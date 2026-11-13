"""LMMS MCP Server - AI-powered music production with LMMS.

This server provides tools, resources, and prompts for creating and
manipulating LMMS projects programmatically via the Model Context Protocol.
"""

import copy
import difflib
import errno
import functools
import inspect
import json
import math
import os
import string
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.etree import ElementTree as ET

from mcp.server import MCPServer

from . import audio_analysis
from . import effect_params
from . import effects as effects_mod
from . import instrument_presets as inst_presets
from . import lmms_app
from . import pitch
from . import presets as zyn_presets
from . import xml_parser
from .models import (
    DEFAULT_BASE_NOTE,
    DEFAULT_KEY,
    KEY_MAX,
    KEY_MIN,
    KEY_NUMBERING_TEXT,
    KEY_RANGE_TEXT,
    MIDI_NOTE_OFFSET,
    NOTE_NAMES,
    TICKS_PER_BAR,
    TICKS_PER_STEP,
    STEPS_PER_BAR,
    TrackType,
    Note,
    bars_to_ticks,
    ticks_to_bars,
)
# The tools below are named note_name_to_key / key_to_note_name too.
from .models import key_to_note_name as _key_to_note_name
from .models import note_name_to_key as _note_name_to_key
from .project import LMMSProject, set_project
from .project import get_project as _get_project

DEFAULT_PROJECTS_DIR = os.environ.get(
    "LMMS_PROJECTS_DIR",
    str(Path.home() / "Desktop" / "Media" / "lmms" / "AI-Projects"),
)


def _expand_user(path):
    """Expand a leading "~" (or "~name") in a file or folder path to the
    home folder, like a shell does; anything else is returned unchanged.

    Every tool that takes a file or folder path runs it through here first
    (tool arguments reach the server as typed, no shell expands them).
    Paths that do not start with "~", such as the factory sample
    "drums/kick01.ogg", stay as they are.
    """
    return os.path.expanduser(path) if isinstance(path, str) else path

# Real LMMS built-in instrument plugin names (as used in the XML
# <instrument name="..."> attribute). Verified against LMMS source.
KNOWN_INSTRUMENTS = {
    "tripleoscillator": "Three-oscillator subtractive synth (default, always works)",
    "kicker": "Kick drum synthesizer",
    "audiofileprocessor": "Audio file player/sampler (WAV, OGG, etc.)",
    "organic": "Additive organ synthesizer",
    "malletsstk": "Physical modeling mallets (STK)",
    "freeboy": "Game Boy sound chip emulator (chiptune)",
    "lb302": "Roland TB-303 style acid bass synth",
    "monstro": "Powerful 3-oscillator polyphonic synth",
    "nes": "NES 8-bit sound chip emulator (chiptune)",
    "opulenz": "OPL3 FM synthesizer (Yamaha DX100 style)",
    "patman": "GUS patch (.pat) sampler (needs a patch file)",
    "sf2player": "SoundFont (.sf2) player: General MIDI piano, guitar, strings, "
                 "drums ... (needs a soundfont file)",
    "sfxr": "Retro sound effect generator (chiptune)",
    "sid": "Commodore 64 SID chip emulator (chiptune)",
    "slicert": "Beat slicer for chopping audio loops",
    "vibedstrings": "Vibrating string physical model",
    "watsyn": "4-oscillator wavetable-style synth",
    "xpressive": "Expressive mono lead synth",
    "zynaddsubfx": "ZynAddSubFX powerful feature-rich synth",
    "gigplayer": "GIG sample library player (needs a .gig file)",
    "bitinvader": "Bit-crushed wavetable synth",
    "vestige": "VST plugin host (Windows only)",
}

# Recommended instruments per use case - helps agents pick valid plugins
INSTRUMENT_RECOMMENDATIONS = {
    "drums": ["kicker", "audiofileprocessor", "sfxr"],
    "bass": ["lb302", "tripleoscillator", "monstro"],
    "lead": ["tripleoscillator", "watsyn", "xpressive", "opulenz"],
    "pad": ["organic", "zynaddsubfx", "monstro"],
    "chiptune": ["freeboy", "nes", "sid", "sfxr"],
    "keys": ["malletsstk", "opulenz", "zynaddsubfx"],
    "strings": ["vibedstrings", "zynaddsubfx"],
}

# Alternative spellings accepted by add_instrument_track
INSTRUMENT_ALIASES = {
    "mallets": "malletsstk",
    "stk": "malletsstk",
    "sf2": "sf2player",
    "fluidsynth": "sf2player",
    "zyn": "zynaddsubfx",
    "zynadd": "zynaddsubfx",
    "audiofile": "audiofileprocessor",
    "afp": "audiofileprocessor",
    "lb-302": "lb302",
    "303": "lb302",
    "opl3": "opulenz",
    "opl2": "opulenz",
}

# Tools that are useless without a plugin; they are removed from the
# server when the installed LMMS lacks it, so agents are never offered them.
# A tuple lists alternatives: the tool stays if any one of them is offered.
TOOL_REQUIRES = {
    "list_zyn_presets": "zynaddsubfx",
    "load_zyn_preset": "zynaddsubfx",
    "set_zyn_params": "zynaddsubfx",
    "set_audiofile_sample": "audiofileprocessor",
    "set_instrument_file": ("sf2player", "gigplayer", "patman"),
    "scan_vst_directory": "vestige",
    "add_vst_track": "vestige",
}

# ──────────────────────────────────────────────────────────────────
# PLUGIN AVAILABILITY (single source of truth, adapts to installed LMMS)
# ──────────────────────────────────────────────────────────────────

# What the installed LMMS offers, computed once at import. Everything the
# agent is shown (instructions, tool descriptions, resources, prompts) and
# every refusal is derived from this. Tests replace it with a fake.
_availability = lmms_app.detect_availability(
    KNOWN_INSTRUMENTS, effects_mod.KNOWN_EFFECTS
)


def get_project() -> LMMSProject:
    """The current project. A first, empty one is written for the installed
    LMMS (its mixer names, see xml_parser.MixerNames)."""
    return _get_project(_availability.version)


def _ordered(known, names) -> list[str]:
    """The known plugin names that are in `names`, in catalogue order."""
    return [n for n in known if n in names]


def _instrument_choices(av) -> list[str]:
    """Available instruments for add_instrument_track (VST has its own tool)."""
    return [n for n in _ordered(KNOWN_INSTRUMENTS, av.instruments)
            if n != "vestige"]


def _effect_choices(av) -> list[str]:
    return sorted(av.effects)


def _lmms_label(av) -> str:
    return f"LMMS {av.version}" if av.version else "the installed LMMS"


def _wrap_names(names, indent: str = " " * 12, width: int = 78) -> str:
    """Comma-separated names wrapped for a docstring (first line unindented)."""
    lines = textwrap.wrap(
        ", ".join(names), width=width - len(indent), break_long_words=False
    )
    return ("\n" + indent).join(lines)


def _file_instruments(av) -> list[str]:
    """The offered instruments that play a file (sf2player, gigplayer, patman)."""
    return [n for n in lmms_app.FILE_INSTRUMENTS if n in av.instruments]


def _file_instrument_details(av) -> list[str]:
    """One short sentence per offered file instrument, for docs and instructions."""
    pool = av.instruments
    details = []
    if "sf2player" in pool:
        details.append(
            "sf2player plays a SoundFont (.sf2) with General MIDI programs: "
            "patch 0 piano, 24 nylon guitar, 32 acoustic bass, 40 violin, "
            "48 strings, 56 trumpet, 73 flute (bank 0). Bank 128 is the "
            "drum kit; its General MIDI drum notes 35-81 are keys 23-69 "
            "here (MIDI note minus 12): key 24 kick, 26 snare, 30 closed "
            "hi-hat, 34 open hi-hat, 37 crash."
        )
    if "gigplayer" in pool:
        details.append(
            "gigplayer plays a .gig sample library (bank 0-999 and patch "
            "0-127 select the instrument inside it)."
        )
    if "patman" in pool:
        details.append(
            "patman plays a GUS patch (.pat), a single instrument without "
            "bank or patch."
        )
    return details


def _pitch_note(av) -> str:
    """Instructions paragraph: key 57 = 440 Hz on every offered instrument."""
    pool = av.instruments
    offered = _ordered(KNOWN_INSTRUMENTS, pool)
    text = "Pitch: every offered instrument is tuned so that key 57 = A4 = 440 Hz."
    compensated = [
        n for n in offered if pitch.plugin_pitch_offset(av.plugin_name(n))
    ]
    if compensated:
        text += (
            f" The track's base note makes up for plugins that sound an "
            f"octave low in LMMS ({', '.join(compensated)}), so never "
            f"transpose notes by hand to correct a pitch."
        )
    exceptions = []
    if "kicker" in pool:
        exceptions.append("kicker is a drum and ignores the key")
    if "sfxr" in pool:
        exceptions.append(
            "sfxr is a sound-effect synth: it follows the key but plays "
            "about 5.5 semitones flat and is silent from key 80"
        )
    if "audiofileprocessor" in pool:
        exceptions.append(
            "audiofileprocessor plays its sample at its original pitch at "
            "the track's base note (key 57 until set_track_tuning or "
            "set_audiofile_sample says at which note the sample was recorded)"
        )
    if exceptions:
        text += " Exceptions: " + "; ".join(exceptions) + "."
    limited = [
        (n, pitch.default_playable_keys(av.plugin_name(n)))
        for n in sorted(offered, key=lambda n: n != "zynaddsubfx")
        if n != "sfxr"
    ]
    limited = [(n, keys) for n, keys in limited if keys]
    if limited:
        name, keys = limited[0]
        text += (
            f" A few instruments do not play the whole key range (e.g. "
            f"{name} plays keys {keys[0]}-{keys[1]} only): "
            "add_instrument_track reports their \"playable_keys\" and "
            "add_note / add_notes_batch warn about notes outside them."
        )
    return text


def _tuning_note(av) -> str:
    """Instructions paragraph: tuning a track with set_track_tuning."""
    text = (
        "Tuning: set_track_tuning(track_index, base_note, pitch_cents) sets "
        "an instrument track's base note (the key that sounds at the "
        "instrument's natural pitch; A4 = key 57 by default) and its fine "
        "tune in cents."
    )
    if "audiofileprocessor" in av.instruments:
        text += (
            " It is how you play a tuned sample: a piano sample recorded at "
            "C4 gets base_note=\"C4\", then every key sounds at its note "
            "name; a sample that is 18 cents flat gets pitch_cents=18. "
            "set_audiofile_sample takes the same two arguments. Never shift "
            "notes by hand to correct a sample's pitch."
        )
    text += (
        " Synths and soundfonts are already tuned: for them a changed base "
        "note only transposes the track."
    )
    return text


def _tuning_exceptions(av) -> str:
    """Sentence naming the offered instruments that cannot follow every
    tuning (empty when there are none); fills $TUNING_LIMITS."""
    parts = []
    for name in _ordered(KNOWN_INSTRUMENTS, av.instruments):
        plugin = av.plugin_name(name)
        limit = pitch.fine_tune_limit(plugin)
        if plugin.lower() in pitch.PLUGINS_IGNORING_KEY:
            parts.append(f"{name} ignores the key, so it cannot be tuned at all")
        elif limit == 0:
            parts.append(f"{name} ignores the fine tune")
        elif limit < pitch.FINE_TUNE_MAX_CENTS:
            parts.append(
                f"{name} follows the fine tune only up to +-{limit} cents")
    if not parts:
        return ""
    return (
        "Limits: " + "; ".join(parts) + ". A call they cannot follow is "
        "refused, nothing changes, and the error says what to do instead."
    )


def _doc_bullets(lines: list[str], indent: str = " " * 4) -> str:
    """Bulleted lines wrapped for a docstring (first line unindented)."""
    return "\n".join(
        textwrap.fill(
            line, width=78, initial_indent=indent + "- ",
            subsequent_indent=indent + "  ",
        )
        for line in lines
    ).lstrip()


def _effect_start_lines(av) -> list[str]:
    """What add_effect starts each offered effect with."""
    return [f"{n}: {effect_params.describe_start(n)}"
            for n in _effect_choices(av) if effect_params.has_params(n)]


def _effect_param_lines(av) -> list[str]:
    """Every offered effect's parameters (name, range unit, default), the
    choices of the long choice lists, and the effects without a table."""
    names = [n for n in _effect_choices(av) if effect_params.has_params(n)]
    lines = [f"{n}: {effect_params.describe_params(n)}" for n in names]
    lines += effect_params.describe_choices(names)
    other = [n for n in _effect_choices(av) if not effect_params.has_params(n)]
    if other:
        lines.append(
            f"{', '.join(other)}: no parameters can be set yet (plugin "
            f"defaults)")
    return lines


def render_plugin_doc(template: str, av) -> str:
    """Fill $INSTRUMENTS / $EFFECTS / $EFFECT_* / $FILE_* / $TUNING_LIMITS in
    a tool docstring template.

    $EFFECT_STARTS lists what add_effect starts each offered effect with and
    $EFFECT_PARAMS the parameters of each (bullets).

    $FILE_INSTRUMENTS names the offered instruments that play a file,
    $FILE_DETAILS describes them (bullets) and $FILE_NOTE is a pointer to
    set_instrument_file (empty when none is offered). $TUNING_LIMITS names
    the offered instruments that cannot follow every set_track_tuning and
    $SAMPLE_TUNING points to set_audiofile_sample (empty without a sampler).
    """
    files = _file_instruments(av)
    return string.Template(template).safe_substitute(
        INSTRUMENTS=_wrap_names(_instrument_choices(av)),
        EFFECTS=_wrap_names(_effect_choices(av)),
        EFFECT_STARTS=_doc_bullets(_effect_start_lines(av)),
        EFFECT_PARAMS=_doc_bullets(_effect_param_lines(av)),
        FILE_INSTRUMENTS=" / ".join(files),
        FILE_DETAILS=_doc_bullets(_file_instrument_details(av)),
        TUNING_LIMITS=textwrap.fill(
            _tuning_exceptions(av), width=74, subsequent_indent=" " * 4),
        SAMPLE_TUNING=textwrap.fill(
            "set_audiofile_sample accepts the same two arguments, to set "
            "the sample and its tuning in one call."
            if "audiofileprocessor" in av.instruments else "",
            width=74, subsequent_indent=" " * 4),
        FILE_NOTE=(
            f"File-based instruments ({', '.join(files)}) get their file and "
            f"sound from set_instrument_file."
            if files else ""
        ),
    )


def _verification_warning(av, name: str) -> str | None:
    """Warning for a version-dependent plugin when nothing could be verified."""
    if av.verified or name not in lmms_app.PLUGIN_REQUIRES:
        return None
    msg = (
        f"No LMMS installation was found, so '{name}' (requires "
        f"{lmms_app.PLUGIN_REQUIRES[name]}) could not be verified; older "
        f"LMMS versions silently ignore it."
    )
    alternative = lmms_app.PLUGIN_SUBSTITUTES.get(name)
    if alternative:
        msg += f" '{alternative}' works in every version."
    return msg


def _refusal(kind: str, name: str, av) -> str:
    """JSON error for a known plugin that the installed LMMS lacks.

    kind is "instrument" or "effect". The project is never modified.
    """
    pool = av.instruments if kind == "instrument" else av.effects
    message = lmms_app.unavailable_reason(name, av.version)
    alternative = lmms_app.PLUGIN_SUBSTITUTES.get(name)
    if alternative not in pool:
        alternative = None
    payload: dict = {}
    if alternative:
        message += f" Use '{alternative}' instead."
    else:
        message += f" Choose one of the available_{kind}s."
    payload["error"] = message
    if alternative:
        payload["alternative"] = alternative
    payload[f"available_{kind}s"] = (
        _instrument_choices(av) if kind == "instrument"
        else _effect_choices(av)
    )
    return json.dumps(payload)


def _needs_file_reason(name: str, av) -> str:
    """Sentence explaining that a file instrument has no file to play."""
    label, _ = lmms_app.FILE_INSTRUMENTS[name]
    searched = ", ".join(av.files.searched.get(name, ())) or "nowhere"
    put_in = av.files.put_in.get(name)
    where = f"Put one in {put_in} and restart the server" if put_in \
        else "Put one in a folder LMMS reads and restart the server"
    return (
        f"'{name}' needs a {lmms_app.extension_text(name)} {label} and none "
        f"was found (searched: {searched}). {where}"
    )


def _needs_file_refusal(name: str, av) -> str:
    """JSON error for a file instrument whose plugin is installed but for
    which no file exists on this machine. The project is never modified."""
    alternatives = [
        n for n in ("sf2player", "audiofileprocessor")
        if n != name and n in av.instruments
    ]
    message = _needs_file_reason(name, av)
    if alternatives:
        message += ", or use " + " / ".join(f"'{n}'" for n in alternatives)
    else:
        message += ", or choose one of the available_instruments"
    payload: dict = {"error": message + "."}
    if alternatives:
        payload["alternative"] = alternatives[0]
    payload["available_instruments"] = _instrument_choices(av)
    return json.dumps(payload)


def _chain_text(chain) -> str:
    return " -> ".join(chain)


def build_instructions(av, projects_dir: str) -> str:
    """Server instructions listing only what the installed LMMS offers."""
    pool = av.instruments

    if av.verified:
        status = (
            f"Installed LMMS: {av.version or 'version unknown'}. This server "
            f"only offers plugins that exist in it: plugin lists in tool "
            f"descriptions and lmms://reference/* are already filtered, and "
            f"add_instrument_track / add_effect refuse anything else without "
            f"changing the project."
        )
    else:
        later = ", ".join(sorted(lmms_app.PLUGIN_REQUIRES))
        status = (
            f"No LMMS installation was detected, so plugin availability could "
            f"not be verified (set LMMS_EXECUTABLE to enable checks). The "
            f"plugin lists show LMMS's standard plugins; some ({later}) need "
            f"LMMS 1.3+ and are silently ignored by older versions."
        )

    pitch_note = _pitch_note(av)

    safe = []
    for names, role in (
        (("tripleoscillator",), "universal synth"),
        (("kicker",), "drums"),
        (("lb302",), "bass"),
        (("freeboy", "nes", "sid"), "chiptune"),
        (("organic",), "pads/organ"),
    ):
        present = [n for n in names if n in pool]
        if present:
            safe.append(f"{'/'.join(present)} ({role})")

    blocks = [
        f"""LMMS MCP Server for AI-powered music production.

This server lets you create, load, modify, and save LMMS music projects.
LMMS is a free, open-source digital audio workstation (DAW).

Key concepts:
- LMMS projects contain tracks (Instrument, Sample, Pattern, Automation)
- Instrument tracks hold synthesizer/sampler plugins and patterns with notes
- Pattern tracks (Beat/Bassline) hold drum patterns in the song editor
- The Mixer has channels (0=Master) where tracks route their audio; a
  channel must exist (add_mixer_channel) before a track is routed to it
  (set_track_mixer_channel re-routes an existing track)
- Time is measured in ticks: 192 ticks = 1 bar (4/4 time)
- {KEY_NUMBERING_TEXT} Valid keys are {KEY_RANGE_TEXT}.
  Tools take either the key number or a note name ("C4", "Bb2").
- Volume: 0-200 (100=normal), Panning: -100 to +100""",
        pitch_note,
        _tuning_note(av),
        status,
        f"""IMPORTANT - Instruments: Only use built-in LMMS instrument names (see the
add_instrument_track tool description or lmms://reference/instruments).
LMMS has NO plugin download mechanism. If asked for a sound you cannot
produce, use tripleoscillator (universal synth) and explain the limitation.
Safe choices: {', '.join(safe)}.""",
    ]

    custom = (
        "Custom plugins: The user may have installed additional LMMS "
        "instrument plugins. list_available_plugins shows them; they can be "
        "used by name in add_instrument_track."
    )
    if "vestige" in pool:
        custom += (
            " For VST .dll files anywhere on disk, use scan_vst_directory "
            "and add_vst_track."
        )
    blocks.append(custom)

    if "audiofileprocessor" in pool:
        blocks.append(
            """Drums with real samples: add one instrument track per drum sound with
instrument "audiofileprocessor", then call set_audiofile_sample on each
(e.g. "drums/kick_hiphop01.ogg", "drums/snare01.ogg",
"drums/hihat_closed02.ogg" from LMMS's factory samples) and play it with
add_note / add_notes_batch at key 57 (the track's base note = original
pitch). A tuned sample (piano, strings ...) is different: give
set_audiofile_sample its base_note (the note it was recorded at) and
pitch_cents. An audiofileprocessor track without a sample is silent."""
            + (" For synthesized drums use kicker instead."
               if "kicker" in pool else "")
        )

    file_lines = []
    if "sf2player" in pool:
        file_lines.append(
            "sf2player plays General MIDI sounds from a SoundFont (.sf2): new "
            f"tracks start on {Path(av.files.default_soundfont).name} (bank "
            "0, patch 0 = piano) and set_instrument_file picks other "
            "programs (strings, guitar, flute ...) or the drum kit (bank 128)."
            if av.files.default_soundfont else
            "sf2player plays General MIDI sounds from a SoundFont (.sf2); "
            "choose them with set_instrument_file."
        )
    others = [n for n in ("gigplayer", "patman") if n in pool]
    if others:
        what = {"gigplayer": "a .gig library", "patman": "a GUS .pat patch"}
        plural = len(others) > 1
        file_lines.append(
            " and ".join(f"{n} plays {what[n]}" for n in others)
            + f"; {'both stay' if plural else 'it stays'} silent until "
              f"set_instrument_file gives {'them' if plural else 'it'} a file."
        )
    if file_lines:
        blocks.append("File-based instruments: " + " ".join(file_lines))

    if "zynaddsubfx" in pool:
        blocks.append(
            """ZynAddSubFX: For rich sounds, add a track with instrument "zynaddsubfx",
then load one of ~950 factory presets via load_zyn_preset (browse with
list_zyn_presets). Fine-tune with set_zyn_params. Every preset plays key 57
as A4 = 440 Hz: an octave shift built into a preset (many basses have one)
is compensated for you."""
        )

    blocks.append(
        "Instrument presets: list_instrument_presets shows the ready-made "
        "sounds (LMMS's factory presets and the user's own saved ones, "
        "often tuned); load_instrument_preset(track_index, \"Folder/Name\") "
        "loads one into an instrument track and keeps its notes, or "
        "creates the track when track_index is null."
    )

    chains = lmms_app.adapt_recommendations(
        {
            "Lead": ["delay", "reverbsc"],
            "Vocals": ["eq", "compressor", "reverbsc"],
            "Master bus": ["eq", "compressor", "stereoenhancer"],
        },
        av.effects,
    )
    blocks.append(
        "Effects: Add built-in effects to tracks or mixer channels with "
        "add_effect (see lmms://reference/effects for the full list and "
        "every parameter). A plain add_effect starts the effect with "
        "audible, musical settings (a delay is synced to the project tempo; "
        "effects that output only the processed signal keep the dry "
        "sound); tune it with set_effect_params and check it with "
        "get_effect_chain. Typical chains:\n"
        + "\n".join(f"- {use}: {_chain_text(c)}" for use, c in chains.items())
    )

    blocks.append(
        f"""Default projects directory: {projects_dir}
When saving a project, use this directory if no specific path is given.
Example: save to {projects_dir}/my_song.mmpz
A relative path in save_project, load_project and the output_path of
render_project is relative to this directory: after save_project("song.mmpz")
load_project("song.mmpz") loads that same file.

Workflow: Create a project -> Add tracks -> Load instruments/presets ->
Add notes -> Add effects -> Configure mixer -> Save"""
    )
    return "\n\n".join(blocks) + "\n"


mcp = MCPServer(
    "LMMS MCP Server",
    instructions=build_instructions(_availability, DEFAULT_PROJECTS_DIR),
)


TOOL_ERROR_HINT = (
    "The tool did not handle this error. The project may be partly "
    "modified; call get_project_info to check. Check the arguments; if it "
    "persists, report it."
)


def _tool_error_response(tool_name: str, exc: Exception) -> str:
    """The JSON a tool answers when it raised an exception.

    The mcp library reports an exception that escapes a tool to the
    client as just "Error executing tool <name>", without its message, so
    an agent cannot tell what went wrong. This puts the exception type and
    message in the answer instead, and the traceback on stderr (a stdio
    server must not print to stdout).
    """
    message = str(exc)
    error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    try:
        print(f"lmms-mcp: tool {tool_name} raised {error}", file=sys.stderr)
        traceback.print_exception(exc, file=sys.stderr)
    except Exception:  # a closed or broken stderr must not hide the error
        pass
    return json.dumps({"error": error, "tool": tool_name, "hint": TOOL_ERROR_HINT})


def _guard_tool(fn):
    """Wrap a tool function so an exception becomes a JSON error answer.

    Only Exception is caught: KeyboardInterrupt, SystemExit and
    asyncio.CancelledError (BaseExceptions) still propagate. functools.wraps
    keeps the name, docstring and signature, so the tool's description and
    input schema are those of fn.
    """
    name = fn.__name__
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def guarded(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                return _tool_error_response(name, exc)
    else:

        @functools.wraps(fn)
        def guarded(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                return _tool_error_response(name, exc)

    return guarded


def _tool(fn):
    """Register fn as an MCP tool (use instead of @mcp.tool()).

    The registered tool is the guarded wrapper (see _guard_tool); fn itself
    is returned, so calling a tool function directly (as the tests do) still
    raises its exceptions. Every tool goes through here so none can fail
    with a message-less error.
    """
    mcp.tool()(_guard_tool(fn))
    return fn


def _plugin_tool(fn):
    """Register a tool whose docstring lists the available plugins.

    The docstring may contain $INSTRUMENTS / $EFFECTS; it is rendered
    from the installed LMMS before registration, so the tool description
    the agent sees names only plugins that exist. The raw template is
    kept in fn.__plugin_doc_template__.
    """
    fn.__plugin_doc_template__ = fn.__doc__ or ""
    fn.__doc__ = render_plugin_doc(fn.__plugin_doc_template__, _availability)
    return _tool(fn)


# ──────────────────────────────────────────────────────────────────
# PROJECT TOOLS
# ──────────────────────────────────────────────────────────────────


@_tool
def create_project(
    bpm: int = 140,
    time_sig_numerator: int = 4,
    time_sig_denominator: int = 4,
    master_volume: int = 100,
    master_pitch: int = 0,
) -> str:
    """Create a new empty LMMS project.

    Args:
        bpm: Tempo in beats per minute (10-999)
        time_sig_numerator: Time signature numerator (e.g. 4 for 4/4)
        time_sig_denominator: Time signature denominator (e.g. 4 for 4/4)
        master_volume: Master volume (0-200, 100=normal)
        master_pitch: Master pitch offset in semitones (-12 to +12)
    """
    proj = LMMSProject()
    proj.new(
        bpm=bpm,
        time_sig_numerator=time_sig_numerator,
        time_sig_denominator=time_sig_denominator,
        master_volume=master_volume,
        master_pitch=master_pitch,
        lmms_version=_availability.version,
    )
    set_project(proj)
    return json.dumps({
        "message": "New project created",
        "bpm": bpm,
        "time_signature": f"{time_sig_numerator}/{time_sig_denominator}",
        "master_volume": master_volume,
        "master_pitch": master_pitch,
    })


_PROJECT_SUFFIXES = (".mmp", ".mmpz")


def _project_files_in(folder: Path, limit: int = 20) -> list[Path]:
    """The .mmp / .mmpz files directly in a folder, by name ([] when the
    folder cannot be listed)."""
    try:
        found = sorted(
            f for f in folder.iterdir()
            if f.suffix.lower() in _PROJECT_SUFFIXES and f.is_file())
    except OSError:
        return []
    return found[:limit]


def _missing_project_hint(path: Path) -> str:
    """What to try when a project file does not exist: the project files
    with a similar name (the other extension, other case, a close
    spelling) in the same folder, or that the folder itself is missing."""
    folder = path.parent
    own = [f for f in _project_files_in(folder) if f != path]
    same_stem = [f for f in own if f.stem.lower() == path.stem.lower()]
    close_names = difflib.get_close_matches(
        path.name.lower(), [f.name.lower() for f in own], n=3, cutoff=0.6)
    similar = same_stem + [
        f for f in own
        if f.name.lower() in close_names and f not in same_stem]
    if similar:
        listed = ", ".join(str(f) for f in similar[:3])
        return (f" A similar file exists: {listed}. Pass that path if it "
                f"is the project you meant.")
    where = ("" if folder.is_dir()
             else f" The folder {folder} does not exist either.")
    return (f"{where} Check the path (use an absolute path, or one "
            f"starting with ~/).")


def _load_error(path: str, exc: Exception, relative: bool = False) -> dict:
    """The {"error", "path"} response of a project file that could not be
    loaded, with the reason and what to try. `path` is the absolute path
    that was tried; `relative` is True when the caller passed a relative
    path (which is looked up in more than one place)."""
    where = Path(path)
    if isinstance(exc, FileNotFoundError):
        searched = ""
        if relative:
            searched = (
                f" A relative path is looked up in the default projects "
                f"folder ({DEFAULT_PROJECTS_DIR}), then in the server's "
                f"working folder ({_working_folder()}).")
        reason = f"no such file.{searched}{_missing_project_hint(where)}"
    elif isinstance(exc, IsADirectoryError):
        files = _project_files_in(where, limit=5)
        reason = "it is a folder, not a project file."
        if files:
            reason += (" Project files in it: "
                       f"{', '.join(str(f) for f in files)}.")
        else:
            reason += (" It has no .mmp or .mmpz file; pass the path of "
                       "the project file itself.")
    elif isinstance(exc, PermissionError):
        reason = (f"{_os_error_reason(exc, path)}. The file or its folder "
                  f"cannot be read by this server; fix the permissions or "
                  f"load a copy.")
    elif isinstance(exc, ValueError):
        # xml_parser.load_project: "Cannot parse <path>: <why>".
        reason = str(exc).removeprefix(f"Cannot parse {where}: ")
        try:
            if where.stat().st_size == 0:
                reason += " (the file is empty)"
        except OSError:
            pass
        reason += (". LMMS projects are .mmp (plain XML) or .mmpz "
                   "(compressed) files; the file may be corrupt or "
                   "another format.")
    else:
        reason = _os_error_reason(exc, path)
    return {"error": f"Could not load {path}: {reason}", "path": path}


def _in_projects_folder(path: str) -> Path:
    """A project file path as save_project places it: a relative path is
    under the default projects folder (the result is absolute even when
    LMMS_PROJECTS_DIR is relative); an absolute path stays as it is."""
    full = Path(DEFAULT_PROJECTS_DIR) / path
    return full if full.is_absolute() else Path.cwd() / full


def _working_folder() -> str:
    """The server process's working folder ("" if it was deleted)."""
    try:
        return str(Path.cwd())
    except OSError:
        return ""


def _resolve_project_file(path: str) -> tuple[str, str | None]:
    """Where load_project reads a project file from, and a note for the
    response when that needs saying.

    An absolute path is used as it is. A relative path is looked up the way
    save_project places it, in the default projects folder, and only if
    nothing is there in the server's working folder. The returned path is
    always absolute. The note names the other file when a relative path
    exists in both places (the default folder wins), or says that the
    working-folder file was loaded because the default folder has none.
    """
    if Path(path).is_absolute():
        return path, None
    in_default = _in_projects_folder(path)
    cwd = _working_folder()
    if not cwd:
        return str(in_default), None
    in_cwd = Path(cwd) / path
    try:
        if in_default.is_file():
            if in_cwd.is_file() and not _same_file(in_default, in_cwd):
                return str(in_default), (
                    f"A file of that name also exists in the server's "
                    f"working folder ({in_cwd}); relative paths mean the "
                    f"default projects folder, so {in_default} was loaded. "
                    f"Pass the full path to load the other one.")
            return str(in_default), None
        if in_cwd.is_file():
            return str(in_cwd), (
                f"{path} is not in the default projects folder "
                f"({DEFAULT_PROJECTS_DIR}); loaded the file of that name "
                f"in the server's working folder instead: {in_cwd}.")
        # Neither is a file: let the load report what is at the default
        # place (no such file, or a folder), unless only the working
        # folder has something there (a folder).
        if not in_default.exists() and in_cwd.exists():
            return str(in_cwd), None
    except OSError:
        # Unreadable folder: loading in_default reports the real reason.
        pass
    return str(in_default), None


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


@_tool
def load_project(path: str) -> str:
    """Load an existing LMMS project file (.mmpz or .mmp).

    Relative paths are relative to the default projects folder
    (LMMS_PROJECTS_DIR), the same as in save_project, so the name that
    save_project was given loads that file again. If a relative path is
    not there but exists in the server's working folder, that file is
    loaded and the response has a "note" saying so; if it exists in both
    places the default folder's file is loaded and the note names the
    other. The response always has the absolute "path" that was loaded.

    If the file cannot be loaded the response is {"error": ..., "path":
    ...} with the reason (no such file, a folder instead of a file, no
    permission, not an LMMS project file), and the project that was open
    stays as it was. When the file does not exist the error lists project
    files with a similar name (e.g. the same name with the other extension
    .mmp / .mmpz).

    A project that uses the mixer names of another LMMS than the installed
    one is converted: LMMS 1.2 ignores the names of LMMS 1.3 (a project an
    older version of this server wrote for LMMS 1.2 has them), so its mixer
    channels (volume, mute, effects) and track routing would have no
    effect. The response then has "converted_mixer" (from, to, channels,
    tracks, message); the file stays as it is until save_project. Nothing
    is converted when no LMMS is installed. "warnings" lists tracks routed
    to a mixer channel the project lacks (LMMS plays those on another
    channel); set_track_mixer_channel routes them to a channel that exists.

    Args:
        path: Path of the LMMS project file: absolute, starting with ~ (the
            home folder), or relative to the default projects folder
            (LMMS_PROJECTS_DIR), e.g. "song.mmpz" or "songs/take1.mmp"
    """
    path = _expand_user(path)
    relative = not Path(path).is_absolute()
    path, note = _resolve_project_file(path)
    proj = LMMSProject()
    try:
        proj.load(path)
    except (OSError, ValueError) as exc:
        # The open project is untouched: set_project only runs below.
        return json.dumps(_load_error(path, exc, relative))
    converted = _convert_mixer(proj, _availability)
    set_project(proj)
    info = proj.get_info()
    response = {
        "message": f"Loaded project from {path}",
        "path": path,
        "bpm": info.bpm,
        "time_signature": info.time_signature,
        "track_count": info.track_count,
        "tracks": [{"index": t.index, "name": t.name, "type": t.type_name} for t in info.tracks],
    }
    if note:
        response["note"] = note
    if converted:
        response["converted_mixer"] = converted
    warning = _unroutable_warning(proj)
    if warning:
        response["warnings"] = [warning]
    return json.dumps(response)


def _convert_mixer(proj, av) -> dict | None:
    """Rename a loaded project's mixer to the names the installed LMMS reads.

    None when nothing needed converting (no LMMS found, LMMS 1.3+ which
    upgrades a project with the 1.2 names itself, or names that already
    match); else the report load_project returns as "converted_mixer".
    """
    names = xml_parser.mixer_names_to_convert_to(av.version)
    if names is None:
        return None
    report = proj.convert_mixer(names)
    if report is None:
        return None
    source = report["from"]
    message = (
        f"The project used the mixer names of LMMS 1.3 "
        f"({source.label}), which {_lmms_label(av)} ignores: its tracks "
        f"would play on Master and its mixer channels (volume, mute, "
        f"effects, sends) would have no effect. They were renamed to the "
        f"names {_lmms_label(av)} reads ({names.label}); the channels and "
        f"tracks keep everything else. The file on disk is unchanged until "
        f"save_project."
    )
    result = {
        "from": source.label,
        "to": names.label,
        "channels": report["channels"],
        "tracks": report["tracks"],
    }
    if report["sample_tracks_on_master"]:
        listed = ", ".join(
            f"'{s['track']}' (channel {s['channel']})"
            for s in report["sample_tracks_on_master"])
        message += (
            f" {_lmms_label(av)} cannot route sample tracks, so these "
            f"now play on Master: {listed}."
        )
        result["sample_tracks_on_master"] = report["sample_tracks_on_master"]
    result["message"] = message
    return result


def _unroutable_warning(proj) -> str | None:
    """Sentence naming the tracks routed to a mixer channel that does not
    exist (LMMS plays those on another channel), or None."""
    found = proj.unroutable_tracks()
    if not found:
        return None
    listed = ", ".join(
        f"track {index} '{name}' (channel {channel})"
        for index, name, channel in found)
    many = len(found) > 1
    next_channel = xml_parser.mixer_channel_numbers(proj.root)[-1] + 1
    return (
        f"Misrouted in LMMS: {listed} {'are' if many else 'is'} "
        f"routed to a mixer channel the project does not have (it has "
        f"{xml_parser.describe_mixer_channels(proj.root)}), so LMMS plays "
        f"{'them' if many else 'it'} on another channel, with "
        f"that channel's volume and effects. Route "
        f"{'each' if many else 'the track'} to a channel that exists with "
        f"set_track_mixer_channel(track_index, mixer_channel), e.g. "
        f"set_track_mixer_channel({found[0][0]}, 0); add_mixer_channel(name) "
        f"first creates another channel (the next one is channel "
        f"{next_channel})."
    )


def _create_parent_folders(path: str | Path) -> list[str]:
    """Create the folders above a file path that do not exist yet.

    Returns the folders it created, outermost first ([] when they all
    existed). If a folder cannot be created, the ones made so far are
    removed again and an OSError that names the folder is raised.
    """
    missing: list[Path] = []
    folder = Path(path).parent
    while not folder.exists():
        missing.append(folder)
        if folder.parent == folder:
            break
        folder = folder.parent
    missing.reverse()
    if folder.exists() and not folder.is_dir():
        # A file where a folder is needed (LMMS and mkdir would only say
        # "could not open file" / "file exists").
        raise OSError(errno.ENOTDIR, f"{folder} is a file, not a folder")
    try:
        if missing:
            missing[-1].mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _remove_folders(str(f) for f in missing)
        raise OSError(
            exc.errno,
            f"could not create the folder {exc.filename or missing[-1]}: "
            f"{exc.strerror or exc}") from exc
    return [str(f) for f in missing]


def _remove_folders(folders) -> None:
    """Remove the given folders again (innermost first) when they are
    empty; used to undo _create_parent_folders after a failure. Anything
    that cannot be removed is left alone."""
    for folder in reversed(list(folders)):
        try:
            os.rmdir(folder)
        except OSError:
            pass


def _os_error_reason(exc: Exception, path: str) -> str:
    """The reason of a file system error, as a short sentence part: the
    system's text (e.g. "Permission denied"), plus the folder or file it
    names when that is not the path itself."""
    if not isinstance(exc, OSError):
        return str(exc)
    reason = exc.strerror or str(exc)
    if exc.filename and str(exc.filename) != path:
        reason += f" ({exc.filename})"
    return reason


@_tool
def save_project(
    path: str = "",
    compressed: bool | None = None,
) -> str:
    """Save the current LMMS project to a file.

    Folders that do not exist yet are created (the response lists them in
    "created_dirs"). If the file cannot be written (no permission, the path
    is an existing folder, disk full, ...) the response is
    {"error": ..., "path": ...} with the reason, and the project keeps its
    previous path.

    Args:
        path: File path to save to. If empty, saves to the default projects directory.
            Relative paths are relative to the default projects folder
            (LMMS_PROJECTS_DIR), the same as in load_project: a filename
            (e.g. "song.mmpz") is saved in that folder, and a relative path
            with folders (e.g. "songs/jazz/take1.mmp") is created under it.
            A path starting with ~ (e.g. "~/songs/take1.mmp") is in the
            home folder.
        compressed: If True, saves as .mmpz (compressed). If False, saves as .mmp
            (plain XML). By default (None) the format follows the file extension:
            ".mmp" is always plain XML, everything else compressed. Writing
            compressed data into a .mmp breaks LMMS and other tools.
    """
    proj = get_project()
    path = _expand_user(path)

    if not path and not proj.path:
        save_path = str(Path(DEFAULT_PROJECTS_DIR) / "untitled.mmpz")
    elif path and not Path(path).is_absolute():
        save_path = str(_in_projects_folder(path))
    else:
        # An absolute path, or no path: save again where the project is.
        save_path = path or proj.path

    if compressed is None:
        # Follow the extension: plain XML for .mmp, compressed otherwise.
        # (LMMSProject.save does the same inference; resolve it here so
        # the response reports the actual format.)
        compressed = not save_path.lower().endswith(".mmp")

    created: list[str] = []
    try:
        if Path(save_path).is_dir():
            return json.dumps({
                "error": f"Could not save the project to {save_path}: it is "
                         f"an existing folder, not a file. Add a file name, "
                         f"e.g. {os.path.join(save_path, 'song.mmpz')}.",
                "path": save_path,
            })
        created = _create_parent_folders(save_path)
        result_path = proj.save(save_path, compressed=compressed)
    except (OSError, ValueError) as exc:
        # LMMSProject.save sets the project's path only after writing, so
        # the project is unchanged here.
        _remove_folders(created)
        return json.dumps({
            "error": f"Could not save the project to {save_path}: "
                     f"{_os_error_reason(exc, save_path)}",
            "path": save_path,
        })

    result = {
        "message": f"Project saved to {result_path}",
        "path": result_path,
        "compressed": compressed,
    }
    if created:
        result["created_dirs"] = created
    return json.dumps(result)


@_tool
def get_project_info() -> str:
    """Get comprehensive information about the current LMMS project.

    Returns tempo, time signature, tracks, mixer channels, and more.
    """
    proj = get_project()
    info = proj.get_info()
    return json.dumps(info.to_dict(), indent=2)


@_tool
def get_project_xml() -> str:
    """Get the raw XML representation of the current project.

    Useful for debugging or understanding the exact project structure.
    """
    proj = get_project()
    return proj.get_xml_string()


# ──────────────────────────────────────────────────────────────────
# TRACK TOOLS
# ──────────────────────────────────────────────────────────────────


def _add_instrument(
    proj, name: str, instrument: str, av, mixer_channel: int, volume: int,
    panning: int, custom: bool = False,
) -> dict:
    """Add an instrument track using the plugin name LMMS actually loads.

    LMMS loads an instrument by the exact plugin name written into the
    project (case-sensitive on Linux), which can differ from the name this
    server offers: opulenz is OPL2 and freeboy is papu in LMMS 1.2. The
    result reports the offered name as "instrument" and, when it differs,
    the name written into the project as "lmms_plugin".
    """
    plugin = av.plugin_name(instrument)
    try:
        result = proj.add_track(
            "instrument", name,
            instrument=plugin,
            mixer_channel=mixer_channel,
            volume=volume,
            panning=panning,
        )
    except ValueError as exc:  # a mixer channel the project lacks
        return {"error": str(exc)}
    result["instrument"] = instrument
    if plugin != instrument:
        result["lmms_plugin"] = plugin
    keys = proj.playable_keys(result["track_index"])
    if keys:
        result["playable_keys"] = list(keys)
    if custom:
        result["custom_plugin"] = True
    else:
        warning = _verification_warning(av, instrument)
        if warning:
            result["warning"] = warning
        if instrument in lmms_app.FILE_INSTRUMENTS:
            result.update(
                _start_file_instrument(proj, result["track_index"], instrument, av)
            )
    return result


def _start_file_instrument(proj, index: int, instrument: str, av) -> dict:
    """What a new sf2player / gigplayer / patman track needs to make sound.

    sf2player starts with the default soundfont (bank 0, patch 0, or the
    soundfont's first preset when it has no bank 0 patch 0) so it plays at
    once; gigplayer and patman have no sensible default and stay silent
    until set_instrument_file gives them a file, which the response says.
    """
    if instrument == "sf2player" and av.files.default_soundfont:
        path = av.files.default_soundfont
        bank, patch = lmms_app.default_soundfont_preset(path)
        try:
            info = xml_parser.set_instrument_file(
                proj.root, index, path, bank, patch)
        except ValueError as exc:
            return {"warning": (
                f"The default soundfont could not be set ({exc}) so the "
                f"track is silent. Call set_instrument_file(track_index="
                f"{index}, file_path=<a .sf2 file>)."
            )}
        proj._modified = True
        preset = f" ({info['preset']})" if info.get("preset") else ""
        return {
            "instrument_file": {
                k: info[k] for k in ("file", "bank", "patch", "preset")
                if k in info
            },
            "hint": (
                f"Plays {Path(path).name} bank {bank} patch {patch}"
                f"{preset} already. Change the sound with "
                f"set_instrument_file(track_index={index}, file_path, bank, "
                f"patch); its description lists General MIDI patch numbers "
                f"and the drum kit (bank 128)."
            ),
        }
    first = av.files.for_instrument(instrument)[0]
    return {"hint": (
        f"{instrument} is silent until it has a "
        f"{lmms_app.extension_text(instrument)} file: call "
        f"set_instrument_file(track_index={index}, file_path='{first}') or "
        f"use another file listed under instrument_files in "
        f"list_available_plugins."
    )}


@_plugin_tool
def add_instrument_track(
    name: str,
    instrument: str = "tripleoscillator",
    mixer_channel: int = 0,
    volume: int = 100,
    panning: int = 0,
) -> str:
    """Add an instrument track to the project.

    Args:
        name: Track name (e.g. "Lead Synth", "Bass")
        instrument: Plugin name (lowercase). Valid options in the installed
            LMMS:
            $INSTRUMENTS
        mixer_channel: Mixer channel number (0=Master, 1+=custom channels).
            The channel must exist (create it with add_mixer_channel
            first): LMMS would play a track routed to a missing channel on
            another one, so such a call is refused.
        volume: Track volume (0-200, 100=normal)
        panning: Track panning (-100 to +100, 0=center)

    The response reports "instrument" (the name you gave). When the
    installed LMMS knows that plugin under another name, it also reports
    "lmms_plugin", the name written into the project; keep using the
    "instrument" name yourself.

    Key 57 = A4 = 440 Hz on the new track (unpitched drum and effect
    instruments excepted). When the instrument does not play all keys
    0-108 at the right pitch, the response has "playable_keys": [lowest,
    highest]; add_note warns about notes outside.

    $FILE_NOTE
    """
    proj = get_project()
    av = _availability

    # Validate instrument name (case-insensitive fuzzy match)
    normalized = instrument.strip().lower()
    if normalized not in KNOWN_INSTRUMENTS:
        resolved = INSTRUMENT_ALIASES.get(normalized)
        if resolved is None:
            # Not a known name - but maybe a custom/newly installed plugin?
            if normalized in av.installed:
                if normalized in lmms_app.NON_TRACK_PLUGINS:
                    return json.dumps({
                        "error": f"'{normalized}' is an LMMS helper plugin "
                                 f"(importer/exporter/analyzer), not an "
                                 f"instrument.",
                        "available_instruments": _instrument_choices(av),
                    })
                return json.dumps(_add_instrument(
                    proj, name, normalized, av, mixer_channel, volume,
                    panning, custom=True,
                ))
            hint = (
                "Use one of valid_instruments. Use list_available_plugins "
                "to see what is installed."
            )
            if "vestige" in av.instruments:
                hint = (
                    "Use one of valid_instruments. For VST plugins use "
                    "add_vst_track with the DLL path. Use "
                    "list_available_plugins to see what is installed."
                )
            return json.dumps({
                "error": f"Unknown instrument '{instrument}'. "
                f"LMMS has no plugin download mechanism - only built-in "
                f"instruments can be used.",
                "valid_instruments": _instrument_choices(av),
                "hint": hint,
            })
        normalized = resolved

    # Refuse plugins the installed LMMS lacks: it would silently load
    # nothing and the track would be silent.
    if normalized in av.needs_file:
        return _needs_file_refusal(normalized, av)
    if normalized not in av.instruments:
        return _refusal("instrument", normalized, av)

    return json.dumps(_add_instrument(
        proj, name, normalized, av, mixer_channel, volume, panning,
    ))


def _sample_path_warning(file_path: str) -> str | None:
    """Soft-check that a sample path exists; return a warning or None.

    Relative paths are looked up in the factory samples directory of the
    installed LMMS (skipped silently if none is found); absolute paths
    must exist on disk.
    """
    if PurePosixPath(file_path).is_absolute() or \
            PureWindowsPath(file_path).is_absolute():
        if not Path(file_path).is_file():
            return f"Sample file not found: {file_path}"
        return None
    samples_dir = lmms_app.get_samples_dir()
    if samples_dir is None:
        return None
    if not (samples_dir / file_path).is_file():
        return (
            f"Sample '{file_path}' was not found in the LMMS factory "
            f"samples directory ({samples_dir}). LMMS will load nothing "
            f"and the track will be silent. Browse that directory for "
            f"valid names (e.g. drums/kick_hiphop01.ogg) or pass an "
            f"absolute path."
        )
    return None


def _tuning_summary(inst_track: ET.Element) -> dict:
    """The tuning fields set_audiofile_sample reports."""
    report = pitch.tuning_report(inst_track)
    return {k: report[k] for k in (
        "base_note", "base_note_name", "pitch_cents", "hint")}


@_tool
def set_audiofile_sample(
    track_index: int,
    file_path: str,
    amp: int = 100,
    reversed: bool = False,
    looped: int = 0,
    stutter: bool = False,
    start: float = 0.0,
    end: float = 1.0,
    base_note: int | str | None = None,
    pitch_cents: float | None = None,
) -> str:
    """Set the sample played by an audiofileprocessor instrument track.

    Without a sample an audiofileprocessor track is silent. Create the
    track first with add_instrument_track(name, instrument=
    "audiofileprocessor"), assign the sample here, then play it with
    add_note / add_notes_batch on that track (it is a normal instrument
    track with a pattern, not a sample track). Calling this again on the
    same track replaces the previous sample and settings.

    Drum-kit workflow: create ONE audiofileprocessor track per drum
    sound (Kick, Snare, Hat, ...), set a sample on each, then write each
    drum's rhythm as notes on its own track. This is much easier to edit
    than placing one sample clip per hit on sample tracks.

    Paths: a relative path is resolved by LMMS against its factory
    samples folder, e.g. "drums/kick_hiphop01.ogg", "drums/snare01.ogg",
    "drums/hihat_closed02.ogg", "basses/bass01.ogg". Use an absolute path
    (or one starting with ~/) for your own files (WAV/OGG/FLAC/MP3). If
    the file cannot be found
    the response contains a "warning" (the sample is still set).

    Pitch: the sample sounds at its original pitch when the note equals
    the track's base note. A new track's base note is key 57 (A4, the LMMS
    default), right for drum one-shots: trigger them with key=57. Other
    keys transpose the sample (higher key = higher pitch and shorter).
    For a tuned sample (piano, strings, a bass note) pass base_note, the
    note the sample was recorded at, and keys then sound at their note
    names: a piano sample recorded at C4 gets base_note="C4", key 48 (C4)
    plays it unchanged and key 57 (A4) plays it as 440 Hz. A sample that
    is slightly out of tune gets pitch_cents (+18 raises every note by 18
    cents, for a sample that is 18 cents flat). The track keeps its tuning
    when you set another sample, so pass both again (or call
    set_track_tuning) when the new sample has another pitch. The same
    tuning is available alone as set_track_tuning.

    Note length: the sample is cut off when the note ends, so use a
    length at least as long as the sample (24 ticks is enough for most
    short hits; use 48+ for open hats, crashes and other long samples).

    Args:
        track_index: Zero-based index of the audiofileprocessor track
        file_path: Sample path, relative to LMMS factory samples, or
            absolute (a path starting with ~ is in the home folder)
        amp: Amplification in percent (0-500, 100=unchanged)
        reversed: Play the sample backwards
        looped: 0=off (one-shot), 1=loop, 2=ping-pong loop
        stutter: Restart the sample from the start for every new note
        start: Sample start point as a fraction 0..1 (e.g. 0.1 skips the
            first 10%); must be smaller than end
        end: Sample end point as a fraction 0..1 (1.0 = whole sample)
        base_note: Note (name like "C4" or LMMS key 0-108) the sample was
            recorded at. Omit to leave the track's base note alone.
        pitch_cents: Fine tune in cents, -1200 to +1200 (+18 for a sample
            that is 18 cents flat). Omit to leave it alone.
    """
    proj = get_project()
    file_path = _expand_user(file_path)
    inst_track = proj._instrument_track_element(track_index) \
        if isinstance(track_index, int) else None
    previous = None if inst_track is None else inst_track.find(
        "instrument/audiofileprocessor")
    previous_src = None if previous is None else previous.get("src")
    try:
        key = None if base_note is None else pitch.parse_base_note(base_note)
        xml_parser.set_audiofile_sample(
            proj.root, track_index, file_path, amp=amp, reversed=reversed,
            looped=looped, stutter=stutter, start=start, end=end,
            base_note=key, pitch_cents=pitch_cents,
        )
    except (ValueError, IndexError) as exc:
        return json.dumps({"error": str(exc)})
    proj._modified = True
    result = {
        "track_index": track_index,
        "file": file_path,
        "message": f"Set sample '{file_path}' on audiofileprocessor "
                   f"track {track_index}",
    }
    warning = _sample_path_warning(file_path)
    if warning:
        result["warning"] = warning
    inst_track = proj._instrument_track_element(track_index)
    given = base_note is not None or pitch_cents is not None
    summary = _tuning_summary(inst_track)
    tuned = summary["base_note"] != DEFAULT_BASE_NOTE \
        or summary["pitch_cents"] != 0
    if given or tuned:
        result["tuning"] = summary
        if not given and previous_src != file_path:
            result["tuning"]["note"] = (
                f"The track keeps the tuning it had. If this sample has "
                f"another pitch, pass base_note / pitch_cents, or reset "
                f"with set_track_tuning({track_index}, base_note=57, "
                f"pitch_cents=0)."
            )
    return json.dumps(result)


@_plugin_tool
def set_instrument_file(
    track_index: int,
    file_path: str,
    bank: int = 0,
    patch: int = 0,
) -> str:
    """Set the file and sound of a file-based instrument track ($FILE_INSTRUMENTS).

    These instruments are silent without a file. Create the track with
    add_instrument_track, call this on it, then play it with add_note /
    add_notes_batch. Calling it again replaces the file and sound. Use an
    absolute path (or one starting with ~/) to an existing file;
    list_available_plugins shows the files found on this machine under
    "instrument_files". (For an
    audiofileprocessor track use set_audiofile_sample instead.)

    $FILE_DETAILS

    A soundfont must contain the bank/patch asked for, otherwise the call
    fails and lists the banks and patches the file has.

    Args:
        track_index: Zero-based index of the instrument track
        file_path: Absolute path of the file (a path starting with ~ is in
            the home folder)
        bank: Bank number 0-999 (0 = melodic sounds)
        patch: Patch / program number 0-127
    """
    proj = get_project()
    av = _availability
    kinds = [
        n for n in lmms_app.FILE_INSTRUMENTS
        if n in av.instruments or n in av.needs_file
    ]
    file_path = _expand_user(file_path)
    try:
        info = xml_parser.set_instrument_file(
            proj.root, track_index, file_path, bank=bank, patch=patch,
            kinds=kinds, offered=_file_instruments(av),
        )
    except (ValueError, IndexError) as exc:
        return json.dumps({"error": str(exc)})
    proj._modified = True
    sound = ""
    if "bank" in info:
        sound = f" bank {info['bank']} patch {info['patch']}"
        if info.get("preset"):
            sound += f" ({info['preset']})"
    return json.dumps({
        "track_index": track_index,
        **info,
        "message": f"Set {Path(info['file']).name}{sound} on "
                   f"{info['instrument']} track {track_index}",
    })


def _tuning_track(proj, track_index) -> tuple[ET.Element | None, str | None]:
    """(<instrumenttrack> element, None), or (None, error message)."""
    error = _track_index_error(proj, track_index)
    if error:
        return None, error
    inst_track = proj._instrument_track_element(track_index)
    if inst_track is None:
        kind = xml_parser.track_type_name(xml_parser.get_track_type(
            xml_parser.find_track_element(proj.root, track_index))).lower()
        return None, (
            f"Track {track_index} is a {kind} track, not an instrument "
            f"track, so it has no base note or fine tune. Tuning applies to "
            f"tracks made with add_instrument_track (list_tracks shows the "
            f"types)."
        )
    return inst_track, None


@_plugin_tool
def set_track_tuning(
    track_index: int,
    base_note: int | str | None = None,
    pitch_cents: float | None = None,
) -> str:
    """Tune an instrument track: its base note and a fine tune in cents.

    base_note is the note at which the instrument or sample sounds at its
    natural, recorded pitch. A new track's base note is key 57 (A4, 440
    Hz). For a sample (audiofileprocessor) this says which note it was
    recorded at: a piano sample recorded at C4 gets base_note="C4", then
    key 48 (C4) plays it unchanged, other keys transpose it, and every key
    sounds at its note name. Without it a tuned sample only plays at its
    original pitch on key 57 and every note would need shifting by hand.
    For synths and soundfonts (already tuned: key 57 = 440 Hz) a changed
    base note only transposes the whole track: base_note="C4" makes key 48
    sound as A4, every key 9 semitones higher than its name. Whatever a
    plugin does on its own is compensated, so the base note you give is
    always the note that sounds at 440 Hz (or at the sample's original
    pitch).

    pitch_cents is a fine tune applied to every note: positive raises,
    negative lowers (+18 for a sample that is 18 cents flat; 100 cents = 1
    semitone; -1200 to +1200). Use base_note for whole semitones.
    $TUNING_LIMITS

    Give either argument or both; omit one to leave it as it is. Called with
    neither, nothing changes and the current tuning is returned. Tuning
    applies to instrument tracks only (not sample, pattern or automation
    tracks).

    $SAMPLE_TUNING

    The response has the track's "base_note" (key) and "base_note_name",
    "written_basenote" (the value stored in the project; it differs from
    base_note on instruments the server compensates), "pitch_cents",
    "pitchrange" (LMMS limits the fine tune to 100 cents times it; it is
    raised for you), "playable_keys" when the instrument cannot play all
    keys 0-108 at the right pitch with this tuning, and a "hint" saying
    what the keys play now.

    Args:
        track_index: Zero-based index of the instrument track
        base_note: Note (name like "C4", "Bb2" or LMMS key 0-108; C4 = 48,
            A4 = 57) at which the instrument/sample sounds at its natural
            pitch. Omit to leave it alone.
        pitch_cents: Fine tune in cents, -1200 to +1200. Omit to leave it
            alone.
    """
    proj = get_project()
    inst_track, error = _tuning_track(proj, track_index)
    if error:
        return json.dumps({"error": error})
    try:
        key = None if base_note is None else pitch.parse_base_note(base_note)
        tuning = pitch.set_tuning(inst_track, key, pitch_cents)
    except ValueError as exc:
        return json.dumps({"error": f"Tuning NOT changed: {exc}"})
    report = pitch.tuning_report(inst_track)
    if tuning == pitch.Tuning():
        message = (
            f"Track {track_index} tuning (unchanged): base note "
            f"{report['base_note_name']} ({report['base_note']}), fine tune "
            f"{report['pitch_cents']:+g} cents."
        )
    else:
        proj._modified = True
        message = (
            f"Set track {track_index} to base note "
            f"{report['base_note_name']} ({report['base_note']}), fine tune "
            f"{report['pitch_cents']:+g} cents."
        )
    return json.dumps({
        "track_index": track_index, "message": message, **report})


@_tool
def add_sample_track(
    name: str,
    mixer_channel: int = 0,
    volume: int = 100,
    panning: int = 0,
) -> str:
    """Add a sample track for arranging audio files.

    Args:
        name: Track name
        mixer_channel: Mixer channel number (0=Master). LMMS 1.2.x cannot
            route sample tracks (they always play on Master), so a channel
            other than 0 is refused there; LMMS 1.3+ routes them. The
            channel must exist (add_mixer_channel creates one).
        volume: Track volume (0-200)
        panning: Track panning (-100 to +100)
    """
    proj = get_project()
    try:
        result = proj.add_track(
            "sample", name,
            mixer_channel=mixer_channel,
            volume=volume,
            panning=panning,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(result)


@_tool
def add_automation_track(name: str = "Automation track") -> str:
    """Add an automation track for parameter automation.

    Args:
        name: Track name
    """
    proj = get_project()
    result = proj.add_track("automation", name)
    return json.dumps(result)


@_tool
def add_pattern_track(name: str = "Pattern 0") -> str:
    """Add a beat/bassline pattern track for drum sequencing.

    A default inner kick instrument is created so notes can be added
    immediately via add_note. Trigger it in the song editor with
    place_bb_clip.

    Args:
        name: Pattern track name
    """
    proj = get_project()
    result = proj.add_track("pattern", name)
    return json.dumps(result)


@_tool
def remove_track(track_index: int) -> str:
    """Remove a track by its index.

    Args:
        track_index: Zero-based index of the track to remove
    """
    proj = get_project()
    result = proj.remove_track(track_index)
    return json.dumps(result)


@_tool
def get_track(track_index: int) -> str:
    """Get detailed information about a specific track.

    Instrument tracks with a base note other than 57 (A4) or a fine tune
    also show "base_note", "base_note_name" and "pitch_cents" (see
    set_track_tuning).

    Args:
        track_index: Zero-based index of the track
    """
    proj = get_project()
    info = proj.get_info()
    if track_index >= len(info.tracks):
        return json.dumps({"error": f"Track index {track_index} out of range. Have {len(info.tracks)} tracks."})
    track = info.tracks[track_index]
    return json.dumps(track.to_dict(), indent=2)


@_tool
def list_tracks() -> str:
    """List all tracks in the current project with their index, name, and type.

    Instrument tracks with a base note other than 57 (A4) or a fine tune
    also show "base_note", "base_note_name" and "pitch_cents" (see
    set_track_tuning).
    """
    proj = get_project()
    info = proj.get_info()
    tracks = [
        {
            "index": t.index,
            "name": t.name,
            "type": t.type_name,
            "instrument": t.instrument,
            **t.tuning_dict(),
            "muted": t.muted,
            "patterns": len(t.patterns),
        }
        for t in info.tracks
    ]
    return json.dumps({"tracks": tracks, "count": len(tracks)}, indent=2)


def _bars_text(ticks: int) -> str:
    """A tick position as a bar number: 384 -> 'bar 2', 480 -> 'bar 2.5'."""
    return f"bar {round(ticks / TICKS_PER_BAR, 2):g}"


def _automation_note(
    proj, parent: ET.Element, name: str, what: str, value: str,
) -> dict:
    """The response fields for a model that was just set, or {}.

    When the model `name` of `parent` is played by automation patterns the
    value just written is its static value, not what is heard everywhere:
    "automated": True and a "note" say where each applies. `what` names
    the parameter ("Track 2's volume"), `value` is the value as given.
    """
    patterns = xml_parser.model_automation(proj.root, parent, name)
    if not patterns:
        return {}
    listed = ", ".join(
        f"automation track {p['track_index']} '{p['track']}'" for p in patterns)
    first = min(p["start"] for p in patterns)
    last = max(p["last_point"] for p in patterns)
    if first <= 0:
        timing = (
            f"the automation starts at the beginning of the song, so "
            f"{value} (its static value) is not heard in playback: the "
            f"curve overrides it wherever it has points, and after its last "
            f"point, at {_bars_text(last)}, LMMS keeps the last curve value."
        )
    else:
        timing = (
            f"{value} is its static value: it applies from the start of the "
            f"song until the automation begins at {_bars_text(first)}. From "
            f"there the curve overrides it wherever it has points (and "
            f"after its last point, at {_bars_text(last)}, LMMS keeps the "
            f"last curve value), so the curve decides what is heard there."
        )
    return {
        "automated": True,
        "note": (
            f"{what} is automated ({listed}): {timing} To change the "
            f"automated part, remove the automation track with remove_track "
            f"and create the curve again with add_automation."
        ),
    }


def _head_automation_note(proj, name: str, what: str, value: str) -> dict:
    """_automation_note for a song setting (tempo, master volume/pitch)."""
    head = xml_parser.find_head(proj.root)
    if head is None:
        return {}
    return _automation_note(proj, head, name, what, value)


def _set_track_model(
    track_index: int, attr: str, label: str, value: int,
) -> str:
    """Set the volume or panning model of an instrument or sample track.

    The model is written where the project stores it (attribute, or the
    child element of an automated model, see xml_parser.set_model_value).
    """
    proj = get_project()
    from .xml_parser import find_tracks
    tracks = find_tracks(proj.root)
    if track_index >= len(tracks):
        return json.dumps({"error": f"Track index {track_index} out of range"})

    track = tracks[track_index]
    track_type = int(track.get("type", "0"))

    settings = None
    if track_type == 0:
        settings = track.find("instrumenttrack")
    elif track_type == 2:
        settings = track.find("sampletrack")

    result: dict = {}
    if settings is not None:
        xml_parser.set_model_value(settings, attr, str(value))
        result = _automation_note(
            proj, settings, attr, f"Track {track_index}'s {label}",
            str(value))

    proj._modified = True
    return json.dumps({
        "message": f"Set track {track_index} {label} to {value}", **result})


@_tool
def set_track_volume(track_index: int, volume: int) -> str:
    """Set the volume of a track.

    If the track's volume is automated (add_automation), the value is its
    static one: it applies until the automation curve starts, and the curve
    overrides it wherever it has points. The response then has
    "automated": true and a "note" saying where each applies.

    Args:
        track_index: Zero-based track index
        volume: Volume level (0-200, 100=normal)
    """
    return _set_track_model(track_index, "vol", "volume", volume)


@_tool
def set_track_panning(track_index: int, panning: int) -> str:
    """Set the panning of a track.

    If the track's panning is automated (add_automation), the value is its
    static one: it applies until the automation curve starts, and the curve
    overrides it wherever it has points. The response then has
    "automated": true and a "note" saying where each applies.

    Args:
        track_index: Zero-based track index
        panning: Pan value (-100=left, 0=center, 100=right)
    """
    return _set_track_model(track_index, "pan", "panning", panning)


@_tool
def set_track_mixer_channel(track_index: int, mixer_channel: int) -> str:
    """Route an existing track to another mixer channel.

    The channel must exist (add_mixer_channel creates one; get_mixer_channels
    lists them): LMMS plays a track routed to a channel the project lacks
    on another channel, so such a route is refused and nothing changes. Use
    this to fix the "warnings" load_project or load_instrument_preset give
    about a track on a missing channel.

    It routes an instrument track, a sample track, or every instrument
    inside a pattern track (the response lists them). LMMS 1.2.x cannot
    route sample tracks (they always play on Master), so a sample track
    there only accepts mixer_channel=0 (nothing to change). The channel is
    written with the spelling the project already uses.

    Args:
        track_index: Zero-based track index
        mixer_channel: Mixer channel number (0=Master)
    """
    proj = get_project()
    error = _track_index_error(proj, track_index)
    if error:
        return json.dumps({"error": error})
    if isinstance(mixer_channel, bool) or not isinstance(mixer_channel, int):
        return json.dumps({
            "error": f"mixer_channel must be an integer, got "
                     f"{mixer_channel!r}."})
    try:
        routed = xml_parser.set_track_mixer_channel(
            proj.root, track_index, mixer_channel)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    channel = xml_parser.channel_label(proj.root, mixer_channel)
    result = {
        "track_index": track_index,
        "mixer_channel": mixer_channel,
        "previous_mixer_channel": (
            routed["routed"][0]["previous"] if routed["routed"] else 0),
    }
    if routed["type"] == "Pattern":
        result["routed"] = routed["routed"]
    if routed["routed"]:
        proj._modified = True
        what = (
            f"every instrument of pattern track {track_index} "
            f"'{routed['track']}'" if routed["type"] == "Pattern"
            else f"track {track_index} '{routed['track']}'")
        result["message"] = f"Routed {what} to mixer channel {channel}"
    else:
        result["message"] = (
            f"Track {track_index} '{routed['track']}' is a sample track: "
            f"LMMS 1.2.x always plays those on Master (channel 0), so there "
            f"is nothing to change")
    return json.dumps(result)


@_tool
def mute_track(track_index: int, muted: bool = True) -> str:
    """Mute or unmute a track.

    Args:
        track_index: Zero-based track index
        muted: True to mute, False to unmute
    """
    proj = get_project()
    from .xml_parser import find_tracks
    tracks = find_tracks(proj.root)
    if track_index >= len(tracks):
        return json.dumps({"error": f"Track index {track_index} out of range"})

    xml_parser.set_model_value(tracks[track_index], "muted", "1" if muted else "0")
    proj._modified = True
    result = _automation_note(
        proj, tracks[track_index], "muted", f"Track {track_index}'s mute",
        "muted" if muted else "unmuted")
    return json.dumps({
        "message": f"Track {track_index} {'muted' if muted else 'unmuted'}",
        **result})


@_tool
def solo_track(track_index: int, solo: bool = True) -> str:
    """Solo or unsolo a track.

    Args:
        track_index: Zero-based track index
        solo: True to solo, False to unsolo
    """
    proj = get_project()
    from .xml_parser import find_tracks
    tracks = find_tracks(proj.root)
    if track_index >= len(tracks):
        return json.dumps({"error": f"Track index {track_index} out of range"})

    xml_parser.set_model_value(tracks[track_index], "solo", "1" if solo else "0")
    proj._modified = True
    result = _automation_note(
        proj, tracks[track_index], "solo", f"Track {track_index}'s solo",
        "soloed" if solo else "unsoloed")
    return json.dumps({
        "message": f"Track {track_index} {'soloed' if solo else 'unsoloed'}",
        **result})


# ──────────────────────────────────────────────────────────────────
# NOTE / PATTERN TOOLS
# ──────────────────────────────────────────────────────────────────


def _as_int(value, label: str) -> int:
    """Return value as an int, or raise ValueError saying what is wrong.

    Whole floats (48.0) are accepted; LMMS rejects float strings such as
    "48.0" in integer attributes, so they are normalised here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an integer, got {value!r}.")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{label} must be a whole number, got {value!r}.")
        return int(value)
    return value


def _validate_note_fields(
    key, pos, length, volume, panning, pattern_index=None
) -> dict:
    """Validate one note's fields; return them normalised as ints.

    key may be an LMMS key number (0-108) or a note name ("C4", "Bb2").
    Raises ValueError with an actionable message on the first problem.
    """
    if isinstance(key, str):
        key = _note_name_to_key(key)
    elif isinstance(key, bool) or not isinstance(key, (int, float)):
        raise ValueError(
            f"key must be an LMMS key number {KEY_MIN}-{KEY_MAX} or a note "
            f"name like 'Bb2', got {key!r}."
        )
    else:
        key = _as_int(key, "key")
    if not KEY_MIN <= key <= KEY_MAX:
        raise ValueError(
            f"key {key} is out of range: valid LMMS keys are "
            f"{KEY_RANGE_TEXT} ({KEY_NUMBERING_TEXT} "
            f"If {key} is a MIDI note number, use {key - MIDI_NOTE_OFFSET}.)"
        )

    pos = _as_int(pos, "pos")
    if pos < 0:
        raise ValueError(
            f"pos must be >= 0 ticks from the pattern start "
            f"(192 = 1 bar), got {pos}."
        )

    length = _as_int(length, "length")
    if length <= 0:
        raise ValueError(
            f"length must be > 0 ticks (12 = 1/16, 24 = 1/8, 48 = 1/4, "
            f"192 = 1 bar), got {length}."
        )

    fields = {
        "key": key,
        "pos": pos,
        "length": length,
        "volume": _as_int(volume, "volume"),
        "panning": _as_int(panning, "panning"),
    }
    if pattern_index is not None:
        fields["pattern_index"] = _as_int(pattern_index, "pattern_index")
    return fields


def _track_index_error(proj: LMMSProject, track_index) -> str | None:
    """Return an error message if track_index is not a valid track, else None."""
    if isinstance(track_index, bool) or not isinstance(track_index, int):
        return f"track_index must be an integer, got {track_index!r}."
    count = len(xml_parser.find_tracks(proj.root))
    if count == 0:
        return (
            f"Track index {track_index} out of range: the project has no "
            f"tracks yet. Add one with add_instrument_track first."
        )
    if not 0 <= track_index < count:
        return (
            f"Track index {track_index} out of range (project has {count} "
            f"tracks, valid indices 0-{count - 1}). Use list_tracks to see "
            f"the tracks."
        )
    return None


def _add_checked_note(proj: LMMSProject, track_index: int, note: dict) -> dict:
    """Validate one note dict and add it. Raises ValueError / IndexError."""
    fields = _validate_note_fields(
        note.get("key", DEFAULT_KEY),
        note.get("pos", 0),
        note.get("length", 48),
        note.get("volume", 100),
        note.get("panning", 0),
        note.get("pattern_index"),
    )
    return proj.add_note(track_index=track_index, **fields)


@_tool
def add_note(
    track_index: int,
    key: int = DEFAULT_KEY,
    pos: int = 0,
    length: int = 48,
    volume: int = 100,
    panning: int = 0,
    pattern_index: int | None = None,
) -> str:
    """Add a note to a track's pattern.

    Keys are LMMS keys: middle C (C4) = 48, A4 = 57 = 440 Hz. A MIDI note
    number is key + 12.

    On invalid input (key outside 0-108, negative pos, length <= 0, bad
    track or pattern index) nothing is added and the response is
    {"error": "..."} explaining what to fix. A key the track's instrument
    cannot play at the right pitch (see "playable_keys" of
    add_instrument_track) is still added but the response has a "warning".

    Args:
        track_index: Zero-based track index (must be an instrument or pattern track)
        key: LMMS key number (0-108, C0 to C9). 48=C4 (middle C), 57=A4
            (440 Hz), 108=C9. To use a note name instead (e.g. 'Bb2') call
            add_note_by_name.
        pos: Position in ticks RELATIVE TO THE PATTERN START
            (192 ticks = 1 bar, 48 = 1/4 note). Notes are placed inside the
            target pattern; if a note ends beyond the pattern clip length,
            the clip is extended automatically (notes beyond the clip end
            would otherwise be silent).
        length: Note length in ticks (12 = 1/16 note, 24 = 1/8, 48 = 1/4 (one beat),
            96 = 1/2, 192 = 1 bar)
        volume: Note velocity (0-200, 100=normal)
        panning: Note panning (-100 to +100)
        pattern_index: Which pattern on the track to edit (default: first).
            Use place_pattern to create additional patterns at specific
            song positions first.
    """
    proj = get_project()
    error = _track_index_error(proj, track_index)
    if error:
        return json.dumps({"error": error})
    try:
        result = _add_checked_note(proj, track_index, {
            "key": key,
            "pos": pos,
            "length": length,
            "volume": volume,
            "panning": panning,
            "pattern_index": pattern_index,
        })
    except (ValueError, IndexError) as e:
        return json.dumps({"error": f"Note NOT added: {e}"})
    return json.dumps(result)


@_tool
def add_note_by_name(
    track_index: int,
    note_name: str = "C4",
    pos: int = 0,
    length: int = 48,
    volume: int = 100,
) -> str:
    """Add a note using a note name (e.g. 'C4', 'A#3', 'F#5', 'Bb2').

    Sharps (#) and flats (b) are both accepted and enharmonic spellings
    map to the same key: 'Bb2' = 'A#2' = key 34, 'Db3' = 'C#3' = 37,
    'Cb4' = 'B3' = 47, 'B#3' = 'C4' = 48. The note letter is
    case-insensitive ('bb2' = Bb2, 'b3' = B3). Valid range is C0 (key 0)
    to C9 (key 108). Keys are LMMS keys: middle C (C4) = 48, A4 = 57 =
    440 Hz. A MIDI note number is key + 12.

    On invalid input (unknown note name, out-of-range note, negative pos,
    length <= 0, bad track index) nothing is added and the response is
    {"error": "..."} explaining the accepted format.

    Args:
        track_index: Zero-based track index
        note_name: Note name like 'C4', 'A#3', 'F5', 'Bb2' (letter A-G,
            optional # or b, octave 0..9; C0 to C9)
        pos: Position in ticks RELATIVE TO THE PATTERN START
            (192 ticks = 1 bar, 48 = 1/4 note)
        length: Note length in ticks (12 = 1/16 note, 24 = 1/8, 48 = 1/4 (one beat),
            96 = 1/2, 192 = 1 bar)
        volume: Note velocity (0-200)
    """
    try:
        key = _note_name_to_key(note_name)
    except ValueError as e:
        return json.dumps({"error": f"Note NOT added: {e}"})

    proj = get_project()
    error = _track_index_error(proj, track_index)
    if error:
        return json.dumps({"error": error})
    try:
        result = _add_checked_note(proj, track_index, {
            "key": key,
            "pos": pos,
            "length": length,
            "volume": volume,
        })
    except (ValueError, IndexError) as e:
        return json.dumps({"error": f"Note NOT added: {e}"})
    return json.dumps(result)


@_tool
def add_notes_batch(
    track_index: int,
    notes: list[dict],
) -> str:
    """Add multiple notes to a track at once.

    Each note is validated on its own: invalid notes are skipped (NOT
    added) and reported, valid ones are still added. Check the result:
    {"added": <count actually added>, "failed": <count>, "errors":
    [{"index": i, "note": <the input>, "error": "..."}], "notes": [...]}.
    When any note failed there is also a top-level "warning"; when every
    note failed (or the track index is bad) there is a top-level "error".
    Notes outside the keys the track's instrument plays at the right pitch
    are added too, with a "warning" on the note and in the top-level one.

    Args:
        track_index: Zero-based track index
        notes: List of note objects, each with: key (LMMS key int 0-108, or
            a note name str like "C4", "A#3", "Bb2" - sharps and flats both
            work, e.g. "Bb2" = "A#2" = 34; range C0..C9; middle C = C4 = 48,
            A4 = 57 = 440 Hz; a MIDI note number is key + 12),
            pos (ticks >= 0, relative to the pattern start; 192 = 1 bar,
            48 = 1/4 note), length (ticks > 0, default 48 = 1/4 note;
            24 = 1/8, 12 = 1/16, 192 = 1 bar), volume (0-200, default 100),
            panning (-100 to +100, default 0), pattern_index (optional)
    """
    proj = get_project()
    error = _track_index_error(proj, track_index)
    if error:
        return json.dumps({"error": error, "added": 0, "failed": 0})
    if not isinstance(notes, list) or not notes:
        return json.dumps({
            "error": "notes must be a non-empty list of note objects, e.g. "
                     '[{"key": "Bb2", "pos": 0, "length": 48}].',
            "added": 0,
            "failed": 0,
        })

    added = []
    errors = []
    for index, note_data in enumerate(notes):
        try:
            if not isinstance(note_data, dict):
                raise ValueError(
                    "each note must be an object like "
                    '{"key": "C4", "pos": 0, "length": 48}.'
                )
            added.append(_add_checked_note(proj, track_index, note_data))
        except (ValueError, IndexError) as e:
            errors.append({"index": index, "note": note_data, "error": str(e)})

    response: dict = {
        "added": len(added),
        "failed": len(errors),
        "errors": errors,
        "notes": added,
    }
    if errors and not added:
        response["error"] = (
            f"None of the {len(notes)} notes were added. First error "
            f"(note {errors[0]['index']}): {errors[0]['error']} "
            f"See 'errors' for every note."
        )
    elif errors:
        response["warning"] = (
            f"{len(errors)} of {len(notes)} notes were NOT added; "
            f"see errors."
        )
    out_of_range = [n for n in added if n.get("warning")]
    if out_of_range:
        keys = proj.playable_keys(track_index)
        extra = (
            f"{len(out_of_range)} of {len(added)} added notes are outside "
            f"the keys this track's instrument plays correctly "
            f"({keys[0]}-{keys[1]}); see notes[].warning."
        )
        response["warning"] = (
            f"{response['warning']} {extra}" if "warning" in response
            else extra
        )
    return json.dumps(response)


# ──────────────────────────────────────────────────────────────────
# MIXER TOOLS
# ──────────────────────────────────────────────────────────────────


@_tool
def add_mixer_channel(name: str, volume: float = 1.0) -> str:
    """Add a new mixer channel.

    The channel is automatically routed to Master via an explicit
    <send> element - LMMS drops the implicit connection when loading,
    so channels without it would be silent.

    Args:
        name: Channel name (e.g. "Drums", "Bass", "Lead")
        volume: Channel volume (0.0-2.0, 1.0=0dB)
    """
    proj = get_project()
    result = proj.add_mixer_channel(name=name, volume=volume)
    return json.dumps(result)


@_tool
def get_mixer_channels() -> str:
    """Get all mixer channels with their settings."""
    proj = get_project()
    info = proj.get_info()
    channels = [ch.to_dict() for ch in info.mixer_channels]
    return json.dumps({"channels": channels, "count": len(channels)}, indent=2)


@_tool
def set_mixer_channel_volume(channel_num: int, volume: float) -> str:
    """Set the volume of a mixer channel.

    If the channel's volume is automated (add_automation), the value is
    its static one: it applies until the automation curve starts, and the
    curve overrides it wherever it has points. The response then has
    "automated": true and a "note" saying where each applies.

    Args:
        channel_num: Channel number (0=Master)
        volume: Volume (0.0-2.0, 1.0=0dB)
    """
    proj = get_project()
    from .xml_parser import find_mixer_channels
    channels = find_mixer_channels(proj.root)
    for ch in channels:
        if int(ch.get("num", "-1")) == channel_num:
            xml_parser.set_model_value(ch, "volume", str(volume))
            proj._modified = True
            return json.dumps({
                "message": f"Set mixer channel {channel_num} volume to {volume}",
                **_automation_note(
                    proj, ch, "volume",
                    f"Mixer channel {channel_num}'s volume", str(volume)),
            })
    return json.dumps({"error": f"Mixer channel {channel_num} not found"})


@_tool
def set_mixer_channel_name(channel_num: int, name: str) -> str:
    """Rename a mixer channel.

    Args:
        channel_num: Channel number (0=Master)
        name: New channel name
    """
    proj = get_project()
    from .xml_parser import find_mixer_channels
    channels = find_mixer_channels(proj.root)
    for ch in channels:
        if int(ch.get("num", "-1")) == channel_num:
            ch.set("name", name)
            proj._modified = True
            return json.dumps({"message": f"Renamed mixer channel {channel_num} to '{name}'"})
    return json.dumps({"error": f"Mixer channel {channel_num} not found"})


# ──────────────────────────────────────────────────────────────────
# EFFECT (FXCHAIN) TOOLS
# ──────────────────────────────────────────────────────────────────


def _resolve_fxchain_target(target_type: str, target_index: int) -> ET.Element:
    """Resolve a track or mixer channel element for fxchain operations."""
    proj = get_project()
    root = proj.root
    if target_type == "track":
        track = xml_parser.find_track_element(root, target_index)
        inst_track = track.find("instrumenttrack")
        if inst_track is None:
            raise ValueError(
                f"Track {target_index} is not an instrument track"
            )
        return inst_track
    if target_type == "mixer":
        channels = xml_parser.find_mixer_channels(root)
        if not 0 <= target_index < len(channels):
            raise ValueError(
                f"Mixer channel {target_index} out of range "
                f"(0-{len(channels) - 1})"
            )
        return channels[target_index]
    raise ValueError("target_type must be 'track' or 'mixer'")


def _param_error(exc: effect_params.ParamError) -> str:
    out = {"error": str(exc)}
    if exc.valid:
        out["valid_parameters"] = exc.valid
    return json.dumps(out)


@_plugin_tool
def add_effect(
    target_type: str,
    target_index: int,
    effect: str,
    wet: float | None = None,
    enabled: bool = True,
    position: int | None = None,
    params: dict | None = None,
) -> str:
    """Add an effect to a track's or mixer channel's effect chain.

    Called with just the effect, it starts with settings that make it audible
    and musical, not the bare plugin defaults (which can be silent, subtle or
    drop the original sound). The response lists the "params" used, the "wet"
    mix, "decay_ms" and a "note" where the start needs your input. Change
    anything later with set_effect_params; see it with get_effect_chain.
    "decay_ms" is how long LMMS keeps an effect running through silence: at
    LMMS's default of 1 it cuts echoes that come after a gap and reverb
    tails, so effects with echoes or tails start at 8000.

    Starting settings:
    $EFFECT_STARTS

    Args:
        target_type: "track" or "mixer"
        target_index: Track index or mixer channel number (0=Master)
        effect: Built-in LMMS effect name. Valid in the installed LMMS:
            $EFFECTS
        wet: Wet/dry mix 0.0-1.0 (0=original sound only, 1=effect only).
            Default: the effect's own start value above. Effects that output
            only the processed signal start below 1.0 so the original
            sound stays; pass 1.0 explicitly to get the effect alone.
        enabled: Whether the effect is active
        position: Chain position to insert at (None=end of chain)
        params: Optional {name: value} settings on top of the starting
            settings, e.g. {"time_s": 0.375, "feedback": 0.5} ("decay_ms"
            too). Names, ranges and units: see set_effect_params. A bad name
            or value adds nothing and the error says what is valid.
    """
    av = _availability
    normalized = effect.strip().lower()
    # Refuse effects the installed LMMS lacks: it would silently ignore
    # them when loading the project. Nothing is modified.
    if normalized in effects_mod.KNOWN_EFFECTS and normalized not in av.effects:
        return _refusal("effect", normalized, av)
    try:
        parent = _resolve_fxchain_target(target_type, target_index)
        result = effects_mod.add_effect(
            parent, effect, wet, enabled, position,
            available=_effect_choices(av), params=params,
            bpm=get_project().get_info().bpm,
        )
        warning = _verification_warning(av, normalized)
        if warning:
            result["warning"] = warning
        return json.dumps(result)
    except effect_params.ParamError as exc:
        return _param_error(exc)
    except ValueError as exc:
        return json.dumps({
            "error": str(exc),
            "valid_effects": _effect_choices(av),
            "recommendations": lmms_app.adapt_recommendations(
                effects_mod.EFFECT_RECOMMENDATIONS, av.effects
            ),
        })


@_plugin_tool
def set_effect_params(
    target_type: str,
    target_index: int,
    effect: str | int,
    params: dict,
) -> str:
    """Change the parameters of an effect already in a chain.

    Only the parameters you pass change; the others keep their values.
    Names are checked and values must be in range: an unknown name or an
    out-of-range value changes nothing and the error lists the valid names
    or the valid range. Each parameter has a friendly name (below); its
    real LMMS attribute name is accepted too, and lmms://reference/effects
    shows both with descriptions. Every effect also has "wet" (0.0-1.0, the
    wet/dry mix) and "decay_ms" (1-8000, how long LMMS keeps it running
    through silence; raise it when echoes or a tail are cut off). The
    response lists the values as stored.

    Parameters (name, range unit, plugin default):
    $EFFECT_PARAMS

    Args:
        target_type: "track" or "mixer"
        target_index: Track index or mixer channel number
        effect: Effect name (e.g. "delay") or chain position (e.g. 0)
        params: {name: value}, e.g. {"time_s": 0.375, "feedback": 0.5}.
            Numbers for knobs, true/false for switches, text for choices
            (a list for the curve parameters: shape, tap_levels_db,
            tap_cutoff_hz).
    """
    try:
        parent = _resolve_fxchain_target(target_type, target_index)
        return json.dumps(effects_mod.set_effect_params(parent, effect, params))
    except effect_params.ParamError as exc:
        return _param_error(exc)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


@_tool
def remove_effect(target_type: str, target_index: int, effect: str | int) -> str:
    """Remove an effect from a chain by name or chain position.

    Args:
        target_type: "track" or "mixer"
        target_index: Track index or mixer channel number
        effect: Effect name (e.g. "delay") or position (e.g. 0)
    """
    try:
        parent = _resolve_fxchain_target(target_type, target_index)
        result = effects_mod.remove_effect(parent, effect)
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


@_tool
def toggle_effect(
    target_type: str, target_index: int, effect: str | int, enabled: bool
) -> str:
    """Enable or disable an effect without removing it.

    Args:
        target_type: "track" or "mixer"
        target_index: Track index or mixer channel number
        effect: Effect name or chain position
        enabled: True to enable, False to bypass
    """
    try:
        parent = _resolve_fxchain_target(target_type, target_index)
        result = effects_mod.set_effect_enabled(parent, effect, enabled)
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


@_tool
def get_effect_chain(target_type: str, target_index: int) -> str:
    """List all effects on a track's or mixer channel's effect chain.

    Each effect shows its position, whether it is enabled, its wet mix,
    decay_ms and "params": the parameters that differ from the plugin
    defaults (friendly names, see set_effect_params).

    Args:
        target_type: "track" or "mixer"
        target_index: Track index or mixer channel number
    """
    try:
        parent = _resolve_fxchain_target(target_type, target_index)
        return json.dumps({
            "target": f"{target_type}[{target_index}]",
            "effects": effects_mod.list_effects(parent),
        })
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


# ──────────────────────────────────────────────────────────────────
# ZYNADDSUBFX PRESET / PARAMETER TOOLS
# ──────────────────────────────────────────────────────────────────


@_tool
def list_zyn_presets(category: str | None = None) -> str:
    """List available ZynAddSubFX presets (.xiz files).

    Args:
        category: Optional category filter (e.g. "Bass", "Strings",
            "Synth", "Pads", "Brass"). Omit to list all categories' presets.
    """
    base = zyn_presets.get_presets_dir()
    if base is None:
        return json.dumps({
            "error": "No ZynAddSubFX presets directory found.",
            "hint": "Set the LMMS_PRESETS_DIR environment variable to your "
                    "LMMS presets/ZynAddSubFX folder (e.g. "
                    "/usr/share/lmms/presets/ZynAddSubFX on Linux or "
                    "data/presets/ZynAddSubFX on Windows).",
        })
    try:
        presets = zyn_presets.list_presets(category)
    except ValueError as exc:
        return json.dumps({
            "error": str(exc),
            "categories": zyn_presets.list_categories(),
        })
    return json.dumps({
        "presets_dir": str(base),
        "categories": zyn_presets.list_categories(),
        "count": len(presets),
        "presets": presets[:200],
        "truncated": len(presets) > 200,
    }, indent=2)


@_tool
def load_zyn_preset(track_index: int, preset: str) -> str:
    """Load a ZynAddSubFX preset (.xiz) into a zynaddsubfx instrument track.

    The track must use the 'zynaddsubfx' instrument. Accepted forms for
    `preset`: "Category/Name" (e.g. "Strings/Dark Strings", preferred
    because it is unambiguous), a plain "Name" (e.g. "Bass 1"), the file
    name with its number ("0034-Dark Strings"), or an absolute path to a
    .xiz file. Numeric prefixes are optional and case is ignored.

    The response says exactly what was loaded: `file` is the .xiz file
    used, `preset` is its "Category/Name", and `internal_name` is the
    name stored inside the file. The internal name often differs from
    the file name; that is normal and does not mean the wrong preset was
    loaded. If a plain name matched several presets, `other_matches`
    lists the alternatives; pass "Category/Name" to choose one. If
    nothing matches, the error suggests close names; list_zyn_presets
    shows everything available.

    Pitch: key 57 plays A4 = 440 Hz with every preset. Some presets carry
    their own octave shift (basses are often an octave lower); it is
    compensated through the track's base note and reported as
    "preset_octave" when it is not 0.

    Args:
        track_index: Index of the target track
        preset: "Category/Name", plain name, "0034-Name" or absolute path
            (a path starting with ~ is in the home folder)
    """
    proj = get_project()
    try:
        preset_xml, match = zyn_presets.load_preset(_expand_user(preset))
        result = xml_parser.embed_zyn_preset(
            proj.root, track_index, preset_xml, match.name
        )
    except zyn_presets.PresetNotFoundError as exc:
        return json.dumps({
            "error": str(exc),
            "suggestions": exc.suggestions,
        })
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return json.dumps({"error": str(exc)})

    internal_name = result.get("internal_name")
    response = {
        "track_index": track_index,
        "requested": preset,
        "file": str(match.path),
        "preset": match.name,
        "internal_name": internal_name,
        "message": (
            f"Loaded ZynAddSubFX preset '{match.name}' (file {match.path}) "
            f"into track {track_index}"
        ),
    }
    notes = []
    keys = proj.playable_keys(track_index)
    if keys:
        response["playable_keys"] = list(keys)
    octave = result.get("preset_octave", 0)
    if octave:
        response["preset_octave"] = octave
        notes.append(
            f"This preset transposes by {octave:+d} octave(s) itself; the "
            f"track's base note compensates it so key 57 still plays "
            f"A4 = 440 Hz."
        )
    if internal_name and internal_name.lower() != (
        match.name.rsplit("/", 1)[-1].lower()
    ):
        notes.append(
            f"The preset file stores the internal name '{internal_name}', "
            f"which differs from its file name. This is normal; "
            f"'{match.name}' is the preset that was loaded."
        )
    if match.other_matches:
        response["other_matches"] = match.other_matches
        more = match.other_total - len(match.other_matches)
        extra = f" (and {more} more)" if more > 0 else ""
        notes.append(
            f"'{preset}' matched several presets; loaded '{match.name}'. "
            f"To pick another, pass \"Category/Name\" from "
            f"other_matches{extra}."
        )
    if notes:
        response["note"] = " ".join(notes)
    return json.dumps(response)


@_tool
def set_zyn_params(
    track_index: int,
    portamento: int | None = None,
    filterfreq: int | None = None,
    filterq: int | None = None,
    bandwidth: int | None = None,
    fmgain: int | None = None,
    rescenterfreq: int | None = None,
    resbandwidth: int | None = None,
) -> str:
    """Set ZynAddSubFX global parameters on a zynaddsubfx track.

    All values are 0-127 as in the ZynAddSubFX UI. Only provided
    parameters are changed.

    Args:
        track_index: Index of the zynaddsubfx track
        portamento: Portamento amount (0-127)
        filterfreq: Filter cutoff frequency (0-127)
        filterq: Filter resonance/Q (0-127)
        bandwidth: Bandwidth (0-127)
        fmgain: FM gain (0-127)
        rescenterfreq: Resonance center frequency (0-127)
        resbandwidth: Resonance bandwidth (0-127)
    """
    proj = get_project()
    params = {}
    for name, value in [
        ("portamento", portamento), ("filterfreq", filterfreq),
        ("filterq", filterq), ("bandwidth", bandwidth),
        ("fmgain", fmgain), ("rescenterfreq", rescenterfreq),
        ("resbandwidth", resbandwidth),
    ]:
        if value is not None:
            if not 0 <= value <= 127:
                return json.dumps({
                    "error": f"{name} must be 0-127, got {value}"
                })
            params[name] = value
    if not params:
        return json.dumps({"error": "No parameters provided"})
    try:
        result = xml_parser.set_instrument_params(proj.root, track_index, params)
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


# ──────────────────────────────────────────────────────────────────
# INSTRUMENT PRESETS (.xpf)
# ──────────────────────────────────────────────────────────────────

_MAX_PRESETS_LISTED = 200
_PRESET_EXAMPLES = 5


def _canonical_instrument(plugin: str) -> str:
    """The offered instrument name for a plugin name as presets store it.

    "OPL2" and "opl2" give opulenz, "papu" freeboy, "stk" malletsstk;
    other names are lowercased (custom plugins).
    """
    low = plugin.strip().lower()
    for known in KNOWN_INSTRUMENTS:
        if low in lmms_app.plugin_candidates(known):
            return known
    return INSTRUMENT_ALIASES.get(low, low)


def _preset_instrument_offered(instrument: str, av) -> bool:
    """Whether presets of this (canonical) instrument can be loaded here."""
    if instrument == "vestige":
        return False
    if instrument in KNOWN_INSTRUMENTS:
        return instrument in av.instruments
    return (instrument in av.installed
            and instrument not in lmms_app.NON_TRACK_PLUGINS)


def _preset_plugin_refusal(plugin: str, instrument: str, av) -> str:
    """JSON error for a preset whose instrument the installed LMMS lacks."""
    if instrument == "vestige":
        message = ("The preset hosts a VST plugin, which a preset cannot "
                   "carry over. Use add_vst_track for VST plugins.")
        payload: dict = {}
    elif instrument in av.needs_file:
        message = _needs_file_reason(instrument, av) + "."
        payload = {}
    elif instrument in KNOWN_INSTRUMENTS:
        message = lmms_app.unavailable_reason(instrument, av.version)
        alternative = lmms_app.PLUGIN_SUBSTITUTES.get(instrument)
        payload = {}
        if alternative in av.instruments:
            message += f" '{alternative}' could stand in for it."
            payload["alternative"] = alternative
    else:
        message = (f"The preset's instrument '{plugin}' is not installed in "
                   f"{_lmms_label(av)}.")
        payload = {}
    payload["error"] = (
        f"{message} Choose a preset of an instrument that is available "
        f"(list_instrument_presets shows those)."
    )
    payload["available_instruments"] = _instrument_choices(av)
    return json.dumps(payload)


def _install_preset(inst_track, preset, plugin: str, instrument: str):
    """Build the <instrumenttrack> that results from loading a preset.

    Works on a copy of `inst_track` (the caller swaps it in once it is
    complete, so a failure changes nothing). The preset replaces the
    instrument and its settings, envelopes/filter, chord and arpeggio
    settings, the effect chain (only when the preset has effects) and the
    tuning; the track keeps its notes (they live beside the instrument
    track), volume, panning, mixer channel, MIDI port and name.

    Returns (new element, replaced parts, kept parts, note or None).
    """
    new = copy.deepcopy(inst_track)
    had_effects = new.find("fxchain/effect") is not None
    stored = preset.plugin.replace(" ", "").lower()
    preset.instrument.set("name", plugin)
    for child in preset.instrument[:1]:
        if child.tag.lower() == stored:
            child.tag = plugin.replace(" ", "")

    inst_presets.put_child(new, preset.instrument, first=True)
    replaced = ["instrument"]
    for tag, label in inst_presets.SOUND_PARTS.items():
        if tag in preset.parts:
            inst_presets.put_child(new, preset.parts[tag])
            replaced.append(label)
    kept = []
    if preset.fxchain is not None:
        inst_presets.put_child(new, preset.fxchain)
        count = len(preset.fxchain.findall("effect"))
        replaced.append(f"effects ({count})")
    elif had_effects:
        kept.append("effects")

    # The tuning is what the preset's author heard in LMMS: stored as is.
    cents = preset.pitch
    pitchrange = max(preset.pitchrange, math.ceil(abs(cents) / 100), 1)
    pitch.apply_tuning(new, pitch.Tuning(preset.basenote, cents, pitchrange))
    replaced.append("tuning")
    note = None
    offset = pitch.track_pitch_offset(new)
    if offset and preset.basenote == DEFAULT_BASE_NOTE:
        # The default base note of a plugin LMMS plays off pitch: calibrate
        # it like a new track so key 57 is A4 = 440 Hz.
        try:
            written = pitch.set_base_note(new, DEFAULT_BASE_NOTE)
        except ValueError:
            pass
        else:
            note = (
                f"The preset has the default base note 57, at which LMMS "
                f"plays {plugin} {abs(offset)} semitones "
                f"{'below' if offset > 0 else 'above'} the key. The track's "
                f"base note is written as {written} to compensate, so key "
                f"57 = A4 = 440 Hz like on a new {instrument} track."
            )
    return new, replaced, kept, note


def _preset_mix_note(index: int, preset, inst_track) -> str | None:
    """Sentence saying the preset was saved with another volume / panning
    than the track that keeps its own."""
    parts = []
    for label, stored, current, tool in (
        ("volume", preset.volume,
         int(inst_presets.number(
             xml_parser.model_value(inst_track, "vol", "100"), 100)),
         "set_track_volume"),
        ("panning", preset.panning,
         int(inst_presets.number(
             xml_parser.model_value(inst_track, "pan", "0"), 0)),
         "set_track_panning"),
    ):
        if stored != current:
            parts.append(
                f"{label} {stored} (the track keeps {current}; "
                f"{tool}({index}, {stored}) to match)")
    if not parts:
        return None
    return "The preset was saved with " + " and ".join(parts) + "."


@_tool
def load_instrument_preset(
    track_index: int | None,
    preset: str,
    name: str | None = None,
    rename: bool = False,
) -> str:
    """Load an LMMS instrument preset (.xpf) into an instrument track.

    A preset is a finished, often tuned sound: the instrument with all its
    settings, envelopes/filter, chords/arpeggio, effects and base note, as
    LMMS's own "Save as preset" writes it. LMMS ships presets for most
    instruments and the user may have saved their own, e.g. a piano
    sample with its base note set so keys sound at their note names
    (list_instrument_presets shows what is available). For ZynAddSubFX
    (.xiz) use load_zyn_preset.

    Pass an existing instrument track as track_index, or null to create a
    new instrument track (named `name`, default the preset's name) from the
    preset in one call.

    What changes on an existing track (the response lists "replaced" and
    "kept"): REPLACED are the instrument and its settings (so the track
    changes instrument if the preset uses another), envelopes and filter,
    chords, arpeggio, the effect chain (only when the preset has effects;
    a preset without effects leaves the track's), and the tuning (base
    note, fine tune). KEPT are the notes and patterns, the track name
    (rename=True, or `name`, sets it; a preset with another volume or
    panning than the track is reported, the track keeps its own), mute and
    solo, mixer channel and MIDI port. A new track takes the preset's
    volume and panning.

    The preset's tuning is stored as the author heard it in LMMS and
    returned as "tuning" with a "hint" saying which key sounds at the
    natural pitch (for a sample preset the key the sample was recorded
    at, for synths key 57 = A4 = 440 Hz). Where LMMS plays an instrument
    off pitch (see add_instrument_track) a preset with the default base
    note 57 is compensated like a new track, and the note says so (an
    OPL2 patch only when it sounds an octave low; most presets do not).
    Change the tuning afterwards with set_track_tuning.

    Nothing is changed when the call is refused: the preset needs an
    instrument the installed LMMS does not have, or a sample/soundfont file
    that is missing (the track would be silent; the error names the file),
    or the track is not an instrument track. A relative sample path is
    found the way LMMS finds it (your LMMS samples folder, then the
    factory samples); a sample lying beside the preset, which LMMS itself
    would not find, is pointed to by its absolute path (the response says
    so).

    The response has "file" (the .xpf used), "preset" ("Folder/Name"),
    "instrument" (and "lmms_plugin" when the project stores another name,
    e.g. OPL2 for opulenz), "replaced", "kept", "tuning",
    "playable_keys" when the instrument cannot play all keys 0-108 at the
    right pitch, "other_matches" when a plain name matched several presets
    (pass "Folder/Name" to pick), "warnings" and "note". If nothing
    matches, the error suggests close names.

    Args:
        track_index: Index of an instrument track, or null to create a
            new instrument track from the preset
        preset: "Folder/Name" as list_instrument_presets shows it (e.g.
            "TripleOscillator/AmazingBubbles"; the folder is the
            instrument), a plain name ("Amazing Bubbles"; case, spaces and
            "-" "_" do not matter), or an absolute path to an .xpf file
            (a path starting with ~ is in the home folder)
        name: Name of the new track (track_index null), default the
            preset's own name; on an existing track it renames the track
        rename: Give an existing track the preset's own name
    """
    proj = get_project()
    av = _availability

    track = None
    if track_index is not None:
        error = _track_index_error(proj, track_index)
        if error is None:
            track = xml_parser.find_track_element(proj.root, track_index)
            if track.find("instrumenttrack") is None:
                kind = xml_parser.track_type_name(
                    xml_parser.get_track_type(track)).lower()
                error = (
                    f"Track {track_index} is a {kind} track, not an "
                    f"instrument track, so it cannot take an instrument "
                    f"preset. Presets load into tracks made with "
                    f"add_instrument_track (list_tracks shows the types), "
                    f"or pass track_index=null to create a new instrument "
                    f"track from the preset."
                )
        if error:
            return json.dumps({"error": error})

    try:
        match = inst_presets.find_preset(
            _expand_user(preset), _canonical_instrument)
        loaded = inst_presets.read_preset(match.path)
    except inst_presets.PresetNotFoundError as exc:
        return json.dumps({"error": str(exc), "suggestions": exc.suggestions})
    except (inst_presets.PresetError, ValueError, OSError) as exc:
        return json.dumps({"error": str(exc)})

    instrument = _canonical_instrument(loaded.plugin)
    if not _preset_instrument_offered(instrument, av):
        return _preset_plugin_refusal(loaded.plugin, instrument, av)
    plugin = av.plugin_name(instrument)

    check = inst_presets.resolve_samples(loaded, lmms_app.sample_search_dirs())
    if check.missing:
        missing = check.missing[0]
        looked = ""
        if not inst_presets.is_absolute_path(missing):
            folders = [str(loaded.path.parent)] + [
                str(d) for d in lmms_app.sample_search_dirs()]
            looked = f" (looked in {', '.join(folders)})"
        return json.dumps({"error": (
            f"Preset '{match.label}' plays the file '{missing}', which was "
            f"not found{looked}. LMMS would load nothing and the track "
            f"would be silent, so nothing was changed. Restore the file, "
            f"or choose another preset (list_instrument_presets)."
        )})

    warnings = []
    if loaded.fxchain is not None:
        dropped = inst_presets.filter_effects(
            loaded, lambda n: not av.verified or n.lower() in av.installed)
        if dropped:
            warnings.append(
                f"Left out effect(s) the installed LMMS lacks: "
                f"{', '.join(dropped)}."
            )
    for src in check.unverified:
        warnings.append(
            f"No LMMS installation was found, so the file '{src}' could "
            f"not be checked; if LMMS cannot find it the track is silent."
        )

    if track is not None:
        # the track keeps its mixer channel; a missing one is misrouted
        channel = xml_parser.get_track_mixer_channel(
            track.find("instrumenttrack"))
        try:
            xml_parser.check_mixer_channel(proj.root, channel, track_index)
        except ValueError as exc:
            warnings.append(f"The track keeps its mixer channel {channel}. {exc}")

    created = track is None
    if created:
        track_name = name or loaded.name
        volume = max(0, min(200, loaded.volume))
        panning = max(-100, min(100, loaded.panning))
        added = proj.add_track(
            "instrument", track_name, instrument=plugin, mixer_channel=0,
            volume=volume, panning=panning)
        track_index = added["track_index"]
        track = xml_parser.find_track_element(proj.root, track_index)
    inst_track = track.find("instrumenttrack")

    try:
        new, replaced, kept, note = _install_preset(
            inst_track, loaded, plugin, instrument)
    except (ValueError, KeyError, IndexError) as exc:
        if created:
            proj.remove_track(track_index)
        return json.dumps({"error": f"Preset not loaded: {exc}"})
    track[list(track).index(inst_track)] = new
    proj._modified = True

    renamed = name if (name and not created) else (
        loaded.name if rename and not created else None)
    if renamed:
        track.set("name", renamed)

    report = pitch.tuning_report(new)
    report.pop("instrument", None)
    playable = report.pop("playable_keys", None)

    notes = [note] if note else []
    if not created:
        mix = _preset_mix_note(track_index, loaded, inst_track)
        if mix:
            notes.append(mix)
        kept = ["notes", "name" if not renamed else None,
                "volume", "panning", "mixer channel", *kept]
        kept = [k for k in kept if k]
    else:
        kept = []
    if check.made_absolute:
        notes.append(
            "LMMS looks for a relative sample path in its samples folders, "
            "not beside the preset, so the sample is set by its absolute "
            "path: " + ", ".join(check.made_absolute.values()) + "."
        )
    if match.other_matches:
        more = match.other_total - len(match.other_matches)
        extra = f" (and {more} more)" if more > 0 else ""
        notes.append(
            f"'{preset}' matched several presets; loaded '{match.label}'. "
            f"To pick another, pass \"Folder/Name\" from "
            f"other_matches{extra}."
        )

    result: dict = {
        "track_index": track_index,
        "file": str(match.path),
        "preset": match.label,
        "instrument": instrument,
    }
    if created:
        result["created"] = True
    if plugin != instrument:
        result["lmms_plugin"] = plugin
    if created or renamed:
        result["name"] = track.get("name")
    result["replaced"] = replaced
    if kept:
        result["kept"] = kept
    result["tuning"] = report
    if playable:
        result["playable_keys"] = list(playable)
    if match.other_matches:
        result["other_matches"] = match.other_matches
    if warnings:
        result["warnings"] = warnings
    if notes:
        result["note"] = " ".join(notes)
    result["message"] = (
        f"Loaded preset '{match.label}' ({instrument}) "
        + (f"into a new track {track_index}" if created
           else f"into track {track_index}")
    )
    return json.dumps(result)


@_tool
def list_instrument_presets(instrument: str | None = None) -> str:
    """List the instrument presets (.xpf) load_instrument_preset can load.

    Presets come from LMMS's factory presets, the user's own LMMS presets
    folder (<LMMS working dir>/presets, where "Save as preset" puts them)
    and the folders in the LMMS_PRESET_DIRS environment variable. Only
    presets of instruments the installed LMMS offers are listed.

    Without arguments this is a short overview: per instrument the number
    of presets and a few example names. With `instrument` it lists every
    preset of that instrument (up to 200); pass a name from there as
    `preset` to load_instrument_preset.

    Args:
        instrument: Instrument to list, e.g. "tripleoscillator" or
            "opulenz" (also "opl2"). Omit for the overview.
    """
    av = _availability
    roots = inst_presets.preset_roots()
    if not roots:
        return json.dumps({
            "error": "No instrument preset folder found.",
            "hint": "LMMS keeps presets in <LMMS working dir>/presets and "
                    "its factory presets folder (e.g. /usr/share/lmms/"
                    "presets). Set the LMMS_PRESET_DIRS environment "
                    "variable to folders holding .xpf files, or pass an "
                    "absolute path to load_instrument_preset.",
        })
    files, truncated = inst_presets.scan_presets(roots)
    groups: dict[str, list[str]] = {}
    unreadable = not_offered = 0
    for f in files:
        plugin = inst_presets.peek_plugin(f.path)
        if plugin is None:
            unreadable += 1
            continue
        kind = _canonical_instrument(plugin)
        if not _preset_instrument_offered(kind, av):
            not_offered += 1
            continue
        groups.setdefault(kind, []).append(f.label)
    order = {name: i for i, name in enumerate(KNOWN_INSTRUMENTS)}
    names = sorted(groups, key=lambda n: (order.get(n, len(order)), n))

    skipped = {}
    if not_offered:
        skipped["instrument_not_available"] = not_offered
    if unreadable:
        skipped["unreadable"] = unreadable
    result: dict = {"preset_dirs": [str(r) for r in roots]}

    if instrument is not None:
        wanted = _canonical_instrument(
            INSTRUMENT_ALIASES.get(instrument.strip().lower(), instrument))
        if wanted not in groups:
            return json.dumps({
                "error": f"No presets found for instrument "
                         f"'{instrument}'.",
                "instruments_with_presets": names,
                **result,
            })
        labels = sorted(groups[wanted], key=str.lower)
        result.update({
            "instrument": wanted,
            "count": len(labels),
            "presets": labels[:_MAX_PRESETS_LISTED],
        })
        if len(labels) > _MAX_PRESETS_LISTED:
            result["truncated"] = True
    else:
        result["total"] = sum(len(v) for v in groups.values())
        result["instruments"] = {
            n: {"count": len(groups[n]),
                "examples": sorted(groups[n], key=str.lower)[:_PRESET_EXAMPLES]}
            for n in names
        }
        result["hint"] = (
            "list_instrument_presets(instrument) lists all presets of one "
            "instrument; load_instrument_preset(track_index, \"Folder/"
            "Name\") loads one."
        )
    if skipped:
        result["skipped"] = skipped
    if truncated:
        result["scan_truncated"] = (
            f"only the first {inst_presets.MAX_SCANNED} preset files were "
            f"scanned")
    return json.dumps(result)


# ──────────────────────────────────────────────────────────────────
# LMMS APP INTEGRATION (VERSION CHECK + RENDER/EXPORT)
# ──────────────────────────────────────────────────────────────────


@_tool
def get_lmms_info() -> str:
    """Get info about the installed LMMS application.

    Shows the detected LMMS version, installation path and which
    instrument/effect plugins are actually available. Use this to check
    whether a plugin is supported before using it.
    """
    exe = lmms_app.find_lmms_exe()
    if exe is None:
        return json.dumps({
            "found": False,
            "message": "LMMS installation not found. Projects can still "
                       "be created and saved, but plugin availability "
                       "cannot be verified and rendering is disabled. "
                       "Set LMMS_EXECUTABLE env var to the LMMS executable "
                       "(lmms.exe / lmms).",
        })
    av = _availability
    installed = sorted(av.installed)
    return json.dumps({
        "found": True,
        "version": av.version,
        "path": str(exe),
        "plugins_verified": av.verified,
        "plugin_count": len(installed),
        "instruments_verified_available": _ordered(
            KNOWN_INSTRUMENTS, av.instruments),
        "effects_verified_available": _ordered(
            effects_mod.KNOWN_EFFECTS, av.effects),
        "all_installed_plugins": installed,
    }, indent=2)


# What render_project does about the level of a render (dBFS).
_RENDER_TARGET_PEAK_DB = -1.0   # the peak auto_fix_clipping brings a clip to
_RENDER_QUIET_PEAK_DB = -12.0   # a render below this gets a "too quiet" note
_MASTER_VOLUME_MAX = 2.0        # LMMS's mixer faders stop at +6 dB
_RENDER_SILENCE_DB = 20 * math.log10(audio_analysis.SILENCE_PEAK)


def _db_text(value: float | None) -> float | None:
    """dB rounded to 0.1 for a response (never -0.0)."""
    return None if value is None else round(value, 1) + 0.0


def _block_json(block) -> dict:
    return {
        "bars": [block.first_bar, block.last_bar],
        "peak_dbfs": _db_text(audio_analysis.to_db(block.peak)),
        "rms_dbfs": _db_text(audio_analysis.to_db(block.rms)),
    }


def _levels_json(levels) -> dict:
    """The "levels" of a render_project response (see its docstring)."""
    blocks = levels.blocks
    audible = [b for b in blocks if not b.silent]
    return {
        "duration_s": round(levels.duration, 2),
        "peak_dbfs": _db_text(audio_analysis.to_db(levels.peak)),
        "clipped_samples": levels.clipped,
        "rms_dbfs": _db_text(audio_analysis.to_db(levels.rms)),
        "blocks": [_block_json(b) for b in blocks],
        "loudest": _block_json(max(blocks, key=lambda b: b.rms)),
        "quietest_nonsilent": (
            _block_json(min(audible, key=lambda b: b.rms)) if audible
            else None),
    }


def _run_lmms_render(
    exe, proj_path: Path, output: Path, fmt: str, samplerate: int,
    bitrate: int, oversampling: int, interp: str, float_output: bool = False,
):
    """One LMMS render of the saved project with its own private config.
    Returns (CompletedProcess, command line text); raises OSError and
    subprocess.TimeoutExpired."""
    import subprocess
    with lmms_app.isolated_config(_availability.version) as config:
        cmd = [
            str(exe), "--config", str(config),
            "render", str(proj_path),
            "-o", str(output), "-f", fmt,
            "-s", str(samplerate), "-b", str(bitrate),
            "-x", str(oversampling), "-i", interp,
        ]
        if float_output:
            cmd.append("-a")
        proc = lmms_app.run_lmms(
            cmd, capture_output=True, text=True, timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return proc, " ".join(cmd)


def _level_advice(proj, levels) -> dict:
    """Response fields that tell what gain would put the render's peak at
    -1 dBFS: "suggested_gain_db" and "suggested_master_volume" for a
    render that clips or is quiet, and a "note" saying so."""
    peak_db = audio_analysis.to_db(levels.peak)
    if peak_db is None or peak_db < _RENDER_SILENCE_DB:
        return {"note": (
            f"The render is silent (its peak is below {_RENDER_SILENCE_DB:.0f}"
            " dBFS). Check that the tracks have notes or clips, are not "
            "muted, and that file-based instruments have their file.")}
    clips = levels.peak > 1.0
    if not clips and peak_db >= _RENDER_QUIET_PEAK_DB:
        return {}
    gain_db = _RENDER_TARGET_PEAK_DB - peak_db
    before = xml_parser.master_volume(proj.root)
    volume = before * audio_analysis.from_db(gain_db)
    out = {
        "suggested_gain_db": round(gain_db, 1),
        "suggested_master_volume": round(min(volume, _MASTER_VOLUME_MAX), 4),
    }
    set_it = (
        f"set_mixer_channel_volume(0, {out['suggested_master_volume']:g})")
    if clips:
        out["note"] = (
            f"The render clips (peak {peak_db:+.1f} dBFS, "
            f"{levels.clipped} samples above 0 dBFS). Lowering the Master "
            f"volume from {before:g} to {out['suggested_master_volume']:g} "
            f"({gain_db:+.1f} dB) puts the peak at "
            f"{_RENDER_TARGET_PEAK_DB:g} dBFS: {set_it}, or render again "
            "with auto_fix_clipping=true.")
    elif volume <= _MASTER_VOLUME_MAX:
        out["note"] = (
            f"The render is quiet (peak {peak_db:+.1f} dBFS). Raising the "
            f"Master volume from {before:g} to "
            f"{out['suggested_master_volume']:g} ({gain_db:+.1f} dB) puts "
            f"the peak at {_RENDER_TARGET_PEAK_DB:g} dBFS: {set_it}.")
    else:
        out["note"] = (
            f"The render is quiet (peak {peak_db:+.1f} dBFS): "
            f"{gain_db:+.1f} dB would put the peak at "
            f"{_RENDER_TARGET_PEAK_DB:g} dBFS, more than the Master volume "
            f"can add (its maximum is {_MASTER_VOLUME_MAX:g}, +6 dB). Raise "
            "the track volumes (set_track_volume) first.")
    return out


@_tool
def render_project(
    output_path: str | None = None,
    file_format: str = "wav",
    samplerate: int = 44100,
    bitrate: int = 160,
    oversampling: int = 1,
    interpolation: str = "sincfastest",
    auto_fix_clipping: bool = True,
) -> str:
    """Export the current project to an audio file using the installed LMMS.

    This launches LMMS in headless render mode (no GUI). Requires a
    working LMMS installation. The project is saved first, then rendered.

    The render uses its own temporary LMMS config (response: "isolated_config":
    true) instead of the user's ~/.lmmsrc.xml, so it never changes the user's
    LMMS settings, is safe next to an open LMMS and other renders, and does
    not depend on the user's GUI options. Only the sample, soundfont and
    plugin folders are copied from the user's config, and effects keep
    running through silence (echoes and reverb tails are not cut off).
    Only one LMMS process of this server runs at a time; a render called
    while another runs waits for it.

    Folders of output_path that do not exist yet are created (the response
    lists them in "created_dirs"). If the output cannot be written, the
    response has an "error" saying why and which path.

    The response measures the render, so no separate analysis is needed:
    "levels" = {"duration_s", "peak_dbfs", "clipped_samples" (samples above
    0 dBFS in what LMMS produced, before any fix), "rms_dbfs", "blocks": [{
    "bars": [first, last] (1-based, inclusive), "peak_dbfs", "rms_dbfs"}]
    for every 4 bars (8, 16, ... when that would be more than 32 blocks;
    the bars follow the project's tempo and time signature), "loudest" and
    "quietest_nonsilent" (the blocks with the highest and lowest RMS; a
    block below -60 dBFS RMS, such as the tail after the last note, counts
    as silence and is not offered as the quietest)}. dB values are rounded
    to 0.1; null is digital silence. RMS covers all channels (a full-scale
    sine is -3.0 dBFS). LMMS itself does not limit its output, so a mix with
    a peak above 0 dBFS would be clipped (harsh distortion) in the file.

    Clipping is fixed in the same call (auto_fix_clipping, default true):
    when the peak is above 0 dBFS the Master mixer volume is lowered so the
    peak becomes -1 dBFS and the project is saved with it, so the next
    render is clean too. The response then has "auto_gain": {"applied_db",
    "master_volume_before", "master_volume_after", "reason"} and
    "levels_after": {"peak_dbfs", "clipped_samples": 0} (computed from the
    measured render and the gain; the saved project plays this loud).
    Automation of the Master volume is scaled with it. A "wav" is written
    from a lossless 32-bit float render with the gain applied; flac/ogg/mp3
    need one more LMMS render of the adjusted project, so they take about
    twice as long. With auto_fix_clipping=false (or when the Master volume
    cannot be adjusted) a clipping mix is rendered once more, as LMMS makes
    it (clipped), and the response gives "suggested_gain_db" and
    "suggested_master_volume" that fix it (apply with
    set_mixer_channel_volume(0, ...)). A render whose peak is below -12 dBFS
    is left alone but has a "note" with the gain that would bring it to
    -1 dBFS (a silent render has a note too). A "note" also says when the
    Master volume cannot be adjusted automatically.

    A "wav" is 16-bit PCM, as LMMS writes it. "command" is the last LMMS
    command that ran (for a wav, the float render into a temporary file).

    Args:
        output_path: Output file path (default: <project>.wav in the
            same directory). A relative path is relative to the default
            projects folder (LMMS_PROJECTS_DIR), the same as in
            save_project and load_project; a path starting with ~ is in
            the home folder
        file_format: "wav", "flac", "ogg" or "mp3"
        samplerate: Sample rate in Hz (44100, 48000, ...)
        bitrate: Bitrate in kbit/s for lossy formats (ogg/mp3)
        oversampling: 1, 2, 4 or 8 (default 1). The default 1x matches what
            LMMS plays in its GUI and the defaults of its Export dialog, so
            the render sounds like what the user hears in LMMS. (The LMMS
            command line alone would default to 2x.) Higher oversampling is
            NOT simply "higher quality": it changes the sound of some
            instruments, e.g. ZynAddSubFX presets get a different timbre
            because they depend on the internal sample rate. Only raise it
            if the result should differ from the GUI.
        interpolation: "linear", "sincfastest", "sincmedium" or "sincbest"
            (default "sincfastest", the LMMS default)
        auto_fix_clipping: Lower the Master volume (and save the project)
            when the render clips, see above (default true). Pass false to
            keep the mix as it is.
    """
    import subprocess
    proj = get_project()
    if not proj.path:
        return json.dumps({
            "error": "Project has never been saved. Call save_project first."
        })
    exe = lmms_app.find_lmms_exe()
    if exe is None:
        return json.dumps({
            "error": "LMMS executable not found. Set LMMS_EXECUTABLE env var "
                     "to the LMMS executable (lmms.exe / lmms).",
        })

    fmt = file_format.lower()
    if fmt not in ("wav", "flac", "ogg", "mp3"):
        return json.dumps({
            "error": f"Invalid format '{file_format}'. Use wav/flac/ogg/mp3."
        })
    if isinstance(oversampling, bool) or oversampling not in (1, 2, 4, 8):
        return json.dumps({
            "error": f"Invalid oversampling '{oversampling}'. Use 1, 2, 4 or 8."
        })
    interp = interpolation.lower() if isinstance(interpolation, str) else ""
    if interp not in ("linear", "sincfastest", "sincmedium", "sincbest"):
        return json.dumps({
            "error": f"Invalid interpolation '{interpolation}'. Use "
                     "linear/sincfastest/sincmedium/sincbest."
        })

    proj_path = Path(proj.path)
    if output_path is None:
        output_path = str(proj_path.with_suffix(f".{fmt}"))
    output_path = _expand_user(output_path)
    if not Path(output_path).is_absolute():
        output_path = str(_in_projects_folder(output_path))

    # Save current state before rendering. Keep the compression format
    # consistent with the file extension - compressed bytes inside a
    # .mmp file cannot be read by LMMS and other tools.
    compressed = str(proj_path).lower().endswith(".mmpz")
    try:
        proj.save(proj.path, compressed=compressed)
    except (OSError, ValueError) as exc:
        return json.dumps({
            "error": f"Could not save the project to {proj.path} before "
                     f"rendering: {_os_error_reason(exc, str(proj.path))}. "
                     "Call save_project with a path that can be written, "
                     "then render again.",
            "path": str(proj.path),
        })

    out_file = Path(output_path)
    created: list[str] = []
    try:
        if out_file.is_dir():
            # LMMS would exit without an error and write nothing.
            return json.dumps({
                "error": f"output_path {output_path} is an existing folder, "
                         f"not a file. Add a file name, e.g. "
                         f"{os.path.join(output_path, 'song.' + fmt)}.",
                "output_file": output_path,
            })
        created = _create_parent_folders(out_file)
        # The audio is written after the (long) render: find out now if it
        # cannot be.
        _check_writable(out_file)
    except (OSError, ValueError) as exc:
        _remove_folders(created)
        return json.dumps({
            "error": f"Could not render to {output_path}: "
                     f"{_os_error_reason(exc, output_path)}",
            "output_file": output_path,
        })

    render_args = (samplerate, bitrate, oversampling, interp)
    extras: dict = {}  # "levels", "auto_gain", "note", ... (see below)

    def finish(success: bool, proc, command: str, error: str | None = None):
        """The response of a render that ran LMMS last as `proc`."""
        result = {
            "command": command,
            "returncode": proc.returncode,
            "output_file": str(out_file),
            "oversampling": oversampling,
            "interpolation": interp,
            "isolated_config": True,
            "success": success,
        }
        if success:
            size = out_file.stat().st_size
            result["message"] = f"Rendered to {out_file} ({size} bytes)"
            result["size_bytes"] = size
            if created:
                result["created_dirs"] = created
        else:
            _remove_folders(created)
            result["error"] = error
        result.update(extras)
        return json.dumps(result)

    try:
        with tempfile.TemporaryDirectory(
                prefix="lmms-mcp-render-",
                ignore_cleanup_errors=True) as work:
            # LMMS writes 16-bit files with its output clipped at 0 dBFS, so
            # the level of the mix is only visible in a float render.
            float_file = Path(work) / "levels.wav"
            proc, command = _run_lmms_render(
                exe, proj_path, float_file, "wav", *render_args,
                float_output=True)
            if proc.returncode != 0 or not float_file.is_file():
                return finish(False, proc, command,
                              _render_failure_message(proc, float_file))
            try:
                audio = audio_analysis.read_wav(float_file)
            except (OSError, audio_analysis.AudioError) as exc:
                audio = None
                extras["levels_error"] = (
                    f"Could not measure the render ({exc}); the file was "
                    "rendered without level analysis.")
            if audio is None:
                proc, command = _run_lmms_render(
                    exe, proj_path, out_file, fmt, *render_args)
                return finish(
                    proc.returncode == 0 and out_file.is_file(), proc,
                    command, _render_failure_message(proc, out_file))

            levels, gain, error = _measure_and_fix(
                proj, audio, auto_fix_clipping, compressed, extras)
            if error:  # the lowered Master volume could not be saved
                return finish(False, proc, command, error)

            # A clean (or fixed) wav is written from the float render. Any
            # other file is made by LMMS: the formats LMMS encodes, and a
            # wav that stays as clipped as LMMS makes it (nothing fixed).
            if fmt == "wav" and not (levels.peak > 1.0 and gain == 1.0):
                try:
                    # LMMS makes its 16-bit samples by truncating
                    # x * 32767 (measured with 1.2.2: the same samples as
                    # this, but for float rounding at the boundary): do the
                    # same, so a clean mix is the file it always was.
                    audio_analysis.write_wav(
                        out_file, audio, bits=16, gain=gain, peak=levels.peak,
                        rounding="toward_zero")
                except (OSError, audio_analysis.AudioError) as exc:
                    _discard(out_file)
                    return finish(
                        False, proc, command,
                        f"Could not write {out_file}: "
                        f"{_os_error_reason(exc, str(out_file))}")
                return finish(True, proc, command)
            proc, command = _run_lmms_render(
                exe, proj_path, out_file, fmt, *render_args)
            return finish(
                proc.returncode == 0 and out_file.is_file(), proc, command,
                _render_failure_message(proc, out_file))
    except subprocess.TimeoutExpired:
        _remove_folders(created)
        return json.dumps({
            "error": "Rendering timed out after 300s. The song may be too "
                     "long.",
            **extras,
        })
    except OSError as exc:
        _remove_folders(created)
        return json.dumps({
            "error": f"Could not start the render: {exc}", **extras})


def _check_writable(path: Path) -> None:
    """Raise PermissionError if a file cannot be written at `path`, as far
    as the file system permissions say (os.access)."""
    target = path if path.exists() else path.parent
    if not os.access(target, os.W_OK):
        what = "file" if path.exists() else "folder"
        raise PermissionError(
            errno.EACCES, f"no write permission for the {what} {target}")


def _discard(path: Path) -> None:
    """Delete a file that was written only in part."""
    try:
        path.unlink()
    except OSError:
        pass


def _measure_and_fix(
    proj, audio, auto_fix_clipping: bool, compressed: bool, result: dict,
):
    """Measure the float render `audio`, fix a clipping Master volume if
    asked, and add "levels" and what follows from them to `result`.

    Returns (levels, gain, error): the gain (a linear factor) to apply to
    the render's samples to get what the project now plays. `error` is a
    text when the lowered Master volume could not be saved (the project is
    then as it was), else None.
    """
    info = proj.get_info()
    bar_length = audio_analysis.bar_seconds(
        info.bpm, info.time_sig_numerator, info.time_sig_denominator)
    levels = audio_analysis.analyze(audio, bar_length)
    result["levels"] = _levels_json(levels)
    head = xml_parser.find_head(proj.root)
    if head is not None and xml_parser.model_automation(
            proj.root, head, "bpm"):
        result["levels"]["bars_note"] = (
            "The tempo is automated: the bar ranges use the static tempo "
            "and are approximate.")

    if levels.peak <= 1.0 or not auto_fix_clipping:
        result.update(_level_advice(proj, levels))
        return levels, 1.0, None

    peak_db = audio_analysis.to_db(levels.peak)
    gain_db = _RENDER_TARGET_PEAK_DB - peak_db
    gain = audio_analysis.from_db(gain_db)
    snapshot, was_modified = copy.deepcopy(proj.root), proj.modified
    try:
        change = xml_parser.scale_master_volume(proj.root, gain)
    except xml_parser.MasterGainError as exc:
        result.update(_level_advice(proj, levels))
        result["note"] = f"Not fixed automatically: {exc}. {result['note']}"
        return levels, 1.0, None
    try:
        proj.save(proj.path, compressed=compressed)
    except (OSError, ValueError) as exc:
        proj.restore(snapshot, was_modified)
        return levels, 1.0, (
            f"The render clips (peak {peak_db:+.1f} dBFS) and lowering the "
            f"Master volume by {-gain_db:.1f} dB needs the project saved, "
            f"but it could not be saved to {proj.path}: "
            f"{_os_error_reason(exc, str(proj.path))}. The project was left "
            "unchanged. Save it to a place that can be written, or render "
            "with auto_fix_clipping=false.")
    curves = change["curves_scaled"]
    result["auto_gain"] = {
        "applied_db": round(gain_db, 2),
        "master_volume_before": round(change["before"], 4),
        "master_volume_after": round(change["after"], 4),
        "reason": (
            f"The render peaked at {peak_db:+.1f} dBFS "
            f"({levels.clipped} samples above 0 dBFS) and would clip. The "
            f"Master mixer volume was lowered from "
            f"{change['before']:.4g} to {change['after']:.4g} "
            f"({gain_db:+.1f} dB) so the peak is at "
            f"{_RENDER_TARGET_PEAK_DB:g} dBFS, and the project was saved "
            "with it."
            + (f" The automation of the Master volume ({curves} "
               f"pattern{'s' if curves != 1 else ''}) was scaled by the same "
               "factor." if curves else "")
            + (" The project had no Master mixer channel; one was added."
               if change["created"] else "")),
    }
    result["levels_after"] = {
        "peak_dbfs": _db_text(peak_db + gain_db),
        "clipped_samples": 0,
    }
    return levels, gain, None


# Lines LMMS prints on every run that say nothing about a failure.
_LMMS_HARMLESS_OUTPUT = (
    "VST sync support disabled in your configuration",
    "Notice: could not set realtime priority.",
)


def _render_failure_message(proc, out_file: Path) -> str:
    """Why a render did not produce its file, from what LMMS printed."""
    detail = ""
    for text in (proc.stderr, proc.stdout):
        lines = [line.strip() for line in (text or "").splitlines()]
        detail = "\n".join(
            line for line in lines
            if line and line not in _LMMS_HARMLESS_OUTPUT)
        if detail:
            break
    if "for writing" in detail:
        # "Could not open file <path> for writing. Please make sure you
        # have write permission to the file and the directory ..."
        return (
            f"Render failed: LMMS could not write the output file "
            f"{out_file} (no write permission for the file or its folder, "
            f"disk full, or the path is too long). LMMS said: {detail}")
    if detail:
        return f"Render failed: {detail}"
    return (
        f"Render failed: LMMS exited with code {proc.returncode} without "
        f"creating {out_file} and printed no error message.")


# Most files per instrument that list_available_plugins shows.
_MAX_LISTED_FILES = 20


@_tool
def list_available_plugins() -> str:
    """List the plugins usable with the installed LMMS.

    Shows the built-in instruments/effects this LMMS provides, any custom
    instrument plugins the user added to its plugins folder (usable by
    name in add_instrument_track), and which known plugins are missing.
    """
    av = _availability
    classified = lmms_app.classify_installed_plugins(
        set(KNOWN_INSTRUMENTS), set(effects_mod.KNOWN_EFFECTS), av.installed
    )
    result = {
        "lmms_found": lmms_app.find_lmms_exe() is not None,
        "version": av.version,
        "built_in_instruments": _ordered(KNOWN_INSTRUMENTS, av.instruments),
        "lmms_plugin_names": {
            n: av.plugin_name(n)
            for n in _ordered(KNOWN_INSTRUMENTS, av.instruments)
            if av.plugin_name(n) != n
        },
        "built_in_effects": _effect_choices(av),
        "custom_plugins_unknown_type": [
            n for n in classified["unknown"]
            if n not in lmms_app.NON_TRACK_PLUGINS
        ],
    }
    if av.verified:
        missing = {}
        for name in (*KNOWN_INSTRUMENTS, *effects_mod.KNOWN_EFFECTS):
            if name in av.instruments or name in av.effects \
                    or name in av.needs_file:
                continue
            required = lmms_app.PLUGIN_REQUIRES.get(name)
            reason = f"requires {required}" if required else "not installed"
            alternative = lmms_app.PLUGIN_SUBSTITUTES.get(name)
            if alternative in av.instruments or alternative in av.effects:
                reason += f"; use {alternative}"
            missing[name] = reason
        for name in sorted(av.needs_file):
            missing[name] = _needs_file_reason(name, av) + "."
        result["unavailable_in_installed_lmms"] = missing
    files = {}
    for name in _file_instruments(av):
        found = av.files.for_instrument(name)
        entry: dict = {"count": len(found), "files": list(found[:_MAX_LISTED_FILES])}
        if len(found) > _MAX_LISTED_FILES:
            entry["note"] = f"first {_MAX_LISTED_FILES} of {len(found)}"
        if name == "sf2player" and av.files.default_soundfont:
            entry["default"] = av.files.default_soundfont
        files[name] = entry
    if files:
        result["instrument_files"] = files
    result["note"] = (
        "Custom plugins can be tried as instruments by their library name "
        "(without extension, and without the 'lib' prefix on Linux) in "
        "add_instrument_track. add_effect only accepts the built-in effects "
        "listed above. lmms_plugin_names lists instruments that LMMS loads "
        "under another name than the one offered; add_instrument_track "
        "handles that, keep using the offered name."
    )
    return json.dumps(result, indent=2)


@_tool
def scan_vst_directory(directory: str, recursive: bool = True) -> str:
    """Scan a folder for VST plugin DLLs (.dll files).

    Args:
        directory: Path to scan (e.g. "C:/VSTPlugins"; a path starting with
            ~ is in the home folder)
        recursive: Include subdirectories
    """
    directory = _expand_user(directory)
    try:
        plugins = lmms_app.find_vst_plugins(directory, recursive)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({
        "directory": str(directory),
        "count": len(plugins),
        "plugins": plugins,
        "hint": "Use add_vst_track with a plugin path. Note: whether a "
                "DLL is an instrument or effect is decided by LMMS on "
                "load.",
    }, indent=2)


@_tool
def add_vst_track(
    name: str,
    dll_path: str,
    mixer_channel: int = 0,
    volume: int = 100,
    panning: int = 0,
) -> str:
    """Add a track hosting a VST plugin (.dll file).

    Uses LMMS's Vestige host. The VST must be compatible with your
    LMMS architecture (64-bit LMMS needs 64-bit VSTs).

    Args:
        name: Track name (e.g. "Spire Lead")
        dll_path: Absolute path to the VST .dll file (a path starting with
            ~ is in the home folder)
        mixer_channel: Mixer channel number (0=Master). The channel must
            exist (add_mixer_channel creates one).
        volume: Track volume (0-200)
        panning: Track panning (-100 to +100)
    """
    proj = get_project()
    av = _availability
    if "vestige" not in av.instruments:
        return json.dumps({
            "error": lmms_app.unavailable_reason("vestige", av.version)
                     + " VST plugins cannot be hosted. Use "
                       "add_instrument_track with an available instrument.",
            "available_instruments": _instrument_choices(av),
        })
    dll_path = _expand_user(dll_path)
    path = Path(dll_path)
    if not path.is_file():
        return json.dumps({
            "error": f"VST file not found: {dll_path}",
            "hint": "Use scan_vst_directory to find VST plugins.",
        })
    if path.suffix.lower() != ".dll":
        return json.dumps({
            "error": f"'{path.name}' is not a .dll file."
        })

    try:
        result = proj.add_track(
            "instrument", name,
            instrument="vestige",
            mixer_channel=mixer_channel,
            volume=volume,
            panning=panning,
        )
    except ValueError as exc:  # a mixer channel the project lacks
        return json.dumps({"error": str(exc)})

    # Set the plugin path on the vestige element
    try:
        track_idx = result.get("track_index")
        track = xml_parser.find_track_element(proj.root, track_idx)
        vestige_el = track.find("instrumenttrack/instrument/vestige")
        if vestige_el is not None:
            # Prefer path relative to LMMS dir when possible
            exe = lmms_app.find_lmms_exe()
            try:
                rel = path.relative_to(exe.parent) if exe else None
            except ValueError:
                rel = None
            vestige_el.set("plugin", str(rel if rel else path))
    except (ValueError, IndexError):
        pass

    result["vst"] = str(path)
    result["message"] = f"Added VST track '{name}' hosting {path.name}"
    return json.dumps(result)


# ──────────────────────────────────────────────────────────────────
# ARRANGEMENT TOOLS (SONG EDITOR TIMELINE)
# ──────────────────────────────────────────────────────────────────


@_tool
def assign_sample_file(track_index: int, file_path: str) -> str:
    """Assign an audio file to all clips on a sample track.

    Use relative paths for LMMS's built-in samples (e.g.
    "drums/kick01.ogg") or absolute paths for your own files (a path
    starting with ~ is in the home folder).

    Args:
        track_index: Index of the sample track
        file_path: Audio file path (WAV/OGG/MP3/FLAC)
    """
    proj = get_project()
    file_path = _expand_user(file_path)
    try:
        track = xml_parser.find_track_element(proj.root, track_index)
        if xml_parser.get_track_type(track) != 2:
            return json.dumps({
                "error": f"Track {track_index} is not a sample track"
            })
        clips = track.findall("sampleclip")
        if not clips:
            clip = xml_parser.add_sample_clip(
                proj.root, track_index, file_path, position=0
            )
            count = 1
        else:
            for clip in clips:
                clip.set("src", file_path)
            count = len(clips)
        return json.dumps({
            "track_index": track_index,
            "file": file_path,
            "clips_updated": count,
            "message": f"Assigned '{file_path}' to {count} clip(s) "
                       f"on track {track_index}",
        })
    except (ValueError, IndexError) as exc:
        return json.dumps({"error": str(exc)})


@_tool
def place_sample_clip(
    track_index: int,
    file_path: str,
    pos_bars: float = 0,
    length_bars: float = 1,
) -> str:
    """Place an audio file clip on a sample track at a given bar.

    Args:
        track_index: Index of the sample track
        file_path: Audio file path (relative to LMMS samples, or absolute;
            a path starting with ~ is in the home folder)
        pos_bars: Start position in bars (0 = beginning)
        length_bars: Clip length in bars
    """
    proj = get_project()
    file_path = _expand_user(file_path)
    pos = bars_to_ticks(pos_bars)
    length = bars_to_ticks(length_bars)
    try:
        clip = xml_parser.add_sample_clip(proj.root, track_index, file_path, pos, length)
        return json.dumps({
            "track_index": track_index,
            "file": file_path,
            "pos_ticks": pos,
            "len_ticks": length,
            "message": f"Placed '{file_path}' at bar {pos_bars} "
                       f"({pos} ticks)",
        })
    except (ValueError, IndexError) as exc:
        return json.dumps({"error": str(exc)})


@_tool
def place_pattern(
    track_index: int,
    pos_bars: float = 0,
    name: str | None = None,
    length_bars: float = 1,
) -> str:
    """Place an empty pattern clip on an instrument track (song arrangement).

    The pattern starts empty; add notes with add_note (they go into the
    first pattern) or use this to sketch the arrangement structure first.

    Args:
        track_index: Index of the instrument track
        pos_bars: Start position in bars (0 = beginning)
        name: Optional pattern name (defaults to track name)
        length_bars: Pattern length in bars
    """
    proj = get_project()
    pos = bars_to_ticks(pos_bars)
    length = bars_to_ticks(length_bars)
    try:
        pattern = xml_parser.place_instrument_pattern(
            proj.root, track_index, pos, name, length
        )
        return json.dumps({
            "track_index": track_index,
            "pattern_name": pattern.get("name"),
            "pos_ticks": pos,
            "len_ticks": length,
            "message": f"Placed pattern '{pattern.get('name')}' at bar "
                       f"{pos_bars} on track {track_index}",
        })
    except (ValueError, IndexError) as exc:
        return json.dumps({"error": str(exc)})


@_tool
def place_bb_clip(
    bb_track_index: int,
    pos_bars: float = 0,
    length_bars: float = 4,
) -> str:
    """Place a beat/bassline clip on a pattern track in the song editor.

    This triggers the BB pattern (created via add_pattern_track and
    filled with notes) to play at the given time.

    Args:
        bb_track_index: Index of the pattern (BB) track
        pos_bars: Start position in bars
        length_bars: Clip length in bars (default 4 = one BB cycle of 16 steps)
    """
    proj = get_project()
    pos = bars_to_ticks(pos_bars)
    length = bars_to_ticks(length_bars)
    try:
        clip = xml_parser.add_bb_clip(proj.root, bb_track_index, pos, length)
        return json.dumps({
            "track_index": bb_track_index,
            "pos_ticks": pos,
            "len_ticks": length,
            "message": f"Placed BB clip at bar {pos_bars} "
                       f"(length {length_bars} bars)",
        })
    except (ValueError, IndexError) as exc:
        return json.dumps({"error": str(exc)})


@_tool
def move_clip(track_index: int, old_pos_bars: float, new_pos_bars: float) -> str:
    """Move a clip (pattern/bbtco/sampleclip) to a new time position.

    Args:
        track_index: Track containing the clip
        old_pos_bars: Current start position in bars
        new_pos_bars: New start position in bars
    """
    proj = get_project()
    try:
        result = xml_parser.move_clip(
            proj.root,
            track_index,
            bars_to_ticks(old_pos_bars),
            bars_to_ticks(new_pos_bars),
        )
        return json.dumps(result)
    except (ValueError, IndexError) as exc:
        return json.dumps({"error": str(exc)})


@_tool
def delete_clip(track_index: int, pos_bars: float) -> str:
    """Delete a clip at a given position from a track.

    Args:
        track_index: Track containing the clip
        pos_bars: Clip start position in bars
    """
    proj = get_project()
    try:
        result = xml_parser.delete_clip(proj.root, track_index, bars_to_ticks(pos_bars))
        return json.dumps(result)
    except (ValueError, IndexError) as exc:
        return json.dumps({"error": str(exc)})


@_tool
def get_arrangement() -> str:
    """Get the full song editor arrangement: all clips sorted by time.

    Shows every pattern, BB clip, sample clip and automation curve with
    their positions and lengths in ticks and bars.
    """
    proj = get_project()
    clips = xml_parser.get_arrangement(proj.root)
    for c in clips:
        c["pos_bars"] = round(ticks_to_bars(c["pos"]), 3)
        c["len_bars"] = round(ticks_to_bars(c["len"]), 3)
    total = max((c["pos"] + c["len"] for c in clips), default=0)
    return json.dumps({
        "clip_count": len(clips),
        "song_length_bars": round(ticks_to_bars(total), 2),
        "clips": clips,
    }, indent=2)


# ──────────────────────────────────────────────────────────────────
# AUTOMATION TOOLS
# ──────────────────────────────────────────────────────────────────


@_tool
def add_automation(
    target_type: str,
    param: str,
    points: list[dict],
    target_index: int = 0,
    name: str | None = None,
    smooth: bool = True,
) -> str:
    """Create an automation curve that controls a parameter over time.

    Automatable targets:
    - song + tempo: Song BPM (e.g. tempo ramps)
    - song + master_volume / master_pitch
    - track + volume / panning (target_index = track index)
    - mixer + volume (target_index = mixer channel number)

    Point positions are absolute song positions in bars. The curve is
    active from the first to the last point only: before the first point
    the parameter keeps its normal (track/mixer/song) value, and after the
    last point it stays at the last value (LMMS keeps the last automated
    value). The normal value stays editable afterwards: set_track_volume,
    set_track_panning, set_mixer_channel_volume, set_tempo and the other
    setters change it (the response says the parameter is automated).

    Args:
        target_type: "song", "track" or "mixer"
        param: Parameter name (see above)
        points: Curve points as list of {"bar": float, "value": float},
            with "bar" the absolute song position (bar 0 = song start).
            Example: [{"bar": 0, "value": 120}, {"bar": 8, "value": 140}]
        target_index: Track index or mixer channel (ignored for "song")
        name: Automation name (defaults to parameter name)
        smooth: True = smooth curves (cubic), False = linear steps
    """
    proj = get_project()
    if not points:
        return json.dumps({"error": "points list is empty"})

    # Convert bar-based points to tick-based
    tick_points: list[tuple[int, float]] = []
    for i, pt in enumerate(points):
        try:
            bar = float(pt["bar"])
            value = float(pt["value"])
        except (KeyError, TypeError, ValueError):
            return json.dumps({
                "error": f"Point {i} must be {{'bar': float, 'value': float}}"
            })
        tick_points.append((bars_to_ticks(bar), value))

    try:
        model_id, current_value = xml_parser.resolve_automation_target(
            proj.root, target_type, target_index, param
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    auto_name = name or param.replace("_", " ").title()
    try:
        track = xml_parser.add_automation_curve_track(
            proj.root,
            auto_name,
            tick_points,
            target_id=model_id,
            progression=1 if smooth else 0,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    tracks = xml_parser.find_tracks(proj.root)
    return json.dumps({
        "automating": f"{target_type}[{target_index}].{param}",
        "current_value": current_value,
        "model_id": model_id,
        "automation_track_index": len(tracks) - 1,
        "points": len(tick_points),
        "smooth": smooth,
        "message": f"Created automation '{auto_name}' controlling "
                   f"{target_type}[{target_index}].{param} with "
                   f"{len(tick_points)} points",
    })


# ──────────────────────────────────────────────────────────────────
# TRANSPORT / SONG SETTINGS TOOLS
# ──────────────────────────────────────────────────────────────────


@_tool
def set_tempo(bpm: int) -> str:
    """Set the song tempo (BPM).

    If the tempo is automated (add_automation), the value is its static
    one: it applies until the automation curve starts, and the curve
    overrides it wherever it has points ("automated" and "note" in the
    response say so).

    Args:
        bpm: Tempo in beats per minute (10-999)
    """
    proj = get_project()
    proj.set_tempo(bpm)
    return json.dumps({
        "message": f"Tempo set to {bpm} BPM",
        **_head_automation_note(proj, "bpm", "The tempo", f"{bpm} BPM"),
    })


@_tool
def set_time_signature(numerator: int, denominator: int) -> str:
    """Set the time signature.

    Args:
        numerator: Beats per bar (e.g. 4, 3, 6)
        denominator: Beat unit (e.g. 4 for quarter notes, 8 for eighth notes)
    """
    proj = get_project()
    proj.set_time_signature(numerator, denominator)
    return json.dumps({"message": f"Time signature set to {numerator}/{denominator}"})


@_tool
def set_master_volume(volume: int) -> str:
    """Set the master volume.

    If the master volume is automated (add_automation), the value is its static
    one: it applies until the automation curve starts, and the curve
    overrides it wherever it has points ("automated" and "note" in the
    response say so).

    Args:
        volume: Master volume (0-200, 100=normal)
    """
    proj = get_project()
    proj.set_master_volume(volume)
    return json.dumps({
        "message": f"Master volume set to {volume}",
        **_head_automation_note(
            proj, "mastervol", "The master volume", str(volume)),
    })


@_tool
def set_master_pitch(pitch: int) -> str:
    """Set the master pitch offset.

    If the master pitch is automated (add_automation), the value is its static
    one: it applies until the automation curve starts, and the curve
    overrides it wherever it has points ("automated" and "note" in the
    response say so).

    Args:
        pitch: Pitch offset in semitones (-12 to +12)
    """
    proj = get_project()
    proj.set_master_pitch(pitch)
    return json.dumps({
        "message": f"Master pitch set to {pitch} semitones",
        **_head_automation_note(
            proj, "masterpitch", "The master pitch", f"{pitch} semitones"),
    })


# ──────────────────────────────────────────────────────────────────
# UTILITY / CONVERSION TOOLS
# ──────────────────────────────────────────────────────────────────


@_tool
def note_name_to_key(name: str) -> str:
    """Convert a note name (e.g. 'C4') to its LMMS key number.

    Keys are LMMS keys: middle C (C4) = 48, A4 = 57 = 440 Hz. A MIDI note
    number is key + 12 (also returned as "midi_note").

    Sharps and flats both work and enharmonics agree ('Bb2' = 'A#2' = 34,
    'Cb4' = 'B3' = 47). Range C0 (0) to C9 (108). Invalid names return
    {"error": "..."} stating the accepted format.

    Args:
        name: Note name like 'C4', 'A#3', 'F#5', 'Bb2'
    """
    try:
        key = _note_name_to_key(name)
        return json.dumps({
            "note": name, "key": key, "midi_note": key + MIDI_NOTE_OFFSET,
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})


@_tool
def key_to_note_name(key: int) -> str:
    """Convert an LMMS key number to its note name.

    Keys are LMMS keys: middle C (C4) = 48, A4 = 57 = 440 Hz. A MIDI note
    number is key + 12 (also returned as "midi_note").

    Args:
        key: LMMS key number (0-108, C0 to C9)
    """
    if isinstance(key, bool) or not isinstance(key, int) or not (
        KEY_MIN <= key <= KEY_MAX
    ):
        return json.dumps({
            "error": f"key must be an integer {KEY_RANGE_TEXT}. "
                     f"{KEY_NUMBERING_TEXT}"
        })
    return json.dumps({
        "key": key, "note": _key_to_note_name(key),
        "midi_note": key + MIDI_NOTE_OFFSET,
    })


@_tool
def bars_to_ticks_converter(bars: int) -> str:
    """Convert bars to ticks (192 ticks per bar in 4/4 time).

    Args:
        bars: Number of bars
    """
    ticks = bars_to_ticks(bars)
    return json.dumps({"bars": bars, "ticks": ticks, "ticks_per_bar": TICKS_PER_BAR})


@_tool
def ticks_to_bars_converter(ticks: int) -> str:
    """Convert ticks to bars (192 ticks per bar in 4/4 time).

    Args:
        ticks: Number of ticks
    """
    bars = ticks_to_bars(ticks)
    return json.dumps({"ticks": ticks, "bars": round(bars, 3), "ticks_per_bar": TICKS_PER_BAR})


@_tool
def generate_scale(
    root_note: str = "C4",
    scale_type: str = "major",
    num_octaves: int = 1,
) -> str:
    """Generate a musical scale as LMMS key numbers and note names.

    Keys are LMMS keys: middle C (C4) = 48, A4 = 57 = 440 Hz. A MIDI note
    number is key + 12. Notes above C9 (key 108) are left out.

    Args:
        root_note: Root note name (e.g. 'C4', 'A3', 'Bb2'; sharps and flats both work)
        scale_type: Scale type: major, minor, dorian, mixolydian, pentatonic_major,
            pentatonic_minor, blues, chromatic, harmonic_minor, melodic_minor
        num_octaves: Number of octaves to generate
    """
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "pentatonic_major": [0, 2, 4, 7, 9],
        "pentatonic_minor": [0, 3, 5, 7, 10],
        "blues": [0, 3, 5, 6, 7, 10],
        "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    }

    if scale_type not in scales:
        return json.dumps({"error": f"Unknown scale type: {scale_type}. Use: {list(scales.keys())}"})

    try:
        root_key = _note_name_to_key(root_note)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    intervals = scales[scale_type]
    notes = []
    for octave in range(num_octaves):
        for interval in intervals:
            key = root_key + octave * 12 + interval
            if key <= KEY_MAX:
                notes.append({
                    "key": key,
                    "note": _key_to_note_name(key),
                    "position": len(notes),
                })

    return json.dumps({
        "root": root_note,
        "scale": scale_type,
        "notes": notes,
        "count": len(notes),
    })


# ──────────────────────────────────────────────────────────────────
# RESOURCES
# ──────────────────────────────────────────────────────────────────


@mcp.resource("lmms://project/info")
def resource_project_info() -> str:
    """Get current LMMS project information."""
    proj = get_project()
    info = proj.get_info()
    return json.dumps(info.to_dict(), indent=2)


@mcp.resource("lmms://project/tracks")
def resource_project_tracks() -> str:
    """Get all tracks in the current project."""
    proj = get_project()
    info = proj.get_info()
    tracks = [t.to_dict() for t in info.tracks]
    return json.dumps({"tracks": tracks}, indent=2)


@mcp.resource("lmms://project/mixer")
def resource_project_mixer() -> str:
    """Get all mixer channels in the current project."""
    proj = get_project()
    info = proj.get_info()
    channels = [ch.to_dict() for ch in info.mixer_channels]
    return json.dumps({"mixer_channels": channels}, indent=2)


@mcp.resource("lmms://project/xml")
def resource_project_xml() -> str:
    """Get the raw XML of the current project."""
    proj = get_project()
    return proj.get_xml_string()


def _reference_note(av, kind: str) -> str:
    if av.verified:
        return (
            f"These are the {kind} available in {_lmms_label(av)}. LMMS "
            f"cannot download additional plugins - do not use any other "
            f"names."
        )
    return (
        f"No LMMS installation was detected, so availability could not be "
        f"verified: these are LMMS's standard {kind}. Plugins that need "
        f"LMMS 1.3+ are silently ignored by older versions."
    )


@mcp.resource("lmms://reference/instruments")
def resource_instruments() -> str:
    """Instruments available in the installed LMMS (verified)."""
    av = _availability
    return json.dumps({
        "note": _reference_note(av, "instruments"),
        "instruments": [
            {"name": name, "description": KNOWN_INSTRUMENTS[name],
             **({"lmms_plugin": av.plugin_name(name)}
                if av.plugin_name(name) != name else {})}
            for name in sorted(av.instruments)
        ],
        "recommendations_by_use_case": lmms_app.adapt_recommendations(
            INSTRUMENT_RECOMMENDATIONS, av.instruments
        ),
    }, indent=2)


def _effect_reference(name: str) -> dict:
    """One effect for lmms://reference/effects: description, controls node
    and, where known, wet default, starting settings and all parameters."""
    entry = {
        "name": name,
        "description": effects_mod.KNOWN_EFFECTS[name][0],
        "controls_node": effects_mod.KNOWN_EFFECTS[name][1],
    }
    if not effect_params.has_params(name):
        entry["parameters"] = "none can be set yet (plugin defaults)"
        return entry
    start = effect_params.EFFECT_START[name]
    entry["outputs_dry_signal"] = not start.why_wet
    entry["wet_default"] = start.wet
    entry["decay_ms_default"] = start.decay_ms
    entry["starting_settings"] = effect_params.describe_start(name)
    if start.note:
        entry["note"] = start.note
    entry["parameters"] = [
        effect_params.param_dict(p) for p in effect_params.params_of(name)]
    if name == "waveshaper":
        entry["shapes"] = effect_params.WAVESHAPE_DOC
    return entry


@mcp.resource("lmms://reference/effects")
def resource_effects() -> str:
    """Effects available in the installed LMMS (verified), with the starting
    settings of add_effect and every parameter set_effect_params accepts."""
    av = _availability
    return json.dumps({
        "note": _reference_note(av, "effects")
                + " Effects that host external plugins (LADSPA, LV2, VST) "
                  "cannot be added with add_effect.",
        "settings_of_every_effect": [
            effect_params.param_dict(p) for p in effect_params.EFFECT_LEVEL],
        "effects": [_effect_reference(name) for name in sorted(av.effects)],
        "recommendations_by_use_case": lmms_app.adapt_recommendations(
            effects_mod.EFFECT_RECOMMENDATIONS, av.effects
        ),
    }, indent=2)


@mcp.resource("lmms://reference/note_names")
def resource_note_names() -> str:
    """LMMS key number to note name mapping reference."""
    mapping = []
    for key in range(KEY_MIN, KEY_MAX + 1):
        mapping.append({
            "key": key,
            "note": _key_to_note_name(key),
            "midi_note": key + MIDI_NOTE_OFFSET,
        })
    return json.dumps({
        "numbering": KEY_NUMBERING_TEXT,
        "mapping": mapping,
        "total": len(mapping),
    }, indent=2)


@mcp.resource("lmms://reference/scales")
def resource_scales() -> str:
    """Available musical scales and their intervals."""
    return json.dumps({
        "scales": {
            "major": {"intervals": [0, 2, 4, 5, 7, 9, 11], "name": "Major (Ionian)"},
            "minor": {"intervals": [0, 2, 3, 5, 7, 8, 10], "name": "Natural Minor (Aeolian)"},
            "dorian": {"intervals": [0, 2, 3, 5, 7, 9, 10], "name": "Dorian Mode"},
            "mixolydian": {"intervals": [0, 2, 4, 5, 7, 9, 10], "name": "Mixolydian Mode"},
            "pentatonic_major": {"intervals": [0, 2, 4, 7, 9], "name": "Major Pentatonic"},
            "pentatonic_minor": {"intervals": [0, 3, 5, 7, 10], "name": "Minor Pentatonic"},
            "blues": {"intervals": [0, 3, 5, 6, 7, 10], "name": "Blues Scale"},
            "chromatic": {"intervals": list(range(12)), "name": "Chromatic"},
            "harmonic_minor": {"intervals": [0, 2, 3, 5, 7, 8, 11], "name": "Harmonic Minor"},
            "melodic_minor": {"intervals": [0, 2, 3, 5, 7, 9, 11], "name": "Melodic Minor"},
        },
    }, indent=2)


# ──────────────────────────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────────────────────────


def _available_of(*names: str) -> list[str]:
    """The given instrument names that the installed LMMS provides."""
    return [n for n in names if n in _availability.instruments] or [
        "tripleoscillator"
    ]


@mcp.prompt()
def create_basic_song(
    genre: str = "electronic",
    bpm: int = 120,
    key: str = "C4",
) -> str:
    """Create a basic song structure with drums, bass, and melody.

    Args:
        genre: Musical genre (electronic, rock, hip-hop, jazz, pop, ambient)
        bpm: Tempo in beats per minute
        key: Root note for the melody as a note name (e.g. 'C4' = middle C =
            key 48, 'A3' = 220 Hz)
    """
    pool = _availability.instruments
    drum_options = []
    if "kicker" in pool:
        drum_options.append('an instrument track "Drums" with kicker')
    if "audiofileprocessor" in pool:
        drum_options.append(
            "one audiofileprocessor track per drum sound (Kick, Snare, "
            "Hat) with a sample set via set_audiofile_sample (e.g. "
            '"drums/kick_hiphop01.ogg")'
        )
    if not drum_options:
        drum_options.append(
            'an instrument track "Drums" with tripleoscillator '
            "(short, percussive notes)"
        )
    drums = ", or ".join(drum_options)
    bass = " or ".join(_available_of("tripleoscillator", "lb302"))
    melody = " or ".join(_available_of("tripleoscillator", "watsyn"))
    return f"""Create a {genre} song at {bpm} BPM in the key of {key}.

Steps:
1. Create a new project with {bpm} BPM
2. Add drum tracks: {drums}
3. Add an instrument track "Bass" with {bass}
4. Add an instrument track "Melody" with {melody}
5. Add an automation track for volume/parameter automation
6. Add notes to each track following {genre} conventions
7. Configure mixer channels
8. Save the project

Use the LMMS MCP tools to build this song step by step."""


@mcp.prompt()
def add_drum_pattern(
    pattern_style: str = "four_on_the_floor",
    steps: int = 16,
) -> str:
    """Create a drum pattern for the current project.

    Args:
        pattern_style: Drum pattern style (four_on_the_floor, breakbeat,
            hiphop, trap, rock_basic, waltz)
        steps: Number of steps in the pattern (16 or 32)
    """
    patterns = {
        "four_on_the_floor": "Classic 4/4 dance beat: kick on 1,3,5,7,9,11,13,15; snare on 5,13; hi-hat on every even step",
        "breakbeat": "Syncopated drum pattern with kick on 1,6,11; snare on 5,13; hi-hats on off-beats",
        "hiphop": "Boom-bap style: kick on 1,7; snare on 5,13; hi-hats on every 2nd step with swing",
        "trap": "Heavy 808s: kick on 1,9; snare on 5,13; rapid hi-hat rolls on every step",
        "rock_basic": "Standard rock beat: kick on 1,9; snare on 5,13; hi-hats on all steps",
        "waltz": "3/4 time: kick on 1; snare on 3,5; hi-hats on every step",
    }

    description = patterns.get(pattern_style, f"Custom {pattern_style} pattern")

    if "audiofileprocessor" in _availability.instruments:
        find_track = """Find or create an instrument track for drums. For real drum samples
   create one audiofileprocessor track per drum sound (Kick, Snare, Hat),
   assign a sample to each with set_audiofile_sample (e.g.
   "drums/kick_hiphop01.ogg", "drums/snare01.ogg",
   "drums/hihat_closed02.ogg") and put that drum's notes at key 57 (A4,
   the track's base note = original sample pitch) on its own track."""
        key_note = (
            "\n   (not needed for per-drum audiofileprocessor tracks, "
            "which use key 57):"
        )
    else:
        find_track = (
            "Find or create an instrument track for drums (for example "
            + " or ".join(_available_of("kicker", "tripleoscillator"))
            + ")."
        )
        key_note = ":"
    gm_kit = (
        "the sf2player drum kit, bank 128"
        if "sf2player" in _availability.instruments
        else "any General MIDI drum instrument"
    )
    return f"""Create a drum pattern with style: {pattern_style}.

Description: {description}

Steps:
1. {find_track}
2. Add notes at the correct positions:
   - Use position values: step * 12 ticks (12 ticks per 1/16 step)
   - Each step is 12 ticks apart
   - Total pattern length: {steps} steps = {steps * 12} ticks
3. Keys for a General MIDI drum kit on a single multi-drum track
   ({gm_kit}){key_note}
   Keys are LMMS keys: a MIDI note number is key + 12, so the GM drum
   notes sit 12 lower than in MIDI charts:
   - Key 24 = Kick Drum (MIDI 36)
   - Key 26 = Snare (MIDI 38)
   - Key 30 = Closed Hi-Hat (MIDI 42)
   - Key 34 = Open Hi-Hat (MIDI 46)
   - Key 37 = Crash Cymbal (MIDI 49)
   - Key 39 = Ride Cymbal (MIDI 51)
4. Use the add_notes_batch tool for efficient note creation"""


@mcp.prompt()
def create_melody(
    scale_type: str = "major",
    root_note: str = "C4",
    bars: int = 4,
    style: str = "simple",
) -> str:
    """Generate a melody for the current project.

    Args:
        scale_type: Musical scale (major, minor, pentatonic_major, blues, etc.)
        root_note: Root note of the scale
        bars: Number of bars for the melody
        style: Melodic style (simple, complex, arpeggiated, chordal)
    """
    melody = ", ".join(_available_of("tripleoscillator", "watsyn"))
    return f"""Create a melody in {scale_type} scale starting on {root_note}.

Parameters:
- Scale: {scale_type} starting at {root_note}
- Duration: {bars} bars ({bars * 192} ticks)
- Style: {style}

Steps:
1. Use the generate_scale tool to get the scale notes
2. Create or find a melody instrument track ({melody}, etc.)
3. Add notes following the {style} style:
   - Simple: Use mostly scale tones on strong beats
   - Complex: Use chromatic passing tones and syncopation
   - Arpeggiated: Break chords into ascending/descending patterns
   - Chordal: Stack notes to form chords
4. Position notes within the pattern:
   - 1 bar = 192 ticks
   - 1/4 note = 48 ticks
   - 1/8 note = 24 ticks
   - 1/16 note = 12 ticks"""


@mcp.prompt()
def mix_and_arrange() -> str:
    """Mix and arrange the current project.

    Guides through volume balancing, panning, and arrangement decisions.
    """
    return """Review and improve the current project's mix and arrangement.

Steps:
1. Get project info to see all tracks and mixer channels
2. For each track, consider:
   - Set appropriate volume levels (drums ~80-100, bass ~70-90, melody ~60-80)
   - Add panning for stereo width (bass center, hi-hats slightly left/right)
3. Create mixer channels if needed for:
   - Drum bus
   - Bass bus
   - Lead/Melody bus
   - FX returns (reverb, delay)
   A track can only be routed to a channel that exists, so create the
   channel first (add_mixer_channel), then add the track with that
   mixer_channel (or move an existing track with set_track_mixer_channel).
4. Adjust mixer channel volumes
5. Save the project"""


@mcp.prompt()
def export_project(
    format: str = "mmpz",
    path: str = "",
) -> str:
    """Export/save the current project.

    Args:
        format: Export format (mmpz=compressed, mmp=uncompressed XML, wav=render audio)
        path: Output file path
    """
    if format == "wav":
        return """To export as WAV audio, the project needs to be rendered using LMMS.

Steps:
1. Save the project first using save_project
2. Render using: lmms --render <project.mmpz> -o <output.wav>
3. Or use the --export option in LMMS GUI

Note: The MCP server can create and modify project files,
but audio rendering requires the LMMS application."""
    else:
        return f"""Save the project in {format} format.

Steps:
1. Use save_project with path='{path}' and compressed={format == 'mmpz'}
2. The project will be saved as {'compressed .mmpz' if format == 'mmpz' else 'uncompressed .mmp'}"""


# ──────────────────────────────────────────────────────────────────
# HIDE TOOLS THE INSTALLED LMMS CANNOT SUPPORT
# ──────────────────────────────────────────────────────────────────


def _remove_unsupported_tools(av) -> list[str]:
    """Unregister tools whose required plugin the installed LMMS lacks."""
    removed = []
    for tool_name, plugins in TOOL_REQUIRES.items():
        needed = (plugins,) if isinstance(plugins, str) else plugins
        if not any(plugin in av.instruments for plugin in needed):
            mcp.remove_tool(tool_name)
            removed.append(tool_name)
    return removed


_remove_unsupported_tools(_availability)


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────


def main():
    """Run the LMMS MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
