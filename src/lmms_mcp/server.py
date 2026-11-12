"""LMMS MCP Server - AI-powered music production with LMMS.

This server provides tools, resources, and prompts for creating and
manipulating LMMS projects programmatically via the Model Context Protocol.
"""

import json
import os
import string
import textwrap
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.etree import ElementTree as ET

from mcp.server import MCPServer

from . import effects as effects_mod
from . import lmms_app
from . import presets as zyn_presets
from . import xml_parser
from .models import (
    MIDI_KEY_MAX,
    MIDI_KEY_MIN,
    NOTE_NAMES,
    TICKS_PER_BAR,
    TICKS_PER_STEP,
    STEPS_PER_BAR,
    TrackType,
    Note,
    midi_to_note_name,
    note_name_to_midi,
    bars_to_ticks,
    ticks_to_bars,
)
from .project import LMMSProject, get_project, set_project

DEFAULT_PROJECTS_DIR = os.environ.get(
    "LMMS_PROJECTS_DIR",
    str(Path.home() / "Desktop" / "Media" / "lmms" / "AI-Projects"),
)

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
    "patman": "GUS patch sampler",
    "sf2player": "SoundFont (.sf2) sample player",
    "sfxr": "Retro sound effect generator (chiptune)",
    "sid": "Commodore 64 SID chip emulator (chiptune)",
    "slicert": "Beat slicer for chopping audio loops",
    "vibedstrings": "Vibrating string physical model",
    "watsyn": "4-oscillator wavetable-style synth",
    "xpressive": "Expressive mono lead synth",
    "zynaddsubfx": "ZynAddSubFX powerful feature-rich synth",
    "gigplayer": "GIG sample library player",
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
TOOL_REQUIRES = {
    "list_zyn_presets": "zynaddsubfx",
    "load_zyn_preset": "zynaddsubfx",
    "set_zyn_params": "zynaddsubfx",
    "set_audiofile_sample": "audiofileprocessor",
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


def render_plugin_doc(template: str, av) -> str:
    """Fill $INSTRUMENTS / $EFFECTS in a tool docstring template."""
    return string.Template(template).safe_substitute(
        INSTRUMENTS=_wrap_names(_instrument_choices(av)),
        EFFECTS=_wrap_names(_effect_choices(av)),
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
        """LMMS MCP Server for AI-powered music production.

This server lets you create, load, modify, and save LMMS music projects.
LMMS is a free, open-source digital audio workstation (DAW).

Key concepts:
- LMMS projects contain tracks (Instrument, Sample, Pattern, Automation)
- Instrument tracks hold synthesizer/sampler plugins and patterns with notes
- Pattern tracks (Beat/Bassline) hold drum patterns in the song editor
- The Mixer has channels (0=Master) where tracks route their audio
- Time is measured in ticks: 192 ticks = 1 bar (4/4 time)
- Notes use MIDI key numbers: 60=C4 (middle C), 69=A4
- Volume: 0-200 (100=normal), Panning: -100 to +100""",
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
pitch). An audiofileprocessor track without a sample is silent."""
            + (" For synthesized drums use kicker instead."
               if "kicker" in pool else "")
        )

    if "zynaddsubfx" in pool:
        blocks.append(
            """ZynAddSubFX: For rich sounds, add a track with instrument "zynaddsubfx",
then load one of ~950 factory presets via load_zyn_preset (browse with
list_zyn_presets). Fine-tune with set_zyn_params."""
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
        "add_effect (see lmms://reference/effects for the full list). "
        "Typical chains:\n"
        + "\n".join(f"- {use}: {_chain_text(c)}" for use, c in chains.items())
    )

    blocks.append(
        f"""Default projects directory: {projects_dir}
When saving a project, use this directory if no specific path is given.
Example: save to {projects_dir}/my_song.mmpz

Workflow: Create a project -> Add tracks -> Load instruments/presets ->
Add notes -> Add effects -> Configure mixer -> Save"""
    )
    return "\n\n".join(blocks) + "\n"


mcp = MCPServer(
    "LMMS MCP Server",
    instructions=build_instructions(_availability, DEFAULT_PROJECTS_DIR),
)


def _plugin_tool(fn):
    """Register a tool whose docstring lists the available plugins.

    The docstring may contain $INSTRUMENTS / $EFFECTS; it is rendered
    from the installed LMMS before registration, so the tool description
    the agent sees names only plugins that exist. The raw template is
    kept in fn.__plugin_doc_template__.
    """
    fn.__plugin_doc_template__ = fn.__doc__ or ""
    fn.__doc__ = render_plugin_doc(fn.__plugin_doc_template__, _availability)
    return mcp.tool()(fn)


# ──────────────────────────────────────────────────────────────────
# PROJECT TOOLS
# ──────────────────────────────────────────────────────────────────


@mcp.tool()
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
    )
    set_project(proj)
    return json.dumps({
        "message": "New project created",
        "bpm": bpm,
        "time_signature": f"{time_sig_numerator}/{time_sig_denominator}",
        "master_volume": master_volume,
        "master_pitch": master_pitch,
    })


@mcp.tool()
def load_project(path: str) -> str:
    """Load an existing LMMS project file (.mmpz or .mmp).

    Args:
        path: Full path to the LMMS project file
    """
    proj = LMMSProject()
    proj.load(path)
    set_project(proj)
    info = proj.get_info()
    return json.dumps({
        "message": f"Loaded project from {path}",
        "bpm": info.bpm,
        "time_signature": info.time_signature,
        "track_count": info.track_count,
        "tracks": [{"index": t.index, "name": t.name, "type": t.type_name} for t in info.tracks],
    })


@mcp.tool()
def save_project(
    path: str = "",
    compressed: bool | None = None,
) -> str:
    """Save the current LMMS project to a file.

    Args:
        path: File path to save to. If empty, saves to the default projects directory.
            If only a filename is given (e.g. "song.mmpz"), it's saved in the default directory.
        compressed: If True, saves as .mmpz (compressed). If False, saves as .mmp
            (plain XML). By default (None) the format follows the file extension:
            ".mmp" is always plain XML, everything else compressed. Writing
            compressed data into a .mmp breaks LMMS and other tools.
    """
    proj = get_project()

    if not path and not proj.path:
        projects_dir = Path(DEFAULT_PROJECTS_DIR)
        projects_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(projects_dir / "untitled.mmpz")
    elif path and not Path(path).is_absolute():
        projects_dir = Path(DEFAULT_PROJECTS_DIR)
        projects_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(projects_dir / path)
    else:
        save_path = path if path else None

    if compressed is None:
        # Follow the extension: plain XML for .mmp, compressed otherwise.
        # (LMMSProject.save does the same inference; resolve it here so
        # the response reports the actual format.)
        compressed = not (
            save_path and str(save_path).lower().endswith(".mmp")
        ) if save_path else True

    result_path = proj.save(save_path, compressed=compressed) if save_path else proj.save(compressed=compressed)

    return json.dumps({
        "message": f"Project saved to {result_path}",
        "path": result_path,
        "compressed": compressed,
    })


@mcp.tool()
def get_project_info() -> str:
    """Get comprehensive information about the current LMMS project.

    Returns tempo, time signature, tracks, mixer channels, and more.
    """
    proj = get_project()
    info = proj.get_info()
    return json.dumps(info.to_dict(), indent=2)


@mcp.tool()
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
    result = proj.add_track(
        "instrument", name,
        instrument=plugin,
        mixer_channel=mixer_channel,
        volume=volume,
        panning=panning,
    )
    result["instrument"] = instrument
    if plugin != instrument:
        result["lmms_plugin"] = plugin
    if custom:
        result["custom_plugin"] = True
    else:
        warning = _verification_warning(av, instrument)
        if warning:
            result["warning"] = warning
    return result


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
        mixer_channel: Mixer channel number (0=Master, 1+=custom channels)
        volume: Track volume (0-200, 100=normal)
        panning: Track panning (-100 to +100, 0=center)

    The response reports "instrument" (the name you gave). When the
    installed LMMS knows that plugin under another name, it also reports
    "lmms_plugin", the name written into the project; keep using the
    "instrument" name yourself.
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


@mcp.tool()
def set_audiofile_sample(
    track_index: int,
    file_path: str,
    amp: int = 100,
    reversed: bool = False,
    looped: int = 0,
    stutter: bool = False,
    start: float = 0.0,
    end: float = 1.0,
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
    for your own files (WAV/OGG/FLAC/MP3). If the file cannot be found
    the response contains a "warning" (the sample is still set).

    Pitch: the sample sounds at its original pitch when the note equals
    the track's base note, which this server creates as key 57 (A3 in this
    server's naming where 60=C4; LMMS's own UI calls it A4). So trigger
    drum one-shots with key=57. Other keys transpose the sample (higher
    key = higher pitch and shorter), which can be used on purpose for
    tuned samples or toms.

    Note length: the sample is cut off when the note ends, so use a
    length at least as long as the sample (24 ticks is enough for most
    short hits; use 48+ for open hats, crashes and other long samples).

    Args:
        track_index: Zero-based index of the audiofileprocessor track
        file_path: Sample path, relative to LMMS factory samples or absolute
        amp: Amplification in percent (0-500, 100=unchanged)
        reversed: Play the sample backwards
        looped: 0=off (one-shot), 1=loop, 2=ping-pong loop
        stutter: Restart the sample from the start for every new note
        start: Sample start point as a fraction 0..1 (e.g. 0.1 skips the
            first 10%); must be smaller than end
        end: Sample end point as a fraction 0..1 (1.0 = whole sample)
    """
    proj = get_project()
    try:
        xml_parser.set_audiofile_sample(
            proj.root, track_index, file_path, amp=amp, reversed=reversed,
            looped=looped, stutter=stutter, start=start, end=end,
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
    return json.dumps(result)


@mcp.tool()
def add_sample_track(
    name: str,
    mixer_channel: int = 0,
    volume: int = 100,
    panning: int = 0,
) -> str:
    """Add a sample track for arranging audio files.

    Args:
        name: Track name
        mixer_channel: Mixer channel number (0=Master)
        volume: Track volume (0-200)
        panning: Track panning (-100 to +100)
    """
    proj = get_project()
    result = proj.add_track(
        "sample", name,
        mixer_channel=mixer_channel,
        volume=volume,
        panning=panning,
    )
    return json.dumps(result)


@mcp.tool()
def add_automation_track(name: str = "Automation track") -> str:
    """Add an automation track for parameter automation.

    Args:
        name: Track name
    """
    proj = get_project()
    result = proj.add_track("automation", name)
    return json.dumps(result)


@mcp.tool()
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


@mcp.tool()
def remove_track(track_index: int) -> str:
    """Remove a track by its index.

    Args:
        track_index: Zero-based index of the track to remove
    """
    proj = get_project()
    result = proj.remove_track(track_index)
    return json.dumps(result)


@mcp.tool()
def get_track(track_index: int) -> str:
    """Get detailed information about a specific track.

    Args:
        track_index: Zero-based index of the track
    """
    proj = get_project()
    info = proj.get_info()
    if track_index >= len(info.tracks):
        return json.dumps({"error": f"Track index {track_index} out of range. Have {len(info.tracks)} tracks."})
    track = info.tracks[track_index]
    return json.dumps(track.to_dict(), indent=2)


@mcp.tool()
def list_tracks() -> str:
    """List all tracks in the current project with their index, name, and type."""
    proj = get_project()
    info = proj.get_info()
    tracks = [
        {
            "index": t.index,
            "name": t.name,
            "type": t.type_name,
            "instrument": t.instrument,
            "muted": t.muted,
            "patterns": len(t.patterns),
        }
        for t in info.tracks
    ]
    return json.dumps({"tracks": tracks, "count": len(tracks)}, indent=2)


@mcp.tool()
def set_track_volume(track_index: int, volume: int) -> str:
    """Set the volume of a track.

    Args:
        track_index: Zero-based track index
        volume: Volume level (0-200, 100=normal)
    """
    proj = get_project()
    from .xml_parser import find_tracks
    tracks = find_tracks(proj.root)
    if track_index >= len(tracks):
        return json.dumps({"error": f"Track index {track_index} out of range"})

    track = tracks[track_index]
    track_type = int(track.get("type", "0"))

    if track_type == 0:
        inst = track.find("instrumenttrack")
        if inst is not None:
            inst.set("vol", str(volume))
    elif track_type == 2:
        samp = track.find("sampletrack")
        if samp is not None:
            samp.set("vol", str(volume))

    proj._modified = True
    return json.dumps({"message": f"Set track {track_index} volume to {volume}"})


@mcp.tool()
def set_track_panning(track_index: int, panning: int) -> str:
    """Set the panning of a track.

    Args:
        track_index: Zero-based track index
        panning: Pan value (-100=left, 0=center, 100=right)
    """
    proj = get_project()
    from .xml_parser import find_tracks
    tracks = find_tracks(proj.root)
    if track_index >= len(tracks):
        return json.dumps({"error": f"Track index {track_index} out of range"})

    track = tracks[track_index]
    track_type = int(track.get("type", "0"))

    if track_type == 0:
        inst = track.find("instrumenttrack")
        if inst is not None:
            inst.set("pan", str(panning))
    elif track_type == 2:
        samp = track.find("sampletrack")
        if samp is not None:
            samp.set("pan", str(panning))

    proj._modified = True
    return json.dumps({"message": f"Set track {track_index} panning to {panning}"})


@mcp.tool()
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

    tracks[track_index].set("muted", "1" if muted else "0")
    proj._modified = True
    return json.dumps({"message": f"Track {track_index} {'muted' if muted else 'unmuted'}"})


@mcp.tool()
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

    tracks[track_index].set("solo", "1" if solo else "0")
    proj._modified = True
    return json.dumps({"message": f"Track {track_index} {'soloed' if solo else 'unsoloed'}"})


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

    key may be a MIDI number (0-127) or a note name ("C4", "Bb2").
    Raises ValueError with an actionable message on the first problem.
    """
    if isinstance(key, str):
        key = note_name_to_midi(key)
    elif isinstance(key, bool) or not isinstance(key, (int, float)):
        raise ValueError(
            f"key must be a MIDI number 0-127 or a note name like 'Bb2', "
            f"got {key!r}."
        )
    else:
        key = _as_int(key, "key")
    if not MIDI_KEY_MIN <= key <= MIDI_KEY_MAX:
        raise ValueError(
            f"key {key} is out of range: MIDI keys are "
            f"{MIDI_KEY_MIN}-{MIDI_KEY_MAX} (C-1 to G9, 60 = C4)."
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
        note.get("key", 60),
        note.get("pos", 0),
        note.get("length", 48),
        note.get("volume", 100),
        note.get("panning", 0),
        note.get("pattern_index"),
    )
    return proj.add_note(track_index=track_index, **fields)


@mcp.tool()
def add_note(
    track_index: int,
    key: int = 60,
    pos: int = 0,
    length: int = 48,
    volume: int = 100,
    panning: int = 0,
    pattern_index: int | None = None,
) -> str:
    """Add a note to a track's pattern.

    On invalid input (key outside 0-127, negative pos, length <= 0, bad
    track or pattern index) nothing is added and the response is
    {"error": "..."} explaining what to fix.

    Args:
        track_index: Zero-based track index (must be an instrument or pattern track)
        key: MIDI note number (0-127). 60=C4 (middle C), 69=A4, 127=G9.
            To use a note name instead (e.g. 'Bb2') call add_note_by_name.
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


@mcp.tool()
def add_note_by_name(
    track_index: int,
    note_name: str = "C4",
    pos: int = 0,
    length: int = 48,
    volume: int = 100,
) -> str:
    """Add a note using a note name (e.g. 'C4', 'A#3', 'F#5', 'Bb2').

    Sharps (#) and flats (b) are both accepted and enharmonic spellings
    map to the same key: 'Bb2' = 'A#2' = MIDI 46, 'Db3' = 'C#3' = 49,
    'Cb4' = 'B3' = 59, 'B#3' = 'C4' = 60. The note letter is
    case-insensitive ('bb2' = Bb2, 'b3' = B3). Valid range is C-1 (key 0)
    to G9 (key 127); C4 = 60.

    On invalid input (unknown note name, out-of-range note, negative pos,
    length <= 0, bad track index) nothing is added and the response is
    {"error": "..."} explaining the accepted format.

    Args:
        track_index: Zero-based track index
        note_name: Note name like 'C4', 'A#3', 'F5', 'Bb2' (letter A-G,
            optional # or b, octave -1..9)
        pos: Position in ticks RELATIVE TO THE PATTERN START
            (192 ticks = 1 bar, 48 = 1/4 note)
        length: Note length in ticks (12 = 1/16 note, 24 = 1/8, 48 = 1/4 (one beat),
            96 = 1/2, 192 = 1 bar)
        volume: Note velocity (0-200)
    """
    try:
        key = note_name_to_midi(note_name)
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


@mcp.tool()
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

    Args:
        track_index: Zero-based track index
        notes: List of note objects, each with: key (int 0-127, or a note
            name str like "C4", "A#3", "Bb2" - sharps and flats both work,
            e.g. "Bb2" = "A#2" = 46; range C-1..G9, 60 = C4),
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
    return json.dumps(response)


# ──────────────────────────────────────────────────────────────────
# MIXER TOOLS
# ──────────────────────────────────────────────────────────────────


@mcp.tool()
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


@mcp.tool()
def get_mixer_channels() -> str:
    """Get all mixer channels with their settings."""
    proj = get_project()
    info = proj.get_info()
    channels = [ch.to_dict() for ch in info.mixer_channels]
    return json.dumps({"channels": channels, "count": len(channels)}, indent=2)


@mcp.tool()
def set_mixer_channel_volume(channel_num: int, volume: float) -> str:
    """Set the volume of a mixer channel.

    Args:
        channel_num: Channel number (0=Master)
        volume: Volume (0.0-2.0, 1.0=0dB)
    """
    proj = get_project()
    from .xml_parser import find_mixer_channels
    channels = find_mixer_channels(proj.root)
    for ch in channels:
        if int(ch.get("num", "-1")) == channel_num:
            ch.set("volume", str(volume))
            proj._modified = True
            return json.dumps({"message": f"Set mixer channel {channel_num} volume to {volume}"})
    return json.dumps({"error": f"Mixer channel {channel_num} not found"})


@mcp.tool()
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
        song = root.find("song")
        mixer = song.find("mixer")
        channels = mixer.findall("mixerchannel")
        if not 0 <= target_index < len(channels):
            raise ValueError(
                f"Mixer channel {target_index} out of range "
                f"(0-{len(channels) - 1})"
            )
        return channels[target_index]
    raise ValueError("target_type must be 'track' or 'mixer'")


@_plugin_tool
def add_effect(
    target_type: str,
    target_index: int,
    effect: str,
    wet: float = 1.0,
    enabled: bool = True,
    position: int | None = None,
) -> str:
    """Add an effect to a track's or mixer channel's effect chain.

    Args:
        target_type: "track" or "mixer"
        target_index: Track index or mixer channel number (0=Master)
        effect: Built-in LMMS effect name. Valid in the installed LMMS:
            $EFFECTS
        wet: Wet/dry mix 0.0-1.0 (1.0=full effect)
        enabled: Whether the effect is active
        position: Chain position to insert at (None=end of chain)
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
            available=_effect_choices(av),
        )
        warning = _verification_warning(av, normalized)
        if warning:
            result["warning"] = warning
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({
            "error": str(exc),
            "valid_effects": _effect_choices(av),
            "recommendations": lmms_app.adapt_recommendations(
                effects_mod.EFFECT_RECOMMENDATIONS, av.effects
            ),
        })


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def get_effect_chain(target_type: str, target_index: int) -> str:
    """List all effects on a track's or mixer channel's effect chain.

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


@mcp.tool()
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


@mcp.tool()
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

    Args:
        track_index: Index of the target track
        preset: "Category/Name", plain name, "0034-Name" or absolute path
    """
    proj = get_project()
    try:
        preset_xml, match = zyn_presets.load_preset(preset)
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


@mcp.tool()
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
# LMMS APP INTEGRATION (VERSION CHECK + RENDER/EXPORT)
# ──────────────────────────────────────────────────────────────────


@mcp.tool()
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


@mcp.tool()
def render_project(
    output_path: str | None = None,
    file_format: str = "wav",
    samplerate: int = 44100,
    bitrate: int = 160,
    oversampling: int = 1,
    interpolation: str = "sincfastest",
) -> str:
    """Export the current project to an audio file using the installed LMMS.

    This launches LMMS in headless render mode (no GUI). Requires a
    working LMMS installation. The project is saved first, then rendered.

    Args:
        output_path: Output file path (default: <project>.wav in the
            same directory)
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
    """
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

    # Save current state before rendering. Keep the compression format
    # consistent with the file extension - compressed bytes inside a
    # .mmp file cannot be read by LMMS and other tools.
    proj.save(proj.path, compressed=str(proj_path).lower().endswith(".mmpz"))

    import subprocess
    cmd = [
        str(exe), "render", str(proj_path),
        "-o", output_path, "-f", fmt,
        "-s", str(samplerate), "-b", str(bitrate),
        "-x", str(oversampling), "-i", interp,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": "Rendering timed out after 300s. The song may be too long."
        })

    out_file = Path(output_path)
    success = proc.returncode == 0 and out_file.is_file()
    result = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "output_file": str(out_file),
        "oversampling": oversampling,
        "interpolation": interp,
        "success": success,
    }
    if success:
        result["message"] = f"Rendered to {out_file} ({out_file.stat().st_size} bytes)"
        result["size_bytes"] = out_file.stat().st_size
    else:
        result["error"] = f"Render failed: {proc.stderr.strip() or proc.stdout.strip()}"
    return json.dumps(result)


@mcp.tool()
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
            if name in av.instruments or name in av.effects:
                continue
            required = lmms_app.PLUGIN_REQUIRES.get(name)
            reason = f"requires {required}" if required else "not installed"
            alternative = lmms_app.PLUGIN_SUBSTITUTES.get(name)
            if alternative in av.instruments or alternative in av.effects:
                reason += f"; use {alternative}"
            missing[name] = reason
        result["unavailable_in_installed_lmms"] = missing
    result["note"] = (
        "Custom plugins can be tried as instruments by their library name "
        "(without extension, and without the 'lib' prefix on Linux) in "
        "add_instrument_track. add_effect only accepts the built-in effects "
        "listed above. lmms_plugin_names lists instruments that LMMS loads "
        "under another name than the one offered; add_instrument_track "
        "handles that, keep using the offered name."
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def scan_vst_directory(directory: str, recursive: bool = True) -> str:
    """Scan a folder for VST plugin DLLs (.dll files).

    Args:
        directory: Path to scan (e.g. "C:/VSTPlugins")
        recursive: Include subdirectories
    """
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


@mcp.tool()
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
        dll_path: Absolute path to the VST .dll file
        mixer_channel: Mixer channel number (0=Master)
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

    result = proj.add_track(
        "instrument", name,
        instrument="vestige",
        mixer_channel=mixer_channel,
        volume=volume,
        panning=panning,
    )

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


@mcp.tool()
def assign_sample_file(track_index: int, file_path: str) -> str:
    """Assign an audio file to all clips on a sample track.

    Use relative paths for LMMS's built-in samples (e.g.
    "drums/kick01.ogg") or absolute paths for your own files.

    Args:
        track_index: Index of the sample track
        file_path: Audio file path (WAV/OGG/MP3/FLAC)
    """
    proj = get_project()
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


@mcp.tool()
def place_sample_clip(
    track_index: int,
    file_path: str,
    pos_bars: float = 0,
    length_bars: float = 1,
) -> str:
    """Place an audio file clip on a sample track at a given bar.

    Args:
        track_index: Index of the sample track
        file_path: Audio file path (relative to LMMS samples or absolute)
        pos_bars: Start position in bars (0 = beginning)
        length_bars: Clip length in bars
    """
    proj = get_project()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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
    value).

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


@mcp.tool()
def set_tempo(bpm: int) -> str:
    """Set the song tempo (BPM).

    Args:
        bpm: Tempo in beats per minute (10-999)
    """
    proj = get_project()
    proj.set_tempo(bpm)
    return json.dumps({"message": f"Tempo set to {bpm} BPM"})


@mcp.tool()
def set_time_signature(numerator: int, denominator: int) -> str:
    """Set the time signature.

    Args:
        numerator: Beats per bar (e.g. 4, 3, 6)
        denominator: Beat unit (e.g. 4 for quarter notes, 8 for eighth notes)
    """
    proj = get_project()
    proj.set_time_signature(numerator, denominator)
    return json.dumps({"message": f"Time signature set to {numerator}/{denominator}"})


@mcp.tool()
def set_master_volume(volume: int) -> str:
    """Set the master volume.

    Args:
        volume: Master volume (0-200, 100=normal)
    """
    proj = get_project()
    proj.set_master_volume(volume)
    return json.dumps({"message": f"Master volume set to {volume}"})


@mcp.tool()
def set_master_pitch(pitch: int) -> str:
    """Set the master pitch offset.

    Args:
        pitch: Pitch offset in semitones (-12 to +12)
    """
    proj = get_project()
    proj.set_master_pitch(pitch)
    return json.dumps({"message": f"Master pitch set to {pitch} semitones"})


# ──────────────────────────────────────────────────────────────────
# UTILITY / CONVERSION TOOLS
# ──────────────────────────────────────────────────────────────────


@mcp.tool()
def note_name_to_key(name: str) -> str:
    """Convert a note name (e.g. 'C4') to its MIDI key number.

    Sharps and flats both work and enharmonics agree ('Bb2' = 'A#2' = 46,
    'Cb4' = 'B3' = 59). Range C-1 (0) to G9 (127); C4 = 60. Invalid names
    return {"error": "..."} stating the accepted format.

    Args:
        name: Note name like 'C4', 'A#3', 'F#5', 'Bb2'
    """
    try:
        key = note_name_to_midi(name)
        return json.dumps({"note": name, "midi_key": key})
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def key_to_note_name(key: int) -> str:
    """Convert a MIDI key number to its note name.

    Args:
        key: MIDI key number (0-127)
    """
    if not 0 <= key <= 127:
        return json.dumps({"error": "MIDI key must be 0-127"})
    return json.dumps({"midi_key": key, "note": midi_to_note_name(key)})


@mcp.tool()
def bars_to_ticks_converter(bars: int) -> str:
    """Convert bars to ticks (192 ticks per bar in 4/4 time).

    Args:
        bars: Number of bars
    """
    ticks = bars_to_ticks(bars)
    return json.dumps({"bars": bars, "ticks": ticks, "ticks_per_bar": TICKS_PER_BAR})


@mcp.tool()
def ticks_to_bars_converter(ticks: int) -> str:
    """Convert ticks to bars (192 ticks per bar in 4/4 time).

    Args:
        ticks: Number of ticks
    """
    bars = ticks_to_bars(ticks)
    return json.dumps({"ticks": ticks, "bars": round(bars, 3), "ticks_per_bar": TICKS_PER_BAR})


@mcp.tool()
def generate_scale(
    root_note: str = "C4",
    scale_type: str = "major",
    num_octaves: int = 1,
) -> str:
    """Generate a musical scale as MIDI key numbers and note names.

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
        root_key = note_name_to_midi(root_note)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    intervals = scales[scale_type]
    notes = []
    for octave in range(num_octaves):
        for interval in intervals:
            key = root_key + octave * 12 + interval
            if key <= 127:
                notes.append({
                    "key": key,
                    "note": midi_to_note_name(key),
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


@mcp.resource("lmms://reference/effects")
def resource_effects() -> str:
    """Effects available in the installed LMMS (verified)."""
    av = _availability
    return json.dumps({
        "note": _reference_note(av, "effects")
                + " Effects that host external plugins (LADSPA, LV2, VST) "
                  "cannot be added with add_effect.",
        "effects": [
            {
                "name": name,
                "description": effects_mod.KNOWN_EFFECTS[name][0],
                "controls_node": effects_mod.KNOWN_EFFECTS[name][1],
            }
            for name in sorted(av.effects)
        ],
        "recommendations_by_use_case": lmms_app.adapt_recommendations(
            effects_mod.EFFECT_RECOMMENDATIONS, av.effects
        ),
    }, indent=2)


@mcp.resource("lmms://reference/note_names")
def resource_note_names() -> str:
    """MIDI note number to note name mapping reference."""
    mapping = []
    for key in range(128):
        mapping.append({
            "key": key,
            "note": midi_to_note_name(key),
        })
    return json.dumps({"mapping": mapping, "total": 128}, indent=2)


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
        key: Root note for the melody (e.g. 'C4', 'A3')
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
   "drums/hihat_closed02.ogg") and put that drum's notes at key 57 (the
   track's base note = original sample pitch) on its own track."""
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
    return f"""Create a drum pattern with style: {pattern_style}.

Description: {description}

Steps:
1. {find_track}
2. Add notes at the correct positions:
   - Use position values: step * 12 ticks (12 ticks per 1/16 step)
   - Each step is 12 ticks apart
   - Total pattern length: {steps} steps = {steps * 12} ticks
3. Typical MIDI keys for drums on a single multi-drum track{key_note}
   - Key 36 = Kick Drum
   - Key 38 = Snare
   - Key 42 = Closed Hi-Hat
   - Key 46 = Open Hi-Hat
   - Key 49 = Crash Cymbal
   - Key 51 = Ride Cymbal
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
   - Route tracks to dedicated mixer channels
3. Create mixer channels if needed for:
   - Drum bus
   - Bass bus
   - Lead/Melody bus
   - FX returns (reverb, delay)
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
    for tool_name, plugin in TOOL_REQUIRES.items():
        if plugin not in av.instruments:
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
