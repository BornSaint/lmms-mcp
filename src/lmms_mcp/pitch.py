"""Pitch calibration: key 57 (A4) sounds at 440 Hz on every offered instrument.

An instrument track has a base note (the <instrumenttrack basenote>
attribute, default 57 = A4). LMMS transposes everything the track plays by
(57 - basenote) semitones: basenote 45 plays 12 semitones higher. Most
plugins sound at the pitch of the key, but some do not. They are measured
(render a note per bar at -x 1, estimate the fundamental) and listed in
PITCH_OFFSET_SEMITONES; the track's written basenote compensates:

    written basenote = user base note - pitch offset

"User base note" is what the agent sees (57 = A4 unless it chose another
tuning); "written" is the attribute in the project file. Use
effective_basenote() to go from the first to the second and
get_base_note() / set_base_note() to read and write a track's. A plugin's
offset is not the whole story for ZynAddSubFX: every preset carries its own
octave transposition (see zyn_preset_octave), which track_pitch_offset()
adds to the table value so each preset sounds at the pitch of the key too.

The agent can also tune a track: set_tuning() writes a base note (the key at
which a sample plays at its recorded pitch, e.g. C4 for a piano sample
recorded at C4) and a fine tune in cents (`pitch`, with the `pitchrange` it
needs). tuning_report() / tuning_hint() describe the result.
"""

import math
import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from .models import (
    DEFAULT_BASE_NOTE,
    KEY_MAX,
    KEY_MIN,
    KEY_RANGE_TEXT,
    key_to_note_name,
    note_name_to_key,
)

# LMMS accepts base notes 0..127 (NumKeys); it clamps anything outside.
BASE_NOTE_MIN = 0
BASE_NOTE_MAX = 127

# Semitones a plugin sounds BELOW the key it is played at (measured with
# LMMS 1.2.2), i.e. how far its track has to be raised. Keyed by the
# lowercase plugin name LMMS loads (the <instrument name> attribute).
# Plugins that are not listed sound at the pitch of the key.
#
# zynaddsubfx: LMMS hands the LMMS key (A4 = 57) to ZynAddSubFX as the MIDI
#   note number (A4 = 69), so every sound is one octave low.
# opl2 (opulenz in LMMS 1.3, not measured there): the default patch
#   frequency-modulates its carrier with an operator at half the carrier
#   frequency, which makes the fundamental of the sound, and the pitch one
#   hears, an octave below the key. This depends on the patch (a preset
#   changes it): see opl2_patch_offset, which track_pitch_offset uses.
PITCH_OFFSET_SEMITONES: dict[str, int] = {
    "zynaddsubfx": 12,
    "opl2": 12,
}

# Keys each plugin's default sound plays at the right pitch, as the plugin
# receives them (the played key plus the track's transposition, see
# playable_key_range). Outside this range the sound is silent or at the
# wrong pitch. Measured with LMMS 1.2.2 by rendering every key 0-108; plugins
# not listed play the whole range (sf2player depends on the soundfont).
#
# zynaddsubfx: drops notes from MIDI note 108 up.
# opl2: the compensated carrier (twice the key's frequency) tops out near
#   6 kHz; above, the note is silent. Measured on the default patch only: a
#   preset whose operators have higher multipliers falls silent earlier.
# papu / freeboy: the Game Boy channel plays C2 to B7 only.
# sid: the 16-bit frequency register wraps above about 3.8 kHz (wrong pitch).
# nes: the waveform period is a whole number of samples, so the pitch is
#   sharp by more than 25 cents above key 72 and by 1.75 semitones at 104.
# vibedstrings: higher strings die out before the note is heard.
# sfxr: a sound-effect synth (unpitched, about 5.5 semitones flat); the
#   sound fades with the key and is silent from key 80.
PLUGIN_KEY_LIMITS: dict[str, tuple[int, int]] = {
    "zynaddsubfx": (0, 107),
    "opl2": (0, 102),
    "papu": (24, 95),
    "freeboy": (24, 95),
    "sid": (2, 94),
    "nes": (0, 72),
    "vibedstrings": (0, 78),
    "sfxr": (0, 78),
}

# Largest track fine tune (the <instrumenttrack pitch> attribute, in cents)
# each plugin follows, measured with LMMS 1.2.2 by rendering a note with a
# fine tune of +-50, +-100, +-150, +-250 and +-1200 cents. Plugins not listed
# follow the fine tune exactly up to FINE_TUNE_MAX_CENTS (LMMS limits it to
# 100 cents times the track's `pitchrange`, which set_tuning raises as needed).
#
# zynaddsubfx: LMMS sends the fine tune as a MIDI pitch bend and ZynAddSubFX
#   always bends +-100 cents at full deflection, whatever pitchrange says. A
#   range above 1 makes the sound pitch / range cents (e.g. +150 with range 2
#   sounds +75), so ZynAddSubFX tracks keep pitchrange 1 and +-100 cents.
# opl2: ignores the fine tune completely.
# kicker: a drum that ignores the key; neither the fine tune nor the base
#   note changes its sound (see PLUGINS_IGNORING_KEY).
PLUGIN_FINE_TUNE_LIMITS: dict[str, int] = {
    "zynaddsubfx": 100,
    "opl2": 0,
    "kicker": 0,
}

# Plugins whose sound does not depend on the key, so tuning them is useless.
PLUGINS_IGNORING_KEY = frozenset({"kicker"})

FINE_TUNE_MAX_CENTS = 1200

_ZYN_PLUGIN = "zynaddsubfx"


def plugin_pitch_offset(plugin: str) -> int:
    """Semitones the plugin sounds below the key (0 when it is accurate)."""
    return PITCH_OFFSET_SEMITONES.get(plugin.strip().lower(), 0)


def effective_basenote(
    plugin: str,
    user_base_note: int = DEFAULT_BASE_NOTE,
    extra_offset: int = 0,
) -> int:
    """The basenote to write for a track whose user-facing base note is given.

    extra_offset is further compensation on top of the plugin's table
    entry (track_pitch_offset() includes it for a ZynAddSubFX preset). The
    result is the user base note minus the total offset, e.g. 45 for a
    ZynAddSubFX track at the default base note 57.

    Raises ValueError when the result is outside what LMMS accepts (0-127).
    """
    written = user_base_note - plugin_pitch_offset(plugin) - extra_offset
    if not BASE_NOTE_MIN <= written <= BASE_NOTE_MAX:
        raise ValueError(
            f"Base note {user_base_note} cannot be used with '{plugin}': "
            f"compensating its pitch would need {written}, but LMMS only "
            f"accepts {BASE_NOTE_MIN}-{BASE_NOTE_MAX}."
        )
    return written


# --- ZynAddSubFX presets ---------------------------------------------------

# Per synthesis engine of a kit item: the flag that enables it and its
# parameter branch, in the order the first enabled one is taken as the
# preset's main engine.
_ZYN_ENGINES = (
    ("add_enabled", "ADD_SYNTH_PARAMETERS"),
    ("pad_enabled", "PAD_SYNTH_PARAMETERS"),
    ("sub_enabled", "SUB_SYNTH_PARAMETERS"),
)


def _is_yes(element: ET.Element | None) -> bool:
    return element is not None and element.get("value") == "yes"


def zyn_octave_from_coarse_detune(coarse_detune: int) -> int:
    """Octave part of a ZynAddSubFX coarse_detune value (-8 .. +7).

    The value has 14 bits: the lowest 10 are semitones (signed), the upper
    4 the octave (two's complement), as ZynAddSubFX decodes it.
    """
    octave = (coarse_detune // 1024) % 16
    return octave - 16 if octave >= 8 else octave


def zyn_preset_octave(instrument: ET.Element | None) -> int:
    """Octave transposition built into a ZynAddSubFX preset (0 for most).

    `instrument` is the preset's INSTRUMENT element (a .xiz root child, or
    MASTER/PART/INSTRUMENT in a project). Every preset stores a coarse
    detune whose octave part shifts the sound; most factory presets use 0
    (85% of LMMS 1.2.2's), the others have e.g. -1 (basses) or +1. It is read from the
    first enabled kit item's first enabled engine (additive, pad, then
    subtractive). Layers whose octave differs from the main engine keep
    their offset to it. Returns 0 when there is no such value.
    """
    if instrument is None:
        return 0
    kit = instrument.find("INSTRUMENT_KIT")
    if kit is None:
        return 0
    for item in kit.findall("INSTRUMENT_KIT_ITEM"):
        enabled = item.find("par_bool[@name='enabled']")
        if enabled is not None and not _is_yes(enabled):
            continue
        for flag, branch in _ZYN_ENGINES:
            if not _is_yes(item.find(f"par_bool[@name='{flag}']")):
                continue
            par = item.find(f"{branch}/FREQUENCY_PARAMETERS/par[@name='coarse_detune']")
            if par is None:
                return 0
            try:
                return zyn_octave_from_coarse_detune(int(par.get("value", "0")))
            except ValueError:
                return 0
        return 0
    return 0


# --- OPL2 patches ------------------------------------------------------------

# The frequency multiplier of an OPL2 operator (the op1_mul / op2_mul
# attribute, 0-15), in half units: 0 is x0.5, 1 is x1, 2 is x2 ...
_OPL2_MULTIPLIER_HALVES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 20, 24, 24, 30, 30)
# LMMS's default patch: op1_mul, op2_mul, op1_lvl.
_OPL2_DEFAULT_MUL = (0, 1)
_OPL2_DEFAULT_LEVEL = 62
# An operator at x0.5 is heard as the pitch (an octave below the key) when it
# is loud enough: its sub-octave partial is within 3 dB of the key's at op1
# level 62, 8 dB at 54, 10 dB at 50 and 12 dB at 46 (measured, FM mode). At
# 46 and below (the factory preset Bubbly_days is 46) the key's pitch wins.
_OPL2_SUBOCTAVE_MIN_LEVEL = 50


def _opl2_attr(settings: ET.Element | None, name: str, default: int) -> int:
    if settings is not None and settings.get(name) is not None:
        try:
            return int(float(settings.get(name)))
        except ValueError:
            pass
    return default


def opl2_patch_offset(settings: ET.Element | None) -> int:
    """Semitones an OPL2 patch sounds below the key it is played at.

    `settings` is the <OPL2> element of the instrument (None = LMMS's
    default patch). LMMS plays the plugin's operators at the key's
    frequency times their multiplier; the pitch one hears is the greatest
    common divisor of the two multipliers, except that a quiet operator at
    x0.5 (op1 level below 50) does not count. The default patch (op1 x0.5,
    op2 x1) is an octave low (12), a patch with both operators at x1 or x2
    sounds at the key (0), both at x2 an octave high (-12). Measured with
    LMMS 1.2.2: all 15 factory OpulenZ presets and a range of operator
    combinations, in FM and additive mode.
    """
    mul1 = max(0, min(15, _opl2_attr(settings, "op1_mul", _OPL2_DEFAULT_MUL[0])))
    mul2 = max(0, min(15, _opl2_attr(settings, "op2_mul", _OPL2_DEFAULT_MUL[1])))
    level1 = _opl2_attr(settings, "op1_lvl", _OPL2_DEFAULT_LEVEL)
    halves = [_OPL2_MULTIPLIER_HALVES[mul2]]
    if mul1 != 0 or level1 >= _OPL2_SUBOCTAVE_MIN_LEVEL:
        halves.append(_OPL2_MULTIPLIER_HALVES[mul1])
    return round(-12 * math.log2(math.gcd(*halves) / 2))


# --- instrument tracks -------------------------------------------------------

def _plugin_of(inst_track: ET.Element) -> str:
    inst = inst_track.find("instrument")
    return (inst.get("name") if inst is not None else "") or ""


def track_pitch_offset(inst_track: ET.Element) -> int:
    """Total semitones the track's sound is below its key right now.

    The plugin's table entry, plus for a ZynAddSubFX track minus the octave
    its loaded preset transposes by (a preset with +1 octave is already
    12 semitones higher, so it needs 12 less). For an OPL2 track it is the
    offset of its patch (see opl2_patch_offset), which is the table entry
    for the default patch. `inst_track` is the <instrumenttrack> element.
    """
    plugin = _plugin_of(inst_track)
    offset = plugin_pitch_offset(plugin)
    if plugin.strip().lower() == _ZYN_PLUGIN:
        part = inst_track.find(
            "instrument/zynaddsubfx/ZynAddSubFX-data/MASTER/PART/INSTRUMENT"
        )
        offset -= 12 * zyn_preset_octave(part)
    elif plugin.strip().lower() == "opl2":
        inst = inst_track.find("instrument")
        offset = opl2_patch_offset(inst[0] if inst is not None and len(inst) else None)
    return offset


def _written_basenote(inst_track: ET.Element) -> int:
    try:
        return int(float(inst_track.get("basenote", DEFAULT_BASE_NOTE)))
    except ValueError:
        return DEFAULT_BASE_NOTE


def get_base_note(inst_track: ET.Element) -> int:
    """The user-facing base note of a track (written basenote + offset)."""
    return _written_basenote(inst_track) + track_pitch_offset(inst_track)


def _written_for(inst_track: ET.Element, base_note: int) -> int:
    """The basenote to write for a user-facing base note, or ValueError.

    Compensating the plugin's pitch (see track_pitch_offset) can need a
    value LMMS does not accept; the error then names the base notes that
    work on this track.
    """
    offset = track_pitch_offset(inst_track)
    written = base_note - offset
    if BASE_NOTE_MIN <= written <= BASE_NOTE_MAX:
        return written
    low = max(KEY_MIN, BASE_NOTE_MIN + offset)
    high = min(KEY_MAX, BASE_NOTE_MAX + offset)
    raise ValueError(
        f"Base note {key_to_note_name(base_note)} ({base_note}) cannot be "
        f"used on this '{_plugin_of(inst_track)}' track: compensating the "
        f"instrument's pitch would need basenote {written}, but LMMS only "
        f"accepts {BASE_NOTE_MIN}-{BASE_NOTE_MAX}. Choose a base note from "
        f"{key_to_note_name(low)} ({low}) to {key_to_note_name(high)} "
        f"({high})."
    )


def set_base_note(inst_track: ET.Element, base_note: int) -> int:
    """Write the basenote for a user-facing base note; returns the written one."""
    written = _written_for(inst_track, base_note)
    inst_track.set("basenote", str(written))
    return written


def keep_base_note(inst_track: ET.Element, offset_before: int) -> int:
    """Re-write the basenote after the track's pitch offset changed.

    Call after replacing a track's instrument data (a ZynAddSubFX preset)
    with the track_pitch_offset() value taken before. The sound keeps the
    same pitch relation to the key as before: only the difference of the
    offsets is applied, so a track nobody calibrated (e.g. from a project
    made elsewhere) is not suddenly shifted. The result is clamped to what
    LMMS accepts. Returns the written basenote.
    """
    delta = track_pitch_offset(inst_track) - offset_before
    written = max(
        BASE_NOTE_MIN,
        min(BASE_NOTE_MAX, _written_basenote(inst_track) - delta),
    )
    inst_track.set("basenote", str(written))
    return written


def _key_window(plugin: str, shift: int) -> tuple[int, int] | None:
    """Keys within 0-108 a plugin plays when the track adds `shift` to them."""
    limits = PLUGIN_KEY_LIMITS.get(plugin.strip().lower())
    if limits is None:
        return None
    low = max(KEY_MIN, limits[0] - shift)
    high = min(KEY_MAX, limits[1] - shift)
    return None if (low, high) == (KEY_MIN, KEY_MAX) else (low, high)


def default_playable_keys(plugin: str) -> tuple[int, int] | None:
    """playable_key_range() of a new track of the plugin (None = all keys)."""
    return _key_window(plugin, plugin_pitch_offset(plugin))


def playable_key_range(inst_track: ET.Element) -> tuple[int, int] | None:
    """Keys (within 0-108) the track plays at the right pitch, or None.

    None means the whole range. LMMS plays a track's key k as key
    k + 57 - basenote, which is what PLUGIN_KEY_LIMITS is about, so the
    limits move with the compensation and with a changed base note (e.g.
    ZynAddSubFX at basenote 45 plays keys up to 95).
    """
    return _key_window(
        _plugin_of(inst_track), DEFAULT_BASE_NOTE - _written_basenote(inst_track)
    )


def key_range_warning(inst_track: ET.Element, key: int) -> str | None:
    """Sentence warning that a key is outside the track's playable range."""
    keys = playable_key_range(inst_track)
    if keys is None or keys[0] <= key <= keys[1]:
        return None
    low, high = keys
    return (
        f"Key {key} ({key_to_note_name(key)}) is outside the keys "
        f"{_plugin_of(inst_track)} plays correctly on this track "
        f"({low}-{high}, {key_to_note_name(low)}-{key_to_note_name(high)}); "
        f"expect silence or a wrong pitch."
    )


# --- tuning a track: base note and fine tune ----------------------------------

def parse_base_note(value) -> int:
    """A base note given as a note name or an LMMS key, as a key (0-108).

    Accepts a note name ("C4", "Bb2"), an int (48), a whole float (48.0) and
    a string of digits ("48"). Raises ValueError saying what to pass.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"base_note must be a note name like 'C4' or an LMMS key number "
            f"({KEY_RANGE_TEXT}), got {value!r}."
        )
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            return note_name_to_key(text)
        value = int(text)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"base_note must be a note name or a whole key number, "
                f"got {value!r}. Use pitch_cents for a fraction of a "
                f"semitone."
            )
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(
            f"base_note must be a note name like 'C4' or an LMMS key number "
            f"({KEY_RANGE_TEXT}), got {value!r}."
        )
    if not KEY_MIN <= value <= KEY_MAX:
        hint = (
            f" If {value} is a MIDI note number, use {value - 12}."
            if value > KEY_MAX else ""
        )
        raise ValueError(
            f"base_note {value} is out of range: valid LMMS keys are "
            f"{KEY_RANGE_TEXT}.{hint}"
        )
    return value


def fine_tune_limit(plugin: str) -> int:
    """Cents of fine tune the plugin follows (FINE_TUNE_MAX_CENTS for most)."""
    return PLUGIN_FINE_TUNE_LIMITS.get(plugin.strip().lower(), FINE_TUNE_MAX_CENTS)


def _float_attr(inst_track: ET.Element, name: str, default: float) -> float:
    try:
        value = float(inst_track.get(name, default))
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def get_pitch_cents(inst_track: ET.Element) -> float:
    """The track's fine tune in cents (0 when none)."""
    return _float_attr(inst_track, "pitch", 0.0)


def get_pitchrange(inst_track: ET.Element) -> int:
    """The track's pitch range in semitones (LMMS default 1)."""
    return max(1, int(_float_attr(inst_track, "pitchrange", 1.0)))


def _format_cents(cents: float) -> str:
    """Cents for the XML or JSON: 18.0 -> "18", 12.5 -> "12.5"."""
    return str(int(cents)) if cents == int(cents) else repr(cents)


def json_cents(cents: float) -> float | int:
    return int(cents) if cents == int(cents) else cents


def _parse_cents(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"pitch_cents must be a number of cents (e.g. 18 or -12.5), "
            f"got {value!r}."
        )
    cents = float(value)
    if not math.isfinite(cents):
        raise ValueError(f"pitch_cents must be a finite number, got {value!r}.")
    if abs(cents) > FINE_TUNE_MAX_CENTS:
        raise ValueError(
            f"pitch_cents {value:g} is out of range: the fine tune is "
            f"-{FINE_TUNE_MAX_CENTS} to +{FINE_TUNE_MAX_CENTS} cents (100 "
            f"cents = 1 semitone). Use base_note to shift by whole "
            f"semitones or octaves."
        )
    return round(cents, 2)


@dataclass(frozen=True)
class Tuning:
    """What apply_tuning() writes to a track; None leaves that value alone."""

    basenote: int | None = None
    pitch: float | None = None
    pitchrange: int | None = None


def plan_tuning(
    inst_track: ET.Element,
    base_note: int | None = None,
    pitch_cents: float | None = None,
) -> Tuning:
    """Check a base note (an LMMS key) and a fine tune for the track.

    Nothing is written; raises ValueError saying what is wrong and what to
    do instead. Hand the result to apply_tuning(). base_note is the key at
    which the instrument or sample sounds at its natural pitch (user-facing,
    see get_base_note); pitch_cents raises (+) or lowers (-) every note.
    Both None gives an empty Tuning. The pitch range is raised to cover the
    fine tune (never lowered), except on plugins that do not follow it (see
    PLUGIN_FINE_TUNE_LIMITS).
    """
    plugin = _plugin_of(inst_track)
    if base_note is None and pitch_cents is None:
        return Tuning()
    if plugin.strip().lower() in PLUGINS_IGNORING_KEY:
        raise ValueError(
            f"'{plugin}' ignores the key it is played at, so neither a base "
            f"note nor a fine tune changes its sound. Use a pitched "
            f"instrument such as tripleoscillator to play notes."
        )

    written = None
    if base_note is not None:
        written = _written_for(inst_track, base_note)

    pitch = pitchrange = None
    if pitch_cents is not None:
        cents = _parse_cents(pitch_cents)
        limit = fine_tune_limit(plugin)
        if abs(cents) > limit:
            if limit == 0:
                raise ValueError(
                    f"'{plugin}' ignores the fine tune (pitch_cents), so "
                    f"{cents:+g} cents would not change its sound. Only "
                    f"whole semitones work: use base_note."
                )
            raise ValueError(
                f"'{plugin}' follows the fine tune only up to +-{limit} "
                f"cents, so {cents:+g} cents would sound wrong. Use "
                f"base_note for whole semitones and pitch_cents "
                f"(within +-{limit}) for the rest."
            )
        needed = max(1, math.ceil(abs(cents) / 100))
        pitch = cents
        pitchrange = (
            needed if limit < FINE_TUNE_MAX_CENTS
            else max(get_pitchrange(inst_track), needed)
        )
    return Tuning(written, pitch, pitchrange)


def apply_tuning(inst_track: ET.Element, tuning: Tuning) -> None:
    """Write a plan_tuning() result to the <instrumenttrack> element."""
    if tuning.basenote is not None:
        inst_track.set("basenote", str(tuning.basenote))
    if tuning.pitch is not None:
        inst_track.set("pitch", _format_cents(tuning.pitch))
        inst_track.set("pitchrange", str(tuning.pitchrange))


def set_tuning(
    inst_track: ET.Element,
    base_note: int | None = None,
    pitch_cents: float | None = None,
) -> Tuning:
    """plan_tuning() and apply_tuning() in one step; nothing is written on error."""
    tuning = plan_tuning(inst_track, base_note, pitch_cents)
    apply_tuning(inst_track, tuning)
    return tuning


def tuning_hint(inst_track: ET.Element) -> str:
    """One sentence saying what the track's keys play now."""
    plugin = _plugin_of(inst_track)
    if plugin.strip().lower() in PLUGINS_IGNORING_KEY:
        return (
            f"{plugin} ignores the key, so the base note and the fine tune "
            f"have no effect on it."
        )
    base = get_base_note(inst_track)
    cents = get_pitch_cents(inst_track)
    name = key_to_note_name(base)
    shift = DEFAULT_BASE_NOTE - base
    direction = "higher" if shift > 0 else "lower"
    tune = (
        f" (fine tune {cents:+g} cents)" if cents else ""
    )
    if plugin.strip().lower() == "audiofileprocessor":
        text = f"Key {base} ({name}) plays the sample at its original pitch"
        if shift:
            text += (
                f"; other keys transpose it (key {DEFAULT_BASE_NOTE} = A4 "
                f"is {abs(shift)} semitones {direction})"
            )
        return text + tune + "."
    if not shift:
        return f"Key {base} ({name}) sounds at 440 Hz{tune}."
    return (
        f"Key {base} ({name}) sounds at 440 Hz (A4): every key sounds "
        f"{abs(shift)} semitones {direction} than its note name{tune}."
    )


def tuning_report(inst_track: ET.Element) -> dict:
    """The track's tuning as a JSON-ready dict (see set_track_tuning)."""
    base = get_base_note(inst_track)
    report = {
        "instrument": _plugin_of(inst_track),
        "base_note": base,
        "base_note_name": key_to_note_name(base),
        "written_basenote": _written_basenote(inst_track),
        "pitch_cents": json_cents(get_pitch_cents(inst_track)),
        "pitchrange": get_pitchrange(inst_track),
    }
    keys = playable_key_range(inst_track)
    if keys:
        report["playable_keys"] = list(keys)
    report["hint"] = tuning_hint(inst_track)
    return report
