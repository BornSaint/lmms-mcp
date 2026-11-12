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
"""

from xml.etree import ElementTree as ET

from .models import DEFAULT_BASE_NOTE, KEY_MAX, KEY_MIN, key_to_note_name

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
#   hears, an octave below the key.
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
#   6 kHz; above, the note is silent.
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


# --- instrument tracks -------------------------------------------------------

def _plugin_of(inst_track: ET.Element) -> str:
    inst = inst_track.find("instrument")
    return (inst.get("name") if inst is not None else "") or ""


def track_pitch_offset(inst_track: ET.Element) -> int:
    """Total semitones the track's sound is below its key right now.

    The plugin's table entry, plus for a ZynAddSubFX track minus the octave
    its loaded preset transposes by (a preset with +1 octave is already
    12 semitones higher, so it needs 12 less). `inst_track` is the
    <instrumenttrack> element.
    """
    plugin = _plugin_of(inst_track)
    offset = plugin_pitch_offset(plugin)
    if plugin.strip().lower() == _ZYN_PLUGIN:
        part = inst_track.find(
            "instrument/zynaddsubfx/ZynAddSubFX-data/MASTER/PART/INSTRUMENT"
        )
        offset -= 12 * zyn_preset_octave(part)
    return offset


def _written_basenote(inst_track: ET.Element) -> int:
    try:
        return int(float(inst_track.get("basenote", DEFAULT_BASE_NOTE)))
    except ValueError:
        return DEFAULT_BASE_NOTE


def get_base_note(inst_track: ET.Element) -> int:
    """The user-facing base note of a track (written basenote + offset)."""
    return _written_basenote(inst_track) + track_pitch_offset(inst_track)


def set_base_note(inst_track: ET.Element, base_note: int) -> int:
    """Write the basenote for a user-facing base note; returns the written one."""
    written = base_note - track_pitch_offset(inst_track)
    if not BASE_NOTE_MIN <= written <= BASE_NOTE_MAX:
        raise ValueError(
            f"Base note {base_note} cannot be used with "
            f"'{_plugin_of(inst_track)}': compensating its pitch would need "
            f"{written}, but LMMS only accepts "
            f"{BASE_NOTE_MIN}-{BASE_NOTE_MAX}."
        )
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
