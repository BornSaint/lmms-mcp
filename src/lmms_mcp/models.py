"""Python data models for LMMS project structures."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum


class TrackType(IntEnum):
    INSTRUMENT = 0
    PATTERN = 1
    SAMPLE = 2
    EVENT = 3
    VIDEO = 4
    AUTOMATION = 5
    HIDDEN_AUTOMATION = 6


TICKS_PER_BAR = 192
TICKS_PER_STEP = 12
STEPS_PER_BAR = 16
DEFAULT_BPM = 140

# Note keys are LMMS keys, not MIDI note numbers: LMMS key = MIDI note - 12.
# C0 = 0, middle C (C4) = 48, A4 = 57 = 440 Hz (the instrument track's
# default base note). LMMS clamps keys above 108 (C9) to 108 when it loads
# a project, so 0..108 is the range whose pitch is played exactly.
KEY_MIN = 0
KEY_MAX = 108
MIDI_NOTE_OFFSET = 12
DEFAULT_KEY = 48  # C4, middle C
# The key an instrument track plays at its natural pitch (A4, 440 Hz; the
# original pitch of a sample). New tracks are calibrated so this holds on
# every offered instrument, see pitch.py.
DEFAULT_BASE_NOTE = 57


@dataclass
class Note:
    key: int = DEFAULT_KEY
    pos: int = 0
    length: int = 48
    volume: int = 100
    panning: int = 0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "pos": self.pos,
            "length": self.length,
            "volume": self.volume,
            "panning": self.panning,
            "note_name": key_to_note_name(self.key),
        }

    @staticmethod
    def from_dict(d: dict) -> Note:
        return Note(
            key=d.get("key", DEFAULT_KEY),
            pos=d.get("pos", 0),
            length=d.get("length", 48),
            volume=d.get("volume", 100),
            panning=d.get("panning", 0),
        )


@dataclass
class Pattern:
    name: str = "Pattern"
    pattern_type: int = 1
    steps: int = 16
    length: int = 192
    muted: bool = False
    notes: list[Note] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.pattern_type,
            "steps": self.steps,
            "length": self.length,
            "muted": self.muted,
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass
class Track:
    track_type: TrackType = TrackType.INSTRUMENT
    name: str = "Unnamed"
    muted: bool = False
    solo: bool = False
    volume: int = 100
    panning: int = 0
    mixer_channel: int = 0
    instrument: str = ""
    patterns: list[Pattern] = field(default_factory=list)
    index: int = 0
    # Tuning of an instrument track (see pitch.py): the key that sounds at
    # the instrument's natural pitch and the fine tune in cents.
    base_note: int = DEFAULT_BASE_NOTE
    pitch_cents: float = 0

    @property
    def type_name(self) -> str:
        return self.track_type.name

    @property
    def tuned(self) -> bool:
        """Whether the base note or the fine tune differs from the default."""
        return self.base_note != DEFAULT_BASE_NOTE or self.pitch_cents != 0

    def tuning_dict(self) -> dict:
        """base_note, base_note_name and pitch_cents; empty at the defaults."""
        if not self.tuned:
            return {}
        return {
            "base_note": self.base_note,
            "base_note_name": key_to_note_name(self.base_note),
            "pitch_cents": self.pitch_cents,
        }

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "type": int(self.track_type),
            "type_name": self.track_type.name,
            "name": self.name,
            "muted": self.muted,
            "solo": self.solo,
            "volume": self.volume,
            "panning": self.panning,
            "mixer_channel": self.mixer_channel,
            "instrument": self.instrument,
            **self.tuning_dict(),
            "patterns": [p.to_dict() for p in self.patterns],
        }


@dataclass
class MixerChannel:
    num: int = 0
    name: str = "Master"
    volume: float = 1.0
    muted: bool = False
    soloed: bool = False
    effects: list[str] = field(default_factory=list)
    sends: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num": self.num,
            "name": self.name,
            "volume": self.volume,
            "muted": self.muted,
            "soloed": self.soloed,
            "effects": self.effects,
            "sends": self.sends,
        }


@dataclass
class SongInfo:
    bpm: int = 140
    time_sig_numerator: int = 4
    time_sig_denominator: int = 4
    master_volume: int = 100
    master_pitch: int = 0
    tracks: list[Track] = field(default_factory=list)
    mixer_channels: list[MixerChannel] = field(default_factory=list)
    file_path: str = ""
    modified: bool = False

    @property
    def time_signature(self) -> str:
        return f"{self.time_sig_numerator}/{self.time_sig_denominator}"

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm,
            "time_signature": f"{self.time_sig_numerator}/{self.time_sig_denominator}",
            "master_volume": self.master_volume,
            "master_pitch": self.master_pitch,
            "file_path": self.file_path,
            "modified": self.modified,
            "track_count": len(self.tracks),
            "tracks": [t.to_dict() for t in self.tracks],
            "mixer_channels": [m.to_dict() for m in self.mixer_channels],
        }


# Note name helpers (LMMS keys, scientific pitch names: C4 = 48, A4 = 57)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def key_to_note_name(key: int) -> str:
    """Convert an LMMS key number to a note name like 'C4', 'A#3'.

    C0 = key 0, C4 (middle C) = 48, A4 = 57. Keys outside the valid range
    (e.g. read from an existing project) still get a name.
    """
    return f"{NOTE_NAMES[key % 12]}{key // 12}"


# Semitone offset of each natural note within an octave.
_NATURAL_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# letter (any case), optional accidental (# or b, single or double),
# octave. The first character is always the note letter, so "b3" is B3
# and "bb3" is Bb3 (B flat) - there is no ambiguity.
_NOTE_NAME_RE = re.compile(r"([A-Ga-g])(##|bb|#|b|)(-?[0-9]+)")

KEY_RANGE_TEXT = (
    f"{KEY_MIN}-{KEY_MAX}, i.e. {key_to_note_name(KEY_MIN)} to "
    f"{key_to_note_name(KEY_MAX)}"
)

KEY_NUMBERING_TEXT = (
    "Keys are LMMS keys: middle C (C4) = 48, A4 = 57 = 440 Hz. "
    "A MIDI note number is key + 12."
)

NOTE_NAME_FORMAT_HINT = (
    "Use a letter A-G, optional # or b, and an octave, "
    "e.g. C4, F#3, Bb2 (C4 = 48, A4 = 57 = 440 Hz)."
)


def note_name_to_key(name: str) -> int:
    """Convert a note name to an LMMS key number (C4 = 48, A4 = 57).

    Accepts sharps and flats ('A#3', 'Bb3', also the unicode sharp and
    flat signs), a case-insensitive note letter ('bb2' = Bb2,
    'b3' = B3) and double accidentals ('F##3', 'Bbb3'). Enharmonic
    spellings wrap across the octave boundary like on a piano:
    Cb4 = B3 = 47, B#3 = C4 = 48, E#4 = F4, Fb4 = E4.

    Raises ValueError with a message explaining the accepted format when
    the name is malformed or the result is outside the valid key range
    (0-108, C0 .. C9).
    """
    if not isinstance(name, str):
        raise ValueError(
            f"Invalid note name {name!r}: expected a string. {NOTE_NAME_FORMAT_HINT}"
        )

    cleaned = name.strip().replace("\u266f", "#").replace("\u266d", "b")
    m = _NOTE_NAME_RE.fullmatch(cleaned)
    if not m:
        hint = NOTE_NAME_FORMAT_HINT
        if re.fullmatch(r"-?[0-9]+", cleaned):
            hint += (
                " A bare number is a key, not a note name: pass it "
                "as an integer (e.g. 48), not a string."
            )
        raise ValueError(f"Invalid note name {name!r}. {hint}")

    letter, accidental, octave_str = m.groups()
    if accidental.startswith("#"):
        offset = len(accidental)
    elif accidental:
        offset = -len(accidental)
    else:
        offset = 0

    key = int(octave_str) * 12 + _NATURAL_SEMITONES[letter.upper()] + offset
    if not KEY_MIN <= key <= KEY_MAX:
        raise ValueError(
            f"Note {name!r} is out of range (it would be key {key}; "
            f"valid keys are {KEY_RANGE_TEXT})."
        )
    return key


def bars_to_ticks(bars: float, ticks_per_bar: int = TICKS_PER_BAR) -> int:
    """Convert bars to ticks.

    Accepts floats (e.g. 16.0) and always returns an int - LMMS rejects
    float strings like "3072.0" in integer attributes.
    """
    return int(round(bars * ticks_per_bar))


def ticks_to_bars(ticks: int, ticks_per_bar: int = TICKS_PER_BAR) -> float:
    """Convert ticks to bars."""
    return ticks / ticks_per_bar
