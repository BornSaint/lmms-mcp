"""How loud an instrument sound is at track volume 100, known before a render.

An agent that picks track volumes without knowing the level of each
instrument, preset or sample gets a mix that clips or is buried and has to
render, measure and fix it. This module measures a sound once (one note and
one chord on a one-track project, rendered with LMMS) and remembers the
result, so the tools that set a sound can report it at once:

    {"note_peak_dbfs", "chord_peak_dbfs", "rms_dbfs"}

- note_peak_dbfs: the peak of one note (key 57, velocity 100, one beat);
- chord_peak_dbfs: the peak of a four-note chord (keys 48, 52, 55, 60);
- rms_dbfs: the RMS (all channels) while that single note is held.

Each is the loudest of three at different places of the song (see
MEASURE_REPEATS), and the RMS is that of the three together.

All of them are for a track at volume 100, Master at 100, no effects (the
track's own effect chain is left out: it can be changed afterwards, and the
mixer is not modelled here either), panning centred.

Where a level comes from (the "source" of a report):

- "seed": SEED_LEVELS below, measured once for every instrument LMMS 1.2.2
  offers with its default settings, so a plain add_instrument_track never
  needs a render;
- "cache": measured before, on this machine or earlier in this session
  (the project's own table, then ${XDG_CACHE_HOME:-~/.cache}/lmms-mcp/
  levels.json);
- "measured": rendered just now (about 1 s), and then cached.

A sound is identified by what makes it sound: the instrument and its
settings (a preset's data, the sample or soundfont with its size and
modification time, bank, patch), the envelope / filter, chord and
arpeggiator settings, and the LMMS version. Volume, panning, mixer channel,
tuning, name and the effect chain are not part of it. Set
LMMS_MCP_MEASURE_LEVELS=0 to forbid the renders (seed and cache are still
used).

Nothing here is written into the project file: LMMS must not see
attributes it does not know.

The track volume of a new track follows from its level (auto_volume): a
track the agent gives no volume gets the volume that puts its sound at a
fixed loudness, so a mix of 6 to 8 tracks lands near 0 dBFS before any
volume is touched.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

from . import audio_analysis, lmms_app, xml_parser
from .models import TICKS_PER_BAR

# Set to 0 (also "false", "no", "off") to forbid renders that measure a sound.
MEASURE_ENV = "LMMS_MCP_MEASURE_LEVELS"

# The measuring project: 60 bpm, so a beat is a second and a bar 4 s. The
# single note plays in bars 0, 2 and 4 and the chord in bars 6, 8 and 10; the
# bar between two of them is empty, so the release of a note (up to 7 s)
# never reaches the next one. Three of each, and the peak taken over the
# three, because the level of some sounds depends on where in the song they
# play: a preset with a tempo-synced LFO or arpeggio changes by up to 3 dB
# between bars, and ZynAddSubFX starts its notes at random phases (a chord
# varies by 2 dB between runs). The loudest is what the headroom has to
# cover. MEASURE_REPEATS = 1 measures one of each.
MEASURE_BPM = 60
NOTE_KEY = 57
CHORD_KEYS = (48, 52, 55, 60)
NOTE_VELOCITY = 100
NOTE_TICKS = TICKS_PER_BAR // 4       # one beat
MEASURE_REPEATS = 3
NOTE_BARS = tuple(2 * i for i in range(MEASURE_REPEATS))
CHORD_BARS = tuple(2 * (MEASURE_REPEATS + i) for i in range(MEASURE_REPEATS))
RENDER_SAMPLERATE = 44100
RENDER_TIMEOUT = 60                   # seconds
# A peak above this (dBFS) is a broken sound (garbage or a runaway filter),
# not a loud one.
IMPLAUSIBLE_DBFS = 60.0

# Most levels the cache file keeps (the oldest go first).
CACHE_LIMIT = 2000
CACHE_FORMAT = 1

SOURCE_SEED = "seed"
SOURCE_CACHE = "cache"
SOURCE_MEASURED = "measured"


class LevelError(Exception):
    """A level could not be measured (no LMMS, a timeout, a silent
    render, ...). The message is short enough for a response note."""


@dataclass(frozen=True)
class Level:
    """The level of a sound at track volume 100 (dBFS, rounded to 0.1)."""

    note_peak_dbfs: float
    chord_peak_dbfs: float
    rms_dbfs: float

    def to_dict(self, source: str | None = None) -> dict:
        out = {
            "note_peak_dbfs": self.note_peak_dbfs,
            "chord_peak_dbfs": self.chord_peak_dbfs,
            "rms_dbfs": self.rms_dbfs,
        }
        if source is not None:
            out["source"] = source
        return out

    @staticmethod
    def from_dict(data) -> Level | None:
        """The level in a dict (None when it is not a valid one)."""
        try:
            values = [float(data[k]) for k in (
                "note_peak_dbfs", "chord_peak_dbfs", "rms_dbfs")]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in values):
            return None
        return Level(*values)


def _round(db: float) -> float:
    return round(db, 1) + 0.0


# ──────────────────────────────────────────────────────────────────
# THE AUTOMATIC TRACK VOLUME
# ──────────────────────────────────────────────────────────────────

# A track whose volume the agent did not give gets the volume that puts its
# sound at AUTO_RMS_TARGET_DBFS (the rms_dbfs of its level: the RMS while one
# note is held, the quantity an ear and analyze_mix compare), but not so loud
# that one note peaks above AUTO_PEAK_LIMIT_DBFS (a drum sample is -23 to
# -29 dB RMS at a 0 dBFS peak, so the RMS alone would put its peak at or
# above 0 dBFS). The two numbers were measured, not guessed: 7 test songs of
# 3 to 10 tracks (13 drum samples and 16 synth, soundfont and ZynAddSubFX
# sounds: kick, snare, hat, clap, bass, keys, lead, pad; 16 bars at 120 bpm)
# were rendered one track at a time at volume 100 and mixed again with each
# rule (the mix is a plain sum of the tracks times their volumes, so this is
# exact), and the mix peak and the analyze_mix flags (dominant, buried) were
# read each time:
#
#   RMS target, peak limit   mix peak, 3-8 tracks      10 tracks   flags (7
#   (dBFS)                   (6 songs, dBFS)           (dBFS)      songs)
#   -20, none                -2.1 .. +12.4             +15.2       4
#   -24, none                -6.1 .. +9.6              +14.6       9
#   -28, -5                  -10.1 .. +3.5             +5.9        2
#   -30, -5  (chosen)        -11.9 .. +3.2             +4.8        1
#   -32, -5                  -12.6 .. +2.1             +3.9        4
#   -30, none                -11.9 .. +3.9             +8.6        6
#   -30, -6                  -11.9 .. +2.5             +4.3        1
#   -30, -8                  -11.9 .. +1.0             +3.3        4
#
# Why -30 and not the -20 dBFS a track has in a finished mix: the RMS of a
# note is far from the RMS of a track that plays it (eighth notes and 3-note
# chords are 3 to 8 dB louder than the one note), and the peaks of N tracks
# that play together add up (+11.7 dB for 6 equal ones, the midpoint of the
# power and amplitude sums), so -20 puts 6 tracks 10 dB over full scale.
# -30 puts 6 to 8 tracks at -1 .. +3.2 dBFS and -16 to -18 dBFS RMS, a normal
# level before mastering; render_project lowers what is left. The peak limit
# keeps the drum samples from setting the mix peak (a hat is -29 dB RMS at a
# 0 dBFS peak, so the RMS target alone leaves it at -1.3 dBFS and the kick,
# snare, hat and clap that play together add up), but -8 buries snares and
# claps (more than 15 dB below the mix). -6 and -5 have the same flags
# offline and -6 peaks 0.2 to 0.7 dB lower; -5 is used because in the real
# end-to-end song with the quietest snare sample (-31 dB RMS) -6 left the
# snare 15.0 dB below the mix (flagged), and -5 puts it 1 dB louder.
AUTO_RMS_TARGET_DBFS = -30.0
AUTO_PEAK_LIMIT_DBFS = -5.0
# The automatic volume stays in this range (set_track_volume still takes
# 0-200): above 200 the mixer has no more gain, and below 10 a sound is
# 20 dB under what it sounds at 100, which is a track the agent should
# decide about (the response says so).
AUTO_VOLUME_MIN = 10
AUTO_VOLUME_MAX = 200
# What a track has when its level is not known (the plain default).
DEFAULT_VOLUME = 100


def _volume_db(volume: int) -> float:
    return 20.0 * math.log10(volume / 100.0)


@dataclass(frozen=True)
class AutoVolume:
    """The volume chosen for a sound, and why (`text` is for the response).
    `limit` is "rms" or "peak" (what set the volume), `clamped` is "min" or
    "max" when the range of AUTO_VOLUME_* cut it, and `off_db` is how far the
    sound lands from what the rule wanted then (positive: below it, at the
    maximum; negative: above it, at the minimum; 0 when not clamped)."""

    volume: int
    limit: str
    clamped: str | None
    off_db: float
    text: str


def auto_volume(level: Level) -> AutoVolume:
    """The track volume (0-200) for a sound of this level: the sound at
    AUTO_RMS_TARGET_DBFS RMS, one note peaking at most at
    AUTO_PEAK_LIMIT_DBFS, within AUTO_VOLUME_MIN..AUTO_VOLUME_MAX. See the
    comment above for how the numbers were chosen."""
    by_rms = AUTO_RMS_TARGET_DBFS - level.rms_dbfs
    by_peak = AUTO_PEAK_LIMIT_DBFS - level.note_peak_dbfs
    limit = "rms" if by_rms <= by_peak else "peak"
    wanted = 100.0 * 10 ** (min(by_rms, by_peak) / 20.0)
    volume = int(round(wanted))
    clamped = None
    off_db = 0.0
    if volume > AUTO_VOLUME_MAX:
        volume, clamped = AUTO_VOLUME_MAX, "max"
    elif volume < AUTO_VOLUME_MIN:
        volume, clamped = AUTO_VOLUME_MIN, "min"
    gain = _volume_db(volume)
    rms_at = level.rms_dbfs + gain
    peak_at = level.note_peak_dbfs + gain
    if clamped:
        # distance from the limit that wanted more (max) or less (min)
        off_db = _round((AUTO_RMS_TARGET_DBFS - rms_at) if limit == "rms"
                        else (AUTO_PEAK_LIMIT_DBFS - peak_at))
    if limit == "rms":
        rule = (f"RMS target {AUTO_RMS_TARGET_DBFS:g} dBFS (level_at_vol_100 "
                f"rms {level.rms_dbfs:g} dB")
    else:
        rule = (f"note peak limit {AUTO_PEAK_LIMIT_DBFS:g} dBFS "
                f"(level_at_vol_100 note_peak {level.note_peak_dbfs:g} dB")
    text = f"auto: {rule} -> {volume})"
    if clamped == "max":
        text += (
            f". Volume {AUTO_VOLUME_MAX} (+6 dB) is the most the mixer "
            f"gives and this sound stays {abs(off_db):.0f} dB below that "
            f"target (rms {_round(rms_at):g} dBFS), so it will sit low in "
            f"the mix; raise the note velocities (up to 200) or lower the "
            f"other tracks.")
    elif clamped == "min":
        text += (
            f". Volume {AUTO_VOLUME_MIN} is the lowest the automatic "
            f"choice sets and this sound is still {abs(off_db):.0f} dB "
            f"above that target; set_track_volume goes lower.")
    return AutoVolume(volume, limit, clamped, off_db, text)


# ──────────────────────────────────────────────────────────────────
# SEED: measured with LMMS 1.2.2, every offered instrument at its defaults
# ──────────────────────────────────────────────────────────────────

# (note_peak_dbfs, chord_peak_dbfs, rms_dbfs) of the track add_instrument_track
# makes for each instrument, at volume 100, measured by measure_level on
# LMMS 1.2.2 (Linux, 44.1 kHz float render at -x 1 with a private config).
# Each is the median of five runs of `python -m lmms_mcp.levels`, which adds
# a new track of every offered instrument (default settings: no preset, no
# sample), renders key 57 at velocity 100 for a beat and then the chord
# 48/52/55/60, each three times (see the module docstring and
# MEASURE_REPEATS), about 1 s an instrument. Most instruments repeat to the
# 0.1 dB; organic varies by up to 1.3 dB and vibedstrings by 0.4 dB between
# runs (their sound starts at random phases). Peaks above 0 dBFS are real:
# the render is 32-bit float, and LMMS's own 16-bit file would clip them.
# Keyed by the offered instrument name (OPL2 is opulenz, papu is freeboy).
#
# Not listed:
# - audiofileprocessor, gigplayer, patman: silent until they have a file
#   (their level comes with the sample / file);
# - bitinvader, watsyn: a track without settings, which is what
#   add_instrument_track writes for them, renders garbage in LMMS 1.2.2 (their
#   wave data is missing, so the level changes from run to run, from -60 to
#   +530 dBFS), so there is no level to seed; see UNSET_IS_BROKEN.
SEED_LEVELS: dict[str, tuple[float, float, float]] = {
    "freeboy": (-5.8, 4.1, -18.5),
    "kicker": (0.0, 12.0, -9.4),
    "lb302": (-6.5, -6.5, -15.0),
    "malletsstk": (0.5, 12.6, -29.6),
    "monstro": (-0.1, 11.6, -2.2),
    "nes": (-3.0, 9.1, -10.0),
    "opulenz": (-23.3, -14.2, -29.4),
    "organic": (-1.4, 6.5, -12.1),
    "sfxr": (-13.9, -1.9, -18.9),
    "sid": (-6.6, 4.9, -11.1),
    "tripleoscillator": (-2.7, 8.8, -7.9),
    "vibedstrings": (-10.8, -0.1, -16.9),
    "zynaddsubfx": (-28.4, -17.3, -32.0),
}

# The default soundfont of a new sf2player track (bank 0, patch 0 = piano),
# measured the same way (median of five runs, identical): used only for a
# file with this name and size.
SEED_SOUNDFONT: dict = {
    "file": "default-GM.sf2", "size": 5969788, "bank": 0, "patch": 0,
    "level": (-26.8, -19.2, -39.9),
}

# The LMMS series the seed was measured with. Another version measures.
SEED_SERIES = "1.2"


def seed_applies(version: str | None) -> bool:
    """Whether the seed is valid for this LMMS version (an unknown version,
    no LMMS installed, counts: nothing can be rendered then anyway)."""
    if not version:
        return True
    return re.match(r"\s*1\.2(\.|$)", version) is not None


# ──────────────────────────────────────────────────────────────────
# WHAT MAKES A SOUND
# ──────────────────────────────────────────────────────────────────

# The parts of an <instrumenttrack> that, besides <instrument>, decide how
# loud a note is: the envelopes and filter, the chord stack, the arpeggio.
SOUND_PARTS = ("eldata", "chordcreator", "arpeggiator")


def _canon(element: ET.Element) -> list:
    """A comparable, JSON-able copy of an element (tag, attributes, text,
    children). Model ids are left out: they change with every session."""
    return [
        element.tag,
        sorted((k, v) for k, v in element.attrib.items() if k != "id"),
        (element.text or "").strip(),
        [_canon(child) for child in element],
    ]


def plugin_of(inst_track: ET.Element) -> str:
    """The plugin name LMMS loads for an <instrumenttrack> ("" if none)."""
    instrument = inst_track.find("instrument")
    return "" if instrument is None else instrument.get("name", "")


def _settings(inst_track: ET.Element) -> ET.Element | None:
    """The instrument's own settings element (LMMS reads the first child)."""
    instrument = inst_track.find("instrument")
    return instrument[0] if instrument is not None and len(instrument) else None


def _file_fingerprints(instrument: ET.Element) -> list:
    """[src, size, mtime_ns] of every file the instrument plays: a changed
    sample or soundfont is another sound. A relative path is looked for the
    way LMMS does (lmms_app.sample_search_dirs); a missing file has None."""
    found = []
    folders = None
    for element in instrument.iter():
        src = (element.get("src") or "").strip()
        if not src:
            continue
        path = Path(src).expanduser()
        if path.is_absolute():
            candidates = [path]
        else:
            if folders is None:
                folders = lmms_app.sample_search_dirs()
            candidates = [d / src for d in folders]
        info = None
        for candidate in candidates:
            try:
                info = candidate.stat()
                break
            except OSError:
                continue
        found.append([
            src, None if info is None else info.st_size,
            None if info is None else info.st_mtime_ns])
    return found


def sound_digest(inst_track: ET.Element) -> str:
    """A short hash of everything that decides the level of the sound of an
    <instrumenttrack> (see the module docstring)."""
    instrument = inst_track.find("instrument")
    payload = [
        None if instrument is None else _canon(instrument),
        [_canon(p) for tag in SOUND_PARTS
         if (p := inst_track.find(tag)) is not None],
        [] if instrument is None else _file_fingerprints(instrument),
    ]
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def level_key(inst_track: ET.Element, version: str | None) -> str:
    """The key of a sound's level in the cache and the project's table:
    "<LMMS version>|<plugin>|<digest>"."""
    return f"{version or 'unknown'}|{plugin_of(inst_track)}|" \
           f"{sound_digest(inst_track)}"


# ──────────────────────────────────────────────────────────────────
# SEED LOOKUP
# ──────────────────────────────────────────────────────────────────


def _seed_name(plugin: str) -> str | None:
    """The offered instrument name (a key of SEED_LEVELS or "sf2player")
    that a plugin name written in a project stands for."""
    low = plugin.lower()
    for name in (*SEED_LEVELS, "sf2player"):
        if low in lmms_app.plugin_candidates(name):
            return name
    return None


@functools.lru_cache(maxsize=None)
def _fresh_canon(plugin: str) -> tuple:
    """(instrument, sound parts) of a track add_instrument_track makes for
    `plugin`, as _canon gives them."""
    root = xml_parser.create_empty_project()
    track = xml_parser.add_instrument_track(root, "fresh", plugin)
    inst_track = track.find("instrumenttrack")
    return (
        json.dumps(_canon(inst_track.find("instrument")), sort_keys=True),
        json.dumps([_canon(inst_track.find(t)) for t in SOUND_PARTS],
                   sort_keys=True),
    )


def _same_parts(inst_track: ET.Element, plugin: str) -> bool:
    """Whether the sound parts of the track are those of a new track."""
    parts = [inst_track.find(t) for t in SOUND_PARTS]
    if any(p is None for p in parts):
        return False
    return json.dumps([_canon(p) for p in parts], sort_keys=True) \
        == _fresh_canon(plugin)[1]


def seed_level(inst_track: ET.Element, version: str | None) -> Level | None:
    """The seeded level of a track that has its instrument's default
    settings (or the default soundfont at bank 0 patch 0), else None."""
    if not seed_applies(version):
        return None
    plugin = plugin_of(inst_track)
    name = _seed_name(plugin)
    instrument = inst_track.find("instrument")
    if name is None or instrument is None or not _same_parts(
            inst_track, plugin):
        return None
    if name == "sf2player":
        return _soundfont_seed(inst_track)
    if name not in SEED_LEVELS:
        return None
    if json.dumps(_canon(instrument), sort_keys=True) != \
            _fresh_canon(plugin)[0]:
        return None
    return Level(*SEED_LEVELS[name])


def _soundfont_seed(inst_track: ET.Element) -> Level | None:
    spec = SEED_SOUNDFONT
    settings = _settings(inst_track)
    if not spec or settings is None or len(settings):
        return None
    attributes = dict(settings.attrib)
    src = attributes.pop("src", "")
    if attributes != {"bank": str(spec["bank"]), "patch": str(spec["patch"]),
                      "gain": "1"} or Path(src).name != spec["file"]:
        return None
    try:
        size = Path(src).stat().st_size
    except OSError:
        return None
    return Level(*spec["level"]) if size == spec["size"] else None


# ──────────────────────────────────────────────────────────────────
# THE CACHE FILE
# ──────────────────────────────────────────────────────────────────


def cache_path() -> Path:
    """${XDG_CACHE_HOME:-~/.cache}/lmms-mcp/levels.json (read at each call)."""
    base = os.environ.get("XDG_CACHE_HOME", "")
    folder = Path(base) if base and Path(base).is_absolute() \
        else Path.home() / ".cache"
    return folder / "lmms-mcp" / "levels.json"


class LevelCache:
    """The levels measured so far, in a JSON file next to the user's other
    caches. A missing, unreadable or corrupt file is an empty cache (it is
    rebuilt by the next write); a write that fails is ignored; a write
    replaces the file atomically."""

    def __init__(self, path: Path | None = None) -> None:
        self._fixed = path
        self._lock = threading.Lock()
        self._memo: tuple | None = None   # (path, mtime_ns, size, entries)

    @property
    def path(self) -> Path:
        return self._fixed if self._fixed is not None else cache_path()

    def _entries(self) -> dict[str, dict]:
        path = self.path
        try:
            info = path.stat()
        except OSError:
            return {}
        stamp = (path, info.st_mtime_ns, info.st_size)
        if self._memo is not None and self._memo[:3] == stamp:
            return self._memo[3]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data["entries"]
            entries = {
                key: value for key, value in raw.items()
                if isinstance(key, str) and Level.from_dict(value)}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            entries = {}
        self._memo = (*stamp, entries)
        return entries

    def get(self, key: str) -> Level | None:
        with self._lock:
            entry = self._entries().get(key)
        return None if entry is None else Level.from_dict(entry)

    def put(self, key: str, level: Level, plugin: str = "") -> bool:
        """Remember a level. False when the file could not be written."""
        entry = level.to_dict()
        if plugin:
            entry["plugin"] = plugin
        with self._lock:
            entries = dict(self._entries())
            entries.pop(key, None)
            entries[key] = entry
            while len(entries) > CACHE_LIMIT:
                del entries[next(iter(entries))]
            return self._write(entries)

    def _write(self, entries: dict) -> bool:
        path = self.path
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                prefix="levels-", suffix=".tmp", dir=path.parent)
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump({"format": CACHE_FORMAT, "entries": entries}, out,
                          separators=(",", ":"))
            os.replace(temporary, path)
            return True
        except OSError:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
            return False


CACHE = LevelCache()


# ──────────────────────────────────────────────────────────────────
# MEASURING
# ──────────────────────────────────────────────────────────────────

# renders the saved project (first path) as a 32-bit float wav (second
# path), at -x 1 with a private config: raises LevelError when it fails.
Render = Callable[[Path, Path], None]


def measure_enabled() -> bool:
    """False when LMMS_MCP_MEASURE_LEVELS says no (0, false, no, off)."""
    value = os.environ.get(MEASURE_ENV, "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def build_measure_project(
    inst_track: ET.Element, version: str | None = None,
) -> ET.Element:
    """The one-track project measure_level renders: a copy of the track at
    volume 100, centred, on Master, without effects, with the single note in
    bar 0 and the chord in bar 2."""
    root = xml_parser.create_empty_project(
        bpm=MEASURE_BPM, lmms_version=version)
    container = root.find("song/trackcontainer[@type='song']")
    track = ET.SubElement(container, "track", {
        "muted": "0", "type": "0", "name": "level", "solo": "0"})
    inst = copy.deepcopy(inst_track)
    for name in ("vol", "pan"):
        for child in inst.findall(name):
            inst.remove(child)
    inst.set("vol", "100")
    inst.set("pan", "0")
    for names in xml_parser.ALL_MIXER_NAMES:
        inst.attrib.pop(names.track_attr, None)
        for child in inst.findall(names.track_attr):
            inst.remove(child)
    inst.set(xml_parser.mixer_names(root).track_attr, "0")
    for chain in inst.findall("fxchain"):
        inst.remove(chain)
    ET.SubElement(inst, "fxchain", {"numofeffects": "0", "enabled": "0"})
    track.append(inst)
    for bar in NOTE_BARS:
        xml_parser.add_note_to_track(
            root, 0, NOTE_KEY, bar * TICKS_PER_BAR, NOTE_TICKS,
            NOTE_VELOCITY)
    for bar in CHORD_BARS:
        for key in CHORD_KEYS:
            xml_parser.add_note_to_track(
                root, 0, key, bar * TICKS_PER_BAR, NOTE_TICKS,
                NOTE_VELOCITY)
    return root


def measure_level(
    inst_track: ET.Element, render: Render, version: str | None = None,
) -> Level:
    """Render a single note and a chord of the sound of an
    <instrumenttrack> (see the module docstring) and measure them.

    `render` does the LMMS run (see Render). Raises LevelError when it
    fails or the sound is silent.
    """
    root = build_measure_project(inst_track, version)
    with tempfile.TemporaryDirectory(
            prefix="lmms-mcp-level-", ignore_cleanup_errors=True) as work:
        project = Path(work) / "level.mmp"
        output = Path(work) / "level.wav"
        try:
            xml_parser.save_project(project, root, compressed=False)
        except OSError as exc:
            raise LevelError(f"could not write the measuring project: {exc}")
        render(project, output)
        try:
            audio = audio_analysis.read_wav(output)
        except (OSError, audio_analysis.AudioError) as exc:
            raise LevelError(f"could not read the measuring render: {exc}")
    bar = audio_analysis.bar_seconds(MEASURE_BPM)
    bars = audio_analysis.bar_stats(audio, bar, CHORD_BARS[-1] + 1)
    note = max(bars[n].peak for n in NOTE_BARS)
    chord = max(bars[n].peak for n in CHORD_BARS)
    if min(note, chord) < audio_analysis.SILENCE_PEAK:
        raise LevelError("the sound rendered silent")
    if max(note, chord) > audio_analysis.from_db(IMPLAUSIBLE_DBFS):
        raise LevelError(
            "the sound renders at an impossible level (above "
            f"{IMPLAUSIBLE_DBFS:+.0f} dBFS), so it is probably broken")
    # the RMS while a note is held (a beat), over all the notes
    held = [
        audio_analysis.window_stats(audio, n * bar, n * bar + 60.0 / MEASURE_BPM)
        for n in NOTE_BARS]
    count = sum(w.count for w in held)
    rms = math.sqrt(sum(w.squares for w in held) / count) if count else 0.0
    return Level(
        _round(audio_analysis.to_db(note)),
        _round(audio_analysis.to_db(chord)),
        _round(audio_analysis.to_db(rms) or -120.0))


# Plugins that play garbage (a different level at every run) when their
# settings element is empty, which is how add_instrument_track writes them:
# their wave data is missing. A level measured from such a track means
# nothing.
UNSET_IS_BROKEN = ("bitinvader", "watsyn")


def _plugin_is_unset(plugin: str, settings: ET.Element | None) -> bool:
    return plugin.lower() in UNSET_IS_BROKEN and (
        settings is None or not (settings.attrib or len(settings)))


def _can_sound(inst_track: ET.Element) -> bool:
    """False when there is nothing to measure: a sampler without its file
    (silent until set_audiofile_sample / set_instrument_file gives one) or
    an instrument that has no settings and plays garbage then."""
    plugin = plugin_of(inst_track)
    settings = _settings(inst_track)
    if _plugin_is_unset(plugin, settings):
        return False
    needs_file = plugin.lower() == "audiofileprocessor" \
        or lmms_app.file_instrument_for(plugin) is not None
    return not needs_file or (
        settings is not None and bool((settings.get("src") or "").strip()))


# ──────────────────────────────────────────────────────────────────
# LOOKUP
# ──────────────────────────────────────────────────────────────────


def find_level(
    inst_track: ET.Element, version: str | None, table: dict | None = None,
    cache: LevelCache | None = None,
) -> tuple[Level, str] | None:
    """The known level of a track's sound and its source ("seed" or
    "cache"), without rendering; None when it is not known. `table` is the
    project's own {key: Level} (see LMMSProject.level_table)."""
    seeded = seed_level(inst_track, version)
    if seeded is not None:
        return seeded, SOURCE_SEED
    key = level_key(inst_track, version)
    if table is not None and key in table:
        return table[key], SOURCE_CACHE
    cached = (cache or CACHE).get(key)
    if cached is not None:
        if table is not None:
            table[key] = cached
        return cached, SOURCE_CACHE
    return None


def resolve_level(
    inst_track: ET.Element, version: str | None, table: dict | None = None,
    render: Render | None = None, cache: LevelCache | None = None,
) -> tuple[Level, str] | None:
    """find_level, and when the level is not known, measure it (and cache
    it and put it in `table`): (level, "seed" | "cache" | "measured").

    None when nothing is known and nothing can be measured: no `render`,
    measuring forbidden (LMMS_MCP_MEASURE_LEVELS=0) or a sampler without a
    file. Raises LevelError when the measuring fails.
    """
    found = find_level(inst_track, version, table, cache)
    if found is not None:
        return found
    if render is None or not measure_enabled() or not _can_sound(inst_track):
        return None
    level = measure_level(inst_track, render, version)
    key = level_key(inst_track, version)
    if table is not None:
        table[key] = level
    (cache or CACHE).put(key, level, plugin_of(inst_track))
    return level, SOURCE_MEASURED


# ──────────────────────────────────────────────────────────────────
# REGENERATING THE SEED
# ──────────────────────────────────────────────────────────────────


def print_seed() -> None:
    """Print SEED_LEVELS and SEED_SOUNDFONT, measured on the installed LMMS
    with a new track of every offered instrument (what the seed is: see
    SEED_LEVELS). A maintainer runs this when the defaults or LMMS change:

        HOME=<scratch folder> python -m lmms_mcp.levels

    It renders every instrument (about half a second each) with a private
    config; use another HOME anyway, LMMS may create folders in it.
    """
    import json

    from . import server
    from .project import LMMSProject

    availability = server._availability
    proj = LMMSProject()
    proj.new(lmms_version=availability.version)
    server.set_project(proj)
    table: dict[str, Level] = {}
    soundfont = None
    for instrument in sorted(availability.instruments):
        before = os.environ.get(MEASURE_ENV)
        os.environ[MEASURE_ENV] = "0"   # adding a track must not measure
        try:
            result = json.loads(server.add_instrument_track(
                instrument, instrument=instrument))
        finally:
            if before is None:
                os.environ.pop(MEASURE_ENV, None)
            else:
                os.environ[MEASURE_ENV] = before
        inst_track = proj._instrument_track_element(
            result.get("track_index", -1))
        if inst_track is None or not _can_sound(inst_track):
            print(f"# {instrument}: nothing to measure (silent without a "
                  f"file, or no settings), no seed")
            continue
        try:
            level = measure_level(
                inst_track, server._render_for_levels, availability.version)
        except LevelError as exc:
            print(f"# {instrument}: {exc}")
            continue
        if instrument == "sf2player":
            settings = _settings(inst_track)
            path = Path(settings.get("src"))
            soundfont = {
                "file": path.name, "size": path.stat().st_size,
                "bank": int(settings.get("bank")),
                "patch": int(settings.get("patch")),
                "level": astuple(level)}
        else:
            table[instrument] = level
    print(f"# LMMS {availability.version}")
    print("SEED_LEVELS = {")
    for name, level in table.items():
        print(f'    "{name}": {astuple(level)},')
    print("}")
    print(f"SEED_SOUNDFONT = {soundfont!r}")


def main() -> None:
    # `python -m` runs this file as __main__, a second copy of the module:
    # the server raises the package's LevelError, so use the package's.
    from . import levels
    levels.print_seed()


if __name__ == "__main__":
    main()
