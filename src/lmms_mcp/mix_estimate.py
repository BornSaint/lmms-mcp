"""A rough peak of the mix from the levels of its instruments, before any render.

levels.py knows how loud each instrument sound is at track volume 100 (the
peak of one note and of a four-note chord). From those levels and the
project (notes, track volumes, velocities, mixer and Master volumes) this
module estimates, bar by bar, the peak the mix will have, so an agent can
set track volumes right the first time instead of rendering, measuring and
fixing the clipping.

Nothing here reads files or runs LMMS: the levels come from a function that
is given the <instrumenttrack> element and answers a levels.Level or None.
Bar numbers are 0-based, like those of analyze_mix.

Each bar's tracks are added two ways: as power (10 * log10 of the sum of
the squared amplitudes: peaks that never meet) and as amplitude (the peaks
add up at the same instant). The real peak lies in between, near the middle
when tracks play on the same beat, so the estimate is the midpoint in dB of
the two and the range is reported with it.

The estimate is not a render. It leaves out what it cannot know: effects,
automation, sample tracks and every track whose level is not known (they
are listed). A track gets its single-note peak in a bar where its notes
sound one at a time and its chord peak where two or more sound together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable
from xml.etree import ElementTree as ET

from . import xml_parser
from .levels import Level
from .models import TICKS_PER_BAR

# What the estimate says about itself (the "estimate_basis" of the response).
BASIS_TEXT = (
    "An estimate, not a render. For each bar the level measured for every "
    "instrument sound at track volume 100 (the peak of a single note, or of "
    "a 4-note chord where 2 or more notes sound together) is raised or "
    "lowered by the track volume, the loudest note's velocity, the "
    "track's mixer channel and the Master volumes. The tracks are added in "
    "two ways: as power (the peaks do not meet, the lower bound) and as "
    "amplitude (every peak at the same instant, the upper bound); "
    "estimate_range_dbfs is [power, amplitude] for the loudest bar and "
    "estimated_mix_peak_dbfs their midpoint in dB. Real mixes land between "
    "the two, nearer the midpoint, because tracks that play on the same "
    "beat (kick and hat) add almost as amplitude. Effects, automation, "
    "sample tracks and tracks without a known level (unknown_level_tracks) "
    "are not included. Bars are 0-based. render_project measures the real "
    "level."
)

# Tracks listed in "loudest_bar_tracks".
MAX_LOUDEST_TRACKS = 5

# Mixer channels followed through sends before giving up (a loop).
_MAX_ROUTE_DEPTH = 8

LevelOf = Callable[[ET.Element], Level | None]


@dataclass
class _Source:
    """One instrument that plays: a track, or an instrument inside a pattern
    track. `index` is the track index (None inside a pattern track)."""

    index: int | None
    name: str
    instrument: str
    inst_track: ET.Element
    notes: list[tuple[int, int, float]] = field(default_factory=list)


def _db(linear: float) -> float:
    return 20.0 * math.log10(linear)


def _rounded(db: float) -> float:
    return round(db, 1) + 0.0


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _plugin(inst_track: ET.Element) -> str:
    instrument = inst_track.find("instrument")
    return "" if instrument is None else instrument.get("name", "")


def ticks_per_bar(root: ET.Element) -> int:
    """Ticks of one bar at the project's time signature (192 in 4/4)."""
    head = xml_parser.find_head(root)
    if head is None:
        return TICKS_PER_BAR
    numerator = max(1, _int(head.get("timesig_numerator"), 4))
    denominator = max(1, _int(head.get("timesig_denominator"), 4))
    return max(1, TICKS_PER_BAR * numerator // denominator)


def _master_gain(root: ET.Element) -> float:
    """The song's Master volume (head mastervol, 100 = 1.0) as a factor."""
    head = xml_parser.find_head(root)
    if head is None:
        return 1.0
    return max(0.0, _number(
        xml_parser.model_value(head, "mastervol", "100"), 100.0) / 100.0)


class _Mixer:
    """The gain a mixer channel gives a track that plays on it: its own
    volume times the sends to the channels it feeds, up to Master (whose
    volume counts). A channel without a send is silent in LMMS."""

    def __init__(self, root: ET.Element) -> None:
        self._channels = {
            _int(c.get("num"), -1): c
            for c in xml_parser.find_mixer_channels(root)}
        self._master = xml_parser.master_volume(root)

    def gain(self, number: int, depth: int = 0) -> float:
        if number == 0 or depth > _MAX_ROUTE_DEPTH:
            return self._master if number == 0 else 0.0
        channel = self._channels.get(number)
        if channel is None:  # a missing channel: LMMS plays it elsewhere
            return self._master
        if xml_parser.model_value(channel, "muted", "0") == "1":
            return 0.0
        volume = _number(xml_parser.model_value(channel, "volume", "1"), 1.0)
        routed = sum(
            _number(xml_parser.model_value(send, "amount", "1"), 1.0)
            * self.gain(_int(send.get("channel"), 0), depth + 1)
            for send in channel.findall("send"))
        return volume * routed


def _pattern_notes(pattern: ET.Element) -> list[tuple[int, int, float]]:
    """(start, end, velocity) in ticks of the notes a pattern plays; notes
    beyond the end of the clip are not played by LMMS."""
    length = _int(pattern.get("len"), TICKS_PER_BAR)
    start = _int(pattern.get("pos"), 0)
    notes = []
    for note in pattern.findall("note"):
        pos = _int(note.get("pos"), 0)
        if not 0 <= pos < length:
            continue
        begin = start + pos
        notes.append((
            begin, begin + max(1, _int(note.get("len"), 48)),
            _number(note.get("vol"), 100.0)))
    return notes


def _track_notes(track: ET.Element) -> list[tuple[int, int, float]]:
    notes = []
    for pattern in track.findall("pattern"):
        if pattern.get("muted", "0") != "1":
            notes.extend(_pattern_notes(pattern))
    return notes


def _inner_tracks(track: ET.Element) -> list[ET.Element]:
    container = track.find("bbtrack/trackcontainer")
    return [] if container is None else container.findall("track")


def _pattern_track_notes(
    track: ET.Element, inner: ET.Element, bar: int,
) -> list[tuple[int, int, float]]:
    """The notes an instrument of a pattern track plays through the clips
    (bbtco) of the track: the pattern repeats every `bar`-rounded length of
    the longest instrument pattern, up to the end of each clip."""
    ends = [
        _int(p.get("pos"), 0) + _int(p.get("len"), bar)
        for t in _inner_tracks(track) for p in t.findall("pattern")]
    span = max(bar, -(-max(ends, default=bar) // bar) * bar)
    own = [
        note for p in inner.findall("pattern")
        if p.get("muted", "0") != "1" for note in _pattern_notes(p)]
    notes = []
    for clip in track.findall("bbtco"):
        if clip.get("muted", "0") == "1":
            continue
        begin = _int(clip.get("pos"), 0)
        end = begin + max(1, _int(clip.get("len"), span))
        for offset in range(begin, end, span):
            for start, stop, velocity in own:
                if offset + start < end:
                    notes.append(
                        (offset + start, offset + stop, velocity))
    return notes


def _sources(
    root: ET.Element, bar: int,
) -> tuple[list[_Source], list[dict]]:
    """The instruments that play (unmuted, with notes) and the tracks that
    play but whose level can never be known here (sample tracks)."""
    tracks = xml_parser.find_tracks(root)
    soloing = any(
        xml_parser.model_value(t, "solo", "0") == "1" for t in tracks)
    sources: list[_Source] = []
    other: list[dict] = []
    for index, track in enumerate(tracks):
        if xml_parser.model_value(track, "muted", "0") == "1":
            continue
        if soloing and xml_parser.model_value(track, "solo", "0") != "1":
            continue
        kind = xml_parser.get_track_type(track)
        name = xml_parser.get_track_name(track)
        if kind == 0:
            inst = track.find("instrumenttrack")
            notes = _track_notes(track)
            if inst is not None and notes:
                sources.append(_Source(
                    index, name, _plugin(inst), inst, notes))
        elif kind == 1:
            for inner in _inner_tracks(track):
                inst = inner.find("instrumenttrack")
                if inst is None or xml_parser.model_value(
                        inner, "muted", "0") == "1":
                    continue
                notes = _pattern_track_notes(track, inner, bar)
                if notes:
                    sources.append(_Source(
                        None, f"{name}/{xml_parser.get_track_name(inner)}",
                        _plugin(inst), inst, notes))
        elif kind == 2:
            if any(c.get("muted", "0") != "1"
                   for c in track.findall("sampleclip")):
                other.append({
                    "index": index, "name": name,
                    "instrument": "sample track"})
    return sources, other


def _max_overlap(spans, low: int, high: int) -> int:
    """The most notes sounding at the same moment within [low, high)."""
    events = []
    for start, end, _ in spans:
        events.append((max(start, low), 1))
        events.append((max(min(end, high), max(start, low)), -1))
    events.sort()
    now = best = 0
    for _, change in events:
        now += change
        best = max(best, now)
    return best


def _bars_of(
    notes: list[tuple[int, int, float]], bar: int,
) -> dict[int, list[tuple[int, int, float]]]:
    """The notes that sound in each bar (a note is in every bar it spans)."""
    found: dict[int, list] = {}
    for start, end, velocity in notes:
        for number in range(start // bar, (max(end, start + 1) - 1) // bar + 1):
            found.setdefault(number, []).append((start, end, velocity))
    return found


def estimate(root: ET.Element, level_of: LevelOf) -> dict:
    """The estimated peak of the mix, as the fields get_project_info adds:
    "estimated_mix_peak_dbfs" (the midpoint in dB of the tracks added as
    power and as amplitude) and "estimate_range_dbfs" ([power, amplitude]),
    both of the loudest bar by that midpoint, "loudest_bar" (0-based) with
    "loudest_bar_tracks", "unknown_level_tracks" (only when there are some)
    and "estimate_basis". Empty when no unmuted track has notes or clips.
    """
    bar = ticks_per_bar(root)
    sources, unknown = _sources(root, bar)
    mixer = _Mixer(root)
    master = _master_gain(root)
    # per bar: [(source, peak dBFS of the source in that bar)]
    per_bar: dict[int, list[tuple[_Source, float]]] = {}
    for source in sources:
        level = level_of(source.inst_track)
        if level is None:
            unknown.append({
                "index": source.index, "name": source.name,
                "instrument": source.instrument})
            continue
        inst = source.inst_track
        volume = _number(xml_parser.model_value(inst, "vol", "100"), 100.0)
        channel = mixer.gain(xml_parser.get_track_mixer_channel(inst))
        fixed = volume / 100.0 * channel * master
        for number, spans in _bars_of(source.notes, bar).items():
            loudest = max(velocity for _, _, velocity in spans)
            gain = fixed * loudest / 100.0
            if gain <= 0:
                continue
            chord = _max_overlap(
                spans, number * bar, (number + 1) * bar) >= 2
            base = level.chord_peak_dbfs if chord else level.note_peak_dbfs
            per_bar.setdefault(number, []).append(
                (source, base + _db(gain)))

    if not per_bar and not unknown:
        return {}
    result: dict = {
        "estimated_mix_peak_dbfs": None,
        "estimate_range_dbfs": None,
    }
    if per_bar:
        # per bar: (power sum dBFS, amplitude sum dBFS); the loudest bar is
        # the one with the highest midpoint (the earliest of equals)
        bounds = {
            n: (10.0 * math.log10(sum(10 ** (peak / 10) for _, peak in items)),
                20.0 * math.log10(sum(10 ** (peak / 20) for _, peak in items)))
            for n, items in per_bar.items()}
        loudest_bar = min(
            bounds, key=lambda n: (-(bounds[n][0] + bounds[n][1]), n))
        power, amplitude = bounds[loudest_bar]
        result["estimated_mix_peak_dbfs"] = _rounded((power + amplitude) / 2)
        result["estimate_range_dbfs"] = [
            _rounded(power), _rounded(amplitude)]
        result["loudest_bar"] = loudest_bar
        top = sorted(per_bar[loudest_bar], key=lambda item: -item[1])
        result["loudest_bar_tracks"] = [
            {"index": s.index, "name": s.name, "peak_dbfs": _rounded(peak)}
            for s, peak in top[:MAX_LOUDEST_TRACKS]]
    if unknown:
        result["unknown_level_tracks"] = unknown
    result["estimate_basis"] = BASIS_TEXT
    return result
