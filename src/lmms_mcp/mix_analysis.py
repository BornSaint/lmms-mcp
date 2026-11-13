"""The balance of a mix track by track, from the track renders of a project.

analyze_mix renders every track on its own (`lmms rendertracks`) and the
whole mix, measures each of them per bar (audio_analysis.bar_stats) and hands
the bar statistics to this module, which answers what an agent would
otherwise work out with a script: how loud each track is in each section of
the song, how much of the mix it is, and which tracks are too loud, too quiet
or add to a peak, with the track volume that would fix it.

Nothing here reads files or runs LMMS: the input is lists of
audio_analysis.BarStats, so the whole logic is tested with synthetic bars.
All bar numbers in this module are 0-based and the end of a range is
exclusive: [12, 20] is bars 12 to 19.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from . import audio_analysis as aa
from .audio_analysis import BarStats

# ──────────────────────────────────────────────────────────────────
# THRESHOLDS (dB are relative levels, dBFS absolute ones)
# ──────────────────────────────────────────────────────────────────

# A track is active in a bar when its RMS there is at least this: -60 dBFS
# (the same silence limit render_project uses for its quietest block).
ACTIVE_BAR_RMS = aa.SILENT_BLOCK_RMS

# A track plays in a section when it is active in at least this fraction of
# its bars (and in one at least). Fills and release tails that spill into a
# section are listed but not judged there.
PLAYING_FRACTION = 0.5

# "dominant": a track whose RMS (over the bars where it plays in a section)
# is more than this many dB above the average RMS of the other tracks
# playing there (needing at least DOMINANT_MIN_OTHERS of them). The suggested
# fix brings it to DOMINANT_TARGET_DB above them.
DOMINANT_DB = 6.0
DOMINANT_TARGET_DB = 3.0
DOMINANT_MIN_OTHERS = 2

# "buried": a track whose RMS (over the bars where it plays in a section) is
# more than this many dB below the RMS of the mix over the same bars. The
# suggested fix brings it to BURIED_TARGET_DB below the mix.
BURIED_DB = 15.0
BURIED_TARGET_DB = 12.0

# "peak_contributors": a section whose mix peak is at or above PEAK_NEAR_DBFS
# gets the tracks whose own peak, within PEAK_WINDOW_S of the mix's peak, is
# not more than CONTRIBUTOR_DB below it (the MAX_CONTRIBUTORS loudest). The
# suggested fix lowers one of them so the mix peak lands at PEAK_TARGET_DBFS
# (a track is lowered by at most 1 / MIN_FACTOR, 20 dB).
# PEAK_NEAR_DBFS is above the -1 dBFS peak render_project's auto_fix_clipping
# leaves (server._RENDER_TARGET_PEAK_DB, +-0.1 dB of measuring): a mix that was
# just fixed is correct and must not be flagged, so the flag is for a mix that
# clips or is within 0.5 dB of full scale. PEAK_TARGET_DBFS is below that fix
# target, so a mix lowered as suggested is not flagged either.
PEAK_NEAR_DBFS = -0.5
PEAK_TARGET_DBFS = -2.0
PEAK_WINDOW_S = 0.025
CONTRIBUTOR_DB = 12.0
MAX_CONTRIBUTORS = 4
MIN_FACTOR = 0.1

# Track volume (set_track_volume) is linear percent, 0-200, 100 = unity.
VOLUME_MAX = 200
# LMMS's mixer faders stop at 2.0 (+6 dB).
MASTER_VOLUME_MAX = 2.0

# Limits that keep a response compact.
MAX_SECTIONS = 64
DEFAULT_MAX_SECTIONS = 16        # sections of the default blocks of bars
MAX_FLAGS_PER_KIND = 12
MAX_RANGES = 8                   # active-bar ranges listed for a track


class SectionError(ValueError):
    """The `sections` argument is not valid; the message says why."""


@dataclass
class Section:
    name: str
    start: int   # first bar (0-based)
    end: int     # one past the last bar


@dataclass
class TrackStem:
    """One rendered track: what it is and its bar statistics.

    `index` is the track's index in the project (what set_track_volume
    takes), None for an instrument inside a pattern (BB) track. `volume` is
    its static volume, 0-200 (None when unknown). `probe_peaks` has the
    track's peak near each PeakProbe, in the order of the probes.
    """

    name: str
    bars: list[BarStats]
    index: int | None = None
    instrument: str = ""
    volume: float | None = None
    volume_automated: bool = False
    probe_peaks: list[float] = field(default_factory=list)


@dataclass
class PeakProbe:
    """Where the mix peaks in a section that is near full scale."""

    section: int      # index into the sections list
    frame: int
    seconds: float
    peak: float       # linear, of the mix


# ──────────────────────────────────────────────────────────────────
# SECTIONS
# ──────────────────────────────────────────────────────────────────

_SECTIONS_FORMAT = (
    '[{"name": "verse", "bars": [4, 12]}, {"name": "hook", "bars": [12, 20]}]'
    " (bars are 0-based and the end is exclusive: [12, 20] is bars 12 to 19)")


def _whole_bar(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SectionError(
            f"{where}: bar numbers must be whole numbers, got {value!r}. "
            f"Format: {_SECTIONS_FORMAT}.")
    if isinstance(value, float):
        if not value.is_integer():
            raise SectionError(
                f"{where}: bar numbers must be whole numbers, got {value!r}. "
                f"Format: {_SECTIONS_FORMAT}.")
        value = int(value)
    return value


def parse_sections(raw) -> list[Section]:
    """The Sections of the `sections` argument of analyze_mix.

    Raises SectionError when it is not a list of {"name": str,
    "bars": [start, end]} with unique names and 0 <= start < end.
    """
    if not isinstance(raw, list) or not raw:
        raise SectionError(
            f"sections must be a non-empty list like {_SECTIONS_FORMAT}, "
            f"got {raw!r}. Leave it out for blocks of 4 or more bars.")
    if len(raw) > MAX_SECTIONS:
        raise SectionError(
            f"Too many sections ({len(raw)}): at most {MAX_SECTIONS}.")
    sections: list[Section] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        where = f"sections[{i}]"
        if not isinstance(item, dict):
            raise SectionError(
                f"{where} must be an object like "
                f'{{"name": "hook", "bars": [12, 20]}}, got {item!r}.')
        extra = sorted(set(item) - {"name", "bars"})
        if extra:
            raise SectionError(
                f"{where} has unknown keys {extra}: use only \"name\" and "
                f"\"bars\" ([start, end], 0-based, end exclusive).")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SectionError(
                f"{where} needs a \"name\" (a non-empty string), got "
                f"{name!r}.")
        name = name.strip()
        where = f"sections[{i}] {name!r}"
        if name in seen:
            raise SectionError(f"{where}: the name is used twice.")
        seen.add(name)
        bars = item.get("bars")
        if not isinstance(bars, (list, tuple)) or len(bars) != 2:
            raise SectionError(
                f"{where}: \"bars\" must be [start, end], got {bars!r}. "
                f"Format: {_SECTIONS_FORMAT}.")
        start = _whole_bar(bars[0], where)
        end = _whole_bar(bars[1], where)
        if start < 0:
            raise SectionError(
                f"{where}: start bar {start} is negative (bars are 0-based).")
        if end <= start:
            raise SectionError(
                f"{where}: bars [{start}, {end}] is empty: the end is "
                f"exclusive and must be greater than the start (bars "
                f"[{start}, {start + 4}] are the 4 bars {start} to "
                f"{start + 3}).")
        sections.append(Section(name, start, end))
    return sections


def default_sections(
    total_bars: int, max_sections: int = DEFAULT_MAX_SECTIONS,
) -> list[Section]:
    """Blocks of 4 bars (8, 16, ... when that gives more than max_sections)
    covering the song."""
    per_block = 4
    while math.ceil(total_bars / per_block) > max_sections:
        per_block *= 2
    return [
        Section(f"bars {first}-{min(first + per_block, total_bars)}",
                first, min(first + per_block, total_bars))
        for first in range(0, total_bars, per_block)
    ]


def fit_sections(
    sections: list[Section], total_bars: int,
) -> tuple[list[Section], list[str]]:
    """The sections that lie in a song of total_bars bars, with an end past
    the song cut to it, and notes for what was cut or left out."""
    fitted: list[Section] = []
    notes: list[str] = []
    for s in sections:
        if s.start >= total_bars:
            notes.append(
                f"Section {s.name!r} (bars [{s.start}, {s.end}]) starts "
                f"after the end of the song ({total_bars} bars) and was "
                "left out.")
        elif s.end > total_bars:
            notes.append(
                f"Section {s.name!r} ends after the song: its bars are cut "
                f"to [{s.start}, {total_bars}].")
            fitted.append(Section(s.name, s.start, total_bars))
        else:
            fitted.append(s)
    return fitted, notes


# ──────────────────────────────────────────────────────────────────
# MEASURING
# ──────────────────────────────────────────────────────────────────


def db_text(value: float | None) -> float | None:
    """dB rounded to 0.1 for a response (never -0.0)."""
    return None if value is None else round(value, 1) + 0.0


def _db(linear: float) -> float | None:
    return db_text(aa.to_db(linear))


def _rms(squares: float, count: int) -> float:
    return math.sqrt(squares / count) if count else 0.0


def find_probes(
    audio: aa.Audio, bar_length: float, sections: list[Section],
    mix_bars: list[BarStats],
) -> list[PeakProbe]:
    """Where the mix peaks in each section whose peak is at or above
    PEAK_NEAR_DBFS."""
    near = aa.from_db(PEAK_NEAR_DBFS)
    total_bars = len(mix_bars)
    probes = []
    for i, s in enumerate(sections):
        if max((b.peak for b in mix_bars[s.start:s.end]), default=0.0) < near:
            continue
        frame, peak = aa.peak_location(
            audio, bar_length, s.start, s.end, total_bars)
        probes.append(PeakProbe(i, frame, frame / audio.samplerate, peak))
    return probes


def probe_half_width(samplerate: int) -> int:
    """Frames on each side of the mix's peak in which a track's own peak
    counts as coinciding with it."""
    return max(1, round(PEAK_WINDOW_S * samplerate))


@dataclass
class _Cell:
    """One track in one section: the bars where it is active."""

    length: int        # bars of the section
    active: int        # of them where the track is active
    rms: float         # of the track over its active bars
    peak: float
    mix_rms: float     # of the mix over the same bars
    energy: float      # sum of squares of the track over the whole section

    @property
    def playing(self) -> bool:
        return self.active >= max(1, math.ceil(PLAYING_FRACTION * self.length))


def _cell(track: list[BarStats], mix: list[BarStats], s: Section) -> _Cell:
    bars = track[s.start:s.end]
    mixed = mix[s.start:s.end]
    energy = sum(b.squares for b in bars)
    active = [i for i, b in enumerate(bars) if b.rms >= ACTIVE_BAR_RMS]
    if not active:
        return _Cell(len(bars), 0, 0.0, 0.0, 0.0, energy)
    return _Cell(
        len(bars), len(active),
        _rms(sum(bars[i].squares for i in active),
             sum(bars[i].count for i in active)),
        max(bars[i].peak for i in active),
        _rms(sum(mixed[i].squares for i in active),
             sum(mixed[i].count for i in active)),
        energy)


def _ranges(flags: list[bool]) -> list[list[int]]:
    """[start, end) ranges of the True runs of flags."""
    out: list[list[int]] = []
    for i, on in enumerate(flags):
        if on and out and out[-1][1] == i:
            out[-1][1] = i + 1
        elif on:
            out.append([i, i + 1])
    return out


# ──────────────────────────────────────────────────────────────────
# SUGGESTIONS
# ──────────────────────────────────────────────────────────────────


def volume_suggestion(track: TrackStem, gain_db: float, others: str) -> str:
    """The call that changes a track's level by gain_db, as text.

    Track volume is linear (dB -> factor), 0-200. When 200 is not enough
    the text says how many dB it does give, and `others` says what else to
    do for the rest.
    """
    gain = f"{gain_db:+.1f} dB"
    if track.index is None or track.volume is None:
        return (
            f"{gain} on '{track.name}', an instrument inside a pattern (BB) "
            "track, which has no track index for set_track_volume: change "
            "the volume of its notes or its mixer channel.")
    wanted = track.volume * aa.from_db(gain_db)
    new = min(max(round(wanted), 0), VOLUME_MAX)
    call = f"set_track_volume({track.index}, {new})"
    text = f"{call}: {gain} (from {round(track.volume)})"
    if round(wanted) > VOLUME_MAX:
        reachable = aa.to_db(VOLUME_MAX / track.volume)
        text = (
            f"{call}: {reachable:+.1f} dB, the most the track volume gives "
            f"(the full {gain} would be volume {round(wanted)}). {others}")
    elif new == round(track.volume):
        text = f"{gain} is below one volume step: leave it."
    if track.volume_automated:
        text += ("" if text.endswith(".") else ".") + (
            " The volume is automated: this changes only its static "
            "value; edit the automation instead.")
    return text


def _master_suggestion(master_volume: float, peak: float) -> str:
    """Lower the Master volume so a mix peak of `peak` (linear) lands at
    PEAK_TARGET_DBFS."""
    gain_db = PEAK_TARGET_DBFS - aa.to_db(peak)
    volume = min(master_volume * aa.from_db(gain_db), MASTER_VOLUME_MAX)
    return f"set_mixer_channel_volume(0, {round(volume, 4):g}): {gain_db:+.1f} dB"


# ──────────────────────────────────────────────────────────────────
# MATCHING THE FILES OF `lmms rendertracks` TO TRACKS
# ──────────────────────────────────────────────────────────────────


@dataclass
class RenderedTrack:
    """A track `lmms rendertracks` renders: an unmuted instrument or sample
    track of the song, in the order of the project."""

    index: int
    name: str
    instrument: str = ""
    volume: float | None = None
    volume_automated: bool = False


# LMMS 1.2 writes "<n>_<track name>.wav": n counts from 1 over the tracks it
# renders (the song's unmuted instrument and sample tracks in order, then
# the instruments of the pattern tracks), muted and automation tracks take
# no number. The name loses characters LMMS cannot put in a file name.
STEM_NAME = re.compile(r"^(\d+)_(.*)\.wav$", re.DOTALL)


def _same_name(track_name: str, file_name: str) -> bool:
    def squeeze(text: str) -> str:
        return re.sub(r"\W", "", text).casefold()
    return squeeze(track_name) == squeeze(file_name)


def match_stems(
    rendered: list[RenderedTrack], files: list[tuple[int, str, object]],
) -> tuple[list[tuple[RenderedTrack | None, str, object]], list[str]]:
    """Which rendered track each stem file is.

    files is (number, name, path) of every "<number>_<name>.wav". A stem is
    the track at that position of `rendered` when the names agree; else the
    one track left with its name; else it has no track (None): an
    instrument inside a pattern track. Returns ([(track or None, name,
    path)], notes) in file number order.
    """
    taken: set[int] = set()
    out: list[tuple[RenderedTrack | None, str, object]] = []
    notes: list[str] = []
    for number, name, path in sorted(files, key=lambda f: f[0]):
        position = number - 1
        track = None
        if 0 <= position < len(rendered) and position not in taken \
                and _same_name(rendered[position].name, name):
            track = position
        else:
            same = [i for i, r in enumerate(rendered)
                    if i not in taken and _same_name(r.name, name)]
            if len(same) == 1:
                track = same[0]
            elif position < len(rendered):
                notes.append(
                    f"The render file {number}_{name}.wav does not match "
                    f"track '{rendered[position].name}' that LMMS should "
                    "have rendered there; it is reported under its file "
                    "name without a track index.")
        if track is not None:
            taken.add(track)
        out.append((None if track is None else rendered[track], name, path))
    for i, r in enumerate(rendered):
        if i not in taken:
            notes.append(
                f"Track {r.index} '{r.name}' has no file in the render "
                "output and is not in the report.")
    return out, notes


# ──────────────────────────────────────────────────────────────────
# THE REPORT
# ──────────────────────────────────────────────────────────────────


def _track_label(t: TrackStem) -> dict:
    return {"track": t.index, "name": t.name}


def _level_flags(
    kind: str, tracks: list[TrackStem], sections: list[Section],
    cells: list[list[_Cell]],
) -> list[dict]:
    """The "dominant" and "buried" flags (one per track, naming its flagged
    sections and the worst of them)."""
    flags = []
    for ti, t in enumerate(tracks):
        hits = []   # (size in dB, section index, details)
        for si, s in enumerate(sections):
            cell = cells[ti][si]
            if not cell.playing:
                continue
            if kind == "buried":
                if cell.mix_rms <= 0.0:
                    continue
                below = aa.to_db(cell.mix_rms) - aa.to_db(cell.rms)
                if below > BURIED_DB:
                    hits.append((below, si, {
                        "rms_dbfs": _db(cell.rms),
                        "mix_rms_dbfs": _db(cell.mix_rms),
                        "below_mix_db": db_text(below)}))
            else:
                others = [
                    cells[oi][si].rms for oi in range(len(tracks))
                    if oi != ti and cells[oi][si].playing]
                if len(others) < DOMINANT_MIN_OTHERS:
                    continue
                mean = sum(others) / len(others)
                above = aa.to_db(cell.rms) - aa.to_db(mean)
                if above > DOMINANT_DB:
                    hits.append((above, si, {
                        "rms_dbfs": _db(cell.rms),
                        "others_mean_dbfs": _db(mean),
                        "above_others_db": db_text(above)}))
        if not hits:
            continue
        size, si, details = max(hits, key=lambda h: h[0])
        if kind == "buried":
            gain_db = size - BURIED_TARGET_DB
            suggestion = volume_suggestion(
                t, gain_db,
                "For the rest lower the louder tracks (see the \"dominant\" "
                "flags) with set_track_volume.")
            what = (f"{t.name} is {size:.1f} dB below the mix when it plays "
                    f"(worst in {sections[si].name!r})")
        else:
            gain_db = -(size - DOMINANT_TARGET_DB)
            suggestion = volume_suggestion(t, gain_db, "")
            what = (f"{t.name} is {size:.1f} dB above the other tracks that "
                    f"play (worst in {sections[si].name!r})")
        flags.append({
            "type": kind,
            **_track_label(t),
            "sections": [sections[h[1]].name for h in hits],
            "worst": {"section": sections[si].name, **details},
            "gain_db": db_text(gain_db),
            "suggestion": suggestion,
            "_size": size,
            "_what": what,
        })
    flags.sort(key=lambda f: -f["_size"])
    return flags


def _peak_flags(
    tracks: list[TrackStem], sections: list[Section],
    probes: list[PeakProbe], master_volume: float,
) -> list[dict]:
    flags = []
    target = aa.from_db(PEAK_TARGET_DBFS)
    for pi, probe in enumerate(probes):
        s = sections[probe.section]
        excess = probe.peak - target
        peaks = [(t, t.probe_peaks[pi]) for t in tracks
                 if pi < len(t.probe_peaks)]
        floor = probe.peak * aa.from_db(-CONTRIBUTOR_DB)
        near = sorted((p for p in peaks if p[1] >= floor),
                      key=lambda p: -p[1])[:MAX_CONTRIBUTORS]
        contributors = []
        for t, peak in near:
            factor = 1.0 - excess / peak
            entry = {
                **_track_label(t),
                "peak_dbfs": _db(peak),
            }
            if factor < MIN_FACTOR:
                entry["gain_db"] = None
                entry["suggestion"] = (
                    f"lowering '{t.name}' alone cannot bring the peak to "
                    f"{PEAK_TARGET_DBFS:g} dBFS: lower several of these "
                    "tracks, or the Master volume (master_suggestion).")
            else:
                gain_db = aa.to_db(factor)
                entry["gain_db"] = db_text(gain_db)
                entry["suggestion"] = volume_suggestion(t, gain_db, "")
            contributors.append(entry)
        flags.append({
            "type": "peak_contributors",
            "section": s.name,
            "bars": [s.start, s.end],
            "mix_peak_dbfs": _db(probe.peak),
            "at_s": round(probe.seconds, 1),
            "contributors": contributors,
            "master_suggestion": (
                f"to bring the peak to {PEAK_TARGET_DBFS:g} dBFS: "
                + _master_suggestion(master_volume, probe.peak)),
            "_size": aa.to_db(probe.peak),
            "_what": (f"the mix peaks at {_db(probe.peak):+.1f} dBFS in "
                      f"{s.name!r}"),
        })
    flags.sort(key=lambda f: -f["_size"])
    return flags


def analyze_balance(
    mix: list[BarStats],
    tracks: list[TrackStem],
    sections: list[Section],
    probes: list[PeakProbe] | None = None,
    master_volume: float = 1.0,
) -> dict:
    """The report of analyze_mix (see its docstring), without the fields
    that come from the render.

    mix and each track's bars are BarStats of the same bars; sections lie
    within them. Returns {"mix": {"sections"}, "tracks", "flags", "notes"}.
    """
    probes = probes or []
    cells = [[_cell(t.bars, mix, s) for s in sections] for t in tracks]
    totals = [sum(cells[ti][si].energy for ti in range(len(tracks)))
              for si in range(len(sections))]

    mix_sections = []
    for s in sections:
        bars = mix[s.start:s.end]
        mix_sections.append({
            "name": s.name,
            "bars": [s.start, s.end],
            "peak_dbfs": _db(max((b.peak for b in bars), default=0.0)),
            "rms_dbfs": _db(_rms(sum(b.squares for b in bars),
                                 sum(b.count for b in bars))),
        })

    notes: list[str] = []
    track_json = []
    for ti, t in enumerate(tracks):
        active_flags = [b.rms >= ACTIVE_BAR_RMS for b in t.bars]
        active = [b for b, on in zip(t.bars, active_flags) if on]
        ranges = _ranges(active_flags)
        entry: dict = {
            "index": t.index,
            "name": t.name,
            "instrument": t.instrument,
        }
        if t.volume is not None:
            entry["volume"] = round(t.volume, 1)
        if t.volume_automated:
            entry["volume_automated"] = True
        entry["peak_dbfs"] = _db(max((b.peak for b in t.bars), default=0.0))
        entry["rms_dbfs"] = _db(_rms(sum(b.squares for b in active),
                                     sum(b.count for b in active)))
        entry["active_bars"] = ranges[:MAX_RANGES]
        if len(ranges) > MAX_RANGES:
            entry["active_bars_more_ranges"] = len(ranges) - MAX_RANGES
        per_section: dict = {}
        silent = []
        for si, s in enumerate(sections):
            cell = cells[ti][si]
            if not cell.active:
                silent.append(s.name)
                continue
            row = {"rms_dbfs": _db(cell.rms), "peak_dbfs": _db(cell.peak)}
            if totals[si] > 0:
                row["share_pct"] = round(100.0 * cell.energy / totals[si], 1)
            if cell.active < cell.length:
                row["active_bars"] = cell.active
            per_section[s.name] = row
        entry["sections"] = per_section
        if silent:
            entry["silent_in"] = silent
        track_json.append(entry)
        if not active:
            notes.append(
                f"Track {t.index} '{t.name}' is silent in the whole render "
                "(no notes, volume 0, or its sound file or plugin is "
                "missing).")

    flags: list[dict] = []
    # A dominant track sets the level of the mix the others are measured
    # against, so it comes before the "buried" flags it causes.
    for kind_flags in (
        _peak_flags(tracks, sections, probes, master_volume),
        _level_flags("dominant", tracks, sections, cells),
        _level_flags("buried", tracks, sections, cells),
    ):
        if len(kind_flags) > MAX_FLAGS_PER_KIND:
            notes.append(
                f"{len(kind_flags) - MAX_FLAGS_PER_KIND} more "
                f"\"{kind_flags[0]['type']}\" flags (the least serious) are "
                "not shown.")
        flags.extend(kind_flags[:MAX_FLAGS_PER_KIND])
    flags = [
        {"type": f.pop("type"), "summary": f.pop("_what"),
         **{k: v for k, v in f.items() if k != "_size"}}
        for f in flags]
    return {
        "mix": {"sections": mix_sections},
        "tracks": track_json,
        "flags": flags,
        "notes": notes,
    }
