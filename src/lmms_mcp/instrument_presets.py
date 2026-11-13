"""Instrument preset (.xpf) discovery and reading.

An .xpf file is the plain XML LMMS writes for "Save as preset" on an
instrument track (it is not compressed). The structure, in every LMMS
version that has it:

    <lmms-project type="instrumenttracksettings" ...>   (root tag
      <head/>                  lmms-project, multimediaproject or
      <instrumenttracksettings name="Amazing Bubbles" ...>   multimedia-project)
        <instrumenttrack vol pan basenote pitch pitchrange fxch|mixch ...>
          <instrument name="tripleoscillator">
            <tripleoscillator .../>     the plugin's settings
          </instrument>
          <eldata/> <chordcreator/> <arpeggiator/> <midiport/> <fxchain/>
        </instrumenttrack>
      </instrumenttracksettings>
    </lmms-project>

Presets saved by LMMS before 1.1 have no <instrument> element: the plugin's
settings sit directly in <instrumenttrack> and the tag is the plugin name
(LMMS loads such a node as an instrument, see Preset.plugin). The oldest
(0.x, type="channelsettings") are a different format and not read.

Loading a preset into a track replaces the instrument and its settings, the
envelope/filter data (eldata), the chord and arpeggio settings, the effect
chain (when the preset has effects) and the tuning (basenote, pitch,
pitchrange). The track keeps its notes, name, volume, panning, mixer channel
and MIDI port (the preset's <midiport> is not used).
"""

import copy
import difflib
import math
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.etree import ElementTree as ET

from . import lmms_app

# Extra preset folders (os.pathsep separated), searched before LMMS's own.
ENV_VAR = "LMMS_PRESET_DIRS"
EXTENSION = ".xpf"

# A folder tree is scanned for this many presets at most.
MAX_SCANNED = 2000
MAX_OTHER_MATCHES = 5
MAX_SUGGESTIONS = 5

DEFAULT_BASE_NOTE = 57

_ROOT_TAGS = {"lmms-project", "multimediaproject", "multimedia-project"}
# Children of <instrumenttrack> that are not an instrument's settings node
# (LMMS treats any other node without an `id` as the old-style instrument).
_NOT_INSTRUMENT = {
    "eldata", "chordcreator", "arpeggiator", "midiport", "fxchain",
    "instrument", "connection",
}
# Parts of the sound a preset replaces besides the instrument itself.
SOUND_PARTS = {
    "eldata": "envelopes and filter",
    "chordcreator": "chords",
    "arpeggiator": "arpeggio",
}
_DOCTYPE = re.compile(r"<!DOCTYPE[^>]*>")


class PresetError(ValueError):
    """A preset file that cannot be used; the message says why."""


class PresetNotFoundError(FileNotFoundError):
    """No preset file matches; carries close-name suggestions."""

    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []


# --- where presets live ------------------------------------------------------

def _system_roots() -> list[Path]:
    """LMMS's own preset folders: the user's, then the factory ones."""
    roots: list[Path] = []
    working = lmms_app.read_lmms_paths().get("workingdir")
    if working:
        roots.append(Path(working).expanduser() / "presets")
    home = Path.home()
    roots += [
        home / "lmms" / "presets",
        home / "Documents" / "LMMS" / "presets",
        home / ".lmms" / "presets",
    ]
    exe = lmms_app.find_lmms_exe()
    if exe is not None:
        real_dir = exe.resolve().parent
        roots += [
            real_dir / "data" / "presets",
            real_dir.parent / "share" / "lmms" / "presets",
        ]
    roots += [
        Path("C:/Program Files/LMMS/data/presets"),
        Path("C:/Program Files (x86)/LMMS/data/presets"),
        Path("/usr/share/lmms/presets"),
        Path("/usr/local/share/lmms/presets"),
    ]
    return roots


def preset_roots() -> list[Path]:
    """The existing folders searched for .xpf presets, in priority order.

    Evaluated on every call. LMMS_PRESET_DIRS (os.pathsep separated)
    comes first, then the user's LMMS presets folder (<working dir>/
    presets), then the factory presets. Each folder may hold instrument
    sub-folders (TripleOscillator/Name.xpf, like LMMS's) or presets
    directly; every .xpf below it counts.
    """
    candidates = [
        Path(part).expanduser()
        for part in os.environ.get(ENV_VAR, "").split(os.pathsep)
        if part.strip()
    ]
    candidates += _system_roots()
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        real = candidate.resolve()
        if real not in seen:
            seen.add(real)
            roots.append(candidate)
    return roots


@dataclass(frozen=True)
class PresetFile:
    """A preset file found in a preset folder.

    label is "Folder/Name": the file's path below its root without the
    extension, e.g. "TripleOscillator/AmazingBubbles" (just "Name" for a
    file directly in a root); it is what load_instrument_preset takes.
    """

    path: Path
    label: str
    root: Path


def scan_presets(roots: Iterable[Path] | None = None) -> tuple[list[PresetFile], bool]:
    """Every .xpf below the roots, in root order then by path.

    Returns (files, truncated); at most MAX_SCANNED files are returned,
    truncated says there were more. The same file is listed once.
    """
    roots = preset_roots() if roots is None else list(roots)
    files: list[PresetFile] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            found = sorted(root.rglob(f"*{EXTENSION}"))
        except OSError:
            continue
        for path in found:
            if not path.is_file():
                continue
            real = path.resolve()
            if real in seen:
                continue
            if len(files) >= MAX_SCANNED:
                return files, True
            seen.add(real)
            label = path.relative_to(root).with_suffix("").as_posix()
            files.append(PresetFile(path, label, root))
    return files, False


# --- reading a preset -------------------------------------------------------

@dataclass
class Preset:
    """What an .xpf holds, ready to be copied into an instrument track.

    plugin: the instrument plugin name as stored in the file (e.g. "OPL2").
    instrument: a fresh <instrument name=plugin> element with the plugin's
        settings (also built for old files without one).
    parts: the eldata / chordcreator / arpeggiator elements the file has.
    fxchain: the file's <fxchain> when it holds effects, else None.
    basenote / pitch / pitchrange: the stored tuning (LMMS defaults when
        the file has none); volume / panning the stored track mix.
    """

    path: Path
    name: str
    plugin: str
    instrument: ET.Element
    parts: dict[str, ET.Element] = field(default_factory=dict)
    fxchain: ET.Element | None = None
    basenote: int = DEFAULT_BASE_NOTE
    pitch: float = 0.0
    pitchrange: int = 1
    volume: int = 100
    panning: int = 0
    creator_version: str = ""


def number(value, default: float) -> float:
    """A float from an XML attribute value (default when missing or bad)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _strip_model_ids(element: ET.Element) -> None:
    """Drop the automation ids of models (<fcut value=".." id="..">).

    They link a model to automation and controllers of the project the
    preset was saved from; kept, they could collide with this project's.
    """
    for el in element.iter():
        if "value" in el.attrib:
            el.attrib.pop("id", None)


def _parse(path: Path) -> tuple[ET.Element, ET.Element, ET.Element]:
    """(root, <instrumenttracksettings>, <instrumenttrack>) of an .xpf."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PresetError(f"Cannot read preset '{path}': {exc}") from exc
    text = _DOCTYPE.sub("", text).lstrip("\ufeff \t\r\n")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise PresetError(
            f"'{path.name}' is not valid XML ({exc}). An instrument preset "
            f"is a plain-XML .xpf file written by LMMS."
        ) from exc
    kind = root.get("type")
    if root.tag not in _ROOT_TAGS:
        raise PresetError(
            f"'{path.name}' is not an LMMS file (root element "
            f"'{root.tag}'), so it is not an instrument preset."
        )
    if kind == "channelsettings":
        raise PresetError(
            f"'{path.name}' is a preset in LMMS's pre-1.0 channel format "
            f"(type 'channelsettings'), which cannot be read. Open it in "
            f"LMMS and save it again as a preset, or choose another one."
        )
    if kind != "instrumenttracksettings":
        raise PresetError(
            f"'{path.name}' is an LMMS file of type '{kind}', not an "
            f"instrument preset (type 'instrumenttracksettings')."
        )
    settings = root.find("instrumenttracksettings")
    track = None if settings is None else settings.find("instrumenttrack")
    if settings is None or track is None:
        raise PresetError(
            f"'{path.name}' has no <instrumenttrack> section, so it holds "
            f"no instrument."
        )
    return root, settings, track


def _instrument_node(track: ET.Element) -> tuple[str, list[ET.Element]]:
    """(plugin name, settings elements) of an <instrumenttrack>."""
    inst = track.find("instrument")
    if inst is not None:
        return inst.get("name") or "", list(inst)
    for child in track:
        if child.tag not in _NOT_INSTRUMENT and "id" not in child.attrib:
            return child.tag, [child]
    return "", []


def peek_plugin(path: Path) -> str | None:
    """The instrument plugin name a preset file uses, None if unreadable."""
    try:
        _, _, track = _parse(path)
    except PresetError:
        return None
    return _instrument_node(track)[0] or None


def read_preset(path: Path) -> Preset:
    """Read an .xpf into a Preset. Raises PresetError saying what is wrong."""
    root, settings, track = _parse(path)
    plugin, nodes = _instrument_node(track)
    if not plugin:
        raise PresetError(f"'{path.name}' contains no instrument.")
    instrument = ET.Element("instrument", {"name": plugin})
    for node in nodes:
        instrument.append(copy.deepcopy(node))
    _strip_model_ids(instrument)

    parts = {}
    for tag in SOUND_PARTS:
        node = track.find(tag)
        if node is not None:
            parts[tag] = copy.deepcopy(node)
            _strip_model_ids(parts[tag])
    fxchain = track.find("fxchain")
    if fxchain is not None and fxchain.find("effect") is not None:
        fxchain = copy.deepcopy(fxchain)
        _strip_model_ids(fxchain)
    else:
        fxchain = None

    base = int(number(track.get("basenote"), DEFAULT_BASE_NOTE))
    return Preset(
        path=path,
        name=(settings.get("name") or track.get("name") or path.stem).strip(),
        plugin=plugin,
        instrument=instrument,
        parts=parts,
        fxchain=fxchain,
        basenote=max(0, min(127, base)),
        pitch=number(track.get("pitch"), 0.0),
        pitchrange=max(1, int(number(track.get("pitchrange"), 1))),
        volume=int(number(track.get("vol"), 100)),
        panning=int(number(track.get("pan"), 0)),
        creator_version=root.get("creatorversion") or "",
    )


def filter_effects(preset: Preset, usable: Callable[[str], bool]) -> list[str]:
    """Drop the preset's effects whose plugin `usable` rejects.

    Returns the dropped plugin names. The preset's chain becomes None when
    nothing usable is left.
    """
    chain = preset.fxchain
    if chain is None:
        return []
    dropped = []
    for effect in chain.findall("effect"):
        plugin = effect.get("name", "")
        if not usable(plugin):
            chain.remove(effect)
            dropped.append(plugin)
    count = len(chain.findall("effect"))
    if count:
        chain.set("numofeffects", str(count))
    else:
        preset.fxchain = None
    return dropped


# --- files the preset plays --------------------------------------------------

@dataclass
class SampleCheck:
    """Result of resolve_samples: files not found, paths made absolute and
    paths that could not be checked."""

    missing: list[str] = field(default_factory=list)
    made_absolute: dict[str, str] = field(default_factory=dict)
    unverified: list[str] = field(default_factory=list)


def is_absolute_path(path: str) -> bool:
    return PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()


def resolve_samples(preset: Preset, search_dirs: list[Path]) -> SampleCheck:
    """Check the sample / soundfont files (`src`) the instrument plays.

    LMMS finds a relative path in the user's samples folder, then the
    factory samples folder (search_dirs, see lmms_app.sample_search_dirs),
    never next to the preset. So an absolute path must exist; a relative
    one is kept when LMMS finds it, rewritten to an absolute path when the
    file lies beside the preset (which LMMS would not find), and reported
    missing otherwise. Without any search folder (no LMMS installation) a
    relative path that is not beside the preset cannot be checked.
    """
    check = SampleCheck()
    for element in preset.instrument.iter():
        src = (element.get("src") or "").strip()
        if not src:
            continue
        if is_absolute_path(src):
            if not Path(src).is_file():
                check.missing.append(src)
            continue
        if any((d / src).is_file() for d in search_dirs):
            continue
        beside = preset.path.parent / src
        if beside.is_file():
            element.set("src", str(beside.resolve()))
            check.made_absolute[src] = str(beside.resolve())
        elif search_dirs:
            check.missing.append(src)
        else:
            check.unverified.append(src)
    return check


# --- finding a preset --------------------------------------------------------

@dataclass
class PresetMatch:
    """A resolved preset file.

    label is "Folder/Name"; exact is True when the request named the file
    (a path); other_matches are other labels the same search matched
    (capped at MAX_OTHER_MATCHES, other_total counts them all).
    """

    path: Path
    label: str
    exact: bool = True
    other_matches: list[str] = field(default_factory=list)
    other_total: int = 0


def _norm(text: str) -> str:
    """Name for comparing: lowercase letters and digits only, so
    "Amazing Bubbles", "amazing_bubbles" and "AmazingBubbles" agree."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _label_of(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return path.relative_to(root).with_suffix("").as_posix()
        except ValueError:
            continue
    return path.stem


def suggest(request: str, files: list[PresetFile]) -> list[str]:
    """Up to MAX_SUGGESTIONS labels that are close to the request."""
    text = request.replace("\\", "/").strip("/")
    if text.lower().endswith(EXTENSION):
        text = text[: -len(EXTENSION)]
    keys: dict[str, list[str]] = {}
    for f in files:
        keys.setdefault(_norm(f.label), []).append(f.label)
        keys.setdefault(_norm(Path(f.label).name), []).append(f.label)
    result: list[str] = []
    for query in (_norm(text), _norm(text.rsplit("/", 1)[-1])):
        if not query:
            continue
        for key in difflib.get_close_matches(query, list(keys),
                                             n=MAX_SUGGESTIONS, cutoff=0.6):
            for label in keys[key]:
                if label not in result:
                    result.append(label)
    return result[:MAX_SUGGESTIONS]


def find_preset(
    request: str,
    canonical: Callable[[str], str] | None = None,
) -> PresetMatch:
    """Resolve a preset request to one .xpf file.

    Accepted: an absolute path (with or without ".xpf"), a path below a
    preset folder ("TripleOscillator/AmazingBubbles"), "Instrument/Name"
    ("TripleOscillator/Amazing Bubbles", "opulenz/Bells") or a plain name
    ("Amazing Bubbles"). Case, spaces, "_" and "-" do not matter, and a
    name that only matches the end of a preset's name is accepted when no
    preset matches whole. `canonical` maps an instrument/plugin name to
    one spelling (opl2 -> opulenz), so "Instrument" may be either the
    folder or the instrument the preset uses.

    Raises PresetNotFoundError (with suggestions) when nothing matches.
    """
    text = request.strip()
    if not text:
        raise PresetNotFoundError(
            "preset must name a preset, e.g. \"TripleOscillator/"
            "AmazingBubbles\". list_instrument_presets shows what is "
            "available.")
    canonical = canonical or (lambda name: name.strip().lower())
    roots = preset_roots()

    given = Path(text).expanduser()
    if given.suffix.lower() == ".xiz":
        raise PresetNotFoundError(
            f"'{text}' is a ZynAddSubFX preset (.xiz): use load_zyn_preset "
            f"for it. load_instrument_preset reads LMMS instrument presets "
            f"(.xpf).")
    if given.is_absolute():
        for candidate in (given, given.with_name(given.name + EXTENSION)):
            if candidate.is_file():
                if candidate.suffix.lower() != EXTENSION:
                    raise PresetNotFoundError(
                        f"'{text}' is not an .xpf file. "
                        f"load_instrument_preset reads LMMS instrument "
                        f"presets (.xpf).")
                return PresetMatch(candidate, _label_of(candidate, roots))
        files, _ = scan_presets(roots)
        raise PresetNotFoundError(
            f"Preset file '{text}' does not exist.", suggest(text, files))

    relative = text.replace("\\", "/")
    for root in roots:
        for candidate in (root / relative, root / (relative + EXTENSION)):
            if candidate.is_file() and candidate.suffix.lower() == EXTENSION:
                return PresetMatch(candidate, _label_of(candidate, roots))

    files, _ = scan_presets(roots)
    parts = [p for p in relative.split("/") if p]
    name = parts[-1] if parts else relative
    if name.lower().endswith(EXTENSION):
        name = name[: -len(EXTENSION)]
    category = parts[:-1]
    wanted, wanted_norm = name.lower(), _norm(name)

    def in_category(f: PresetFile) -> bool:
        if not category:
            return True
        folders = _norm("".join(Path(f.label).parent.parts))
        wanted_cat = _norm("".join(category))
        if wanted_cat and folders.endswith(wanted_cat):
            return True
        plugin = peek_plugin(f.path)
        return plugin is not None and canonical(plugin) == canonical(category[-1])

    tiers: list[Callable[[PresetFile], bool]] = [
        lambda f: f.path.stem.lower() == wanted,
        lambda f: bool(wanted_norm) and _norm(f.path.stem) == wanted_norm,
        lambda f: bool(wanted_norm) and _norm(f.path.stem).endswith(wanted_norm),
    ]
    for tier in tiers:
        matches = [f for f in files if tier(f) and in_category(f)]
        if matches:
            break
    else:
        message = f"Preset '{request}' not found."
        suggestions = suggest(request, files)
        if suggestions:
            message += " Did you mean: " + ", ".join(
                f"'{s}'" for s in suggestions) + "?"
        message += (
            " list_instrument_presets shows the available presets; pass "
            "\"Instrument/Name\"."
        )
        raise PresetNotFoundError(message, suggestions)

    # Closest name first (shortest), then earlier folder, then path.
    order = {root: i for i, root in enumerate(roots)}
    matches.sort(key=lambda f: (
        len(f.path.stem), order.get(f.root, len(order)), str(f.path)))
    best = matches[0]
    others = [
        lbl for lbl in dict.fromkeys(f.label for f in matches[1:])
        if lbl != best.label
    ]
    return PresetMatch(
        best.path, best.label, exact=False,
        other_matches=others[:MAX_OTHER_MATCHES], other_total=len(others),
    )


# --- writing a preset into a track ------------------------------------------

def put_child(parent: ET.Element, element: ET.Element, first: bool = False) -> None:
    """Replace the child of `parent` with the element's tag, in place (or add it)."""
    for i, child in enumerate(parent):
        if child.tag == element.tag:
            parent[i] = element
            return
    if first:
        parent.insert(0, element)
    else:
        parent.append(element)
