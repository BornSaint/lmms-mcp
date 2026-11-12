"""ZynAddSubFX preset (.xiz) discovery and loading.

.xiz files are gzip-compressed XML documents with a <ZynAddSubFX-data>
root element. LMMS ships ~950 presets in data/presets/ZynAddSubFX.
"""

import difflib
import gzip
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import lmms_app

PRESET_EXTENSIONS = {".xiz"}


def _static_preset_candidates() -> list[Path]:
    """Well-known preset locations (Windows, Linux system, per-user)."""
    home = Path.home()
    return [
        # Windows installs
        Path("C:/Program Files/LMMS/data/presets/ZynAddSubFX"),
        Path("C:/Program Files (x86)/LMMS/data/presets/ZynAddSubFX"),
        # Linux system installs
        Path("/usr/share/lmms/presets/ZynAddSubFX"),
        Path("/usr/local/share/lmms/presets/ZynAddSubFX"),
        # Per-user LMMS working directories
        home / "lmms/presets/ZynAddSubFX",
        home / "Documents/LMMS/presets/ZynAddSubFX",
        home / ".lmms/presets/ZynAddSubFX",
    ]


def get_presets_dir() -> Path | None:
    """Return the first existing ZynAddSubFX presets directory, if any.

    Evaluated on every call (not cached). Order:
    1. LMMS_PRESETS_DIR env var: either the ZynAddSubFX folder itself or
       its parent presets folder (containing a ZynAddSubFX/ subfolder).
    2. Relative to the detected LMMS executable (symlinks resolved):
       <exe dir>/data/presets/ZynAddSubFX (Windows) and
       <exe dir>/../share/lmms/presets/ZynAddSubFX (Linux prefix).
    3. Well-known Windows, Linux system and per-user locations.
    """
    candidates: list[Path] = []

    override = os.environ.get("LMMS_PRESETS_DIR")
    if override:
        given = Path(override)
        candidates.append(given / "ZynAddSubFX")
        candidates.append(given)

    exe = lmms_app.find_lmms_exe()
    if exe is not None:
        real_dir = exe.resolve().parent
        candidates.append(real_dir / "data/presets/ZynAddSubFX")
        candidates.append(
            real_dir.parent / "share/lmms/presets/ZynAddSubFX"
        )

    candidates.extend(_static_preset_candidates())

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def list_categories() -> list[str]:
    """List preset categories (subdirectories) in the presets dir."""
    base = get_presets_dir()
    if base is None:
        return []
    cats = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and any(d.glob("*.xiz"))
    )
    # Presets may also live directly in the root dir
    if any(base.glob("*.xiz")):
        cats.insert(0, "(root)")
    return cats


def list_presets(category: str | None = None) -> list[dict]:
    """List available .xiz presets, optionally filtered by category.

    Returns a list of dicts with name, category and path.
    """
    base = get_presets_dir()
    if base is None:
        return []

    results = []
    if category and category != "(root)":
        search_dir = base / category
        if not search_dir.is_dir():
            raise ValueError(
                f"Unknown preset category '{category}'. "
                f"Available: {', '.join(list_categories())}"
            )
        dirs_to_scan = [(category, search_dir)]
    elif category == "(root)":
        dirs_to_scan = [("(root)", base)]
    else:
        dirs_to_scan = [
            (d.name, d) for d in sorted(base.iterdir()) if d.is_dir()
        ]
        if any(base.glob("*.xiz")):
            dirs_to_scan.append(("(root)", base))

    for cat_name, cat_dir in dirs_to_scan:
        for xiz in sorted(cat_dir.glob("*.xiz")):
            results.append({
                "name": xiz.stem,
                "category": cat_name,
                "path": str(xiz),
            })
    return results


class PresetNotFoundError(FileNotFoundError):
    """Raised when no preset file matches; carries close-name suggestions."""

    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []


@dataclass
class PresetMatch:
    """A resolved preset file.

    Attributes:
        path: Absolute path of the .xiz file that was chosen.
        name: Display label "Category/Name" (numeric prefix removed;
            just "Name" for presets outside any category folder).
        exact: True when the request named the file directly (absolute
            path or path relative to the presets dir); False when it was
            resolved by searching the library.
        other_matches: Other "Category/Name" labels that the same search
            matched (only filled when a search matched several files).
        other_total: Total number of other matches (other_matches itself
            is capped at MAX_OTHER_MATCHES).
    """

    path: Path
    name: str
    exact: bool = True
    other_matches: list[str] = field(default_factory=list)
    other_total: int = 0


MAX_OTHER_MATCHES = 5
MAX_SUGGESTIONS = 5

_NUMERIC_PREFIX = re.compile(r"^\d+-")


def display_name(stem: str) -> str:
    """Preset name without the factory "0034-" style numeric prefix."""
    return _NUMERIC_PREFIX.sub("", stem)


def preset_label(path: Path, base: Path | None = None) -> str:
    """Return the "Category/Name" label for a preset file.

    The category is the folder relative to the presets dir. Presets
    directly in the presets dir, or outside of it, get just "Name".
    """
    name = display_name(path.stem)
    if base is None:
        base = get_presets_dir()
    if base is not None:
        try:
            rel_parent = path.resolve().relative_to(base.resolve()).parent
        except ValueError:
            return name
        if rel_parent.parts:
            return f"{rel_parent.as_posix()}/{name}"
    return name


def find_preset(path: str) -> PresetMatch:
    """Resolve a preset request to a single .xiz file.

    Accepted forms: absolute path, path relative to the presets dir
    ("Strings/0034-Dark Strings"), "Category/Name" ("Strings/Dark
    Strings"), or a plain name ("Dark Strings" or "0034-Dark Strings").
    Numeric prefixes like "0001-" are optional and the comparison is
    case-insensitive. A name matching a preset's name exactly wins over
    a name that only matches the end of another preset's name.

    Raises:
        RuntimeError: No presets directory found.
        PresetNotFoundError: Nothing matched (carries suggestions).
    """
    given = Path(path)

    # 1. Direct hit (absolute path or path relative to the presets dir)
    if given.is_absolute() and given.is_file():
        return PresetMatch(given, preset_label(given), exact=True)
    base = get_presets_dir()
    if not given.is_absolute() and base is not None:
        direct = base / (
            given.with_suffix(".xiz") if not given.suffix else given
        )
        if direct.is_file():
            return PresetMatch(direct, preset_label(direct, base), exact=True)

    # 2. Search the library by name (ignores "0001-" style prefixes)
    if base is None:
        raise RuntimeError(
            "No ZynAddSubFX presets directory found. Set LMMS_PRESETS_DIR."
        )
    name_part = given.stem if given.suffix == ".xiz" else given.name
    wanted = display_name(name_part).lower()
    wanted_raw = name_part.lower()

    # "Category/Name": search only that category when the folder exists.
    search_dir = base
    if not given.is_absolute() and given.parent.parts:
        category_dir = _find_category_dir(base, given.parent)
        if category_dir is not None:
            search_dir = category_dir

    files = sorted(search_dir.rglob("*.xiz"))
    # Tier 1: the whole name matches (with or without numeric prefix).
    matches = [
        f for f in files
        if f.stem.lower() == wanted_raw or display_name(f.stem).lower() == wanted
    ]
    if not matches:
        # Tier 2: the name only matches the end of the preset name.
        matches = [f for f in files if f.stem.lower().endswith(wanted_raw)]
    if not matches:
        suggestions = suggest_presets(path)
        message = f"Preset '{path}' not found."
        if suggestions:
            message += " Did you mean: " + ", ".join(
                f"'{s}'" for s in suggestions
            ) + "?"
        message += (
            " Use list_zyn_presets (optionally with a category) to browse "
            "available presets, then pass \"Category/Name\"."
        )
        raise PresetNotFoundError(message, suggestions)

    # Prefer the closest name (shortest), then a stable order by path.
    matches.sort(key=lambda p: (len(p.stem), str(p)))
    best = matches[0]
    labels = [preset_label(m, base) for m in matches[1:]]
    best_label = preset_label(best, base)
    # Distinct labels only (a label can repeat for e.g. 0001-X / 0002-X).
    others = [lbl for lbl in dict.fromkeys(labels) if lbl != best_label]
    return PresetMatch(
        best,
        best_label,
        exact=False,
        other_matches=others[:MAX_OTHER_MATCHES],
        other_total=len(others),
    )


def _find_category_dir(base: Path, category: Path) -> Path | None:
    """Find the (case-insensitive) preset category folder, if it exists."""
    candidate = base / category
    if candidate.is_dir():
        return candidate
    try:
        wanted = [part.lower() for part in category.parts]
        current = base
        for part in wanted:
            current = next(
                (d for d in current.iterdir()
                 if d.is_dir() and d.name.lower() == part),
                None,
            )
            if current is None:
                return None
        return current
    except OSError:
        return None


def suggest_presets(request: str, limit: int = MAX_SUGGESTIONS) -> list[str]:
    """Return up to `limit` "Category/Name" labels close to `request`."""
    base = get_presets_dir()
    if base is None:
        return []
    labels = [preset_label(f, base) for f in sorted(base.rglob("*.xiz"))]
    requested = Path(request)
    stem = requested.stem if requested.suffix == ".xiz" else requested.name
    query_name = display_name(stem).lower()
    if requested.parent.parts and not requested.is_absolute():
        query_full = (
            requested.parent.as_posix() + "/" + display_name(stem)
        ).lower()
    else:
        query_full = ""

    # Candidate keys: plain names and full "Category/Name" labels.
    keys: dict[str, list[str]] = {}
    for label in labels:
        keys.setdefault(label.lower(), []).append(label)
        keys.setdefault(label.rsplit("/", 1)[-1].lower(), []).append(label)

    result: list[str] = []
    for query in (query_full, query_name):
        if not query:
            continue
        for key in difflib.get_close_matches(query, list(keys), n=limit,
                                             cutoff=0.6):
            for label in keys[key]:
                if label not in result:
                    result.append(label)
    return result[:limit]


def load_preset(path: str) -> tuple[str, PresetMatch]:
    """Resolve a preset request and return (xml, match)."""
    match = find_preset(path)
    return _read_preset(match.path), match


def load_preset_xml(path: str) -> str:
    """Load a .xiz preset file and return its XML content as string.

    Supports absolute paths, paths relative to the presets dir, and
    fuzzy "Category/Name" or plain "Name" lookups against the installed
    preset library (numeric prefixes like "0001-" are ignored). Use
    load_preset() to also learn which file was used.
    """
    return load_preset(path)[0]


def _read_preset(path: Path) -> str:
    """Read a preset file, transparently decompressing gzip (.xiz)."""
    if _is_gzip(path):
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return f.read()
        except (OSError, EOFError) as exc:
            raise ValueError(f"Cannot read preset '{path}': {exc}") from exc
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Cannot read preset '{path}': {exc}") from exc


def _is_gzip(path: Path) -> bool:
    """Check gzip magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False
