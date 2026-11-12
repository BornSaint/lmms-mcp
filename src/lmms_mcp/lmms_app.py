"""Detect the installed LMMS application and its available plugins.

The MCP server manipulates project files directly without launching
LMMS. However, knowing which plugins the installed LMMS actually ships
prevents generating projects with missing-plugin warnings.

LMMS 1.2.x and 1.3.x differ in plugin availability, e.g.:
- SlicerT, Xpressive: 1.3+ only
- Compressor, Dispersion, FrequencyShifter, SlewDistortion effects: 1.3+ only
"""

import glob
import os
import re
import shutil
import subprocess
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

# Support libraries shipped next to the real plugins on Linux (.so files
# that are not LMMS plugins). Names are lowercase with the "lib" prefix
# already stripped.
_SUPPORT_LIBS = {"zynaddsubfxcore"}


def _static_exe_candidates() -> list[Path]:
    """Well-known install locations (Windows and Linux)."""
    return [
        Path("C:/Program Files/LMMS/lmms.exe"),
        Path("C:/Program Files (x86)/LMMS/lmms.exe"),
        Path.home() / "AppData/Local/Programs/LMMS/lmms.exe",
        Path("/usr/bin/lmms"),
        Path("/usr/local/bin/lmms"),
    ]


def find_lmms_exe() -> Path | None:
    """Locate the installed LMMS executable.

    Order: LMMS_EXECUTABLE env var, well-known Windows locations, the
    executable named "lmms" on PATH, well-known Linux locations.
    Evaluated on every call (not cached).
    """
    candidates: list[Path] = []
    env_exe = os.environ.get("LMMS_EXECUTABLE")
    if env_exe:
        candidates.append(Path(env_exe))
    static = _static_exe_candidates()
    candidates.extend(p for p in static if p.suffix.lower() == ".exe")
    on_path = shutil.which("lmms")
    if on_path:
        candidates.append(Path(on_path))
    candidates.extend(p for p in static if p.suffix.lower() != ".exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _has_plugin_files(directory: Path) -> bool:
    """True if the directory directly contains .dll or .so files."""
    if not directory.is_dir():
        return False
    return any(directory.glob("*.dll")) or any(directory.glob("*.so"))


def get_plugins_dir() -> Path | None:
    """Return the plugins directory of the installed LMMS.

    Order: LMMS_PLUGINS_DIR env var, Windows layout
    (<exe dir>/plugins), then Linux layouts (<prefix>/lib/lmms relative
    to the executable, /usr/lib/<multiarch>/lmms, /usr/lib/lmms,
    /usr/lib64/lmms, /usr/local/lib/lmms). The first directory that
    contains plugin files wins.
    """
    override = os.environ.get("LMMS_PLUGINS_DIR")
    if override and Path(override).is_dir():
        return Path(override)

    candidates: list[Path] = []
    exe = find_lmms_exe()
    if exe is not None:
        real_dir = exe.resolve().parent
        candidates.append(real_dir / "plugins")
        candidates.append(real_dir.parent / "lib" / "lmms")
        candidates.append(real_dir.parent / "lib64" / "lmms")
    candidates.extend(
        Path(p) for p in sorted(glob.glob("/usr/lib/*/lmms"))
    )
    candidates.extend([
        Path("/usr/lib/lmms"),
        Path("/usr/lib64/lmms"),
        Path("/usr/local/lib/lmms"),
    ])
    for candidate in candidates:
        if _has_plugin_files(candidate):
            return candidate
    return None


def get_samples_dir() -> Path | None:
    """Return the factory samples directory of the installed LMMS.

    Relative sample paths in projects (e.g. "drums/kick01.ogg") are
    resolved against this directory by LMMS. Order: Windows layout
    (<exe dir>/data/samples), Linux layout relative to the resolved
    executable (<exe dir>/../share/lmms/samples), then
    /usr/share/lmms/samples and /usr/local/share/lmms/samples. Returns
    None if no such directory exists.
    """
    candidates: list[Path] = []
    exe = find_lmms_exe()
    if exe is not None:
        real_dir = exe.resolve().parent
        candidates.append(real_dir / "data" / "samples")
        candidates.append(real_dir.parent / "share" / "lmms" / "samples")
    candidates.extend([
        Path("/usr/share/lmms/samples"),
        Path("/usr/local/share/lmms/samples"),
    ])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def get_installed_plugins() -> set[str]:
    """Set of plugin library names shipped with the installed LMMS.

    Names are lowercase library basenames without extension (e.g.
    "tripleoscillator", "reverbsc"). On Linux (.so files) the leading
    "lib" prefix is stripped, so libOPL2.so becomes "opl2"; support
    libraries such as libZynAddSubFxCore.so are excluded. Only files
    directly in the plugins dir are considered, so the ladspa/
    subdirectory is ignored. Returns an empty set if no plugins are
    found.
    """
    plugins_dir = get_plugins_dir()
    if plugins_dir is None:
        return set()
    names = {f.stem.lower() for f in plugins_dir.glob("*.dll")}
    for f in plugins_dir.glob("*.so"):
        if not f.is_file():
            continue
        name = f.stem.lower()
        if name.startswith("lib"):
            name = name[3:]
        if name and name not in _SUPPORT_LIBS:
            names.add(name)
    return names


def get_lmms_version() -> str | None:
    """Version string of the installed LMMS (e.g. '1.2.2'), or None."""
    exe = find_lmms_exe()
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [str(exe), "--version"],
            capture_output=True, text=True, timeout=10,
        )
        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", out.stdout + out.stderr)
        return match.group(1) if match else None
    except (OSError, subprocess.TimeoutExpired):
        return None


# Known aliases across versions (XML name -> possible DLL names).
# Some instruments are statically linked into lmms.exe and have no DLL;
# those are listed in STATIC_PLUGINS.
STATIC_PLUGINS = {
    # Official Windows builds link FreeBoy into the main binary
    "freeboy",
}

PLUGIN_ALIASES = {
    "nes": {"nes", "papu", "nescaline"},
    "papu": {"nes", "papu", "nescaline"},
    "malletsstk": {"malletsstk", "stk"},
    "audiofileprocessor": {"audiofileprocessor"},
    "sf2player": {"sf2player", "fluidsynth"},
    "opulenz": {"opl2", "opulenz"},  # named OPL2 in LMMS 1.2.x
}


# Known plugins that only exist in newer LMMS versions. Used to explain a
# refusal ("requires LMMS 1.3+") and to warn when availability cannot be
# verified because no LMMS installation was found.
PLUGIN_REQUIRES = {
    "slicert": "LMMS 1.3+",
    "xpressive": "LMMS 1.3+",
    "compressor": "LMMS 1.3+",
    "dispersion": "LMMS 1.3+",
    "frequencyshifter": "LMMS 1.3+",
    "slewdistortion": "LMMS 1.3+",
}

# Closest plugin to suggest when a known plugin is not available, and the
# replacement used when adapting recommended instrument/effect lists. The
# single place that defines substitutions (instruments and effects).
PLUGIN_SUBSTITUTES = {
    "slicert": "audiofileprocessor",
    "xpressive": "watsyn",
    "compressor": "dynamicsprocessor",
    "dispersion": "flanger",
    "frequencyshifter": "eq",
    "slewdistortion": "waveshaper",
}

# Plugin libraries that ship with LMMS but are neither instruments nor
# effects this server can add to a project (importers, exporters, browsers,
# analyzers, the LADSPA effect host). They are never offered as custom
# plugins.
NON_TRACK_PLUGINS = {
    "flpimport",
    "hydrogenimport",
    "ladspabrowser",
    "ladspaeffect",
    "midiexport",
    "midiimport",
    "peakcontrollereffect",
    "spectrumanalyzer",
}


def _installed_match(name: str, installed: Collection[str]) -> str | None:
    """How a lowercase plugin name is satisfied by the installed set.

    Returns "built-in", "installed", "installed (alias)" or None.
    """
    if name in STATIC_PLUGINS:
        return "built-in"
    if name in installed:
        return "installed"
    if any(alias in installed for alias in PLUGIN_ALIASES.get(name, ())):
        return "installed (alias)"
    return None


def check_plugin_available(plugin_name: str) -> tuple[bool, str]:
    """Check whether a plugin exists in the installed LMMS.

    Returns (available, reason). If no LMMS installation is detected,
    everything is considered available (cannot verify).
    """
    name = plugin_name.lower()
    installed = set() if name in STATIC_PLUGINS else get_installed_plugins()
    match = _installed_match(name, installed)
    if match:
        return True, match
    if not installed:
        return True, "LMMS installation not found - cannot verify"
    return False, (
        f"'{plugin_name}' is not included in your installed LMMS "
        f"(found {len(installed)} plugins). It may require a newer "
        f"LMMS version."
    )


def compute_available(
    known: Iterable[str], installed: Collection[str]
) -> set[str]:
    """Which of the known plugin names the given installed set provides.

    Aliases (opulenz <-> opl2, ...) and statically linked plugins are
    respected. An empty installed set means no LMMS installation was
    found, so nothing can be verified and every known name is returned.
    """
    known = {name.lower() for name in known}
    if not installed:
        return known
    return {name for name in known if _installed_match(name, installed)}


@dataclass(frozen=True)
class PluginAvailability:
    """Snapshot of what the installed LMMS offers.

    verified: False if no installation/plugins were found (then
        instruments and effects hold every known name).
    instruments / effects: the known instrument / effect names that are
        available (all known names when not verified).
    installed: raw plugin library names found (empty when not verified).
    """

    verified: bool
    version: str | None
    installed: frozenset[str]
    instruments: frozenset[str]
    effects: frozenset[str]


def build_availability(
    known_instruments: Iterable[str],
    known_effects: Iterable[str],
    installed: Collection[str],
    version: str | None,
) -> PluginAvailability:
    """Build an availability snapshot from an installed-plugin set."""
    installed = frozenset(installed)
    return PluginAvailability(
        verified=bool(installed),
        version=version,
        installed=installed,
        instruments=frozenset(compute_available(known_instruments, installed)),
        effects=frozenset(compute_available(known_effects, installed)),
    )


def detect_availability(
    known_instruments: Iterable[str], known_effects: Iterable[str]
) -> PluginAvailability:
    """Inspect the installed LMMS and build the availability snapshot."""
    installed = get_installed_plugins()
    version = get_lmms_version() if installed else None
    return build_availability(
        known_instruments, known_effects, installed, version
    )


def unavailable_reason(name: str, version: str | None) -> str:
    """Sentence explaining why a known plugin cannot be used."""
    where = f"the installed LMMS {version}" if version else "the installed LMMS"
    required = PLUGIN_REQUIRES.get(name)
    detail = (
        f"requires {required}" if required
        else "its plugin library was not found in the LMMS plugins folder"
    )
    return f"'{name}' is not available in {where} ({detail})."


def adapt_recommendations(
    recommendations: dict[str, list[str]], available: Collection[str]
) -> dict[str, list[str]]:
    """Recommended name lists limited to available plugins.

    An unavailable entry is replaced by its PLUGIN_SUBSTITUTES entry if
    that is available (and not already listed), otherwise dropped. Lists
    that end up empty are omitted.
    """
    adapted: dict[str, list[str]] = {}
    for use_case, names in recommendations.items():
        chosen: list[str] = []
        for name in names:
            if name not in available:
                name = PLUGIN_SUBSTITUTES.get(name, "")
                if name not in available:
                    continue
            if name not in chosen:
                chosen.append(name)
        if chosen:
            adapted[use_case] = chosen
    return adapted


def classify_installed_plugins(
    known_instruments: set[str],
    known_effects: set[str],
    installed: Collection[str] | None = None,
) -> dict[str, list[str]]:
    """Classify all installed plugin libraries by comparing with known names.

    Returns dict with keys "instruments", "effects", "unknown".
    Unknown plugins are custom/newer plugins the user added - they can be
    used but their type (instrument vs effect) is not known from the
    filename alone. Pass `installed` to classify a given set instead of
    scanning the installed LMMS.
    """
    if installed is None:
        installed = get_installed_plugins()
    result: dict[str, list[str]] = {"instruments": [], "effects": [], "unknown": []}
    for name in sorted(installed):
        if name.startswith("lib"):
            continue  # Windows support libraries, not plugins
        if name in known_instruments or any(
            key in known_instruments and name in aliases
            for key, aliases in PLUGIN_ALIASES.items()
        ):
            result["instruments"].append(name)
        elif name in known_effects:
            result["effects"].append(name)
        else:
            result["unknown"].append(name)
    return result


def find_vst_plugins(directory: str | Path, recursive: bool = True) -> list[dict]:
    """Scan a directory for VST plugin DLLs.

    Returns list of dicts with name and path. Note: any .dll could be a
    VST effect or instrument - LMMS decides on load.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    it = root.rglob("*.dll") if recursive else root.glob("*.dll")
    skip = {"lmms.exe", "remotevstplugin.exe", "32bitvsthelper.exe"}
    results = []
    for dll in sorted(it):
        if dll.stem.lower() in skip:
            continue
        results.append({
            "name": dll.stem,
            "path": str(dll),
            "size_kb": round(dll.stat().st_size / 1024),
        })
    return results
