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
import struct
import subprocess
import sys
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

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


# ──────────────────────────────────────────────────────────────────
# FILES THAT SAMPLER INSTRUMENTS PLAY
# ──────────────────────────────────────────────────────────────────

# Instruments that make no sound without a file: instrument -> (what the
# file is, accepted extensions). No tool other than set_instrument_file
# can give them one, so an instrument is only offered when at least one
# matching file exists on the machine (see find_instrument_files).
# Compressed .sf3 soundfonts are deliberately not accepted: LMMS 1.2.2
# crashes (segfault in fluidsynth's sample loading) when sf2player loads
# one, although fluidsynth alone plays it.
FILE_INSTRUMENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "sf2player": ("soundfont", (".sf2",)),
    "gigplayer": ("sample library", (".gig",)),
    "patman": ("GUS patch", (".pat",)),
}

# Extra directories (os.pathsep separated) searched for soundfonts, GIG
# libraries and GUS patches, before the usual places.
INSTRUMENT_FILE_DIRS_ENV = "LMMS_INSTRUMENT_FILE_DIRS"

# Where Linux distributions put soundfonts and GUS patches.
SYSTEM_FILE_DIRS: dict[str, tuple[str, ...]] = {
    "sf2player": ("/usr/share/sounds/sf2", "/usr/share/soundfonts"),
    "gigplayer": (),
    "patman": ("/usr/share/midi/freepats",),
}

# A directory is searched this many levels deep and a kind stops at this
# many files, so a huge samples folder cannot slow the start-up down.
_SCAN_DEPTH = 2
_SCAN_LIMIT = 500


def extension_text(instrument: str) -> str:
    """Accepted extensions of a file instrument, e.g. ".sf2"."""
    return "/".join(FILE_INSTRUMENTS[instrument][1])


def file_instrument_for(plugin_name: str) -> str | None:
    """The file instrument (sf2player, ...) a plugin name in a project is."""
    name = plugin_name.lower()
    for instrument in FILE_INSTRUMENTS:
        if name in plugin_candidates(instrument):
            return instrument
    return None


def read_lmms_paths(config: str | Path | None = None) -> dict[str, str]:
    """The attributes of <paths> in LMMS's config file (~/.lmmsrc.xml).

    Holds the user's sf2dir, gigdir, defaultsf2, workingdir, ... Returns
    an empty dict if the file is missing or unreadable.
    """
    path = Path(config) if config else Path.home() / ".lmmsrc.xml"
    try:
        node = ET.parse(path).getroot().find("paths")
    except (OSError, ET.ParseError):
        return {}
    return dict(node.attrib) if node is not None else {}


@dataclass(frozen=True)
class InstrumentFiles:
    """The soundfonts, GIG libraries and GUS patches found on this machine.

    files / searched / put_in are keyed by instrument name (sf2player,
    gigplayer, patman): the files found (absolute paths, no duplicates),
    every directory looked in, and the directory where a user should put
    a file (LMMS's own folder for it). default_soundfont is the one new
    sf2player tracks start with.
    """

    files: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    searched: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    put_in: Mapping[str, str] = field(default_factory=dict)
    default_soundfont: str | None = None

    def for_instrument(self, name: str) -> tuple[str, ...]:
        return tuple(self.files.get(name, ()))


def _scan_files(
    directory: Path, extensions: tuple[str, ...], limit: int
) -> list[str]:
    """Files with one of the extensions in a directory, a few levels deep.

    Names are sorted case-insensitively and symlinks to files are followed
    (dangling ones are skipped); symlinked directories are not entered.
    """
    found: list[str] = []
    if not directory.is_dir():
        return found
    base = len(directory.parts)
    for current, dirs, names in os.walk(directory):
        dirs.sort(key=str.lower)
        if len(Path(current).parts) - base >= _SCAN_DEPTH:
            dirs.clear()
        for name in sorted(names, key=str.lower):
            full = Path(current, name)
            if name.lower().endswith(extensions) and full.is_file():
                found.append(str(full))
                if len(found) >= limit:
                    return found
    return found


def find_instrument_files(
    config_paths: Mapping[str, str] | None = None,
    extra_dirs: Iterable[str | Path] | None = None,
    system_dirs: Mapping[str, Iterable[str | Path]] | None = None,
    home: str | Path | None = None,
    samples_dir: str | Path | None = None,
) -> InstrumentFiles:
    """Find the soundfonts, GIG libraries and GUS patches on this machine.

    Per instrument the directories are searched in this order: extra_dirs
    (default: LMMS_INSTRUMENT_FILE_DIRS), the folders from LMMS's config
    (sf2dir, gigdir, the folder of defaultsf2, and the working directory's
    samples folder), the LMMS factory samples folder (GUS patches),
    ~/.local/share/soundfonts and the system folders (SYSTEM_FILE_DIRS).
    Each is searched a few levels deep, bounded in size. The arguments
    replace the real machine's values (config_paths: the <paths> of
    ~/.lmmsrc.xml; samples_dir: the factory samples folder) for tests.
    """
    paths = read_lmms_paths() if config_paths is None else dict(config_paths)
    home_dir = Path(home) if home else Path.home()
    if extra_dirs is None:
        extra_dirs = [
            d for d in os.environ.get(INSTRUMENT_FILE_DIRS_ENV, "").split(
                os.pathsep) if d
        ]
    extra = [Path(d).expanduser() for d in extra_dirs]
    if samples_dir is None:
        samples_dir = get_samples_dir()
    system = SYSTEM_FILE_DIRS if system_dirs is None else system_dirs

    user_samples = Path(paths.get("workingdir") or home_dir / "lmms") / "samples"
    sf2_dir = Path(paths.get("sf2dir") or user_samples / "soundfonts")
    gig_dir = Path(paths.get("gigdir") or user_samples / "gig")
    patch_dir = user_samples / "patches"
    default_sf = None
    if paths.get("defaultsf2"):
        default_sf = Path(paths["defaultsf2"])
        if not default_sf.is_absolute():
            default_sf = sf2_dir / default_sf
    factory = Path(samples_dir) if samples_dir else None

    dirs: dict[str, list[Path]] = {
        "sf2player": [*extra, sf2_dir, home_dir / ".local/share/soundfonts"],
        "gigplayer": [*extra, gig_dir],
        "patman": [*extra, patch_dir, *([factory] if factory else [])],
    }
    put_in = {
        "sf2player": str(sf2_dir),
        "gigplayer": str(gig_dir),
        "patman": str(patch_dir),
    }

    files: dict[str, tuple[str, ...]] = {}
    searched: dict[str, tuple[str, ...]] = {}
    for instrument, (_label, extensions) in FILE_INSTRUMENTS.items():
        candidates = [
            *dirs[instrument], *(Path(d) for d in system.get(instrument, ()))
        ]
        seen_dirs: dict[str, None] = {}
        seen_files: dict[str, str] = {}  # real path -> first path found
        if instrument == "sf2player" and default_sf is not None \
                and default_sf.suffix.lower() in extensions \
                and default_sf.is_file():
            seen_files[os.path.realpath(default_sf)] = str(default_sf)
        for directory in candidates:
            seen_dirs.setdefault(str(directory))
            for found in _scan_files(directory, extensions, _SCAN_LIMIT):
                seen_files.setdefault(os.path.realpath(found), found)
        files[instrument] = tuple(seen_files.values())[:_SCAN_LIMIT]
        searched[instrument] = tuple(seen_dirs)

    default = None
    if default_sf is not None and files["sf2player"] \
            and str(default_sf) == files["sf2player"][0]:
        default = str(default_sf)
    elif files["sf2player"]:
        default = files["sf2player"][0]
    return InstrumentFiles(
        files=files, searched=searched, put_in=put_in,
        default_soundfont=default,
    )


def read_soundfont_presets(path: str | Path) -> dict[tuple[int, int], str] | None:
    """The presets of a SoundFont as {(bank, patch): name}.

    Reads only the preset list (phdr chunk) of an .sf2 file.
    Returns None if the file cannot be read as a SoundFont.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"sfbk":
                return None
            while True:
                chunk = f.read(8)
                if len(chunk) < 8:
                    return None
                chunk_id, size = struct.unpack("<4sI", chunk)
                if chunk_id != b"LIST" or f.read(4) != b"pdta":
                    # (a non-pdta LIST has its 4-byte form type consumed)
                    skip = size - 4 if chunk_id == b"LIST" else size
                    f.seek(skip + (size & 1), os.SEEK_CUR)
                    continue
                end = f.tell() + size - 4
                while f.tell() < end:
                    sub = f.read(8)
                    if len(sub) < 8:
                        return None
                    sub_id, sub_size = struct.unpack("<4sI", sub)
                    if sub_id != b"phdr":
                        f.seek(sub_size + (sub_size & 1), os.SEEK_CUR)
                        continue
                    data = f.read(sub_size)
                    presets = {}
                    for i in range(0, len(data) - 38, 38):  # last = terminator
                        name, patch, bank = struct.unpack_from(
                            "<20sHH", data, i)
                        presets[(bank, patch)] = (
                            name.split(b"\0")[0].decode("latin-1").strip()
                        )
                    return presets
                return None
    except (OSError, struct.error):
        return None


def default_soundfont_preset(path: str | Path) -> tuple[int, int]:
    """(bank, patch) a new sf2player track should start with.

    General MIDI piano (bank 0, patch 0) if the soundfont has it, else its
    first preset; (0, 0) if the file cannot be read.
    """
    presets = read_soundfont_presets(path)
    if not presets or (0, 0) in presets:
        return (0, 0)
    return min(presets)


def _ranges(numbers: Iterable[int]) -> str:
    """"0-2, 5, 8-9" for [0, 1, 2, 5, 8, 9]."""
    runs: list[list[int]] = []
    for n in sorted(set(numbers)):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ", ".join(
        str(a) if a == b else f"{a}-{b}" for a, b in runs
    )


def describe_soundfont_banks(presets: Mapping[tuple[int, int], str]) -> str:
    """The banks and patches a SoundFont has, e.g. "bank 0: patches 0-127"."""
    banks: dict[int, list[int]] = {}
    for bank, patch in presets:
        banks.setdefault(bank, []).append(patch)
    return "; ".join(
        f"bank {bank}: patches {_ranges(patches)}"
        for bank, patches in sorted(banks.items())
    )


def get_installed_stems() -> dict[str, str]:
    """Installed plugin libraries as {lowercase name: original-case stem}.

    The stem is the library file name without extension (and without the
    "lib" prefix on Linux), in its original case, e.g. libOPL2.so gives
    {"opl2": "OPL2"}. LMMS loads an instrument by the plugin name in the
    project XML, and on Linux that name is matched case-sensitively
    against the library stem (OPL2 loads, opl2 loads nothing), so the
    original case is what must be written into projects. Support
    libraries such as libZynAddSubFxCore.so are excluded. Only files
    directly in the plugins dir are considered, so the ladspa/
    subdirectory is ignored. Returns an empty dict if no plugins are
    found.
    """
    plugins_dir = get_plugins_dir()
    if plugins_dir is None:
        return {}
    stems = {f.stem.lower(): f.stem for f in plugins_dir.glob("*.dll")}
    for f in plugins_dir.glob("*.so"):
        if not f.is_file():
            continue
        stem = f.stem
        if stem.lower().startswith("lib"):
            stem = stem[3:]
        if stem and stem.lower() not in _SUPPORT_LIBS:
            stems[stem.lower()] = stem
    return stems


def get_installed_plugins() -> set[str]:
    """Set of plugin library names shipped with the installed LMMS.

    Names are lowercase library basenames without extension (e.g.
    "tripleoscillator", "reverbsc"). On Linux (.so files) the leading
    "lib" prefix is stripped, so libOPL2.so becomes "opl2"; support
    libraries such as libZynAddSubFxCore.so are excluded. Only files
    directly in the plugins dir are considered, so the ladspa/
    subdirectory is ignored. Returns an empty set if no plugins are
    found. See get_installed_stems for the original-case names.
    """
    return set(get_installed_stems())


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


# Plugins that official Windows builds link into the main binary instead
# of shipping a library. Only trusted on Windows: on Linux a plugin must
# have a library, otherwise LMMS loads nothing and the track is silent.
STATIC_PLUGINS = {
    "freeboy",
}

# MCP-facing instrument name -> library names (lowercase) that provide it
# in some LMMS version/platform, preferred first. The name that is written
# into the project XML is the original-case stem of the first one that is
# installed (see resolve_plugin), e.g. opulenz -> OPL2 in LMMS 1.2.x
# (libOPL2.so) but opulenz in 1.3 (libopulenz.so); freeboy -> papu on
# Linux 1.2.x, where the FreeBoy instrument is the library libpapu.so.
# nes (Nescaline, libnes.so) is a different instrument from papu and never
# resolves to it.
PLUGIN_ALIASES = {
    "nes": ("nes", "nescaline"),
    "freeboy": ("freeboy", "papu"),
    "malletsstk": ("malletsstk", "stk"),
    "audiofileprocessor": ("audiofileprocessor",),
    "sf2player": ("sf2player", "fluidsynth"),
    "opulenz": ("opulenz", "opl2"),  # named OPL2 in LMMS 1.2.x
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


def _is_windows(windows: bool | None) -> bool:
    return sys.platform == "win32" if windows is None else windows


def plugin_candidates(name: str) -> tuple[str, ...]:
    """Library names (lowercase) that can provide a plugin, preferred first."""
    name = name.lower()
    return (name, *(a for a in PLUGIN_ALIASES.get(name, ()) if a != name))


def resolve_plugin(
    name: str,
    installed: Collection[str],
    stems: Mapping[str, str] | None = None,
    windows: bool | None = None,
) -> str | None:
    """The plugin name to write into a project for an MCP-facing name.

    Returns the original-case stem (from `stems`, a lowercase -> stem
    mapping; the lowercase name if unknown) of the first installed library
    among plugin_candidates(name), e.g. opulenz -> "OPL2" when libOPL2.so
    is installed. A statically linked plugin (STATIC_PLUGINS) with no
    library resolves to its own name on Windows only. Returns None if the
    installed LMMS cannot load the plugin.
    """
    for candidate in plugin_candidates(name):
        if candidate in installed:
            return (stems or {}).get(candidate, candidate)
    if name.lower() in STATIC_PLUGINS and _is_windows(windows):
        return name.lower()
    return None


def _installed_match(
    name: str, installed: Collection[str], windows: bool | None = None
) -> str | None:
    """How a lowercase plugin name is satisfied by the installed set.

    Returns "installed", "installed (alias)", "built-in" or None.
    """
    for candidate in plugin_candidates(name):
        if candidate in installed:
            return "installed" if candidate == name else "installed (alias)"
    if name in STATIC_PLUGINS and _is_windows(windows):
        return "built-in"
    return None


def check_plugin_available(plugin_name: str) -> tuple[bool, str]:
    """Check whether a plugin exists in the installed LMMS.

    Returns (available, reason). If no LMMS installation is detected,
    everything is considered available (cannot verify).
    """
    name = plugin_name.lower()
    installed = get_installed_plugins()
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
    known: Iterable[str],
    installed: Collection[str],
    windows: bool | None = None,
) -> set[str]:
    """Which of the known plugin names the given installed set provides.

    Aliases (opulenz <-> opl2, freeboy <-> papu, ...) are respected;
    statically linked plugins only count on Windows. An empty installed
    set means no LMMS installation was found, so nothing can be verified
    and every known name is returned.
    """
    known = {name.lower() for name in known}
    if not installed:
        return known
    return {name for name in known if _installed_match(name, installed, windows)}


@dataclass(frozen=True)
class PluginAvailability:
    """Snapshot of what the installed LMMS offers.

    verified: False if no installation/plugins were found (then
        instruments and effects hold every known name).
    instruments / effects: the known instrument / effect names that are
        available (all known names when not verified).
    installed: raw plugin library names found (empty when not verified).
    stems: installed library name (lowercase) -> stem in its original
        case, e.g. {"opl2": "OPL2"}.
    plugin_names: available known instrument -> the plugin name LMMS
        loads for it and that is written into projects, e.g. opulenz ->
        "OPL2", freeboy -> "papu" (see resolve_plugin).
    files: the soundfonts, GIG libraries and GUS patches found.
    needs_file: file instruments (FILE_INSTRUMENTS) whose plugin is
        installed but for which no file was found. They are left out of
        `instruments`: without a file they could only make silent tracks.
    """

    verified: bool
    version: str | None
    installed: frozenset[str]
    instruments: frozenset[str]
    effects: frozenset[str]
    stems: Mapping[str, str] = field(default_factory=dict)
    plugin_names: Mapping[str, str] = field(default_factory=dict)
    files: InstrumentFiles = field(default_factory=InstrumentFiles)
    needs_file: frozenset[str] = frozenset()

    def plugin_name(self, name: str) -> str:
        """Plugin name to write into the project for an instrument name.

        Known instruments use their resolved name, any other installed
        library its original-case stem. Without a resolution (nothing
        could be verified) the name itself is returned.
        """
        name = name.lower()
        return self.plugin_names.get(name) or self.stems.get(name, name)


def build_availability(
    known_instruments: Iterable[str],
    known_effects: Iterable[str],
    installed: Collection[str],
    version: str | None,
    stems: Mapping[str, str] | None = None,
    windows: bool | None = None,
    files: InstrumentFiles | None = None,
) -> PluginAvailability:
    """Build an availability snapshot from an installed-plugin set.

    stems maps the lowercase installed names to their original-case
    library stems (see get_installed_stems); names it lacks are used in
    lowercase. windows overrides the platform check for statically linked
    plugins (default: the running platform). files are the soundfonts, GIG
    libraries and GUS patches found (default: none): a file instrument
    without a file is not offered (see PluginAvailability.needs_file).
    """
    installed = frozenset(installed)
    stems = {k: v for k, v in (stems or {}).items() if k in installed}
    files = files if files is not None else InstrumentFiles()
    instruments = compute_available(known_instruments, installed, windows)
    needs_file = frozenset(
        n for n in instruments
        if n in FILE_INSTRUMENTS and not files.for_instrument(n)
    )
    instruments -= needs_file
    return PluginAvailability(
        verified=bool(installed),
        version=version,
        installed=installed,
        instruments=frozenset(instruments),
        effects=frozenset(compute_available(known_effects, installed, windows)),
        stems=stems,
        plugin_names={
            name: resolve_plugin(name, installed, stems, windows) or name
            for name in sorted(instruments)
        },
        files=files,
        needs_file=needs_file,
    )


def detect_availability(
    known_instruments: Iterable[str],
    known_effects: Iterable[str],
    files: InstrumentFiles | None = None,
) -> PluginAvailability:
    """Inspect the installed LMMS and build the availability snapshot.

    files default to what find_instrument_files finds on this machine.
    """
    installed = get_installed_plugins()
    version = get_lmms_version() if installed else None
    stems = get_installed_stems() if installed else {}
    if files is None:
        files = find_instrument_files()
    return build_availability(
        known_instruments, known_effects, installed, version, stems,
        files=files,
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
