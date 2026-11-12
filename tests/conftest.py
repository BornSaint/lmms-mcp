"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def complete_lmms_install(monkeypatch, tmp_path):
    """Make server tests independent of the LMMS installed on this machine.

    The server computes plugin availability once at import from the real
    installation. Replace that snapshot with a complete LMMS (every plugin
    the server knows about, plus the legacy custom plugin "papu"), so
    tests behave the same with any, or no, LMMS installed. Every plugin is
    installed under its own lowercase name, so instruments are written into
    projects under the names the server offers; Windows-only built-ins are
    off. The machine also has a soundfont, a GIG library and a GUS patch
    (empty stand-in files in tmp_path/instrument_files; sf2player starts
    with the soundfont), so every file instrument is offered. Tests that
    need a specific install (library file names with
    their case, as on Linux) replace the snapshot again (see
    TestPluginAvailability and TestInstrumentPluginNames).
    """
    from lmms_mcp import lmms_app
    from lmms_mcp import server as srv

    installed = (
        set(srv.KNOWN_INSTRUMENTS) | set(srv.effects_mod.KNOWN_EFFECTS)
        | {"papu"}
    )
    monkeypatch.setattr(
        srv, "_availability",
        lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
            installed, "1.3.0", windows=False,
            files=fake_instrument_files(tmp_path / "instrument_files"),
        ),
    )


def fake_instrument_files(directory):
    """InstrumentFiles with one empty stand-in file of each kind in a folder."""
    from lmms_mcp import lmms_app

    directory.mkdir(parents=True, exist_ok=True)
    found = {}
    for instrument, name in (
        ("sf2player", "default-GM.sf2"), ("gigplayer", "piano.gig"),
        ("patman", "flute.pat"),
    ):
        (directory / name).write_bytes(b"")
        found[instrument] = (str(directory / name),)
    return lmms_app.InstrumentFiles(
        files=found,
        searched={k: (str(directory),) for k in found},
        put_in={k: str(directory) for k in found},
        default_soundfont=found["sf2player"][0],
    )


@pytest.fixture
def server_for(monkeypatch, tmp_path):
    """Reload the server module against a fake install (what an agent
    sees at startup: instructions, tool list, descriptions). The fake
    machine has a soundfont, a GIG library and a GUS patch unless
    `files` (an InstrumentFiles) says otherwise."""
    import importlib

    from lmms_mcp import lmms_app
    from lmms_mcp import server as srv

    def make(installed, version="1.2.2", stems=None, files=None):
        if files is None:
            files = fake_instrument_files(tmp_path / "instrument_files")
        monkeypatch.setattr(
            lmms_app, "get_installed_plugins", lambda: set(installed))
        monkeypatch.setattr(
            lmms_app, "get_installed_stems",
            lambda: dict(stems) if stems else {n: n for n in installed})
        monkeypatch.setattr(
            lmms_app, "get_lmms_version", lambda: version)
        monkeypatch.setattr(
            lmms_app, "find_instrument_files", lambda: files)
        return importlib.reload(srv)

    yield make
    monkeypatch.undo()
    importlib.reload(srv)
