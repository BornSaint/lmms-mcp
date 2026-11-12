"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def complete_lmms_install(monkeypatch):
    """Make server tests independent of the LMMS installed on this machine.

    The server computes plugin availability once at import from the real
    installation. Replace that snapshot with a complete LMMS (every plugin
    the server knows about, plus the legacy custom plugin "papu"), so
    tests behave the same with any, or no, LMMS installed. Every plugin is
    installed under its own lowercase name, so instruments are written into
    projects under the names the server offers; Windows-only built-ins are
    off. Tests that need a specific install (library file names with
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
        ),
    )
