"""Tests for the guard around every tool: an unexpected exception must reach
the client as a JSON error with its type and message (the mcp library drops
the message of an exception that escapes a tool)."""

import asyncio
import inspect
import json

import pytest
from mcp.server import MCPServer


def _srv():
    from lmms_mcp import server as srv
    return srv


def _boom(x: int = 1, label: str = "a") -> str:
    """Dummy tool that always fails.

    Args:
        x: A number
        label: A label
    """
    raise ValueError(f"boom {x} {label}")


class TestGuardTool:
    def test_exception_becomes_json_error(self):
        guarded = _srv()._guard_tool(_boom)
        result = json.loads(guarded(3, label="z"))
        assert result["error"] == "ValueError: boom 3 z"
        assert result["tool"] == "_boom"
        assert "get_project_info" in result["hint"]
        assert set(result) == {"error", "tool", "hint"}

    def test_exception_type_is_named_for_any_exception(self):
        def tool_index() -> str:
            return [][0]

        result = json.loads(_srv()._guard_tool(tool_index)())
        assert result["error"] == "IndexError: list index out of range"
        assert result["tool"] == "tool_index"

    def test_exception_without_message_still_names_its_type(self):
        def tool_empty() -> str:
            raise RuntimeError()

        result = json.loads(_srv()._guard_tool(tool_empty)())
        assert result["error"] == "RuntimeError"

    def test_result_passes_through(self):
        def tool_ok(a: int) -> str:
            return json.dumps({"a": a})

        assert _srv()._guard_tool(tool_ok)(5) == '{"a": 5}'

    class _Stop(BaseException):
        pass

    @pytest.mark.parametrize("exc", [
        KeyboardInterrupt, SystemExit, asyncio.CancelledError, _Stop,
        GeneratorExit,
    ])
    def test_base_exceptions_propagate(self, exc):
        def tool_stop() -> str:
            raise exc()

        with pytest.raises(exc):
            _srv()._guard_tool(tool_stop)()

    def test_traceback_goes_to_stderr_not_stdout(self, capsys):
        _srv()._guard_tool(_boom)()
        out, err = capsys.readouterr()
        assert out == ""
        assert "Traceback" in err
        assert "ValueError: boom 1 a" in err
        assert "_boom" in err

    def test_broken_stderr_does_not_hide_the_error(self, monkeypatch):
        class Broken:
            def write(self, _):
                raise OSError("stderr is closed")

            def flush(self):
                raise OSError("stderr is closed")

        monkeypatch.setattr("sys.stderr", Broken())
        result = json.loads(_srv()._guard_tool(_boom)())
        assert result["error"] == "ValueError: boom 1 a"

    def test_wrapper_keeps_name_doc_and_signature(self):
        guarded = _srv()._guard_tool(_boom)
        assert guarded is not _boom
        assert guarded.__name__ == "_boom"
        assert guarded.__doc__ == _boom.__doc__
        assert guarded.__wrapped__ is _boom
        assert inspect.signature(guarded) == inspect.signature(_boom)
        assert inspect.signature(guarded, eval_str=True).return_annotation is str

    def test_async_tool_is_guarded_and_stays_async(self):
        async def tool_async(n: int) -> str:
            if n < 0:
                raise KeyError("negative")
            return "ok"

        guarded = _srv()._guard_tool(tool_async)
        assert inspect.iscoroutinefunction(guarded)
        assert asyncio.run(guarded(1)) == "ok"
        result = json.loads(asyncio.run(guarded(-1)))
        assert result["error"] == "KeyError: 'negative'"
        assert result["tool"] == "tool_async"

    def test_async_tool_propagates_cancellation(self):
        async def tool_cancelled() -> str:
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(_srv()._guard_tool(tool_cancelled)())


class TestToolDecorator:
    """_tool registers the guarded function and returns the original."""

    def test_registers_guarded_tool_and_returns_original(self, monkeypatch):
        srv = _srv()
        server = MCPServer("test")
        monkeypatch.setattr(srv, "mcp", server)
        decorated = srv._tool(_boom)
        assert decorated is _boom
        with pytest.raises(ValueError, match="boom"):
            decorated()  # a direct call still raises
        res = asyncio.run(server.call_tool("_boom", {"x": 7}))
        assert not res.is_error
        payload = json.loads(res.content[0].text)
        assert payload["error"] == "ValueError: boom 7 a"
        assert payload["tool"] == "_boom"

    def test_plugin_tool_is_guarded_too(self, monkeypatch):
        srv = _srv()
        server = MCPServer("test")
        monkeypatch.setattr(srv, "mcp", server)

        def tool_plugins() -> str:
            """Lists $INSTRUMENTS."""
            raise RuntimeError("no luck")

        srv._plugin_tool(tool_plugins)
        res = asyncio.run(server.call_tool("tool_plugins", {}))
        assert json.loads(res.content[0].text)["error"] == "RuntimeError: no luck"
        assert "$INSTRUMENTS" not in tool_plugins.__doc__

    def test_unexpected_exception_in_a_real_tool_reaches_the_client(
            self, monkeypatch):
        srv = _srv()

        def broken_project():
            raise RuntimeError("project store exploded")

        monkeypatch.setattr(srv, "get_project", broken_project)
        res = asyncio.run(srv.mcp.call_tool("get_project_info", {}))
        assert not res.is_error
        payload = json.loads(res.content[0].text)
        assert payload["error"] == "RuntimeError: project store exploded"
        assert payload["tool"] == "get_project_info"
        assert payload["hint"]

    def test_deliberate_value_error_message_reaches_the_client(self):
        # Some tools raise ValueError("...") with an explanation (set_tempo
        # rejects a BPM out of range); the mcp library would drop it.
        srv = _srv()
        srv.create_project()
        with pytest.raises(ValueError, match="BPM must be between"):
            srv.set_tempo(0)
        res = asyncio.run(srv.mcp.call_tool("set_tempo", {"bpm": 0}))
        assert not res.is_error
        payload = json.loads(res.content[0].text)
        assert payload["tool"] == "set_tempo"
        assert payload["error"] == "ValueError: BPM must be between 10 and 999"

    def test_index_error_message_reaches_the_client(self):
        srv = _srv()
        srv.create_project()
        res = asyncio.run(srv.mcp.call_tool(
            "remove_track", {"track_index": 999999}))
        payload = json.loads(res.content[0].text)
        assert payload["tool"] == "remove_track"
        assert payload["error"].startswith("IndexError: ")
        assert "999999" in payload["error"]


class TestEveryToolIsGuarded:
    def _registered(self):
        return _srv().mcp._tool_manager.list_tools()

    def test_every_registered_tool_is_the_guarded_wrapper(self):
        srv = _srv()
        tools = self._registered()
        assert len(tools) > 50
        for tool in tools:
            original = getattr(srv, tool.name)
            assert tool.fn is not original, tool.name
            assert tool.fn.__wrapped__ is original, tool.name

    def test_no_tool_is_registered_around_the_guard(self):
        # Registering with @mcp.tool() directly would bypass the guard.
        import re
        from pathlib import Path
        source = Path(_srv().__file__).read_text(encoding="utf-8")
        assert not re.search(r"^\s*@mcp\.tool\(", source, re.M)

    def test_schemas_equal_those_of_the_unwrapped_functions(self):
        """Name, description and input/output schema of every registered
        tool equal what MCPServer builds from the plain function."""
        srv = _srv()
        registered = asyncio.run(srv.mcp.list_tools())
        assert len(registered) > 50

        plain = MCPServer("plain")
        for tool in registered:
            plain.add_tool(getattr(srv, tool.name))
        expected = asyncio.run(plain.list_tools())

        def dump(tools):
            return sorted(
                (t.model_dump(mode="json", by_alias=True) for t in tools),
                key=lambda d: d["name"])

        assert dump(registered) == dump(expected)
        for tool in registered:
            assert tool.description == getattr(srv, tool.name).__doc__
            assert tool.input_schema["type"] == "object"
