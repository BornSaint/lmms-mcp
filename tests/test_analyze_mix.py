"""analyze_mix: renders the mix and every track (LMMS replaced by
FakeMixLmms, tests/audio_helpers.py, which writes synthetic float stems the
way LMMS 1.2.2 does) and reports the balance. What is checked is what the
tool does with the files and the project: numbers, active bars, sections,
flags and their suggestions, the mapping of files to tracks, the cleanup of
the temporary files and the errors."""

import asyncio
import inspect
import json
import math
import re
import subprocess
from pathlib import Path

import pytest

from audio_helpers import FakeMixLmms, SAMPLERATE, silence, sine
from lmms_mcp import audio_analysis, lmms_app, mix_analysis, xml_parser
from lmms_mcp.project import LMMSProject

BAR = 2.0   # seconds of a bar at 120 bpm in 4/4


def tone(freq, amp, bars):
    """`bars` bars of a sine (250, 500 and 1000 Hz have whole periods at
    8 kHz)."""
    return sine(freq, amp, bars * BAR)


def rest(bars):
    return silence(bars * BAR)


def db(x):
    return 20 * math.log10(x)


def setup(monkeypatch, tmp_path, signals, order=None, **fake_kwargs):
    """A saved project with an instrument track for every key of `signals`
    (or of `order`) and the fake LMMS. Returns (server, project, fake)."""
    from lmms_mcp import server as srv
    proj = LMMSProject()
    proj.new(bpm=120)
    srv.set_project(proj)
    for name in (order or signals):
        srv.add_instrument_track(name, "tripleoscillator")
    proj.save(str(tmp_path / "song.mmp"))
    monkeypatch.setattr(lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
    fake = FakeMixLmms(signals, **fake_kwargs)
    monkeypatch.setattr("subprocess.run", fake)
    return srv, proj, fake


def analyze(srv, **kwargs):
    return json.loads(srv.analyze_mix(**kwargs))


def track(result, name):
    [t] = [t for t in result["tracks"] if t["name"] == name]
    return t


def flags(result, kind):
    return [f for f in result["flags"] if f["type"] == kind]


def volume_call(text):
    """(track index, volume) of the set_track_volume call a suggestion
    starts with."""
    m = re.match(r"set_track_volume\((\d+), (\d+)\)", text)
    return int(m.group(1)), int(m.group(2))


class TestNumbers:
    SIGNALS = {
        "Kick": tone(250, 0.5, 8),                 # -9.0 dBFS RMS, -6.0 peak
        "Lead": rest(2) + tone(500, 0.1, 6),       # enters at bar 2
        "Pad": tone(1000, 0.1, 8),
    }

    def test_levels_of_the_tracks_the_mix_and_the_sections(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, self.SIGNALS)
        result = analyze(srv, sections=[
            {"name": "intro", "bars": [0, 2]},
            {"name": "main", "bars": [2, 8]}])
        assert result["bars"] == 8 and result["duration_s"] == 16.0
        assert result["oversampling"] == 1
        assert result["isolated_config"] is True
        assert isinstance(result["elapsed_s"], float)

        kick = track(result, "Kick")
        assert (kick["index"], kick["instrument"]) == (0, "tripleoscillator")
        assert kick["volume"] == 100.0
        assert kick["peak_dbfs"] == -6.0 and kick["rms_dbfs"] == -9.0
        assert kick["active_bars"] == [[0, 8]]
        assert kick["sections"]["main"]["rms_dbfs"] == -9.0
        assert kick["sections"]["main"]["peak_dbfs"] == -6.0
        lead = track(result, "Lead")
        assert lead["index"] == 1
        assert lead["peak_dbfs"] == -20.0 and lead["rms_dbfs"] == -23.0

        # The mix is the sum of the three stems, as the fake renders it.
        total = [sum(v) for v in zip(*[
            [x for x in self.SIGNALS[n]] for n in ("Kick", "Lead", "Pad")])]
        peak = max(abs(x) for x in total)
        rms = math.sqrt(sum(x * x for x in total) / len(total))
        mix = result["mix"]
        assert mix["peak_dbfs"] == round(db(peak), 1)
        assert mix["rms_dbfs"] == round(db(rms), 1)
        assert mix["clipped_samples"] == 0
        assert [s["name"] for s in mix["sections"]] == ["intro", "main"]
        assert mix["sections"][0]["bars"] == [0, 2]
        bar_frames = int(BAR * SAMPLERATE) * 2
        intro = total[:2 * bar_frames]
        assert mix["sections"][0]["rms_dbfs"] == round(
            db(math.sqrt(sum(x * x for x in intro) / len(intro))), 1)

    def test_share_of_the_energy_in_a_section(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, self.SIGNALS)
        result = analyze(srv, sections=[{"name": "main", "bars": [2, 8]}])
        shares = {t["name"]: t["sections"]["main"]["share_pct"]
                  for t in result["tracks"]}
        # energies are amp^2 / 2 per sample: 0.125, 0.005, 0.005
        assert shares == {"Kick": 92.6, "Lead": 3.7, "Pad": 3.7}

    def test_active_bars_and_silent_sections(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, self.SIGNALS)
        result = analyze(srv, sections=[
            {"name": "intro", "bars": [0, 2]},
            {"name": "main", "bars": [2, 8]}])
        lead = track(result, "Lead")
        assert lead["active_bars"] == [[2, 8]]
        assert lead["silent_in"] == ["intro"]
        assert list(lead["sections"]) == ["main"]
        assert "silent_in" not in track(result, "Kick")

    def test_default_sections_are_blocks_of_four_bars(
            self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, self.SIGNALS)
        result = analyze(srv)
        assert [(s["name"], s["bars"]) for s in result["mix"]["sections"]] == [
            ("bars 0-4", [0, 4]), ("bars 4-8", [4, 8])]

    def test_sections_that_overlap_or_leave_gaps(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, self.SIGNALS)
        result = analyze(srv, sections=[
            {"name": "all", "bars": [0, 8]}, {"name": "late", "bars": [6, 8]}])
        assert [s["bars"] for s in result["mix"]["sections"]] == [
            [0, 8], [6, 8]]

    def test_an_end_past_the_song_is_cut_and_noted(
            self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, self.SIGNALS)
        result = analyze(srv, sections=[
            {"name": "a", "bars": [0, 4]}, {"name": "outro", "bars": [4, 12]},
            {"name": "never", "bars": [20, 24]}])
        assert [s["bars"] for s in result["mix"]["sections"]] == [
            [0, 4], [4, 8]]
        assert any("'outro'" in n and "[4, 8]" in n for n in result["notes"])
        assert any("'never'" in n and "left out" in n for n in result["notes"])

    def test_no_section_inside_the_song_is_an_error(
            self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, self.SIGNALS)
        result = analyze(srv, sections=[{"name": "far", "bars": [20, 24]}])
        assert "None of the sections is inside the song" in result["error"]
        assert result["bars"] == 8

    def test_clipped_samples_of_the_mix_are_counted(
            self, monkeypatch, tmp_path):
        signals = {"A": tone(250, 0.7, 4), "B": tone(250, 0.5, 4)}
        srv, _, _ = setup(monkeypatch, tmp_path, signals)
        mix = analyze(srv)["mix"]
        assert mix["peak_dbfs"] == round(db(1.2), 1)
        assert mix["clipped_samples"] > 100

    def test_the_response_is_compact(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, self.SIGNALS)
        text = srv.analyze_mix()
        assert text == json.dumps(json.loads(text), separators=(",", ":"))
        assert len(text) < 3000


class TestBadArguments:
    def test_bad_sections_are_refused_before_any_render(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        for sections in ([], "hook", [{"name": "a", "bars": [4, 4]}],
                         [{"name": "a", "bars": [0, 4.5]}],
                         [{"name": "a", "bars": [0, 4]},
                          {"name": "a", "bars": [4, 8]}],
                         [{"name": "a", "bars": [-2, 4]}],
                         [{"name": "a"}]):
            result = analyze(srv, sections=sections)
            assert result["error"], sections
            assert set(result) == {"error"}
        assert fake.calls == []

    def test_the_error_for_a_bad_range_explains_the_convention(
            self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        error = analyze(srv, sections=[{"name": "hook", "bars": [20, 12]}])[
            "error"]
        assert "'hook'" in error and "exclusive" in error

    @pytest.mark.parametrize("value", [0, 3, True, "2", 16])
    def test_invalid_oversampling(self, monkeypatch, tmp_path, value):
        srv, _, fake = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        result = analyze(srv, oversampling=value)
        assert "Invalid oversampling" in result["error"]
        assert fake.calls == []

    def test_a_project_that_was_never_saved(self, monkeypatch, tmp_path):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new(bpm=120)
        srv.set_project(proj)
        srv.add_instrument_track("A", "tripleoscillator")
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        fake = FakeMixLmms({"A": tone(250, 0.1, 4)})
        monkeypatch.setattr("subprocess.run", fake)
        result = analyze(srv)
        assert result["error"] == (
            "Project has never been saved. Call save_project first.")
        assert fake.calls == []

    def test_no_lmms(self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        monkeypatch.setattr(lmms_app, "find_lmms_exe", lambda: None)
        assert "LMMS executable not found" in analyze(srv)["error"]
        assert fake.calls == []

    def test_a_project_without_unmuted_tracks(self, monkeypatch, tmp_path):
        srv, proj, fake = setup(
            monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        srv.mute_track(0, True)
        result = analyze(srv)
        assert "no track to analyze" in result["error"]
        assert "muted" in result["error"]
        assert result["skipped_tracks"] == [
            {"index": 0, "name": "A", "reason": "muted"}]
        assert fake.calls == []

    def test_a_silent_mix(self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, {"A": rest(4)})
        result = analyze(srv)
        assert "The mix is silent" in result["error"]
        # nothing to analyze: the track renders are not made
        assert [c["command"] for c in fake.calls] == ["render"]


class TestRenders:
    def test_one_mix_render_then_one_render_of_all_tracks(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, {
            "A": tone(250, 0.1, 4), "B": tone(500, 0.1, 4)})
        result = analyze(srv, oversampling=2)
        assert "error" not in result
        assert [c["command"] for c in fake.calls] == ["render", "rendertracks"]
        for call in fake.calls:
            cmd = call["cmd"]
            assert call["float"]            # -a: levels above 0 dBFS stay
            assert cmd[cmd.index("-x") + 1] == "2"
            assert cmd[cmd.index("-f") + 1] == "wav"
            assert cmd[cmd.index("-s") + 1] == "44100"
            assert cmd[cmd.index(call["command"]) + 1] == str(
                tmp_path / "song.mmp")
        assert result["oversampling"] == 2

    def test_default_oversampling_is_1_like_the_gui(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        analyze(srv)
        for call in fake.calls:
            assert call["cmd"][call["cmd"].index("-x") + 1] == "1"
        assert inspect.signature(srv.analyze_mix).parameters[
            "oversampling"].default == 1
        assert inspect.signature(srv.analyze_mix).parameters[
            "sections"].default is None

    def test_runs_in_a_private_config_through_the_lmms_lock(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        locked = []
        real = lmms_app.run_lmms

        def spy(cmd, **kwargs):
            locked.append(lmms_app._LMMS_RUN_LOCK.locked() is False)
            return real(cmd, **kwargs)

        monkeypatch.setattr(lmms_app, "run_lmms", spy)
        analyze(srv)
        assert len(locked) == 2
        for call in fake.calls:
            assert Path(call["cmd"][call["cmd"].index("--config") + 1]).name \
                == "lmmsrc.xml"
            assert str(Path.home() / ".lmmsrc.xml") not in call["cmd"]
            assert "disableautoquit" in call["config_text"]

    def test_rendertracks_gets_a_time_limit_that_grows_with_the_tracks(
            self, monkeypatch, tmp_path):
        signals = {f"T{i}": tone(250, 0.1, 4) for i in range(5)}
        srv, _, fake = setup(monkeypatch, tmp_path, signals)
        analyze(srv)
        mix_call, tracks_call = fake.calls
        assert mix_call["kwargs"]["timeout"] == 300
        assert tracks_call["kwargs"]["timeout"] == 300 + 60 * 5

    def test_the_project_is_saved_first_and_not_changed(
            self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, {
            "A": tone(250, 0.2, 4), "B": tone(500, 0.2, 4)})
        srv.set_track_volume(0, 50)       # unsaved change
        assert proj.modified
        result = analyze(srv)
        # the fake read the saved file: A plays at half level
        assert track(result, "A")["volume"] == 50.0
        assert track(result, "A")["peak_dbfs"] == round(db(0.1), 1)
        assert not proj.modified
        before = (tmp_path / "song.mmp").read_text()
        analyze(srv)
        assert (tmp_path / "song.mmp").read_text() == before

    def test_a_compressed_project_stays_compressed(
            self, monkeypatch, tmp_path):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new(bpm=120)
        srv.set_project(proj)
        srv.add_instrument_track("A", "tripleoscillator")
        proj.save(str(tmp_path / "song.mmpz"), compressed=True)
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        monkeypatch.setattr(
            "subprocess.run", FakeMixLmms({"A": tone(250, 0.1, 4)}))
        assert "error" not in analyze(srv)
        # still a valid compressed project LMMS (and the fake) can read
        assert xml_parser.load_project(tmp_path / "song.mmpz") is not None

    def test_a_project_that_cannot_be_saved(self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})

        def refuse(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(proj, "save", refuse)
        result = analyze(srv)
        assert "Could not save the project" in result["error"]
        assert fake.calls == []


class TestTemporaryFiles:
    def test_every_temporary_file_is_removed(self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, {
            "A": tone(250, 0.1, 4), "B": tone(500, 0.1, 4)})
        before = sorted(p.name for p in tmp_path.iterdir())
        analyze(srv)
        mix_file, stems_dir = fake.outputs
        assert mix_file.name == "mix.wav" and stems_dir.name == "stems"
        assert not mix_file.exists() and not stems_dir.exists()
        assert not mix_file.parent.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    @pytest.mark.parametrize("failing", ["render", "rendertracks"])
    def test_removed_when_a_render_fails(self, monkeypatch, tmp_path, failing):
        srv, _, fake = setup(
            monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)}, fail=failing)
        result = analyze(srv)
        assert "Could not load project: broken" in result["error"]
        assert not fake.outputs[0].parent.exists()

    def test_removed_when_a_render_times_out(self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})

        def slow(cmd, **kwargs):
            if "rendertracks" in cmd:
                raise subprocess.TimeoutExpired(cmd, 360)
            return fake(cmd, **kwargs)

        monkeypatch.setattr("subprocess.run", slow)
        result = analyze(srv)
        assert "timed out after 360 s" in result["error"]
        assert not fake.outputs[0].parent.exists()

    def test_the_track_files_are_deleted_as_they_are_read(
            self, monkeypatch, tmp_path):
        signals = {f"T{i}": tone(250, 0.1, 4) for i in range(3)}
        srv, _, fake = setup(monkeypatch, tmp_path, signals)
        seen = []
        real = audio_analysis.read_wav

        def spy(path):
            seen.append(sorted(p.name for p in Path(path).parent.iterdir()))
            return real(path)

        monkeypatch.setattr(audio_analysis, "read_wav", spy)
        analyze(srv)
        # mix.wav first (it is deleted once read), then the 3 track files,
        # one fewer each time
        assert seen[0] == ["mix.wav"]
        assert [len(s) for s in seen[1:]] == [3, 2, 1]

    def test_not_enough_disk_space_is_refused_before_the_track_renders(
            self, monkeypatch, tmp_path):
        import collections
        srv, _, fake = setup(monkeypatch, tmp_path, {
            "A": tone(250, 0.1, 4), "B": tone(500, 0.1, 4)})
        usage = collections.namedtuple("usage", "total used free")
        monkeypatch.setattr(
            "shutil.disk_usage", lambda path: usage(10**9, 10**9 - 1000, 1000))
        result = analyze(srv)
        assert "Not enough free disk space" in result["error"]
        assert "TMPDIR" in result["error"]
        assert [c["command"] for c in fake.calls] == ["render"]
        assert not fake.outputs[0].parent.exists()


class TestFailures:
    def test_the_mix_render_fails(self, monkeypatch, tmp_path):
        srv, _, fake = setup(
            monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)}, fail="render")
        result = analyze(srv)
        assert result["error"].startswith("Render failed: ")
        assert "Could not load project: broken" in result["error"]
        assert "command" in result
        assert [c["command"] for c in fake.calls] == ["render"]

    def test_the_mix_render_writes_no_file(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)},
                          write_nothing="render")
        assert "without creating" in analyze(srv)["error"]

    def test_the_track_render_fails(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)},
                          fail="rendertracks")
        result = analyze(srv)
        assert "Could not load project: broken" in result["error"]

    def test_the_track_render_writes_no_files(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)},
                          write_nothing="rendertracks")
        assert "LMMS rendered no track files" in analyze(srv)["error"]

    def test_a_track_file_that_cannot_be_read_is_left_out_and_noted(
            self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, {
            "Good": tone(250, 0.1, 4), "Bad": tone(500, 0.1, 4)})
        real = audio_analysis.read_wav

        def read(path):
            if Path(path).name.startswith("2_"):
                raise audio_analysis.AudioError("not a RIFF/WAVE file")
            return real(path)

        monkeypatch.setattr(audio_analysis, "read_wav", read)
        result = analyze(srv)
        assert [t["name"] for t in result["tracks"]] == ["Good"]
        assert any("Could not read the render of 'Bad'" in n
                   for n in result["notes"])

    def test_an_unexpected_exception_is_a_json_error_through_the_guard(
            self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        monkeypatch.setattr(
            mix_analysis, "analyze_balance",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        guarded = srv._guard_tool(srv.analyze_mix)
        assert "RuntimeError: boom" in json.loads(guarded())["error"]


class TestTrackMapping:
    def build(self, monkeypatch, tmp_path, **fake_kwargs):
        """Lead, automation, muted pad, 'Ba/ss é+1', pattern track, sample
        track and a second Lead: LMMS renders 4 of them."""
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new(bpm=120)
        srv.set_project(proj)
        srv.add_instrument_track("Lead", "tripleoscillator")      # 0
        srv.add_automation_track("Auto")                          # 1
        srv.add_instrument_track("Muted Pad", "tripleoscillator")  # 2
        srv.add_instrument_track("Ba/ss é+1", "tripleoscillator")  # 3
        srv.add_pattern_track("Beat")                             # 4
        srv.add_sample_track("Smp")                               # 5
        srv.add_instrument_track("Lead", "tripleoscillator")      # 6
        srv.mute_track(2, True)
        proj.save(str(tmp_path / "song.mmp"))
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        fake = FakeMixLmms({
            "Lead": tone(250, 0.1, 4), "Muted Pad": tone(250, 0.3, 4),
            "Ba/ss é+1": tone(500, 0.2, 4), "Smp": tone(1000, 0.05, 4),
        }, **fake_kwargs)
        monkeypatch.setattr("subprocess.run", fake)
        return srv, fake

    def test_files_are_matched_to_the_tracks_that_were_rendered(
            self, monkeypatch, tmp_path):
        srv, fake = self.build(
            monkeypatch, tmp_path,
            pattern_instruments={"Beat": tone(250, 0.3, 4)})
        result = analyze(srv)
        assert [(t["index"], t["name"]) for t in result["tracks"]] == [
            (0, "Lead"), (3, "Ba/ss é+1"), (5, "Smp"), (6, "Lead"),
            (None, "Beat")]
        assert [t["instrument"] for t in result["tracks"]] == [
            "tripleoscillator", "tripleoscillator", "sample track",
            "tripleoscillator", "pattern instrument"]
        assert track(result, "Ba/ss é+1")["peak_dbfs"] == round(db(0.2), 1)
        assert track(result, "Smp")["peak_dbfs"] == round(db(0.05), 1)
        assert track(result, "Beat")["peak_dbfs"] == round(db(0.3), 1)
        assert "volume" not in track(result, "Beat")
        assert result["notes"] == []

    def test_muted_tracks_are_skipped_and_listed(self, monkeypatch, tmp_path):
        srv, _ = self.build(monkeypatch, tmp_path)
        result = analyze(srv)
        assert result["skipped_tracks"] == [
            {"index": 2, "name": "Muted Pad", "reason": "muted"}]
        assert "Muted Pad" not in [t["name"] for t in result["tracks"]]

    def test_a_track_without_a_file_is_noted(self, monkeypatch, tmp_path):
        srv, fake = self.build(monkeypatch, tmp_path)
        original = fake.__call__

        class Dropping:
            """The fake LMMS, but its rendertracks drops the last file."""

            def __call__(self, cmd, **kwargs):
                proc = original(cmd, **kwargs)
                if "rendertracks" in cmd:
                    out = Path(cmd[cmd.index("-o") + 1])
                    last = max(out.iterdir(),
                               key=lambda p: int(p.name.split("_")[0]))
                    last.unlink()
                return proc

        monkeypatch.setattr("subprocess.run", Dropping())
        result = analyze(srv)
        assert [t["index"] for t in result["tracks"]] == [0, 3, 5]
        assert any("Track 6 'Lead' has no file" in n for n in result["notes"])

    def test_a_pattern_instrument_only_project(self, monkeypatch, tmp_path):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new(bpm=120)
        srv.set_project(proj)
        srv.add_pattern_track("Beat")
        proj.save(str(tmp_path / "song.mmp"))
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        monkeypatch.setattr("subprocess.run", FakeMixLmms(
            {}, pattern_instruments={"Beat": tone(250, 0.3, 4)}))
        result = analyze(srv)
        assert [(t["index"], t["name"]) for t in result["tracks"]] == [
            (None, "Beat")]

    def test_the_volume_of_an_automated_track_is_marked(
            self, monkeypatch, tmp_path):
        srv, proj, _ = setup(monkeypatch, tmp_path, {
            "A": tone(250, 0.1, 4), "B": tone(500, 0.1, 4)})
        added = json.loads(srv.add_automation(
            "track", "volume", [{"bar": 0, "value": 80}, {"bar": 4, "value": 100}],
            target_index=1))
        assert "error" not in added
        proj.save(str(tmp_path / "song.mmp"))
        result = analyze(srv)
        assert track(result, "B").get("volume_automated") is True
        assert "volume_automated" not in track(result, "A")

    def test_a_tempo_that_is_automated_is_noted(self, monkeypatch, tmp_path):
        srv, proj, _ = setup(monkeypatch, tmp_path, {"A": tone(250, 0.1, 4)})
        added = json.loads(srv.add_automation(
            "song", "tempo", [{"bar": 0, "value": 120}, {"bar": 2, "value": 130}]))
        assert "error" not in added
        proj.save(str(tmp_path / "song.mmp"))
        result = analyze(srv)
        assert any("tempo is automated" in n for n in result["notes"])


class TestFlagsEndToEnd:
    def test_a_dominant_kick_and_buried_tracks_and_the_fix(
            self, monkeypatch, tmp_path):
        signals = {"Kick": tone(250, 0.2, 8), "Lead": tone(500, 0.03, 8),
                   "Pad": tone(1000, 0.03, 8)}
        srv, _, _ = setup(monkeypatch, tmp_path, signals)
        sections = [{"name": "verse", "bars": [0, 4]},
                    {"name": "hook", "bars": [4, 8]}]
        result = analyze(srv, sections=sections)
        [dominant] = flags(result, "dominant")
        assert (dominant["track"], dominant["name"]) == (0, "Kick")
        assert dominant["sections"] == ["verse", "hook"]
        # Kick -17.0 vs the others' -33.5: 16.5 dB above; to 3 dB: -13.5 dB
        assert dominant["worst"]["above_others_db"] == 16.5
        assert dominant["gain_db"] == -13.5
        index, volume = volume_call(dominant["suggestion"])
        assert (index, volume) == (0, 21)       # 100 * 10^(-13.5/20) = 21.1
        buried = flags(result, "buried")
        assert sorted(f["name"] for f in buried) == ["Lead", "Pad"]
        for flag in buried:
            assert flag["worst"]["below_mix_db"] == pytest.approx(16.5, abs=0.2)
            assert flag["suggestion"].startswith(
                f"set_track_volume({flag['track']}, ")
        # the flags are ordered peak, dominant, buried
        assert [f["type"] for f in result["flags"]] == [
            "dominant", "buried", "buried"]

        # Applying the suggestion fixes all of it: the second run is clean.
        applied = json.loads(srv.set_track_volume(index, volume))
        assert "error" not in applied
        again = analyze(srv, sections=sections)
        assert again["flags"] == []
        assert track(again, "Kick")["volume"] == 21.0
        assert track(again, "Kick")["rms_dbfs"] == pytest.approx(
            -17.0 - 13.5, abs=0.2)

    def test_a_buried_lead_gets_the_volume_that_brings_it_to_12_db(
            self, monkeypatch, tmp_path):
        # Kick -17 dBFS RMS, Lead -33.5: 16.5 dB below the mix (-17.0)
        signals = {"Kick": tone(250, 0.2, 8), "Lead": tone(500, 0.03, 8)}
        srv, _, _ = setup(monkeypatch, tmp_path, signals)
        [flag] = flags(analyze(srv), "buried")
        assert flag["name"] == "Lead"
        mix_rms = math.sqrt((0.2 ** 2 + 0.03 ** 2) / 2)
        gap = db(mix_rms) - db(0.03 / math.sqrt(2))
        assert flag["worst"]["below_mix_db"] == round(gap, 1)
        gain = gap - mix_analysis.BURIED_TARGET_DB
        assert flag["gain_db"] == round(gain, 1)
        want = round(100 * 10 ** (gain / 20))
        assert volume_call(flag["suggestion"]) == (1, want)
        assert 100 < want < 200
        srv.set_track_volume(1, want)
        assert analyze(srv)["flags"] == []

    def test_a_volume_the_track_cannot_reach_says_what_else_to_do(
            self, monkeypatch, tmp_path):
        signals = {"Kick": tone(250, 0.2, 8), "Lead": tone(500, 0.005, 8)}
        srv, _, _ = setup(monkeypatch, tmp_path, signals)
        [flag] = flags(analyze(srv), "buried")
        assert flag["suggestion"].startswith(
            "set_track_volume(1, 200): +6.0 dB, the most the track volume "
            "gives")
        assert "lower the louder tracks" in flag["suggestion"]

    def test_peak_contributors_with_suggestions(self, monkeypatch, tmp_path):
        # In phase: the mix peaks at 1.27 (+2.1 dBFS) where Kick (0.8) and
        # Snare (0.45) peak together; the Hat is far too small to matter.
        signals = {"Kick": tone(250, 0.8, 8), "Snare": tone(250, 0.45, 8),
                   "Hat": tone(250, 0.02, 8)}
        srv, _, _ = setup(monkeypatch, tmp_path, signals)
        result = analyze(srv, sections=[{"name": "hook", "bars": [0, 8]}])
        assert result["mix"]["clipped_samples"] > 0
        [flag] = flags(result, "peak_contributors")
        assert flag["section"] == "hook" and flag["bars"] == [0, 8]
        assert flag["mix_peak_dbfs"] == round(db(1.27), 1)
        assert "at_s" in flag
        assert [c["name"] for c in flag["contributors"]] == ["Kick", "Snare"]
        kick, snare = flag["contributors"]
        assert kick["peak_dbfs"] == round(db(0.8), 1)
        target = 10 ** (mix_analysis.PEAK_TARGET_DBFS / 20)
        factor = 1 - (1.27 - target) / 0.8
        assert volume_call(kick["suggestion"]) == (0, round(100 * factor))
        # removing all of 0.45 would still not take 0.48 off the peak
        assert snare["gain_db"] is None
        assert flag["master_suggestion"].startswith(
            "to bring the peak to -2 dBFS: set_mixer_channel_volume(0, 0.")
        # Lowering the Kick as suggested brings the peak under -1 dBFS.
        srv.set_track_volume(0, round(100 * factor))
        assert flags(analyze(srv), "peak_contributors") == []

    def test_peak_flag_for_the_section_that_peaks_only(
            self, monkeypatch, tmp_path):
        signals = {"A": tone(250, 0.1, 4) + tone(250, 0.7, 4),
                   "B": tone(250, 0.1, 4) + tone(250, 0.5, 4)}
        srv, _, _ = setup(monkeypatch, tmp_path, signals)
        result = analyze(srv, sections=[{"name": "quiet", "bars": [0, 4]},
                                        {"name": "loud", "bars": [4, 8]}])
        assert [f["section"] for f in flags(result, "peak_contributors")] == [
            "loud"]

    def test_a_balanced_mix_has_no_flags(self, monkeypatch, tmp_path):
        signals = {"A": tone(250, 0.1, 8), "B": tone(500, 0.1, 8),
                   "C": tone(1000, 0.1, 8)}
        srv, _, _ = setup(monkeypatch, tmp_path, signals)
        result = analyze(srv)
        assert result["flags"] == []
        assert result["notes"] == []


class TestDocumentation:
    def test_the_docstring_states_the_conventions_and_thresholds(self):
        from lmms_mcp import server as srv
        doc = " ".join(srv.analyze_mix.__doc__.split())
        for text in ("0-based", "exclusive", "1-based and inclusive",
                     "rendertracks", "32-bit float", "skipped_tracks",
                     "dominant", "buried", "peak_contributors", "share_pct",
                     "active_bars", "master_suggestion", "set_track_volume",
                     "add_automation", "no tokens", "one LMMS render per",
                     "-60 dBFS"):
            assert text in doc, text
        # the numbers in the text are the constants
        assert f"more than {mix_analysis.DOMINANT_DB:g} dB above" in doc
        assert f"{mix_analysis.DOMINANT_TARGET_DB:g} dB above them" in doc
        assert f"more than {mix_analysis.BURIED_DB:g} dB below" in doc
        assert f"{mix_analysis.BURIED_TARGET_DB:g} dB below the mix" in doc
        assert f"at or above {mix_analysis.PEAK_NEAR_DBFS:g} dBFS" in doc
        assert f"at {mix_analysis.PEAK_TARGET_DBFS:g} dBFS" in doc
        assert f"within {mix_analysis.PEAK_WINDOW_S * 1000:g} ms" in doc
        assert f"within {mix_analysis.CONTRIBUTOR_DB:g} dB" in doc
        assert f"up to {mix_analysis.MAX_CONTRIBUTORS}" in doc
        assert f"more than {mix_analysis.DEFAULT_MAX_SECTIONS}" in doc
        assert f"volume {mix_analysis.VOLUME_MAX}" in doc

    def test_it_is_a_registered_tool(self):
        from lmms_mcp import server as srv
        tools = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}
        tool = tools["analyze_mix"]
        assert tool.description == srv.analyze_mix.__doc__
        assert set(tool.input_schema["properties"]) == {
            "sections", "oversampling"}
        assert tool.input_schema.get("required", []) == []

    def test_the_readme_has_a_row_for_it(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        row = [line for line in readme.splitlines()
               if line.startswith("| `analyze_mix`")]
        assert len(row) == 1
        assert "sections" in row[0] and "buried" in row[0]
