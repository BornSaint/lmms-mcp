"""render_project: levels, automatic fix of a clipping mix, and the lock
that keeps LMMS processes from running together. LMMS is replaced by
FakeLmms (tests/audio_helpers.py), which "plays" a synthetic signal scaled
by the saved project's Master volume, so what is checked here is what the
server does with the audio and the project."""

import array
import json
import math
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from audio_helpers import FakeLmms, SAMPLERATE, read_pcm, sine
from lmms_mcp import lmms_app, xml_parser
from lmms_mcp.project import LMMSProject

# 4 bars at 120 bpm in 4/4 are 8 seconds. 250 Hz at 8 kHz has whole periods.
CLIPPING = sine(250, 2.0, 8)       # peak +6.02 dBFS
NORMAL = sine(250, 0.5, 8)         # peak -6.02 dBFS, RMS -9.03 dBFS
QUIET = sine(250, 0.02, 8)         # peak -33.98 dBFS


def db(x):
    return 20 * math.log10(x)


def setup(monkeypatch, tmp_path, signal, bpm=120, numerator=4,
          denominator=4, lmms_version=None, master_volume=None):
    from lmms_mcp import server as srv
    proj = LMMSProject()
    proj.new(bpm=bpm, time_sig_numerator=numerator,
             time_sig_denominator=denominator, lmms_version=lmms_version)
    if master_volume is not None:
        channel = xml_parser.ensure_master_channel(proj.root)
        xml_parser.set_model_value(channel, "volume", str(master_volume))
    srv.set_project(proj)
    proj.save(str(tmp_path / "song.mmp"))
    monkeypatch.setattr(lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
    fake = FakeLmms(signal)
    monkeypatch.setattr("subprocess.run", fake)
    return srv, proj, fake


def saved_master_volume(tmp_path):
    return xml_parser.master_volume(
        xml_parser.load_project(tmp_path / "song.mmp"))


def render(srv, tmp_path, name="out.wav", **kwargs):
    return json.loads(srv.render_project(
        output_path=str(tmp_path / name), **kwargs))


class TestLevels:
    def test_levels_of_a_clean_render(self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, NORMAL)
        result = render(srv, tmp_path)
        assert result["success"] is True
        levels = result["levels"]
        assert levels["peak_dbfs"] == -6.0
        assert levels["rms_dbfs"] == -9.0
        assert levels["clipped_samples"] == 0
        assert levels["duration_s"] == 8.0
        block = {"bars": [1, 4], "peak_dbfs": -6.0, "rms_dbfs": -9.0}
        assert levels["blocks"] == [block]
        assert levels["loudest"] == block
        assert levels["quietest_nonsilent"] == block
        for key in ("auto_gain", "levels_after", "note",
                    "suggested_master_volume"):
            assert key not in result

    def test_clean_render_is_one_float_render_and_a_16_bit_wav(
            self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, NORMAL)
        result = render(srv, tmp_path)
        assert len(fake.calls) == 1 and fake.calls[0]["float"]
        rate, channels, bits, ints = read_pcm(tmp_path / "out.wav")
        assert (rate, channels, bits) == (SAMPLERATE, 2, 16)
        assert len(ints) == len(NORMAL)
        assert max(abs(v) for v in ints) == pytest.approx(
            0.5 * 32767, abs=1)
        # The samples LMMS writes: x * 32767 truncated toward zero.
        assert ints == [int(v * 32767) for v in NORMAL]
        assert result["size_bytes"] == (tmp_path / "out.wav").stat().st_size
        assert result["message"].startswith(f"Rendered to {tmp_path / 'out.wav'}")
        # Unchanged project, still marked as saved.
        assert saved_master_volume(tmp_path) == 1.0
        assert not proj.modified

    def test_clipped_samples_are_counted_before_any_fix(
            self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, CLIPPING)
        result = render(srv, tmp_path)
        expected = sum(1 for v in CLIPPING if abs(v) > 1.0)
        assert expected > 1000
        assert result["levels"]["clipped_samples"] == expected
        assert result["levels"]["peak_dbfs"] == 6.0
        assert result["levels"]["rms_dbfs"] == 3.0

    def test_blocks_loudest_and_quietest(self, monkeypatch, tmp_path):
        # 12 bars: 4 loud, 4 quiet, 4 in between.
        signal = (sine(250, 0.5, 8) + sine(250, 0.1, 8) + sine(250, 0.25, 8))
        srv, _, _ = setup(monkeypatch, tmp_path, signal)
        levels = render(srv, tmp_path)["levels"]
        assert [b["bars"] for b in levels["blocks"]] == [
            [1, 4], [5, 8], [9, 12]]
        assert [b["peak_dbfs"] for b in levels["blocks"]] == [
            -6.0, -20.0, -12.0]
        assert [b["rms_dbfs"] for b in levels["blocks"]] == [
            -9.0, -23.0, -15.1]
        assert levels["loudest"]["bars"] == [1, 4]
        assert levels["quietest_nonsilent"]["bars"] == [5, 8]
        assert levels["peak_dbfs"] == -6.0

    def test_silent_blocks_are_not_the_quietest(self, monkeypatch, tmp_path):
        signal = sine(250, 0.5, 8) + array.array("f", [0.0] * (SAMPLERATE * 2 * 16))
        srv, _, _ = setup(monkeypatch, tmp_path, signal)
        levels = render(srv, tmp_path)["levels"]
        assert [b["peak_dbfs"] for b in levels["blocks"]] == [-6.0, None, None]
        assert levels["blocks"][1]["rms_dbfs"] is None
        assert levels["quietest_nonsilent"]["bars"] == [1, 4]

    def test_a_tail_below_minus_60_dbfs_is_not_the_quietest_part(
            self, monkeypatch, tmp_path):
        # 4 bars of music, then a bar of tail at -69 dBFS RMS.
        signal = sine(250, 0.5, 8) + sine(250, 0.0005, 2)
        srv, _, _ = setup(monkeypatch, tmp_path, signal)
        levels = render(srv, tmp_path)["levels"]
        assert [b["bars"] for b in levels["blocks"]] == [[1, 4], [5, 5]]
        assert levels["blocks"][1]["rms_dbfs"] == -69.0
        assert levels["quietest_nonsilent"]["bars"] == [1, 4]

    def test_bars_follow_the_projects_tempo_and_time_signature(
            self, monkeypatch, tmp_path):
        # 3/4 at 90 bpm: a bar is 2 s, so 12 s of audio are 6 bars: the
        # last 4 s (bars 5-6) are loud.
        signal = sine(250, 0.1, 8) + sine(250, 0.8, 4)
        srv, _, _ = setup(monkeypatch, tmp_path, signal, bpm=90,
                          numerator=3, denominator=4)
        levels = render(srv, tmp_path)["levels"]
        assert [b["bars"] for b in levels["blocks"]] == [[1, 4], [5, 6]]
        assert levels["loudest"]["bars"] == [5, 6]
        # At 60 bpm the bars are 3 s long: 4 bars.
        srv.get_project().set_tempo(60)
        levels = render(srv, tmp_path)["levels"]
        assert [b["bars"] for b in levels["blocks"]] == [[1, 4]]

    def test_automated_tempo_is_said(self, monkeypatch, tmp_path):
        srv, proj, _ = setup(monkeypatch, tmp_path, NORMAL)
        srv.add_automation("song", "tempo", [
            {"bar": 0, "value": 120}, {"bar": 4, "value": 140}])
        levels = render(srv, tmp_path)["levels"]
        assert "tempo is automated" in levels["bars_note"]

    def test_silent_render_has_a_note(self, monkeypatch, tmp_path):
        signal = array.array("f", [0.0] * (SAMPLERATE * 2 * 16))
        srv, _, _ = setup(monkeypatch, tmp_path, signal)
        result = render(srv, tmp_path)
        assert result["success"] is True
        assert result["levels"]["peak_dbfs"] is None
        assert result["levels"]["quietest_nonsilent"] is None
        assert "silent" in result["note"]
        assert "auto_gain" not in result

    def test_existing_fields_are_kept(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, NORMAL)
        result = render(srv, tmp_path, oversampling=2,
                        interpolation="sincbest")
        for key in ("success", "command", "returncode", "output_file",
                    "oversampling", "interpolation", "isolated_config",
                    "message", "size_bytes"):
            assert key in result
        assert result["returncode"] == 0
        assert result["isolated_config"] is True
        assert result["oversampling"] == 2
        assert "-x 2 -i sincbest" in result["command"]

    def test_docstring_and_signature(self):
        import inspect
        from lmms_mcp import server as srv
        doc = " ".join(srv.render_project.__doc__.split())
        for text in ("levels", "clipped_samples", "auto_gain",
                     "auto_fix_clipping", "levels_after", "loudest",
                     "quietest_nonsilent", "suggested_master_volume",
                     "-12 dBFS"):
            assert text in doc
        parameter = inspect.signature(srv.render_project).parameters[
            "auto_fix_clipping"]
        assert parameter.default is True


class TestAutoFix:
    GAIN = 10 ** (-1 / 20) / 2.0   # 0.4456: brings the +6.02 dBFS peak to -1 dBFS

    def test_master_volume_is_lowered_and_the_project_saved(
            self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, CLIPPING)
        result = render(srv, tmp_path)
        assert result["success"] is True
        gain = result["auto_gain"]
        assert gain["applied_db"] == pytest.approx(-7.02, abs=0.01)
        assert gain["master_volume_before"] == 1.0
        assert gain["master_volume_after"] == pytest.approx(self.GAIN, abs=1e-4)
        assert "6.0 dBFS" in gain["reason"] and "saved" in gain["reason"]
        assert saved_master_volume(tmp_path) == pytest.approx(self.GAIN, abs=1e-6)
        assert xml_parser.master_volume(proj.root) == pytest.approx(
            self.GAIN, abs=1e-6)
        assert not proj.modified
        # The levels are those of what LMMS made, before the fix.
        assert result["levels"]["peak_dbfs"] == 6.0
        assert result["levels"]["clipped_samples"] > 1000
        assert result["levels_after"] == {
            "peak_dbfs": -1.0, "clipped_samples": 0}

    def test_wav_is_written_from_the_float_render_without_a_second_one(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        render(srv, tmp_path)
        assert len(fake.calls) == 1 and fake.calls[0]["float"]
        _, _, bits, ints = read_pcm(tmp_path / "out.wav")
        peak = max(abs(v) for v in ints)
        assert bits == 16
        assert peak == pytest.approx(10 ** (-1 / 20) * 32767, abs=1)
        assert db(peak / 32767) == pytest.approx(-1.0, abs=0.001)
        assert all(abs(v) < 32767 for v in ints)

    def test_the_saved_project_renders_clean(self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        render(srv, tmp_path)
        again = render(srv, tmp_path, name="again.wav",
                       auto_fix_clipping=False)
        assert again["levels"]["clipped_samples"] == 0
        assert again["levels"]["peak_dbfs"] == -1.0
        assert "auto_gain" not in again and "note" not in again
        assert fake.calls[-1]["master_volume"] == pytest.approx(
            self.GAIN, abs=1e-6)

    def test_an_existing_master_volume_is_scaled(self, monkeypatch, tmp_path):
        # The project plays the signal at its Master volume 0.8 already:
        # the render peaks at 1.6 (+4.08 dBFS).
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING,
                             master_volume=0.8)
        result = render(srv, tmp_path)
        gain = 10 ** (-1 / 20) / 1.6
        assert result["levels"]["peak_dbfs"] == 4.1
        assert result["auto_gain"]["master_volume_before"] == 0.8
        assert result["auto_gain"]["master_volume_after"] == pytest.approx(
            0.8 * gain, abs=1e-4)
        assert saved_master_volume(tmp_path) == pytest.approx(0.8 * gain, abs=1e-6)
        rendered = render(srv, tmp_path, name="b.wav",
                          auto_fix_clipping=False)
        assert rendered["levels"]["peak_dbfs"] == -1.0

    def test_no_fix_when_asked_not_to(self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, CLIPPING)
        result = render(srv, tmp_path, auto_fix_clipping=False)
        assert result["success"] is True
        assert "auto_gain" not in result and "levels_after" not in result
        assert result["suggested_gain_db"] == pytest.approx(-7.0, abs=0.05)
        assert result["suggested_master_volume"] == pytest.approx(
            self.GAIN, abs=1e-4)
        assert "clips" in result["note"]
        assert "set_mixer_channel_volume(0, 0.4456)" in result["note"]
        assert saved_master_volume(tmp_path) == 1.0
        assert xml_parser.master_volume(proj.root) == 1.0
        assert not proj.modified
        # The file is what LMMS writes (clipped at 0 dBFS): LMMS renders
        # the unchanged project once more, as it did before the levels.
        assert [c["float"] for c in fake.calls] == [True, False]
        assert fake.calls[1]["master_volume"] == 1.0
        cmd = fake.commands[1]
        assert cmd[cmd.index("-o") + 1] == str(tmp_path / "out.wav")
        assert cmd[cmd.index("-f") + 1] == "wav"
        assert (tmp_path / "out.wav").read_bytes() == b"ENCODED:wav"
        assert result["command"] == " ".join(cmd)

    def test_a_clean_render_is_not_rendered_twice_without_the_fix(
            self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, NORMAL)
        render(srv, tmp_path, auto_fix_clipping=False)
        assert [c["float"] for c in fake.calls] == [True]
        assert read_pcm(tmp_path / "out.wav")[2] == 16

    def test_quiet_render_is_left_alone_with_a_note(self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, QUIET)
        result = render(srv, tmp_path)
        assert result["levels"]["peak_dbfs"] == -34.0
        assert "auto_gain" not in result
        assert result["suggested_gain_db"] == pytest.approx(33.0, abs=0.05)
        # More than the Master volume can add: say so, cap the suggestion.
        assert result["suggested_master_volume"] == 2.0
        assert "quiet" in result["note"]
        assert "track volumes" in result["note"]
        assert saved_master_volume(tmp_path) == 1.0
        _, _, _, ints = read_pcm(tmp_path / "out.wav")
        assert max(ints) == pytest.approx(0.02 * 32767, abs=1)

    def test_quiet_render_with_room_on_the_master_volume(
            self, monkeypatch, tmp_path):
        srv, _, _ = setup(monkeypatch, tmp_path, sine(250, 0.25, 8),
                          master_volume=0.1)
        # Plays at 0.025 (-32 dBFS); +31 dB would be volume 3.6: too much.
        result = render(srv, tmp_path)
        assert result["suggested_master_volume"] == 2.0
        srv, _, _ = setup(monkeypatch, tmp_path, sine(250, 5.0, 8),
                          master_volume=0.04)
        # Plays at 0.2 (-14 dBFS); +13 dB is volume 0.04 * 4.47.
        result = render(srv, tmp_path)
        assert result["levels"]["peak_dbfs"] == -14.0
        assert result["suggested_gain_db"] == pytest.approx(13.0, abs=0.05)
        assert result["suggested_master_volume"] == pytest.approx(
            0.04 * 10 ** (13.0 / 20), abs=1e-3)
        assert "Raising the Master volume from 0.04" in result["note"]
        assert "auto_gain" not in result

    @pytest.mark.parametrize("amplitude", [0.5, 0.3, 1.0])
    def test_a_peak_in_the_normal_range_is_not_touched(
            self, monkeypatch, tmp_path, amplitude):
        srv, _, fake = setup(monkeypatch, tmp_path,
                             sine(250, amplitude, 8))
        result = render(srv, tmp_path)
        assert "auto_gain" not in result and "note" not in result
        assert saved_master_volume(tmp_path) == 1.0

    @pytest.mark.parametrize("fmt", ["flac", "ogg", "mp3"])
    def test_other_formats_render_the_adjusted_project(
            self, monkeypatch, tmp_path, fmt):
        srv, proj, fake = setup(monkeypatch, tmp_path, CLIPPING)
        result = render(srv, tmp_path, name=f"out.{fmt}", file_format=fmt)
        assert result["success"] is True
        assert len(fake.calls) == 2
        first, second = fake.calls
        assert first["float"] and first["master_volume"] == 1.0
        assert not second["float"]
        # The second render read the project saved with the lower volume.
        assert second["master_volume"] == pytest.approx(self.GAIN, abs=1e-6)
        assert second["cmd"][second["cmd"].index("-o") + 1] == str(
            tmp_path / f"out.{fmt}")
        assert second["cmd"][second["cmd"].index("-f") + 1] == fmt
        assert (tmp_path / f"out.{fmt}").read_bytes() == b"ENCODED:" + fmt.encode()
        assert result["auto_gain"]["master_volume_after"] == pytest.approx(
            self.GAIN, abs=1e-4)
        assert result["levels_after"]["clipped_samples"] == 0
        assert result["levels"]["clipped_samples"] > 0

    def test_other_formats_without_clipping_still_render_once_more(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, NORMAL)
        result = render(srv, tmp_path, name="out.flac", file_format="flac")
        assert [c["float"] for c in fake.calls] == [True, False]
        assert fake.calls[1]["master_volume"] == 1.0
        assert "auto_gain" not in result
        assert result["levels"]["peak_dbfs"] == -6.0

    def test_other_format_without_the_fix_renders_the_unchanged_project(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        result = render(srv, tmp_path, name="out.ogg", file_format="ogg",
                        auto_fix_clipping=False)
        assert fake.calls[1]["master_volume"] == 1.0
        assert "suggested_master_volume" in result

    def test_second_render_failure_keeps_the_levels_and_the_fix(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        real = fake

        def run(cmd, **kwargs):
            if "-a" in cmd:
                return real(cmd, **kwargs)
            return subprocess.CompletedProcess(cmd, 1, "", "encoder broke")

        monkeypatch.setattr("subprocess.run", run)
        result = render(srv, tmp_path, name="out.ogg", file_format="ogg")
        assert result["success"] is False
        assert result["error"] == "Render failed: encoder broke"
        assert result["levels"]["peak_dbfs"] == 6.0
        assert "auto_gain" in result
        assert saved_master_volume(tmp_path) == pytest.approx(self.GAIN, abs=1e-6)

    def test_second_render_timeout_is_reported(self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, NORMAL)

        def run(cmd, **kwargs):
            if "-a" in cmd:
                return fake(cmd, **kwargs)
            raise subprocess.TimeoutExpired(cmd, 300)

        monkeypatch.setattr("subprocess.run", run)
        result = render(srv, tmp_path, name="out.mp3", file_format="mp3")
        assert "timed out" in result["error"]
        assert result["levels"]["peak_dbfs"] == -6.0
        assert not (tmp_path / "out.mp3").exists()

    def test_project_that_cannot_be_saved_is_left_unchanged(
            self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, CLIPPING)
        real_save = proj.save
        saves = []

        def save(path=None, compressed=None):
            saves.append(path)
            if len(saves) == 2:  # the save of the lowered volume
                raise PermissionError(13, "Permission denied", str(path))
            return real_save(path, compressed=compressed)

        monkeypatch.setattr(proj, "save", save)
        result = render(srv, tmp_path)
        assert result["success"] is False
        assert "could not be saved" in result["error"]
        assert "Permission denied" in result["error"]
        assert "left unchanged" in result["error"]
        assert "auto_gain" not in result
        assert result["levels"]["peak_dbfs"] == 6.0
        assert xml_parser.master_volume(proj.root) == 1.0
        assert saved_master_volume(tmp_path) == 1.0
        assert not (tmp_path / "out.wav").exists()

    def test_mmpz_projects_are_saved_compressed_again(
            self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, CLIPPING)
        proj.save(str(tmp_path / "song.mmpz"))
        result = render(srv, tmp_path)
        assert "auto_gain" in result
        loaded = xml_parser.load_project(tmp_path / "song.mmpz")
        assert xml_parser.master_volume(loaded) == pytest.approx(
            self.GAIN, abs=1e-6)
        assert (tmp_path / "song.mmpz").read_bytes()[:5] != b"<?xml"


class TestMasterVolumeSpellings:
    GAIN = 10 ** (-1 / 20) / 2.0

    def test_lmms_1_3_names_are_kept(self, monkeypatch, tmp_path):
        srv, proj, _ = setup(monkeypatch, tmp_path, CLIPPING,
                             lmms_version="1.3.0")
        assert proj.root.find("song/mixer/mixerchannel") is not None
        result = render(srv, tmp_path)
        saved = xml_parser.load_project(tmp_path / "song.mmp")
        channel = saved.find("song/mixer/mixerchannel[@num='0']")
        assert float(channel.get("volume")) == pytest.approx(self.GAIN, abs=1e-6)
        assert saved.find("song/fxmixer") is None
        assert result["auto_gain"]["master_volume_after"] == pytest.approx(
            self.GAIN, abs=1e-4)

    def test_a_project_without_a_mixer_gets_one(self, monkeypatch, tmp_path):
        srv, proj, _ = setup(monkeypatch, tmp_path, CLIPPING)
        song = proj.root.find("song")
        song.remove(xml_parser.find_mixer(proj.root))
        proj.save(str(tmp_path / "song.mmp"))
        result = render(srv, tmp_path)
        assert result["auto_gain"]["master_volume_before"] == 1.0
        assert "no Master mixer channel" in result["auto_gain"]["reason"]
        saved = xml_parser.load_project(tmp_path / "song.mmp")
        channel = saved.find("song/fxmixer/fxchannel[@num='0']")
        assert channel.get("name") == "Master"
        assert float(channel.get("volume")) == pytest.approx(self.GAIN, abs=1e-6)
        again = render(srv, tmp_path, name="b.wav", auto_fix_clipping=False)
        assert again["levels"]["peak_dbfs"] == -1.0

    def test_a_mixer_without_channel_0_gets_a_master(self, monkeypatch, tmp_path):
        srv, proj, _ = setup(monkeypatch, tmp_path, CLIPPING)
        mixer = xml_parser.find_mixer(proj.root)
        for channel in list(mixer):
            mixer.remove(channel)
        proj.save(str(tmp_path / "song.mmp"))
        result = render(srv, tmp_path)
        assert "auto_gain" in result
        saved = xml_parser.load_project(tmp_path / "song.mmp")
        channels = saved.findall("song/fxmixer/fxchannel")
        assert [c.get("num") for c in channels] == ["0"]

    def test_other_channels_are_untouched(self, monkeypatch, tmp_path):
        srv, proj, _ = setup(monkeypatch, tmp_path, CLIPPING)
        srv.add_mixer_channel("Drums", 0.7)
        render(srv, tmp_path)
        volumes = {c.get("name"): c.get("volume")
                   for c in xml_parser.find_mixer_channels(proj.root)}
        assert volumes["Drums"] == "0.7"
        assert float(volumes["Master"]) == pytest.approx(self.GAIN, abs=1e-6)

    def test_automated_master_volume_is_scaled_with_its_curve(
            self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, CLIPPING)
        srv.add_automation("mixer", "volume", [
            {"bar": 0, "value": 1.0}, {"bar": 2, "value": 0.5},
            {"bar": 4, "value": 1.0}], target_index=0, smooth=True)
        result = render(srv, tmp_path)
        assert result["auto_gain"]["master_volume_after"] == pytest.approx(
            self.GAIN, abs=1e-4)
        assert ("The automation of the Master volume (1 pattern) was scaled "
                "by the same factor.") in result["auto_gain"]["reason"]
        saved = xml_parser.load_project(tmp_path / "song.mmp")
        pattern = saved.find(".//automationpattern")
        values = [float(t.get("value")) for t in pattern.findall("time")]
        assert values == pytest.approx([self.GAIN, self.GAIN * 0.5, self.GAIN])
        outs = [float(t.get("outValue")) for t in pattern.findall("time")]
        assert outs == pytest.approx(values)
        channel = xml_parser.find_master_channel(saved)
        assert float(channel.find("volume").get("value")) == pytest.approx(
            self.GAIN, abs=1e-6)
        assert channel.find("volume").get("id")  # still automated

    def test_a_curve_shared_with_other_parameters_is_not_fixed(
            self, monkeypatch, tmp_path):
        srv, proj, fake = setup(monkeypatch, tmp_path, CLIPPING)
        srv.add_instrument_track("Lead")
        srv.add_automation("mixer", "volume", [
            {"bar": 0, "value": 1.0}, {"bar": 4, "value": 0.9}],
            target_index=0)
        pattern = proj.root.find(".//automationpattern")
        import xml.etree.ElementTree as ET
        ET.SubElement(pattern, "object", {"id": "999999"})
        proj.save(str(tmp_path / "song.mmp"))
        before = saved_master_volume(tmp_path)
        result = render(srv, tmp_path)
        assert result["success"] is True
        assert "auto_gain" not in result
        assert result["note"].startswith("Not fixed automatically")
        assert "also controls other parameters" in result["note"]
        assert result["suggested_master_volume"] == pytest.approx(
            self.GAIN, abs=1e-4)
        assert saved_master_volume(tmp_path) == before
        # Written by LMMS, as before.
        assert [c["float"] for c in fake.calls] == [True, False]
        assert (tmp_path / "out.wav").read_bytes() == b"ENCODED:wav"


class TestMasterVolumeHelpers:
    def test_scale_master_volume(self):
        proj = LMMSProject()
        proj.new()
        info = xml_parser.scale_master_volume(proj.root, 0.5)
        assert info == {"before": 1.0, "after": 0.5, "curves_scaled": 0,
                        "created": False}
        assert xml_parser.master_volume(proj.root) == 0.5
        assert xml_parser.scale_master_volume(proj.root, 0.5)["after"] == 0.25

    def test_scaling_keeps_an_automated_models_id(self):
        proj = LMMSProject()
        proj.new()
        channel = xml_parser.find_master_channel(proj.root)
        model_id = xml_parser.automate_attribute(channel, "volume", "1")
        xml_parser.scale_master_volume(proj.root, 0.25)
        child = channel.find("volume")
        assert child.get("id") == str(model_id)
        assert float(child.get("value")) == 0.25
        assert xml_parser.master_volume(proj.root) == 0.25

    def test_a_missing_master_volume_defaults_to_one(self):
        proj = LMMSProject()
        proj.new()
        channel = xml_parser.find_master_channel(proj.root)
        del channel.attrib["volume"]
        assert xml_parser.master_volume(proj.root) == 1.0
        assert xml_parser.scale_master_volume(proj.root, 0.5)["after"] == 0.5
        assert channel.get("volume") == "0.5"

    def test_invalid_factor(self):
        proj = LMMSProject()
        proj.new()
        for factor in (0, -1, float("nan")):
            with pytest.raises(ValueError):
                xml_parser.scale_master_volume(proj.root, factor)

    def test_find_master_channel_ignores_other_channels(self):
        proj = LMMSProject()
        proj.new()
        proj.add_mixer_channel("Bass")
        assert xml_parser.find_master_channel(proj.root).get("name") == "Master"
        assert xml_parser.ensure_master_channel(proj.root).get("num") == "0"

    def test_model_automation_still_reports_patterns(self):
        # Refactored to share a helper with the Master volume scaling.
        proj = LMMSProject()
        proj.new()
        channel = xml_parser.find_master_channel(proj.root)
        assert xml_parser.model_automation(proj.root, channel, "volume") == []
        model_id = xml_parser.automate_attribute(channel, "volume", "1")
        xml_parser.add_automation_curve_track(
            proj.root, "Fade", [(192, 1.0), (576, 0.0)], target_id=model_id)
        assert xml_parser.model_automation(proj.root, channel, "volume") == [{
            "track_index": 0, "track": "Fade", "start": 192,
            "last_point": 192 + 384}]


class TestTempFilesAndLock:
    @pytest.fixture
    def tmp_dir(self, monkeypatch, tmp_path):
        folder = tmp_path / "tmp"
        folder.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(folder))
        return folder

    def test_nothing_is_left_after_a_wav_render(
            self, monkeypatch, tmp_path, tmp_dir):
        seen = []
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        fake.on_run = lambda cmd: seen.append(
            Path(cmd[cmd.index("-o") + 1]).parent.parent)
        render(srv, tmp_path)
        assert seen == [tmp_dir]  # the float render went to a temp folder
        assert list(tmp_dir.iterdir()) == []

    def test_nothing_is_left_after_another_format(
            self, monkeypatch, tmp_path, tmp_dir):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        render(srv, tmp_path, name="out.flac", file_format="flac")
        assert list(tmp_dir.iterdir()) == []

    def test_nothing_is_left_when_the_render_fails(
            self, monkeypatch, tmp_path, tmp_dir):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)

        def failing(cmd, **kwargs):
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"partial")
            return subprocess.CompletedProcess(cmd, 1, "", "boom")

        monkeypatch.setattr("subprocess.run", failing)
        result = render(srv, tmp_path)
        assert result["success"] is False
        assert list(tmp_dir.iterdir()) == []

    def test_nothing_is_left_after_a_timeout(
            self, monkeypatch, tmp_path, tmp_dir):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)

        def hang(cmd, **kwargs):
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"partial")
            raise subprocess.TimeoutExpired(cmd, 300)

        monkeypatch.setattr("subprocess.run", hang)
        result = render(srv, tmp_path)
        assert "timed out" in result["error"]
        assert list(tmp_dir.iterdir()) == []

    def test_nothing_is_left_when_the_audio_cannot_be_written(
            self, monkeypatch, tmp_path, tmp_dir):
        from lmms_mcp import audio_analysis
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        monkeypatch.setattr(
            audio_analysis, "write_wav",
            lambda *a, **k: (_ for _ in ()).throw(OSError(28, "Disk full")))
        assert render(srv, tmp_path)["success"] is False
        assert list(tmp_dir.iterdir()) == []

    def test_an_unreadable_float_render_falls_back_to_a_plain_render(
            self, monkeypatch, tmp_path, tmp_dir):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        calls = []

        def odd(cmd, **kwargs):
            calls.append(cmd)
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"RIFF....")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("subprocess.run", odd)
        result = render(srv, tmp_path)
        assert result["success"] is True
        assert "Could not measure the render" in result["levels_error"]
        assert "levels" not in result and "auto_gain" not in result
        assert len(calls) == 2 and "-a" in calls[0] and "-a" not in calls[1]
        assert calls[1][calls[1].index("-o") + 1] == str(tmp_path / "out.wav")
        assert (tmp_path / "out.wav").read_bytes() == b"RIFF...."
        assert list(tmp_dir.iterdir()) == []

    def test_every_lmms_process_goes_through_the_lock(
            self, monkeypatch, tmp_path):
        srv, _, fake = setup(monkeypatch, tmp_path, CLIPPING)
        held = []
        real = lmms_app._LMMS_RUN_LOCK

        def check(cmd):
            held.append(real.locked())

        fake.on_run = check
        render(srv, tmp_path, name="out.ogg", file_format="ogg")
        assert held == [True, True]  # the float render and the second one

    def test_lmms_processes_never_overlap(self, monkeypatch):
        active = []
        most = [0]
        guard = threading.Lock()

        def run(cmd, **kwargs):
            with guard:
                active.append(1)
                most[0] = max(most[0], len(active))
            time.sleep(0.03)
            with guard:
                active.pop()
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("subprocess.run", run)
        threads = [threading.Thread(target=lmms_app.run_lmms, args=(["lmms"],))
                   for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert most[0] == 1

    def test_the_lock_is_released_when_lmms_fails_to_start(self, monkeypatch):
        def run(cmd, **kwargs):
            raise OSError("no such file")

        monkeypatch.setattr("subprocess.run", run)
        for _ in range(2):
            with pytest.raises(OSError):
                lmms_app.run_lmms(["lmms"])
        assert not lmms_app._LMMS_RUN_LOCK.locked()

    def test_the_version_check_waits_for_a_running_render_too(
            self, monkeypatch):
        monkeypatch.setattr(lmms_app, "find_lmms_exe",
                            lambda: Path("/fake/lmms"))
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0, "LMMS 1.2.2", ""))
        result = []
        with lmms_app._LMMS_RUN_LOCK:
            thread = threading.Thread(
                target=lambda: result.append(lmms_app.get_lmms_version()))
            thread.start()
            thread.join(0.2)
            assert result == []  # still waiting
        thread.join(10)
        assert result == ["1.2.2"]
