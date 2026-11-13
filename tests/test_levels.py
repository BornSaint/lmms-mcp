"""Levels: how loud each sound is at track volume 100 (lmms_mcp.levels), the
responses that report it and the estimate of the mix peak (mix_estimate).

LMMS is replaced by FakeLevelLmms (tests/audio_helpers.py), which writes a
float WAV with known amplitudes for the measuring project, so what is
checked is how the module builds that project, reads the render, caches
the result and what the tools do with it.
"""

import dataclasses
import gzip
import json
import math
import os
from pathlib import Path

import pytest

from audio_helpers import FakeLevelLmms, SAMPLERATE, sine
from lmms_mcp import (
    audio_analysis, instrument_presets, levels, lmms_app, mix_estimate,
    presets, xml_parser,
)
from lmms_mcp.levels import Level
from lmms_mcp.project import LMMSProject


def db(x):
    return 20 * math.log10(x)


class Lab:
    """A server with a fake LMMS that measures levels, an empty project and
    the fake LMMS version (no seed unless version is 1.2.x)."""

    def __init__(self, monkeypatch, tmp_path, version="1.3.0", **fake):
        from lmms_mcp import server as srv

        self.srv = srv
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        monkeypatch.setenv("LMMS_MCP_MEASURE_LEVELS", "1")
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        self.fake = FakeLevelLmms(**fake)
        monkeypatch.setattr("subprocess.run", self.fake)
        monkeypatch.setattr(srv, "_availability", dataclasses.replace(
            srv._availability, version=version))
        self.proj = LMMSProject()
        self.proj.new(lmms_version=version)
        srv.set_project(self.proj)

    @property
    def renders(self):
        return len(self.fake.commands)

    def add(self, instrument="tripleoscillator", **kwargs):
        return json.loads(self.srv.add_instrument_track(
            instrument.title(), instrument=instrument, **kwargs))

    def inst(self, index=0):
        return self.proj._instrument_track_element(index)


@pytest.fixture
def lab(monkeypatch, tmp_path):
    return Lab(monkeypatch, tmp_path)


@pytest.fixture
def lab12(monkeypatch, tmp_path):
    """LMMS 1.2.2: the seed applies."""
    return Lab(monkeypatch, tmp_path, version="1.2.2")


def level_field(result):
    return result.get("level_at_vol_100")


# ──────────────────────────────────────────────────────────────────
# measure_level
# ──────────────────────────────────────────────────────────────────


class TestMeasureLevel:
    def test_reads_note_chord_and_rms(self, lab):
        lab.fake.amplitudes = {"tripleoscillator": (0.25, 0.5)}
        lab.add()
        level = levels.measure_level(
            lab.inst(), lab.srv._render_for_levels, "1.3.0")
        assert level.note_peak_dbfs == pytest.approx(db(0.25), abs=0.05)
        assert level.chord_peak_dbfs == pytest.approx(db(0.5), abs=0.05)
        # a sine of amplitude a has an RMS of a / sqrt(2) while it is held
        assert level.rms_dbfs == pytest.approx(db(0.25 / math.sqrt(2)),
                                               abs=0.1)
        assert level == Level(
            round(level.note_peak_dbfs, 1), round(level.chord_peak_dbfs, 1),
            round(level.rms_dbfs, 1))

    def test_measuring_project(self, lab):
        lab.add(volume=37, panning=40)
        levels.measure_level(lab.inst(), lab.srv._render_for_levels, "1.3.0")
        root = lab.fake.projects[-1]
        head = xml_parser.find_head(root)
        assert head.get("bpm") == "60"
        assert head.get("mastervol") == "100"
        assert xml_parser.master_volume(root) == 1.0
        tracks = xml_parser.find_tracks(root)
        assert len(tracks) == 1
        inst = tracks[0].find("instrumenttrack")
        assert inst.get("vol") == "100" and inst.get("pan") == "0"
        assert xml_parser.get_track_mixer_channel(inst) == 0
        assert inst.find("fxchain/effect") is None
        notes = [
            (int(n.get("key")), int(n.get("pos")), int(n.get("len")),
             int(n.get("vol")))
            for n in tracks[0].findall("pattern/note")]
        # three single notes (bars 0, 2, 4), three chords (bars 6, 8, 10)
        assert notes == [(57, pos, 48, 100) for pos in (0, 384, 768)] + [
            (key, pos, 48, 100) for pos in (1152, 1536, 1920)
            for key in (48, 52, 55, 60)]
        assert levels.NOTE_BARS == (0, 2, 4)
        assert levels.CHORD_BARS == (6, 8, 10)

    def test_the_loudest_repetition_counts(self, monkeypatch, tmp_path):
        # louder in some places of the song than in others (a tempo-synced
        # LFO, random phases): the peaks are the largest of the three
        lab = Lab(monkeypatch, tmp_path, default=(0.5, 0.25),
                  repeat_scales=(0.5, 1.0, 0.7))
        lab.add()
        level = lab.proj.level_table[levels.level_key(lab.inst(), "1.3.0")]
        assert level.note_peak_dbfs == pytest.approx(db(0.5), abs=0.05)
        assert level.chord_peak_dbfs == pytest.approx(db(0.25), abs=0.05)
        # the RMS is that of all three notes together
        scales = (0.5, 1.0, 0.7)
        mean_square = sum(f * f for f in scales) / 3
        assert level.rms_dbfs == pytest.approx(
            db(0.5 / math.sqrt(2) * math.sqrt(mean_square)), abs=0.1)

    def test_render_command(self, lab):
        lab.add()
        levels.measure_level(lab.inst(), lab.srv._render_for_levels, "1.3.0")
        cmd = lab.fake.commands[-1]
        assert cmd[cmd.index("-x") + 1] == "1"
        assert "-a" in cmd                       # 32-bit float
        assert cmd[cmd.index("-f") + 1] == "wav"
        assert "--config" in cmd                 # the isolated config
        assert cmd[cmd.index("-s") + 1] == "44100"
        assert lab.fake.calls[-1]["kwargs"]["timeout"] == levels.RENDER_TIMEOUT

    def test_copies_the_sound_and_drops_effects(self, lab):
        lab.add()
        inst = lab.inst()
        inst.find("instrument")[0].set("coarse0", "-12")
        inst.find("eldata").set("fcut", "484")
        inst.find("fxchain").set("numofeffects", "1")
        effect = xml_parser.ET.SubElement(
            inst.find("fxchain"), "effect", {"name": "reverbsc"})
        before = xml_parser.ET.tostring(inst)
        levels.measure_level(inst, lab.srv._render_for_levels, "1.3.0")
        sent = lab.fake.projects[-1]
        copy = xml_parser.find_tracks(sent)[0].find("instrumenttrack")
        assert copy.find("instrument")[0].get("coarse0") == "-12"
        assert copy.find("eldata").get("fcut") == "484"
        assert copy.find("fxchain/effect") is None
        assert xml_parser.ET.tostring(inst) == before   # the track is intact
        assert effect in list(inst.find("fxchain"))

    @pytest.mark.parametrize("version,attr", [
        ("1.2.2", "fxch"), ("1.3.0", "mixch")])
    def test_routing_uses_the_versions_attribute(
            self, monkeypatch, tmp_path, version, attr):
        lab = Lab(monkeypatch, tmp_path, version=version)
        lab.srv.add_mixer_channel("Bus")
        lab.add(mixer_channel=1)
        levels.measure_level(
            lab.inst(), lab.srv._render_for_levels, version)
        copy = xml_parser.find_tracks(
            lab.fake.projects[-1])[0].find("instrumenttrack")
        assert copy.get(attr) == "0"

    def test_silent_sound(self, monkeypatch, tmp_path):
        lab = Lab(monkeypatch, tmp_path, silent=True)
        lab.add()
        with pytest.raises(levels.LevelError, match="silent"):
            levels.measure_level(
                lab.inst(), lab.srv._render_for_levels, "1.3.0")

    def test_impossible_level_is_an_error(self, monkeypatch, tmp_path):
        lab = Lab(monkeypatch, tmp_path, default=(1e12, 1e12))
        lab.add()
        with pytest.raises(levels.LevelError, match="impossible"):
            levels.measure_level(
                lab.inst(), lab.srv._render_for_levels, "1.3.0")

    @pytest.mark.parametrize("fake,message", [
        ({"fail": True}, "Render failed"),
        ({"timeout": True}, "did not finish"),
        ({"os_error": True}, "could not be started"),
        ({"write_nothing": True}, "Render failed"),
    ])
    def test_render_failures(self, monkeypatch, tmp_path, fake, message):
        lab = Lab(monkeypatch, tmp_path, **fake)
        lab.add()
        with pytest.raises(levels.LevelError, match=message):
            levels.measure_level(
                lab.inst(), lab.srv._render_for_levels, "1.3.0")

    def test_no_lmms(self, lab, monkeypatch):
        lab.add()
        monkeypatch.setattr(lmms_app, "find_lmms_exe", lambda: None)
        with pytest.raises(levels.LevelError, match="not found"):
            levels.measure_level(
                lab.inst(), lab.srv._render_for_levels, "1.3.0")

    def test_unreadable_render(self, lab, monkeypatch):
        lab.add()

        def render(project, output):
            output.write_bytes(b"not a wav")

        with pytest.raises(levels.LevelError, match="could not read"):
            levels.measure_level(lab.inst(), render, "1.3.0")


# ──────────────────────────────────────────────────────────────────
# the cache
# ──────────────────────────────────────────────────────────────────


class TestCache:
    def resolve(self, lab, index=0, table=None):
        return levels.resolve_level(
            lab.inst(index), "1.3.0", table,
            render=lab.srv._render_for_levels)

    def test_miss_then_hit(self, lab):
        added = lab.add()
        assert level_field(added)["source"] == "measured"
        assert lab.renders == 1
        level, source = self.resolve(lab)
        assert source == "cache" and lab.renders == 1
        assert level.to_dict("measured") == level_field(added)
        # a track of another sound is a miss
        lab.add("kicker")
        assert lab.renders == 2

    def test_the_file_is_in_the_cache_folder(self, lab, tmp_path):
        lab.add()
        self.resolve(lab)
        path = tmp_path / "xdg_cache" / "lmms-mcp" / "levels.json"
        assert levels.cache_path() == path
        data = json.loads(path.read_text())
        (key, entry), = data["entries"].items()
        assert key.startswith("1.3.0|tripleoscillator|")
        assert Level.from_dict(entry) is not None
        assert entry["plugin"] == "tripleoscillator"

    def test_default_folder_is_dot_cache(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XDG_CACHE_HOME")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert levels.cache_path() == tmp_path / ".cache" / "lmms-mcp" \
            / "levels.json"
        monkeypatch.setenv("XDG_CACHE_HOME", "relative/dir")  # not valid
        assert levels.cache_path() == tmp_path / ".cache" / "lmms-mcp" \
            / "levels.json"

    def test_a_new_process_reads_the_file(self, lab):
        lab.add()
        level, _ = self.resolve(lab)
        found = levels.find_level(
            lab.inst(), "1.3.0", None, levels.LevelCache())
        assert found == (level, "cache")

    @pytest.mark.parametrize("content", [
        "not json {", "", "[1, 2]", '{"entries": [1]}', '{"entries": 5}',
        '{"format": 1}', '{"entries": {"k": "x"}}'])
    def test_a_corrupt_cache_is_rebuilt(self, lab, content):
        lab.add()
        path = levels.cache_path()
        path.write_text(content)
        level, source = self.resolve(lab)
        assert source == "measured" and lab.renders == 2
        json.loads(path.read_text())            # valid again
        assert self.resolve(lab) == (level, "cache")
        assert lab.renders == 2

    def test_invalid_entries_are_ignored(self, lab):
        lab.add()
        key = levels.level_key(lab.inst(), "1.3.0")
        path = levels.cache_path()
        path.write_text(json.dumps({"entries": {
            key: {"note_peak_dbfs": "loud"},
            "other": {"note_peak_dbfs": -1, "chord_peak_dbfs": 0,
                      "rms_dbfs": float("nan")}}}))
        assert levels.CACHE.get(key) is None
        assert levels.CACHE.get("other") is None

    def test_an_unwritable_cache_is_ignored(self, lab, tmp_path, monkeypatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a folder")
        monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
        lab.proj.level_table.clear()
        result = lab.add()                    # no error, the level is there
        assert level_field(result)["source"] == "measured"
        assert not levels.cache_path().exists()
        # the project's own table still remembers it
        assert level_field(lab.add())["source"] == "cache"
        assert lab.renders == 1

    def test_writes_are_atomic(self, lab):
        lab.add()
        self.resolve(lab)
        folder = levels.cache_path().parent
        assert [p.name for p in folder.iterdir()] == ["levels.json"]

    def test_the_cache_keeps_the_newest_levels(self, tmp_path, monkeypatch):
        monkeypatch.setattr(levels, "CACHE_LIMIT", 2)
        cache = levels.LevelCache(tmp_path / "c.json")
        for i in range(3):
            assert cache.put(f"k{i}", Level(-1.0 * i, 0.0, -20.0))
        assert cache.get("k0") is None
        assert cache.get("k1") == Level(-1.0, 0.0, -20.0)
        assert cache.get("k2") == Level(-2.0, 0.0, -20.0)

    def test_put_replaces_an_entry(self, tmp_path):
        cache = levels.LevelCache(tmp_path / "c.json")
        cache.put("k", Level(-1.0, 0.0, -20.0), "tripleoscillator")
        cache.put("k", Level(-5.0, 0.0, -20.0))
        assert cache.get("k") == Level(-5.0, 0.0, -20.0)

    def test_the_project_table_answers_before_the_file(self, lab):
        lab.add()
        table = {}
        level, _ = self.resolve(lab, table=table)
        levels.cache_path().unlink()
        assert levels.find_level(lab.inst(), "1.3.0", table) \
            == (level, "cache")
        assert levels.find_level(lab.inst(), "1.3.0", {}) is None

    # --- what a key is made of -------------------------------------

    def key(self, lab, index=0, version="1.3.0"):
        return levels.level_key(lab.inst(index), version)

    def test_key_ignores_volume_panning_name_routing_tuning_effects(
            self, lab):
        lab.srv.add_mixer_channel("Bus")
        lab.add(volume=100)
        lab.add(volume=20, panning=50, mixer_channel=1)
        lab.srv.set_track_tuning(1, base_note="C4", pitch_cents=20)
        lab.srv.add_effect("track", 1, "reverbsc")
        lab.proj.root.find("song/trackcontainer").findall(
            "track")[1].set("name", "Renamed")
        assert self.key(lab, 0) == self.key(lab, 1)

    def test_key_changes_with_the_settings(self, lab):
        lab.add()
        lab.add()
        lab.inst(1).find("instrument")[0].set("coarse0", "-12")
        assert self.key(lab, 0) != self.key(lab, 1)
        lab.add()
        lab.inst(2).find("eldata").set("fcut", "484")
        assert self.key(lab, 0) != self.key(lab, 2)
        lab.add()
        lab.inst(3).find("arpeggiator").set("arp-enabled", "1")
        assert self.key(lab, 0) != self.key(lab, 3)

    def test_key_changes_with_the_version_and_plugin(self, lab):
        lab.add()
        lab.add("kicker")
        assert self.key(lab, 0) != self.key(lab, 0, "1.2.2")
        assert self.key(lab, 0) != self.key(lab, 1)
        assert "|kicker|" in self.key(lab, 1)

    def test_key_ignores_model_ids(self, lab):
        lab.add()
        lab.add()
        for index, number in ((0, "11"), (1, "22")):
            xml_parser.ET.SubElement(
                lab.inst(index).find("eldata"), "fcut",
                {"id": number, "value": "100"})
        assert self.key(lab, 0) == self.key(lab, 1)

    def test_a_changed_sample_is_measured_again(self, lab, tmp_path):
        sample = tmp_path / "kick.wav"
        sample.write_bytes(b"RIFF....")
        lab.add("audiofileprocessor")
        for _ in range(2):
            result = json.loads(lab.srv.set_audiofile_sample(0, str(sample)))
        assert level_field(result)["source"] == "cache"
        assert lab.renders == 1
        later = sample.stat().st_mtime + 100
        os.utime(sample, (later, later))                 # modified
        result = json.loads(lab.srv.set_audiofile_sample(0, str(sample)))
        assert level_field(result)["source"] == "measured"
        assert lab.renders == 2
        sample.write_bytes(b"RIFF........")              # another size
        result = json.loads(lab.srv.set_audiofile_sample(0, str(sample)))
        assert level_field(result)["source"] == "measured"
        assert lab.renders == 3

    def test_a_relative_sample_is_found_like_lmms_does(
            self, lab, tmp_path, monkeypatch):
        folder = tmp_path / "samples"
        (folder / "drums").mkdir(parents=True)
        (folder / "drums" / "kick.ogg").write_bytes(b"1234")
        monkeypatch.setattr(lmms_app, "sample_search_dirs", lambda: [folder])
        lab.add("audiofileprocessor")
        json.loads(lab.srv.set_audiofile_sample(0, "drums/kick.ogg"))
        first = self.key(lab)
        (folder / "drums" / "kick.ogg").write_bytes(b"12345678")
        json.loads(lab.srv.set_audiofile_sample(0, "drums/kick.ogg"))
        assert self.key(lab) != first

    def test_sample_settings_are_part_of_the_key(self, lab, tmp_path):
        sample = tmp_path / "kick.wav"
        sample.write_bytes(b"RIFF....")
        lab.add("audiofileprocessor")
        json.loads(lab.srv.set_audiofile_sample(0, str(sample), amp=100))
        json.loads(lab.srv.set_audiofile_sample(0, str(sample), amp=300))
        assert lab.renders == 2


# ──────────────────────────────────────────────────────────────────
# the seed
# ──────────────────────────────────────────────────────────────────


class TestSeed:
    def test_every_seeded_instrument_is_reported_without_a_render(
            self, lab12):
        for name, values in levels.SEED_LEVELS.items():
            result = lab12.add(name)
            assert level_field(result) == dict(
                zip(("note_peak_dbfs", "chord_peak_dbfs", "rms_dbfs"),
                    values), source="seed"), name
        assert lab12.renders == 0

    def test_the_seed_names_are_offered_instruments(self):
        from lmms_mcp import server as srv
        assert set(levels.SEED_LEVELS) <= set(srv.KNOWN_INSTRUMENTS)
        assert "sf2player" not in levels.SEED_LEVELS

    def test_the_seed_values_are_levels(self):
        for values in (*levels.SEED_LEVELS.values(),
                       levels.SEED_SOUNDFONT["level"]):
            assert Level.from_dict(dict(zip(
                ("note_peak_dbfs", "chord_peak_dbfs", "rms_dbfs"),
                values))) is not None

    def test_the_seed_is_named_for_the_plugin_lmms_loads(
            self, lab12, monkeypatch):
        # opulenz is OPL2 in LMMS 1.2: the track is a plain OPL2 one
        monkeypatch.setattr(lab12.srv, "_availability", lmms_app.build_availability(
            lab12.srv.KNOWN_INSTRUMENTS, lab12.srv.effects_mod.KNOWN_EFFECTS,
            {"OPL2".lower(), "tripleoscillator"}, "1.2.2",
            stems={"opl2": "OPL2"}, windows=False))
        result = lab12.add("opulenz")
        assert result["lmms_plugin"] == "OPL2"
        assert level_field(result)["source"] == "seed"
        assert level_field(result)["note_peak_dbfs"] \
            == levels.SEED_LEVELS["opulenz"][0]

    def test_no_seed_for_another_lmms_series(self, lab):
        result = lab.add()      # LMMS 1.3.0: rendered, not seeded
        assert level_field(result)["source"] == "measured"
        assert lab.renders == 1

    @pytest.mark.parametrize("version,applies", [
        (None, True), ("", True), ("1.2.2", True), ("1.2", True),
        ("1.2.0", True), ("1.3.0", False), ("1.20.1", False),
        ("1.1.3", False)])
    def test_seed_applies(self, version, applies):
        assert levels.seed_applies(version) is applies

    def test_a_changed_track_is_not_the_seed(self, lab12):
        lab12.add()
        lab12.inst().find("instrument")[0].set("coarse0", "-12")
        assert levels.seed_level(lab12.inst(), "1.2.2") is None

    def test_a_changed_envelope_is_not_the_seed(self, lab12):
        lab12.add()
        assert levels.seed_level(lab12.inst(), "1.2.2") is not None
        lab12.inst().find("eldata/elvol").set("amt", "1")
        assert levels.seed_level(lab12.inst(), "1.2.2") is None

    def test_volume_and_tuning_do_not_change_the_seed(self, lab12):
        lab12.add(volume=30, panning=-20)
        lab12.srv.set_track_tuning(0, base_note="C3")
        assert levels.seed_level(lab12.inst(), "1.2.2") == Level(
            *levels.SEED_LEVELS["tripleoscillator"])

    def test_default_soundfont(self, lab12, monkeypatch):
        files = lab12.srv._availability.files
        path = Path(files.default_soundfont)
        monkeypatch.setattr(levels, "SEED_SOUNDFONT", {
            "file": path.name, "size": path.stat().st_size, "bank": 0,
            "patch": 0, "level": (-26.8, -19.2, -39.9)})
        result = lab12.add("sf2player")
        assert level_field(result) == {
            "note_peak_dbfs": -26.8, "chord_peak_dbfs": -19.2,
            "rms_dbfs": -39.9, "source": "seed"}
        assert lab12.renders == 0

    def test_another_soundfont_or_patch_is_not_the_seed(
            self, lab12, monkeypatch, tmp_path):
        files = lab12.srv._availability.files
        path = Path(files.default_soundfont)
        spec = {"file": path.name, "size": path.stat().st_size, "bank": 0,
                "patch": 0, "level": (-26.8, -19.2, -39.9)}
        lab12.add("sf2player")
        monkeypatch.setattr(levels, "SEED_SOUNDFONT", dict(spec, size=99))
        assert levels.seed_level(lab12.inst(), "1.2.2") is None     # size
        monkeypatch.setattr(levels, "SEED_SOUNDFONT", spec)
        assert levels.seed_level(lab12.inst(), "1.2.2") is not None
        other = tmp_path / "other.sf2"
        other.write_bytes(b"")
        json.loads(lab12.srv.set_instrument_file(0, str(other)))
        assert levels.seed_level(lab12.inst(), "1.2.2") is None     # name
        json.loads(lab12.srv.set_instrument_file(0, str(path), 0, 5))
        assert levels.seed_level(lab12.inst(), "1.2.2") is None     # patch

    @pytest.mark.parametrize("instrument", [
        "audiofileprocessor", "bitinvader", "watsyn", "gigplayer", "patman"])
    def test_unseeded_instruments_have_no_level_and_no_render(
            self, lab12, instrument):
        result = lab12.add(instrument)
        assert "level_at_vol_100" not in result
        assert "note" not in result
        assert lab12.renders == 0

    def test_seeded_names_are_the_ones_that_can_sound(self):
        from lmms_mcp import server as srv
        unseeded = set(srv.KNOWN_INSTRUMENTS) - set(levels.SEED_LEVELS) \
            - {"sf2player"}
        for name in ("audiofileprocessor", "bitinvader", "watsyn",
                     "gigplayer", "patman"):
            assert name in unseeded
        assert levels.UNSET_IS_BROKEN == ("bitinvader", "watsyn")


# ──────────────────────────────────────────────────────────────────
# LMMS_MCP_MEASURE_LEVELS
# ──────────────────────────────────────────────────────────────────


class TestMeasuringSwitch:
    @pytest.mark.parametrize("value,enabled", [
        ("0", False), ("false", False), ("No", False), ("off", False),
        (" OFF ", False), ("1", True), ("true", True), ("", True),
        ("anything", True)])
    def test_values(self, monkeypatch, value, enabled):
        monkeypatch.setenv(levels.MEASURE_ENV, value)
        assert levels.measure_enabled() is enabled

    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv(levels.MEASURE_ENV)
        assert levels.measure_enabled() is True

    def test_disabled_never_renders(self, lab, monkeypatch):
        monkeypatch.setenv(levels.MEASURE_ENV, "0")
        result = lab.add()
        assert "level_at_vol_100" not in result and "note" not in result
        assert lab.renders == 0

    def test_disabled_still_uses_the_seed_and_the_cache(
            self, lab12, monkeypatch):
        lab12.add()                                  # seeded
        monkeypatch.setenv(levels.MEASURE_ENV, "1")
        lab12.inst().find("instrument")[0].set("coarse0", "-12")
        first = levels.resolve_level(
            lab12.inst(), "1.2.2", None, render=lab12.srv._render_for_levels)
        assert first[1] == "measured"
        monkeypatch.setenv(levels.MEASURE_ENV, "0")
        assert levels.resolve_level(
            lab12.inst(), "1.2.2", None,
            render=lab12.srv._render_for_levels)[1] == "cache"
        plain = lab12.add()
        assert level_field(plain)["source"] == "seed"
        lab12.inst(1).find("instrument")[0].set("coarse0", "5")
        assert levels.resolve_level(
            lab12.inst(1), "1.2.2", None,
            render=lab12.srv._render_for_levels) is None
        assert lab12.renders == 1


# ──────────────────────────────────────────────────────────────────
# the tools
# ──────────────────────────────────────────────────────────────────


XPF = (
    '<?xml version="1.0"?>\n<!DOCTYPE lmms-project>\n'
    '<lmms-project version="1.0" creator="LMMS" creatorversion="1.0.2" '
    'type="instrumenttracksettings">\n  <head/>\n'
    '  <instrumenttracksettings muted="0" type="0" name="Test Sound" '
    'solo="0">\n'
    '    <instrumenttrack pan="0" fxch="0" pitchrange="1" pitch="0" '
    'basenote="57" vol="100">\n'
    '      <instrument name="tripleoscillator">'
    '<tripleoscillator coarse0="-12"/></instrument>\n'
    '      <eldata fres="0.7" ftype="6" fcut="484" fwet="1"/>\n'
    '      <chordcreator chord="3" chordrange="1" chord-enabled="1"/>\n'
    '      <arpeggiator arp="2" arprange="1" arp-enabled="1"/>\n'
    '      <midiport inputcontroller="0"/>\n'
    '      <fxchain numofeffects="0" enabled="0"/>\n'
    '    </instrumenttrack>\n  </instrumenttracksettings>\n'
    '</lmms-project>\n'
)


@pytest.fixture
def zyn_dir(tmp_path, monkeypatch):
    folder = tmp_path / "ZynAddSubFX"
    for rel, name in (("Bass/0001-Deep.xiz", "Deep"),
                      ("Bass/0002-Wide.xiz", "Wide")):
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as out:
            out.write((
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<ZynAddSubFX-data version-major="1" version-minor="0">'
                '<INFORMATION/><BASE_PARAMETERS/><INSTRUMENT><INFO>'
                f'<string name="name">{name}</string></INFO>'
                '</INSTRUMENT></ZynAddSubFX-data>').encode())
    monkeypatch.setenv("LMMS_PRESETS_DIR", str(folder))
    monkeypatch.setattr(presets, "_static_preset_candidates", lambda: [])
    return folder


@pytest.fixture
def xpf_dir(tmp_path, monkeypatch):
    folder = tmp_path / "xpf"
    (folder / "TripleOscillator").mkdir(parents=True)
    (folder / "TripleOscillator" / "Test Sound.xpf").write_text(XPF)
    monkeypatch.setenv("LMMS_PRESET_DIRS", str(folder))
    monkeypatch.setattr(instrument_presets, "_system_roots", lambda: [])
    return folder


class TestTools:
    def test_add_instrument_track(self, lab):
        lab.fake.default = (0.5, 0.25)
        result = lab.add()
        assert result["track_index"] == 0
        field = level_field(result)
        assert field["source"] == "measured"
        assert field["note_peak_dbfs"] == pytest.approx(db(0.5), abs=0.05)
        assert field["chord_peak_dbfs"] == pytest.approx(db(0.25), abs=0.05)
        assert set(field) == {
            "note_peak_dbfs", "chord_peak_dbfs", "rms_dbfs", "source"}
        assert level_field(lab.add())["source"] == "cache"
        assert lab.renders == 1

    def test_a_custom_plugin_is_not_rendered(self, lab, monkeypatch):
        srv = lab.srv
        monkeypatch.setattr(srv, "_availability", lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
            set(srv.KNOWN_INSTRUMENTS) | {"mynewsynth"}, "1.3.0",
            windows=False))
        result = json.loads(srv.add_instrument_track(
            "X", instrument="mynewsynth"))
        assert result["custom_plugin"] is True
        assert "level_at_vol_100" not in result
        assert lab.renders == 0

    def test_load_zyn_preset(self, lab, zyn_dir):
        lab.add("zynaddsubfx")
        lab.add("zynaddsubfx")
        first = json.loads(lab.srv.load_zyn_preset(0, "Bass/Deep"))
        assert level_field(first)["source"] == "measured"
        assert lab.renders == 2          # the plain track, then the preset
        copy = xml_parser.find_tracks(lab.fake.projects[-1])[0].find(
            "instrumenttrack/instrument/zynaddsubfx/ZynAddSubFX-data")
        assert copy is not None          # the preset is what was measured
        again = json.loads(lab.srv.load_zyn_preset(1, "Bass/Deep"))
        assert level_field(again)["source"] == "cache"
        assert level_field(again)["note_peak_dbfs"] \
            == level_field(first)["note_peak_dbfs"]
        assert lab.renders == 2
        other = json.loads(lab.srv.load_zyn_preset(1, "Bass/Wide"))
        assert level_field(other)["source"] == "measured"
        assert lab.renders == 3

    def test_load_instrument_preset(self, lab, xpf_dir):
        lab.fake.amplitudes = (
            lambda inst: (0.1, 0.2)
            if inst.find("instrument")[0].get("coarse0") else (0.5, 1.0))
        lab.add()
        result = json.loads(lab.srv.load_instrument_preset(
            0, "TripleOscillator/Test Sound"))
        assert level_field(result)["source"] == "measured"
        assert level_field(result)["note_peak_dbfs"] \
            == pytest.approx(db(0.1), abs=0.05)
        # the preset's chord and arpeggio settings went into the measuring
        sent = xml_parser.find_tracks(lab.fake.projects[-1])[0]
        assert sent.find("instrumenttrack/chordcreator").get("chord") == "3"
        created = json.loads(lab.srv.load_instrument_preset(
            None, "TripleOscillator/Test Sound"))
        assert created["created"] is True
        assert level_field(created)["source"] == "cache"
        assert lab.renders == 2

    def test_set_audiofile_sample(self, lab, tmp_path):
        sample = tmp_path / "kick.wav"
        sample.write_bytes(b"RIFF....")
        lab.add("audiofileprocessor")
        result = json.loads(lab.srv.set_audiofile_sample(0, str(sample)))
        assert level_field(result)["source"] == "measured"
        sent = xml_parser.find_tracks(lab.fake.projects[-1])[0]
        assert sent.find(
            "instrumenttrack/instrument/audiofileprocessor").get(
                "src") == str(sample)

    def test_a_missing_sample_is_not_measured(self, lab, tmp_path):
        lab.add("audiofileprocessor")
        result = json.loads(lab.srv.set_audiofile_sample(
            0, str(tmp_path / "nope.wav")))
        assert "not found" in result["warning"]
        assert "level_at_vol_100" not in result and "note" not in result
        assert lab.renders == 0

    def test_set_instrument_file(self, lab):
        lab.add("sf2player")
        path = lab.srv._availability.files.default_soundfont
        result = json.loads(lab.srv.set_instrument_file(0, path, 0, 1))
        assert level_field(result)["source"] == "measured"
        again = json.loads(lab.srv.set_instrument_file(0, path, 0, 1))
        assert level_field(again)["source"] == "cache"
        other = json.loads(lab.srv.set_instrument_file(0, path, 0, 2))
        assert level_field(other)["source"] == "measured"

    @pytest.mark.parametrize("fake", [
        {"fail": True}, {"timeout": True}, {"os_error": True},
        {"silent": True}])
    def test_a_failed_measurement_is_a_note_not_an_error(
            self, monkeypatch, tmp_path, zyn_dir, xpf_dir, fake):
        lab = Lab(monkeypatch, tmp_path, **fake)
        sample = tmp_path / "kick.wav"
        sample.write_bytes(b"RIFF....")
        results = [lab.add()]
        lab.add("zynaddsubfx")
        results.append(json.loads(lab.srv.load_zyn_preset(1, "Bass/Deep")))
        results.append(json.loads(lab.srv.load_instrument_preset(
            0, "TripleOscillator/Test Sound")))
        lab.add("audiofileprocessor")
        results.append(json.loads(
            lab.srv.set_audiofile_sample(2, str(sample))))
        results.append(lab.add("sf2player"))
        results.append(json.loads(lab.srv.set_instrument_file(
            3, lab.srv._availability.files.default_soundfont)))
        assert len(results) == 6
        for result in results:
            assert "error" not in result
            assert "level_at_vol_100" not in result
            assert "could not be measured" in result["note"]
        assert not levels.cache_path().exists()    # nothing cached

    def test_the_note_is_added_to_the_existing_one(self, lab):
        lab.fake.fail = True
        lab.add()
        result = {"note": "First."}
        lab.srv._add_level(result, lab.proj, lab.inst())
        assert result["note"].startswith("First. The level of this sound")

    def test_no_lmms_is_a_note(self, lab, monkeypatch):
        monkeypatch.setattr(lmms_app, "find_lmms_exe", lambda: None)
        result = lab.add()
        assert "error" not in result and "LMMS was not found" in result["note"]

    def test_the_xml_is_not_touched(self, lab, zyn_dir):
        lab.add("zynaddsubfx")
        before = lab.proj.get_xml_string()
        json.loads(lab.srv.load_zyn_preset(0, "Bass/Deep"))
        text = lab.proj.get_xml_string()
        assert before != text                    # (the preset itself)
        assert "level" not in text and "dbfs" not in text

    def test_the_project_table_remembers_levels(self, lab):
        lab.add()
        assert len(lab.proj.level_table) == 1
        (key, level), = lab.proj.level_table.items()
        assert key == levels.level_key(lab.inst(), "1.3.0")
        assert isinstance(level, Level)

    def test_levels_survive_saving_and_loading(self, lab, tmp_path):
        lab.add()
        path = tmp_path / "song.mmp"
        lab.proj.save(str(path))
        loaded = LMMSProject()
        loaded.load(path)
        track = loaded._instrument_track_element(0)
        found = levels.find_level(track, "1.3.0", loaded.level_table)
        assert found is not None and found[1] == "cache"
        assert lab.renders == 1


# ──────────────────────────────────────────────────────────────────
# get_project_info: the estimate
# ──────────────────────────────────────────────────────────────────

LEVELS = {
    "tripleoscillator": Level(-10.0, -4.0, -20.0),
    "lb302": Level(-20.0, -20.0, -30.0),
    "kicker": Level(-6.0, -6.0, -16.0),
}


def level_of(inst_track):
    return LEVELS.get(levels.plugin_of(inst_track))


def project(*tracks, bpm=140):
    """A project with tracks [(instrument, volume, [(key, pos, len, vel)])]."""
    proj = LMMSProject()
    proj.new(bpm=bpm, lmms_version="1.2.2")
    for i, (instrument, volume, notes) in enumerate(tracks):
        proj.add_track(
            "instrument", f"T{i}", instrument=instrument, volume=volume)
        for key, pos, length, velocity in notes:
            proj.add_note(i, key, pos, length, velocity)
    return proj


def est(proj, lookup=level_of):
    return mix_estimate.estimate(proj.root, lookup)


def bounds(*peaks):
    """(power sum, amplitude sum) in dBFS of tracks with these peaks."""
    return (10 * math.log10(sum(10 ** (p / 10) for p in peaks)),
            20 * math.log10(sum(10 ** (p / 20) for p in peaks)))


def check_estimate(result, *peaks):
    """The result is the midpoint of the two sums and their range."""
    power, amplitude = bounds(*peaks)
    assert result["estimate_range_dbfs"] == [
        round(power, 1), round(amplitude, 1)]
    assert result["estimated_mix_peak_dbfs"] == round(
        (power + amplitude) / 2, 1)
    assert result["estimate_range_dbfs"][0] \
        <= result["estimated_mix_peak_dbfs"] \
        <= result["estimate_range_dbfs"][1]


class TestEstimate:
    def test_one_note(self):
        result = est(project(("tripleoscillator", 100, [(57, 0, 48, 100)])))
        assert result["estimated_mix_peak_dbfs"] == -10.0
        assert result["estimate_range_dbfs"] == [-10.0, -10.0]
        assert "worst_case_peak_dbfs" not in result
        assert result["loudest_bar"] == 0
        assert result["loudest_bar_tracks"] == [
            {"index": 0, "name": "T0", "peak_dbfs": -10.0}]
        assert result["estimate_basis"] == mix_estimate.BASIS_TEXT
        assert "unknown_level_tracks" not in result

    @pytest.mark.parametrize("volume,expected", [
        (100, -10.0), (50, -16.0), (200, -4.0), (25, -22.0)])
    def test_track_volume(self, volume, expected):
        result = est(project(
            ("tripleoscillator", volume, [(57, 0, 48, 100)])))
        assert result["estimated_mix_peak_dbfs"] == expected

    def test_velocity_of_the_loudest_note(self):
        result = est(project(("tripleoscillator", 100, [
            (57, 0, 48, 50), (59, 96, 48, 80), (60, 144, 48, 20)])))
        assert result["estimated_mix_peak_dbfs"] == round(-10 + db(0.8), 1)
        loud = est(project(("tripleoscillator", 100, [(57, 0, 48, 200)])))
        assert loud["estimated_mix_peak_dbfs"] == -4.0

    def test_a_silent_note_adds_nothing(self):
        assert est(project(
            ("tripleoscillator", 100, [(57, 0, 48, 0)]))) == {}
        assert est(project(
            ("tripleoscillator", 0, [(57, 0, 48, 100)]))) == {}

    def test_master_volumes(self):
        proj = project(("tripleoscillator", 100, [(57, 0, 48, 100)]))
        xml_parser.set_head_attribute(proj.root, "mastervol", "50")
        assert est(proj)["estimated_mix_peak_dbfs"] == -16.0
        xml_parser.set_model_value(
            xml_parser.ensure_master_channel(proj.root), "volume", "0.5")
        result = est(proj)
        assert result["estimated_mix_peak_dbfs"] == round(-10 + 2 * db(0.5), 1)

    def test_the_estimate_is_the_midpoint_of_power_and_amplitude_sums(self):
        proj = project(
            ("tripleoscillator", 100, [(57, 0, 48, 100)]),
            ("tripleoscillator", 100, [(57, 0, 48, 100)]))
        result = est(proj)
        # two -10 dBFS peaks: +3.0 dB as power, +6.0 dB as amplitude
        assert result["estimate_range_dbfs"] == [-7.0, -4.0]
        assert result["estimated_mix_peak_dbfs"] == -5.5
        check_estimate(result, -10.0, -10.0)

    def test_four_tracks(self):
        result = est(project(*[
            ("tripleoscillator", 100, [(57, 0, 48, 100)])] * 4))
        assert result["estimate_range_dbfs"] == [-4.0, 2.0]
        assert result["estimated_mix_peak_dbfs"] == -1.0

    def test_tracks_of_different_levels(self):
        result = est(project(
            ("tripleoscillator", 100, [(57, 0, 48, 100)]),
            ("lb302", 100, [(45, 0, 48, 100)])))
        check_estimate(result, -10.0, -20.0)
        assert [t["name"] for t in result["loudest_bar_tracks"]] \
            == ["T0", "T1"]

    def test_chord_or_single_note(self):
        together = est(project(("tripleoscillator", 100, [
            (48, 0, 48, 100), (52, 0, 48, 100)])))
        assert together["estimated_mix_peak_dbfs"] == -4.0      # chord peak
        overlapping = est(project(("tripleoscillator", 100, [
            (48, 0, 48, 100), (52, 24, 48, 100)])))
        assert overlapping["estimated_mix_peak_dbfs"] == -4.0
        one_after_the_other = est(project(("tripleoscillator", 100, [
            (48, 0, 48, 100), (52, 48, 48, 100), (55, 96, 48, 100)])))
        assert one_after_the_other["estimated_mix_peak_dbfs"] == -10.0

    def test_chord_is_judged_per_bar(self):
        result = est(project(("tripleoscillator", 100, [
            (48, 0, 48, 100),                               # bar 0: single
            (48, 192, 48, 100), (52, 192, 48, 100)])))      # bar 1: chord
        assert result["estimated_mix_peak_dbfs"] == -4.0
        assert result["loudest_bar"] == 1

    def test_loudest_bar(self):
        result = est(project(
            ("tripleoscillator", 100, [(57, 0, 48, 100)]),
            ("tripleoscillator", 200, [(57, 576, 48, 100)])))
        assert result["loudest_bar"] == 3
        assert result["estimated_mix_peak_dbfs"] == -4.0
        assert result["loudest_bar_tracks"] == [
            {"index": 1, "name": "T1", "peak_dbfs": -4.0}]

    def test_a_long_note_sounds_in_every_bar_it_spans(self):
        result = est(project(
            ("tripleoscillator", 100, [(57, 0, 576, 100)]),    # bars 0-2
            ("tripleoscillator", 100, [(57, 192, 48, 100)])))  # bar 1
        assert result["loudest_bar"] == 1
        check_estimate(result, -10.0, -10.0)

    def test_the_loudest_bar_is_the_one_with_the_highest_midpoint(self):
        # bar 0: one track at -4.5 dBFS (power = amplitude = -4.5);
        # bar 1: three at -10.5, which are -5.7 as power but -1.0 as
        # amplitude, so the midpoint (-3.3) is the louder
        proj = project(
            ("tripleoscillator", 188, [(57, 0, 48, 100)]),
            ("tripleoscillator", 94, [(57, 192, 48, 100)]),
            ("tripleoscillator", 94, [(57, 192, 48, 100)]),
            ("tripleoscillator", 94, [(57, 192, 48, 100)]))
        result = est(proj)
        assert result["loudest_bar"] == 1
        high = -10 + db(1.88)
        low = -10 + db(0.94)
        assert result["estimate_range_dbfs"][0] < round(high, 1)
        assert bounds(low, low, low)[0] < high
        check_estimate(result, low, low, low)
        assert result["loudest_bar_tracks"][0]["peak_dbfs"] == round(low, 1)

    def test_the_loudest_bar_of_a_tie_is_the_first(self):
        result = est(project(("tripleoscillator", 100, [
            (57, 192, 48, 100), (57, 576, 48, 100)])))
        assert result["loudest_bar"] == 1

    def test_unknown_levels_are_listed_and_left_out(self):
        proj = project(
            ("tripleoscillator", 100, [(57, 0, 48, 100)]),
            ("watsyn", 100, [(57, 0, 48, 100)]))
        result = est(proj)
        assert result["estimated_mix_peak_dbfs"] == -10.0
        assert result["unknown_level_tracks"] == [
            {"index": 1, "name": "T1", "instrument": "watsyn"}]

    def test_only_unknown_levels(self):
        result = est(project(("watsyn", 100, [(57, 0, 48, 100)])))
        assert result["estimated_mix_peak_dbfs"] is None
        assert result["estimate_range_dbfs"] is None
        assert "loudest_bar" not in result
        assert result["unknown_level_tracks"][0]["name"] == "T0"
        assert result["estimate_basis"]

    def test_sample_tracks_are_listed_as_unknown(self):
        proj = project(("tripleoscillator", 100, [(57, 0, 48, 100)]))
        proj.add_track("sample", "Vocals")
        xml_parser.add_sample_clip(proj.root, 1, "vox.wav", 0, 192)
        proj.add_track("sample", "Empty")
        result = est(proj)
        assert result["unknown_level_tracks"] == [
            {"index": 1, "name": "Vocals", "instrument": "sample track"}]

    def test_muted_and_soloed_tracks(self):
        proj = project(
            ("tripleoscillator", 100, [(57, 0, 48, 100)]),
            ("tripleoscillator", 100, [(57, 0, 48, 100)]),
            ("watsyn", 100, [(57, 0, 48, 100)]))
        tracks = xml_parser.find_tracks(proj.root)
        tracks[1].set("muted", "1")
        tracks[2].set("muted", "1")
        result = est(proj)
        assert result["estimated_mix_peak_dbfs"] == -10.0
        assert "unknown_level_tracks" not in result
        tracks[1].set("muted", "0")
        tracks[1].set("solo", "1")
        result = est(proj)
        assert [t["name"] for t in result["loudest_bar_tracks"]] == ["T1"]

    def test_mixer_channel_volume_and_sends(self):
        proj = project(("tripleoscillator", 100, [(57, 0, 48, 100)]))
        proj.add_mixer_channel("Bus", 0.5)
        xml_parser.set_model_value(
            xml_parser.find_tracks(proj.root)[0].find("instrumenttrack"),
            "fxch", "1")
        assert est(proj)["estimated_mix_peak_dbfs"] == -16.0
        send = xml_parser.find_mixer_channels(proj.root)[1].find("send")
        send.set("amount", "0.5")
        assert est(proj)["estimated_mix_peak_dbfs"] == -22.0
        xml_parser.find_mixer_channels(proj.root)[1].set("muted", "1")
        assert est(proj) == {}

    def test_a_channel_without_a_send_is_silent(self):
        proj = project(("tripleoscillator", 100, [(57, 0, 48, 100)]))
        proj.add_mixer_channel("Bus")
        channel = xml_parser.find_mixer_channels(proj.root)[1]
        channel.remove(channel.find("send"))
        xml_parser.find_tracks(proj.root)[0].find(
            "instrumenttrack").set("fxch", "1")
        assert est(proj) == {}

    def test_a_loop_of_sends_ends(self):
        proj = project(("tripleoscillator", 100, [(57, 0, 48, 100)]))
        proj.add_mixer_channel("A")
        proj.add_mixer_channel("B")
        channels = xml_parser.find_mixer_channels(proj.root)
        channels[1].find("send").set("channel", "2")
        channels[2].find("send").set("channel", "1")
        xml_parser.find_tracks(proj.root)[0].find(
            "instrumenttrack").set("fxch", "1")
        assert est(proj) == {}

    def test_pattern_tracks_repeat_their_pattern_in_each_clip(self):
        proj = project(("tripleoscillator", 100, [(57, 576, 48, 100)]))
        proj.add_track("pattern", "Beat")
        proj.add_note(1, 57, 0, 48, 100)
        xml_parser.add_bb_clip(proj.root, 1, position=384, length=384)
        result = est(proj)
        # the clip plays bars 2 and 3; bar 3 also has the instrument track
        assert result["loudest_bar"] == 3
        check_estimate(result, -6.0, -10.0)
        names = [(t["index"], t["name"]) for t in result["loudest_bar_tracks"]]
        assert names == [(None, "Beat/Beat"), (0, "T0")]

    def test_a_pattern_track_without_clips_is_silent(self):
        proj = LMMSProject()
        proj.new()
        proj.add_track("pattern", "Beat")
        proj.add_note(0, 57, 0, 48, 100)
        assert est(proj) == {}

    def test_time_signature(self):
        proj = project(("tripleoscillator", 100, [(57, 144, 48, 100)]))
        proj.set_time_signature(3, 4)           # 144 ticks a bar
        assert est(proj)["loudest_bar"] == 1
        proj.set_time_signature(4, 4)
        assert est(proj)["loudest_bar"] == 0
        assert mix_estimate.ticks_per_bar(proj.root) == 192
        proj.set_time_signature(6, 8)
        assert mix_estimate.ticks_per_bar(proj.root) == 144

    def test_notes_beyond_the_pattern_are_not_played(self):
        proj = project(("tripleoscillator", 100, [(57, 0, 48, 100)]))
        pattern = xml_parser.find_tracks(proj.root)[0].find("pattern")
        pattern.set("len", "96")
        xml_parser.ET.SubElement(pattern, "note", {
            "key": "57", "pos": "400", "len": "48", "vol": "100",
            "pan": "0"})
        assert est(proj)["loudest_bar"] == 0
        pattern.set("muted", "1")
        assert est(proj) == {}

    def test_no_notes_no_estimate(self):
        assert est(project(("tripleoscillator", 100, []))) == {}
        assert est(project()) == {}

    def test_the_tool_adds_the_estimate(self, lab12):
        lab12.add(volume=100)
        lab12.srv.add_note(0, 57, 0, 48, 100)
        info = json.loads(lab12.srv.get_project_info())
        assert info["estimated_mix_peak_dbfs"] == \
            levels.SEED_LEVELS["tripleoscillator"][0]
        assert info["loudest_bar"] == 0
        assert info["estimate_range_dbfs"] == [
            levels.SEED_LEVELS["tripleoscillator"][0]] * 2
        assert "worst_case_peak_dbfs" not in info
        assert info["tracks"][0]["name"] == "Tripleoscillator"
        lab12.srv.set_track_volume(0, 50)
        info = json.loads(lab12.srv.get_project_info())
        assert info["estimated_mix_peak_dbfs"] == round(
            levels.SEED_LEVELS["tripleoscillator"][0] + db(0.5), 1)
        assert lab12.renders == 0           # no render for an estimate

    def test_the_tool_uses_measured_levels_and_lists_unknown_ones(self, lab):
        lab.fake.amplitudes = {"tripleoscillator": (0.1, 0.2)}
        lab.add()
        lab.add("watsyn")
        for index in (0, 1):
            lab.srv.add_note(index, 57, 0, 48, 100)
        info = json.loads(lab.srv.get_project_info())
        assert info["estimated_mix_peak_dbfs"] == pytest.approx(
            db(0.1), abs=0.06)
        assert [t["name"] for t in info["unknown_level_tracks"]] \
            == ["Watsyn"]
        assert "estimate" in info["estimate_basis"]

    def test_an_empty_project_has_no_estimate_fields(self, lab12):
        info = json.loads(lab12.srv.get_project_info())
        assert not any(k in info for k in (
            "estimated_mix_peak_dbfs", "estimate_range_dbfs",
            "estimate_basis", "unknown_level_tracks"))


# ──────────────────────────────────────────────────────────────────
# the seed generator
# ──────────────────────────────────────────────────────────────────


class TestDocumentation:
    def test_instructions_tell_agents_about_levels(self, lab12):
        text = " ".join(lab12.srv.build_instructions(
            lab12.srv._availability, "/tmp/p").split())
        assert "level_at_vol_100" in text
        assert "estimated_mix_peak_dbfs" in text

    def test_tool_descriptions(self):
        from lmms_mcp import server as srv
        for tool in (srv.add_instrument_track, srv.load_zyn_preset,
                     srv.load_instrument_preset, srv.set_audiofile_sample,
                     srv.set_instrument_file):
            assert "level_at_vol_100" in " ".join(tool.__doc__.split())
        assert "estimated_mix_peak_dbfs" in srv.get_project_info.__doc__
        assert "estimate_range_dbfs" in srv.get_project_info.__doc__

    def test_readme(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        for word in ("level_at_vol_100", "estimated_mix_peak_dbfs",
                     "estimate_range_dbfs", "LMMS_MCP_MEASURE_LEVELS",
                     "levels.json", "unknown_level_tracks"):
            assert word in readme, word


class TestPrintSeed:
    def test_prints_the_tables(self, lab12, capsys):
        lab12.fake.amplitudes = {"kicker": (1.0, 1.0)}
        levels.main()
        out = capsys.readouterr().out
        assert "SEED_LEVELS = {" in out
        assert '"kicker": (0.0, 0.0,' in out
        assert '"tripleoscillator": (-12.0, -6.0,' in out
        assert "SEED_SOUNDFONT = {'file': 'default-GM.sf2'" in out
        assert "bitinvader: nothing to measure" in out
        assert "audiofileprocessor: nothing to measure" in out
        assert os.environ["LMMS_MCP_MEASURE_LEVELS"] == "1"


class TestWindowStats:
    def test_window_of_a_sine(self):
        audio = audio_analysis.Audio(
            SAMPLERATE, 2, sine(250, 0.5, 2.0))
        stats = audio_analysis.window_stats(audio, 0.5, 1.0)
        assert stats.peak == pytest.approx(0.5, abs=1e-6)
        assert stats.count == int(0.5 * SAMPLERATE) * 2
        assert stats.rms == pytest.approx(0.5 / math.sqrt(2), abs=1e-4)

    def test_cut_to_the_audio(self):
        audio = audio_analysis.Audio(SAMPLERATE, 2, sine(250, 0.5, 1.0))
        assert audio_analysis.window_stats(audio, 0.5, 9.0).count \
            == int(0.5 * SAMPLERATE) * 2
        assert audio_analysis.window_stats(audio, 5.0, 6.0).count == 0
        assert audio_analysis.window_stats(audio, 5.0, 6.0).peak == 0.0
        assert audio_analysis.window_stats(audio, 0.5, 0.2).count == 0
