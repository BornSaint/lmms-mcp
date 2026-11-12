"""Tests for LMMS MCP Server."""

import json
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from lmms_mcp.models import Note, midi_to_note_name, note_name_to_midi, bars_to_ticks, ticks_to_bars
from lmms_mcp.xml_parser import create_empty_project, load_project, save_project, find_tracks, add_instrument_track, add_note_to_track, add_pattern_track, add_sample_track
from lmms_mcp.project import LMMSProject


class TestModels:
    def test_midi_to_note_name(self):
        assert midi_to_note_name(60) == "C4"
        assert midi_to_note_name(69) == "A4"
        assert midi_to_note_name(0) == "C-1"
        assert midi_to_note_name(127) == "G9"

    def test_note_name_to_midi(self):
        assert note_name_to_midi("C4") == 60
        assert note_name_to_midi("A4") == 69
        assert note_name_to_midi("A#3") == 58

    def test_roundtrip_note_names(self):
        for key in [0, 24, 48, 60, 69, 72, 96, 127]:
            name = midi_to_note_name(key)
            assert note_name_to_midi(name) == key

    @pytest.mark.parametrize("name,key", [
        ("Bb2", 46), ("A#2", 46), ("Eb4", 63), ("Db3", 49),
        ("Cb4", 59), ("B#3", 60), ("E#4", 65), ("Fb4", 64),
        ("bb2", 46), ("b3", 59), ("B3", 59), ("c4", 60), ("c#4", 61),
        ("C-1", 0), ("G9", 127), ("Cb0", 11), ("B\u266d2", 46),
        ("F\u266f3", 54), ("Bbb3", 57), ("F##3", 55), (" Bb2 ", 46),
    ])
    def test_note_name_to_midi_flats_and_enharmonics(self, name, key):
        assert note_name_to_midi(name) == key

    def test_flat_and_sharp_enharmonics_agree(self):
        # Every sharp name (C#, D#, F#, G#, A#) has a flat twin one letter
        # up in the same octave (Db, Eb, Gb, Ab, Bb).
        for key in range(128):
            sharp = midi_to_note_name(key)
            if "#" not in sharp:
                continue
            flat_letter = "ABCDEFG"[("ABCDEFG".index(sharp[0]) + 1) % 7]
            octave = sharp[2:]
            assert note_name_to_midi(f"{flat_letter}b{octave}") == key

    @pytest.mark.parametrize("name", ["G#9", "Ab9", "C10", "Cb-1", "C-2"])
    def test_note_name_out_of_midi_range(self, name):
        with pytest.raises(ValueError, match="out of range.*0-127"):
            note_name_to_midi(name)

    @pytest.mark.parametrize("name", [
        "H2", "", "  ", "C", "4", "60", "BB2", "C#b4", "Cbbb4", "bbbb3",
        "C 4", "Do4", "C#",
    ])
    def test_note_name_invalid_states_format(self, name):
        with pytest.raises(ValueError) as exc:
            note_name_to_midi(name)
        msg = str(exc.value)
        assert f"Invalid note name {name!r}" in msg
        assert "Bb2" in msg and "F#3" in msg and "C4 = 60" in msg

    def test_note_name_non_string_rejected(self):
        with pytest.raises(ValueError, match="Invalid note name"):
            note_name_to_midi(60)

    def test_bars_to_ticks(self):
        assert bars_to_ticks(1) == 192
        assert bars_to_ticks(4) == 768

    def test_ticks_to_bars(self):
        assert ticks_to_bars(192) == 1.0
        assert ticks_to_bars(768) == 4.0

    def test_note_to_dict(self):
        note = Note(key=60, pos=0, length=48, volume=100)
        d = note.to_dict()
        assert d["key"] == 60
        assert d["note_name"] == "C4"

    def test_note_from_dict(self):
        note = Note.from_dict({"key": 69, "pos": 48, "length": 96})
        assert note.key == 69
        assert note.pos == 48


class TestXMLParser:
    def test_create_empty_project(self):
        root = create_empty_project(bpm=120)
        head = root.find("head")
        assert head is not None
        assert head.get("bpm") == "120"

    def test_save_load_roundtrip(self):
        root = create_empty_project(bpm=150)
        with tempfile.NamedTemporaryFile(suffix=".mmpz", delete=False) as f:
            path = f.name

        save_project(path, root)
        loaded = load_project(path)
        head = loaded.find("head")
        assert head.get("bpm") == "150"

        Path(path).unlink()

    def test_save_load_uncompressed(self):
        root = create_empty_project(bpm=130)
        with tempfile.NamedTemporaryFile(suffix=".mmp", delete=False) as f:
            path = f.name

        save_project(path, root, compressed=False)
        loaded = load_project(path)
        head = loaded.find("head")
        assert head.get("bpm") == "130"

        Path(path).unlink()

    def test_add_instrument_track(self):
        root = create_empty_project()
        add_instrument_track(root, "My Synth", instrument="tripleoscillator")
        tracks = find_tracks(root)
        assert len(tracks) == 1
        assert tracks[0].get("name") == "My Synth"
        assert tracks[0].get("type") == "0"

    @staticmethod
    def _settings(root, instrument):
        """The plugin settings element of a fresh track of `instrument`."""
        add_instrument_track(root, "T", instrument=instrument)
        inst = find_tracks(root)[-1].find("instrumenttrack/instrument")
        assert inst.get("name") == instrument
        return list(inst)[0]

    def test_vibedstrings_track_starts_with_an_active_string(self):
        # A fresh Vibed has no active string and renders silence (verified
        # with LMMS 1.2.2), so the track must be created with string 0 on.
        settings = self._settings(create_empty_project(), "vibedstrings")
        assert settings.tag == "vibedstrings"
        assert settings.get("active0") == "1"

    def test_other_instruments_get_no_default_attributes(self):
        root = create_empty_project()
        for name in ("tripleoscillator", "kicker", "organic", "zynaddsubfx"):
            assert self._settings(root, name).attrib == {}

    def test_instrument_defaults_are_not_shared_between_tracks(self):
        root = create_empty_project()
        first = self._settings(root, "vibedstrings")
        first.set("active0", "0")
        assert self._settings(root, "vibedstrings").get("active0") == "1"

    def test_add_note(self):
        root = create_empty_project()
        add_instrument_track(root, "Test Track")
        add_note_to_track(root, 0, key=60, pos=0, length=48)
        tracks = find_tracks(root)
        pattern = tracks[0].find("pattern")
        assert pattern is not None
        notes = pattern.findall("note")
        assert len(notes) == 1
        assert notes[0].get("key") == "60"


class TestProject:
    def test_new_project(self):
        proj = LMMSProject()
        proj.new(bpm=120)
        info = proj.get_info()
        assert info.bpm == 120
        assert info.time_sig_numerator == 4

    def test_set_tempo(self):
        proj = LMMSProject()
        proj.new()
        proj.set_tempo(180)
        info = proj.get_info()
        assert info.bpm == 180

    def test_add_track(self):
        proj = LMMSProject()
        proj.new()
        result = proj.add_track("instrument", "Bass", instrument="LB302")
        assert result["track_index"] == 0
        info = proj.get_info()
        assert len(info.tracks) == 1
        assert info.tracks[0].name == "Bass"

    def test_add_multiple_tracks(self):
        proj = LMMSProject()
        proj.new()
        proj.add_track("instrument", "Drums", instrument="kicker")
        proj.add_track("instrument", "Bass", instrument="tripleoscillator")
        proj.add_track("sample", "Samples")
        info = proj.get_info()
        assert len(info.tracks) == 3

    def test_add_note(self):
        proj = LMMSProject()
        proj.new()
        proj.add_track("instrument", "Melody")
        result = proj.add_note(0, key=60, pos=0, length=48)
        assert "Added note" in result["message"]

    def test_save_and_load(self):
        proj = LMMSProject()
        proj.new(bpm=160)
        proj.add_track("instrument", "Lead")

        with tempfile.NamedTemporaryFile(suffix=".mmpz", delete=False) as f:
            path = f.name

        proj.save(path)

        proj2 = LMMSProject()
        proj2.load(path)
        info = proj2.get_info()
        assert info.bpm == 160
        assert len(info.tracks) == 1

        Path(path).unlink()

    def test_to_dict(self):
        proj = LMMSProject()
        proj.new()
        proj.add_track("instrument", "Test")
        d = proj.to_dict()
        assert "bpm" in d
        assert "tracks" in d
        assert len(d["tracks"]) == 1


class TestEffects:
    """Tests for the effect chain (fxchain) module."""

    def _make_project(self):
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        return root

    def test_known_effects_complete(self):
        from lmms_mcp.effects import KNOWN_EFFECTS
        expected = {"delay", "reverbsc", "eq", "compressor", "flanger",
                    "bassbooster", "waveshaper", "stereoenhancer"}
        assert expected.issubset(set(KNOWN_EFFECTS.keys()))

    def test_add_effect_to_track(self):
        from lmms_mcp.effects import add_effect
        root = self._make_project()
        track = find_tracks(root)[0]
        result = add_effect(track, "delay")
        assert result["effect"] == "delay"
        assert result["chain_size"] == 1

    def test_track_fxchain_lives_in_instrumenttrack(self):
        """Regression: effects must go into <instrumenttrack>/<fxchain>,
        not directly under <track> - LMMS ignores the latter."""
        from lmms_mcp import server as srv
        from lmms_mcp.project import LMMSProject
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("G", instrument="tripleoscillator")
        srv.add_effect("track", 0, "delay")
        track = find_tracks(proj.root)[0]
        assert track.find("instrumenttrack/fxchain/effect") is not None
        assert track.find("fxchain") is None

    def test_add_duplicate_effect_raises(self):
        from lmms_mcp.effects import add_effect
        root = self._make_project()
        track = find_tracks(root)[0]
        add_effect(track, "delay")
        with pytest.raises(ValueError, match="already exists"):
            add_effect(track, "delay")

    def test_add_unknown_effect_raises(self):
        from lmms_mcp.effects import add_effect
        root = self._make_project()
        track = find_tracks(root)[0]
        with pytest.raises(ValueError, match="Unknown effect"):
            add_effect(track, "greatest_reverb_9000")

    def test_external_effect_rejected(self):
        from lmms_mcp.effects import add_effect
        root = self._make_project()
        track = find_tracks(root)[0]
        with pytest.raises(ValueError, match="external"):
            add_effect(track, "vsteffect")

    def test_remove_effect_by_name(self):
        from lmms_mcp.effects import add_effect, remove_effect
        root = self._make_project()
        track = find_tracks(root)[0]
        add_effect(track, "delay")
        result = remove_effect(track, "delay")
        assert result["removed"] == "delay"
        assert result["chain_size"] == 0

    def test_remove_effect_by_position(self):
        from lmms_mcp.effects import add_effect, remove_effect
        root = self._make_project()
        track = find_tracks(root)[0]
        add_effect(track, "delay")
        add_effect(track, "eq")
        result = remove_effect(track, 0)
        assert result["removed"] == "delay"

    def test_list_effects(self):
        from lmms_mcp.effects import add_effect, list_effects
        root = self._make_project()
        track = find_tracks(root)[0]
        add_effect(track, "delay", wet=0.7)
        add_effect(track, "reverbsc", enabled=False)
        effects = list_effects(track)
        assert len(effects) == 2
        assert effects[0]["name"] == "delay"
        assert effects[0]["wet"] == 0.7
        assert effects[1]["enabled"] is False

    def test_toggle_effect(self):
        from lmms_mcp.effects import add_effect, set_effect_enabled
        root = self._make_project()
        track = find_tracks(root)[0]
        add_effect(track, "delay")
        result = set_effect_enabled(track, "delay", False)
        assert result["enabled"] is False

    def test_fxchain_survives_roundtrip(self):
        from lmms_mcp.effects import add_effect
        root = self._make_project()
        track = find_tracks(root)[0]
        add_effect(track, "reverbsc")

        with tempfile.NamedTemporaryFile(suffix=".mmpz", delete=False) as f:
            path = f.name
        save_project(path, root)
        root2 = load_project(path)
        fxchain = find_tracks(root2)[0].find("fxchain")
        eff = fxchain.find("effect")
        assert eff.get("name") == "reverbsc"
        Path(path).unlink()


class TestZynAddSubFX:
    """Tests for ZynAddSubFX preset and parameter support."""

    def _make_zyn_project(self):
        root = create_empty_project()
        add_instrument_track(root, "Zyn", instrument="zynaddsubfx")
        return root

    ZYN_PRESET_XML = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE ZynAddSubFX-data>\n'
        '<ZynAddSubFX-data version-major="1" version-minor="0">'
        '<INFORMATION><par_bool name="PADsynth_used" value="no"/>'
        '</INFORMATION>'
        '<BASE_PARAMETERS><par name="max_midi_parts" value="16"/>'
        '</BASE_PARAMETERS>'
        '<INSTRUMENT><INFO>'
        "<string name=\"name\">TestPatch</string>"
        '</INFO></INSTRUMENT></ZynAddSubFX-data>'
    )

    @staticmethod
    def _zyn_inst(root):
        return find_tracks(root)[0].find("instrumenttrack/instrument")

    @staticmethod
    def _part_par(part, name):
        return part.find(f"*[@name='{name}']").get("value")

    def test_embed_preset(self):
        """LMMS only loads a preset laid out as instrument/zynaddsubfx/
        ZynAddSubFX-data/MASTER/PART[0]/INSTRUMENT (Master::loadXML needs
        the MASTER branch)."""
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = self._make_zyn_project()
        result = embed_zyn_preset(root, 0, self.ZYN_PRESET_XML)
        assert result["preset"] == "TestPatch"
        inst = self._zyn_inst(root)
        # not in the legacy place directly under <instrument>
        assert inst.find("ZynAddSubFX-data") is None
        data = inst.find("zynaddsubfx/ZynAddSubFX-data")
        assert data is not None
        assert data.get("version-major") == "1"
        assert data.find("INFORMATION/par_bool") is not None
        assert data.find("BASE_PARAMETERS/par") is not None
        # the instrument lives inside the enabled part 0, not at top level
        assert data.find("INSTRUMENT") is None
        part = data.find("MASTER/PART[@id='0']")
        assert part is not None
        assert self._part_par(part, "enabled") == "yes"
        assert self._part_par(part, "panning") == "64"
        assert self._part_par(part, "min_key") == "0"
        assert self._part_par(part, "max_key") == "127"
        assert self._part_par(part, "key_shift") == "64"
        assert self._part_par(part, "rcv_chn") == "0"
        assert self._part_par(part, "note_on") == "yes"
        assert self._part_par(part, "poly_mode") == "yes"
        name = part.find("INSTRUMENT/INFO/string[@name='name']")
        assert name.text == "TestPatch"
        assert data.find("MASTER/par[@name='volume']") is not None

    def test_embed_preset_twice_keeps_one_data_block(self):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = self._make_zyn_project()
        embed_zyn_preset(root, 0, self.ZYN_PRESET_XML)
        embed_zyn_preset(root, 0, self.ZYN_PRESET_XML.replace(
            "TestPatch", "Other"))
        inst = self._zyn_inst(root)
        assert len(inst.findall("zynaddsubfx")) == 1
        assert len(inst.findall("zynaddsubfx/ZynAddSubFX-data")) == 1
        assert len(inst.findall("ZynAddSubFX-data")) == 0
        names = inst.findall(
            "zynaddsubfx/ZynAddSubFX-data/MASTER/PART/INSTRUMENT/INFO/"
            "string[@name='name']")
        assert [n.text for n in names] == ["Other"]

    def test_embed_preset_cleans_legacy_layout(self):
        """Old projects had the data and knobs on <instrument> itself."""
        import xml.etree.ElementTree as ET
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = self._make_zyn_project()
        inst = self._zyn_inst(root)
        inst.set("filterfreq", "70")
        inst.set("modifiedcontrollers", "2")
        ET.SubElement(inst, "ZynAddSubFX-data")
        embed_zyn_preset(root, 0, self.ZYN_PRESET_XML)
        assert inst.find("ZynAddSubFX-data") is None
        assert "filterfreq" not in inst.attrib
        assert "modifiedcontrollers" not in inst.attrib
        settings = inst.find("zynaddsubfx")
        assert settings.get("filterfreq") == "70"
        assert settings.get("modifiedcontrollers") == "74"
        assert len(settings.findall("ZynAddSubFX-data")) == 1

    def test_embed_preset_without_instrument_branch(self):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = self._make_zyn_project()
        with pytest.raises(ValueError, match="INSTRUMENT"):
            embed_zyn_preset(root, 0, "<ZynAddSubFX-data/>")

    def test_embed_preset_wrong_instrument(self):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = create_empty_project()
        add_instrument_track(root, "Not Zyn")  # tripleoscillator default
        with pytest.raises(ValueError, match="not 'zynaddsubfx'"):
            embed_zyn_preset(root, 0, self.ZYN_PRESET_XML)

    def test_embed_invalid_xml(self):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = self._make_zyn_project()
        with pytest.raises(ValueError, match="Invalid preset XML"):
            embed_zyn_preset(root, 0, "<not-valid-xml")

    def test_embed_wrong_root_element(self):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = self._make_zyn_project()
        with pytest.raises(ValueError, match="expected"):
            embed_zyn_preset(root, 0, "<something-else/>")

    def test_set_instrument_params(self):
        """Knobs are attributes of <zynaddsubfx>, not of <instrument>, and
        modifiedcontrollers lists their MIDI CC numbers (portamento=65,
        filterfreq=74, ...)."""
        from lmms_mcp.xml_parser import set_instrument_params
        root = self._make_zyn_project()
        result = set_instrument_params(
            root, 0, {"filterfreq": 100, "portamento": 30}
        )
        assert result["applied"]["filterfreq"] == 100
        assert result["modifiedcontrollers"] == [65, 74]
        inst = self._zyn_inst(root)
        assert "filterfreq" not in inst.attrib
        assert "portamento" not in inst.attrib
        assert "modifiedcontrollers" not in inst.attrib
        settings = inst.find("zynaddsubfx")
        assert settings.get("filterfreq") == "100"
        assert settings.get("portamento") == "30"
        assert settings.get("modifiedcontrollers") == "65,74"
        assert settings.get("forwardmidicc") == "1"
        # LMMS ignores the knobs of a settings element without children,
        # so a default part is written.
        data = settings.find("ZynAddSubFX-data")
        assert self._part_par(data.find("MASTER/PART[@id='0']"),
                              "enabled") == "yes"

    def test_all_knob_controller_numbers(self):
        from lmms_mcp.xml_parser import set_instrument_params
        root = self._make_zyn_project()
        set_instrument_params(root, 0, {
            "portamento": 1, "filterfreq": 2, "filterq": 3, "bandwidth": 4,
            "fmgain": 5, "rescenterfreq": 6, "resbandwidth": 7,
        })
        settings = self._zyn_inst(root).find("zynaddsubfx")
        assert settings.get("modifiedcontrollers") == "65,71,74,75,76,77,78"
        assert settings.get("resbandwidth") == "7"

    def test_set_params_merges_modified_controllers(self):
        from lmms_mcp.xml_parser import set_instrument_params
        root = self._make_zyn_project()
        settings = self._zyn_inst(root).find("zynaddsubfx")
        settings.set("modifiedcontrollers", "1,74,")
        settings.set("forwardmidicc", "0")
        result = set_instrument_params(root, 0, {"filterq": 90})
        assert result["modifiedcontrollers"] == [1, 71, 74]
        assert settings.get("modifiedcontrollers") == "1,71,74"
        assert settings.get("forwardmidicc") == "0"  # not overwritten

    def test_set_params_keeps_loaded_preset(self):
        from lmms_mcp.xml_parser import embed_zyn_preset, set_instrument_params
        root = self._make_zyn_project()
        embed_zyn_preset(root, 0, self.ZYN_PRESET_XML)
        set_instrument_params(root, 0, {"filterfreq": 20})
        settings = self._zyn_inst(root).find("zynaddsubfx")
        assert len(settings.findall("ZynAddSubFX-data")) == 1
        assert settings.find(
            "ZynAddSubFX-data/MASTER/PART[@id='0']/INSTRUMENT") is not None
        # and loading a preset afterwards keeps the knobs
        embed_zyn_preset(root, 0, self.ZYN_PRESET_XML)
        assert settings.get("filterfreq") == "20"
        assert settings.get("modifiedcontrollers") == "74"

    def test_set_params_migrates_legacy_attributes(self):
        from lmms_mcp.xml_parser import set_instrument_params
        root = self._make_zyn_project()
        inst = self._zyn_inst(root)
        inst.set("filterq", "50")
        inst.set("modifiedcontrollers", "3")  # made-up legacy index
        set_instrument_params(root, 0, {"filterfreq": 20})
        assert "filterq" not in inst.attrib
        assert "modifiedcontrollers" not in inst.attrib
        settings = inst.find("zynaddsubfx")
        assert settings.get("filterq") == "50"
        assert settings.get("filterfreq") == "20"
        assert settings.get("modifiedcontrollers") == "71,74"

    def test_set_params_unknown_parameter(self):
        from lmms_mcp.xml_parser import set_instrument_params
        root = self._make_zyn_project()
        with pytest.raises(ValueError, match="Unknown"):
            set_instrument_params(root, 0, {"cutoff": 10})

    def test_set_params_wrong_instrument(self):
        from lmms_mcp.xml_parser import set_instrument_params
        root = create_empty_project()
        add_instrument_track(root, "Not Zyn")
        with pytest.raises(ValueError, match="not 'zynaddsubfx'"):
            set_instrument_params(root, 0, {"filterfreq": 10})

    def test_set_params_out_of_range_track(self):
        from lmms_mcp.xml_parser import set_instrument_params
        root = self._make_zyn_project()
        with pytest.raises(ValueError, match="out of range"):
            set_instrument_params(root, 99, {"portamento": 10})

    def test_find_track_element_range_check(self):
        from lmms_mcp.xml_parser import find_track_element
        root = self._make_zyn_project()
        with pytest.raises(ValueError, match="out of range"):
            find_track_element(root, 5)

    def test_preset_and_params_survive_save_load(self, tmp_path):
        from lmms_mcp.xml_parser import embed_zyn_preset, set_instrument_params
        root = self._make_zyn_project()
        embed_zyn_preset(root, 0, self.ZYN_PRESET_XML)
        set_instrument_params(root, 0, {"filterfreq": 20, "filterq": 100})
        path = tmp_path / "zyn.mmpz"
        save_project(path, root)
        loaded = load_project(path)
        inst = self._zyn_inst(loaded)
        assert inst.get("name") == "zynaddsubfx"
        assert inst.find("ZynAddSubFX-data") is None
        settings = inst.find("zynaddsubfx")
        assert settings.get("filterfreq") == "20"
        assert settings.get("filterq") == "100"
        assert settings.get("modifiedcontrollers") == "71,74"
        assert settings.get("forwardmidicc") == "1"
        instrument = settings.find(
            "ZynAddSubFX-data/MASTER/PART[@id='0']/INSTRUMENT")
        assert instrument.find("INFO/string[@name='name']").text == "TestPatch"
        part = settings.find("ZynAddSubFX-data/MASTER/PART[@id='0']")
        assert self._part_par(part, "enabled") == "yes"

    def test_server_tools(self, monkeypatch):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("Z", instrument="zynaddsubfx")
        from lmms_mcp.presets import PresetMatch
        monkeypatch.setattr(
            srv.zyn_presets, "load_preset",
            lambda name: (
                self.ZYN_PRESET_XML,
                PresetMatch(Path("/x/Any/0001-Thing.xiz"), "Any/Thing"),
            ),
        )
        loaded = json.loads(srv.load_zyn_preset(0, "Any/Thing"))
        # `preset` is the file-derived name, not the name inside the file.
        assert loaded["preset"] == "Any/Thing"
        assert loaded["internal_name"] == "TestPatch"
        params = json.loads(srv.set_zyn_params(0, filterfreq=20))
        assert params["applied"] == {"filterfreq": 20}
        assert params["modifiedcontrollers"] == [74]
        inst = self._zyn_inst(proj.root)
        assert inst.find("zynaddsubfx").get("filterfreq") == "20"
        assert inst.get("filterfreq") is None
        assert "error" in json.loads(srv.set_zyn_params(0, filterfreq=200))


class TestArrangement:
    """Tests for song editor arrangement (clips, patterns, samples)."""

    def _project_with_tracks(self):
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        add_pattern_track(root, "BB")
        return root

    def test_place_instrument_pattern(self):
        from lmms_mcp.xml_parser import place_instrument_pattern
        root = self._project_with_tracks()
        pattern = place_instrument_pattern(root, 0, position=768, name="Drop")
        assert pattern.get("pos") == "768"
        assert pattern.get("name") == "Drop"

    def test_place_pattern_wrong_track_type(self):
        from lmms_mcp.xml_parser import place_instrument_pattern
        root = self._project_with_tracks()
        with pytest.raises(ValueError, match="not an instrument track"):
            place_instrument_pattern(root, 1, position=0)

    def test_add_sample_clip(self):
        from lmms_mcp.xml_parser import add_sample_track, add_sample_clip
        root = self._project_with_tracks()
        add_sample_track(root, "FX")
        clip = add_sample_clip(root, 2, "drums/kick01.ogg", 192, 384)
        assert clip.get("src") == "drums/kick01.ogg"
        assert clip.get("pos") == "192"

    def test_add_sample_clip_requires_sample_track(self):
        from lmms_mcp.xml_parser import add_sample_clip
        root = self._project_with_tracks()
        with pytest.raises(ValueError, match="not a sample track"):
            add_sample_clip(root, 0, "test.wav")

    def test_get_arrangement_sorted(self):
        from lmms_mcp.xml_parser import (
            place_instrument_pattern, add_bb_clip, get_arrangement,
            add_sample_track, add_sample_clip,
        )
        root = self._project_with_tracks()
        place_instrument_pattern(root, 0, 768)
        add_bb_clip(root, 1, 0)
        add_sample_track(root, "S")
        add_sample_clip(root, 2, "a.wav", 384)

        clips = get_arrangement(root)
        kinds = [c["kind"] for c in clips]
        positions = [c["pos"] for c in clips]
        assert kinds == ["bbtco", "sampleclip", "pattern"]
        assert positions == sorted(positions)

    def test_move_and_delete_clip(self):
        from lmms_mcp.xml_parser import (
            place_instrument_pattern, move_clip, delete_clip, get_arrangement,
        )
        root = self._project_with_tracks()
        place_instrument_pattern(root, 0, 192)
        result = move_clip(root, 0, 192, 960)
        assert result["new_pos"] == 960
        clips = get_arrangement(root)
        assert clips[0]["pos"] == 960
        result = delete_clip(root, 0, 960)
        assert len(get_arrangement(root)) == 0

    def test_move_nonexistent_clip(self):
        from lmms_mcp.xml_parser import move_clip
        root = self._project_with_tracks()
        with pytest.raises(ValueError, match="No clip found"):
            move_clip(root, 0, 999, 100)


class TestAutomation:
    """Tests for automation tracks and curves."""

    def _root(self):
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        return root

    def test_automate_attribute_creates_element(self):
        from lmms_mcp.xml_parser import automate_attribute
        root = self._root()
        track = find_tracks(root)[0]
        inst = track.find("instrumenttrack")
        model_id = automate_attribute(inst, "vol", "100")
        assert model_id > 0
        el = inst.find("vol")
        assert el is not None
        assert el.get("id") == str(model_id)
        assert "vol" not in inst.attrib

    def test_automate_attribute_idempotent(self):
        from lmms_mcp.xml_parser import automate_attribute
        root = self._root()
        track = find_tracks(root)[0]
        inst = track.find("instrumenttrack")
        id1 = automate_attribute(inst, "vol", "100")
        id2 = automate_attribute(inst, "vol", "100")
        assert id1 == id2
        assert len(inst.findall("vol")) == 1

    def test_add_automation_curve_track(self):
        from lmms_mcp.xml_parser import add_automation_curve_track
        root = self._root()
        track = add_automation_curve_track(
            root, "Ramp", [(0, 120), (1536, 140)], target_id=42
        )
        assert track.get("type") == "5"
        pattern = track.find("automationpattern")
        times = pattern.findall("time")
        assert len(times) == 2
        assert times[0].get("value") == "120"
        obj = pattern.find("object")
        assert obj.get("id") == "42"

    def test_automation_pattern_starts_at_first_point(self):
        """Pattern is placed at the first point; times are relative to it."""
        from lmms_mcp.xml_parser import add_automation_curve_track
        root = self._root()
        track = add_automation_curve_track(
            root, "Late", [(5376, 20), (6125, 60), (6144, 45)], target_id=1
        )
        pattern = track.find("automationpattern")
        assert pattern.get("pos") == "5376"
        assert pattern.get("autoresize") == "1"
        times = [(t.get("pos"), t.get("value")) for t in pattern.findall("time")]
        assert times == [("0", "20"), ("749", "60"), ("768", "45")]
        # Length covers the whole span (last relative time 768 -> >= 769)
        assert int(pattern.get("len")) >= 769

    def test_automation_pattern_len_covers_long_span(self):
        from lmms_mcp.xml_parser import add_automation_curve_track
        root = self._root()
        track = add_automation_curve_track(
            root, "Long", [(192, 10), (192 + 3000, 90)], target_id=1
        )
        pattern = track.find("automationpattern")
        assert pattern.get("pos") == "192"
        assert pattern.find("time[last()]").get("pos") == "3000"
        assert int(pattern.get("len")) == 3001

    def test_automation_pattern_from_song_start(self):
        """First point at tick 0 keeps the pattern at the song start."""
        from lmms_mcp.xml_parser import add_automation_curve_track
        root = self._root()
        track = add_automation_curve_track(
            root, "Ramp", [(0, 120), (1536, 140)], target_id=1
        )
        pattern = track.find("automationpattern")
        assert pattern.get("pos") == "0"
        assert [t.get("pos") for t in pattern.findall("time")] == ["0", "1536"]
        assert int(pattern.get("len")) == 1537

    def test_automation_single_point(self):
        from lmms_mcp.xml_parser import add_automation_curve_track
        root = self._root()
        track = add_automation_curve_track(root, "One", [(768, 33)], target_id=1)
        pattern = track.find("automationpattern")
        assert pattern.get("pos") == "768"
        times = pattern.findall("time")
        assert len(times) == 1
        assert times[0].get("pos") == "0"
        assert times[0].get("value") == "33"
        assert int(pattern.get("len")) == 192

    def test_automation_unsorted_points(self):
        from lmms_mcp.xml_parser import add_automation_curve_track
        root = self._root()
        track = add_automation_curve_track(
            root, "Unsorted", [(1000, 3), (400, 1), (700, 2)], target_id=1
        )
        pattern = track.find("automationpattern")
        assert pattern.get("pos") == "400"
        times = [(t.get("pos"), t.get("value")) for t in pattern.findall("time")]
        assert times == [("0", "1"), ("300", "2"), ("600", "3")]

    def test_automation_position_offset(self):
        """`position` is added to the pattern start; times stay relative."""
        from lmms_mcp.xml_parser import add_automation_curve_track
        root = self._root()
        track = add_automation_curve_track(
            root, "Offset", [(100, 1), (300, 2)], target_id=1, position=192
        )
        pattern = track.find("automationpattern")
        assert pattern.get("pos") == "292"
        assert [t.get("pos") for t in pattern.findall("time")] == ["0", "200"]

    def test_get_arrangement_reports_absolute_automation_position(self):
        from lmms_mcp.xml_parser import add_automation_curve_track, get_arrangement
        root = self._root()
        add_automation_curve_track(
            root, "Late", [(5376, 20), (6144, 45)], target_id=1
        )
        clips = [c for c in get_arrangement(root) if c["kind"] == "automation"]
        assert len(clips) == 1
        assert clips[0]["pos"] == 5376
        assert clips[0]["len"] == 769
        assert clips[0]["points"] == 2

    def test_add_automation_tool_uses_absolute_bars(self):
        """The add_automation tool keeps taking absolute bars."""
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        add_instrument_track(proj.root, "Strings")
        result = json.loads(srv.add_automation(
            target_type="track", param="volume", target_index=0,
            points=[{"bar": 28, "value": 20}, {"bar": 31.9, "value": 60},
                    {"bar": 32, "value": 45}],
        ))
        assert "error" not in result
        track = find_tracks(proj.root)[result["automation_track_index"]]
        pattern = track.find("automationpattern")
        assert pattern.get("pos") == "5376"
        assert [t.get("pos") for t in pattern.findall("time")] == [
            "0", "749", "768"]

    def test_add_automation_track_tool_creates_empty_track(self):
        """The add_automation_track MCP tool creates an empty track."""
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        before = len(find_tracks(proj.root))
        result = json.loads(srv.add_automation_track("Auto"))
        assert "error" not in result
        tracks = find_tracks(proj.root)
        assert len(tracks) == before + 1
        track = tracks[-1]
        assert track.get("type") == "5"
        assert track.get("name") == "Auto"
        assert track.find("automationtrack") is not None
        assert track.find("automationpattern") is None

    def test_add_automation_curve_track_empty_points(self):
        from lmms_mcp.xml_parser import add_automation_curve_track
        root = self._root()
        with pytest.raises(ValueError, match="At least one point"):
            add_automation_curve_track(root, "Empty", [])

    def test_resolve_song_target(self):
        from lmms_mcp.xml_parser import resolve_automation_target
        root = self._root()
        model_id, value = resolve_automation_target(root, "song", 0, "tempo")
        assert value == 140.0
        head = root.find("head")
        bpm_el = head.find("bpm")
        assert bpm_el is not None
        assert bpm_el.get("id") == str(model_id)

    def test_resolve_track_volume(self):
        from lmms_mcp.xml_parser import resolve_automation_target
        root = self._root()
        model_id, value = resolve_automation_target(
            root, "track", 0, "volume"
        )
        assert value == 100.0
        inst = find_tracks(root)[0].find("instrumenttrack")
        assert inst.find("vol").get("id") == str(model_id)

    def test_resolve_invalid_param(self):
        from lmms_mcp.xml_parser import resolve_automation_target
        root = self._root()
        with pytest.raises(ValueError, match="Unknown song parameter"):
            resolve_automation_target(root, "song", 0, "nonexistent")

    def test_resolve_mixer_out_of_range(self):
        from lmms_mcp.xml_parser import resolve_automation_target
        root = self._root()
        with pytest.raises(ValueError, match="out of range"):
            resolve_automation_target(root, "mixer", 99, "volume")

    def test_full_automation_roundtrip(self):
        """Automation must survive .mmpz save/load with intact ID links."""
        from lmms_mcp.xml_parser import (
            resolve_automation_target, add_automation_curve_track,
        )
        root = self._root()
        model_id, _ = resolve_automation_target(root, "song", 0, "tempo")
        add_automation_curve_track(
            root, "Tempo Ramp", [(0, 120), (1536, 140)], target_id=model_id
        )

        with tempfile.NamedTemporaryFile(suffix=".mmpz", delete=False) as f:
            path = f.name
        save_project(path, root)
        root2 = load_project(path)

        # Model element with id survives
        bpm_el = root2.find("head/bpm")
        assert bpm_el is not None
        saved_id = int(bpm_el.get("id"))

        # Automation object reference matches the model id
        auto_tracks = [
            t for t in find_tracks(root2) if t.get("type") == "5"
        ]
        assert len(auto_tracks) == 1
        obj = auto_tracks[0].find("automationpattern/object")
        assert int(obj.get("id")) == saved_id
        Path(path).unlink()


class TestLmmsApp:
    """Tests for installed-LMMS detection."""

    def test_find_exe(self):
        from lmms_mcp import lmms_app
        exe = lmms_app.find_lmms_exe()
        # On the dev machine LMMS is installed; skip elsewhere
        if exe is None:
            pytest.skip("LMMS not installed")
        assert exe.is_file()

    def test_check_plugin_known_builtin(self):
        from lmms_mcp import lmms_app
        ok, _ = lmms_app.check_plugin_available("tripleoscillator")
        assert ok is True

    def test_check_static_plugin_only_built_in_on_windows(self, monkeypatch):
        from types import SimpleNamespace

        from lmms_mcp import lmms_app
        monkeypatch.setattr(
            lmms_app, "get_installed_plugins", lambda: {"nes", "sid"})
        monkeypatch.setattr(lmms_app, "sys", SimpleNamespace(platform="win32"))
        ok, reason = lmms_app.check_plugin_available("freeboy")
        assert ok is True
        assert "built-in" in reason
        # On Linux there is no such thing: FreeBoy needs a library
        monkeypatch.setattr(lmms_app, "sys", SimpleNamespace(platform="linux"))
        ok, reason = lmms_app.check_plugin_available("freeboy")
        assert ok is False
        assert "not included" in reason

    def test_check_freeboy_is_the_papu_library(self, monkeypatch):
        from lmms_mcp import lmms_app
        monkeypatch.setattr(
            lmms_app, "get_installed_plugins", lambda: {"nes", "papu"})
        ok, reason = lmms_app.check_plugin_available("freeboy")
        assert ok is True
        assert reason == "installed (alias)"

    def test_check_plugin_matches_dll_reality(self):
        """check_plugin_available must agree with the actual plugins dir."""
        from lmms_mcp import lmms_app
        if lmms_app.find_lmms_exe() is None:
            pytest.skip("LMMS not installed")
        installed = lmms_app.get_installed_plugins()
        # slicert: available iff its DLL exists (user may add/remove it)
        ok, _ = lmms_app.check_plugin_available("slicert")
        assert ok == ("slicert" in installed)

    def test_check_unknown_plugin_reports_missing(self):
        from lmms_mcp import lmms_app
        if lmms_app.find_lmms_exe() is None:
            pytest.skip("LMMS not installed")
        ok, reason = lmms_app.check_plugin_available(
            "definitely_not_a_real_plugin_12345"
        )
        assert ok is False
        assert "not included" in reason

    def test_classify_plugins(self):
        from lmms_mcp import lmms_app
        if lmms_app.find_lmms_exe() is None:
            pytest.skip("LMMS not installed")
        result = lmms_app.classify_installed_plugins(
            {"tripleoscillator"}, {"delay"}
        )
        assert "tripleoscillator" in result["instruments"]
        assert "delay" in result["effects"]
        # everything else lands in unknown
        all_names = (
            result["instruments"] + result["effects"] + result["unknown"]
        )
        installed = lmms_app.get_installed_plugins()
        for name in installed:
            if not name.startswith("lib"):
                assert name in all_names

    def test_find_vst_plugins(self):
        from lmms_mcp import lmms_app
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dll = Path(tmp) / "fakevst.dll"
            dll.write_bytes(b"MZ fake")
            (Path(tmp) / "sub").mkdir()
            nested = Path(tmp) / "sub" / "nested.dll"
            nested.write_bytes(b"MZ fake")
            results = lmms_app.find_vst_plugins(tmp)
            assert len(results) == 2
            shallow = lmms_app.find_vst_plugins(tmp, recursive=False)
            assert len(shallow) == 1
            assert shallow[0]["name"] == "fakevst"

    def test_find_vst_plugins_missing_dir(self):
        from lmms_mcp import lmms_app
        with pytest.raises(ValueError, match="not found"):
            lmms_app.find_vst_plugins("Z:/no/such/dir")


class TestLmmsAppDetection:
    """Machine-independent tests for exe/plugin detection (fake dirs)."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("LMMS_EXECUTABLE", raising=False)
        monkeypatch.delenv("LMMS_PLUGINS_DIR", raising=False)
        return monkeypatch

    @staticmethod
    def _fake_linux_plugins(root):
        plugins = root / "lmms"
        (plugins / "ladspa").mkdir(parents=True)
        for name in (
            "libtripleoscillator.so", "libOPL2.so", "libreverbsc.so",
            "libZynAddSubFxCore.so", "RemoteZynAddSubFx",
        ):
            (plugins / name).write_bytes(b"\x7fELF fake")
        (plugins / "ladspa" / "amp_1181.so").write_bytes(b"\x7fELF fake")
        return plugins

    def test_linux_so_plugins_detected(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        plugins = self._fake_linux_plugins(tmp_path)
        clean_env.setenv("LMMS_PLUGINS_DIR", str(plugins))
        assert lmms_app.get_plugins_dir() == plugins
        # lib prefix stripped, lowercased; support lib, ladspa/ and the
        # RemoteZynAddSubFx executable are not plugins
        assert lmms_app.get_installed_plugins() == {
            "tripleoscillator", "opl2", "reverbsc",
        }

    def test_installed_stems_keep_the_library_case(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        plugins = self._fake_linux_plugins(tmp_path)
        (plugins / "libpapu.so").write_bytes(b"\x7fELF fake")
        clean_env.setenv("LMMS_PLUGINS_DIR", str(plugins))
        # libOPL2.so is loaded as "OPL2" (case-sensitive), not "opl2"
        assert lmms_app.get_installed_stems() == {
            "tripleoscillator": "tripleoscillator", "opl2": "OPL2",
            "reverbsc": "reverbsc", "papu": "papu",
        }
        # the lowercase set is unchanged
        assert lmms_app.get_installed_plugins() == {
            "tripleoscillator", "opl2", "reverbsc", "papu",
        }

    def test_installed_stems_empty_without_plugins(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        clean_env.setenv("LMMS_PLUGINS_DIR", str(tmp_path / "missing"))
        clean_env.setattr(lmms_app, "get_plugins_dir", lambda: None)
        assert lmms_app.get_installed_stems() == {}

    def test_samples_dir_linux_layout(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        exe = tmp_path / "prefix" / "bin" / "lmms"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"x")
        samples = tmp_path / "prefix" / "share" / "lmms" / "samples"
        samples.mkdir(parents=True)
        clean_env.setenv("LMMS_EXECUTABLE", str(exe))
        assert lmms_app.get_samples_dir() == samples

    def test_samples_dir_windows_layout(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        exe = tmp_path / "LMMS" / "lmms.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"x")
        samples = tmp_path / "LMMS" / "data" / "samples"
        samples.mkdir(parents=True)
        clean_env.setenv("LMMS_EXECUTABLE", str(exe))
        assert lmms_app.get_samples_dir() == samples

    def test_linux_plugin_availability_and_alias(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        clean_env.setenv(
            "LMMS_PLUGINS_DIR", str(self._fake_linux_plugins(tmp_path))
        )
        ok, _ = lmms_app.check_plugin_available("tripleoscillator")
        assert ok is True
        # opulenz is shipped as OPL2 in LMMS 1.2.x
        ok, reason = lmms_app.check_plugin_available("opulenz")
        assert ok is True
        assert "alias" in reason
        ok, reason = lmms_app.check_plugin_available("slicert")
        assert ok is False
        assert "not included" in reason

    def test_linux_classify(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        clean_env.setenv(
            "LMMS_PLUGINS_DIR", str(self._fake_linux_plugins(tmp_path))
        )
        result = lmms_app.classify_installed_plugins(
            {"tripleoscillator", "opulenz"}, {"reverbsc", "delay"}
        )
        assert result["instruments"] == ["opl2", "tripleoscillator"]
        assert result["effects"] == ["reverbsc"]
        assert result["unknown"] == []

    def test_windows_dll_plugins_unchanged(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        plugins = tmp_path / "plugins"
        plugins.mkdir()
        for name in ("tripleoscillator.dll", "libfoo.dll", "reverbsc.dll"):
            (plugins / name).write_bytes(b"MZ fake")
        clean_env.setenv("LMMS_PLUGINS_DIR", str(plugins))
        # no prefix stripping for .dll: lib*.dll are support libraries
        assert lmms_app.get_installed_plugins() == {
            "tripleoscillator", "libfoo", "reverbsc",
        }
        result = lmms_app.classify_installed_plugins(
            {"tripleoscillator"}, {"reverbsc"}
        )
        assert result["instruments"] == ["tripleoscillator"]
        assert result["effects"] == ["reverbsc"]
        assert "libfoo" not in result["unknown"]

    def test_plugins_dir_override_wins_over_exe(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        bin_dir = tmp_path / "prefix" / "bin"
        bin_dir.mkdir(parents=True)
        exe = bin_dir / "lmms"
        exe.write_bytes(b"")
        lib = tmp_path / "prefix" / "lib" / "lmms"
        lib.mkdir(parents=True)
        (lib / "libkicker.so").write_bytes(b"")
        override = tmp_path / "other"
        override.mkdir()
        (override / "libmonstro.so").write_bytes(b"")
        clean_env.setenv("LMMS_EXECUTABLE", str(exe))
        assert lmms_app.get_plugins_dir() == lib
        clean_env.setenv("LMMS_PLUGINS_DIR", str(override))
        assert lmms_app.get_plugins_dir() == override
        assert lmms_app.get_installed_plugins() == {"monstro"}

    def test_plugins_dir_relative_to_symlinked_exe(self, tmp_path, clean_env):
        """<prefix>/bin/lmms -> <real prefix>/lib/lmms via symlink."""
        from lmms_mcp import lmms_app
        real_bin = tmp_path / "opt" / "lmms" / "bin"
        real_bin.mkdir(parents=True)
        (real_bin / "lmms").write_bytes(b"")
        lib = tmp_path / "opt" / "lmms" / "lib" / "lmms"
        lib.mkdir(parents=True)
        (lib / "libkicker.so").write_bytes(b"")
        link_dir = tmp_path / "usr" / "bin"
        link_dir.mkdir(parents=True)
        try:
            (link_dir / "lmms").symlink_to(real_bin / "lmms")
        except OSError:
            pytest.skip("symlinks not supported")
        clean_env.setenv("LMMS_EXECUTABLE", str(link_dir / "lmms"))
        assert lmms_app.get_plugins_dir() == lib
        assert lmms_app.get_installed_plugins() == {"kicker"}

    def test_find_exe_falls_back_to_path_lookup(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        fake = tmp_path / "lmms"
        fake.write_bytes(b"")
        clean_env.setattr(lmms_app, "_static_exe_candidates", lambda: [])
        clean_env.setattr(lmms_app.shutil, "which", lambda name: None)
        assert lmms_app.find_lmms_exe() is None
        clean_env.setattr(
            lmms_app.shutil, "which",
            lambda name: str(fake) if name == "lmms" else None,
        )
        assert lmms_app.find_lmms_exe() == fake

    def test_find_exe_env_var_takes_precedence(self, tmp_path, clean_env):
        from lmms_mcp import lmms_app
        env_exe = tmp_path / "custom-lmms"
        env_exe.write_bytes(b"")
        path_exe = tmp_path / "lmms"
        path_exe.write_bytes(b"")
        clean_env.setattr(
            lmms_app.shutil, "which", lambda name: str(path_exe)
        )
        clean_env.setenv("LMMS_EXECUTABLE", str(env_exe))
        assert lmms_app.find_lmms_exe() == env_exe


class TestPresetsDirDetection:
    """Machine-independent tests for ZynAddSubFX preset dir detection."""

    XIZ_XML = b"<ZynAddSubFX-data/>"

    @pytest.fixture
    def isolated(self, monkeypatch):
        """No env vars, no machine-specific exe or static candidates."""
        from lmms_mcp import lmms_app, presets
        monkeypatch.delenv("LMMS_PRESETS_DIR", raising=False)
        monkeypatch.delenv("LMMS_EXECUTABLE", raising=False)
        monkeypatch.setattr(presets, "_static_preset_candidates", lambda: [])
        monkeypatch.setattr(lmms_app, "_static_exe_candidates", lambda: [])
        monkeypatch.setattr(lmms_app.shutil, "which", lambda name: None)
        return monkeypatch

    @classmethod
    def _make_zyn_dir(cls, zyn_dir):
        import gzip
        (zyn_dir / "Bass").mkdir(parents=True)
        with gzip.open(zyn_dir / "Bass" / "x.xiz", "wb") as f:
            f.write(cls.XIZ_XML)
        return zyn_dir

    def test_none_when_nothing_found(self, isolated):
        from lmms_mcp import presets
        assert presets.get_presets_dir() is None
        assert presets.list_presets() == []

    def test_env_var_pointing_at_zyn_dir(self, tmp_path, isolated):
        from lmms_mcp import presets
        zyn = self._make_zyn_dir(tmp_path / "custom" / "ZynAddSubFX")
        isolated.setenv("LMMS_PRESETS_DIR", str(zyn))
        assert presets.get_presets_dir() == zyn

    def test_env_var_pointing_at_parent_presets_dir(self, tmp_path, isolated):
        from lmms_mcp import presets
        parent = tmp_path / "presets"
        zyn = self._make_zyn_dir(parent / "ZynAddSubFX")
        (parent / "TripleOscillator").mkdir()
        isolated.setenv("LMMS_PRESETS_DIR", str(parent))
        assert presets.get_presets_dir() == zyn
        assert presets.list_categories() == ["Bass"]

    def test_exe_relative_linux_prefix(self, tmp_path, isolated):
        from lmms_mcp import presets
        prefix = tmp_path / "prefix"
        (prefix / "bin").mkdir(parents=True)
        exe = prefix / "bin" / "lmms"
        exe.write_bytes(b"")
        zyn = self._make_zyn_dir(
            prefix / "share" / "lmms" / "presets" / "ZynAddSubFX"
        )
        isolated.setenv("LMMS_EXECUTABLE", str(exe))
        assert presets.get_presets_dir() == zyn
        xml = presets.load_preset_xml("Bass/x")
        assert xml == self.XIZ_XML.decode()
        assert presets.load_preset_xml("x") == self.XIZ_XML.decode()
        assert [p["name"] for p in presets.list_presets("Bass")] == ["x"]

    def test_exe_relative_windows_data_dir(self, tmp_path, isolated):
        from lmms_mcp import presets
        install = tmp_path / "LMMS"
        install.mkdir()
        exe = install / "lmms.exe"
        exe.write_bytes(b"")
        zyn = self._make_zyn_dir(install / "data" / "presets" / "ZynAddSubFX")
        isolated.setenv("LMMS_EXECUTABLE", str(exe))
        assert presets.get_presets_dir() == zyn

    def test_env_var_wins_over_exe_relative(self, tmp_path, isolated):
        from lmms_mcp import presets
        prefix = tmp_path / "prefix"
        (prefix / "bin").mkdir(parents=True)
        exe = prefix / "bin" / "lmms"
        exe.write_bytes(b"")
        self._make_zyn_dir(
            prefix / "share" / "lmms" / "presets" / "ZynAddSubFX"
        )
        override = self._make_zyn_dir(tmp_path / "mine" / "ZynAddSubFX")
        isolated.setenv("LMMS_EXECUTABLE", str(exe))
        isolated.setenv("LMMS_PRESETS_DIR", str(override))
        assert presets.get_presets_dir() == override

    def test_static_candidates_used_as_fallback(self, tmp_path, isolated):
        from lmms_mcp import presets
        zyn = self._make_zyn_dir(tmp_path / "static" / "ZynAddSubFX")
        isolated.setattr(
            presets, "_static_preset_candidates",
            lambda: [tmp_path / "missing", zyn],
        )
        assert presets.get_presets_dir() == zyn

    def test_stale_env_var_falls_through(self, tmp_path, isolated):
        from lmms_mcp import presets
        zyn = self._make_zyn_dir(tmp_path / "static" / "ZynAddSubFX")
        isolated.setattr(presets, "_static_preset_candidates", lambda: [zyn])
        isolated.setenv("LMMS_PRESETS_DIR", str(tmp_path / "does-not-exist"))
        assert presets.get_presets_dir() == zyn


class TestLoadZynPresetReport:
    """load_zyn_preset must say which file it loaded and never mislead.

    Uses a fake presets dir whose files have internal names that differ
    from their file names (like the factory library does).
    """

    @staticmethod
    def _xiz(internal_name):
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE ZynAddSubFX-data>\n'
            '<ZynAddSubFX-data version-major="1" version-minor="0">'
            '<INFORMATION/><BASE_PARAMETERS/>'
            '<INSTRUMENT><INFO>'
            f'<string name="name">{internal_name}</string>'
            '</INFO></INSTRUMENT></ZynAddSubFX-data>'
        ).encode()

    @pytest.fixture
    def zyn(self, tmp_path, monkeypatch):
        """Fake presets dir + a project with one zynaddsubfx track."""
        import gzip
        from lmms_mcp import lmms_app, presets
        from lmms_mcp import server as srv

        zyn_dir = tmp_path / "ZynAddSubFX"
        files = {
            "Strings/0034-Dark Strings.xiz": "Internal X",
            "Strings/0035-Bright Strings.xiz": "Bright Strings",
            "Pads/0001-Warm Pad.xiz": "Something Else",
            "Bass/0001-Bass 1.xiz": "Plain Bass",
            "Bass/0035-FM Bass 1.xiz": "FM Thing",
            "Synth/0007-Bass 1.xiz": "Synth Bass One",
            "0002-Root Preset.xiz": "Root Preset",
        }
        for rel, internal in files.items():
            path = zyn_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wb") as f:
                f.write(self._xiz(internal))
        monkeypatch.setenv("LMMS_PRESETS_DIR", str(zyn_dir))
        monkeypatch.setattr(presets, "_static_preset_candidates", lambda: [])
        monkeypatch.setattr(lmms_app, "find_lmms_exe", lambda: None)

        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("Z", instrument="zynaddsubfx")
        return srv, zyn_dir

    def test_reports_file_and_names(self, zyn):
        srv, zyn_dir = zyn
        result = json.loads(
            srv.load_zyn_preset(0, "Strings/0034-Dark Strings"))
        file = zyn_dir / "Strings" / "0034-Dark Strings.xiz"
        assert result["track_index"] == 0
        assert result["requested"] == "Strings/0034-Dark Strings"
        assert result["file"] == str(file)
        assert result["preset"] == "Strings/Dark Strings"
        assert result["internal_name"] == "Internal X"
        assert result["message"] == (
            f"Loaded ZynAddSubFX preset 'Strings/Dark Strings' "
            f"(file {file}) into track 0"
        )
        assert "other_matches" not in result

    def test_note_when_internal_name_differs(self, zyn):
        srv, _ = zyn
        result = json.loads(srv.load_zyn_preset(0, "Strings/Dark Strings"))
        assert result["preset"] == "Strings/Dark Strings"
        assert "Internal X" in result["note"]
        assert "normal" in result["note"]

    def test_no_note_when_names_agree(self, zyn):
        srv, _ = zyn
        result = json.loads(srv.load_zyn_preset(0, "Strings/Bright Strings"))
        assert result["internal_name"] == "Bright Strings"
        assert "note" not in result

    @pytest.mark.parametrize("request_name", [
        "Strings/Dark Strings", "Dark Strings", "0034-Dark Strings",
        "dark strings", "Strings/0034-Dark Strings.xiz",
    ])
    def test_accepted_forms_load_same_file(self, zyn, request_name):
        srv, zyn_dir = zyn
        result = json.loads(srv.load_zyn_preset(0, request_name))
        assert result["file"] == str(
            zyn_dir / "Strings" / "0034-Dark Strings.xiz")
        assert result["preset"] == "Strings/Dark Strings"
        assert result["requested"] == request_name

    def test_absolute_path(self, zyn):
        srv, zyn_dir = zyn
        file = zyn_dir / "Pads" / "0001-Warm Pad.xiz"
        result = json.loads(srv.load_zyn_preset(0, str(file)))
        assert result["file"] == str(file)
        assert result["preset"] == "Pads/Warm Pad"
        assert result["internal_name"] == "Something Else"

    def test_root_preset_has_plain_name(self, zyn):
        srv, zyn_dir = zyn
        result = json.loads(srv.load_zyn_preset(0, "Root Preset"))
        assert result["file"] == str(zyn_dir / "0002-Root Preset.xiz")
        assert result["preset"] == "Root Preset"

    def test_ambiguous_plain_name_lists_other_matches(self, zyn):
        srv, zyn_dir = zyn
        result = json.loads(srv.load_zyn_preset(0, "Bass 1"))
        assert result["preset"] == "Bass/Bass 1"
        assert result["file"] == str(zyn_dir / "Bass" / "0001-Bass 1.xiz")
        # Same name in another category; the "FM Bass 1" suffix-only match
        # is not an equally good match and is not listed.
        assert result["other_matches"] == ["Synth/Bass 1"]
        assert "Category/Name" in result["note"]
        assert "other_matches" in result["note"]

    def test_category_name_picks_the_other_match(self, zyn):
        srv, zyn_dir = zyn
        result = json.loads(srv.load_zyn_preset(0, "Synth/Bass 1"))
        assert result["preset"] == "Synth/Bass 1"
        assert result["file"] == str(zyn_dir / "Synth" / "0007-Bass 1.xiz")
        assert result["internal_name"] == "Synth Bass One"
        assert "other_matches" not in result

    def test_end_of_name_match_is_a_fallback(self, zyn):
        srv, _ = zyn
        # No preset is called exactly "Strings", but two end with it.
        result = json.loads(srv.load_zyn_preset(0, "Strings"))
        assert result["preset"].endswith("Strings")
        assert len(result["other_matches"]) == 1
        # A fragment that is not at the end of any name is not a match.
        assert "error" in json.loads(srv.load_zyn_preset(0, "FM Bass"))

    def test_other_matches_capped_at_five(self, zyn):
        import gzip
        srv, zyn_dir = zyn
        for i in range(8):
            d = zyn_dir / f"Cat{i}"
            d.mkdir()
            with gzip.open(d / f"{i:04d}-Dup.xiz", "wb") as f:
                f.write(self._xiz("Dup"))
        result = json.loads(srv.load_zyn_preset(0, "Dup"))
        assert len(result["other_matches"]) == 5
        assert "and 2 more" in result["note"]

    def test_not_found_suggests_close_names(self, zyn):
        srv, _ = zyn
        result = json.loads(srv.load_zyn_preset(0, "Dark Strngs"))
        assert "error" in result
        assert result["suggestions"][0] == "Strings/Dark Strings"
        assert len(result["suggestions"]) <= 5
        assert "Strings/Dark Strings" in result["error"]
        assert "list_zyn_presets" in result["error"]

    def test_not_found_without_close_names_still_points_to_list(self, zyn):
        srv, _ = zyn
        result = json.loads(srv.load_zyn_preset(0, "qqqqzzzz"))
        assert result["suggestions"] == []
        assert "list_zyn_presets" in result["error"]

    def test_unknown_category_falls_back_to_name_search(self, zyn):
        srv, zyn_dir = zyn
        result = json.loads(srv.load_zyn_preset(0, "Nope/Dark Strings"))
        assert result["preset"] == "Strings/Dark Strings"
        assert result["requested"] == "Nope/Dark Strings"

    def test_wrong_category_is_an_error_not_a_silent_swap(self, zyn):
        srv, _ = zyn
        result = json.loads(srv.load_zyn_preset(0, "Pads/Dark Strings"))
        assert "error" in result
        assert "Strings/Dark Strings" in result["suggestions"]

    def test_failed_load_leaves_track_unchanged(self, zyn):
        srv, _ = zyn
        before = ET.tostring(srv.get_project().root)
        json.loads(srv.load_zyn_preset(0, "qqqqzzzz"))
        assert ET.tostring(srv.get_project().root) == before

    def test_xml_keeps_internal_name_for_lmms(self, zyn):
        srv, _ = zyn
        srv.load_zyn_preset(0, "Strings/Dark Strings")
        inst = find_tracks(srv.get_project().root)[0].find(
            "instrumenttrack/instrument")
        info = inst.find(
            "zynaddsubfx/ZynAddSubFX-data/MASTER/PART[@id='0']/INSTRUMENT/"
            "INFO/string[@name='name']")
        assert info.text == "Internal X"

    def test_load_preset_xml_still_returns_text(self, zyn):
        from lmms_mcp import presets
        xml = presets.load_preset_xml("Strings/Dark Strings")
        assert "Internal X" in xml

    def test_embed_result_has_internal_name(self):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        xml = self._xiz("Named").decode()
        assert embed_zyn_preset(root, 0, xml, "File Name")[
            "internal_name"] == "Named"
        nameless = xml.replace('<string name="name">Named</string>', "")
        result = embed_zyn_preset(root, 0, nameless, "File Name")
        assert result["internal_name"] is None
        assert result["preset"] == "File Name"


class TestCustomPluginsAndVst:
    """Tests for dynamic plugin usage and VST tracks."""

    def test_custom_plugin_accepted(self):
        """A DLL that exists in the plugins dir is accepted by name."""
        from lmms_mcp import server as srv
        from lmms_mcp.project import LMMSProject
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        # papu ships as legacy DLL; if absent skip
        from lmms_mcp import lmms_app
        if "papu" not in lmms_app.get_installed_plugins():
            pytest.skip("papu.dll not installed")
        response = json.loads(
            srv.add_instrument_track("Legacy", instrument="papu")
        )
        assert response.get("custom_plugin") is True

    def test_unknown_plugin_rejected(self):
        from lmms_mcp import server as srv
        from lmms_mcp.project import LMMSProject
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        response = json.loads(
            srv.add_instrument_track("X", instrument="not_a_plugin_xyz")
        )
        assert "error" in response

    def test_add_vst_track_sets_plugin_path(self):
        from lmms_mcp import server as srv
        from lmms_mcp.project import LMMSProject
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        # Use any existing file as stand-in DLL (only path handling tested)
        dll = Path(r"C:\Program Files\LMMS\plugins\tripleoscillator.dll")
        if not dll.is_file():
            pytest.skip("LMMS plugins folder not found")
        response = json.loads(srv.add_vst_track("VST Test", str(dll)))
        assert response.get("vst") == str(dll)
        track = find_tracks(proj.root)[0]
        vestige_el = track.find("instrumenttrack/instrument/vestige")
        assert vestige_el is not None
        assert vestige_el.get("plugin", "").lower().endswith(".dll")

    def test_add_vst_track_missing_file(self):
        from lmms_mcp import server as srv
        from lmms_mcp.project import LMMSProject
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        response = json.loads(
            srv.add_vst_track("VST", "Z:/nonexistent/plugin.dll")
        )
        assert "error" in response


class TestSilenceRegressions:
    """Regressions for bugs that rendered whole songs silent."""

    def test_mixer_channel_gets_send_to_master(self):
        """LMMS deletes the implicit send-to-master when allocating
        channels on load (Mixer::allocateChannelsTo) - without an
        explicit <send> element every sub-channel renders silent."""
        from lmms_mcp.xml_parser import add_mixer_channel
        root = create_empty_project()
        ch = add_mixer_channel(root, "Drums")
        send = ch.find("send")
        assert send is not None
        assert send.get("channel") == "0"
        assert float(send.get("amount")) > 0

    def test_note_beyond_pattern_len_extends_clip(self):
        """LMMS does not play notes outside a pattern clip's window;
        add_note must extend the clip instead of leaving notes silent."""
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        res = add_note_to_track(root, 0, key=57, pos=768, length=96)
        pattern = find_tracks(root)[0].find("pattern")
        assert pattern.get("len") == "864"
        assert res["extended_len"] is True

    def test_note_within_len_not_extended(self):
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        res = add_note_to_track(root, 0, key=57, pos=48, length=48)
        pattern = find_tracks(root)[0].find("pattern")
        assert pattern.get("len") == "192"
        assert res["extended_len"] is False

    def test_bars_to_ticks_returns_int_for_float_input(self):
        """place_pattern(pos_bars=16.0) used to write len="3072.0",
        which LMMS cannot parse (Qt toInt fails -> clip collapses)."""
        assert bars_to_ticks(16.0) == 3072
        assert isinstance(bars_to_ticks(2.5), int)
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        from lmms_mcp.xml_parser import place_instrument_pattern
        pat = place_instrument_pattern(root, 0, position=bars_to_ticks(8.0),
                                       length=bars_to_ticks(16.0))
        assert pat.get("pos") == "1536"
        assert pat.get("len") == "3072"

    def test_move_and_delete_clip_tolerate_float_strings(self):
        from lmms_mcp.xml_parser import (
            place_instrument_pattern, move_clip, delete_clip,
        )
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        pat = place_instrument_pattern(root, 0, 192)
        pat.set("pos", "192.0")  # legacy broken file content
        result = move_clip(root, 0, 192, 960)
        assert result["new_pos"] == 960
        result = delete_clip(root, 0, 960)
        assert "Deleted" in result["message"]

    def test_pattern_track_has_inner_instrument(self):
        """add_note on a BB track failed with 'no inner instruments' -
        the documented workflow must work out of the box."""
        root = create_empty_project()
        add_pattern_track(root, "Drums")
        bb = find_tracks(root)[0].find("bbtrack")
        inner = bb.findall("trackcontainer/track")
        assert len(inner) == 1
        assert inner[0].get("type") == "0"
        res = add_note_to_track(root, 0, key=36, pos=0, length=48)
        assert res["note"] is not None

    def test_add_note_target_specific_pattern(self):
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        from lmms_mcp.xml_parser import place_instrument_pattern
        place_instrument_pattern(root, 0, position=384, name="Bar3")
        place_instrument_pattern(root, 0, position=768, name="Bar5")
        add_note_to_track(root, 0, key=60, pos=48, length=48,
                          pattern_index=1)
        pats = find_tracks(root)[0].findall("pattern")
        assert len(pats) == 2
        assert len(pats[0].findall("note")) == 0
        assert pats[1].get("name") == "Bar5"
        assert len(pats[1].findall("note")) == 1

    def test_save_project_infers_format_from_extension(self):
        """Compressed qCompress bytes inside .mmp are unreadable for
        LMMS/tooling - saving must follow the file extension."""
        import tempfile
        from lmms_mcp.project import LMMSProject
        proj = LMMSProject()
        proj.new()
        proj.add_track("instrument", "T")

        with tempfile.TemporaryDirectory() as tmp:
            mmp = str(Path(tmp) / "song.mmp")
            mmpz = str(Path(tmp) / "song.mmpz")
            proj.save(mmp)          # default: follow extension
            proj.save(mmpz)
            assert Path(mmp).read_bytes().startswith(b"<?xml")
            assert not Path(mmpz).read_bytes().startswith(b"<?xml")


class TestAudioFileProcessorSample:
    """set_audiofile_sample: give an audiofileprocessor track its sample."""

    @staticmethod
    def _afp(root, index=0):
        track = find_tracks(root)[index]
        return track.find("instrumenttrack/instrument/audiofileprocessor")

    def test_sets_attributes(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "Kick", instrument="audiofileprocessor")
        set_audiofile_sample(
            root, 0, "drums/kick_hiphop01.ogg", amp=150, reversed=True,
            looped=2, stutter=True, start=0.25, end=0.75,
        )
        afp = self._afp(root)
        assert afp.get("src") == "drums/kick_hiphop01.ogg"
        assert afp.get("amp") == "150"
        assert afp.get("reversed") == "1"
        assert afp.get("looped") == "2"
        assert afp.get("stutter") == "1"
        assert afp.get("interp") == "1"
        assert afp.get("sframe") == "0.25"
        assert afp.get("eframe") == "0.75"
        assert afp.get("lframe") == "0"

    def test_defaults(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "Kick", instrument="audiofileprocessor")
        set_audiofile_sample(root, 0, "drums/kick01.ogg")
        afp = self._afp(root)
        assert afp.get("amp") == "100"
        assert afp.get("reversed") == "0"
        assert afp.get("looped") == "0"
        assert afp.get("stutter") == "0"
        assert afp.get("sframe") == "0"
        assert afp.get("eframe") == "1"

    def test_creates_missing_element(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "Kick", instrument="audiofileprocessor")
        inst = find_tracks(root)[0].find("instrumenttrack/instrument")
        inst.remove(inst.find("audiofileprocessor"))
        set_audiofile_sample(root, 0, "drums/kick01.ogg")
        assert self._afp(root).get("src") == "drums/kick01.ogg"

    def test_overwrites_on_second_call(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "Kick", instrument="audiofileprocessor")
        set_audiofile_sample(root, 0, "drums/kick01.ogg", amp=200,
                             reversed=True, looped=1)
        set_audiofile_sample(root, 0, "drums/snare01.ogg")
        inst = find_tracks(root)[0].find("instrumenttrack/instrument")
        assert len(inst.findall("audiofileprocessor")) == 1
        afp = self._afp(root)
        assert afp.get("src") == "drums/snare01.ogg"
        assert afp.get("amp") == "100"
        assert afp.get("reversed") == "0"
        assert afp.get("looped") == "0"

    def test_error_on_other_instrument(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "Lead", instrument="tripleoscillator")
        with pytest.raises(ValueError, match="audiofileprocessor"):
            set_audiofile_sample(root, 0, "drums/kick01.ogg")

    def test_error_on_sample_track(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_sample_track(root, "Samples")
        with pytest.raises(ValueError, match="add_instrument_track"):
            set_audiofile_sample(root, 0, "drums/kick01.ogg")

    def test_error_on_bad_track_index(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "Kick", instrument="audiofileprocessor")
        with pytest.raises(IndexError):
            set_audiofile_sample(root, 5, "drums/kick01.ogg")
        with pytest.raises(IndexError):
            set_audiofile_sample(root, -1, "drums/kick01.ogg")

    @pytest.mark.parametrize("kwargs", [
        {"amp": -1},
        {"amp": 501},
        {"looped": 3},
        {"looped": -1},
        {"start": 0.5, "end": 0.5},
        {"start": 0.8, "end": 0.2},
        {"start": -0.1},
        {"end": 1.5},
    ])
    def test_error_on_invalid_values(self, kwargs):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "Kick", instrument="audiofileprocessor")
        with pytest.raises(ValueError):
            set_audiofile_sample(root, 0, "drums/kick01.ogg", **kwargs)
        # a rejected call must not leave a half-written sample behind
        assert self._afp(root).get("src") is None

    def test_save_load_roundtrip(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "Kick", instrument="audiofileprocessor")
        set_audiofile_sample(root, 0, "drums/kick_hiphop01.ogg", amp=120,
                             reversed=True, looped=1, start=0.1, end=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "afp.mmpz"
            save_project(path, root)
            loaded = load_project(path)
        afp = self._afp(loaded)
        assert afp.get("src") == "drums/kick_hiphop01.ogg"
        assert afp.get("amp") == "120"
        assert afp.get("reversed") == "1"
        assert afp.get("looped") == "1"
        assert afp.get("sframe") == "0.1"
        assert afp.get("eframe") == "0.9"

    # ── tool level ───────────────────────────────────────────────

    @staticmethod
    def _new_project():
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        return srv, proj

    def test_tool_sets_sample(self, monkeypatch, tmp_path):
        from lmms_mcp import lmms_app
        (tmp_path / "drums").mkdir()
        (tmp_path / "drums" / "kick01.ogg").write_bytes(b"x")
        monkeypatch.setattr(lmms_app, "get_samples_dir", lambda: tmp_path)
        srv, proj = self._new_project()
        srv.add_instrument_track("Kick", instrument="audiofileprocessor")
        response = json.loads(srv.set_audiofile_sample(
            0, "drums/kick01.ogg", amp=130, looped=1))
        assert "error" not in response
        assert response["track_index"] == 0
        assert response["file"] == "drums/kick01.ogg"
        assert "message" in response
        assert "warning" not in response
        afp = self._afp(proj.root)
        assert afp.get("src") == "drums/kick01.ogg"
        assert afp.get("amp") == "130"
        assert afp.get("looped") == "1"
        assert proj.modified is True

    def test_tool_warns_for_missing_relative_sample(self, monkeypatch,
                                                    tmp_path):
        from lmms_mcp import lmms_app
        monkeypatch.setattr(lmms_app, "get_samples_dir", lambda: tmp_path)
        srv, proj = self._new_project()
        srv.add_instrument_track("Kick", instrument="audiofileprocessor")
        response = json.loads(
            srv.set_audiofile_sample(0, "drums/nope.ogg"))
        assert "error" not in response
        assert "warning" in response
        assert "drums/nope.ogg" in response["warning"]
        # the sample is still written
        assert self._afp(proj.root).get("src") == "drums/nope.ogg"

    def test_tool_skips_check_without_samples_dir(self, monkeypatch):
        from lmms_mcp import lmms_app
        monkeypatch.setattr(lmms_app, "get_samples_dir", lambda: None)
        srv, _ = self._new_project()
        srv.add_instrument_track("Kick", instrument="audiofileprocessor")
        response = json.loads(
            srv.set_audiofile_sample(0, "drums/whatever.ogg"))
        assert "error" not in response
        assert "warning" not in response

    def test_tool_absolute_path_warning(self, tmp_path):
        srv, _ = self._new_project()
        srv.add_instrument_track("Kick", instrument="audiofileprocessor")
        missing = tmp_path / "missing.wav"
        response = json.loads(srv.set_audiofile_sample(0, str(missing)))
        assert "warning" in response
        existing = tmp_path / "exists.wav"
        existing.write_bytes(b"x")
        response = json.loads(srv.set_audiofile_sample(0, str(existing)))
        assert "error" not in response
        assert "warning" not in response

    def test_tool_errors_are_json(self):
        srv, _ = self._new_project()
        srv.add_instrument_track("Lead", instrument="tripleoscillator")
        srv.add_sample_track("Samples")
        for index in (0, 1, 9):
            response = json.loads(
                srv.set_audiofile_sample(index, "drums/kick01.ogg"))
            assert "error" in response
        srv.add_instrument_track("Kick", instrument="audiofileprocessor")
        response = json.loads(
            srv.set_audiofile_sample(2, "drums/kick01.ogg", amp=900))
        assert "error" in response

    def test_get_samples_dir_returns_existing_dir_or_none(self):
        from lmms_mcp import lmms_app
        result = lmms_app.get_samples_dir()
        assert result is None or result.is_dir()


class TestRenderProjectQuality:
    """render_project passes -x/-i to LMMS (default 1x, like the GUI)."""

    @staticmethod
    def _setup(monkeypatch, tmp_path):
        import subprocess
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv

        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        proj.save(str(tmp_path / "song.mmp"))
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("subprocess.run", fake_run)
        return srv, calls

    def test_defaults_are_1x_sincfastest(self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        out = tmp_path / "out.wav"
        result = json.loads(srv.render_project(output_path=str(out)))
        assert result["success"] is True
        assert len(calls) == 1
        cmd = calls[0]
        assert cmd[cmd.index("-x") + 1] == "1"
        assert cmd[cmd.index("-i") + 1] == "sincfastest"
        assert "-x 1 -i sincfastest" in result["command"]
        assert result["oversampling"] == 1
        assert result["interpolation"] == "sincfastest"
        # Existing arguments are unchanged.
        assert cmd[1] == "render"
        assert cmd[cmd.index("-o") + 1] == str(out)
        assert cmd[cmd.index("-f") + 1] == "wav"
        assert cmd[cmd.index("-s") + 1] == "44100"
        assert cmd[cmd.index("-b") + 1] == "160"

    def test_custom_values_are_passed_through(self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "out.wav"),
            oversampling=4, interpolation="sincbest"))
        assert result["success"] is True
        cmd = calls[0]
        assert cmd[cmd.index("-x") + 1] == "4"
        assert cmd[cmd.index("-i") + 1] == "sincbest"
        assert result["oversampling"] == 4
        assert result["interpolation"] == "sincbest"

    @pytest.mark.parametrize("value", [1, 2, 4, 8])
    def test_all_oversampling_values_allowed(self, monkeypatch, tmp_path, value):
        srv, calls = self._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "out.wav"), oversampling=value))
        assert "error" not in result
        assert calls[0][calls[0].index("-x") + 1] == str(value)

    @pytest.mark.parametrize("value", ["linear", "sincfastest",
                                        "sincmedium", "sincbest"])
    def test_all_interpolations_allowed(self, monkeypatch, tmp_path, value):
        srv, calls = self._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "out.wav"), interpolation=value))
        assert "error" not in result
        assert calls[0][calls[0].index("-i") + 1] == value

    @pytest.mark.parametrize("value", [0, 3, 16, -1, True])
    def test_invalid_oversampling_rejected(self, monkeypatch, tmp_path, value):
        srv, calls = self._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "out.wav"), oversampling=value))
        assert "error" in result
        assert "oversampling" in result["error"]
        assert calls == []

    @pytest.mark.parametrize("value", ["", "cubic", "sinc"])
    def test_invalid_interpolation_rejected(self, monkeypatch, tmp_path, value):
        srv, calls = self._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "out.wav"), interpolation=value))
        assert "error" in result
        assert "interpolation" in result["error"]
        assert calls == []


if __name__ == "__main__":
    import sys
    print("Run with: pytest tests/")
    sys.exit(1)


class TestNoteValidation:
    """add_note / add_note_by_name / add_notes_batch must never silently
    swallow bad input: failures are reported clearly."""

    def _srv(self):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("Bass", instrument="tripleoscillator")
        return srv, proj

    @staticmethod
    def _note_elems(proj):
        return find_tracks(proj.root)[0].findall(".//note")

    def test_add_note_by_name_flat(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_note_by_name(0, "Bb2", pos=0, length=48))
        assert "error" not in res
        notes = self._note_elems(proj)
        assert [n.get("key") for n in notes] == ["46"]

    def test_add_note_by_name_invalid_reports_error(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_note_by_name(0, "H2"))
        assert "error" in res
        assert "Invalid note name 'H2'" in res["error"]
        assert "Bb2" in res["error"]
        assert self._note_elems(proj) == []

    def test_add_note_by_name_out_of_range(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_note_by_name(0, "G#9"))
        assert "error" in res and "out of range" in res["error"]
        assert self._note_elems(proj) == []

    def test_add_note_validation(self):
        srv, proj = self._srv()
        for kwargs, fragment in [
            ({"key": 128}, "key 128 is out of range"),
            ({"key": -1}, "key -1 is out of range"),
            ({"pos": -48}, "pos must be >= 0"),
            ({"length": 0}, "length must be > 0"),
            ({"length": -12}, "length must be > 0"),
        ]:
            res = json.loads(srv.add_note(0, **kwargs))
            assert "error" in res, kwargs
            assert fragment in res["error"], res
        assert self._note_elems(proj) == []

    def test_add_note_bad_track_index(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_note(5, key=60))
        assert "error" in res and "Track index 5 out of range" in res["error"]
        res = json.loads(srv.add_note(-1, key=60))
        assert "error" in res
        res = json.loads(srv.add_note_by_name(5, "C4"))
        assert "error" in res

    def test_add_note_valid_still_works(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_note(0, key=127, pos=0, length=12))
        assert "error" not in res and "message" in res
        assert [n.get("key") for n in self._note_elems(proj)] == ["127"]

    def test_batch_all_valid(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_notes_batch(0, [
            {"key": "Bb2", "pos": 0, "length": 24},
            {"key": "C#3", "pos": 24, "length": 24},
            {"key": 60, "pos": 48},
        ]))
        assert res["added"] == 3
        assert res["failed"] == 0
        assert res["errors"] == []
        assert len(res["notes"]) == 3
        assert "warning" not in res and "error" not in res
        assert [n.get("key") for n in self._note_elems(proj)] == ["46", "49", "60"]

    def test_batch_mixed_valid_and_invalid(self):
        srv, proj = self._srv()
        bad = {"key": "H2", "pos": 0, "length": 48}
        good = {"key": "Bb2", "pos": 48, "length": 48}
        res = json.loads(srv.add_notes_batch(0, [bad, good]))
        assert res["added"] == 1
        assert res["failed"] == 1
        assert len(res["notes"]) == 1
        assert "error" not in res
        assert "1 of 2 notes were NOT added" in res["warning"]
        assert len(res["errors"]) == 1
        err = res["errors"][0]
        assert err["index"] == 0
        assert err["note"] == bad
        assert "Invalid note name 'H2'" in err["error"]
        assert [n.get("key") for n in self._note_elems(proj)] == ["46"]

    def test_batch_numeric_and_range_validation(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_notes_batch(0, [
            {"key": 128, "pos": 0},
            {"key": -1, "pos": 0},
            {"key": 60, "pos": -1},
            {"key": 60, "pos": 0, "length": 0},
            {"key": 60, "pos": "x"},
            {"key": 60, "pos": 0, "length": 24},
        ]))
        assert res["added"] == 1
        assert res["failed"] == 5
        assert [e["index"] for e in res["errors"]] == [0, 1, 2, 3, 4]
        assert "out of range" in res["errors"][0]["error"]
        assert "out of range" in res["errors"][1]["error"]
        assert "pos must be >= 0" in res["errors"][2]["error"]
        assert "length must be > 0" in res["errors"][3]["error"]
        assert "pos must be an integer" in res["errors"][4]["error"]
        assert "5 of 6 notes were NOT added" in res["warning"]
        assert len(self._note_elems(proj)) == 1

    def test_batch_all_invalid_returns_error(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_notes_batch(0, [
            {"key": 128, "pos": 0},
            {"key": "H2", "pos": 0},
        ]))
        assert res["added"] == 0
        assert res["failed"] == 2
        assert "error" in res
        assert "None of the 2 notes were added" in res["error"]
        assert len(res["errors"]) == 2
        assert self._note_elems(proj) == []

    def test_batch_non_dict_note_and_empty_list(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_notes_batch(0, [60, {"key": 60}]))
        assert res["added"] == 1 and res["failed"] == 1
        assert res["errors"][0]["index"] == 0
        assert "each note must be an object" in res["errors"][0]["error"]
        res = json.loads(srv.add_notes_batch(0, []))
        assert "error" in res and res["added"] == 0

    def test_batch_bad_track_index(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_notes_batch(7, [{"key": 60}]))
        assert "error" in res and "Track index 7 out of range" in res["error"]
        assert res["added"] == 0
        assert self._note_elems(proj) == []

    def test_batch_bad_pattern_index_reported_per_note(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_notes_batch(0, [
            {"key": 60, "pattern_index": 9},
            {"key": 62},
        ]))
        assert res["added"] == 1 and res["failed"] == 1
        assert "Pattern index 9 out of range" in res["errors"][0]["error"]

    def test_note_name_to_key_tool_flats(self):
        from lmms_mcp import server as srv
        assert json.loads(srv.note_name_to_key("Bb2"))["midi_key"] == 46
        assert "error" in json.loads(srv.note_name_to_key("H2"))
        scale = json.loads(srv.generate_scale("Bb2", "major"))
        assert scale["notes"][0]["key"] == 46


# A made-up LMMS 1.2.2 install: independent of the machine running the
# tests. Lacks everything that needs LMMS 1.3+ and the VST host.
LMMS_122_PLUGINS = frozenset({
    "tripleoscillator", "kicker", "audiofileprocessor", "organic",
    "malletsstk", "lb302", "monstro", "nes", "papu", "opl2", "patman",
    "sf2player", "sfxr", "sid", "vibedstrings", "watsyn", "zynaddsubfx",
    "gigplayer", "bitinvader",
    "amplifier", "bassbooster", "bitcrush", "crossovereq", "delay",
    "dualfilter", "dynamicsprocessor", "eq", "flanger", "multitapecho",
    "reverbsc", "stereoenhancer", "stereomatrix", "waveshaper",
    "ladspaeffect", "midiexport", "spectrumanalyzer",
    "mycustomsynth",
})
# Known only from LMMS 1.3 on, or hosting external plugins
UNAVAILABLE_IN_122 = (
    "compressor", "slicert", "xpressive", "dispersion", "frequencyshifter",
    "slewdistortion", "vestige",
)


class TestPluginAvailability:
    """The MCP only offers, and only accepts, plugins the installed LMMS has."""

    @pytest.fixture
    def lmms122(self, monkeypatch):
        """(server, project) with a fake LMMS 1.2.2 as the installed LMMS."""
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv
        monkeypatch.setattr(
            srv, "_availability",
            lmms_app.build_availability(
                srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
                LMMS_122_PLUGINS, "1.2.2",
            ),
        )
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        return srv, proj

    @pytest.fixture
    def unverified(self, monkeypatch):
        """(server, project) when no LMMS installation is found."""
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv
        monkeypatch.setattr(
            srv, "_availability",
            lmms_app.build_availability(
                srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
                set(), None,
            ),
        )
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        return srv, proj

    @pytest.fixture
    def server_for(self, monkeypatch):
        """Reload the server module against a fake install (what an agent
        sees at startup: instructions, tool list, descriptions)."""
        import importlib

        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv

        def make(installed, version="1.2.2", stems=None):
            monkeypatch.setattr(
                lmms_app, "get_installed_plugins", lambda: set(installed))
            monkeypatch.setattr(
                lmms_app, "get_installed_stems",
                lambda: dict(stems) if stems else {n: n for n in installed})
            monkeypatch.setattr(
                lmms_app, "get_lmms_version", lambda: version)
            return importlib.reload(srv)

        yield make
        monkeypatch.undo()
        importlib.reload(srv)

    # --- pure detection -------------------------------------------------

    def test_compute_available_respects_aliases_and_static(self):
        from lmms_mcp import lmms_app
        known = ["tripleoscillator", "opulenz", "nes", "slicert", "freeboy"]
        # opl2 satisfies opulenz and papu satisfies freeboy; papu is the
        # Game Boy chip, so it does not make nes (Nescaline) available
        available = lmms_app.compute_available(
            known, {"tripleoscillator", "opl2", "papu"}, windows=False)
        assert available == {"tripleoscillator", "opulenz", "freeboy"}
        available = lmms_app.compute_available(
            known, {"tripleoscillator", "nes", "opulenz"}, windows=False)
        assert available == {"tripleoscillator", "opulenz", "nes"}

    def test_freeboy_is_static_only_on_windows(self):
        from lmms_mcp import lmms_app
        installed = {"tripleoscillator", "nes"}
        assert lmms_app.compute_available(
            ["freeboy", "nes"], installed, windows=True
        ) == {"freeboy", "nes"}
        assert lmms_app.compute_available(
            ["freeboy", "nes"], installed, windows=False
        ) == {"nes"}

    def test_compute_available_without_install_allows_everything(self):
        from lmms_mcp import lmms_app
        known = ["slicert", "compressor", "tripleoscillator"]
        assert lmms_app.compute_available(known, set()) == set(known)

    def test_build_availability_marks_verified(self):
        from lmms_mcp import lmms_app
        av = lmms_app.build_availability(
            ["a", "b"], ["c"], {"a", "c"}, "1.2.2")
        assert av.verified and av.version == "1.2.2"
        assert av.instruments == {"a"} and av.effects == {"c"}
        none = lmms_app.build_availability(["a", "b"], ["c"], set(), None)
        assert not none.verified
        assert none.instruments == {"a", "b"} and none.effects == {"c"}

    def test_detect_availability_reads_installed_lmms(self, monkeypatch):
        from lmms_mcp import lmms_app
        monkeypatch.setattr(
            lmms_app, "get_installed_plugins", lambda: {"delay", "kicker"})
        monkeypatch.setattr(lmms_app, "get_lmms_version", lambda: "1.2.2")
        av = lmms_app.detect_availability(
            ["kicker", "slicert"], ["delay", "compressor"])
        assert av.verified and av.version == "1.2.2"
        assert av.instruments == {"kicker"}
        assert av.effects == {"delay"}

    def test_detect_availability_without_lmms(self, monkeypatch):
        from lmms_mcp import lmms_app
        monkeypatch.setattr(lmms_app, "get_installed_plugins", lambda: set())
        av = lmms_app.detect_availability(["kicker", "slicert"], ["delay"])
        assert not av.verified and av.version is None
        assert av.instruments == {"kicker", "slicert"}

    def test_adapt_recommendations_substitutes_and_drops(self):
        from lmms_mcp import lmms_app
        recs = {
            "vocals": ["eq", "compressor", "reverbsc"],
            "dup": ["compressor", "dynamicsprocessor"],
            "lead": ["xpressive", "tripleoscillator"],
            "gone": ["slicert"],
        }
        available = {"eq", "reverbsc", "dynamicsprocessor", "tripleoscillator"}
        assert lmms_app.adapt_recommendations(recs, available) == {
            "vocals": ["eq", "dynamicsprocessor", "reverbsc"],
            "dup": ["dynamicsprocessor"],
            "lead": ["tripleoscillator"],
        }

    def test_classify_given_installed_set(self):
        from lmms_mcp import lmms_app
        result = lmms_app.classify_installed_plugins(
            {"kicker"}, {"delay"}, installed={"kicker", "delay", "other"})
        assert result == {
            "instruments": ["kicker"], "effects": ["delay"],
            "unknown": ["other"],
        }

    # --- refusals -------------------------------------------------------

    def test_add_effect_refuses_missing_effect(self, lmms122):
        srv, proj = lmms122
        srv.add_instrument_track("Lead")
        before = proj.get_xml_string()
        result = json.loads(srv.add_effect("track", 0, "compressor"))
        assert "'compressor' is not available in the installed LMMS 1.2.2" \
            in result["error"]
        assert "requires LMMS 1.3+" in result["error"]
        assert "Use 'dynamicsprocessor' instead" in result["error"]
        assert result["alternative"] == "dynamicsprocessor"
        assert "dynamicsprocessor" in result["available_effects"]
        for name in UNAVAILABLE_IN_122:
            assert name not in result["available_effects"]
        assert proj.get_xml_string() == before
        assert find_tracks(proj.root)[0].find(
            "instrumenttrack/fxchain/effect") is None

    def test_add_effect_refusal_ignores_case_and_spaces(self, lmms122):
        srv, _ = lmms122
        srv.add_instrument_track("Lead")
        result = json.loads(srv.add_effect("track", 0, " Compressor "))
        assert result["alternative"] == "dynamicsprocessor"

    def test_add_effect_refuses_on_mixer_too(self, lmms122):
        srv, proj = lmms122
        before = proj.get_xml_string()
        result = json.loads(srv.add_effect("mixer", 0, "dispersion"))
        assert result["alternative"] == "flanger"
        assert proj.get_xml_string() == before

    def test_add_effect_available_effect_works(self, lmms122):
        srv, _ = lmms122
        srv.add_instrument_track("Lead")
        result = json.loads(srv.add_effect("track", 0, "dynamicsprocessor"))
        assert result["effect"] == "dynamicsprocessor"
        assert "warning" not in result and "error" not in result

    def test_unknown_effect_error_lists_only_available(self, lmms122):
        srv, _ = lmms122
        srv.add_instrument_track("Lead")
        result = json.loads(srv.add_effect("track", 0, "greatest_reverb_9000"))
        assert "Unknown effect" in result["error"]
        assert result["valid_effects"] == sorted(
            srv._availability.effects)
        text = json.dumps(result)
        for name in UNAVAILABLE_IN_122:
            assert name not in text
        assert result["recommendations"]["vocals"] == [
            "eq", "dynamicsprocessor", "reverbsc", "delay"]

    def test_add_instrument_track_refuses_missing_instrument(self, lmms122):
        srv, proj = lmms122
        before = proj.get_xml_string()
        result = json.loads(
            srv.add_instrument_track("Slices", instrument="slicert"))
        assert "'slicert' is not available in the installed LMMS 1.2.2" \
            in result["error"]
        assert "requires LMMS 1.3+" in result["error"]
        assert result["alternative"] == "audiofileprocessor"
        assert "audiofileprocessor" in result["available_instruments"]
        for name in UNAVAILABLE_IN_122:
            assert name not in result["available_instruments"]
        assert find_tracks(proj.root) == []
        assert proj.get_xml_string() == before

    def test_xpressive_refused_with_watsyn_alternative(self, lmms122):
        srv, proj = lmms122
        result = json.loads(
            srv.add_instrument_track("Lead", instrument="xpressive"))
        assert result["alternative"] == "watsyn"
        assert find_tracks(proj.root) == []

    def test_alias_to_missing_instrument_is_refused(self, monkeypatch, lmms122):
        from lmms_mcp import lmms_app
        srv, proj = lmms122
        installed = LMMS_122_PLUGINS - {"sf2player"}
        monkeypatch.setattr(
            srv, "_availability",
            lmms_app.build_availability(
                srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
                installed, "1.2.2"))
        result = json.loads(srv.add_instrument_track("SF", instrument="sf2"))
        assert "'sf2player' is not available" in result["error"]
        assert "alternative" not in result
        assert find_tracks(proj.root) == []

    def test_available_instruments_work(self, lmms122):
        srv, proj = lmms122
        for inst in ("tripleoscillator", "KICKER", "opulenz", "zyn"):
            result = json.loads(srv.add_instrument_track("T", instrument=inst))
            assert "error" not in result and "warning" not in result, inst
        assert len(find_tracks(proj.root)) == 4

    def test_custom_installed_plugin_still_accepted(self, lmms122):
        srv, proj = lmms122
        result = json.loads(
            srv.add_instrument_track("Mine", instrument="mycustomsynth"))
        assert result.get("custom_plugin") is True
        assert len(find_tracks(proj.root)) == 1

    def test_helper_plugin_is_not_an_instrument(self, lmms122):
        srv, proj = lmms122
        result = json.loads(
            srv.add_instrument_track("X", instrument="midiexport"))
        assert "not an instrument" in result["error"]
        assert find_tracks(proj.root) == []

    def test_unknown_instrument_error_lists_only_available(self, lmms122):
        srv, _ = lmms122
        result = json.loads(
            srv.add_instrument_track("X", instrument="not_a_plugin_xyz"))
        assert "Unknown instrument" in result["error"]
        assert result["valid_instruments"] == srv._instrument_choices(
            srv._availability)
        text = json.dumps(result)
        for name in UNAVAILABLE_IN_122:
            assert name not in text

    def test_add_vst_track_refused_without_vestige(self, lmms122, tmp_path):
        srv, proj = lmms122
        dll = tmp_path / "synth.dll"
        dll.write_bytes(b"MZ")
        before = proj.get_xml_string()
        result = json.loads(srv.add_vst_track("VST", str(dll)))
        assert "'vestige' is not available in the installed LMMS 1.2.2" \
            in result["error"]
        assert proj.get_xml_string() == before

    def test_add_vst_track_works_with_vestige(self, monkeypatch, lmms122,
                                              tmp_path):
        from lmms_mcp import lmms_app
        srv, proj = lmms122
        monkeypatch.setattr(
            srv, "_availability",
            lmms_app.build_availability(
                srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
                LMMS_122_PLUGINS | {"vestige"}, "1.2.2"))
        dll = tmp_path / "synth.dll"
        dll.write_bytes(b"MZ")
        result = json.loads(srv.add_vst_track("VST", str(dll)))
        assert "error" not in result
        assert len(find_tracks(proj.root)) == 1

    # --- no LMMS found: everything known is allowed ----------------------

    def test_without_lmms_known_plugins_are_allowed(self, unverified):
        srv, _ = unverified
        added = json.loads(
            srv.add_instrument_track("Lead", instrument="tripleoscillator"))
        assert "error" not in added and "warning" not in added
        fx = json.loads(srv.add_effect("track", 0, "delay"))
        assert "error" not in fx and "warning" not in fx

    def test_without_lmms_version_dependent_plugins_warn(self, unverified):
        srv, proj = unverified
        track = json.loads(
            srv.add_instrument_track("Slices", instrument="slicert"))
        assert "error" not in track
        assert "could not be verified" in track["warning"]
        assert "audiofileprocessor" in track["warning"]
        fx = json.loads(srv.add_effect("track", 0, "compressor"))
        assert "error" not in fx
        assert "could not be verified" in fx["warning"]
        assert "dynamicsprocessor" in fx["warning"]
        assert len(find_tracks(proj.root)) == 1
        assert find_tracks(proj.root)[0].find(
            "instrumenttrack/fxchain/effect[@name='compressor']") is not None

    def test_without_lmms_instructions_say_so(self, unverified):
        srv, _ = unverified
        text = srv.build_instructions(srv._availability, "/tmp/p")
        assert "No LMMS installation was detected" in text
        assert "compressor" in text  # standard chains, unverifiable
        assert srv.mcp.instructions is not None

    # --- what the agent is shown -----------------------------------------

    def test_instructions_omit_unavailable_plugins(self, lmms122):
        srv, _ = lmms122
        text = srv.build_instructions(srv._availability, "/tmp/projects")
        for name in UNAVAILABLE_IN_122:
            assert name not in text, name
        assert "Installed LMMS: 1.2.2" in text
        assert "eq -> dynamicsprocessor -> reverbsc" in text
        assert "Master bus: eq -> dynamicsprocessor -> stereoenhancer" in text
        assert "scan_vst_directory" not in text
        assert "/tmp/projects" in text

    def test_instructions_follow_missing_zyn_and_sampler(self, monkeypatch,
                                                         lmms122):
        from lmms_mcp import lmms_app
        srv, _ = lmms122
        av = lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
            LMMS_122_PLUGINS - {"zynaddsubfx", "audiofileprocessor"},
            "1.2.2")
        text = srv.build_instructions(av, "/tmp/p")
        assert "zynaddsubfx" not in text
        assert "load_zyn_preset" not in text
        assert "audiofileprocessor" not in text
        assert "set_audiofile_sample" not in text

    def test_instructions_offer_everything_on_a_full_install(self, monkeypatch,
                                                             lmms122):
        from lmms_mcp import lmms_app
        srv, _ = lmms122
        av = lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
            LMMS_122_PLUGINS | set(UNAVAILABLE_IN_122), "1.3.0")
        text = srv.build_instructions(av, "/tmp/p")
        assert "Vocals: eq -> compressor -> reverbsc" in text
        assert "scan_vst_directory" in text

    def test_tool_docstrings_list_only_available(self, lmms122):
        srv, _ = lmms122
        for fn in (srv.add_instrument_track, srv.add_effect):
            doc = srv.render_plugin_doc(
                fn.__plugin_doc_template__, srv._availability)
            assert "$" not in doc
            for name in UNAVAILABLE_IN_122:
                assert name not in doc, (fn.__name__, name)
        inst_doc = srv.render_plugin_doc(
            srv.add_instrument_track.__plugin_doc_template__,
            srv._availability)
        assert "tripleoscillator" in inst_doc and "zynaddsubfx" in inst_doc
        fx_doc = srv.render_plugin_doc(
            srv.add_effect.__plugin_doc_template__, srv._availability)
        assert "dynamicsprocessor" in fx_doc

    def test_resources_list_only_available(self, lmms122):
        srv, _ = lmms122
        instruments = srv.resource_instruments()
        effects = srv.resource_effects()
        for name in UNAVAILABLE_IN_122:
            assert f'"{name}"' not in instruments, name
            assert f'"{name}"' not in effects, name
        assert "1.2.2" in effects
        assert "dynamicsprocessor" in effects
        eff = json.loads(effects)
        assert eff["recommendations_by_use_case"]["drums"] == [
            "dynamicsprocessor", "eq", "waveshaper"]
        inst = json.loads(instruments)
        assert inst["recommendations_by_use_case"]["lead"] == [
            "tripleoscillator", "watsyn", "opulenz"]

    def test_effects_resource_works_on_a_full_install(self):
        """Regression: the resource raised NameError (KNOWN_EFFECTS)."""
        from lmms_mcp import server as srv
        data = json.loads(srv.resource_effects())
        names = {e["name"] for e in data["effects"]}
        assert {"compressor", "delay"} <= names

    def test_prompts_only_mention_available_instruments(self, lmms122):
        srv, _ = lmms122
        texts = [
            srv.create_basic_song(),
            srv.add_drum_pattern(),
            srv.create_melody(),
        ]
        for text in texts:
            for name in UNAVAILABLE_IN_122:
                assert name not in text, name

    def test_prompts_without_sampler(self, monkeypatch, lmms122):
        from lmms_mcp import lmms_app
        srv, _ = lmms122
        monkeypatch.setattr(
            srv, "_availability",
            lmms_app.build_availability(
                srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
                LMMS_122_PLUGINS - {"audiofileprocessor"}, "1.2.2"))
        for text in (srv.create_basic_song(), srv.add_drum_pattern()):
            assert "audiofileprocessor" not in text
            assert "set_audiofile_sample" not in text

    def test_list_available_plugins_reports_missing_with_reasons(self, lmms122):
        srv, _ = lmms122
        result = json.loads(srv.list_available_plugins())
        assert "compressor" not in result["built_in_effects"]
        assert "slicert" not in result["built_in_instruments"]
        missing = result["unavailable_in_installed_lmms"]
        assert missing["compressor"] == "requires LMMS 1.3+; use dynamicsprocessor"
        assert missing["slicert"] == "requires LMMS 1.3+; use audiofileprocessor"
        assert "vestige" in missing
        assert result["version"] == "1.2.2"
        # helper plugins are not offered, real custom ones are
        assert result["custom_plugins_unknown_type"] == ["mycustomsynth"]

    def test_get_lmms_info_uses_the_snapshot(self, monkeypatch, lmms122):
        from lmms_mcp import lmms_app
        srv, _ = lmms122
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/usr/bin/lmms"))
        info = json.loads(srv.get_lmms_info())
        assert info["version"] == "1.2.2"
        assert "compressor" not in info["effects_verified_available"]
        assert "delay" in info["effects_verified_available"]
        assert "slicert" not in info["instruments_verified_available"]

    # --- end to end through the MCP registry ------------------------------

    def test_registered_server_hides_unavailable_plugins(self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS)
        tools = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}
        # Tools needing the missing VST host are not offered at all
        assert "add_vst_track" not in tools
        assert "scan_vst_directory" not in tools
        assert "add_effect" in tools and "load_zyn_preset" in tools
        everything = srv.mcp.instructions + "\n".join(
            t.description or "" for t in tools.values())
        for name in UNAVAILABLE_IN_122:
            assert name not in everything, name
        assert "dynamicsprocessor" in tools["add_effect"].description
        assert "tripleoscillator" in tools["add_instrument_track"].description

    def test_registered_server_hides_tools_of_missing_plugins(self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS - {"zynaddsubfx"})
        names = {t.name for t in asyncio.run(srv.mcp.list_tools())}
        assert not {"list_zyn_presets", "load_zyn_preset",
                    "set_zyn_params"} & names
        assert "set_audiofile_sample" in names

    def test_registered_server_offers_everything_on_a_full_install(
            self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS | set(UNAVAILABLE_IN_122), "1.3.0")
        tools = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}
        assert {"add_vst_track", "scan_vst_directory",
                "load_zyn_preset"} <= set(tools)
        assert "slicert" in tools["add_instrument_track"].description
        assert "compressor" in tools["add_effect"].description

    def test_registered_server_without_lmms_offers_everything(self, server_for):
        import asyncio
        srv = server_for(set(), None)
        tools = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}
        assert "add_vst_track" in tools
        assert "compressor" in tools["add_effect"].description
        assert "No LMMS installation was detected" in srv.mcp.instructions

    def test_registered_server_resolves_names_from_the_install(
            self, server_for):
        stems = _stems_of(LINUX_122_LIBS)
        srv = server_for(set(stems), stems=stems)
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        result = json.loads(srv.add_instrument_track("FM", instrument="opulenz"))
        assert result["lmms_plugin"] == "OPL2"
        assert srv._availability.plugin_names["freeboy"] == "papu"


# The plugin library files of a made-up Linux LMMS 1.2.2: the case of the
# file names matters, LMMS loads libOPL2.so only as "OPL2".
LINUX_122_LIBS = (
    "libtripleoscillator.so", "libkicker.so", "libaudiofileprocessor.so",
    "libnes.so", "libpapu.so", "libOPL2.so", "libsid.so", "libsf2player.so",
    "libzynaddsubfx.so", "libMyCustomSynth.so", "libeq.so", "libdelay.so",
)


def _stems_of(libs):
    """{lowercase name: original-case stem} for Linux library file names."""
    return {f[3:-3].lower(): f[3:-3] for f in libs}


class TestInstrumentPluginNames:
    """Instruments are written into projects under the name LMMS loads.

    LMMS matches the <instrument name="..."> of a project case-sensitively
    against its plugin names: opulenz only plays as "OPL2" in LMMS 1.2 and
    the FreeBoy instrument is the library "papu". Writing the offered name
    produces a silent track.
    """

    @staticmethod
    def _availability(libs=LINUX_122_LIBS, version="1.2.2", windows=False):
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv
        stems = _stems_of(libs)
        return lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
            set(stems), version, stems=stems, windows=windows)

    @pytest.fixture
    def linux122(self, monkeypatch):
        """(server, project) with the fake Linux LMMS 1.2.2 installed."""
        from lmms_mcp import server as srv
        monkeypatch.setattr(srv, "_availability", self._availability())
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        return srv, proj

    @staticmethod
    def _instrument_xml(proj, index=0):
        inst = find_tracks(proj.root)[index].find("instrumenttrack/instrument")
        return inst.get("name"), [child.tag for child in inst]

    # --- resolution ------------------------------------------------------

    def test_resolve_opulenz_to_the_opl2_library(self):
        from lmms_mcp import lmms_app
        stems = _stems_of(LINUX_122_LIBS)
        assert lmms_app.resolve_plugin("opulenz", set(stems), stems) == "OPL2"

    def test_resolve_opulenz_when_the_library_is_called_opulenz(self):
        from lmms_mcp import lmms_app
        stems = _stems_of(["libopulenz.so", "libnes.so"])  # LMMS 1.3
        assert lmms_app.resolve_plugin(
            "opulenz", set(stems), stems) == "opulenz"

    def test_resolve_prefers_the_exact_library(self):
        from lmms_mcp import lmms_app
        stems = _stems_of(["libopulenz.so", "libOPL2.so"])
        assert lmms_app.resolve_plugin(
            "opulenz", set(stems), stems) == "opulenz"

    def test_resolve_freeboy_to_the_papu_library(self):
        from lmms_mcp import lmms_app
        stems = _stems_of(LINUX_122_LIBS)
        assert lmms_app.resolve_plugin(
            "freeboy", set(stems), stems, windows=False) == "papu"

    def test_resolve_freeboy_prefers_a_freeboy_library(self):
        from lmms_mcp import lmms_app
        stems = _stems_of(["libpapu.so", "libfreeboy.so"])
        assert lmms_app.resolve_plugin(
            "freeboy", set(stems), stems, windows=False) == "freeboy"

    def test_resolve_nes_stays_nes_and_never_becomes_papu(self):
        from lmms_mcp import lmms_app
        stems = _stems_of(LINUX_122_LIBS)
        assert lmms_app.resolve_plugin("nes", set(stems), stems) == "nes"
        only_papu = _stems_of(["libpapu.so", "libsid.so"])
        assert lmms_app.resolve_plugin(
            "nes", set(only_papu), only_papu) is None

    def test_resolve_ignores_case_of_the_requested_name(self):
        from lmms_mcp import lmms_app
        stems = _stems_of(LINUX_122_LIBS)
        assert lmms_app.resolve_plugin("OpUlEnZ", set(stems), stems) == "OPL2"

    def test_resolve_without_stems_uses_the_lowercase_name(self):
        from lmms_mcp import lmms_app
        assert lmms_app.resolve_plugin("opulenz", {"opl2"}) == "opl2"

    def test_freeboy_without_library_resolves_only_on_windows(self):
        from lmms_mcp import lmms_app
        stems = _stems_of(["libnes.so", "libsid.so"])
        assert lmms_app.resolve_plugin(
            "freeboy", set(stems), stems, windows=False) is None
        assert lmms_app.resolve_plugin(
            "freeboy", set(stems), stems, windows=True) == "freeboy"

    # --- the availability snapshot ---------------------------------------

    def test_snapshot_maps_offered_names_to_plugin_names(self):
        av = self._availability()
        assert av.plugin_names["opulenz"] == "OPL2"
        assert av.plugin_names["freeboy"] == "papu"
        assert av.plugin_names["nes"] == "nes"
        assert av.plugin_names["tripleoscillator"] == "tripleoscillator"
        assert set(av.plugin_names) == set(av.instruments)
        assert av.stems["opl2"] == "OPL2"

    def test_snapshot_plugin_name_covers_custom_plugins(self):
        av = self._availability()
        assert av.plugin_name("mycustomsynth") == "MyCustomSynth"
        assert av.plugin_name("OPULENZ") == "OPL2"
        assert av.plugin_name("not_installed") == "not_installed"

    def test_snapshot_freeboy_unavailable_on_linux_without_library(self):
        libs = [f for f in LINUX_122_LIBS if f != "libpapu.so"]
        av = self._availability(libs)
        assert "freeboy" not in av.instruments
        assert "freeboy" not in av.plugin_names
        assert "nes" in av.instruments

    def test_snapshot_freeboy_static_on_windows(self):
        libs = [f for f in LINUX_122_LIBS if f != "libpapu.so"]
        av = self._availability(libs, windows=True)
        assert "freeboy" in av.instruments
        assert av.plugin_names["freeboy"] == "freeboy"

    def test_snapshot_without_lmms_uses_the_offered_names(self):
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv
        av = lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS, set(), None)
        assert not av.verified
        assert av.plugin_name("opulenz") == "opulenz"
        assert av.plugin_name("freeboy") == "freeboy"

    def test_detect_availability_keeps_the_library_case(self, monkeypatch):
        from lmms_mcp import lmms_app
        stems = _stems_of(["libOPL2.so", "libpapu.so", "libdelay.so"])
        monkeypatch.setattr(
            lmms_app, "get_installed_plugins", lambda: set(stems))
        monkeypatch.setattr(lmms_app, "get_installed_stems", lambda: stems)
        monkeypatch.setattr(lmms_app, "get_lmms_version", lambda: "1.2.2")
        av = lmms_app.detect_availability(
            ["opulenz", "freeboy", "nes"], ["delay"])
        assert av.instruments == {"opulenz", "freeboy"}
        assert av.plugin_names == {"opulenz": "OPL2", "freeboy": "papu"}

    # --- add_instrument_track --------------------------------------------

    def test_opulenz_is_written_as_opl2(self, linux122):
        srv, proj = linux122
        result = json.loads(srv.add_instrument_track("FM", instrument="opulenz"))
        assert "error" not in result and "warning" not in result
        assert result["instrument"] == "opulenz"
        assert result["lmms_plugin"] == "OPL2"
        name, children = self._instrument_xml(proj)
        assert name == "OPL2"
        assert children == ["OPL2"]
        assert 'name="OPL2"' in proj.get_xml_string()

    def test_opulenz_spellings_resolve_to_the_same_plugin(self, linux122):
        srv, proj = linux122
        for i, spelling in enumerate(("OPL3", " Opulenz ", "opl2")):
            result = json.loads(
                srv.add_instrument_track("FM", instrument=spelling))
            assert result["instrument"] == "opulenz", spelling
            assert result["lmms_plugin"] == "OPL2", spelling
            assert self._instrument_xml(proj, i)[0] == "OPL2"

    def test_freeboy_is_written_as_papu(self, linux122):
        srv, proj = linux122
        result = json.loads(srv.add_instrument_track("GB", instrument="freeboy"))
        assert "error" not in result
        assert result["instrument"] == "freeboy"
        assert result["lmms_plugin"] == "papu"
        assert self._instrument_xml(proj) == ("papu", ["papu"])

    def test_same_name_plugins_report_no_lmms_plugin(self, linux122):
        srv, proj = linux122
        for i, inst in enumerate(("nes", "tripleoscillator", "zyn")):
            result = json.loads(srv.add_instrument_track("T", instrument=inst))
            assert "lmms_plugin" not in result, inst
            assert result["instrument"] == (
                "zynaddsubfx" if inst == "zyn" else inst)
            assert self._instrument_xml(proj, i)[0] == result["instrument"]

    def test_freeboy_refused_on_linux_without_papu(self, monkeypatch, linux122):
        srv, proj = linux122
        monkeypatch.setattr(srv, "_availability", self._availability(
            [f for f in LINUX_122_LIBS if f != "libpapu.so"]))
        result = json.loads(srv.add_instrument_track("GB", instrument="freeboy"))
        assert "'freeboy' is not available" in result["error"]
        assert "freeboy" not in result["available_instruments"]
        assert "nes" in result["available_instruments"]
        assert find_tracks(proj.root) == []
        # and it is not advertised anywhere
        assert "freeboy" not in srv.build_instructions(
            srv._availability, "/tmp/p")
        assert "freeboy" not in srv.render_plugin_doc(
            srv.add_instrument_track.__plugin_doc_template__,
            srv._availability)

    def test_freeboy_accepted_on_windows_without_library(self, monkeypatch,
                                                         linux122):
        srv, proj = linux122
        monkeypatch.setattr(srv, "_availability", self._availability(
            [f for f in LINUX_122_LIBS if f != "libpapu.so"], windows=True))
        result = json.loads(srv.add_instrument_track("GB", instrument="freeboy"))
        assert "error" not in result
        assert "lmms_plugin" not in result
        assert self._instrument_xml(proj)[0] == "freeboy"

    def test_custom_plugin_keeps_its_library_case(self, linux122):
        srv, proj = linux122
        result = json.loads(
            srv.add_instrument_track("Mine", instrument="mycustomsynth"))
        assert result["custom_plugin"] is True
        assert result["instrument"] == "mycustomsynth"
        assert result["lmms_plugin"] == "MyCustomSynth"
        assert self._instrument_xml(proj) == (
            "MyCustomSynth", ["MyCustomSynth"])

    def test_custom_plugin_with_lowercase_library(self, linux122):
        srv, proj = linux122
        result = json.loads(srv.add_instrument_track("GB", instrument="papu"))
        assert result["custom_plugin"] is True
        assert "lmms_plugin" not in result
        assert self._instrument_xml(proj)[0] == "papu"

    def test_added_tracks_keep_working_with_other_tools(self, linux122,
                                                        tmp_path):
        """Name checks in other tools only look at zynaddsubfx/afp."""
        srv, proj = linux122
        srv.add_instrument_track("FM", instrument="opulenz")
        srv.add_instrument_track("Zyn", instrument="zynaddsubfx")
        srv.add_instrument_track("Drum", instrument="audiofileprocessor")
        error = json.loads(srv.set_audiofile_sample(0, "drums/kick01.ogg"))
        assert "uses instrument 'OPL2'" in error["error"]
        assert "error" not in json.loads(
            srv.set_audiofile_sample(2, "drums/kick01.ogg"))
        assert "error" not in json.loads(srv.set_zyn_params(1, portamento=0))
        assert "error" not in json.loads(srv.add_note(0, 60, 0, 48, 100))
        tracks = json.loads(srv.list_tracks())["tracks"]
        assert [t["instrument"] for t in tracks] == [
            "OPL2", "zynaddsubfx", "audiofileprocessor"]
        saved = Path(proj.save(tmp_path / "p.mmp", compressed=False))
        assert '<instrument name="OPL2">' in saved.read_text()

    # --- what the agent is shown -----------------------------------------

    def test_list_available_plugins_reports_renamed_instruments(
            self, linux122):
        srv, _ = linux122
        result = json.loads(srv.list_available_plugins())
        assert result["lmms_plugin_names"] == {
            "opulenz": "OPL2", "freeboy": "papu"}
        assert "opulenz" in result["built_in_instruments"]
        assert "papu" not in result["custom_plugins_unknown_type"]
        assert "opl2" not in result["custom_plugins_unknown_type"]

    def test_instruments_resource_reports_renamed_instruments(self, linux122):
        srv, _ = linux122
        entries = {
            e["name"]: e
            for e in json.loads(srv.resource_instruments())["instruments"]
        }
        assert entries["opulenz"]["lmms_plugin"] == "OPL2"
        assert entries["freeboy"]["lmms_plugin"] == "papu"
        assert "lmms_plugin" not in entries["nes"]

    def test_tool_description_offers_the_renamed_instruments(self, linux122):
        srv, _ = linux122
        doc = srv.render_plugin_doc(
            srv.add_instrument_track.__plugin_doc_template__,
            srv._availability)
        assert "opulenz" in doc and "freeboy" in doc
        assert "lmms_plugin" in doc
