"""Tests for LMMS MCP Server."""

import json
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from lmms_mcp.models import (
    KEY_MAX,
    KEY_MIN,
    MIDI_NOTE_OFFSET,
    Note,
    bars_to_ticks,
    key_to_note_name,
    note_name_to_key,
    ticks_to_bars,
)
from lmms_mcp.xml_parser import create_empty_project, load_project, save_project, find_tracks, add_instrument_track, add_note_to_track, add_pattern_track, add_sample_track, get_track_mixer_channel
from lmms_mcp.project import LMMSProject

from audio_helpers import FakeLmms, read_pcm, sine


class TestModels:
    def test_key_to_note_name(self):
        assert key_to_note_name(48) == "C4"
        assert key_to_note_name(57) == "A4"
        assert key_to_note_name(0) == "C0"
        assert key_to_note_name(108) == "C9"
        assert key_to_note_name(34) == "A#2"

    def test_note_name_to_key(self):
        assert note_name_to_key("C4") == 48
        assert note_name_to_key("A4") == 57
        assert note_name_to_key("A#3") == 46
        assert note_name_to_key("C0") == 0
        assert note_name_to_key("C9") == 108

    def test_numbering_matches_measured_pitch(self):
        # Measured with LMMS 1.2.2 renders: key 45 -> 220 Hz, key 57 ->
        # 440 Hz, key 69 -> 880 Hz, key 48 -> 261.63 Hz (middle C).
        def hz(name):
            return 440 * 2 ** ((note_name_to_key(name) - 57) / 12)

        assert hz("A3") == pytest.approx(220)
        assert hz("A4") == pytest.approx(440)
        assert hz("A5") == pytest.approx(880)
        assert hz("C4") == pytest.approx(261.63, abs=0.01)

    def test_midi_note_is_key_plus_12(self):
        assert MIDI_NOTE_OFFSET == 12
        assert note_name_to_key("C4") + MIDI_NOTE_OFFSET == 60  # MIDI middle C
        assert note_name_to_key("A4") + MIDI_NOTE_OFFSET == 69  # MIDI A440

    def test_key_range_is_c0_to_c9(self):
        assert (KEY_MIN, KEY_MAX) == (0, 108)
        assert key_to_note_name(KEY_MIN) == "C0"
        assert key_to_note_name(KEY_MAX) == "C9"

    def test_roundtrip_note_names(self):
        for key in range(KEY_MIN, KEY_MAX + 1):
            name = key_to_note_name(key)
            assert note_name_to_key(name) == key

    @pytest.mark.parametrize("name,key", [
        ("Bb2", 34), ("A#2", 34), ("Eb4", 51), ("Db3", 37),
        ("Cb4", 47), ("B#3", 48), ("E#4", 53), ("Fb4", 52),
        ("bb2", 34), ("b3", 47), ("B3", 47), ("c4", 48), ("c#4", 49),
        ("C0", 0), ("C9", 108), ("Cb1", 11), ("B\u266d2", 34),
        ("F\u266f3", 42), ("Bbb3", 45), ("F##3", 43), (" Bb2 ", 34),
        ("B#0", 12), ("B#8", 108),
    ])
    def test_note_name_to_key_flats_and_enharmonics(self, name, key):
        assert note_name_to_key(name) == key

    def test_flat_and_sharp_enharmonics_agree(self):
        # Every sharp name (C#, D#, F#, G#, A#) has a flat twin one letter
        # up in the same octave (Db, Eb, Gb, Ab, Bb).
        for key in range(KEY_MIN, KEY_MAX + 1):
            sharp = key_to_note_name(key)
            if "#" not in sharp:
                continue
            flat_letter = "ABCDEFG"[("ABCDEFG".index(sharp[0]) + 1) % 7]
            octave = sharp[2:]
            assert note_name_to_key(f"{flat_letter}b{octave}") == key

    @pytest.mark.parametrize("name", [
        "C#9", "Db9", "C10", "Cb0", "C-1", "B-1", "Bb-1", "A9",
    ])
    def test_note_name_out_of_range(self, name):
        with pytest.raises(ValueError, match=r"out of range.*0-108.*C0 to C9"):
            note_name_to_key(name)

    @pytest.mark.parametrize("name", [
        "H2", "", "  ", "C", "4", "60", "BB2", "C#b4", "Cbbb4", "bbbb3",
        "C 4", "Do4", "C#",
    ])
    def test_note_name_invalid_states_format(self, name):
        with pytest.raises(ValueError) as exc:
            note_name_to_key(name)
        msg = str(exc.value)
        assert f"Invalid note name {name!r}" in msg
        assert "Bb2" in msg and "F#3" in msg and "C4 = 48" in msg

    def test_note_name_non_string_rejected(self):
        with pytest.raises(ValueError, match="Invalid note name"):
            note_name_to_key(48)

    def test_old_midi_helper_names_are_gone(self):
        from lmms_mcp import models

        assert not hasattr(models, "note_name_to_midi")
        assert not hasattr(models, "midi_to_note_name")

    def test_bars_to_ticks(self):
        assert bars_to_ticks(1) == 192
        assert bars_to_ticks(4) == 768

    def test_ticks_to_bars(self):
        assert ticks_to_bars(192) == 1.0
        assert ticks_to_bars(768) == 4.0

    def test_note_to_dict(self):
        note = Note(key=48, pos=0, length=48, volume=100)
        d = note.to_dict()
        assert d["key"] == 48
        assert d["note_name"] == "C4"

    def test_note_default_key_is_middle_c(self):
        assert Note().key == 48
        assert Note().to_dict()["note_name"] == "C4"
        assert Note.from_dict({}).key == 48

    def test_note_from_dict(self):
        note = Note.from_dict({"key": 57, "pos": 48, "length": 96})
        assert note.key == 57
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


def _floats(text):
    from lmms_mcp.effect_params import unpack_floats
    return unpack_floats(text)


class TestEffectParamTable:
    """The per-effect parameter table (names, ranges, defaults)."""

    def test_every_offered_effect_has_a_table_and_a_start(self):
        from lmms_mcp import effect_params as ep
        from lmms_mcp.effects import KNOWN_EFFECTS
        # the 1.3-only plugins have no table yet; everything else does
        without = set(KNOWN_EFFECTS) - set(ep.EFFECT_PARAMS)
        assert without == {"compressor", "dispersion", "frequencyshifter",
                           "slewdistortion"}
        assert set(ep.EFFECT_START) == set(ep.EFFECT_PARAMS)

    def test_names_and_attributes_are_unique_per_effect(self):
        from lmms_mcp import effect_params as ep
        for effect, params in ep.EFFECT_PARAMS.items():
            names = [p.name.lower() for p in params]
            attrs = [p.attr.lower() for p in params if p.kind != "curve"]
            assert len(set(names)) == len(names), effect
            assert len(set(attrs)) == len(attrs), effect
            # a name never means a different parameter's attribute
            for p in params:
                other = [q for q in params if q is not p]
                assert p.name.lower() not in {q.attr.lower() for q in other
                                              if q.kind != "curve"}, (effect, p.name)

    def test_defaults_lie_in_the_range_and_everything_is_described(self):
        from lmms_mcp import effect_params as ep
        for effect, params in ep.EFFECT_PARAMS.items():
            for p in params:
                assert p.description, (effect, p.name)
                if p.minimum is not None:
                    assert p.minimum < p.maximum, (effect, p.name)
                    assert p.minimum <= p.default <= p.maximum, (effect, p.name)
                if p.kind == "choice":
                    assert p.default in p.choices, (effect, p.name)

    def test_real_attribute_names(self):
        """The attribute names LMMS 1.2 reads (checked against the plugins)."""
        from lmms_mcp import effect_params as ep
        attrs = {e: {p.name: p.attr for p in ps}
                 for e, ps in ep.EFFECT_PARAMS.items()}
        assert attrs["delay"] == {
            "time_s": "DelayTimeSamples", "feedback": "FeebackAmount",
            "lfo_period_s": "LfoFrequency", "lfo_amount_s": "LfoAmount",
            "out_gain_db": "OutGain"}
        assert attrs["bitcrush"]["sample_rate_hz"] == "rate"
        assert attrs["bitcrush"]["bit_levels"] == "levels"
        assert attrs["bitcrush"]["rate_on"] == "rateon"
        assert attrs["reverbsc"] == {
            "input_gain_db": "input_gain", "size": "size",
            "color_hz": "color", "output_gain_db": "output_gain"}
        assert attrs["eq"]["peak1_gain_db"] == "Peak1gain"
        assert attrs["eq"]["highshelf_gain_db"] == "HighShelfgain"
        assert attrs["eq"]["highshelf_on"] == "Highshelfactive"
        assert attrs["stereomatrix"]["left_to_right"] == "l-r"
        assert attrs["crossovereq"]["band1_on"] == "mute1"

    def test_documented_defaults_are_the_plugin_defaults(self):
        from lmms_mcp import effect_params as ep
        d = {e: {p.name: p.default for p in ps}
             for e, ps in ep.EFFECT_PARAMS.items()}
        assert d["delay"]["time_s"] == 0.5 and d["delay"]["feedback"] == 0
        assert d["reverbsc"]["size"] == 0.89
        assert d["bitcrush"]["sample_rate_hz"] == 44100
        assert d["bitcrush"]["bit_levels"] == 256
        assert d["dualfilter"]["cutoff1_hz"] == 7000
        assert d["flanger"]["lfo_period_s"] == 0.25

    def test_wet_is_below_one_exactly_where_the_dry_signal_is_missing(self):
        """delay, reverbsc and flanger output only the processed signal
        (verified by rendering with LMMS 1.2.2); all others start at 1."""
        from lmms_mcp import effect_params as ep
        low = {e for e, s in ep.EFFECT_START.items() if s.wet < 1.0}
        assert low == {"delay", "reverbsc", "flanger"}
        assert {e for e, s in ep.EFFECT_START.items() if s.why_wet} == low
        assert 0.3 <= ep.EFFECT_START["delay"].wet <= 0.5
        assert 0.3 <= ep.EFFECT_START["reverbsc"].wet <= 0.5

    def test_start_settings_are_valid_for_every_effect_and_tempo(self):
        import xml.etree.ElementTree as ET
        from lmms_mcp import effect_params as ep
        for effect in ep.EFFECT_START:
            for bpm in (10, 60, 120, 140, 999):
                controls = ET.Element("c")
                ep.ensure_graph_defaults(effect, controls)
                ep.apply_params(effect, controls, ep.start_params(effect, bpm))


class TestAddEffectParams:
    """add_effect: starting settings, params and wet."""

    @pytest.fixture
    def proj(self):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new(bpm=120)
        srv.set_project(proj)
        srv.add_instrument_track("Lead")
        return proj

    @staticmethod
    def controls(proj, effect):
        node = proj.root.find(
            f".//instrumenttrack/fxchain/effect[@name='{effect}']")
        assert node is not None
        return node, node[0]

    @staticmethod
    def add(*args, **kwargs):
        from lmms_mcp import server as srv
        return json.loads(srv.add_effect("track", 0, *args, **kwargs))

    def test_plain_delay_keeps_the_dry_sound_and_repeats(self, proj):
        result = self.add("delay")
        assert result["wet"] == 0.4
        assert result["params"] == {"time_s": 0.25, "feedback": 0.3}
        assert "dry sound" in result["note"] and "wet" in result["note"]
        effect, controls = self.controls(proj, "delay")
        assert effect.get("wet") == "0.4"
        assert controls.tag == "Delay"
        assert controls.get("DelayTimeSamples") == "0.25"
        assert controls.get("FeebackAmount") == "0.3"

    def test_delay_time_is_an_eighth_note_of_the_project_tempo(self, proj):
        from lmms_mcp import server as srv
        for bpm, seconds in ((60, 0.5), (90, 0.3333), (120, 0.25),
                             (140, 0.2143), (200, 0.15), (999, 0.03)):
            srv.set_tempo(bpm)
            srv.remove_effect("track", 0, "delay")
            assert self.add("delay")["params"]["time_s"] == pytest.approx(
                seconds, abs=1e-4), bpm
        srv.set_tempo(10)  # 3 s, still inside the plugin's 5 s
        srv.remove_effect("track", 0, "delay")
        assert self.add("delay")["params"]["time_s"] == 3.0

    def test_explicit_wet_wins_and_has_no_wet_note(self, proj):
        result = self.add("delay", wet=1.0)
        assert result["wet"] == 1.0
        assert "wet defaulted" not in result.get("note", "")
        assert self.controls(proj, "delay")[0].get("wet") == "1"

    def test_wet_default_per_effect(self, proj):
        from lmms_mcp import effect_params as ep
        for effect in ep.EFFECT_START:
            result = self.add(effect)
            assert "error" not in result, result
            assert result["wet"] == ep.EFFECT_START[effect].wet, effect
            assert ("wet defaulted" in result.get("note", "")) == (
                ep.EFFECT_START[effect].wet != 1.0), effect

    def test_bad_wet_is_refused(self, proj):
        for wet in (1.5, -0.1):
            result = self.add("delay", wet=wet)
            assert "'wet'" in result["error"] and "0 to 1" in result["error"]
        assert "must be a number" in self.add("delay", wet="loud")["error"]
        assert "Give wet once" in self.add(
            "delay", wet=0.5, params={"wet": 0.2})["error"]
        assert proj.root.find(".//instrumenttrack/fxchain/effect") is None

    def test_wet_may_be_given_in_params(self, proj):
        result = self.add("delay", params={"wet": 0.6, "time_s": 0.2})
        assert result["wet"] == 0.6 and result["params"] == {
            "time_s": 0.2, "feedback": 0.3}

    def test_effects_with_echoes_or_tails_start_with_a_long_decay(self, proj):
        """LMMS stops an effect after "decay" ms of silent output: at its
        default of 1 ms a delay loses its echo after a gap (rendered with
        LMMS 1.2.2 without disableautoquit: echo at 1 s gone at decay 1 and
        100, present at 1000 and 8000)."""
        from lmms_mcp import effect_params as ep
        from lmms_mcp import server as srv
        long_decay = {e for e, s in ep.EFFECT_START.items() if s.decay_ms > 1}
        assert long_decay == {"delay", "reverbsc", "multitapecho", "flanger"}
        for effect in ep.EFFECT_START:
            result = self.add(effect)
            expected = 8000 if effect in long_decay else 1
            assert result["decay_ms"] == expected, effect
            assert self.controls(proj, effect)[0].get("autoquit") == str(
                expected), effect
            assert ("decay_ms defaulted" in result.get("note", "")) == (
                effect in long_decay), effect
        srv.remove_effect("track", 0, "delay")
        result = self.add("delay", params={"decay_ms": 500})
        assert result["decay_ms"] == 500
        assert "decay_ms defaulted" not in result.get("note", "")
        srv.remove_effect("track", 0, "reverbsc")
        assert "range" in self.add(
            "reverbsc", params={"decay_ms": 9000})["error"]

    def test_params_override_the_start_and_keep_the_rest(self, proj):
        result = self.add("delay", params={"time_s": 0.375})
        assert result["params"] == {"time_s": 0.375, "feedback": 0.3}
        assert "note" in result  # wet explanation still applies
        _, controls = self.controls(proj, "delay")
        assert controls.get("DelayTimeSamples") == "0.375"

    def test_xml_attribute_names_are_accepted_too(self, proj):
        result = self.add("delay", params={"FeebackAmount": 0.6,
                                          "delaytimesamples": 0.2})
        assert result["params"] == {"time_s": 0.2, "feedback": 0.6}

    def test_unknown_name_adds_nothing_and_lists_valid_ones(self, proj):
        result = self.add("delay", params={"echo": 1})
        assert "no parameter 'echo'" in result["error"]
        assert result["valid_parameters"] == [
            "time_s", "feedback", "lfo_period_s", "lfo_amount_s",
            "out_gain_db"]
        assert "time_s" in result["error"]
        assert proj.root.find(".//instrumenttrack/fxchain/effect") is None

    def test_out_of_range_is_refused_with_the_range(self, proj):
        result = self.add("delay", params={"time_s": 9})
        assert "out of range" in result["error"]
        assert "0.01 to 5 s" in result["error"]
        result = self.add("reverbsc", params={"size": -0.1})
        assert "0 to 1" in result["error"]
        assert proj.root.find(".//instrumenttrack/fxchain/effect") is None

    def test_value_types(self, proj):
        assert "must be a number" in self.add(
            "delay", params={"feedback": "high"})["error"]
        assert "must be a number" in self.add(
            "delay", params={"feedback": True})["error"]
        assert "whole number" in self.add(
            "bitcrush", params={"bit_levels": 3.5})["error"]
        assert "true or false" in self.add(
            "bitcrush", params={"rate_on": 2})["error"]
        assert "must be one of" in self.add(
            "dualfilter", params={"filter1_type": "wobble"})["error"]
        assert "finite" in self.add(
            "delay", params={"feedback": float("nan")})["error"]
        assert proj.root.find(".//instrumenttrack/fxchain/effect") is None

    def test_the_same_parameter_twice_is_refused(self, proj):
        result = self.add("delay", params={"time_s": 0.2,
                                          "DelayTimeSamples": 0.3})
        assert "twice" in result["error"]

    def test_values_are_written_as_lmms_reads_them(self, proj):
        self.add("bitcrush", params={
            "bit_levels": 8, "rate_on": False, "output_gain_db": -3.5})
        _, c = self.controls(proj, "bitcrush")
        assert c.tag == "bitcrushcontrols"
        assert (c.get("levels"), c.get("rateon"), c.get("outgain")) == (
            "8", "0", "-3.5")
        assert c.get("rate") == "8000"  # the start, not overridden

    def test_choices_by_name_and_index(self, proj):
        self.add("dualfilter", params={"filter1_type": "HighPass",
                                      "filter2_type": 6})
        _, c = self.controls(proj, "dualfilter")
        assert (c.get("filter1"), c.get("filter2")) == ("1", "6")

    def test_eq_slope_writes_the_flags_lmms_reads(self, proj):
        self.add("eq", params={"lp_slope_db": 24, "hp_slope_db": 48})
        _, c = self.controls(proj, "eq")
        assert (c.get("LP"), c.get("LP12"), c.get("LP24"), c.get("LP48")) == (
            "1", "0", "1", "0")
        assert (c.get("HP"), c.get("HP12"), c.get("HP24"), c.get("HP48")) == (
            "2", "0", "0", "1")

    def test_neutral_effects_say_so_unless_params_are_given(self, proj):
        for effect in ("amplifier", "crossovereq", "eq", "stereomatrix"):
            result = self.add(effect)
            assert "Neutral" in result["note"], effect
            assert "set_effect_params" in result["note"], effect
        from lmms_mcp import server as srv
        srv.remove_effect("track", 0, "amplifier")
        result = self.add("amplifier", params={"volume": 150})
        assert "note" not in result
        assert result["params"] == {"volume": 150.0}

    def test_effects_without_a_table_take_no_params(self, proj):
        result = self.add("compressor")
        assert result["wet"] == 1.0 and result["params"] == {}
        result = self.add("dispersion", params={"amount": 1})
        assert "no settable parameters" in result["error"]

    def test_mixer_channel_target(self, proj):
        from lmms_mcp import server as srv
        srv.add_mixer_channel("Bus")
        result = json.loads(srv.add_effect(
            "mixer", 1, "reverbsc", params={"size": 0.5}))
        assert result["params"] == {"size": 0.5} and result["wet"] == 0.3

    def test_graph_effects_always_get_their_curves(self, proj):
        """LMMS plays uninitialised memory when a curve is missing."""
        self.add("waveshaper", params={"shape": "linear", "input_gain": 1,
                                       "output_gain": 1})
        self.add("dynamicsprocessor", params={"threshold_db": 0, "ratio": 1})
        self.add("multitapecho")
        _, ws = self.controls(proj, "waveshaper")
        assert len(_floats(ws.get("waveShape"))) == 200
        _, dp = self.controls(proj, "dynamicsprocessor")
        assert _floats(dp.get("waveShape")) == pytest.approx(
            [(i + 1) / 200 for i in range(200)])
        _, mt = self.controls(proj, "multitapecho")
        assert len(_floats(mt.get("ampsteps"))) == 8
        assert len(_floats(mt.get("lpsteps"))) == 8
        assert mt.get("steps") == "8"

    def test_waveshaper_shapes(self, proj):
        from lmms_mcp import effect_params as ep
        for name in ep.WAVESHAPES:
            curve = ep.waveshape_curve(name)
            assert len(curve) == 200 and all(0 <= v <= 1 for v in curve), name
        assert ep.waveshape_curve("linear")[99] == pytest.approx(0.5)
        assert ep.waveshape_curve("soft_clip")[-1] == pytest.approx(1.0)
        result = self.add("waveshaper", params={"shape": "fuzz"})
        assert result["params"]["shape"] == "fuzz"
        _, c = self.controls(proj, "waveshaper")
        assert _floats(c.get("waveShape")) == pytest.approx(
            ep.waveshape_curve("fuzz"), abs=1e-6)

    def test_waveshaper_custom_curve_and_bad_curves(self, proj):
        custom = [min(1.0, i / 100) for i in range(200)]
        result = self.add("waveshaper", params={"shape": custom})
        assert result["params"]["shape"] == "custom"
        from lmms_mcp import server as srv
        srv.remove_effect("track", 0, "waveshaper")
        for bad in ("wobble", [0.5] * 10, [2.0] * 200, 3):
            assert "error" in self.add("waveshaper", params={"shape": bad})

    def test_dynamics_curve_is_a_compressor_curve(self, proj):
        import math
        result = self.add("dynamicsprocessor",
                          params={"threshold_db": -12, "ratio": 4})
        assert result["params"]["threshold_db"] == -12
        _, c = self.controls(proj, "dynamicsprocessor")
        curve = _floats(c.get("waveShape"))
        t = 10 ** (-12 / 20)
        for i, y in enumerate(curve):
            x = (i + 1) / 200
            if x <= t:
                assert y == pytest.approx(x, abs=1e-6)
        # the full-scale level comes out 9 dB down: -12 + 12/4
        assert 20 * math.log10(curve[-1]) == pytest.approx(-9.0, abs=0.05)

    def test_multitapecho_taps(self, proj):
        from lmms_mcp import effect_params as ep
        result = self.add("multitapecho", params={
            "steps": 4, "tap_levels_db": [-6, -12, -18, -24],
            "tap_cutoff_hz": 2000})
        assert result["params"]["tap_levels_db"] == [-6, -12, -18, -24]
        assert result["params"]["tap_cutoff_hz"] == [2000.0] * 4
        _, c = self.controls(proj, "multitapecho")
        assert _floats(c.get("ampsteps")) == [-6, -12, -18, -24]
        assert _floats(c.get("lpsteps")) == pytest.approx(
            [math_log(2000 / 20)] * 4, abs=1e-5)
        assert c.get("steps") == "4"
        assert ep.decay_levels(3) == [-9.0, -13.0, -17.0]

    def test_multitapecho_step_length_is_a_sixteenth_note(self, proj):
        from lmms_mcp import server as srv
        srv.set_tempo(120)
        assert self.add("multitapecho")["params"]["step_length_ms"] == 125.0
        srv.remove_effect("track", 0, "multitapecho")
        srv.set_tempo(20)  # 750 ms would exceed the plugin's 500 ms
        assert self.add("multitapecho")["params"]["step_length_ms"] == 500.0

    def test_multitapecho_wrong_number_of_taps(self, proj):
        result = self.add("multitapecho", params={
            "steps": 4, "tap_levels_db": [-6, -12]})
        assert "needs 4 values" in result["error"]

    def test_crossovers_must_ascend(self, proj):
        result = self.add("crossovereq", params={"xover_1_2_hz": 3000})
        assert "must ascend" in result["error"]
        result = self.add("crossovereq", params={
            "xover_1_2_hz": 300, "xover_2_3_hz": 900, "xover_3_4_hz": 3000})
        assert "error" not in result


def math_log(x):
    import math
    return math.log10(x)


class TestSetEffectParams:
    """set_effect_params and get_effect_chain."""

    @pytest.fixture
    def proj(self):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new(bpm=120)
        srv.set_project(proj)
        srv.add_instrument_track("Lead")
        srv.add_mixer_channel("Bus")
        return proj

    @staticmethod
    def call(fn, *args):
        return json.loads(fn(*args))

    def test_sets_on_a_track_and_keeps_the_other_params(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "delay")
        result = self.call(srv.set_effect_params, "track", 0, "delay",
                           {"time_s": 0.5, "feedback": 0.6})
        assert result["params"] == {"time_s": 0.5, "feedback": 0.6}
        assert result["effect"] == "delay" and result["position"] == 0
        controls = proj.root.find(".//instrumenttrack/fxchain/effect/Delay")
        assert controls.get("DelayTimeSamples") == "0.5"
        assert controls.get("FeebackAmount") == "0.6"
        self.call(srv.set_effect_params, "track", 0, "delay", {"feedback": 0.1})
        assert controls.get("DelayTimeSamples") == "0.5"
        assert controls.get("FeebackAmount") == "0.1"

    def test_sets_on_a_mixer_channel_by_position(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("mixer", 1, "eq")
        srv.add_effect("mixer", 1, "reverbsc")
        result = self.call(srv.set_effect_params, "mixer", 1, 1,
                           {"size": 0.7, "color_hz": 4000})
        assert result["effect"] == "reverbsc" and result["position"] == 1
        node = [c for c in xml_parser_channels(proj)
                if c.get("num") == "1"][0]
        controls = node.find("fxchain/effect[@name='reverbsc']/ReverbSCControls")
        assert (controls.get("size"), controls.get("color")) == ("0.7", "4000")

    def test_wet_can_be_changed_and_is_validated(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "delay")
        result = self.call(srv.set_effect_params, "track", 0, "delay",
                           {"wet": 0.25, "feedback": 0.2})
        assert result["params"] == {"wet": 0.25, "feedback": 0.2}
        effect = proj.root.find(".//instrumenttrack/fxchain/effect")
        assert effect.get("wet") == "0.25"
        bad = self.call(srv.set_effect_params, "track", 0, "delay", {"wet": 2})
        assert "'wet'" in bad["error"] and "0 to 1" in bad["error"]
        assert effect.get("wet") == "0.25"

    def test_decay_can_be_changed_and_is_listed(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "delay")
        srv.add_effect("track", 0, "eq")
        result = self.call(srv.set_effect_params, "track", 0, "eq",
                           {"decay_ms": 2000, "hp_on": True})
        assert result["params"] == {"decay_ms": 2000, "hp_on": True}
        effect = proj.root.find(".//instrumenttrack/fxchain/effect[@name='eq']")
        assert effect.get("autoquit") == "2000"
        chain = self.call(srv.get_effect_chain, "track", 0)["effects"]
        assert [e["decay_ms"] for e in chain] == [8000, 2000]
        # either spelling
        self.call(srv.set_effect_params, "track", 0, "eq", {"autoquit": 5})
        assert effect.get("autoquit") == "5"
        assert "range" in self.call(srv.set_effect_params, "track", 0, "eq",
                                    {"decay_ms": 0})["error"]

    def test_a_refused_call_changes_nothing(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "delay")
        controls = proj.root.find(".//instrumenttrack/fxchain/effect/Delay")
        before = dict(controls.attrib)
        for params in ({"time_s": 0.9, "feedback": 7},
                       {"time_s": 0.9, "nonsense": 1}):
            result = self.call(srv.set_effect_params, "track", 0, "delay",
                               params)
            assert "error" in result
            assert dict(controls.attrib) == before
        result = self.call(srv.set_effect_params, "track", 0, "delay",
                           {"nonsense": 1})
        assert result["valid_parameters"][0] == "time_s"
        assert "0 to 1" in self.call(
            srv.set_effect_params, "track", 0, "delay",
            {"feedback": 7})["error"]

    def test_errors(self, proj):
        from lmms_mcp import server as srv
        assert "not found" in self.call(
            srv.set_effect_params, "track", 0, "delay", {"time_s": 1})["error"]
        srv.add_effect("track", 0, "delay")
        assert "not found" in self.call(
            srv.set_effect_params, "track", 0, "eq", {"hp_on": True})["error"]
        assert "not found" in self.call(
            srv.set_effect_params, "track", 0, 4, {"hp_on": True})["error"]
        assert "at least one" in self.call(
            srv.set_effect_params, "track", 0, "delay", {})["error"]
        assert "error" in self.call(
            srv.set_effect_params, "track", 7, "delay", {"time_s": 1})
        assert "error" in self.call(
            srv.set_effect_params, "bus", 0, "delay", {"time_s": 1})

    def test_multitapecho_steps_resize_the_curves(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "multitapecho")
        result = self.call(srv.set_effect_params, "track", 0, "multitapecho",
                           {"steps": 12})
        assert result["params"]["steps"] == 12
        c = proj.root.find(".//instrumenttrack/fxchain/effect/multitapechocontrols")
        assert len(_floats(c.get("ampsteps"))) == 12
        assert len(_floats(c.get("lpsteps"))) == 12
        self.call(srv.set_effect_params, "track", 0, "multitapecho",
                  {"steps": 5})
        assert len(_floats(c.get("ampsteps"))) == 5
        result = self.call(srv.set_effect_params, "track", 0, "multitapecho",
                           {"tap_levels_db": -20})
        assert _floats(c.get("ampsteps")) == [-20.0] * 5

    def test_dynamics_hints_when_nothing_would_be_reduced(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "dynamicsprocessor",
                       params={"threshold_db": -6, "ratio": 1})
        result = self.call(srv.set_effect_params, "track", 0,
                           "dynamicsprocessor", {"threshold_db": -20})
        assert "ratio is 1" in result["note"]
        # a curve with ratio 1 holds no threshold, so it reads back as 0 dB
        result = self.call(srv.set_effect_params, "track", 0,
                           "dynamicsprocessor", {"ratio": 3})
        assert "threshold_db is 0" in result["note"]
        result = self.call(srv.set_effect_params, "track", 0,
                           "dynamicsprocessor",
                           {"ratio": 3, "threshold_db": -20})
        assert "note" not in result

    def test_dynamics_threshold_and_ratio_are_set_separately(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "dynamicsprocessor")
        self.call(srv.set_effect_params, "track", 0, "dynamicsprocessor",
                  {"ratio": 8})
        chain = self.call(srv.get_effect_chain, "track", 0)
        assert chain["effects"][0]["params"] == {
            "threshold_db": -12.0, "ratio": 8.0}

    def test_dualfilter_and_eq_names(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "dualfilter")
        result = self.call(srv.set_effect_params, "track", 0, "dualfilter",
                           {"filter1_type": "notch", "mix": -1})
        assert result["params"] == {"filter1_type": "notch", "mix": -1.0}
        srv.add_effect("track", 0, "eq")
        result = self.call(srv.set_effect_params, "track", 0, "eq", {
            "peak2_on": True, "peak2_gain_db": -6, "peak2_freq_hz": 300,
            "peak2_bw": 1.5, "hp_on": True, "hp_freq_hz": 40})
        c = proj.root.find(".//instrumenttrack/fxchain/effect/Eq")
        assert (c.get("Peak2active"), c.get("Peak2gain"),
                c.get("Peak2freq"), c.get("Peak2bw")) == (
                    "1", "-6", "300", "1.5")

    def test_get_effect_chain_lists_only_changed_params(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "delay")
        srv.add_effect("track", 0, "eq")
        srv.add_effect("track", 0, "waveshaper")
        srv.add_effect("track", 0, "compressor")
        srv.set_effect_params("track", 0, "eq", {"lp_on": True,
                                                  "lp_slope_db": 24})
        srv.set_effect_params("track", 0, "waveshaper", {"output_gain": 1})
        chain = self.call(srv.get_effect_chain, "track", 0)["effects"]
        by = {e["name"]: e for e in chain}
        assert by["delay"]["params"] == {"time_s": 0.25, "feedback": 0.3}
        assert by["delay"]["wet"] == 0.4
        assert by["eq"]["params"] == {"lp_on": True, "lp_slope_db": 24}
        assert by["waveshaper"]["params"] == {
            "input_gain": 1.5, "shape": "soft_clip"}
        assert "params" not in by["compressor"]
        # a switched-off value goes back to being unlisted
        srv.set_effect_params("track", 0, "delay", {"feedback": 0})
        chain = self.call(srv.get_effect_chain, "track", 0)["effects"]
        assert chain[0]["params"] == {"time_s": 0.25}

    def test_chain_of_a_mixer_channel_shows_params(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("mixer", 1, "bitcrush")
        chain = self.call(srv.get_effect_chain, "mixer", 1)["effects"]
        assert chain[0]["params"] == {"sample_rate_hz": 8000, "bit_levels": 16}

    def test_custom_curves_are_reported_as_custom(self, proj):
        from lmms_mcp import effect_params as ep
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "waveshaper")
        srv.add_effect("track", 0, "dynamicsprocessor")
        srv.set_effect_params("track", 0, "waveshaper",
                              {"shape": [0.5] * 200})
        c = proj.root.find(".//instrumenttrack/fxchain/effect/dynamicsprocessor_controls")
        c.set("waveShape", ep.pack_floats([0.5] * 200))
        chain = {e["name"]: e["params"] for e in self.call(
            srv.get_effect_chain, "track", 0)["effects"]}
        assert chain["waveshaper"]["shape"] == "custom"
        assert chain["dynamicsprocessor"]["curve"] == "custom"

    def test_legacy_project_with_empty_controls(self, proj):
        """Effects written by earlier versions have no curve (LMMS plays
        garbage); changing any parameter repairs that."""
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "waveshaper")
        c = proj.root.find(".//instrumenttrack/fxchain/effect/waveshapercontrols")
        c.attrib.clear()
        chain = self.call(srv.get_effect_chain, "track", 0)["effects"]
        assert chain[0]["params"] == {}
        self.call(srv.set_effect_params, "track", 0, "waveshaper",
                  {"input_gain": 2})
        assert len(_floats(c.get("waveShape"))) == 200

    def test_loaded_project_with_log_scale_children(self, proj):
        """LMMS saves some knobs (eq frequencies...) as child elements."""
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "eq")
        c = proj.root.find(".//instrumenttrack/fxchain/effect/Eq")
        child = ET.SubElement(c, "Peak1freq", {
            "id": "1234", "value": "800", "scale_type": "log"})
        ET.SubElement(c, "Peak1gain", {"id": "1235", "value": "4.5",
                                      "scale_type": "log"})
        chain = self.call(srv.get_effect_chain, "track", 0)["effects"]
        assert chain[0]["params"] == {"peak1_freq_hz": 800.0,
                                      "peak1_gain_db": 4.5}
        self.call(srv.set_effect_params, "track", 0, "eq",
                  {"peak1_freq_hz": 900})
        assert child.get("value") == "900" and child.get("id") == "1234"
        assert c.get("Peak1freq") == "900"

    def test_effects_without_a_table(self, proj):
        from lmms_mcp import server as srv
        srv.add_effect("track", 0, "compressor")
        result = self.call(srv.set_effect_params, "track", 0, "compressor",
                           {"threshold": -10})
        assert "no settable parameters" in result["error"]
        # an effect from a loaded project that this server knows nothing of
        fx = proj.root.find(".//instrumenttrack/fxchain")
        ET.SubElement(fx, "effect", {"name": "ladspaeffect", "on": "1",
                                     "wet": "1"}).append(ET.Element("ladspacontrols"))
        entry = self.call(srv.get_effect_chain, "track", 0)["effects"][-1]
        assert entry["name"] == "ladspaeffect" and "params" not in entry
        assert "error" in self.call(srv.set_effect_params, "track", 0,
                                    "ladspaeffect", {"x": 1})


def xml_parser_channels(proj):
    from lmms_mcp import xml_parser
    return xml_parser.find_mixer_channels(proj.root)


class TestEffectDocs:
    """Tool descriptions and the effects resource name the parameters."""

    def test_add_effect_lists_the_starting_settings(self):
        from lmms_mcp import server as srv
        doc = srv.add_effect.__doc__
        assert "Starting settings:" in doc
        assert "delay: wet 0.4, feedback 0.3" in doc
        assert "1/8 note" in doc and "reverbsc: wet 0.3" in doc
        assert "set_effect_params" in doc and "params:" in doc

    def test_set_effect_params_lists_every_effects_parameters(self):
        from lmms_mcp import effect_params as ep
        from lmms_mcp import server as srv
        doc = srv.set_effect_params.__doc__
        assert "$" not in doc
        for effect, params in ep.EFFECT_PARAMS.items():
            assert f"- {effect}:" in doc, effect
            for p in params:
                if effect != "eq":  # eq is described by band
                    assert p.name in doc, (effect, p.name)
        assert "time_s (0.01..5 s, 0.5)" in doc
        assert "rc_lowpass_24" in doc  # the dualfilter types
        assert "lfo_period_s is the period" in doc
        for token in ("<band>_on", "peak1, peak2", "hp_slope_db", "lowshelf",
                      "highshelf"):
            assert token in doc

    def test_docs_only_name_the_effects_this_lmms_has(self, server_for):
        from lmms_mcp.effects import KNOWN_EFFECTS
        installed = {"tripleoscillator", "delay", "reverbsc", "eq",
                     "dynamicsprocessor"}
        srv = server_for(installed)
        for fn in (srv.add_effect, srv.set_effect_params):
            doc = fn.__doc__
            assert "- reverbsc:" in doc and "- eq:" in doc
            for name in set(KNOWN_EFFECTS) - installed:
                assert f"- {name}:" not in doc, (fn.__name__, name)
                assert f"{name}," not in doc, (fn.__name__, name)
        assert "bitcrush" not in srv.resource_effects()

    def test_effects_without_a_table_are_listed_as_such(self, server_for):
        srv = server_for({"tripleoscillator", "delay", "compressor"}, "1.3.0")
        assert "compressor: no parameters can be set yet" in \
            srv.set_effect_params.__doc__

    def test_resource_has_parameters_defaults_and_attributes(self):
        from lmms_mcp import server as srv
        data = json.loads(srv.resource_effects())
        by = {e["name"]: e for e in data["effects"]}
        delay = by["delay"]
        assert delay["wet_default"] == 0.4
        assert delay["decay_ms_default"] == 8000
        assert {p["name"] for p in data["settings_of_every_effect"]} == {
            "wet", "decay_ms"}
        assert delay["outputs_dry_signal"] is False
        assert "feedback 0.3" in delay["starting_settings"]
        params = {p["name"]: p for p in delay["parameters"]}
        assert params["feedback"]["xml_attribute"] == "FeebackAmount"
        assert (params["time_s"]["min"], params["time_s"]["max"],
                params["time_s"]["default"], params["time_s"]["unit"]) == (
                    0.01, 5, 0.5, "s")
        assert by["bassbooster"]["outputs_dry_signal"] is True
        assert "Neutral" in by["eq"]["note"]
        assert "soft_clip" in by["waveshaper"]["shapes"]
        assert by["compressor"]["parameters"].startswith("none")

    def test_instructions_point_to_set_effect_params(self):
        from lmms_mcp import server as srv
        assert "set_effect_params" in srv.mcp.instructions

    def test_tool_is_registered(self):
        import asyncio
        from lmms_mcp import server as srv
        tools = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}
        assert "set_effect_params" in tools
        assert "time_s" in tools["set_effect_params"].description


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
        # A project that plays a clean sine at -10.5 dBFS.
        fake = FakeLmms(sine(220, 0.3, 4))
        monkeypatch.setattr("subprocess.run", fake)
        return srv, fake.commands

    def test_defaults_are_1x_sincfastest(self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        out = tmp_path / "out.wav"
        result = json.loads(srv.render_project(output_path=str(out)))
        assert result["success"] is True
        # A wav is made from one 32-bit float render (-a) into a temporary
        # file, which is not the output.
        assert len(calls) == 1
        cmd = calls[0]
        assert cmd[cmd.index("-x") + 1] == "1"
        assert cmd[cmd.index("-i") + 1] == "sincfastest"
        assert "-x 1 -i sincfastest" in result["command"]
        assert result["oversampling"] == 1
        assert result["interpolation"] == "sincfastest"
        # Existing arguments are unchanged.
        assert cmd[cmd.index("render") - 2] == "--config"
        assert cmd[cmd.index("render") + 1] == str(tmp_path / "song.mmp")
        assert cmd[cmd.index("-o") + 1] != str(out)
        assert "-a" in cmd
        assert cmd[cmd.index("-f") + 1] == "wav"
        assert cmd[cmd.index("-s") + 1] == "44100"
        assert cmd[cmd.index("-b") + 1] == "160"
        assert result["output_file"] == str(out)
        assert read_pcm(out)[2] == 16

    def test_other_formats_render_the_output_with_lmms_after_the_float_one(
            self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        out = tmp_path / "out.ogg"
        result = json.loads(srv.render_project(
            output_path=str(out), file_format="ogg", bitrate=192))
        assert result["success"] is True
        assert len(calls) == 2
        first, second = calls
        assert "-a" in first and first[first.index("-f") + 1] == "wav"
        assert first[first.index("-o") + 1] != str(out)
        assert "-a" not in second
        assert second[second.index("-o") + 1] == str(out)
        assert second[second.index("-f") + 1] == "ogg"
        assert second[second.index("-b") + 1] == "192"
        assert second[second.index("-x") + 1] == "1"
        assert result["command"] == " ".join(second)
        assert out.read_bytes() == b"ENCODED:ogg"

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


class TestRenderProjectIsolatedConfig:
    """render_project runs LMMS with a private config, not ~/.lmmsrc.xml."""

    USER_CONFIG = (
        '<?xml version="1.0"?>\n<!DOCTYPE lmms-config-file>\n'
        '<lmms version="1.2.2">'
        '<audioalsa device="hw:1" channels="2"/>'
        '<paths stkdir="/stk/" gigdir="/home/u/lmms/gig/" '
        'laddir="/home/u/lmms/ladspa/" artwork="data:/themes/default/" '
        'sf2dir="/home/u/lmms/sf2/" defaultsf2="/home/u/lmms/sf2/a.sf2" '
        'backgroundartwork="" vstdir="/home/u/lmms/vst/" '
        'workingdir="/home/u/lmms/"/>'
        '<ui disableautoquit="0" saveinterval="2"/>'
        '</lmms>\n')

    @staticmethod
    def _setup(monkeypatch, tmp_path, user_config=None, runner=None):
        """A fake LMMS whose --config file is read while the run is in
        progress (it is deleted afterwards). Returns (srv, runs, home) with
        runs a list of {"cmd", "config", "text", "dir"} per LMMS call."""
        import subprocess
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv

        home = tmp_path / "home"
        home.mkdir()
        if user_config is not None:
            (home / ".lmmsrc.xml").write_text(user_config)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        proj.save(str(tmp_path / "song.mmp"))
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        runs = []

        fake = FakeLmms(sine(220, 0.3, 4))

        def fake_run(cmd, **kwargs):
            config = Path(cmd[cmd.index("--config") + 1])
            runs.append({
                "cmd": cmd, "config": config, "dir": config.parent,
                "text": config.read_text(encoding="utf-8"),
            })
            if runner is not None:
                runner(cmd)
            return fake(cmd, **kwargs)

        monkeypatch.setattr("subprocess.run", fake_run)
        return srv, runs, home

    @staticmethod
    def _parsed(text):
        from xml.etree import ElementTree as ET
        return ET.fromstring(text.split("\n", 2)[2])

    def test_runs_lmms_with_a_config_that_is_not_the_users(
            self, monkeypatch, tmp_path):
        srv, runs, home = self._setup(
            monkeypatch, tmp_path, self.USER_CONFIG)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "out.wav")))
        assert result["success"] is True
        assert result["isolated_config"] is True
        cmd = runs[0]["cmd"]
        # A global option: it comes before the "render" action.
        assert cmd.index("--config") < cmd.index("render")
        assert runs[0]["config"] != home / ".lmmsrc.xml"
        assert str(home) not in " ".join(cmd)
        assert "--config" in result["command"]

    def test_copies_only_the_render_paths_from_the_users_config(
            self, monkeypatch, tmp_path):
        srv, runs, _ = self._setup(monkeypatch, tmp_path, self.USER_CONFIG)
        srv.render_project(output_path=str(tmp_path / "out.wav"))
        root = self._parsed(runs[0]["text"])
        assert root.find("paths").attrib == {
            "stkdir": "/stk/", "gigdir": "/home/u/lmms/gig/",
            "laddir": "/home/u/lmms/ladspa/", "sf2dir": "/home/u/lmms/sf2/",
            "defaultsf2": "/home/u/lmms/sf2/a.sf2",
            "vstdir": "/home/u/lmms/vst/", "workingdir": "/home/u/lmms/"}
        # Nothing else of the user's config is carried over.
        assert [c.tag for c in root] == ["app", "paths", "ui"]
        assert root.find("ui").attrib == {"disableautoquit": "1"}
        assert "hw:1" not in runs[0]["text"]
        assert "saveinterval" not in runs[0]["text"]

    def test_effects_keep_running_whatever_the_users_gui_setting(
            self, monkeypatch, tmp_path):
        # The user's config says disableautoquit="0"; renders always use 1
        # so echoes and reverb tails survive silent gaps.
        srv, runs, _ = self._setup(monkeypatch, tmp_path, self.USER_CONFIG)
        srv.render_project(output_path=str(tmp_path / "out.wav"))
        root = self._parsed(runs[0]["text"])
        assert root.find("ui").get("disableautoquit") == "1"
        assert root.find("app").get("nanhandler") == "1"
        assert root.get("version") == "1.3.0"

    def test_works_without_a_user_config(self, monkeypatch, tmp_path):
        srv, runs, home = self._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "out.wav")))
        assert result["success"] is True
        root = self._parsed(runs[0]["text"])
        assert root.find("paths") is None
        assert root.find("ui").get("disableautoquit") == "1"
        assert not (home / ".lmmsrc.xml").exists()

    def test_users_config_is_never_written(self, monkeypatch, tmp_path):
        srv, runs, home = self._setup(monkeypatch, tmp_path, self.USER_CONFIG)
        import os
        user_rc = home / ".lmmsrc.xml"
        os.utime(user_rc, ns=(10**18, 10**18))
        before = user_rc.stat().st_mtime_ns
        for _ in range(2):
            srv.render_project(output_path=str(tmp_path / "out.wav"))
        assert user_rc.read_text() == self.USER_CONFIG
        assert user_rc.stat().st_mtime_ns == before
        assert sorted(p.name for p in home.iterdir()) == [".lmmsrc.xml"]

    def test_every_render_gets_its_own_config_and_cleans_it_up(
            self, monkeypatch, tmp_path):
        srv, runs, _ = self._setup(monkeypatch, tmp_path, self.USER_CONFIG)
        for _ in range(2):
            srv.render_project(output_path=str(tmp_path / "out.wav"))
        assert len(runs) == 2
        assert runs[0]["config"] != runs[1]["config"]
        assert runs[0]["dir"] != runs[1]["dir"]
        for run in runs:
            assert not run["dir"].exists()

    def test_overlapping_renders_wait_for_each_other_and_do_not_share_a_config(
            self, monkeypatch, tmp_path):
        import threading
        first_running = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        order = []

        def block_the_first(cmd):
            # The first render is still running while a second one is asked
            # for in another thread.
            order.append(("start", cmd[cmd.index("-o") + 1]))
            if len(order) == 1:
                first_running.set()
                assert release_first.wait(10)
            order.append(("end", cmd[cmd.index("-o") + 1]))

        srv, runs, _ = self._setup(
            monkeypatch, tmp_path, self.USER_CONFIG, runner=block_the_first)
        results = {}

        def render(name):
            if name == "second":
                second_started.set()
            results[name] = json.loads(srv.render_project(
                output_path=str(tmp_path / f"{name}.wav")))

        first = threading.Thread(target=render, args=("first",))
        first.start()
        assert first_running.wait(10)
        second = threading.Thread(target=render, args=("second",))
        second.start()
        assert second_started.wait(10)
        # Give the second render time to (wrongly) start next to the first.
        second.join(0.3)
        assert len(order) == 1
        release_first.set()
        first.join(10)
        second.join(10)
        assert results["first"]["success"] and results["second"]["success"]
        assert [kind for kind, _ in order] == ["start", "end", "start", "end"]
        assert len(runs) == 2
        assert runs[0]["dir"] != runs[1]["dir"]
        assert not runs[0]["dir"].exists() and not runs[1]["dir"].exists()

    def test_config_is_removed_when_the_render_times_out(
            self, monkeypatch, tmp_path):
        import subprocess

        def hang(cmd):
            raise subprocess.TimeoutExpired(cmd, 300)

        srv, runs, _ = self._setup(
            monkeypatch, tmp_path, self.USER_CONFIG, runner=hang)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "out.wav")))
        assert "timed out" in result["error"]
        assert not runs[0]["dir"].exists()

    def test_config_is_removed_when_lmms_fails(self, monkeypatch, tmp_path):
        import subprocess
        srv, runs, _ = self._setup(monkeypatch, tmp_path, self.USER_CONFIG)

        def failing(cmd, **kwargs):
            runs.append({"dir": Path(cmd[cmd.index("--config") + 1]).parent})
            return subprocess.CompletedProcess(cmd, 1, "", "boom")

        monkeypatch.setattr("subprocess.run", failing)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "gone.wav")))
        assert result["success"] is False
        assert result["isolated_config"] is True
        assert "boom" in result["error"]
        assert not runs[-1]["dir"].exists()

    def test_render_config_xml_keeps_only_render_paths(self):
        from lmms_mcp import lmms_app
        text = lmms_app.render_config_xml(
            {"workingdir": "/w & <x>/", "artwork": "data:/t/",
             "laddir": 'a"b', "recent": "z"})
        root = self._parsed(text)
        assert root.get("version") is None
        assert root.find("paths").attrib == {
            "workingdir": "/w & <x>/", "laddir": 'a"b'}
        assert text.startswith('<?xml version="1.0"?>\n<!DOCTYPE lmms-config-file>')

    def test_isolated_config_reads_the_given_config_and_removes_its_folder(
            self, tmp_path):
        from lmms_mcp import lmms_app
        user_rc = tmp_path / "user.xml"
        user_rc.write_text(self.USER_CONFIG)
        before = user_rc.stat().st_mtime_ns
        with lmms_app.isolated_config("1.2.2", user_rc) as first:
            with lmms_app.isolated_config("1.2.2", user_rc) as second:
                assert first != second and first.parent != second.parent
                assert first.is_file() and second.is_file()
            assert not second.parent.exists() and first.is_file()
            root = self._parsed(first.read_text(encoding="utf-8"))
            assert root.get("version") == "1.2.2"
            assert root.find("paths").get("sf2dir") == "/home/u/lmms/sf2/"
        assert not first.parent.exists()
        assert user_rc.stat().st_mtime_ns == before
        assert user_rc.read_text() == self.USER_CONFIG

    def test_isolated_config_is_removed_after_an_error(self):
        from lmms_mcp import lmms_app
        with pytest.raises(RuntimeError):
            with lmms_app.isolated_config() as config:
                folder = config.parent
                raise RuntimeError("render failed")
        assert not folder.exists()

    def test_lmms_version_check_does_not_use_a_config(self, monkeypatch):
        # `lmms --version` exits before LMMS loads or saves a config file
        # (checked with LMMS 1.2.2), so it runs as is.
        import subprocess
        from lmms_mcp import lmms_app
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "LMMS 1.2.2\n", "")

        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        monkeypatch.setattr("subprocess.run", fake_run)
        assert lmms_app.get_lmms_version() == "1.2.2"
        assert seen == [["/fake/lmms", "--version"]]


if __name__ == "__main__":
    import sys
    print("Run with: pytest tests/")
    sys.exit(1)


def _running_as_root():
    import os
    return hasattr(os, "geteuid") and os.geteuid() == 0


class TestSaveProjectFolders:
    """save_project creates missing folders and turns every file system
    problem into a JSON error (an exception reaches the MCP client only as
    "Error executing tool save_project")."""

    @staticmethod
    def _project(saved_at=None):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        if saved_at is not None:
            proj.save(str(saved_at))
        return srv, proj

    def test_absolute_path_in_missing_nested_folders(self, tmp_path):
        srv, proj = self._project()
        target = tmp_path / "new" / "nested" / "deep" / "test.mmp"
        result = json.loads(srv.save_project(str(target)))
        assert "error" not in result
        assert result["path"] == str(target)
        assert result["created_dirs"] == [
            str(tmp_path / "new"),
            str(tmp_path / "new" / "nested"),
            str(tmp_path / "new" / "nested" / "deep"),
        ]
        assert target.read_bytes().startswith(b"<?xml")
        assert proj.path == str(target)
        assert load_project(target) is not None

    def test_only_the_missing_folders_are_listed(self, tmp_path):
        srv, _ = self._project()
        (tmp_path / "have").mkdir()
        result = json.loads(srv.save_project(
            str(tmp_path / "have" / "sub" / "x.mmpz")))
        assert result["created_dirs"] == [str(tmp_path / "have" / "sub")]

    def test_relative_path_with_subfolders_goes_under_the_default_dir(
            self, monkeypatch, tmp_path):
        srv, proj = self._project()
        default = tmp_path / "projects"
        default.mkdir()
        monkeypatch.setattr(srv, "DEFAULT_PROJECTS_DIR", str(default))
        result = json.loads(srv.save_project("songs/x/y.mmp"))
        target = default / "songs" / "x" / "y.mmp"
        assert result["path"] == str(target)
        assert result["created_dirs"] == [
            str(default / "songs"), str(default / "songs" / "x")]
        assert target.is_file()
        assert proj.path == str(target)

    def test_missing_default_dir_is_created_and_listed(
            self, monkeypatch, tmp_path):
        srv, _ = self._project()
        default = tmp_path / "not" / "yet"
        monkeypatch.setattr(srv, "DEFAULT_PROJECTS_DIR", str(default))
        result = json.loads(srv.save_project("song.mmpz"))
        assert (default / "song.mmpz").is_file()
        assert result["created_dirs"] == [
            str(tmp_path / "not"), str(default)]
        # No path at all: untitled.mmpz in the same folder.
        srv, _ = self._project()
        result = json.loads(srv.save_project())
        assert result["path"] == str(default / "untitled.mmpz")
        assert "created_dirs" not in result

    def test_existing_folder_has_no_created_dirs(self, tmp_path):
        srv, _ = self._project()
        result = json.loads(srv.save_project(str(tmp_path / "x.mmp")))
        assert result["message"] == f"Project saved to {tmp_path / 'x.mmp'}"
        assert "created_dirs" not in result
        assert "error" not in result

    def test_save_again_without_a_path_recreates_a_deleted_folder(
            self, tmp_path):
        import shutil
        srv, proj = self._project(saved_at=None)
        (tmp_path / "gone").mkdir()
        proj.save(str(tmp_path / "gone" / "song.mmp"))
        shutil.rmtree(tmp_path / "gone")
        result = json.loads(srv.save_project())
        assert result["created_dirs"] == [str(tmp_path / "gone")]
        # The format follows the project's own .mmp extension.
        assert result["compressed"] is False
        assert (tmp_path / "gone" / "song.mmp").read_bytes().startswith(
            b"<?xml")

    @pytest.mark.skipif(_running_as_root(), reason="root ignores permissions")
    def test_read_only_folder_gives_a_json_error(self, tmp_path):
        srv, proj = self._project(saved_at=tmp_path / "first.mmp")
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            target = locked / "x.mmp"
            result = json.loads(srv.save_project(str(target)))
        finally:
            locked.chmod(0o700)
        assert result["path"] == str(target)
        assert result["error"].startswith(
            f"Could not save the project to {target}: ")
        assert "Permission denied" in result["error"]
        assert "created_dirs" not in result
        assert not target.exists()
        assert proj.path == str(tmp_path / "first.mmp")

    @pytest.mark.skipif(_running_as_root(), reason="root ignores permissions")
    def test_cannot_create_a_folder_in_a_read_only_folder(self, tmp_path):
        srv, proj = self._project()
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            result = json.loads(srv.save_project(
                str(locked / "a" / "b" / "x.mmp")))
        finally:
            locked.chmod(0o700)
        assert "Permission denied" in result["error"]
        assert f"could not create the folder {locked / 'a'}" in result["error"]
        assert list(locked.iterdir()) == []
        assert not proj.path

    def test_path_that_is_an_existing_folder(self, tmp_path):
        srv, proj = self._project(saved_at=tmp_path / "first.mmp")
        folder = tmp_path / "songs.mmp"
        folder.mkdir()
        result = json.loads(srv.save_project(str(folder)))
        assert result["path"] == str(folder)
        assert "existing folder" in result["error"]
        assert str(folder / "song.mmpz") in result["error"]
        assert folder.is_dir() and list(folder.iterdir()) == []
        assert proj.path == str(tmp_path / "first.mmp")

    def test_a_file_in_the_folder_path_gives_a_json_error(self, tmp_path):
        srv, proj = self._project()
        (tmp_path / "plain.txt").write_text("x")
        target = tmp_path / "plain.txt" / "sub" / "x.mmp"
        result = json.loads(srv.save_project(str(target)))
        assert result["error"] == (
            f"Could not save the project to {target}: "
            f"{tmp_path / 'plain.txt'} is a file, not a folder")
        assert result["path"] == str(target)
        assert (tmp_path / "plain.txt").read_text() == "x"
        assert not proj.path

    def test_a_failed_write_removes_the_folders_it_created(
            self, monkeypatch, tmp_path):
        srv, proj = self._project()

        def disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("lmms_mcp.project.save_project", disk_full)
        target = tmp_path / "a" / "b" / "x.mmp"
        result = json.loads(srv.save_project(str(target)))
        assert "No space left on device" in result["error"]
        assert str(target) in result["error"]
        assert not (tmp_path / "a").exists()
        assert not proj.path
        assert proj.modified is True

    def test_invalid_path_gives_a_json_error(self, tmp_path):
        srv, proj = self._project()
        result = json.loads(srv.save_project(str(tmp_path / "bad\0name.mmp")))
        assert result["error"].startswith("Could not save the project to ")
        assert "null" in result["error"]
        assert not proj.path


class TestRenderProjectFolders:
    """render_project creates the folders of output_path and reports
    problems with the output path as JSON errors that name the path."""

    @staticmethod
    def _setup(monkeypatch, tmp_path):
        return TestRenderProjectQuality._setup(monkeypatch, tmp_path)

    def test_missing_output_folders_are_created_before_lmms_runs(
            self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        out = tmp_path / "out" / "sub" / "x.wav"
        seen = []
        fake = FakeLmms(sine(220, 0.3, 4),
                        on_run=lambda cmd: seen.append(out.parent.is_dir()))
        monkeypatch.setattr("subprocess.run", fake)
        result = json.loads(srv.render_project(output_path=str(out)))
        assert seen == [True]
        assert result["success"] is True
        assert result["created_dirs"] == [
            str(tmp_path / "out"), str(tmp_path / "out" / "sub")]
        assert out.is_file()

    def test_existing_output_folder_has_no_created_dirs(
            self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "x.wav")))
        assert result["success"] is True
        assert "created_dirs" not in result

    def test_output_path_that_is_an_existing_folder(
            self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        folder = tmp_path / "out.wav"
        folder.mkdir()
        result = json.loads(srv.render_project(output_path=str(folder)))
        assert "existing folder" in result["error"]
        assert str(folder / "song.wav") in result["error"]
        assert result["output_file"] == str(folder)
        assert calls == []

    @pytest.mark.skipif(_running_as_root(), reason="root ignores permissions")
    def test_cannot_create_the_output_folder(self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            out = locked / "sub" / "x.wav"
            result = json.loads(srv.render_project(output_path=str(out)))
        finally:
            locked.chmod(0o700)
        assert result["error"].startswith(
            f"Could not render to {out}: ")
        assert "Permission denied" in result["error"]
        assert f"could not create the folder {locked / 'sub'}" in result["error"]
        assert result["output_file"] == str(out)
        assert calls == []

    def test_a_file_in_the_output_folder_path(self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        (tmp_path / "plain.txt").write_text("x")
        out = tmp_path / "plain.txt" / "x.wav"
        result = json.loads(srv.render_project(output_path=str(out)))
        assert result["error"] == (
            f"Could not render to {out}: "
            f"{tmp_path / 'plain.txt'} is a file, not a folder")
        assert calls == []

    def test_lmms_cannot_write_its_render(self, monkeypatch, tmp_path):
        import subprocess
        srv, _ = self._setup(monkeypatch, tmp_path)
        out = tmp_path / "new" / "x.wav"

        def fake_run(cmd, **kwargs):
            # What LMMS 1.2.2 prints and returns when it cannot write (here
            # the temporary file of the level measurement).
            target = cmd[cmd.index("-o") + 1]
            return subprocess.CompletedProcess(
                cmd, 1,
                "Notice: could not set realtime priority.\n"
                "Loading project...\nDone\n",
                "VST sync support disabled in your configuration\n"
                f"Could not open file {target} for writing.\n"
                "Please make sure you have write permission to the file "
                "and the directory containing the file and try again!\n")

        monkeypatch.setattr("subprocess.run", fake_run)
        result = json.loads(srv.render_project(output_path=str(out)))
        assert result["success"] is False
        error = result["error"]
        assert error.startswith("Render failed: LMMS could not write the "
                                "output file ")
        assert " for writing." in error
        assert "VST sync" not in error
        # The folder made for the failed render is removed again.
        assert "created_dirs" not in result
        assert not (tmp_path / "new").exists()

    @pytest.mark.skipif(_running_as_root(), reason="root ignores permissions")
    def test_output_folder_without_write_permission_fails_before_rendering(
            self, monkeypatch, tmp_path):
        srv, calls = self._setup(monkeypatch, tmp_path)
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            out = locked / "x.wav"
            result = json.loads(srv.render_project(output_path=str(out)))
        finally:
            locked.chmod(0o700)
        assert result["error"] == (
            f"Could not render to {out}: "
            f"no write permission for the folder {locked}")
        assert result["output_file"] == str(out)
        assert calls == []

    def test_output_that_cannot_be_written_after_the_render(
            self, monkeypatch, tmp_path):
        import errno
        from lmms_mcp import audio_analysis
        srv, calls = self._setup(monkeypatch, tmp_path)
        out = tmp_path / "made" / "x.wav"

        def disk_full(path, *args, **kwargs):
            Path(path).write_bytes(b"RIFF half a file")
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(audio_analysis, "write_wav", disk_full)
        result = json.loads(srv.render_project(output_path=str(out)))
        assert result["success"] is False
        assert result["error"] == (
            f"Could not write {out}: No space left on device")
        assert not out.exists()
        assert not (tmp_path / "made").exists()

    def test_lmms_fails_without_a_message(self, monkeypatch, tmp_path):
        import subprocess
        srv, _ = self._setup(monkeypatch, tmp_path)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, "", "VST sync support disabled in your configuration\n")

        monkeypatch.setattr("subprocess.run", fake_run)
        out = tmp_path / "x.wav"
        result = json.loads(srv.render_project(output_path=str(out)))
        assert result["success"] is False
        assert result["error"].startswith(
            "Render failed: LMMS exited with code 0 without creating ")
        assert result["error"].endswith(" and printed no error message.")
        assert not out.exists()

    def test_other_lmms_errors_keep_their_text(self, monkeypatch, tmp_path):
        import subprocess
        srv, _ = self._setup(monkeypatch, tmp_path)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 1, "", "Could not load project\n")

        monkeypatch.setattr("subprocess.run", fake_run)
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "x.wav")))
        assert result["error"] == "Render failed: Could not load project"

    def test_project_folder_gone_gives_a_json_error(
            self, monkeypatch, tmp_path):
        import shutil
        srv, calls = self._setup(monkeypatch, tmp_path)
        proj = LMMSProject()
        proj.new()
        (tmp_path / "gone").mkdir()
        proj.save(str(tmp_path / "gone" / "song.mmp"))
        srv.set_project(proj)
        shutil.rmtree(tmp_path / "gone")
        result = json.loads(srv.render_project(
            output_path=str(tmp_path / "x.wav")))
        assert "before rendering" in result["error"]
        assert str(tmp_path / "gone" / "song.mmp") in result["error"]
        assert "save_project" in result["error"]
        assert calls == []


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
        assert [n.get("key") for n in notes] == ["34"]

    def test_add_note_by_name_invalid_reports_error(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_note_by_name(0, "H2"))
        assert "error" in res
        assert "Invalid note name 'H2'" in res["error"]
        assert "Bb2" in res["error"]
        assert self._note_elems(proj) == []

    def test_add_note_by_name_out_of_range(self):
        srv, proj = self._srv()
        for name in ("C#9", "G#9", "C-1"):
            res = json.loads(srv.add_note_by_name(0, name))
            assert "error" in res and "out of range" in res["error"], name
            assert "C0 to C9" in res["error"], name
        assert self._note_elems(proj) == []

    def test_add_note_validation(self):
        srv, proj = self._srv()
        for kwargs, fragment in [
            ({"key": 128}, "key 128 is out of range"),
            ({"key": 109}, "key 109 is out of range"),
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
        res = json.loads(srv.add_note(0, key=108, pos=0, length=12))
        assert "error" not in res and "message" in res
        assert [n.get("key") for n in self._note_elems(proj)] == ["108"]

    def test_batch_all_valid(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_notes_batch(0, [
            {"key": "Bb2", "pos": 0, "length": 24},
            {"key": "C#3", "pos": 24, "length": 24},
            {"key": 48, "pos": 48},
        ]))
        assert res["added"] == 3
        assert res["failed"] == 0
        assert res["errors"] == []
        assert len(res["notes"]) == 3
        assert "warning" not in res and "error" not in res
        assert [n.get("key") for n in self._note_elems(proj)] == ["34", "37", "48"]

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
        assert [n.get("key") for n in self._note_elems(proj)] == ["34"]

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
        res = json.loads(srv.note_name_to_key("Bb2"))
        assert res["key"] == 34 and res["midi_note"] == 46
        assert "error" in json.loads(srv.note_name_to_key("H2"))
        scale = json.loads(srv.generate_scale("Bb2", "major"))
        assert scale["notes"][0]["key"] == 34

    # --- LMMS key numbering (C4 = 48, A4 = 57 = 440 Hz) -------------------

    def test_note_names_become_lmms_keys(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_notes_batch(0, [
            {"key": "A3", "pos": 0}, {"key": "A4", "pos": 192},
            {"key": "C4", "pos": 384}, {"key": "A5", "pos": 576},
            {"key": "Cb4", "pos": 768}, {"key": "B#3", "pos": 960},
            {"key": "C0", "pos": 1152}, {"key": "C9", "pos": 1344},
        ]))
        assert res["failed"] == 0
        assert [n.get("key") for n in self._note_elems(proj)] == [
            "45", "57", "48", "69", "47", "48", "0", "108"]
        # each result says which key (and name) the note was written as
        assert "key=45 (A3)" in res["notes"][0]["message"]
        assert "key=57 (A4)" in res["notes"][1]["message"]
        assert "key=48 (C4)" in res["notes"][2]["message"]
        assert "key=69 (A5)" in res["notes"][3]["message"]

    def test_add_note_by_name_a4_is_key_57(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_note_by_name(0, "A4"))
        assert "error" not in res
        assert [n.get("key") for n in self._note_elems(proj)] == ["57"]

    def test_add_note_default_key_is_middle_c(self):
        srv, proj = self._srv()
        assert "error" not in json.loads(srv.add_note(0))
        assert [n.get("key") for n in self._note_elems(proj)] == ["48"]
        assert json.loads(srv.add_note_by_name(0))  # default name C4
        assert [n.get("key") for n in self._note_elems(proj)] == ["48", "48"]
        # a batch note without a key is middle C too
        assert json.loads(srv.add_notes_batch(0, [{"pos": 48}]))["added"] == 1
        assert [n.get("key") for n in self._note_elems(proj)][-1] == "48"

    def test_key_range_errors_name_the_range_and_the_midi_offset(self):
        srv, proj = self._srv()
        res = json.loads(srv.add_note(0, key=120))
        msg = res["error"]
        assert "Note NOT added" in msg
        assert "0-108" in msg and "C0 to C9" in msg
        assert "middle C (C4) = 48" in msg and "A4 = 57 = 440 Hz" in msg
        assert "MIDI note number is key + 12" in msg
        assert "use 108" in msg  # 120 as a MIDI note is key 108
        res = json.loads(srv.add_notes_batch(0, [{"key": 60}, {"key": 109}]))
        assert res["added"] == 1 and res["failed"] == 1
        assert "key 109 is out of range" in res["errors"][0]["error"]
        assert [n.get("key") for n in self._note_elems(proj)] == ["60"]

    def test_lowest_and_highest_keys_are_accepted(self):
        srv, proj = self._srv()
        for key in (0, 108):
            assert "error" not in json.loads(srv.add_note(0, key=key))
        assert [n.get("key") for n in self._note_elems(proj)] == ["0", "108"]

    def test_key_to_note_name_tool(self):
        from lmms_mcp import server as srv
        res = json.loads(srv.key_to_note_name(48))
        assert res == {"key": 48, "note": "C4", "midi_note": 60}
        res = json.loads(srv.key_to_note_name(57))
        assert res["note"] == "A4" and res["midi_note"] == 69
        assert json.loads(srv.key_to_note_name(0))["note"] == "C0"
        assert json.loads(srv.key_to_note_name(108))["note"] == "C9"
        for bad in (-1, 109, 127):
            res = json.loads(srv.key_to_note_name(bad))
            assert "0-108" in res["error"] and "C0 to C9" in res["error"]

    def test_key_and_name_tools_round_trip_the_whole_range(self):
        from lmms_mcp import server as srv
        for key in range(0, 109):
            name = json.loads(srv.key_to_note_name(key))["note"]
            back = json.loads(srv.note_name_to_key(name))
            assert back["key"] == key
            assert back["midi_note"] == key + 12

    def test_note_name_to_key_tool_range_errors(self):
        from lmms_mcp import server as srv
        for name in ("C#9", "C-1", "C10"):
            res = json.loads(srv.note_name_to_key(name))
            assert "out of range" in res["error"], name

    def test_generate_scale_uses_lmms_keys_and_stops_at_c9(self):
        from lmms_mcp import server as srv
        scale = json.loads(srv.generate_scale("C4", "major"))
        assert [n["key"] for n in scale["notes"]] == [48, 50, 52, 53, 55, 57, 59]
        assert [n["note"] for n in scale["notes"]] == [
            "C4", "D4", "E4", "F4", "G4", "A4", "B4"]
        top = json.loads(srv.generate_scale("A8", "major", num_octaves=2))
        assert max(n["key"] for n in top["notes"]) <= 108
        assert top["notes"][0]["key"] == 105
        assert top["count"] == len(top["notes"]) < 14
        assert "error" in json.loads(srv.generate_scale("C10"))

    def test_note_names_resource_is_the_lmms_mapping(self):
        from lmms_mcp import server as srv
        data = json.loads(srv.resource_note_names())
        assert data["total"] == 109 == len(data["mapping"])
        by_key = {m["key"]: m for m in data["mapping"]}
        assert by_key[0]["note"] == "C0" and by_key[108]["note"] == "C9"
        assert by_key[48]["note"] == "C4" and by_key[48]["midi_note"] == 60
        assert by_key[57]["note"] == "A4" and by_key[57]["midi_note"] == 69
        assert "MIDI note number is key + 12" in data["numbering"]

    def test_notes_read_back_with_lmms_names(self):
        srv, proj = self._srv()
        srv.add_note(0, key=57)
        track = json.loads(srv.get_track(0))
        names = [n["note_name"] for p in track["patterns"] for n in p["notes"]]
        assert names == ["A4"]


class TestKeyNumberingDocs:
    """Every doc, prompt and resource states the LMMS numbering and none
    still says MIDI numbers (60 = C4, 69 = A4, kick = 36)."""

    STALE = (
        "60=C4", "60 = C4", "69=A4", "69 = A4", "C-1", "G9",
        "outside 0-127", "int 0-127", "number (0-127)", "keys 0-127",
        "MIDI key", "midi_key", "A3 in this server", "Key 36", "Key 38",
        "one octave down",
    )
    STATEMENT = (
        "Keys are LMMS keys: middle C (C4) = 48, A4 = 57 = 440 Hz. "
        "A MIDI note number is key + 12."
    )

    def _texts(self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS)
        tools = {t.name: t.description or ""
                 for t in asyncio.run(srv.mcp.list_tools())}
        prompts = {
            "create_basic_song": srv.create_basic_song(),
            "add_drum_pattern": srv.add_drum_pattern(),
            "create_melody": srv.create_melody(),
        }
        return srv, tools, prompts

    def test_instructions_state_the_numbering_once(self, server_for):
        srv, _, _ = self._texts(server_for)
        text = " ".join(srv.mcp.instructions.split())
        assert self.STATEMENT in text
        assert text.count("A MIDI note number is key + 12") == 1
        assert "Valid keys are 0-108, i.e. C0 to C9." in text

    def test_note_tools_state_the_numbering(self, server_for):
        _, tools, _ = self._texts(server_for)
        for name in ("add_note", "add_note_by_name", "note_name_to_key",
                     "key_to_note_name", "generate_scale"):
            flat = " ".join(tools[name].split())
            assert "middle C (C4) = 48" in flat, name
            assert "A4 = 57 = 440 Hz" in flat, name
            assert "MIDI note number is key + 12" in flat, name
        flat = " ".join(tools["add_notes_batch"].split())
        assert "middle C = C4 = 48" in flat and "key + 12" in flat
        assert "0-108" in flat and "C0..C9" in flat

    def test_no_text_still_uses_midi_numbering(self, server_for):
        srv, tools, prompts = self._texts(server_for)
        everything = {"instructions": srv.mcp.instructions, **tools,
                      **prompts}
        for where, text in everything.items():
            for stale in self.STALE:
                assert stale not in text, (where, stale)

    def test_audiofile_sample_doc_calls_key_57_a4(self, server_for):
        _, tools, _ = self._texts(server_for)
        flat = " ".join(tools["set_audiofile_sample"].split())
        assert "key 57 (A4, the LMMS default)" in flat

    def test_sf2_drum_keys_are_lmms_keys(self, server_for):
        _, tools, _ = self._texts(server_for)
        flat = " ".join(tools["set_instrument_file"].split())
        assert "notes 35-81 are keys 23-69 here (MIDI note minus 12)" in flat
        assert ("key 24 kick, 26 snare, 30 closed hi-hat, 34 open hi-hat, "
                "37 crash") in flat

    def test_drum_prompt_lists_lmms_keys_with_midi_numbers(self, server_for):
        _, _, prompts = self._texts(server_for)
        text = prompts["add_drum_pattern"]
        for line in (
            "Key 24 = Kick Drum (MIDI 36)", "Key 26 = Snare (MIDI 38)",
            "Key 30 = Closed Hi-Hat (MIDI 42)",
            "Key 34 = Open Hi-Hat (MIDI 46)",
            "Key 37 = Crash Cymbal (MIDI 49)",
            "Key 39 = Ride Cymbal (MIDI 51)",
        ):
            assert line in text
        assert "a MIDI note number is key + 12" in " ".join(text.split())
        assert "key 57 (A4," in " ".join(text.split())

    def test_drum_prompt_names_the_kit_only_when_offered(self, server_for):
        srv = server_for(LMMS_122_PLUGINS - {"sf2player"})
        text = srv.add_drum_pattern()
        assert "sf2player" not in text
        assert "any General MIDI drum instrument" in text
        assert "Key 24 = Kick Drum" in text

    def test_readme_matches(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        flat = " ".join(readme.split())
        assert "middle C (C4) = 48, A4 = 57 = 440 Hz. A MIDI note number is key + 12." in flat
        assert "Note 60" not in readme and "Note 69" not in readme
        assert "key 24 is the kick (MIDI 36)" in flat


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
        for fn in (srv.add_instrument_track, srv.add_effect,
                   srv.set_effect_params):
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


# ──────────────────────────────────────────────────────────────────
# Instruments that play a file: sf2player, gigplayer, patman
# ──────────────────────────────────────────────────────────────────


def make_sf2(path, presets=((0, 0, "Piano"),)):
    """Write a minimal SoundFont whose preset list is `presets`.

    presets: (bank, patch, name) tuples. Only the RIFF skeleton and the
    phdr chunk exist; enough for reading the preset list.
    """
    import struct

    def chunk(cid, body):
        return cid + struct.pack("<I", len(body)) + body + (
            b"\0" if len(body) & 1 else b"")

    records = b"".join(
        struct.pack("<20sHHHIII", name.encode(), patch, bank, 0, 0, 0, 0)
        for bank, patch, name in presets
    ) + struct.pack("<20sHHHIII", b"EOP", 255, 255, 0, 0, 0, 0)
    body = (
        b"sfbk"
        + chunk(b"LIST", b"INFO" + chunk(b"ifil", struct.pack("<HH", 2, 1)))
        + chunk(b"LIST", b"sdta" + chunk(b"smpl", b"\0" * 8))
        + chunk(b"LIST", b"pdta" + chunk(b"phdr", records))
    )
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return Path(path)


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class TestInstrumentFileDiscovery:
    """Finding soundfonts, GIG libraries and GUS patches (fake folders)."""

    @staticmethod
    def find(tmp_path, config=None, extra=(), system=None, samples=""):
        """Search only the given folders ("" = no LMMS factory samples)."""
        from lmms_mcp import lmms_app
        return lmms_app.find_instrument_files(
            config_paths=config or {}, extra_dirs=list(extra),
            system_dirs=system or {}, home=tmp_path / "home",
            samples_dir=samples)

    def test_finds_each_kind_in_the_lmms_folders(self, tmp_path):
        sf2 = touch(tmp_path / "sf" / "a.sf2")
        gig = touch(tmp_path / "gig" / "b.gig")
        pat = touch(tmp_path / "home/lmms/samples/patches/c.pat")
        touch(tmp_path / "sf" / "notes.txt")
        touch(tmp_path / "sf" / "wrong.gig")  # a gig in the soundfont folder
        found = self.find(tmp_path, config={
            "sf2dir": str(tmp_path / "sf"), "gigdir": str(tmp_path / "gig")})
        assert found.files["sf2player"] == (str(sf2),)
        assert found.files["gigplayer"] == (str(gig),)
        assert found.files["patman"] == (str(pat),)

    def test_extension_case_is_ignored_and_sf3_is_not_a_soundfont(
            self, tmp_path):
        loud = touch(tmp_path / "sf" / "LOUD.SF2")
        touch(tmp_path / "sf" / "compressed.sf3")
        found = self.find(tmp_path, config={"sf2dir": str(tmp_path / "sf")})
        assert found.files["sf2player"] == (str(loud),)

    def test_default_folders_follow_the_working_directory(self, tmp_path):
        work = tmp_path / "work"
        sf2 = touch(work / "samples/soundfonts/x.sf2")
        gig = touch(work / "samples/gig/y.gig")
        found = self.find(tmp_path, config={"workingdir": str(work)})
        assert found.files["sf2player"] == (str(sf2),)
        assert found.files["gigplayer"] == (str(gig),)
        assert found.put_in["gigplayer"] == str(work / "samples/gig")
        assert found.put_in["sf2player"] == str(work / "samples/soundfonts")

    def test_without_config_the_folders_are_under_home(self, tmp_path):
        found = self.find(tmp_path)
        home = tmp_path / "home"
        assert found.put_in["gigplayer"] == str(home / "lmms/samples/gig")
        assert str(home / ".local/share/soundfonts") in \
            found.searched["sf2player"]

    def test_system_folders_and_factory_patches_are_searched(self, tmp_path):
        sf2 = touch(tmp_path / "usr" / "GM.sf2")
        pat = touch(tmp_path / "factory" / "patches" / "flute.pat")
        free = touch(tmp_path / "freepats" / "Tone_000" / "piano.pat")
        found = self.find(
            tmp_path,
            system={"sf2player": [tmp_path / "usr"],
                    "patman": [tmp_path / "freepats"]},
            samples=tmp_path / "factory")
        assert found.files["sf2player"] == (str(sf2),)
        assert found.files["patman"] == (str(pat), str(free))

    def test_extra_folders_come_first_and_apply_to_every_kind(self, tmp_path):
        mine = tmp_path / "mine"
        sf_mine = touch(mine / "mine.sf2")
        gig_mine = touch(mine / "mine.gig")
        pat_mine = touch(mine / "mine.pat")
        sf_lmms = touch(tmp_path / "sf" / "lmms.sf2")
        found = self.find(tmp_path, extra=[mine],
                          config={"sf2dir": str(tmp_path / "sf")})
        assert found.files["sf2player"] == (str(sf_mine), str(sf_lmms))
        assert found.files["gigplayer"] == (str(gig_mine),)
        assert found.files["patman"] == (str(pat_mine),)
        assert found.default_soundfont == str(sf_mine)

    def test_extra_folders_come_from_the_environment(
            self, tmp_path, monkeypatch):
        import os

        from lmms_mcp import lmms_app
        one, two = tmp_path / "one", tmp_path / "two"
        a, b = touch(one / "a.sf2"), touch(two / "b.sf2")
        monkeypatch.setenv(
            lmms_app.INSTRUMENT_FILE_DIRS_ENV, f"{one}{os.pathsep}{two}")
        found = lmms_app.find_instrument_files(
            config_paths={}, system_dirs={}, home=tmp_path / "home",
            samples_dir="")
        assert found.files["sf2player"] == (str(a), str(b))

    def test_search_is_bounded_in_depth(self, tmp_path):
        top = touch(tmp_path / "sf" / "top.sf2")
        level2 = touch(tmp_path / "sf" / "a" / "b" / "deep.sf2")
        touch(tmp_path / "sf" / "a" / "b" / "c" / "too_deep.sf2")
        found = self.find(tmp_path, config={"sf2dir": str(tmp_path / "sf")})
        assert found.files["sf2player"] == (str(top), str(level2))

    def test_search_is_bounded_in_size(self, tmp_path, monkeypatch):
        from lmms_mcp import lmms_app
        monkeypatch.setattr(lmms_app, "_SCAN_LIMIT", 3)
        for i in range(10):
            touch(tmp_path / "sf" / f"f{i}.sf2")
        found = self.find(tmp_path, config={"sf2dir": str(tmp_path / "sf")})
        assert len(found.files["sf2player"]) == 3

    def test_duplicates_through_links_are_listed_once(self, tmp_path):
        real = touch(tmp_path / "sf" / "TimGM.sf2")
        (tmp_path / "sf" / "default-GM.sf2").symlink_to(real)
        (tmp_path / "sf" / "dangling.sf2").symlink_to(tmp_path / "gone.sf2")
        found = self.find(tmp_path, config={"sf2dir": str(tmp_path / "sf")})
        # names sort case-insensitively: the link comes first and wins
        assert found.files["sf2player"] == (str(tmp_path / "sf/default-GM.sf2"),)

    def test_nothing_found(self, tmp_path):
        found = self.find(tmp_path)
        assert all(not v for v in found.files.values())
        assert found.default_soundfont is None
        assert found.for_instrument("gigplayer") == ()
        assert found.for_instrument("tripleoscillator") == ()

    def test_searched_folders_are_reported_even_if_missing(self, tmp_path):
        found = self.find(
            tmp_path, config={"gigdir": str(tmp_path / "nope")},
            extra=[tmp_path / "extra"])
        assert found.searched["gigplayer"] == (
            str(tmp_path / "extra"), str(tmp_path / "nope"))

    # --- the default soundfont ---------------------------------------------

    def test_lmms_default_soundfont_wins(self, tmp_path):
        touch(tmp_path / "sf" / "a.sf2")
        chosen = touch(tmp_path / "elsewhere" / "mine.sf2")
        found = self.find(tmp_path, config={
            "sf2dir": str(tmp_path / "sf"), "defaultsf2": str(chosen)})
        assert found.default_soundfont == str(chosen)
        assert str(chosen) in found.files["sf2player"]

    def test_relative_default_soundfont_is_in_the_soundfont_folder(
            self, tmp_path):
        touch(tmp_path / "sf" / "a.sf2")
        chosen = touch(tmp_path / "sf" / "b.sf2")
        found = self.find(tmp_path, config={
            "sf2dir": str(tmp_path / "sf"), "defaultsf2": "b.sf2"})
        assert found.default_soundfont == str(chosen)

    @pytest.mark.parametrize("default", ["", "gone.sf2", "notes.txt"])
    def test_unusable_default_soundfont_is_ignored(self, tmp_path, default):
        first = touch(tmp_path / "sf" / "a.sf2")
        touch(tmp_path / "sf" / "notes.txt")
        found = self.find(tmp_path, config={
            "sf2dir": str(tmp_path / "sf"), "defaultsf2": default})
        assert found.default_soundfont == str(first)

    def test_default_soundfont_prefers_the_gm_default_name(self, tmp_path):
        touch(tmp_path / "usr" / "TimGM6mb.sf2")
        gm = touch(tmp_path / "usr" / "default-GM.sf2")
        found = self.find(tmp_path, system={"sf2player": [tmp_path / "usr"]})
        assert found.default_soundfont == str(gm)

    # --- LMMS's config file ---------------------------------------------------

    def test_read_lmms_paths(self, tmp_path):
        from lmms_mcp import lmms_app
        rc = tmp_path / ".lmmsrc.xml"
        rc.write_text(
            '<lmms><paths sf2dir="/x/sf/" gigdir="/x/gig/" defaultsf2=""/>'
            '<ui/></lmms>')
        assert lmms_app.read_lmms_paths(rc) == {
            "sf2dir": "/x/sf/", "gigdir": "/x/gig/", "defaultsf2": ""}

    def test_read_lmms_paths_tolerates_a_missing_or_broken_file(
            self, tmp_path):
        from lmms_mcp import lmms_app
        assert lmms_app.read_lmms_paths(tmp_path / "missing.xml") == {}
        broken = tmp_path / "broken.xml"
        broken.write_text("<lmms><paths")
        assert lmms_app.read_lmms_paths(broken) == {}
        no_paths = tmp_path / "none.xml"
        no_paths.write_text("<lmms><ui/></lmms>")
        assert lmms_app.read_lmms_paths(no_paths) == {}


class TestSoundfontPresets:
    def test_reads_banks_patches_and_names(self, tmp_path):
        from lmms_mcp import lmms_app
        sf2 = make_sf2(tmp_path / "t.sf2", [
            (0, 0, "Piano"), (0, 48, "Strings"), (128, 0, "Standard")])
        assert lmms_app.read_soundfont_presets(sf2) == {
            (0, 0): "Piano", (0, 48): "Strings", (128, 0): "Standard"}

    @pytest.mark.parametrize("content", [b"", b"RIFF", b"not a soundfont" * 4])
    def test_unreadable_files_give_none(self, tmp_path, content):
        from lmms_mcp import lmms_app
        bad = tmp_path / "bad.sf2"
        bad.write_bytes(content)
        assert lmms_app.read_soundfont_presets(bad) is None
        assert lmms_app.read_soundfont_presets(tmp_path / "gone.sf2") is None

    def test_default_preset_is_piano_or_the_first_preset(self, tmp_path):
        from lmms_mcp import lmms_app
        gm = make_sf2(tmp_path / "gm.sf2", [(0, 5, "X"), (0, 0, "Piano")])
        drums = make_sf2(tmp_path / "drums.sf2", [(128, 8, "Room"), (1, 3, "Y")])
        assert lmms_app.default_soundfont_preset(gm) == (0, 0)
        assert lmms_app.default_soundfont_preset(drums) == (1, 3)
        assert lmms_app.default_soundfont_preset(tmp_path / "gone") == (0, 0)

    def test_describe_banks(self):
        from lmms_mcp import lmms_app
        presets = {(0, p): "x" for p in (0, 1, 2, 5)}
        presets.update({(128, 0): "d", (128, 8): "d"})
        assert lmms_app.describe_soundfont_banks(presets) == (
            "bank 0: patches 0-2, 5; bank 128: patches 0, 8")


def _availability_with(files, installed=LMMS_122_PLUGINS, **kwargs):
    from lmms_mcp import lmms_app
    from lmms_mcp import server as srv
    return lmms_app.build_availability(
        srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
        set(installed), "1.2.2", files=files, **kwargs)


def _files(**found):
    """InstrumentFiles from {instrument: [paths]}."""
    from lmms_mcp import lmms_app
    sf = found.get("sf2player", [])
    return lmms_app.InstrumentFiles(
        files={k: tuple(str(p) for p in v) for k, v in found.items()},
        searched={k: ("/search/" + k, "/other/" + k)
                  for k in lmms_app.FILE_INSTRUMENTS},
        put_in={k: "/put/" + k for k in lmms_app.FILE_INSTRUMENTS},
        default_soundfont=str(sf[0]) if sf else None,
    )


class TestFileInstrumentOffering:
    """sf2player, gigplayer and patman are only offered with a file."""

    @pytest.fixture
    def setup(self, monkeypatch, tmp_path):
        """(server, project, install) builder for a given set of files."""
        from lmms_mcp import server as srv

        def make(**found):
            files = _files(**{k: [touch(tmp_path / v) for v in vs]
                              if vs and isinstance(vs[0], str) else vs
                              for k, vs in found.items()})
            monkeypatch.setattr(srv, "_availability", _availability_with(files))
            proj = LMMSProject()
            proj.new()
            srv.set_project(proj)
            return srv, proj
        return make

    def test_snapshot_offers_an_instrument_only_with_a_file(self):
        av = _availability_with(_files(sf2player=["/a.sf2"]))
        assert "sf2player" in av.instruments
        assert "gigplayer" not in av.instruments
        assert "patman" not in av.instruments
        assert av.needs_file == {"gigplayer", "patman"}
        assert "gigplayer" not in av.plugin_names
        assert av.files.default_soundfont == "/a.sf2"

    def test_snapshot_without_any_files_offers_none(self):
        av = _availability_with(None)
        assert not {"sf2player", "gigplayer", "patman"} & av.instruments
        assert av.needs_file == {"sf2player", "gigplayer", "patman"}
        assert "tripleoscillator" in av.instruments

    def test_a_missing_library_is_not_a_missing_file(self):
        av = _availability_with(
            _files(gigplayer=["/a.gig"]),
            installed=LMMS_122_PLUGINS - {"gigplayer", "patman"})
        assert not {"gigplayer", "patman"} & (av.instruments | av.needs_file)

    def test_files_also_gate_when_no_lmms_was_found(self):
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv
        av = lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS, set(), None,
            files=_files(patman=["/a.pat"]))
        assert "patman" in av.instruments
        assert "gigplayer" not in av.instruments
        assert "sf2player" not in av.instruments

    def test_instrument_lists_leave_out_file_instruments_without_files(
            self, setup):
        srv, _ = setup(sf2player=["a.sf2"])
        assert "gigplayer" not in srv._instrument_choices(srv._availability)
        doc = srv.render_plugin_doc(
            srv.add_instrument_track.__plugin_doc_template__,
            srv._availability)
        assert "sf2player" in doc and "gigplayer" not in doc \
            and "patman" not in doc
        info = json.loads(srv.list_available_plugins())
        assert "gigplayer" not in info["built_in_instruments"]
        entries = json.loads(srv.resource_instruments())["instruments"]
        assert "gigplayer" not in {e["name"] for e in entries}

    # --- refusals ----------------------------------------------------------

    def test_gigplayer_is_refused_without_a_gig_file(self, setup):
        srv, proj = setup(sf2player=["a.sf2"])
        before = proj.get_xml_string()
        result = json.loads(srv.add_instrument_track("G", "gigplayer"))
        assert result["error"] == (
            "'gigplayer' needs a .gig sample library and none was found "
            "(searched: /search/gigplayer, /other/gigplayer). Put one in "
            "/put/gigplayer and restart the server, or use 'sf2player' / "
            "'audiofileprocessor'.")
        assert result["alternative"] == "sf2player"
        assert "sf2player" in result["available_instruments"]
        assert "gigplayer" not in result["available_instruments"]
        assert proj.get_xml_string() == before

    def test_patman_is_refused_without_a_pat_file(self, setup):
        srv, proj = setup(sf2player=["a.sf2"])
        result = json.loads(srv.add_instrument_track("P", "patman"))
        assert "'patman' needs a .pat GUS patch and none was found" \
            in result["error"]
        assert "Put one in /put/patman and restart the server" \
            in result["error"]
        assert find_tracks(proj.root) == []

    def test_sf2player_is_refused_without_a_soundfont(self, setup):
        srv, proj = setup(gigplayer=["a.gig"])
        for spelling in ("sf2player", "sf2", "fluidsynth"):
            result = json.loads(srv.add_instrument_track("S", spelling))
            assert "'sf2player' needs a .sf2 soundfont and none was found" \
                in result["error"], spelling
            # gigplayer needs a file too, but this one has one, and the
            # suggested alternatives are only offered instruments
            assert result["alternative"] == "audiofileprocessor"
            assert "'audiofileprocessor'" in result["error"]
        assert find_tracks(proj.root) == []

    def test_refusal_without_any_alternative(self, setup, monkeypatch):
        srv, _ = setup(patman=["a.pat"])
        monkeypatch.setattr(srv, "_availability", _availability_with(
            _files(patman=["/a.pat"]),
            installed=LMMS_122_PLUGINS - {"audiofileprocessor"}))
        result = json.loads(srv.add_instrument_track("G", "gigplayer"))
        assert result["error"].endswith(
            "restart the server, or choose one of the available_instruments.")
        assert "alternative" not in result
        assert "patman" in result["available_instruments"]

    def test_alias_of_a_missing_library_is_still_the_plain_refusal(
            self, setup, monkeypatch):
        srv, _ = setup(sf2player=["a.sf2"])
        monkeypatch.setattr(srv, "_availability", _availability_with(
            _files(sf2player=["/a.sf2"]),
            installed=LMMS_122_PLUGINS - {"gigplayer"}))
        result = json.loads(srv.add_instrument_track("G", "gigplayer"))
        assert "'gigplayer' is not available in the installed LMMS" \
            in result["error"]
        assert "needs a" not in result["error"]

    def test_list_available_plugins_explains_missing_files(self, setup):
        srv, _ = setup(sf2player=["a.sf2"])
        missing = json.loads(
            srv.list_available_plugins())["unavailable_in_installed_lmms"]
        assert missing["gigplayer"].startswith("'gigplayer' needs a .gig")
        assert missing["patman"].endswith("restart the server.")
        assert "sf2player" not in missing

    # --- new tracks --------------------------------------------------------

    def test_new_sf2player_track_plays_the_default_soundfont(
            self, setup, tmp_path):
        sf2 = make_sf2(tmp_path / "gm.sf2", [(0, 0, "Grand Piano")])
        srv, proj = setup(sf2player=[sf2])
        result = json.loads(srv.add_instrument_track("Keys", "sf2"))
        assert "error" not in result and "warning" not in result
        assert result["instrument"] == "sf2player"
        assert result["instrument_file"] == {
            "file": str(sf2), "bank": 0, "patch": 0, "preset": "Grand Piano"}
        assert "gm.sf2 bank 0 patch 0 (Grand Piano)" in result["hint"]
        assert "set_instrument_file(track_index=0" in result["hint"]
        inst = find_tracks(proj.root)[0].find("instrumenttrack/instrument")
        assert inst.get("name") == "sf2player"
        assert inst.find("sf2player").attrib == {
            "src": str(sf2), "bank": "0", "patch": "0", "gain": "1"}
        assert proj._modified

    def test_default_soundfont_without_bank_zero_starts_on_its_first_preset(
            self, setup, tmp_path):
        sf2 = make_sf2(tmp_path / "drums.sf2", [(128, 0, "Kit")])
        srv, proj = setup(sf2player=[sf2])
        result = json.loads(srv.add_instrument_track("Drums", "sf2player"))
        assert result["instrument_file"]["bank"] == 128
        assert find_tracks(proj.root)[0].find(
            "instrumenttrack/instrument/sf2player").get("bank") == "128"

    def test_unreadable_default_soundfont_is_still_set(self, setup):
        srv, _ = setup(sf2player=["plain.sf2"])  # empty stand-in file
        result = json.loads(srv.add_instrument_track("K", "sf2player"))
        assert result["instrument_file"]["bank"] == 0
        assert "preset" not in result["instrument_file"]

    def test_vanished_default_soundfont_warns_and_says_what_to_do(
            self, setup, tmp_path):
        srv, proj = setup(sf2player=[tmp_path / "gone.sf2"])
        result = json.loads(srv.add_instrument_track("K", "sf2player"))
        assert "File not found" in result["warning"]
        assert "set_instrument_file(track_index=0" in result["warning"]
        assert "instrument_file" not in result
        assert len(find_tracks(proj.root)) == 1

    def test_new_gigplayer_and_patman_tracks_say_to_set_a_file(
            self, setup, tmp_path):
        gig, pat = touch(tmp_path / "x.gig"), touch(tmp_path / "x.pat")
        srv, proj = setup(gigplayer=[gig], patman=[pat])
        for name, file in (("gigplayer", gig), ("patman", pat)):
            result = json.loads(srv.add_instrument_track("T", name))
            assert "error" not in result
            assert f"{name} is silent until it has a" in result["hint"]
            assert f"file_path='{file}'" in result["hint"]
            assert "instrument_file" not in result
        for track in find_tracks(proj.root):
            settings = track.find("instrumenttrack/instrument")[0]
            assert "src" not in settings.attrib

    def test_other_instruments_get_no_file_hint(self, setup):
        srv, _ = setup(sf2player=["a.sf2"])
        result = json.loads(srv.add_instrument_track("T", "tripleoscillator"))
        assert "hint" not in result and "instrument_file" not in result

    # --- what the agent is told -------------------------------------------

    def test_instructions_mention_only_offered_file_instruments(self, setup):
        srv, _ = setup(sf2player=["a.sf2"])
        text = srv.build_instructions(srv._availability, "/tmp/p")
        assert "sf2player plays General MIDI" in text
        assert "set_instrument_file" in text
        assert "gigplayer" not in text and "patman" not in text

    def test_instructions_for_gig_and_patch_only(self, setup):
        srv, _ = setup(gigplayer=["a.gig"], patman=["a.pat"])
        text = srv.build_instructions(srv._availability, "/tmp/p")
        assert "gigplayer plays a .gig library and patman plays a GUS .pat" \
            in text
        assert "sf2player" not in text

    def test_instructions_without_file_instruments(self, setup):
        srv, _ = setup()
        text = srv.build_instructions(srv._availability, "/tmp/p")
        assert "File-based instruments" not in text
        assert "set_instrument_file" not in text

    def test_tool_description_follows_the_offered_instruments(self, setup):
        srv, _ = setup(sf2player=["a.sf2"])
        doc = srv.render_plugin_doc(
            srv.set_instrument_file.__plugin_doc_template__,
            srv._availability)
        assert "(sf2player)" in doc
        flat = " ".join(doc.split())
        assert "patch 0 piano, 24 nylon guitar, 32 acoustic bass" in flat
        assert "73 flute" in flat and "Bank 128 is the drum kit" in flat
        assert "gigplayer" not in doc and "patman" not in doc
        srv2, _ = setup(gigplayer=["a.gig"], patman=["a.pat"])
        doc = srv2.render_plugin_doc(
            srv2.set_instrument_file.__plugin_doc_template__,
            srv2._availability)
        assert "(gigplayer / patman)" in doc
        assert "sf2player" not in doc and "General MIDI" not in doc

    def test_add_instrument_track_points_to_the_file_tool(self, setup):
        srv, _ = setup(sf2player=["a.sf2"])
        doc = srv.render_plugin_doc(
            srv.add_instrument_track.__plugin_doc_template__,
            srv._availability)
        assert "set_instrument_file" in doc
        srv, _ = setup()
        doc = srv.render_plugin_doc(
            srv.add_instrument_track.__plugin_doc_template__,
            srv._availability)
        assert "set_instrument_file" not in doc

    def test_list_available_plugins_shows_the_files(self, setup, tmp_path):
        sfs = [tmp_path / f"f{i:02}.sf2" for i in range(25)]
        srv, _ = setup(sf2player=[touch(p) for p in sfs],
                       patman=[touch(tmp_path / "x.pat")])
        listed = json.loads(srv.list_available_plugins())["instrument_files"]
        assert set(listed) == {"sf2player", "patman"}
        assert listed["patman"] == {"count": 1, "files": [str(tmp_path / "x.pat")]}
        assert listed["sf2player"]["count"] == 25
        assert listed["sf2player"]["files"] == [str(p) for p in sfs[:20]]
        assert listed["sf2player"]["note"] == "first 20 of 25"
        assert listed["sf2player"]["default"] == str(sfs[0])

    def test_list_available_plugins_without_files_has_no_file_section(
            self, setup):
        srv, _ = setup()
        assert "instrument_files" not in json.loads(srv.list_available_plugins())

    # --- the tool is registered only when it can be used ---------------------

    @pytest.mark.parametrize("found,registered", [
        ({"sf2player": ("/a.sf2",)}, True),
        ({"gigplayer": ("/a.gig",)}, True),
        ({"patman": ("/a.pat",)}, True),
        ({}, False),
    ])
    def test_tool_is_registered_only_if_a_file_instrument_is_offered(
            self, server_for, found, registered):
        import asyncio
        files = _files(**{k: list(v) for k, v in found.items()})
        srv = server_for(LMMS_122_PLUGINS, files=files)
        tools = {t.name for t in asyncio.run(srv.mcp.list_tools())}
        assert ("set_instrument_file" in tools) is registered
        assert "add_instrument_track" in tools

    def test_tool_is_not_registered_when_the_libraries_are_missing(
            self, server_for, tmp_path):
        import asyncio
        srv = server_for(
            LMMS_122_PLUGINS - {"sf2player", "gigplayer", "patman"})
        tools = {t.name for t in asyncio.run(srv.mcp.list_tools())}
        assert "set_instrument_file" not in tools

    def test_registered_descriptions_match_the_machine(self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS, files=_files(sf2player=["/a.sf2"]))
        tools = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}
        text = srv.mcp.instructions + "\n".join(
            t.description or "" for t in tools.values())
        assert "gigplayer" not in text and "patman" not in text
        assert "sf2player" in tools["set_instrument_file"].description


class TestSetInstrumentFile:
    @pytest.fixture
    def env(self, monkeypatch, tmp_path):
        """(server, project, folder): the conftest install with real files."""
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        return srv, proj, tmp_path

    @staticmethod
    def settings(proj, index=0):
        inst = find_tracks(proj.root)[index].find("instrumenttrack/instrument")
        return inst.get("name"), inst[0]

    def test_sets_a_soundfont_with_bank_and_patch(self, env):
        srv, proj, tmp = env
        sf2 = make_sf2(tmp / "gm.sf2", [(0, 0, "Piano"), (0, 48, "Strings"),
                                        (128, 0, "Standard")])
        srv.add_instrument_track("Keys", "sf2player")
        result = json.loads(srv.set_instrument_file(0, str(sf2), patch=48))
        assert result == {
            "track_index": 0, "instrument": "sf2player", "file": str(sf2),
            "bank": 0, "patch": 48, "preset": "Strings",
            "message": "Set gm.sf2 bank 0 patch 48 (Strings) on sf2player "
                       "track 0"}
        name, node = self.settings(proj)
        assert name == "sf2player" and node.tag == "sf2player"
        assert node.attrib == {
            "src": str(sf2), "bank": "0", "patch": "48", "gain": "1"}

    def test_replaces_the_file_and_keeps_the_gain(self, env):
        srv, proj, tmp = env
        a, b = make_sf2(tmp / "a.sf2"), make_sf2(tmp / "b.sf2")
        srv.add_instrument_track("Keys", "sf2player")
        _, node = self.settings(proj)
        node.set("gain", "2.5")
        json.loads(srv.set_instrument_file(0, str(a)))
        json.loads(srv.set_instrument_file(0, str(b)))
        assert node.get("src") == str(b)
        assert node.get("gain") == "2.5"
        assert len(list(proj.root.iter("sf2player"))) == 1

    def test_drum_bank(self, env):
        srv, proj, tmp = env
        sf2 = make_sf2(tmp / "gm.sf2", [(0, 0, "Piano"), (128, 0, "Standard")])
        srv.add_instrument_track("Drums", "sf2player")
        result = json.loads(srv.set_instrument_file(0, str(sf2), bank=128))
        assert result["preset"] == "Standard"
        assert self.settings(proj)[1].get("bank") == "128"

    def test_gigplayer_gets_src_bank_patch_and_gain(self, env):
        srv, proj, tmp = env
        gig = touch(tmp / "lib.gig")
        srv.add_instrument_track("G", "gigplayer")
        result = json.loads(
            srv.set_instrument_file(0, str(gig), bank=3, patch=7))
        assert result["instrument"] == "gigplayer"
        assert "preset" not in result
        name, node = self.settings(proj)
        assert name == "gigplayer"
        assert node.attrib == {
            "src": str(gig), "bank": "3", "patch": "7", "gain": "1"}

    def test_patman_gets_only_src(self, env):
        srv, proj, tmp = env
        pat = touch(tmp / "flute.pat")
        srv.add_instrument_track("P", "patman")
        result = json.loads(srv.set_instrument_file(0, str(pat)))
        assert result["message"] == "Set flute.pat on patman track 0"
        assert "bank" not in result and "patch" not in result
        assert self.settings(proj)[1].attrib == {"src": str(pat)}

    @pytest.mark.parametrize("kwargs", [{"bank": 1}, {"patch": 5}])
    def test_patman_rejects_bank_and_patch(self, env, kwargs):
        srv, proj, tmp = env
        pat = touch(tmp / "flute.pat")
        srv.add_instrument_track("P", "patman")
        result = json.loads(srv.set_instrument_file(0, str(pat), **kwargs))
        assert "patman has no bank or patch" in result["error"]
        assert "src" not in self.settings(proj)[1].attrib

    def test_saved_project_keeps_the_file(self, env):
        srv, proj, tmp = env
        sf2 = make_sf2(tmp / "gm.sf2")
        srv.add_instrument_track("Keys", "sf2player")
        srv.set_instrument_file(0, str(sf2), patch=0)
        saved = Path(proj.save(tmp / "p.mmp", compressed=False))
        assert f'src="{sf2}"' in saved.read_text()
        reloaded = LMMSProject()
        reloaded.load(str(saved))
        assert find_tracks(reloaded.root)[0].find(
            "instrumenttrack/instrument/sf2player").get("src") == str(sf2)

    # --- wrong track -------------------------------------------------------

    def test_other_instruments_are_refused(self, env):
        srv, proj, tmp = env
        sf2 = make_sf2(tmp / "gm.sf2")
        srv.add_instrument_track("Lead", "tripleoscillator")
        before = proj.get_xml_string()
        result = json.loads(srv.set_instrument_file(0, str(sf2)))
        assert "uses instrument 'tripleoscillator', which does not play a " \
            "file" in result["error"]
        assert "sf2player / gigplayer / patman" in result["error"]
        assert proj.get_xml_string() == before

    def test_audiofileprocessor_is_sent_to_its_own_tool(self, env):
        srv, _, tmp = env
        srv.add_instrument_track("Kick", "audiofileprocessor")
        result = json.loads(srv.set_instrument_file(
            0, str(touch(tmp / "kick.wav"))))
        assert "use set_audiofile_sample" in result["error"]

    def test_non_instrument_tracks_and_bad_indexes_are_refused(self, env):
        srv, _, tmp = env
        sf2 = make_sf2(tmp / "gm.sf2")
        srv.add_sample_track("Audio")
        assert "not an instrument track" in json.loads(
            srv.set_instrument_file(0, str(sf2)))["error"]
        for index in (1, -1):
            assert "out of range" in json.loads(
                srv.set_instrument_file(index, str(sf2)))["error"]

    def test_the_other_file_tool_refuses_these_tracks(self, env):
        srv, _, _tmp = env
        srv.add_instrument_track("Keys", "sf2player")
        error = json.loads(srv.set_audiofile_sample(0, "drums/kick01.ogg"))
        assert "not audiofileprocessor" in error["error"]

    def test_only_offered_kinds_are_named_in_errors(self, env, monkeypatch):
        srv, _, tmp = env
        monkeypatch.setattr(srv, "_availability", _availability_with(
            _files(sf2player=["/a.sf2"])))
        srv.add_instrument_track("Lead", "tripleoscillator")
        error = json.loads(srv.set_instrument_file(0, str(tmp / "x.sf2")))
        assert "sf2player" in error["error"]
        assert "patman" not in error["error"]
        assert "gigplayer" not in error["error"]

    def test_a_track_whose_file_is_missing_everywhere_can_still_be_set(
            self, env, monkeypatch):
        """A loaded project may hold a gigplayer track while this machine
        has no .gig library yet; pointing it at a file still works."""
        srv, proj, tmp = env
        monkeypatch.setattr(srv, "_availability", _availability_with(
            _files(sf2player=["/a.sf2"])))
        proj.add_track("instrument", "G", instrument="gigplayer")
        gig = touch(tmp / "lib.gig")
        result = json.loads(srv.set_instrument_file(0, str(gig)))
        assert "error" not in result
        assert self.settings(proj)[1].get("src") == str(gig)

    # --- wrong file ----------------------------------------------------------

    def test_missing_file_is_refused(self, env):
        srv, _, tmp = env
        srv.add_instrument_track("Keys", "sf2player")
        result = json.loads(srv.set_instrument_file(0, str(tmp / "gone.sf2")))
        assert result["error"] == f"File not found: {tmp / 'gone.sf2'}"

    def test_a_folder_is_not_a_file(self, env):
        srv, _, tmp = env
        (tmp / "dir.sf2").mkdir()
        srv.add_instrument_track("Keys", "sf2player")
        assert "File not found" in json.loads(
            srv.set_instrument_file(0, str(tmp / "dir.sf2")))["error"]

    @pytest.mark.parametrize("instrument,name", [
        ("sf2player", "song.wav"), ("sf2player", "lib.gig"),
        ("gigplayer", "font.sf2"), ("patman", "font.sf2"),
        ("sf2player", "noextension"),
    ])
    def test_wrong_extension_is_refused(self, env, instrument, name):
        srv, proj, tmp = env
        srv.add_instrument_track("T", instrument)
        before = proj.get_xml_string()
        result = json.loads(
            srv.set_instrument_file(0, str(touch(tmp / name))))
        assert f"{instrument} plays " in result["error"]
        assert f"got '{name}'" in result["error"]
        assert proj.get_xml_string() == before

    def test_compressed_soundfonts_are_refused_with_the_reason(self, env):
        srv, _, tmp = env
        srv.add_instrument_track("Keys", "sf2player")
        result = json.loads(
            srv.set_instrument_file(0, str(touch(tmp / "big.sf3"))))
        assert "sf2player plays .sf2 files" in result["error"]
        assert "crash" in result["error"]

    def test_extension_case_is_ignored(self, env):
        srv, _, tmp = env
        srv.add_instrument_track("Keys", "sf2player")
        assert "error" not in json.loads(
            srv.set_instrument_file(0, str(touch(tmp / "LOUD.SF2"))))

    def test_relative_and_empty_paths_are_refused(self, env):
        srv, _, _tmp = env
        srv.add_instrument_track("Keys", "sf2player")
        assert "absolute path" in json.loads(
            srv.set_instrument_file(0, "gm.sf2"))["error"]
        assert "must not be empty" in json.loads(
            srv.set_instrument_file(0, "  "))["error"]

    # --- bank and patch ------------------------------------------------------

    @pytest.mark.parametrize("bank,patch,message", [
        (-1, 0, "bank must be 0-999"), (1000, 0, "bank must be 0-999"),
        (0, -1, "patch must be 0-127"), (0, 128, "patch must be 0-127"),
    ])
    def test_bank_and_patch_ranges(self, env, bank, patch, message):
        srv, _, tmp = env
        gig = touch(tmp / "lib.gig")
        srv.add_instrument_track("G", "gigplayer")
        result = json.loads(
            srv.set_instrument_file(0, str(gig), bank=bank, patch=patch))
        assert message in result["error"]

    def test_range_limits_are_accepted(self, env):
        srv, _, tmp = env
        gig = touch(tmp / "lib.gig")
        srv.add_instrument_track("G", "gigplayer")
        assert "error" not in json.loads(
            srv.set_instrument_file(0, str(gig), bank=999, patch=127))

    def test_missing_preset_is_refused_and_lists_what_exists(self, env):
        srv, proj, tmp = env
        sf2 = make_sf2(tmp / "gm.sf2", [
            (0, 0, "A"), (0, 1, "B"), (0, 2, "C"), (128, 0, "Kit")])
        srv.add_instrument_track("Keys", "sf2player")
        before = proj.get_xml_string()
        result = json.loads(srv.set_instrument_file(0, str(sf2), bank=5))
        assert result["error"] == (
            "gm.sf2 has no preset at bank 5, patch 0. It has: "
            "bank 0: patches 0-2; bank 128: patches 0.")
        assert "no preset at bank 0, patch 9" in json.loads(
            srv.set_instrument_file(0, str(sf2), patch=9))["error"]
        assert proj.get_xml_string() == before

    def test_gig_banks_are_not_checked_against_presets(self, env):
        srv, _, tmp = env
        gig = touch(tmp / "lib.gig")
        srv.add_instrument_track("G", "gigplayer")
        assert "error" not in json.loads(
            srv.set_instrument_file(0, str(gig), bank=9, patch=99))

    def test_registered_tool_schema(self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS)
        tool = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}[
            "set_instrument_file"]
        props = tool.input_schema["properties"]
        assert set(props) == {"track_index", "file_path", "bank", "patch"}
        assert tool.input_schema["required"] == ["track_index", "file_path"]


class TestPitchCalibration:
    """Key 57 = A4 = 440 Hz on every offered instrument (src/lmms_mcp/pitch.py).

    The offsets are measured with LMMS 1.2.2: ZynAddSubFX and OPL2 sound
    12 semitones below the key. LMMS transposes a track by 57 - basenote,
    so their tracks are created with basenote 45.
    """

    @staticmethod
    def _written(root, index=0):
        track = find_tracks(root)[index]
        return track.find("instrumenttrack").get("basenote")

    @staticmethod
    def _inst_track(root, index=0):
        return find_tracks(root)[index].find("instrumenttrack")

    @staticmethod
    def _preset_xml(octave=0, engine="add", name="P", extra_items=""):
        """A minimal .xiz whose kit item 0 has the given global octave."""
        coarse = (octave * 1024) % 16384
        branch = {"add": "ADD_SYNTH_PARAMETERS", "pad": "PAD_SYNTH_PARAMETERS",
                  "sub": "SUB_SYNTH_PARAMETERS"}[engine]
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<ZynAddSubFX-data version-major="1" version-minor="0">'
            '<INFORMATION/><BASE_PARAMETERS/>'
            f'<INSTRUMENT><INFO><string name="name">{name}</string></INFO>'
            '<INSTRUMENT_KIT><INSTRUMENT_KIT_ITEM id="0">'
            '<par_bool name="enabled" value="yes"/>'
            f'<par_bool name="{engine}_enabled" value="yes"/>'
            f'<{branch}><FREQUENCY_PARAMETERS>'
            f'<par name="coarse_detune" value="{coarse}"/>'
            f'</FREQUENCY_PARAMETERS></{branch}>'
            '</INSTRUMENT_KIT_ITEM>'
            f'{extra_items}'
            '</INSTRUMENT_KIT></INSTRUMENT></ZynAddSubFX-data>'
        )

    # --- the table and the helper ------------------------------------------

    def test_table_lists_the_measured_offsets(self):
        from lmms_mcp import pitch
        assert pitch.PITCH_OFFSET_SEMITONES == {
            "zynaddsubfx": 12, "opl2": 12}
        # a sampler keeps key 57 = original pitch
        assert pitch.plugin_pitch_offset("audiofileprocessor") == 0
        assert pitch.plugin_pitch_offset("tripleoscillator") == 0

    def test_offset_lookup_ignores_case(self):
        from lmms_mcp import pitch
        assert pitch.plugin_pitch_offset("OPL2") == 12
        assert pitch.plugin_pitch_offset("ZynAddSubFX") == 12

    def test_effective_basenote(self):
        from lmms_mcp import pitch
        assert pitch.effective_basenote("zynaddsubfx") == 45
        assert pitch.effective_basenote("OPL2") == 45
        assert pitch.effective_basenote("tripleoscillator") == 57
        assert pitch.effective_basenote("audiofileprocessor") == 57
        # a user base note maps to written = user - offset
        assert pitch.effective_basenote("zynaddsubfx", 60) == 48
        assert pitch.effective_basenote("tripleoscillator", 60) == 60
        # preset compensation comes on top of the table
        assert pitch.effective_basenote("zynaddsubfx", 57, 12) == 33
        assert pitch.effective_basenote("zynaddsubfx", 57, -12) == 57

    def test_effective_basenote_refuses_what_lmms_cannot_store(self):
        from lmms_mcp import pitch
        with pytest.raises(ValueError, match="only accepts 0-127"):
            pitch.effective_basenote("zynaddsubfx", 5)
        with pytest.raises(ValueError):
            pitch.effective_basenote("tripleoscillator", 128)

    # --- new tracks --------------------------------------------------------

    def test_new_zynaddsubfx_track_gets_basenote_45(self):
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        assert self._written(root) == "45"

    def test_opl2_track_gets_basenote_45_under_either_name_case(self):
        root = create_empty_project()
        add_instrument_track(root, "O", instrument="OPL2")
        assert self._written(root) == "45"

    def test_other_tracks_keep_basenote_57(self):
        from lmms_mcp import server as srv
        root = create_empty_project()
        names = [n for n in srv.KNOWN_INSTRUMENTS
                 if n not in ("zynaddsubfx", "opulenz")]
        for n in names:
            add_instrument_track(root, n, instrument=n)
        # papu is what freeboy is called in LMMS 1.2 on Linux
        add_instrument_track(root, "papu", instrument="papu")
        assert {self._written(root, i) for i in range(len(names) + 1)} == {"57"}

    def test_pattern_track_kicker_keeps_basenote_57(self):
        root = create_empty_project()
        add_pattern_track(root, "Beat")
        kicker = root.find(".//bbtrack//instrumenttrack")
        assert kicker.get("basenote") == "57"

    def test_server_tool_writes_the_compensated_basenote(self):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("Z", instrument="zynaddsubfx")
        srv.add_instrument_track("T", instrument="tripleoscillator")
        srv.add_instrument_track("S", instrument="audiofileprocessor")
        assert [self._written(proj.root, i) for i in range(3)] == [
            "45", "57", "57"]

    # --- a track's base note ------------------------------------------------

    def test_user_base_note_round_trip(self):
        from lmms_mcp import pitch
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        add_instrument_track(root, "T", instrument="tripleoscillator")
        zyn, tri = self._inst_track(root, 0), self._inst_track(root, 1)
        assert pitch.get_base_note(zyn) == 57
        assert pitch.get_base_note(tri) == 57
        assert pitch.set_base_note(zyn, 60) == 48
        assert zyn.get("basenote") == "48"
        assert pitch.get_base_note(zyn) == 60
        assert pitch.set_base_note(tri, 60) == 60
        with pytest.raises(ValueError):
            pitch.set_base_note(zyn, 3)  # would need basenote -9
        assert zyn.get("basenote") == "48"  # unchanged

    # --- ZynAddSubFX presets carry their own octave -------------------------

    @pytest.mark.parametrize("coarse, octave", [
        (0, 0), (1024, 1), (2048, 2), (7 * 1024, 7), (8 * 1024, -8),
        (15 * 1024, -1), (14 * 1024, -2),
        (1024 + 5, 1), (15 * 1024 + 1023, -1),  # semitone bits are not octaves
    ])
    def test_zyn_octave_from_coarse_detune(self, coarse, octave):
        from lmms_mcp import pitch
        assert pitch.zyn_octave_from_coarse_detune(coarse) == octave

    def test_preset_octave_reads_the_first_enabled_engine(self):
        from lmms_mcp import pitch
        for octave in (-2, -1, 0, 1, 2):
            for engine in ("add", "pad", "sub"):
                root = ET.fromstring(self._preset_xml(octave, engine))
                assert pitch.zyn_preset_octave(root.find("INSTRUMENT")) == octave
        assert pitch.zyn_preset_octave(None) == 0
        assert pitch.zyn_preset_octave(ET.fromstring("<INSTRUMENT/>")) == 0

    def test_preset_octave_prefers_additive_over_pad_and_skips_disabled(self):
        from lmms_mcp import pitch
        both = self._preset_xml(1, "add").replace(
            '<par_bool name="add_enabled" value="yes"/>',
            '<par_bool name="add_enabled" value="yes"/>'
            '<par_bool name="pad_enabled" value="yes"/>'
            '<PAD_SYNTH_PARAMETERS><FREQUENCY_PARAMETERS>'
            '<par name="coarse_detune" value="15360"/>'
            '</FREQUENCY_PARAMETERS></PAD_SYNTH_PARAMETERS>')
        assert pitch.zyn_preset_octave(
            ET.fromstring(both).find("INSTRUMENT")) == 1
        # a disabled first item is skipped
        skipped = self._preset_xml(2, "add").replace(
            '<par_bool name="enabled" value="yes"/>',
            '<par_bool name="enabled" value="no"/>', 1)
        assert pitch.zyn_preset_octave(
            ET.fromstring(skipped).find("INSTRUMENT")) == 0

    @pytest.mark.parametrize("octave, written", [
        (0, "45"), (1, "57"), (-1, "33"), (2, "69"), (-2, "21"),
    ])
    def test_loading_a_preset_compensates_its_octave(self, octave, written):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        result = embed_zyn_preset(root, 0, self._preset_xml(octave))
        assert result["preset_octave"] == octave
        assert self._written(root) == written
        # the user-facing base note stays A4
        from lmms_mcp import pitch
        assert pitch.get_base_note(self._inst_track(root)) == 57

    def test_switching_presets_follows_the_octave_each_time(self):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        seen = []
        for octave in (1, 0, -1, 1, 0):
            embed_zyn_preset(root, 0, self._preset_xml(octave))
            seen.append(self._written(root))
        assert seen == ["57", "45", "33", "57", "45"]

    def test_custom_base_note_survives_a_preset_change(self):
        from lmms_mcp import pitch
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        pitch.set_base_note(self._inst_track(root), 60)
        embed_zyn_preset(root, 0, self._preset_xml(1))
        assert pitch.get_base_note(self._inst_track(root)) == 60
        embed_zyn_preset(root, 0, self._preset_xml(0))
        assert pitch.get_base_note(self._inst_track(root)) == 60

    def test_uncalibrated_zyn_track_is_not_shifted_by_a_flat_preset(self):
        """A track made elsewhere (basenote 57) only moves by the octave
        difference its preset brings, not by the table offset."""
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        self._inst_track(root).set("basenote", "57")
        embed_zyn_preset(root, 0, self._preset_xml(0))
        assert self._written(root) == "57"

    def test_failed_preset_load_leaves_the_basenote(self):
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        with pytest.raises(ValueError):
            embed_zyn_preset(root, 0, "<not-a-preset/>")
        assert self._written(root) == "45"

    def test_load_zyn_preset_tool_reports_the_octave(self, tmp_path, monkeypatch):
        import gzip
        from lmms_mcp import lmms_app, presets
        from lmms_mcp import server as srv
        zyn_dir = tmp_path / "ZynAddSubFX"
        (zyn_dir / "Bass").mkdir(parents=True)
        for rel, octave, name in (("Bass/0001-Deep.xiz", -1, "Deep"),
                                  ("Bass/0002-Plain.xiz", 0, "Plain"),
                                  ("Bass/0003-Bell.xiz", 1, "Bell")):
            with gzip.open(zyn_dir / rel, "wb") as f:
                f.write(self._preset_xml(octave, name=name).encode())
        monkeypatch.setenv("LMMS_PRESETS_DIR", str(zyn_dir))
        monkeypatch.setattr(presets, "_static_preset_candidates", lambda: [])
        monkeypatch.setattr(lmms_app, "find_lmms_exe", lambda: None)
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("Z", instrument="zynaddsubfx")

        deep = json.loads(srv.load_zyn_preset(0, "Bass/Deep"))
        assert deep["preset_octave"] == -1
        assert "-1 octave" in deep["note"] and "A4 = 440 Hz" in deep["note"]
        assert deep["playable_keys"] == [0, 83]
        assert self._written(proj.root) == "33"

        plain = json.loads(srv.load_zyn_preset(0, "Bass/Plain"))
        assert "preset_octave" not in plain and "note" not in plain
        assert plain["playable_keys"] == [0, 95]
        assert self._written(proj.root) == "45"

        bell = json.loads(srv.load_zyn_preset(0, "Bass/Bell"))
        assert bell["preset_octave"] == 1
        assert bell["playable_keys"] == [0, 107]
        assert self._written(proj.root) == "57"

    # --- keys a plugin cannot play ----------------------------------------

    def test_playable_key_ranges(self):
        from lmms_mcp import pitch
        root = create_empty_project()
        for n in ("zynaddsubfx", "tripleoscillator", "papu", "sid", "OPL2"):
            add_instrument_track(root, n, instrument=n)
        ranges = [pitch.playable_key_range(self._inst_track(root, i))
                  for i in range(5)]
        assert ranges == [(0, 95), None, (24, 95), (2, 94), (0, 90)]

    def test_playable_keys_follow_the_written_basenote(self):
        from lmms_mcp import pitch
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        track = self._inst_track(root)
        assert pitch.playable_key_range(track) == (0, 95)
        track.set("basenote", "57")  # nothing added to the key any more
        assert pitch.playable_key_range(track) == (0, 107)
        track.set("basenote", "33")  # 24 semitones added
        assert pitch.playable_key_range(track) == (0, 83)

    def test_key_range_warning(self):
        from lmms_mcp import pitch
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        add_instrument_track(root, "T", instrument="tripleoscillator")
        zyn, tri = self._inst_track(root, 0), self._inst_track(root, 1)
        assert pitch.key_range_warning(zyn, 95) is None
        warning = pitch.key_range_warning(zyn, 96)
        assert "Key 96 (C8)" in warning and "0-95" in warning
        assert pitch.key_range_warning(tri, 108) is None

    def test_add_instrument_track_reports_playable_keys(self):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        zyn = json.loads(srv.add_instrument_track("Z", instrument="zynaddsubfx"))
        tri = json.loads(srv.add_instrument_track("T", instrument="tripleoscillator"))
        assert zyn["playable_keys"] == [0, 95]
        assert "playable_keys" not in tri

    def test_add_note_warns_outside_the_playable_keys_but_adds_it(self):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("Z", instrument="zynaddsubfx")
        srv.add_instrument_track("T", instrument="tripleoscillator")
        ok = json.loads(srv.add_note(0, key=95))
        assert "warning" not in ok
        high = json.loads(srv.add_note(0, key=100))
        assert "Key 100" in high["warning"] and "Added note" in high["message"]
        assert "warning" not in json.loads(srv.add_note(1, key=100))
        by_name = json.loads(srv.add_note_by_name(0, "C8"))
        assert "Key 96" in by_name["warning"]
        keys = [n.get("key") for n in proj.root.iter("note")]
        assert keys == ["95", "100", "96", "100"]

    def test_add_notes_batch_warns_per_note_and_overall(self):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("Z", instrument="zynaddsubfx")
        res = json.loads(srv.add_notes_batch(0, [
            {"key": 57, "pos": 0}, {"key": 96, "pos": 48},
            {"key": 108, "pos": 96}, {"key": 200, "pos": 144},
        ]))
        assert res["added"] == 3 and res["failed"] == 1
        assert [("warning" in n) for n in res["notes"]] == [False, True, True]
        assert "2 of 3 added notes are outside" in res["warning"]
        assert "1 of 4 notes were NOT added" in res["warning"]
        clean = json.loads(srv.add_notes_batch(0, [{"key": 60, "pos": 0}]))
        assert "warning" not in clean

    # --- what the agent is told -------------------------------------------

    def test_instructions_state_the_pitch_and_its_exceptions(self):
        from lmms_mcp import server as srv
        text = " ".join(srv.build_instructions(
            srv._availability, "/tmp/p").split())
        assert "every offered instrument is tuned so that key 57 = A4 = 440 Hz" in text
        # compensated: zynaddsubfx everywhere; opulenz only where LMMS
        # loads it as OPL2 (the fake install here has a plugin "opulenz")
        assert "octave low in LMMS (zynaddsubfx)" in text
        assert "kicker is a drum and ignores the key" in text
        assert "sfxr is a sound-effect synth" in text
        assert "audiofileprocessor plays its sample at its original pitch" in text
        assert "e.g. zynaddsubfx plays keys 0-95 only" in text
        assert "an octave shift built into a preset" in text

    def test_instructions_only_name_offered_instruments(self):
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv
        av = lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
            {"tripleoscillator", "lb302"}, "1.2.2", windows=False)
        text = " ".join(srv.build_instructions(av, "/tmp/p").split())
        assert "key 57 = A4 = 440 Hz" in text
        for word in ("kicker", "sfxr", "zynaddsubfx", "opulenz",
                     "audiofileprocessor", "playable_keys"):
            assert word not in text, word

    def test_instructions_name_opulenz_when_lmms_loads_it_as_opl2(self):
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv
        av = lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
            {"tripleoscillator", "opl2", "zynaddsubfx"}, "1.2.2",
            stems={"opl2": "OPL2"}, windows=False)
        assert av.plugin_name("opulenz") == "OPL2"
        text = " ".join(srv.build_instructions(av, "/tmp/p").split())
        assert "octave low in LMMS (opulenz, zynaddsubfx)" in text

    def test_default_playable_keys_of_a_new_track(self):
        from lmms_mcp import pitch
        assert pitch.default_playable_keys("zynaddsubfx") == (0, 95)
        assert pitch.default_playable_keys("OPL2") == (0, 90)
        assert pitch.default_playable_keys("papu") == (24, 95)
        assert pitch.default_playable_keys("monstro") is None


class TestTrackTuning:
    """set_track_tuning: a track's base note and fine tune (cents).

    LMMS plays a sample at its original pitch when the key equals the
    track's base note and moves every note by `pitch` cents (limited to
    100 cents times `pitchrange`). Both were measured with LMMS 1.2.2
    renders; see pitch.py for the plugins that do not follow the fine tune.
    """

    @staticmethod
    def _inst(root, index=0):
        return find_tracks(root)[index].find("instrumenttrack")

    @staticmethod
    def _new_project():
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        return srv, proj

    def _track(self, instrument):
        root = create_empty_project()
        add_instrument_track(root, "T", instrument=instrument)
        return self._inst(root)

    # --- parse_base_note -----------------------------------------------------

    @pytest.mark.parametrize("value, key", [
        ("C4", 48), ("A4", 57), ("Bb2", 34), ("bb2", 34), ("c0", 0),
        ("C9", 108), (48, 48), (0, 0), (108, 108), (48.0, 48), ("48", 48),
        (" D3 ", 38),
    ])
    def test_parse_base_note(self, value, key):
        from lmms_mcp import pitch
        assert pitch.parse_base_note(value) == key

    @pytest.mark.parametrize("value", [
        "H4", "", "C", "C10", True, 48.5, -1, 109, 200, None, [48], "4.5",
    ])
    def test_parse_base_note_rejects(self, value):
        from lmms_mcp import pitch
        with pytest.raises(ValueError):
            pitch.parse_base_note(value)

    def test_parse_base_note_out_of_range_mentions_midi(self):
        from lmms_mcp import pitch
        with pytest.raises(ValueError, match="use 108"):
            pitch.parse_base_note(120)

    # --- the base note maps through the plugin's offset ----------------------

    def test_base_note_name_and_key_on_a_sampler(self):
        from lmms_mcp import pitch
        track = self._track("audiofileprocessor")
        assert pitch.set_tuning(track, pitch.parse_base_note("C4")).basenote == 48
        assert track.get("basenote") == "48"
        pitch.set_tuning(track, 26)
        assert track.get("basenote") == "26"
        assert pitch.get_base_note(track) == 26

    @pytest.mark.parametrize("base, written", [
        (57, "45"),   # A4: the calibrated default stays
        (48, "36"),   # C4
        (45, "33"),   # A3
        (60, "48"),
        (12, "0"),    # lowest base note the compensation allows
    ])
    def test_base_note_keeps_the_zynaddsubfx_offset(self, base, written):
        from lmms_mcp import pitch
        track = self._track("zynaddsubfx")
        pitch.set_tuning(track, base)
        assert track.get("basenote") == written
        assert pitch.get_base_note(track) == base

    def test_base_note_keeps_the_opl2_offset(self):
        from lmms_mcp import pitch
        for name in ("OPL2", "opl2"):
            track = self._track(name)
            pitch.set_tuning(track, 48)
            assert track.get("basenote") == "36"
            assert pitch.get_base_note(track) == 48

    def test_base_note_keeps_a_zyn_preset_octave(self):
        from lmms_mcp import pitch
        from lmms_mcp.xml_parser import embed_zyn_preset
        root = create_empty_project()
        add_instrument_track(root, "Z", instrument="zynaddsubfx")
        embed_zyn_preset(root, 0, TestPitchCalibration._preset_xml(1))
        track = self._inst(root)
        assert track.get("basenote") == "57"   # +1 octave needs no offset
        pitch.set_tuning(track, 48)
        assert track.get("basenote") == "48"
        assert pitch.get_base_note(track) == 48
        # loading another preset keeps the user's base note
        embed_zyn_preset(root, 0, TestPitchCalibration._preset_xml(-1))
        assert pitch.get_base_note(track) == 48
        assert track.get("basenote") == "24"

    def test_zyn_base_note_too_low_names_the_lowest_one(self):
        from lmms_mcp import pitch
        track = self._track("zynaddsubfx")
        with pytest.raises(ValueError) as exc:
            pitch.set_tuning(track, 5)
        assert "F0 (5)" in str(exc.value) or "Base note F0" in str(exc.value)
        assert "C1 (12)" in str(exc.value)
        assert track.get("basenote") == "45"

    # --- the fine tune -----------------------------------------------------

    @pytest.mark.parametrize("cents, pitch_attr, rng", [
        (18, "18", "1"), (-18, "-18", "1"), (0, "0", "1"), (100, "100", "1"),
        (-100, "-100", "1"), (100.5, "100.5", "2"), (150, "150", "2"),
        (-250.5, "-250.5", "3"), (299.99, "299.99", "3"), (300, "300", "3"),
        (1200, "1200", "12"), (-1200, "-1200", "12"), (18.0, "18", "1"),
        (12.345, "12.35", "1"),
    ])
    def test_pitch_cents_writes_pitch_and_the_range_it_needs(
            self, cents, pitch_attr, rng):
        from lmms_mcp import pitch
        track = self._track("tripleoscillator")
        pitch.set_tuning(track, pitch_cents=cents)
        assert track.get("pitch") == pitch_attr
        assert track.get("pitchrange") == rng
        assert track.get("basenote") == "57"      # untouched

    def test_pitch_range_is_raised_never_lowered(self):
        from lmms_mcp import pitch
        track = self._track("tripleoscillator")
        track.set("pitchrange", "5")
        pitch.set_tuning(track, pitch_cents=150)
        assert track.get("pitchrange") == "5"
        pitch.set_tuning(track, pitch_cents=900)
        assert track.get("pitchrange") == "9"
        pitch.set_tuning(track, pitch_cents=10)
        assert track.get("pitchrange") == "9"

    def test_reads_back_pitch_cents_and_range(self):
        from lmms_mcp import pitch
        track = self._track("tripleoscillator")
        assert pitch.get_pitch_cents(track) == 0
        assert pitch.get_pitchrange(track) == 1
        pitch.set_tuning(track, pitch_cents=-250)
        assert pitch.get_pitch_cents(track) == -250
        assert pitch.get_pitchrange(track) == 3
        track.set("pitch", "garbage")
        assert pitch.get_pitch_cents(track) == 0

    @pytest.mark.parametrize("cents", [
        1200.5, -1201, 5000, float("nan"), float("inf"), True, "18", None,
    ])
    def test_pitch_cents_validation(self, cents):
        from lmms_mcp import pitch
        track = self._track("tripleoscillator")
        if cents is None:   # None means "leave alone"
            assert pitch.plan_tuning(track, None, None) == pitch.Tuning()
            return
        with pytest.raises(ValueError):
            pitch.set_tuning(track, pitch_cents=cents)
        assert track.get("pitch") == "0"

    def test_out_of_range_error_points_to_base_note(self):
        from lmms_mcp import pitch
        track = self._track("tripleoscillator")
        with pytest.raises(ValueError, match="-1200 to \\+1200.*base_note"):
            pitch.set_tuning(track, pitch_cents=1500)

    def test_nothing_is_written_when_one_argument_is_invalid(self):
        from lmms_mcp import pitch
        track = self._track("tripleoscillator")
        with pytest.raises(ValueError):
            pitch.set_tuning(track, 48, 5000)
        assert (track.get("basenote"), track.get("pitch")) == ("57", "0")
        zyn = self._track("zynaddsubfx")
        with pytest.raises(ValueError):
            pitch.set_tuning(zyn, 48, 150)      # fine tune too large for zyn
        assert (zyn.get("basenote"), zyn.get("pitch")) == ("45", "0")

    # --- plugins that do not follow the fine tune ----------------------------

    def test_zynaddsubfx_fine_tune_is_limited_to_100_cents(self):
        from lmms_mcp import pitch
        zyn = self._track("zynaddsubfx")
        pitch.set_tuning(zyn, pitch_cents=-100)
        assert zyn.get("pitch") == "-100"
        with pytest.raises(ValueError, match="only up to \\+-100 cents.*base_note"):
            pitch.set_tuning(zyn, pitch_cents=150)

    def test_zynaddsubfx_keeps_pitchrange_1(self):
        """A bigger range would divide the fine tune (measured: +150 cents
        at range 2 sounds +75)."""
        from lmms_mcp import pitch
        zyn = self._track("zynaddsubfx")
        zyn.set("pitchrange", "12")
        pitch.set_tuning(zyn, pitch_cents=20)
        assert zyn.get("pitchrange") == "1"

    def test_opl2_ignores_the_fine_tune_but_takes_a_base_note(self):
        from lmms_mcp import pitch
        track = self._track("OPL2")
        with pytest.raises(ValueError, match="ignores the fine tune.*base_note"):
            pitch.set_tuning(track, pitch_cents=50)
        pitch.set_tuning(track, pitch_cents=0)      # resetting is fine
        pitch.set_tuning(track, base_note=48)
        assert track.get("basenote") == "36"

    def test_kicker_cannot_be_tuned(self):
        from lmms_mcp import pitch
        track = self._track("kicker")
        for kwargs in ({"base_note": 48}, {"pitch_cents": 10}):
            with pytest.raises(ValueError, match="ignores the key"):
                pitch.set_tuning(track, **kwargs)
        assert pitch.plan_tuning(track) == pitch.Tuning()   # reading is fine
        assert "ignores the key" in pitch.tuning_hint(track)

    # --- reports and hints ---------------------------------------------------

    def test_tuning_report_of_a_default_and_a_tuned_track(self):
        from lmms_mcp import pitch
        track = self._track("zynaddsubfx")
        report = pitch.tuning_report(track)
        assert report["base_note"] == 57 and report["base_note_name"] == "A4"
        assert report["written_basenote"] == 45
        assert report["pitch_cents"] == 0 and report["pitchrange"] == 1
        assert report["playable_keys"] == [0, 95]
        assert report["instrument"] == "zynaddsubfx"
        pitch.set_tuning(track, 48, 25.5)
        report = pitch.tuning_report(track)
        assert (report["base_note"], report["base_note_name"]) == (48, "C4")
        assert report["written_basenote"] == 36
        assert report["pitch_cents"] == 25.5
        assert report["playable_keys"] == [0, 86]
        assert isinstance(pitch.tuning_report(self._track("monstro"))["pitch_cents"], int)
        assert "playable_keys" not in pitch.tuning_report(self._track("monstro"))

    def test_hint_for_a_sampler(self):
        from lmms_mcp import pitch
        track = self._track("audiofileprocessor")
        assert pitch.tuning_hint(track) == (
            "Key 57 (A4) plays the sample at its original pitch.")
        pitch.set_tuning(track, 48, 18)
        hint = pitch.tuning_hint(track)
        assert hint.startswith("Key 48 (C4) plays the sample at its original pitch")
        assert "key 57 = A4 is 9 semitones higher" in hint
        assert "fine tune +18 cents" in hint
        pitch.set_tuning(track, 60)
        assert "key 57 = A4 is 3 semitones lower" in pitch.tuning_hint(track)

    def test_hint_for_a_synth(self):
        from lmms_mcp import pitch
        track = self._track("tripleoscillator")
        assert pitch.tuning_hint(track) == "Key 57 (A4) sounds at 440 Hz."
        pitch.set_tuning(track, 48, -30)
        hint = pitch.tuning_hint(track)
        assert hint.startswith("Key 48 (C4) sounds at 440 Hz (A4)")
        assert "every key sounds 9 semitones higher than its note name" in hint
        assert "fine tune -30 cents" in hint
        pitch.set_tuning(track, 69)
        assert "12 semitones lower" in pitch.tuning_hint(track)

    # --- the tool ------------------------------------------------------------

    def test_tool_tunes_a_piano_sample_recorded_at_c4(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("Piano", instrument="audiofileprocessor")
        result = json.loads(srv.set_track_tuning(0, base_note="C4"))
        assert "error" not in result
        assert result["track_index"] == 0
        assert result["base_note"] == 48
        assert result["base_note_name"] == "C4"
        assert result["written_basenote"] == 48
        assert result["pitch_cents"] == 0
        assert result["pitchrange"] == 1
        assert "playable_keys" not in result
        assert result["hint"].startswith("Key 48 (C4) plays the sample")
        assert "C4" in result["message"]
        assert self._inst(proj.root).get("basenote") == "48"
        assert proj.modified is True

    def test_tool_accepts_a_key_number_and_both_arguments(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("Cello", instrument="audiofileprocessor")
        result = json.loads(srv.set_track_tuning(0, base_note=38, pitch_cents=18))
        assert result["base_note"] == 38 and result["base_note_name"] == "D3"
        assert result["pitch_cents"] == 18 and result["pitchrange"] == 1
        track = self._inst(proj.root)
        assert (track.get("basenote"), track.get("pitch")) == ("38", "18")
        assert "+18" in result["message"] and "D3" in result["message"]

    def test_tool_leaves_an_omitted_argument_alone(self):
        srv, _ = self._new_project()
        srv.add_instrument_track("S", instrument="audiofileprocessor")
        srv.set_track_tuning(0, base_note="G4", pitch_cents=-12)
        first = json.loads(srv.set_track_tuning(0, pitch_cents=7))
        assert first["base_note"] == 55 and first["pitch_cents"] == 7
        second = json.loads(srv.set_track_tuning(0, base_note="C4"))
        assert second["base_note"] == 48 and second["pitch_cents"] == 7

    def test_tool_without_arguments_returns_the_tuning_and_changes_nothing(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("S", instrument="audiofileprocessor")
        srv.set_track_tuning(0, base_note="C4", pitch_cents=18)
        proj._modified = False
        before = ET.tostring(proj.root)
        result = json.loads(srv.set_track_tuning(0))
        assert result["base_note_name"] == "C4" and result["pitch_cents"] == 18
        assert "unchanged" in result["message"]
        assert proj.modified is False
        assert ET.tostring(proj.root) == before

    def test_tool_pitch_cents_beyond_100_raises_the_range(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("L", instrument="tripleoscillator")
        result = json.loads(srv.set_track_tuning(0, pitch_cents=-250))
        assert result["pitch_cents"] == -250 and result["pitchrange"] == 3
        assert self._inst(proj.root).get("pitchrange") == "3"

    def test_tool_on_zynaddsubfx_reports_playable_keys_and_offset(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("Z", instrument="zynaddsubfx")
        default = json.loads(srv.set_track_tuning(0))
        assert default["base_note"] == 57 and default["written_basenote"] == 45
        assert default["playable_keys"] == [0, 95]
        result = json.loads(srv.set_track_tuning(0, base_note="C4"))
        assert result["base_note"] == 48 and result["written_basenote"] == 36
        assert result["playable_keys"] == [0, 86]
        too_low = json.loads(srv.set_track_tuning(0, base_note="C0"))
        assert "C1 (12)" in too_low["error"]
        too_far = json.loads(srv.set_track_tuning(0, pitch_cents=150))
        assert "+-100 cents" in too_far["error"]
        assert self._inst(proj.root).get("basenote") == "36"

    def test_tool_errors_are_json_and_change_nothing(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("S", instrument="audiofileprocessor")
        before = ET.tostring(proj.root)
        for kwargs in ({"base_note": "H4"}, {"base_note": 200},
                       {"base_note": "C4", "pitch_cents": 2000},
                       {"pitch_cents": float("nan")}, {"base_note": ""}):
            result = json.loads(srv.set_track_tuning(0, **kwargs))
            assert result["error"].startswith("Tuning NOT changed:"), kwargs
        assert ET.tostring(proj.root) == before
        for index in (5, -1):
            assert "error" in json.loads(srv.set_track_tuning(index, base_note="C4"))

    def test_tool_refuses_tracks_that_are_not_instrument_tracks(self):
        srv, _ = self._new_project()
        srv.add_sample_track("Samples")
        srv.add_pattern_track("Beat")
        srv.add_automation_track("Auto")
        for index, kind in ((0, "sample"), (1, "pattern"), (2, "automation")):
            result = json.loads(srv.set_track_tuning(index, base_note="C4"))
            assert f"is a {kind} track, not an instrument track" in result["error"]
            assert "add_instrument_track" in result["error"]
        empty = LMMSProject()
        empty.new()
        srv.set_project(empty)
        assert "no tracks" in json.loads(srv.set_track_tuning(0))["error"]

    def test_tool_refuses_kicker_and_opl2_fine_tune(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("K", instrument="kicker")
        # LMMS 1.2 loads opulenz as "OPL2" (the test install has "opulenz")
        proj.add_track("instrument", "FM", instrument="OPL2")
        assert "ignores the key" in json.loads(
            srv.set_track_tuning(0, base_note="C4"))["error"]
        assert "ignores the fine tune" in json.loads(
            srv.set_track_tuning(1, pitch_cents=10))["error"]
        ok = json.loads(srv.set_track_tuning(1, base_note="C4"))
        assert ok["base_note"] == 48 and ok["written_basenote"] == 36

    # --- get_track / list_tracks ---------------------------------------------

    def test_get_track_and_list_tracks_show_a_non_default_tuning(self):
        srv, _ = self._new_project()
        srv.add_instrument_track("Plain", instrument="tripleoscillator")
        srv.add_instrument_track("Zyn", instrument="zynaddsubfx")
        srv.add_instrument_track("Cello", instrument="audiofileprocessor")
        srv.add_instrument_track("Pad", instrument="audiofileprocessor")
        srv.add_sample_track("Samples")
        srv.set_track_tuning(2, base_note="D3", pitch_cents=18)
        srv.set_track_tuning(3, pitch_cents=-7.5)
        for index in (0, 1, 4):     # default tuning: no extra fields
            track = json.loads(srv.get_track(index))
            assert not {"base_note", "base_note_name", "pitch_cents"} & set(track)
        cello = json.loads(srv.get_track(2))
        assert cello["base_note"] == 38 and cello["base_note_name"] == "D3"
        assert cello["pitch_cents"] == 18
        pad = json.loads(srv.get_track(3))
        assert pad["base_note"] == 57 and pad["base_note_name"] == "A4"
        assert pad["pitch_cents"] == -7.5
        listed = {t["index"]: t for t in json.loads(srv.list_tracks())["tracks"]}
        assert listed[2]["base_note_name"] == "D3" and listed[2]["pitch_cents"] == 18
        assert listed[3]["pitch_cents"] == -7.5
        for index in (0, 1, 4):
            assert "base_note" not in listed[index]
            assert "pitch_cents" not in listed[index]

    def test_zyn_track_shows_the_user_facing_base_note(self):
        srv, _ = self._new_project()
        srv.add_instrument_track("Zyn", instrument="zynaddsubfx")
        srv.set_track_tuning(0, base_note="C4")       # written 36
        track = json.loads(srv.get_track(0))
        assert track["base_note"] == 48 and track["base_note_name"] == "C4"

    # --- set_audiofile_sample takes the tuning too -------------------------------

    def test_set_audiofile_sample_sets_sample_and_tuning_in_one_call(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("Cello", instrument="audiofileprocessor")
        result = json.loads(srv.set_audiofile_sample(
            0, "instruments/cello01.ogg", base_note="D3", pitch_cents=18))
        assert "error" not in result
        assert result["tuning"]["base_note"] == 38
        assert result["tuning"]["base_note_name"] == "D3"
        assert result["tuning"]["pitch_cents"] == 18
        assert "Key 38 (D3) plays the sample at its original pitch" in result["tuning"]["hint"]
        assert "note" not in result["tuning"]
        track = self._inst(proj.root)
        assert (track.get("basenote"), track.get("pitch"),
                track.get("pitchrange")) == ("38", "18", "1")
        assert track.find("instrument/audiofileprocessor").get("src") == \
            "instruments/cello01.ogg"

    def test_set_audiofile_sample_with_a_key_number_and_only_one_value(self):
        srv, _ = self._new_project()
        srv.add_instrument_track("P", instrument="audiofileprocessor")
        result = json.loads(srv.set_audiofile_sample(
            0, "instruments/piano01.ogg", base_note=48))
        assert result["tuning"]["base_note_name"] == "C4"
        assert result["tuning"]["pitch_cents"] == 0
        result = json.loads(srv.set_audiofile_sample(
            0, "instruments/piano01.ogg", pitch_cents=-4))
        assert result["tuning"]["base_note"] == 48      # kept
        assert result["tuning"]["pitch_cents"] == -4

    def test_set_audiofile_sample_default_response_has_no_tuning(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("Kick", instrument="audiofileprocessor")
        result = json.loads(srv.set_audiofile_sample(0, "drums/kick01.ogg"))
        assert "tuning" not in result
        assert self._inst(proj.root).get("basenote") == "57"

    def test_set_audiofile_sample_with_bad_tuning_changes_nothing(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("S", instrument="audiofileprocessor")
        for kwargs in ({"base_note": "H4"}, {"pitch_cents": 5000},
                       {"base_note": "C4", "pitch_cents": 5000}):
            result = json.loads(srv.set_audiofile_sample(
                0, "drums/kick01.ogg", **kwargs))
            assert "error" in result, kwargs
        track = self._inst(proj.root)
        assert "src" not in track.find("instrument/audiofileprocessor").attrib
        assert (track.get("basenote"), track.get("pitch")) == ("57", "0")

    def test_set_audiofile_sample_bad_tuning_keeps_the_previous_sample(self):
        srv, proj = self._new_project()
        srv.add_instrument_track("S", instrument="audiofileprocessor")
        srv.set_audiofile_sample(0, "drums/kick01.ogg", amp=150)
        result = json.loads(srv.set_audiofile_sample(
            0, "drums/snare01.ogg", amp=50, base_note="C4", pitch_cents=-2000))
        assert "error" in result
        afp = self._inst(proj.root).find("instrument/audiofileprocessor")
        assert (afp.get("src"), afp.get("amp")) == ("drums/kick01.ogg", "150")

    def test_new_sample_on_a_tuned_track_reports_the_kept_tuning(self):
        srv, _ = self._new_project()
        srv.add_instrument_track("S", instrument="audiofileprocessor")
        srv.set_audiofile_sample(0, "instruments/piano01.ogg", base_note="C4")
        other = json.loads(srv.set_audiofile_sample(
            0, "instruments/violin_pizzicato01.ogg"))
        assert other["tuning"]["base_note_name"] == "C4"
        assert "keeps the tuning it had" in other["tuning"]["note"]
        assert "set_track_tuning(0, base_note=57, pitch_cents=0)" in other["tuning"]["note"]
        # the same sample again (e.g. to change amp): no note
        again = json.loads(srv.set_audiofile_sample(
            0, "instruments/violin_pizzicato01.ogg", amp=80))
        assert "note" not in again["tuning"]
        # giving the tuning for the new sample: no note either
        fixed = json.loads(srv.set_audiofile_sample(
            0, "instruments/flute01.ogg", base_note="A5"))
        assert "note" not in fixed["tuning"]
        reset = json.loads(srv.set_track_tuning(0, base_note=57, pitch_cents=0))
        assert reset["base_note_name"] == "A4" and reset["pitch_cents"] == 0

    def test_set_audiofile_sample_xml_level_tuning(self):
        from lmms_mcp.xml_parser import set_audiofile_sample
        root = create_empty_project()
        add_instrument_track(root, "S", instrument="audiofileprocessor")
        set_audiofile_sample(root, 0, "x.ogg", base_note=38, pitch_cents=18)
        track = self._inst(root)
        assert (track.get("basenote"), track.get("pitch")) == ("38", "18")
        with pytest.raises(ValueError):
            set_audiofile_sample(root, 0, "y.ogg", base_note=1000)
        assert track.find("instrument/audiofileprocessor").get("src") == "x.ogg"
        assert track.get("basenote") == "38"

    # --- save and load -----------------------------------------------------

    @pytest.mark.parametrize("suffix", [".mmp", ".mmpz"])
    def test_tuning_survives_save_and_load(self, tmp_path, suffix):
        srv, proj = self._new_project()
        srv.add_instrument_track("Cello", instrument="audiofileprocessor")
        srv.add_instrument_track("Zyn", instrument="zynaddsubfx")
        srv.add_instrument_track("Lead", instrument="tripleoscillator")
        srv.set_audiofile_sample(0, "instruments/cello01.ogg",
                                 base_note="D3", pitch_cents=17.5)
        srv.set_track_tuning(1, base_note="C4", pitch_cents=-40)
        srv.set_track_tuning(2, pitch_cents=-250)
        path = tmp_path / f"tuned{suffix}"
        proj.save(path)
        loaded = LMMSProject()
        loaded.load(path)
        srv.set_project(loaded)
        for index, (base, name, cents, written) in enumerate((
                (38, "D3", 17.5, 38), (48, "C4", -40, 36), (57, "A4", -250, 57))):
            report = json.loads(srv.set_track_tuning(index))
            assert report["base_note"] == base and report["base_note_name"] == name
            assert report["pitch_cents"] == cents
            assert report["written_basenote"] == written
        assert json.loads(srv.set_track_tuning(2))["pitchrange"] == 3
        track = json.loads(srv.get_track(0))
        assert track["base_note_name"] == "D3" and track["pitch_cents"] == 17.5

    # --- what the agent is told -------------------------------------------------

    def test_tool_is_offered_with_its_arguments(self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS)
        tools = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}
        schema = tools["set_track_tuning"].input_schema
        assert set(schema["properties"]) == {
            "track_index", "base_note", "pitch_cents"}
        assert schema["required"] == ["track_index"]
        props = tools["set_audiofile_sample"].input_schema["properties"]
        assert {"base_note", "pitch_cents"} <= set(props)

    def test_tool_description_explains_the_sample_example(self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS)
        tools = {t.name: t.description or ""
                 for t in asyncio.run(srv.mcp.list_tools())}
        flat = " ".join(tools["set_track_tuning"].split())
        assert 'a piano sample recorded at C4 gets base_note="C4"' in flat
        assert "key 48 (C4) plays it unchanged" in flat
        assert "+18 for a sample that is 18 cents flat" in flat
        assert "-1200 to +1200" in flat
        assert "set_audiofile_sample accepts the same two arguments" in flat
        # the limits name only offered instruments
        assert "zynaddsubfx follows the fine tune only up to +-100 cents" in flat
        assert "opulenz ignores the fine tune" in flat
        assert "kicker ignores the key" in flat
        sample = " ".join(tools["set_audiofile_sample"].split())
        assert 'base_note="C4"' in sample and "pitch_cents" in sample
        assert "key 57 (A4, the LMMS default)" in sample

    def test_tool_description_names_only_offered_instruments(self, server_for):
        import asyncio
        srv = server_for({"tripleoscillator", "monstro", "sf2player"})
        tools = {t.name: t.description or ""
                 for t in asyncio.run(srv.mcp.list_tools())}
        flat = " ".join(tools["set_track_tuning"].split())
        for word in ("zynaddsubfx", "opulenz", "kicker", "Limits:",
                     "set_audiofile_sample"):
            assert word not in flat, word
        assert "set_audiofile_sample" not in tools

    def test_instructions_explain_tuning(self):
        from lmms_mcp import server as srv
        text = " ".join(srv.build_instructions(
            srv._availability, "/tmp/p").split())
        assert "set_track_tuning(track_index, base_note, pitch_cents)" in text
        assert 'a piano sample recorded at C4 gets base_note="C4"' in text
        assert "pitch_cents=18" in text
        assert "set_audiofile_sample takes the same two arguments" in text
        assert "give set_audiofile_sample its base_note" in text

    def test_instructions_without_a_sampler_do_not_mention_samples(self):
        from lmms_mcp import lmms_app
        from lmms_mcp import server as srv
        av = lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS,
            {"tripleoscillator", "lb302"}, "1.2.2", windows=False)
        text = " ".join(srv.build_instructions(av, "/tmp/p").split())
        assert "set_track_tuning" in text
        for word in ("audiofileprocessor", "set_audiofile_sample",
                     "tuned sample", "pitch_cents=18"):
            assert word not in text, word

    def test_readme_documents_the_tool(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        assert "`set_track_tuning`" in readme
        flat = " ".join(readme.split())
        assert "base_note" in flat and "pitch_cents" in flat


# ──────────────────────────────────────────────────────────────────
# Instrument presets (.xpf)
# ──────────────────────────────────────────────────────────────────


def _xpf(name="Test Sound", plugin="tripleoscillator", settings='coarse0="-12"',
         *, basenote=57, pitch=0, pitchrange=1, vol=100, pan=0, compat=False,
         root="lmms-project", effects=(), ids=False, midiport=True,
         sound_parts=True, settings_xml=None, kind="instrumenttracksettings"):
    """An .xpf modelled on the files LMMS writes (see /usr/share/lmms/presets).

    `compat` writes the pre-1.1 layout (settings directly in
    <instrumenttrack>, no <instrument>); `effects` are (plugin name,
    attributes) pairs for the <fxchain>. The sound parts carry marker
    values: filter cutoff 484, chord 3, arpeggio 2.
    """
    model = ' id="5593744"' if ids else ""
    body = settings_xml if settings_xml is not None else (
        f"<{plugin} {settings}/>")
    instrument = body if compat else (
        f'<instrument name="{plugin}">{body}</instrument>')
    parts = (
        '<eldata fres="0.7" ftype="6" fcut="484" fwet="1">'
        f'<elvol lspd_denominator="4" sustain="1" amt="1" lamt="0.18"/>'
        f'<fcut value="12819"{model}/>'
        '</eldata>'
        '<chordcreator chord="3" chordrange="1" chord-enabled="1"/>'
        '<arpeggiator arp="2" arprange="1" arp-enabled="1" arpgate="100"/>'
    ) if sound_parts else ""
    port = (
        '<midiport inputcontroller="0" outputchannel="7" readable="1" '
        'writable="1" basevelocity="63"/>'
    ) if midiport else ""
    fx = "".join(
        f'<effect name="{n}" wet="1" on="1" {a}/>' for n, a in effects)
    chain = (
        f'<fxchain numofeffects="{len(effects)}" enabled="1">{fx}</fxchain>'
        if effects else '<fxchain numofeffects="0" enabled="0"/>'
    )
    return (
        '<?xml version="1.0"?>\n<!DOCTYPE lmms-project>\n'
        f'<{root} version="1.0" creator="LMMS" creatorversion="1.0.2" '
        f'type="{kind}">\n  <head/>\n'
        f'  <instrumenttracksettings muted="0" type="0" name="{name}" solo="0">\n'
        f'    <instrumenttrack pan="{pan}" fxch="9" pitchrange="{pitchrange}" '
        f'pitch="{pitch}" basenote="{basenote}" vol="{vol}">\n'
        f'      {instrument}{parts}{port}{chain}\n'
        '    </instrumenttrack>\n  </instrumenttracksettings>\n'
        f'</{root}>\n'
    )


class _PresetLab:
    """A fake preset folder, samples folder and a one-track project."""

    def __init__(self, tmp_path, monkeypatch):
        from lmms_mcp import instrument_presets, lmms_app
        from lmms_mcp import server as srv

        self.srv = srv
        self.root = tmp_path / "presets"
        self.root.mkdir()
        self.samples = tmp_path / "lmms_samples"
        self.samples.mkdir()
        monkeypatch.setenv("LMMS_PRESET_DIRS", str(self.root))
        monkeypatch.setattr(instrument_presets, "_system_roots", lambda: [])
        monkeypatch.setattr(
            lmms_app, "sample_search_dirs", lambda: [self.samples])
        self.proj = LMMSProject()
        self.proj.new()
        srv.set_project(self.proj)
        srv.add_mixer_channel("Bus 1")
        srv.add_mixer_channel("Bus 2")
        srv.add_instrument_track(
            "Lead", instrument="tripleoscillator", mixer_channel=2,
            volume=80, panning=20)
        srv.add_note(0, 60, 0, 48, 100)
        srv.add_note(0, 64, 48, 48, 100)

    def put(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def load(self, index, preset, **kwargs):
        return json.loads(
            self.srv.load_instrument_preset(index, preset, **kwargs))

    def inst_track(self, index=0):
        return find_tracks(self.proj.root)[index].find("instrumenttrack")

    def xml(self):
        return self.proj.get_xml_string()


@pytest.fixture
def lab(tmp_path, monkeypatch):
    return _PresetLab(tmp_path, monkeypatch)


class TestLoadInstrumentPreset:
    """load_instrument_preset: an .xpf replaces the sound of a track."""

    def test_replaces_the_sound_and_keeps_the_rest(self, lab):
        lab.put("TripleOscillator/Amazing Bubbles.xpf", _xpf(
            "Amazing Bubbles", vol=50, pan=-30, basenote=57))
        result = lab.load(0, "TripleOscillator/Amazing Bubbles")
        assert result["track_index"] == 0
        assert result["preset"] == "TripleOscillator/Amazing Bubbles"
        assert result["file"].endswith("Amazing Bubbles.xpf")
        assert result["instrument"] == "tripleoscillator"
        assert "lmms_plugin" not in result and "created" not in result
        assert result["replaced"] == [
            "instrument", "envelopes and filter", "chords", "arpeggio",
            "tuning"]
        assert result["kept"] == [
            "notes", "name", "volume", "panning", "mixer channel"]

        inst = lab.inst_track()
        assert inst.find("instrument/tripleoscillator").get("coarse0") == "-12"
        assert inst.find("eldata").get("fcut") == "484"
        assert inst.find("chordcreator").get("chord") == "3"
        assert inst.find("arpeggiator").get("arp") == "2"
        # The track keeps its mix, mixer channel, MIDI port and notes/name
        assert (inst.get("vol"), inst.get("pan"), get_track_mixer_channel(inst)) == (
            "80", "20", 2)
        assert inst.find("midiport").get("outputchannel") == "1"
        track = find_tracks(lab.proj.root)[0]
        assert track.get("name") == "Lead"
        assert len(track.findall("pattern/note")) == 2
        # ... and says that the preset was saved with another mix
        assert "volume 50 (the track keeps 80" in result["note"]
        assert "set_track_volume(0, 50)" in result["note"]
        assert "panning -30 (the track keeps 20" in result["note"]
        assert lab.proj.modified

    def test_tuning_is_reported_as_stored(self, lab):
        lab.put("Pads/Tuned.xpf", _xpf(
            plugin="audiofileprocessor", settings='src="x.ogg" amp="100"',
            basenote=48, pitch=18, pitchrange=1,
            settings_xml="<audiofileprocessor/>"))
        result = lab.load(0, "Pads/Tuned")
        assert "error" not in result, result
        tuning = result["tuning"]
        assert tuning["base_note"] == 48 and tuning["base_note_name"] == "C4"
        assert tuning["pitch_cents"] == 18 and tuning["written_basenote"] == 48
        assert "Key 48 (C4) plays the sample at its original pitch" in tuning["hint"]
        assert "instrument" not in tuning and "playable_keys" not in tuning
        inst = lab.inst_track()
        assert (inst.get("basenote"), inst.get("pitch")) == ("48", "18")

    def test_pitchrange_covers_the_fine_tune(self, lab):
        lab.put("A/B.xpf", _xpf(pitch=-250, pitchrange=1))
        lab.load(0, "A/B")
        assert lab.inst_track().get("pitchrange") == "3"
        assert lab.inst_track().get("pitch") == "-250"

    def test_rename_and_name(self, lab):
        lab.put("A/Warm.xpf", _xpf("Warm Pad"))
        assert lab.load(0, "A/Warm")["kept"][1] == "name"
        assert find_tracks(lab.proj.root)[0].get("name") == "Lead"
        result = lab.load(0, "A/Warm", rename=True)
        assert result["name"] == "Warm Pad"
        assert "name" not in result["kept"]
        assert find_tracks(lab.proj.root)[0].get("name") == "Warm Pad"
        assert lab.load(0, "A/Warm", name="Pad 2")["name"] == "Pad 2"
        assert find_tracks(lab.proj.root)[0].get("name") == "Pad 2"

    def test_creates_a_track(self, lab):
        lab.put("A/Warm.xpf", _xpf("Warm Pad", vol=60, pan=-10))
        result = lab.load(None, "A/Warm")
        assert result["track_index"] == 1 and result["created"] is True
        assert result["name"] == "Warm Pad"
        assert "kept" not in result
        inst = lab.inst_track(1)
        assert (inst.get("vol"), inst.get("pan")) == ("60", "-10")
        assert inst.find("eldata").get("fcut") == "484"
        assert find_tracks(lab.proj.root)[1].get("name") == "Warm Pad"
        named = lab.load(None, "A/Warm", name="Pad two")
        assert named["track_index"] == 2 and named["name"] == "Pad two"
        # the first track is untouched
        assert lab.inst_track(0).find("eldata").get("fcut") == "14000"

    def test_effect_chain_replaced_only_when_the_preset_has_effects(self, lab):
        lab.srv.add_effect("track", 0, "reverbsc")
        lab.put("A/Dry.xpf", _xpf("Dry"))
        dry = lab.load(0, "A/Dry")
        assert "kept" in dry and dry["kept"][-1] == "effects"
        assert [e.get("name") for e in lab.inst_track().findall(
            "fxchain/effect")] == ["reverbsc"]
        lab.put("A/Wet.xpf", _xpf("Wet", effects=[
            ("delay", ""), ("amplifier", "")]))
        wet = lab.load(0, "A/Wet")
        assert "effects (2)" in wet["replaced"]
        assert "effects" not in wet["kept"]
        chain = lab.inst_track().find("fxchain")
        assert [e.get("name") for e in chain.findall("effect")] == [
            "delay", "amplifier"]
        assert chain.get("numofeffects") == "2"

    def test_effects_the_install_lacks_are_left_out(self, lab, monkeypatch):
        from lmms_mcp import lmms_app
        srv = lab.srv
        installed = (set(srv.KNOWN_INSTRUMENTS)
                     | set(srv.effects_mod.KNOWN_EFFECTS)) - {"delay"}
        monkeypatch.setattr(srv, "_availability", lmms_app.build_availability(
            srv.KNOWN_INSTRUMENTS, srv.effects_mod.KNOWN_EFFECTS, installed,
            "1.2.2", windows=False))
        lab.put("A/Wet.xpf", _xpf(effects=[("delay", ""), ("amplifier", "")]))
        result = lab.load(0, "A/Wet")
        assert result["warnings"] == [
            "Left out effect(s) the installed LMMS lacks: delay."]
        chain = lab.inst_track().find("fxchain")
        assert [e.get("name") for e in chain.findall("effect")] == ["amplifier"]
        assert chain.get("numofeffects") == "1"

    def test_old_layout_without_instrument_element(self, lab):
        lab.put("TripleOscillator/Old.xpf", _xpf(
            "Old", compat=True, root="multimedia-project"))
        result = lab.load(0, "TripleOscillator/Old")
        assert "error" not in result, result
        inst = lab.inst_track().find("instrument")
        assert inst.get("name") == "tripleoscillator"
        assert [c.tag for c in inst] == ["tripleoscillator"]
        # LMMS-1.0-style roots read as well
        lab.put("X/Y.xpf", _xpf(root="multimediaproject"))
        assert "error" not in lab.load(0, "X/Y")

    def test_model_ids_are_not_copied(self, lab):
        lab.put("A/Ids.xpf", _xpf(ids=True))
        lab.load(0, "A/Ids")
        fcut = lab.inst_track().find("eldata/fcut")
        assert fcut.get("value") == "12819" and fcut.get("id") is None

    def test_result_saves_and_reloads(self, lab, tmp_path):
        lab.put("A/B.xpf", _xpf(settings='coarse0="7"'))
        lab.load(0, "A/B")
        path = lab.proj.save(tmp_path / "out.mmp")
        again = LMMSProject()
        again.load(path)
        inst = find_tracks(again.root)[0].find("instrumenttrack")
        assert inst.find("instrument/tripleoscillator").get("coarse0") == "7"

    # --- instrument availability and names --------------------------------

    @pytest.fixture
    def linux122(self, lab, monkeypatch):
        monkeypatch.setattr(
            lab.srv, "_availability",
            TestInstrumentPluginNames._availability(
                LINUX_122_LIBS + ("libwatsyn.so",)))
        return lab

    def test_opl2_preset_maps_to_opulenz(self, linux122):
        lab = linux122
        lab.put("OpulenZ/Bells.xpf", _xpf(
            "Bells", plugin="OPL2", settings='op1_mul="1" op2_mul="1"'))
        result = lab.load(None, "OpulenZ/Bells")
        assert result["instrument"] == "opulenz"
        assert result["lmms_plugin"] == "OPL2"
        inst = lab.inst_track(1).find("instrument")
        assert inst.get("name") == "OPL2" and inst[0].tag == "OPL2"

    def test_preset_names_follow_the_installed_library(self, linux122):
        lab = linux122
        lab.put("A/Opulenz.xpf", _xpf(plugin="opulenz", settings='op2_mul="1"'))
        lab.put("A/Gb.xpf", _xpf(plugin="freeboy", settings='sweep="0"'))
        lab.put("A/Custom.xpf", _xpf(plugin="mycustomsynth", settings='a="1"'))
        opl = lab.load(None, "A/Opulenz")
        assert (opl["instrument"], opl["lmms_plugin"]) == ("opulenz", "OPL2")
        gb = lab.load(None, "A/Gb")
        assert (gb["instrument"], gb["lmms_plugin"]) == ("freeboy", "papu")
        assert lab.inst_track(2).find("instrument").get("name") == "papu"
        custom = lab.load(None, "A/Custom")
        assert custom["instrument"] == "mycustomsynth"
        assert custom["lmms_plugin"] == "MyCustomSynth"

    def test_unavailable_instrument_is_refused(self, linux122):
        lab = linux122
        lab.put("A/Lead.xpf", _xpf(plugin="xpressive", settings='a="1"'))
        before = lab.xml()
        result = lab.load(0, "A/Lead")
        assert "'xpressive' is not available" in result["error"]
        assert result["alternative"] == "watsyn"
        assert "list_instrument_presets" in result["error"]
        assert "tripleoscillator" in result["available_instruments"]
        assert lab.load(None, "A/Lead")["error"] == result["error"]
        assert lab.xml() == before

    def test_unknown_plugin_without_alternative(self, linux122):
        lab = linux122
        lab.put("A/Ghost.xpf", _xpf(plugin="ghostsynth", settings='a="1"'))
        result = lab.load(0, "A/Ghost")
        assert "'ghostsynth' is not installed" in result["error"]
        assert "alternative" not in result

    def test_vst_preset_is_refused(self, lab):
        lab.put("A/Vst.xpf", _xpf(plugin="vestige", settings='plugin="x.dll"'))
        assert "add_vst_track" in lab.load(0, "A/Vst")["error"]

    # --- pitch calibration of plugins LMMS plays off pitch -----------------

    def test_default_base_note_is_calibrated_for_an_octave_low_patch(self, linux122):
        lab = linux122
        lab.put("OpulenZ/Low.xpf", _xpf(
            plugin="OPL2", settings='op1_mul="0" op2_mul="1" op1_lvl="62"'))
        result = lab.load(None, "OpulenZ/Low")
        assert lab.inst_track(1).get("basenote") == "45"
        assert result["tuning"]["base_note"] == 57
        assert result["tuning"]["written_basenote"] == 45
        assert "key 57 = A4 = 440 Hz" in result["note"]
        assert "12 semitones below" in result["note"]
        assert result["playable_keys"] == [0, 90]

    def test_default_base_note_is_kept_for_a_patch_at_pitch(self, linux122):
        lab = linux122
        lab.put("OpulenZ/Right.xpf", _xpf(
            plugin="OPL2", settings='op1_mul="1" op2_mul="1"'))
        result = lab.load(None, "OpulenZ/Right")
        assert lab.inst_track(1).get("basenote") == "57"
        assert result["tuning"]["base_note"] == 57
        assert "note" not in result
        assert result["playable_keys"] == [0, 102]

    def test_other_base_notes_are_kept_exactly(self, linux122):
        lab = linux122
        lab.put("OpulenZ/Odd.xpf", _xpf(
            plugin="OPL2", basenote=60, settings='op1_mul="0" op2_mul="1"'))
        result = lab.load(None, "OpulenZ/Odd")
        assert lab.inst_track(1).get("basenote") == "60"
        assert result["tuning"]["written_basenote"] == 60
        assert "note" not in result

    def test_zynaddsubfx_preset_is_calibrated(self, linux122):
        lab = linux122
        zyn = ('<zynaddsubfx><ZynAddSubFX-data><MASTER><PART><INSTRUMENT/>'
               '</PART></MASTER></ZynAddSubFX-data></zynaddsubfx>')
        lab.put("Zyn/Pad.xpf", _xpf(plugin="zynaddsubfx", settings_xml=zyn))
        result = lab.load(None, "Zyn/Pad")
        assert lab.inst_track(1).get("basenote") == "45"
        assert result["tuning"]["base_note"] == 57
        assert "key 57 = A4 = 440 Hz" in result["note"]

    # --- files the preset plays ----------------------------------------------

    def _sample_preset(self, lab, src, **kwargs):
        return lab.put("AudioFileProcessor/Piano.xpf", _xpf(
            "Piano", plugin="audiofileprocessor",
            settings=f'src="{src}" amp="100" looped="0"', basenote=48,
            **kwargs))

    def test_absolute_sample_that_exists(self, lab, tmp_path):
        sample = tmp_path / "elsewhere" / "piano.ogg"
        sample.parent.mkdir()
        sample.write_bytes(b"x")
        self._sample_preset(lab, sample)
        result = lab.load(0, "Piano")
        assert "error" not in result, result
        afp = lab.inst_track().find("instrument/audiofileprocessor")
        assert afp.get("src") == str(sample)
        assert "warnings" not in result

    def test_relative_sample_found_the_way_lmms_finds_it(self, lab):
        (lab.samples / "drums").mkdir()
        (lab.samples / "drums" / "kick.ogg").write_bytes(b"x")
        self._sample_preset(lab, "drums/kick.ogg")
        result = lab.load(0, "Piano")
        assert "error" not in result, result
        afp = lab.inst_track().find("instrument/audiofileprocessor")
        assert afp.get("src") == "drums/kick.ogg"
        assert "absolute path" not in result.get("note", "")

    def test_relative_sample_beside_the_preset_gets_an_absolute_path(self, lab):
        beside = lab.put("AudioFileProcessor/piano.ogg", "x")
        self._sample_preset(lab, "piano.ogg")
        result = lab.load(0, "Piano")
        assert "error" not in result, result
        afp = lab.inst_track().find("instrument/audiofileprocessor")
        assert afp.get("src") == str(beside.resolve())
        assert "absolute path" in result["note"]
        assert str(beside.resolve()) in result["note"]

    def test_missing_relative_sample_is_refused(self, lab):
        self._sample_preset(lab, "gone/piano.ogg")
        before = lab.xml()
        result = lab.load(0, "Piano")
        assert "'gone/piano.ogg'" in result["error"]
        assert "silent" in result["error"] and "nothing was changed" in result["error"]
        assert str(lab.samples) in result["error"]
        assert lab.xml() == before
        assert len(find_tracks(lab.proj.root)) == 1
        # creating a track refuses just the same and leaves no track behind
        assert "error" in lab.load(None, "Piano")
        assert len(find_tracks(lab.proj.root)) == 1

    def test_missing_absolute_sample_is_refused(self, lab, tmp_path):
        self._sample_preset(lab, tmp_path / "nowhere" / "piano.ogg")
        result = lab.load(0, "Piano")
        assert str(tmp_path / "nowhere" / "piano.ogg") in result["error"]
        assert "looked in" not in result["error"]

    def test_unverifiable_relative_sample_only_warns(self, lab, monkeypatch):
        from lmms_mcp import lmms_app
        monkeypatch.setattr(lmms_app, "sample_search_dirs", lambda: [])
        self._sample_preset(lab, "drums/kick.ogg")
        result = lab.load(0, "Piano")
        assert "error" not in result
        assert "could not be checked" in result["warnings"][0]

    # --- refusals that change nothing ---------------------------------------

    def test_non_instrument_track_is_refused(self, lab):
        lab.srv.add_sample_track("Vox")
        lab.put("A/B.xpf", _xpf())
        before = lab.xml()
        result = lab.load(1, "A/B")
        assert "Track 1 is a sample track, not an instrument track" in result["error"]
        assert "track_index=null" in result["error"]
        assert lab.xml() == before

    @pytest.mark.parametrize("index", [5, -1, True, "0"])
    def test_bad_track_index(self, lab, index):
        lab.put("A/B.xpf", _xpf())
        assert "error" in lab.load(index, "A/B")

    def test_no_tracks_at_all(self, lab):
        lab.srv.remove_track(0)
        lab.put("A/B.xpf", _xpf())
        assert "no tracks yet" in lab.load(0, "A/B")["error"]
        assert lab.load(None, "A/B")["track_index"] == 0

    @pytest.mark.parametrize("text,fragment", [
        ("not xml at all", "not valid XML"),
        ("<song/>", "not an LMMS file"),
        (_xpf(kind="song"), "type 'song', not an instrument preset"),
        (_xpf(kind="channelsettings"), "pre-1.0 channel format"),
        ('<lmms-project type="instrumenttracksettings"><head/></lmms-project>',
         "no <instrumenttrack>"),
        (_xpf(plugin="", settings_xml=""), "contains no instrument"),
    ])
    def test_unusable_files_are_refused(self, lab, text, fragment):
        lab.put("Bad/File.xpf", text)
        before = lab.xml()
        result = lab.load(0, "Bad/File")
        assert fragment in result["error"]
        assert lab.xml() == before


class TestInstrumentPresetLookup:
    """Finding the preset a request names."""

    @pytest.fixture
    def library(self, lab):
        lab.put("TripleOscillator/AmazingBubbles.xpf", _xpf("Amazing Bubbles"))
        lab.put("TripleOscillator/Wobble Bass.xpf", _xpf("Wobble"))
        lab.put("Kicker/Wobble Bass.xpf", _xpf("Kick Wobble", plugin="kicker"))
        lab.put("OpulenZ/Bells.xpf", _xpf(
            plugin="OPL2", settings='op1_mul="1" op2_mul="1"'))
        lab.put("Flat.xpf", _xpf("Flat"))
        return lab

    def test_label_forms(self, library):
        for request in (
            "TripleOscillator/AmazingBubbles",
            "TripleOscillator/AmazingBubbles.xpf",
            "tripleoscillator/amazing bubbles",
            "Amazing Bubbles", "amazing_bubbles", "AMAZING-BUBBLES",
        ):
            result = library.load(0, request)
            assert result["preset"] == "TripleOscillator/AmazingBubbles", request
            assert "other_matches" not in result

    def test_name_matching_the_end_of_a_name(self, library):
        assert library.load(0, "Bubbles")["preset"] == (
            "TripleOscillator/AmazingBubbles")

    def test_instrument_name_may_replace_the_folder(self, library):
        # the folder is OpulenZ, the instrument opulenz / OPL2
        for request in ("OpulenZ/Bells", "opulenz/Bells", "opl2/Bells"):
            assert library.load(None, request)["preset"] == "OpulenZ/Bells", request
        assert library.load(None, "tripleoscillator/Bells").get("error")

    def test_flat_file_and_absolute_path(self, library):
        assert library.load(0, "Flat")["preset"] == "Flat"
        path = library.root / "TripleOscillator" / "AmazingBubbles.xpf"
        for given in (str(path), str(path)[:-4]):
            result = library.load(0, given)
            assert result["file"] == str(path)
            assert result["preset"] == "TripleOscillator/AmazingBubbles"

    def test_ambiguous_name_lists_other_matches(self, library):
        result = library.load(0, "Wobble Bass")
        assert result["preset"] == "Kicker/Wobble Bass" \
            or result["preset"] == "TripleOscillator/Wobble Bass"
        other = {"Kicker/Wobble Bass", "TripleOscillator/Wobble Bass"} \
            - {result["preset"]}
        assert set(result["other_matches"]) == other
        assert "matched several presets" in result["note"]
        # a folder picks one
        picked = library.load(0, "Kicker/Wobble Bass")
        assert picked["preset"] == "Kicker/Wobble Bass"
        assert "other_matches" not in picked

    def test_not_found_suggests_close_names(self, library):
        before = library.xml()
        result = library.load(0, "Amazin Bubles")
        assert "not found" in result["error"]
        assert "TripleOscillator/AmazingBubbles" in result["suggestions"]
        assert "Did you mean" in result["error"]
        assert "list_instrument_presets" in result["error"]
        nothing = library.load(0, "qqqqzzzz")
        assert nothing["suggestions"] == []
        assert library.xml() == before

    def test_wrong_folder_is_not_ignored(self, library):
        result = library.load(0, "Kicker/Amazing Bubbles")
        assert "not found" in result["error"]

    def test_absolute_path_that_does_not_exist(self, library, tmp_path):
        result = library.load(0, str(tmp_path / "nope.xpf"))
        assert "does not exist" in result["error"]

    def test_zyn_preset_points_to_load_zyn_preset(self, library):
        assert "load_zyn_preset" in library.load(0, "Bass/0001-Deep.xiz")["error"]

    def test_empty_request(self, library):
        assert "must name a preset" in library.load(0, "  ")["error"]

    def test_user_folder_comes_before_factory_folder(self, lab, tmp_path, monkeypatch):
        from lmms_mcp import instrument_presets
        factory = tmp_path / "factory"
        monkeypatch.setattr(
            instrument_presets, "_system_roots", lambda: [factory])
        (factory / "TripleOscillator").mkdir(parents=True)
        (factory / "TripleOscillator" / "Same.xpf").write_text(
            _xpf(settings='coarse0="1"'))
        lab.put("TripleOscillator/Same.xpf", _xpf(settings='coarse0="2"'))
        result = lab.load(0, "Same")
        assert result["file"] == str(lab.root / "TripleOscillator" / "Same.xpf")
        assert "other_matches" not in result
        assert lab.inst_track().find("instrument/tripleoscillator").get(
            "coarse0") == "2"

    def test_env_var_accepts_several_folders(self, lab, tmp_path, monkeypatch):
        import os
        second = tmp_path / "second"
        (second / "Mine").mkdir(parents=True)
        (second / "Mine" / "Tuned.xpf").write_text(_xpf("Tuned"))
        monkeypatch.setenv(
            "LMMS_PRESET_DIRS",
            f"{lab.root}{os.pathsep}{second}{os.pathsep}{tmp_path / 'gone'}")
        assert lab.load(0, "Mine/Tuned")["preset"] == "Mine/Tuned"


class TestListInstrumentPresets:
    def _list(self, lab, *args):
        return json.loads(lab.srv.list_instrument_presets(*args))

    @pytest.fixture
    def library(self, lab, monkeypatch):
        monkeypatch.setattr(
            lab.srv, "_availability",
            TestInstrumentPluginNames._availability())
        for i in range(7):
            lab.put(f"TripleOscillator/Tri {i}.xpf", _xpf(f"Tri {i}"))
        lab.put("OpulenZ/Bells.xpf", _xpf(plugin="OPL2", settings='op2_mul="1"'))
        lab.put("Mine/Own.xpf", _xpf(plugin="opulenz", settings='op2_mul="1"'))
        lab.put("Slicer/Chop.xpf", _xpf(plugin="xpressive", settings='a="1"'))
        lab.put("Broken/Bad.xpf", "garbage")
        lab.put("Old/Chan.xpf", _xpf(kind="channelsettings"))
        return lab

    def test_overview_has_counts_and_a_few_examples(self, library):
        result = self._list(library)
        assert result["preset_dirs"] == [str(library.root)]
        assert result["total"] == 9
        tri = result["instruments"]["tripleoscillator"]
        assert tri["count"] == 7 and len(tri["examples"]) == 5
        # OPL2 and opulenz presets are the same instrument
        assert result["instruments"]["opulenz"]["count"] == 2
        assert "xpressive" not in result["instruments"]
        assert list(result["instruments"]) == ["tripleoscillator", "opulenz"]
        assert result["skipped"] == {
            "instrument_not_available": 1, "unreadable": 2}
        assert "list_instrument_presets(instrument)" in result["hint"]

    def test_one_instrument_lists_all_its_presets(self, library):
        for given in ("opulenz", "OPL2", "OpulenZ"):
            result = self._list(library, given)
            assert result["instrument"] == "opulenz"
            assert result["count"] == 2
            assert result["presets"] == ["Mine/Own", "OpulenZ/Bells"]
        tri = self._list(library, "tripleoscillator")
        assert tri["presets"][0] == "TripleOscillator/Tri 0"
        assert "truncated" not in tri
        # listed names load
        assert library.load(None, "Mine/Own")["instrument"] == "opulenz"

    def test_unknown_or_unavailable_instrument(self, library):
        for given in ("xpressive", "nonsense"):
            result = self._list(library, given)
            assert "No presets found" in result["error"]
            assert result["instruments_with_presets"] == [
                "tripleoscillator", "opulenz"]

    def test_listing_is_capped(self, lab):
        for i in range(205):
            lab.put(f"TripleOscillator/P{i:03d}.xpf", _xpf())
        result = self._list(lab, "tripleoscillator")
        assert result["count"] == 205 and len(result["presets"]) == 200
        assert result["truncated"] is True

    def test_no_preset_folder(self, lab, monkeypatch, tmp_path):
        monkeypatch.setenv("LMMS_PRESET_DIRS", str(tmp_path / "missing"))
        result = self._list(lab)
        assert "No instrument preset folder" in result["error"]
        assert "LMMS_PRESET_DIRS" in result["hint"]

    def test_folder_without_presets(self, lab):
        result = self._list(lab)
        assert result["total"] == 0 and result["instruments"] == {}


class TestInstrumentPresetTools:
    def test_tools_are_registered_with_their_arguments(self, server_for):
        import asyncio
        srv = server_for(LMMS_122_PLUGINS)
        tools = {t.name: t for t in asyncio.run(srv.mcp.list_tools())}
        load = tools["load_instrument_preset"]
        assert set(load.input_schema["properties"]) == {
            "track_index", "preset", "name", "rename"}
        assert set(load.input_schema["required"]) == {"track_index", "preset"}
        assert "list_instrument_presets" in tools
        assert "set_track_tuning" in load.description

    def test_instructions_mention_presets(self):
        from lmms_mcp import server as srv
        text = " ".join(srv.build_instructions(
            srv._availability, "/tmp/p").split())
        assert "load_instrument_preset(track_index" in text
        assert "list_instrument_presets" in text

    def test_readme_documents_the_tools(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        assert "`load_instrument_preset`" in readme
        assert "`list_instrument_presets`" in readme
        assert "LMMS_PRESET_DIRS" in readme


class TestOpl2PatchPitch:
    """OPL2's pitch depends on the patch (operator multipliers)."""

    @staticmethod
    def _offset(**attrs):
        from lmms_mcp import pitch
        settings = ET.Element("OPL2", {k: str(v) for k, v in attrs.items()})
        return pitch.opl2_patch_offset(settings)

    def test_default_patch_is_an_octave_low(self):
        from lmms_mcp import pitch
        assert pitch.opl2_patch_offset(None) == 12
        assert self._offset() == 12
        assert pitch.PITCH_OFFSET_SEMITONES["opl2"] == 12

    @pytest.mark.parametrize("attrs,offset", [
        (dict(op1_mul=1, op2_mul=1), 0),       # measured 441 Hz at key 57
        (dict(op1_mul=1, op2_mul=2), 0),       # Bagpipe
        (dict(op1_mul=7, op2_mul=1), 0),       # Bells
        (dict(op1_mul=0, op2_mul=1, op1_lvl=62), 12),   # Funky
        (dict(op1_mul=0, op2_mul=0), 12),
        (dict(op1_mul=2, op2_mul=2), -12),     # measured 882 Hz
        (dict(op1_mul=2, op2_mul=4), -12),
        (dict(op1_mul=3, op2_mul=3), -19),     # a twelfth up
        (dict(op2_mul=1), 12),                 # op1 at its default x0.5
        (dict(op1_mul=1), 0),                  # op2 at its default x1
    ])
    def test_offsets(self, attrs, offset):
        assert self._offset(**attrs) == offset

    def test_a_quiet_half_speed_operator_does_not_count(self):
        # Bubbly_days: op1 x0.5 at level 46 sounds at the key's pitch
        assert self._offset(op1_mul=0, op2_mul=1, op1_lvl=46) == 0
        assert self._offset(op1_mul=0, op2_mul=1, op1_lvl=50) == 12
        assert self._offset(op1_mul=0, op2_mul=2, op1_lvl=30) == -12

    def test_bad_values_fall_back_to_the_defaults(self):
        assert self._offset(op1_mul="x", op2_mul="y") == 12
        # out-of-range multipliers are clamped to 0-15
        assert self._offset(op1_mul=99, op2_mul=-4) == 12

    def test_track_offset_follows_the_patch(self):
        from lmms_mcp import pitch
        root = create_empty_project()
        add_instrument_track(root, "O", instrument="OPL2")
        inst_track = find_tracks(root)[0].find("instrumenttrack")
        assert pitch.track_pitch_offset(inst_track) == 12
        assert inst_track.get("basenote") == "45"
        settings = inst_track.find("instrument")[0]
        settings.set("op1_mul", "1")
        assert pitch.track_pitch_offset(inst_track) == 0
        assert pitch.get_base_note(inst_track) == 45
        pitch.set_base_note(inst_track, 57)
        assert inst_track.get("basenote") == "57"


# ──────────────────────────────────────────────────────────────────
# MIXER NAMES: fxmixer/fxchannel/fxch (LMMS 1.2.x) vs mixer/mixerchannel/mixch
# ──────────────────────────────────────────────────────────────────

def _mixer_project_xml(names, vol=80):
    """A project written like LMMS writes it, with the given MixerNames.

    Modelled on the LMMS 1.2.2 demo projects (/usr/share/lmms/projects):
    a bass track on channel 1, a sample track, and a mixer with Master and
    a "drums" channel that has an effect and a <send> to Master.
    """
    return f"""<?xml version="1.0"?>
<!DOCTYPE lmms-project>
<lmms-project version="1.0" creator="LMMS" creatorversion="1.2.0" type="song">
  <head bpm="120" mastervol="100" masterpitch="0" timesig_numerator="4" timesig_denominator="4"/>
  <song>
    <trackcontainer type="song" visible="1" width="600" height="300" x="5" y="5" maximized="0" minimized="0">
      <track muted="0" type="0" name="Bass" solo="0">
        <instrumenttrack pan="0" {names.track_attr}="1" usemasterpitch="1" pitchrange="1" pitch="0" basenote="57" vol="{vol}">
          <instrument name="tripleoscillator"><tripleoscillator/></instrument>
          <fxchain numofeffects="0" enabled="0"/>
        </instrumenttrack>
      </track>
      <track muted="0" type="2" name="Loop" solo="0">
        <sampletrack vol="100" pan="0"><fxchain numofeffects="0" enabled="0"/></sampletrack>
      </track>
    </trackcontainer>
    <{names.mixer} visible="1" width="561" height="332" x="5" y="310" maximized="0" minimized="0">
      <{names.channel} num="0" muted="0" volume="1.2" name="Master" soloed="0">
        <fxchain numofeffects="0" enabled="0"/>
      </{names.channel}>
      <{names.channel} num="1" muted="0" volume="0.65" name="drums" soloed="0">
        <fxchain numofeffects="1" enabled="1">
          <effect name="delay" on="1" wet="1" gate="0" autoquit="1"><delaycontrols/></effect>
        </fxchain>
        <send channel="0" amount="1"/>
      </{names.channel}>
    </{names.mixer}>
  </song>
</lmms-project>
"""


def _spellings(root):
    """(element tags, attribute names) used anywhere in a project."""
    tags = {e.tag for e in root.iter()}
    attrs = {a for e in root.iter() for a in e.attrib}
    return tags, attrs


def _assert_only_spelling(root, names, tracks=True):
    """The project uses `names` and none of the other spelling.

    tracks=False for a project without tracks (only the mixer is checked).
    """
    from lmms_mcp import xml_parser
    tags, attrs = _spellings(root)
    assert {names.mixer, names.channel} <= tags
    if tracks:
        assert names.track_attr in attrs
    for other in xml_parser.ALL_MIXER_NAMES:
        if other != names:
            assert not {other.mixer, other.channel} & tags
            assert other.track_attr not in attrs


def find_mixer_channel_element(root, num):
    """The mixer channel element with the given number."""
    from lmms_mcp import xml_parser
    for channel in xml_parser.find_mixer_channels(root):
        if channel.get("num") == str(num):
            return channel
    raise AssertionError(f"no mixer channel {num}")


@pytest.fixture
def installed(monkeypatch):
    """installed(version): the server runs against that LMMS version
    (None: no LMMS detected) and no project exists yet. Returns the server
    module."""
    import dataclasses
    from lmms_mcp import project as project_mod
    from lmms_mcp import server as srv

    def use(version):
        monkeypatch.setattr(
            srv, "_availability",
            dataclasses.replace(srv._availability, version=version))
        monkeypatch.setattr(project_mod, "_current_project", None)
        return srv

    return use


class TestMixerNames:
    """Projects are written with the mixer names the installed LMMS reads.

    LMMS 1.2.2 knows only fxmixer/fxchannel/fxch (its binary and all its
    demo projects), 1.3 mixer/mixerchannel/mixch; a name LMMS does not know
    is ignored, so a track routed with mixch played on Master in 1.2.2.
    """

    VERSIONS = [("1.2.2", "FX_MIXER"), ("1.2.0", "FX_MIXER"),
                ("1.3.0", "MIXER"), ("1.3.0-alpha.1", "MIXER"),
                (None, "FX_MIXER")]

    @staticmethod
    def _names(attr):
        from lmms_mcp import xml_parser
        return getattr(xml_parser, attr)

    # --- which names a version reads --------------------------------

    @pytest.mark.parametrize("version,attr", VERSIONS + [
        ("1.10.0", "MIXER"), ("2.0.0", "MIXER"), ("1.1.3", "FX_MIXER"),
        ("garbage", "FX_MIXER"), ("", "FX_MIXER")])
    def test_names_by_version(self, version, attr):
        from lmms_mcp import xml_parser
        assert xml_parser.mixer_names_for_version(version) == self._names(attr)

    def test_the_two_spellings(self):
        from lmms_mcp import xml_parser
        fx, new = xml_parser.FX_MIXER, xml_parser.MIXER
        assert (fx.mixer, fx.channel, fx.track_attr) == (
            "fxmixer", "fxchannel", "fxch")
        assert (new.mixer, new.channel, new.track_attr) == (
            "mixer", "mixerchannel", "mixch")
        # 1.2.x sample tracks ignore a channel, 1.3 sample tracks have one
        assert fx.sample_track_attr is None
        assert new.sample_track_attr == "mixch"

    def test_without_a_version_the_names_every_lmms_loads(self):
        # 1.3 upgrades a file that says it was created by LMMS 1.2
        from lmms_mcp import xml_parser
        root = create_empty_project()
        assert root.get("creatorversion") == "1.2.0"
        assert xml_parser.DEFAULT_MIXER_NAMES == xml_parser.FX_MIXER
        _assert_only_spelling(root, xml_parser.FX_MIXER, tracks=False)

    @pytest.mark.parametrize("version,attr", VERSIONS)
    def test_new_project_mixer(self, version, attr):
        names = self._names(attr)
        root = create_empty_project(lmms_version=version)
        mixer = root.find(f"song/{names.mixer}")
        assert mixer is not None
        channels = list(mixer)
        assert [(c.tag, c.get("num"), c.get("name")) for c in channels] == [
            (names.channel, "0", "Master")]
        _assert_only_spelling(root, names, tracks=False)

    # --- tracks, channels, effects, automation ---------------------

    @pytest.mark.parametrize("version,attr", VERSIONS)
    def test_tools_write_the_names_of_the_installed_lmms(
            self, installed, version, attr):
        srv = installed(version)
        names = self._names(attr)
        srv.create_project()
        assert json.loads(srv.add_mixer_channel("Lead bus"))["channel_num"] == 1
        assert "error" not in json.loads(srv.add_instrument_track(
            "Lead", "tripleoscillator", mixer_channel=1))
        srv.add_pattern_track("Beat")
        root = srv.get_project().root

        _assert_only_spelling(root, names)
        track = find_tracks(root)[0].find("instrumenttrack")
        assert track.get(names.track_attr) == "1"
        channel = find_mixer_channel_element(root, 1)
        assert channel.tag == names.channel and channel.get("name") == "Lead bus"
        # explicit send: LMMS drops the implicit connection to Master
        assert channel.find("send").attrib == {"channel": "0", "amount": "1"}
        # the pattern track's inner kicker is routed to Master the same way
        inner = find_tracks(root)[1].find(
            "bbtrack/trackcontainer/track/instrumenttrack")
        assert inner.get(names.track_attr) == "0"

    @pytest.mark.parametrize("version,attr", VERSIONS)
    def test_mixer_tools_work_with_the_names(self, installed, version, attr):
        srv = installed(version)
        srv.create_project()
        srv.add_mixer_channel("Lead bus")
        srv.add_instrument_track("Lead", mixer_channel=1)

        assert "error" not in json.loads(srv.set_mixer_channel_volume(1, 0.0))
        assert "error" not in json.loads(srv.set_mixer_channel_name(1, "Keys"))
        assert "error" in json.loads(srv.set_mixer_channel_volume(7, 0.5))
        channels = json.loads(srv.get_mixer_channels())
        assert channels["count"] == 2
        assert [(c["num"], c["name"], c["volume"]) for c in channels["channels"]
                ] == [(0, "Master", 1.0), (1, "Keys", 0.0)]
        assert channels["channels"][1]["sends"] == [
            {"channel": 0, "amount": 1.0}]

        info = json.loads(srv.get_project_info())
        assert info["tracks"][0]["mixer_channel"] == 1
        assert json.loads(srv.get_mixer_channels())["count"] == len(
            json.loads(srv.get_project_info())["mixer_channels"])

        # effect chain of a mixer channel
        added = json.loads(srv.add_effect("mixer", 1, "bitcrush"))
        assert "error" not in added
        chain = json.loads(srv.get_effect_chain("mixer", 1))
        assert [e["name"] for e in chain["effects"]] == ["bitcrush"]
        assert json.loads(srv.get_mixer_channels())["channels"][1][
            "effects"] == ["bitcrush"]
        assert "error" not in json.loads(srv.toggle_effect(
            "mixer", 1, "bitcrush", False))
        assert "error" in json.loads(srv.get_effect_chain("mixer", 5))

        # automation of a mixer channel's volume
        result = json.loads(srv.add_automation(
            "mixer", "volume",
            [{"bar": 0, "value": 1.0}, {"bar": 2, "value": 0.5}],
            target_index=1))
        assert "error" not in result, result
        channel = find_mixer_channel_element(srv.get_project().root, 1)
        assert channel.find("volume").get("id")

    @pytest.mark.parametrize("version,attr", VERSIONS)
    def test_first_project_is_made_for_the_installed_lmms(
            self, installed, version, attr):
        # no create_project: the default project is written for that LMMS
        srv = installed(version)
        srv.add_instrument_track("Lead")
        _assert_only_spelling(
            srv.get_project().root, self._names(attr))

    @pytest.mark.parametrize("version,attr", VERSIONS)
    def test_save_and_load_keep_the_names(
            self, installed, tmp_path, version, attr):
        srv = installed(version)
        names = self._names(attr)
        srv.create_project()
        srv.add_mixer_channel("Lead bus")
        srv.add_instrument_track("Lead", mixer_channel=1)
        for suffix in (".mmpz", ".mmp"):
            path = tmp_path / f"song{suffix}"
            srv.save_project(str(path))
            srv.create_project()  # forget the project, then read the file
            assert "error" not in json.loads(srv.load_project(str(path)))
            root = srv.get_project().root
            _assert_only_spelling(root, names)
            info = json.loads(srv.get_project_info())
            assert info["tracks"][0]["mixer_channel"] == 1
            assert [m["name"] for m in info["mixer_channels"]] == [
                "Master", "Lead bus"]

    # --- sample tracks -----------------------------------------------

    def test_sample_track_cannot_be_routed_in_lmms_12(self, installed):
        # verified with LMMS 1.2.2: a sample track ignores fxch and plays
        # on Master, so asking for another channel must not pretend to work
        srv = installed("1.2.2")
        srv.create_project()
        srv.add_mixer_channel("Loops")
        before = srv.get_project().get_xml_string()
        result = json.loads(srv.add_sample_track("Loop", mixer_channel=1))
        assert "Master" in result["error"] and "1.2" in result["error"]
        assert srv.get_project().get_xml_string() == before

        result = json.loads(srv.add_sample_track("Loop", mixer_channel=0))
        assert "error" not in result
        sample = find_tracks(srv.get_project().root)[0].find("sampletrack")
        assert set(sample.attrib) == {"vol", "pan"}

    def test_sample_track_refused_without_a_known_lmms_too(self, installed):
        srv = installed(None)
        srv.create_project()
        assert "error" in json.loads(srv.add_sample_track("Loop", mixer_channel=2))

    def test_sample_track_is_routed_in_lmms_13(self, installed):
        srv = installed("1.3.0")
        srv.create_project()
        srv.add_mixer_channel("Loops")
        result = json.loads(srv.add_sample_track("Loop", mixer_channel=1))
        assert "error" not in result
        sample = find_tracks(srv.get_project().root)[0].find("sampletrack")
        assert sample.get("mixch") == "1" and "fxch" not in sample.attrib
        info = json.loads(srv.get_project_info())
        assert info["tracks"][0]["mixer_channel"] == 1

    # --- existing projects -----------------------------------------

    @staticmethod
    def _write(tmp_path, names, suffix=".mmp", text=None):
        """Write a project spelled with `names` (or the given text)."""
        path = tmp_path / f"existing{suffix}"
        text = _mixer_project_xml(names) if text is None else text
        if suffix == ".mmp":
            path.write_text(text)
        else:
            from lmms_mcp.xml_parser import _q_compress
            path.write_bytes(_q_compress(text.encode()))
        return path

    @classmethod
    def _load(cls, srv, tmp_path, names, suffix=".mmp"):
        path = cls._write(tmp_path, names, suffix)
        result = json.loads(srv.load_project(str(path)))
        assert "error" not in result, result
        return path

    @pytest.mark.parametrize("attr", ["FX_MIXER", "MIXER"])
    @pytest.mark.parametrize("suffix", [".mmp", ".mmpz"])
    @pytest.mark.parametrize("version", ["1.2.2", "1.3.0", None])
    def test_both_spellings_are_read(
            self, installed, tmp_path, attr, suffix, version):
        srv = installed(version)
        self._load(srv, tmp_path, self._names(attr), suffix)

        info = json.loads(srv.get_project_info())
        assert [(t["name"], t["mixer_channel"]) for t in info["tracks"]] == [
            ("Bass", 1), ("Loop", 0)]
        assert [(m["num"], m["name"], m["volume"], m["effects"], m["sends"])
                for m in info["mixer_channels"]] == [
            (0, "Master", 1.2, [], []),
            (1, "drums", 0.65, ["delay"], [{"channel": 0, "amount": 1.0}])]
        assert json.loads(srv.get_mixer_channels())["count"] == 2
        assert json.loads(srv.list_tracks())["tracks"][0]["name"] == "Bass"
        chain = json.loads(srv.get_effect_chain("mixer", 1))
        assert [e["name"] for e in chain["effects"]] == ["delay"]
        assert "error" not in json.loads(srv.set_mixer_channel_volume(1, 0.3))
        assert find_mixer_channel_element(
            srv.get_project().root, 1).get("volume") == "0.3"
        assert "error" not in json.loads(srv.set_mixer_channel_name(1, "kit"))

    @pytest.mark.parametrize("attr", ["FX_MIXER", "MIXER"])
    @pytest.mark.parametrize("version", ["1.2.2", "1.3.0", None])
    def test_edits_keep_the_spelling_of_the_loaded_project(
            self, installed, tmp_path, attr, version):
        srv = installed(version)
        path = self._load(srv, tmp_path, self._names(attr))
        # the project's own spelling, except 1.3 names under LMMS 1.2.2
        # (converted on load, see TestMixerConversion)
        names = self._names(
            "FX_MIXER" if (attr, version) == ("MIXER", "1.2.2") else attr)

        assert json.loads(srv.add_mixer_channel("Pads"))["channel_num"] == 2
        assert "error" not in json.loads(srv.add_instrument_track(
            "Pad", mixer_channel=2))
        srv.add_pattern_track("Beat")
        assert "error" not in json.loads(srv.add_effect("mixer", 2, "bitcrush"))
        assert "error" not in json.loads(srv.add_effect("mixer", 1, "bitcrush"))
        assert "error" not in json.loads(srv.add_automation(
            "mixer", "volume",
            [{"bar": 0, "value": 1.0}, {"bar": 1, "value": 0.2}],
            target_index=2))

        root = srv.get_project().root
        _assert_only_spelling(root, names)
        assert [find_mixer_channel_element(root, n).tag for n in range(3)] == [
            names.channel] * 3
        pad = find_tracks(root)[2].find("instrumenttrack")
        assert pad.get(names.track_attr) == "2"

        # ... and saving and loading again changes nothing
        srv.save_project(str(path))
        srv.create_project()
        srv.load_project(str(path))
        _assert_only_spelling(srv.get_project().root, names)
        info = json.loads(srv.get_project_info())
        assert [m["name"] for m in info["mixer_channels"]] == [
            "Master", "drums", "Pads"]
        assert info["tracks"][2]["mixer_channel"] == 2
        assert info["mixer_channels"][1]["effects"] == ["delay", "bitcrush"]

    def test_sample_track_in_a_loaded_lmms12_project(self, installed, tmp_path):
        # the project is in the 1.2 format whatever LMMS is installed: no
        # mixch is mixed into it, and a channel it cannot honour is refused
        srv = installed("1.3.0")
        self._load(srv, tmp_path, self._names("FX_MIXER"))
        assert "error" in json.loads(srv.add_sample_track("L", mixer_channel=1))
        assert "error" not in json.loads(srv.add_sample_track("L"))
        sample = find_tracks(srv.get_project().root)[-1].find("sampletrack")
        assert "mixch" not in sample.attrib and "fxch" not in sample.attrib

    @pytest.mark.parametrize("version", ["1.3.0", None])
    def test_sample_track_in_a_loaded_lmms13_project(
            self, installed, tmp_path, version):
        # no conversion here (1.3 installed, or no LMMS detected), so the
        # project's mixch sample routing stays
        srv = installed(version)
        self._load(srv, tmp_path, self._names("MIXER"))
        assert "error" not in json.loads(srv.add_sample_track("L", mixer_channel=1))
        sample = find_tracks(srv.get_project().root)[-1].find("sampletrack")
        assert sample.get("mixch") == "1"

    def test_loaded_preset_keeps_the_tracks_channel(self, lab):
        # the preset's own channel (fxch="9" in the .xpf) is not used
        lab.put("TripleOscillator/Bubbles.xpf", _xpf("Bubbles"))
        assert "error" not in lab.load(0, "TripleOscillator/Bubbles")
        inst = lab.inst_track()
        assert inst.get("fxch") == "2" and "mixch" not in inst.attrib

    # --- helpers ---------------------------------------------------

    def test_track_channel_reads_either_attribute(self):
        def read(**attrs):
            return get_track_mixer_channel(ET.Element("instrumenttrack", attrs))
        assert read(fxch="3") == 3
        assert read(mixch="4") == 4
        assert read(fxch="2.0") == 2
        assert read() == 0
        assert read(fxch="junk") == 0

    def test_names_are_taken_from_the_project(self):
        from lmms_mcp import xml_parser
        for names in xml_parser.ALL_MIXER_NAMES:
            root = ET.fromstring(_mixer_project_xml(names))
            assert xml_parser.mixer_names(root) == names
            assert xml_parser.find_mixer(root).tag == names.mixer
            assert [c.get("num") for c in xml_parser.find_mixer_channels(root)
                    ] == ["0", "1"]

    def test_names_come_from_the_tracks_when_there_is_no_mixer(self):
        from lmms_mcp import xml_parser
        for names in xml_parser.ALL_MIXER_NAMES:
            root = ET.fromstring(_mixer_project_xml(names))
            song = root.find("song")
            song.remove(song.find(names.mixer))
            assert xml_parser.find_mixer(root) is None
            assert xml_parser.find_mixer_channels(root) == []
            assert xml_parser.mixer_names(root) == names
            # a mixer is created, in the project's spelling
            channel = xml_parser.add_mixer_channel(root, "X")
            assert channel.tag == names.channel and channel.get("num") == "1"
            assert xml_parser.find_mixer(root).tag == names.mixer
            assert [c.get("num") for c in xml_parser.find_mixer_channels(root)
                    ] == ["0", "1"]
        assert xml_parser.mixer_names(ET.Element("lmms-project")) == (
            xml_parser.DEFAULT_MIXER_NAMES)



# ──────────────────────────────────────────────────────────────────
# MIXER ROUTING: a track must be routed to a channel that exists
# ──────────────────────────────────────────────────────────────────

class TestMixerChannelMustExist:
    """LMMS does not route a track to a mixer channel that is not in the
    project: 1.2.2 plays it on the last channel instead (with that
    channel's volume; real renders), so nothing may be routed to one: every
    tool that routes a track refuses it and says which channels exist."""

    VERSIONS = ["1.2.2", "1.3.0", None]

    @staticmethod
    def _project(installed, version):
        """Server with a project that has the channels 0 Master, 1 Drums."""
        srv = installed(version)
        srv.create_project()
        srv.add_mixer_channel("Drums")
        return srv

    @staticmethod
    def _assert_refused(result, channel=5):
        assert f"Mixer channel {channel} does not exist" in result["error"], result
        # the channels there are, with their names, and how to get another
        assert "0 (Master), 1 (Drums)" in result["error"]
        assert "add_mixer_channel" in result["error"]
        assert "channel 2" in result["error"]  # the one it would create next
        # what LMMS would do instead (verified with LMMS 1.2.2: the track
        # plays on the last channel, with that channel's volume)
        assert "plays such a track on the last channel, 1 (Drums)" in (
            result["error"])

    @pytest.mark.parametrize("version", VERSIONS)
    @pytest.mark.parametrize("instrument", ["tripleoscillator", "papu"])
    def test_instrument_track(self, installed, version, instrument):
        # papu is not a known instrument: the custom-plugin path
        srv = self._project(installed, version)
        before = srv.get_project().get_xml_string()
        self._assert_refused(json.loads(srv.add_instrument_track(
            "Lead", instrument, mixer_channel=5)))
        assert srv.get_project().get_xml_string() == before
        assert not find_tracks(srv.get_project().root)

        # ... and works once the channel exists
        for _ in range(4):
            srv.add_mixer_channel("More")
        assert "error" not in json.loads(srv.add_instrument_track(
            "Lead", instrument, mixer_channel=5))
        track = find_tracks(srv.get_project().root)[0]
        assert get_track_mixer_channel(track.find("instrumenttrack")) == 5

    @pytest.mark.parametrize("version", VERSIONS)
    def test_master_and_existing_channels_are_accepted(
            self, installed, version):
        srv = self._project(installed, version)
        for channel in (0, 1):
            assert "error" not in json.loads(srv.add_instrument_track(
                "T", mixer_channel=channel))

    @pytest.mark.parametrize("channel", [-1, 2, 99])
    def test_other_numbers_are_refused(self, installed, channel):
        srv = self._project(installed, "1.2.2")
        result = json.loads(srv.add_instrument_track(
            "T", mixer_channel=channel))
        assert f"Mixer channel {channel} does not exist" in result["error"]

    def test_master_always_exists_even_without_a_mixer(self):
        # LMMS creates Master when the file has no mixer at all
        from lmms_mcp import xml_parser
        root = create_empty_project()
        song = root.find("song")
        song.remove(song.find("fxmixer"))
        add_instrument_track(root, "T")  # Master: no error
        with pytest.raises(ValueError) as err:
            add_instrument_track(root, "T", mixer_channel=1)
        assert "channels 0 (Master)" in str(err.value)
        assert "next one it creates is channel 1)" in str(err.value)
        # ... and add_mixer_channel, which the error points to, works: it
        # adds the missing mixer (with Master) in the project's spelling
        channel = xml_parser.add_mixer_channel(root, "Drums")
        assert channel.get("num") == "1" and channel.tag == "fxchannel"
        assert [c.get("num") for c in xml_parser.find_mixer_channels(root)] == [
            "0", "1"]
        add_instrument_track(root, "T", mixer_channel=1)

    def test_the_error_says_what_lmms_does_instead(self):
        # verified with LMMS 1.2.2 (real renders): a number past the last
        # channel plays on the last channel; a number inside a gap is an
        # empty channel without a send to Master (silent)
        from lmms_mcp import xml_parser
        root = create_empty_project()
        xml_parser.add_mixer_channel(root, "One")
        xml_parser.add_mixer_channel(root, "Two").set("num", "3")

        def refusal(channel):
            with pytest.raises(ValueError) as err:
                xml_parser.check_mixer_channel(root, channel)
            return str(err.value)

        past = refusal(9)
        assert "last channel, 3 (Two)" in past and "silent if" in past
        assert "next one it creates is channel 4" in past
        gap = refusal(2)
        assert "without a send to Master" in gap and "silent" in gap
        assert "last channel" not in gap
        negative = refusal(-1)
        assert "Mixer channel -1 does not exist" in negative
        assert "last channel" not in negative and "silent" not in negative

    def test_a_gap_in_the_numbers_is_not_a_channel(self):
        # LMMS allocates the numbers between channels empty and without a
        # send to Master: silent
        from lmms_mcp import xml_parser
        root = create_empty_project()
        xml_parser.add_mixer_channel(root, "One")
        two = xml_parser.add_mixer_channel(root, "Two")
        two.set("num", "3")
        assert xml_parser.mixer_channel_numbers(root) == [0, 1, 3]
        add_instrument_track(root, "T", mixer_channel=3)
        with pytest.raises(ValueError, match="0 \\(Master\\), 1 \\(One\\), 3 \\(Two\\)"):
            add_instrument_track(root, "T", mixer_channel=2)

    @pytest.mark.parametrize("version,reason", [
        ("1.3.0", "does not exist"), ("1.2.2", "LMMS 1.2 mixer format")])
    def test_sample_track(self, installed, version, reason):
        # LMMS 1.3 routes sample tracks, so the channel must exist; 1.2.x
        # cannot route them at all (its own refusal, unchanged)
        srv = self._project(installed, version)
        before = srv.get_project().get_xml_string()
        result = json.loads(srv.add_sample_track("Loop", mixer_channel=5))
        assert reason in result["error"]
        assert srv.get_project().get_xml_string() == before
        assert "error" not in json.loads(
            srv.add_sample_track("Loop", mixer_channel=0))
        if version == "1.3.0":
            assert "error" not in json.loads(
                srv.add_sample_track("Loop", mixer_channel=1))
            self._assert_refused(json.loads(
                srv.add_sample_track("Loop", mixer_channel=5)))

    @pytest.mark.parametrize("version", VERSIONS)
    def test_vst_track(self, installed, tmp_path, version):
        srv = self._project(installed, version)
        dll = tmp_path / "synth.dll"
        dll.write_bytes(b"MZ")
        before = srv.get_project().get_xml_string()
        self._assert_refused(json.loads(srv.add_vst_track(
            "VST", str(dll), mixer_channel=5)))
        assert srv.get_project().get_xml_string() == before
        assert "error" not in json.loads(srv.add_vst_track(
            "VST", str(dll), mixer_channel=1))

    def test_pattern_track_is_routed_to_master(self, installed):
        # add_pattern_track has no mixer_channel: its kicker is on Master,
        # which always exists
        srv = installed("1.2.2")
        srv.create_project()
        assert "error" not in json.loads(srv.add_pattern_track("Beat"))
        assert srv.get_project().unroutable_tracks() == []

    def test_project_add_track_validates_too(self):
        proj = LMMSProject()
        proj.new(lmms_version="1.3.0")  # (1.2 refuses sample routing anyway)
        for kind in ("instrument", "sample"):
            with pytest.raises(ValueError, match="does not exist"):
                proj.add_track(kind, "T", mixer_channel=3)
        assert not find_tracks(proj.root)

    # --- existing routing (a loaded project, a preset) ---------------

    @staticmethod
    def _stale_project(tmp_path, channel=5):
        from lmms_mcp import xml_parser
        text = _mixer_project_xml(xml_parser.FX_MIXER).replace(
            'fxch="1"', f'fxch="{channel}"')
        return TestMixerNames._write(tmp_path, None, text=text)

    def test_load_project_warns_about_a_track_on_a_missing_channel(
            self, installed, tmp_path):
        srv = installed("1.2.2")
        path = self._stale_project(tmp_path)
        result = json.loads(srv.load_project(str(path)))
        (warning,) = result["warnings"]
        assert "track 0 'Bass' (channel 5)" in warning
        assert "0 (Master), 1 (drums)" in warning
        # the fix is a re-route, not removing and re-adding the track
        assert "set_track_mixer_channel(0, 0)" in warning
        assert "add_mixer_channel" in warning
        assert "remove_track" not in warning
        assert "another channel" in warning
        assert srv.get_project().unroutable_tracks() == [(0, "Bass", 5)]

    def test_load_project_is_quiet_when_the_routing_is_valid(
            self, installed, tmp_path):
        from lmms_mcp import xml_parser
        srv = installed("1.2.2")
        path = TestMixerNames._write(tmp_path, xml_parser.FX_MIXER)
        assert "warnings" not in json.loads(srv.load_project(str(path)))

    def test_creating_the_missing_channels_fixes_the_warning(
            self, installed, tmp_path):
        srv = installed("1.2.2")
        srv.load_project(str(self._stale_project(tmp_path, channel=3)))
        assert srv.get_project().unroutable_tracks()
        srv.add_mixer_channel("Two")
        srv.add_mixer_channel("Three")
        assert srv.get_project().unroutable_tracks() == []

    def test_unroutable_tracks_looks_inside_pattern_tracks(self):
        from lmms_mcp import xml_parser
        root = create_empty_project()
        add_pattern_track(root, "Beat")
        inner = root.find(".//bbtrack//instrumenttrack")
        inner.set("fxch", "4")
        assert xml_parser.unroutable_tracks(root) == [(0, "Beat", 4)]

    def test_preset_into_a_track_on_a_missing_channel_warns(self, lab):
        # the track keeps its channel (the preset's own is not used), so a
        # stale one stays misrouted: say so
        lab.put("TripleOscillator/Bubbles.xpf", _xpf("Bubbles"))
        clean = lab.load(0, "TripleOscillator/Bubbles")
        assert "warnings" not in clean
        lab.inst_track().set("fxch", "9")
        result = lab.load(0, "TripleOscillator/Bubbles")
        (warning,) = result["warnings"]
        assert "keeps its mixer channel 9" in warning
        assert "does not exist" in warning and "add_mixer_channel" in warning
        assert "set_track_mixer_channel(0, channel)" in warning
        assert "last channel" in warning
        # a new track takes channel 0
        assert "warnings" not in lab.load(None, "TripleOscillator/Bubbles")


# ──────────────────────────────────────────────────────────────────
# set_track_mixer_channel: re-route a track that already exists
# ──────────────────────────────────────────────────────────────────

class TestSetTrackMixerChannel:
    """A track on a missing channel (a loaded project, a preset) is fixed
    by routing it to a channel that exists: no remove_track and re-add."""

    @staticmethod
    def _project(installed, version="1.2.2"):
        """Channels 0 Master, 1 Drums, 2 Bass; track 0 'Lead' on channel 1."""
        srv = installed(version)
        srv.create_project()
        srv.add_mixer_channel("Drums")
        srv.add_mixer_channel("Bass")
        srv.add_instrument_track("Lead", mixer_channel=1)
        return srv

    @staticmethod
    def _lead(srv):
        return find_tracks(srv.get_project().root)[0].find("instrumenttrack")

    def test_routes_an_instrument_track(self, installed):
        srv = self._project(installed)
        srv.get_project()._modified = False
        result = json.loads(srv.set_track_mixer_channel(0, 2))
        assert result["track_index"] == 0
        assert (result["mixer_channel"], result["previous_mixer_channel"]) == (2, 1)
        assert result["message"] == "Routed track 0 'Lead' to mixer channel 2 (Bass)"
        assert "error" not in result
        assert get_track_mixer_channel(self._lead(srv)) == 2
        assert json.loads(srv.get_track(0))["mixer_channel"] == 2
        assert srv.get_project().modified
        # Master and back
        assert "error" not in json.loads(srv.set_track_mixer_channel(0, 0))
        assert get_track_mixer_channel(self._lead(srv)) == 0

    @pytest.mark.parametrize("version", ["1.2.2", "1.3.0", None])
    def test_a_channel_the_project_lacks_is_refused(self, installed, version):
        srv = self._project(installed, version)
        before = srv.get_project().get_xml_string()
        srv.get_project()._modified = False
        result = json.loads(srv.set_track_mixer_channel(0, 5))
        assert "Mixer channel 5 does not exist" in result["error"]
        assert "0 (Master), 1 (Drums), 2 (Bass)" in result["error"]
        assert "add_mixer_channel" in result["error"]
        assert "channel 3" in result["error"]  # the next one it creates
        assert "set_track_mixer_channel(0, channel)" in result["error"]
        assert "plays such a track on the last channel, 2 (Bass)" in (
            result["error"])
        assert srv.get_project().get_xml_string() == before
        assert not srv.get_project().modified
        # a channel created first can be routed to
        for _ in range(3):
            srv.add_mixer_channel("More")
        assert "error" not in json.loads(srv.set_track_mixer_channel(0, 5))
        assert get_track_mixer_channel(self._lead(srv)) == 5

    @pytest.mark.parametrize("channel", [-1, 3, 99])
    def test_other_numbers_are_refused(self, installed, channel):
        srv = self._project(installed)
        result = json.loads(srv.set_track_mixer_channel(0, channel))
        assert f"Mixer channel {channel} does not exist" in result["error"]
        assert get_track_mixer_channel(self._lead(srv)) == 1

    def test_bad_arguments(self, installed):
        srv = self._project(installed)
        assert "out of range" in json.loads(
            srv.set_track_mixer_channel(4, 1))["error"]
        assert "out of range" in json.loads(
            srv.set_track_mixer_channel(-1, 1))["error"]
        assert "must be an integer" in json.loads(
            srv.set_track_mixer_channel(0, "1"))["error"]
        assert "must be an integer" in json.loads(
            srv.set_track_mixer_channel(0, True))["error"]
        assert get_track_mixer_channel(self._lead(srv)) == 1

    @pytest.mark.parametrize("version,attr", [
        ("1.2.2", "FX_MIXER"), ("1.3.0", "MIXER"), (None, "FX_MIXER")])
    def test_the_projects_spelling_is_kept(self, installed, version, attr):
        from lmms_mcp import xml_parser
        names = getattr(xml_parser, attr)
        srv = self._project(installed, version)
        srv.set_track_mixer_channel(0, 2)
        assert self._lead(srv).get(names.track_attr) == "2"
        _assert_only_spelling(srv.get_project().root, names)

    def test_a_loaded_project_keeps_its_spelling_and_drops_a_leftover(
            self, installed, tmp_path):
        from lmms_mcp import xml_parser
        srv = installed("1.2.2")
        text = _mixer_project_xml(xml_parser.FX_MIXER).replace(
            'fxch="1"', 'fxch="1" mixch="1"')  # a leftover of the 1.3 name
        path = TestMixerNames._write(tmp_path, None, text=text)
        assert "error" not in json.loads(srv.load_project(str(path)))
        result = json.loads(srv.set_track_mixer_channel(0, 0))
        assert "error" not in result
        bass = self._lead(srv)
        assert bass.get("fxch") == "0" and "mixch" not in bass.attrib

    def test_an_automated_channel_changes_the_child_element(self, installed):
        from lmms_mcp import xml_parser
        srv = self._project(installed)
        inst = self._lead(srv)
        del inst.attrib["fxch"]
        ET.SubElement(inst, "fxch", {"id": "77", "value": "1"})
        srv.set_track_mixer_channel(0, 2)
        assert inst.find("fxch").attrib == {"id": "77", "value": "2"}
        assert "fxch" not in inst.attrib
        assert xml_parser.get_track_mixer_channel(inst) == 2

    def test_a_track_on_a_missing_channel_is_fixed_by_the_tool(
            self, installed, tmp_path):
        srv = installed("1.2.2")
        path = TestMixerChannelMustExist._stale_project(tmp_path)
        (warning,) = json.loads(srv.load_project(str(path)))["warnings"]
        assert "set_track_mixer_channel" in warning
        assert srv.get_project().unroutable_tracks() == [(0, "Bass", 5)]
        result = json.loads(srv.set_track_mixer_channel(0, 1))
        assert result["previous_mixer_channel"] == 5
        assert srv.get_project().unroutable_tracks() == []

    def test_a_pattern_track_routes_its_instruments(self, installed):
        srv = self._project(installed)
        srv.add_pattern_track("Beat")
        inner = srv.get_project().root.find(".//bbtrack//instrumenttrack")
        result = json.loads(srv.set_track_mixer_channel(1, 2))
        assert result["message"] == (
            "Routed every instrument of pattern track 1 'Beat' to mixer "
            "channel 2 (Bass)")
        assert result["routed"] == [{"name": "Beat", "previous": 0}]
        assert get_track_mixer_channel(inner) == 2
        assert srv.get_project().unroutable_tracks() == []
        # the instrument track is untouched
        assert get_track_mixer_channel(self._lead(srv)) == 1

    def test_a_pattern_track_on_a_missing_channel_is_fixed(self, installed):
        from lmms_mcp import xml_parser
        srv = self._project(installed)
        srv.add_pattern_track("Beat")
        srv.get_project().root.find(".//bbtrack//instrumenttrack").set(
            "fxch", "9")
        assert srv.get_project().unroutable_tracks() == [(1, "Beat", 9)]
        srv.set_track_mixer_channel(1, 0)
        assert srv.get_project().unroutable_tracks() == []

    def test_an_automation_track_has_no_channel(self, installed):
        srv = self._project(installed)
        srv.add_automation_track("Auto")
        before = srv.get_project().get_xml_string()
        result = json.loads(srv.set_track_mixer_channel(1, 1))
        assert "automation track" in result["error"]
        assert "no audio output" in result["error"]
        assert srv.get_project().get_xml_string() == before

    def test_sample_track_on_lmms_12(self, installed):
        # 1.2.x sample tracks always play on Master: the same refusal as
        # add_sample_track, and Master is accepted without a change
        srv = self._project(installed, "1.2.2")
        srv.add_sample_track("Loop")
        before = srv.get_project().get_xml_string()
        result = json.loads(srv.set_track_mixer_channel(1, 1))
        assert "Sample tracks cannot be routed to mixer channel 1" in (
            result["error"])
        assert "LMMS 1.2 mixer format" in result["error"]
        assert srv.get_project().get_xml_string() == before
        # the refusal comes before the check of the channel
        assert "LMMS 1.2 mixer format" in json.loads(
            srv.set_track_mixer_channel(1, 9))["error"]
        ok = json.loads(srv.set_track_mixer_channel(1, 0))
        assert "error" not in ok
        assert "nothing to change" in ok["message"]
        assert srv.get_project().get_xml_string() == before
        sample = find_tracks(srv.get_project().root)[1].find("sampletrack")
        assert get_track_mixer_channel(sample) == 0

    def test_sample_track_on_lmms_13(self, installed):
        srv = self._project(installed, "1.3.0")
        srv.add_sample_track("Loop")
        sample = find_tracks(srv.get_project().root)[1].find("sampletrack")
        result = json.loads(srv.set_track_mixer_channel(1, 2))
        assert "error" not in result
        assert result["message"] == "Routed track 1 'Loop' to mixer channel 2 (Bass)"
        assert sample.get("mixch") == "2" and "fxch" not in sample.attrib
        # ... but only to a channel that exists
        assert "Mixer channel 7 does not exist" in json.loads(
            srv.set_track_mixer_channel(1, 7))["error"]
        assert sample.get("mixch") == "2"

    def test_the_project_function_raises_for_the_same_cases(self):
        from lmms_mcp import xml_parser
        root = create_empty_project()
        xml_parser.add_mixer_channel(root, "Drums")
        add_instrument_track(root, "Lead")
        info = xml_parser.set_track_mixer_channel(root, 0, 1)
        assert info == {"track": "Lead", "type": "Instrument", "channel": 1,
                        "routed": [{"name": "Lead", "previous": 0}]}
        with pytest.raises(ValueError, match="does not exist"):
            xml_parser.set_track_mixer_channel(root, 0, 4)
        with pytest.raises(ValueError, match="out of range"):
            xml_parser.set_track_mixer_channel(root, 3, 0)


# ──────────────────────────────────────────────────────────────────
# MIXER CONVERSION: LMMS 1.3 names in a project, LMMS 1.2 installed
# ──────────────────────────────────────────────────────────────────

class TestMixerConversion:
    """A project an older version of this server wrote for LMMS 1.2.2 has
    the LMMS 1.3 mixer names, which 1.2.2 ignores (verified: routing,
    channel volume and effects had no effect). load_project renames them to
    the installed LMMS's, keeping everything else."""

    @staticmethod
    def _names(attr):
        return TestMixerNames._names(attr)

    @staticmethod
    def _load(srv, tmp_path, names, suffix=".mmp", text=None):
        path = TestMixerNames._write(tmp_path, names, suffix, text)
        return path, json.loads(srv.load_project(str(path)))

    @staticmethod
    def _data(srv):
        """Everything a project holds about its mixer and routing."""
        info = json.loads(srv.get_project_info())
        return info["mixer_channels"], [
            (t["name"], t["mixer_channel"], t["volume"]) for t in info["tracks"]]

    @pytest.mark.parametrize("suffix", [".mmp", ".mmpz"])
    def test_a_13_project_is_converted_under_lmms_12(
            self, installed, tmp_path, suffix):
        srv = installed("1.2.2")
        path, result = self._load(
            srv, tmp_path, self._names("MIXER"), suffix)

        report = result["converted_mixer"]
        assert report["from"] == "mixer/mixerchannel/mixch"
        assert report["to"] == "fxmixer/fxchannel/fxch"
        assert (report["channels"], report["tracks"]) == (2, 1)
        assert "LMMS 1.2.2" in report["message"]
        assert "save_project" in report["message"]
        assert "sample_tracks_on_master" not in report

        proj = srv.get_project()
        _assert_only_spelling(proj.root, self._names("FX_MIXER"))
        assert proj.modified  # differs from the file until it is saved
        # all the channel data and the routing survived
        channels, tracks = self._data(srv)
        assert [(m["num"], m["name"], m["volume"], m["muted"], m["soloed"],
                 m["effects"], m["sends"]) for m in channels] == [
            (0, "Master", 1.2, False, False, [], []),
            (1, "drums", 0.65, False, False, ["delay"],
             [{"channel": 0, "amount": 1.0}])]
        assert tracks == [("Bass", 1, 80), ("Loop", 0, 100)]
        assert proj.unroutable_tracks() == []
        # the file on disk is unchanged until it is saved: loading it
        # again converts it again
        assert "converted_mixer" in json.loads(srv.load_project(str(path)))

    def test_save_and_load_again_round_trips(self, installed, tmp_path):
        srv = installed("1.2.2")
        path, _ = self._load(srv, tmp_path, self._names("MIXER"))
        converted = self._data(srv)
        for suffix in (".mmpz", ".mmp"):
            out = tmp_path / f"saved{suffix}"
            srv.save_project(str(out))
            srv.create_project()
            result = json.loads(srv.load_project(str(out)))
            assert "converted_mixer" not in result  # nothing left to convert
            _assert_only_spelling(srv.get_project().root,
                                  self._names("FX_MIXER"))
            assert self._data(srv) == converted
            assert not srv.get_project().modified
        # and later edits keep the names LMMS 1.2.2 reads
        srv.add_mixer_channel("Pads")
        srv.add_instrument_track("Pad", mixer_channel=2)
        srv.add_pattern_track("Beat")
        _assert_only_spelling(srv.get_project().root, self._names("FX_MIXER"))

    def test_converting_in_place_keeps_each_element_and_attribute(self):
        # the channels keep every attribute, child and their order
        from lmms_mcp import xml_parser
        text = _mixer_project_xml(xml_parser.MIXER)
        root = ET.fromstring(text)
        report = xml_parser.convert_mixer_names(root, xml_parser.FX_MIXER)
        assert (report["channels"], report["tracks"]) == (2, 1)
        expected = text.replace("mixerchannel", "fxchannel").replace(
            "<mixer ", "<fxmixer ").replace("</mixer>", "</fxmixer>").replace(
            "mixch=", "fxch=")
        assert ET.tostring(root) == ET.tostring(ET.fromstring(expected))
        # nothing left to convert; the other direction restores the original
        assert xml_parser.convert_mixer_names(
            root, xml_parser.FX_MIXER) is None
        back = xml_parser.convert_mixer_names(root, xml_parser.MIXER)
        assert (back["channels"], back["tracks"]) == (2, 1)
        assert ET.tostring(root) == ET.tostring(ET.fromstring(text))

    def test_pattern_track_and_mixer_state_are_converted(self):
        from lmms_mcp import xml_parser
        root = create_empty_project(lmms_version="1.3.0")
        xml_parser.add_mixer_channel(root, "Drums", volume=0.4)
        add_instrument_track(root, "Lead", mixer_channel=1)
        add_pattern_track(root, "Beat")
        channel = find_mixer_channel_element(root, 1)
        channel.set("muted", "1")
        channel.set("soloed", "1")
        channel.find("send").set("amount", "0.5")
        extra = ET.SubElement(channel, "send", {"channel": "2", "amount": "0.25"})
        assert extra is not None

        report = xml_parser.convert_mixer_names(root, xml_parser.FX_MIXER)
        assert (report["channels"], report["tracks"]) == (2, 2)  # + inner kicker
        _assert_only_spelling(root, xml_parser.FX_MIXER)
        channel = find_mixer_channel_element(root, 1)
        assert channel.tag == "fxchannel"
        assert (channel.get("name"), channel.get("volume"),
                channel.get("muted"), channel.get("soloed")) == (
            "Drums", "0.4", "1", "1")
        assert [s.attrib for s in channel.findall("send")] == [
            {"channel": "0", "amount": "0.5"},
            {"channel": "2", "amount": "0.25"}]
        inner = root.find(".//bbtrack//instrumenttrack")
        assert inner.get("fxch") == "0" and "mixch" not in inner.attrib

    def test_sample_track_routing_is_reported_not_silently_lost(
            self, installed, tmp_path):
        # LMMS 1.2 cannot route sample tracks: one that was on a channel
        # now plays on Master, and the response says which
        from lmms_mcp import xml_parser
        text = _mixer_project_xml(xml_parser.MIXER).replace(
            '<sampletrack vol="100" pan="0">',
            '<sampletrack vol="100" pan="0" mixch="1">')
        srv = installed("1.2.2")
        _, result = self._load(srv, tmp_path, None, text=text)
        report = result["converted_mixer"]
        assert report["sample_tracks_on_master"] == [
            {"track": "Loop", "channel": 1}]
        assert "'Loop' (channel 1)" in report["message"]
        assert "Master" in report["message"]
        assert report["tracks"] == 2
        sample = find_tracks(srv.get_project().root)[1].find("sampletrack")
        assert set(sample.attrib) == {"vol", "pan"}
        _assert_only_spelling(srv.get_project().root, self._names("FX_MIXER"))

    def test_a_sample_track_on_master_is_not_reported(self, installed, tmp_path):
        from lmms_mcp import xml_parser
        text = _mixer_project_xml(xml_parser.MIXER).replace(
            '<sampletrack vol="100" pan="0">',
            '<sampletrack vol="100" pan="0" mixch="0">')
        srv = installed("1.2.2")
        _, result = self._load(srv, tmp_path, None, text=text)
        assert "sample_tracks_on_master" not in result["converted_mixer"]

    # --- when nothing is converted ----------------------------------

    def test_no_conversion_without_a_detected_lmms(self, installed, tmp_path):
        # nothing says which names the LMMS reads: the project stays as is
        for version in (None, "", "garbage"):
            srv = installed(version)
            _, result = self._load(srv, tmp_path, self._names("MIXER"))
            assert "converted_mixer" not in result
            proj = srv.get_project()
            _assert_only_spelling(proj.root, self._names("MIXER"))
            assert not proj.modified

    @pytest.mark.parametrize("version", ["1.3.0", "1.4.1", "2.0.0"])
    def test_no_conversion_under_lmms_13(self, installed, tmp_path, version):
        srv = installed(version)
        _, result = self._load(srv, tmp_path, self._names("MIXER"))
        assert "converted_mixer" not in result
        _assert_only_spelling(srv.get_project().root, self._names("MIXER"))

    @pytest.mark.parametrize("version", ["1.3.0", "1.2.2", None])
    def test_a_project_in_the_installed_spelling_is_untouched(
            self, installed, tmp_path, version):
        # LMMS 1.3+ upgrades a 1.2 project itself on load (it renames
        # fxmixer/fxchannel/fxch), so the 1.2 names are never converted
        srv = installed(version)
        _, result = self._load(srv, tmp_path, self._names("FX_MIXER"))
        assert "converted_mixer" not in result
        proj = srv.get_project()
        _assert_only_spelling(proj.root, self._names("FX_MIXER"))
        assert not proj.modified

    def test_a_project_without_a_mixer_only_has_its_tracks_converted(self):
        # (so that add_mixer_channel, which then adds a mixer in the
        # tracks' spelling, writes names the installed LMMS reads)
        from lmms_mcp import xml_parser
        root = ET.fromstring(_mixer_project_xml(xml_parser.MIXER))
        song = root.find("song")
        song.remove(song.find("mixer"))
        report = xml_parser.convert_mixer_names(root, xml_parser.FX_MIXER)
        assert (report["channels"], report["tracks"]) == (0, 1)
        assert root.find(".//instrumenttrack").get("fxch") == "1"
        assert xml_parser.mixer_names(root) == xml_parser.FX_MIXER
        xml_parser.add_mixer_channel(root, "X")
        _assert_only_spelling(root, xml_parser.FX_MIXER)

    def test_a_project_without_a_mixer_or_routing_needs_no_conversion(self):
        from lmms_mcp import xml_parser
        assert xml_parser.convert_mixer_names(
            ET.Element("lmms-project"), xml_parser.FX_MIXER) is None
        root = create_empty_project(lmms_version="1.3.0")
        song = root.find("song")
        song.remove(song.find("mixer"))
        add_instrument_track(root, "T")
        root.find(".//instrumenttrack").attrib.pop("fxch")  # no routing at all
        assert xml_parser.convert_mixer_names(root, xml_parser.FX_MIXER) is None

    def test_a_project_with_both_mixers_is_left_alone(self):
        from lmms_mcp import xml_parser
        root = ET.fromstring(_mixer_project_xml(xml_parser.MIXER))
        ET.SubElement(root.find("song"), "fxmixer")
        before = ET.tostring(root)
        assert xml_parser.convert_mixer_names(root, xml_parser.FX_MIXER) is None
        assert ET.tostring(root) == before

    @pytest.mark.parametrize("version,names", [
        ("1.2.2", "FX_MIXER"), ("1.2.0", "FX_MIXER"), ("1.1.3", "FX_MIXER"),
        ("1.3.0", None), ("1.10.0", None), (None, None), ("", None),
        ("garbage", None)])
    def test_which_installed_lmms_needs_a_conversion(self, version, names):
        from lmms_mcp import xml_parser
        expected = None if names is None else self._names(names)
        assert xml_parser.mixer_names_to_convert_to(version) == expected


# ──────────────────────────────────────────────────────────────────
# AUTOMATED MODEL VALUES: <volume id=".." value=".."/> instead of volume=".."
# ──────────────────────────────────────────────────────────────────

class TestAutomatedValues:
    """LMMS saves a model that is automated as a child element carrying its
    value; the readers must see it (get_mixer_channels reported 1.0)."""

    def test_model_value(self):
        from lmms_mcp.xml_parser import model_value
        plain = ET.fromstring('<ch volume="0.7"/>')
        child = ET.fromstring('<ch><volume id="3" value="0.4"/></ch>')
        both = ET.fromstring('<ch volume="0.7"><volume id="3" value="0.4"/></ch>')
        no_value = ET.fromstring('<ch volume="0.7"><volume id="3"/></ch>')
        assert model_value(plain, "volume", "1") == "0.7"
        assert model_value(child, "volume", "1") == "0.4"
        assert model_value(both, "volume", "1") == "0.4"  # LMMS reads the child
        assert model_value(no_value, "volume", "1") == "0.7"
        assert model_value(ET.Element("ch"), "volume", "1") == "1"

    @staticmethod
    def _automated_xml():
        from lmms_mcp import xml_parser
        text = _mixer_project_xml(xml_parser.FX_MIXER)
        return (
            text.replace(
                '<fxchannel num="1" muted="0" volume="0.65" name="drums" soloed="0">',
                '<fxchannel num="1" name="drums">'
                '<volume id="11" value="0.4" scale_type="linear"/>'
                '<muted id="12" value="1"/><soloed id="13" value="1"/>')
            .replace('pan="0" fxch="1" usemasterpitch="1" pitchrange="1" '
                     'pitch="0" basenote="57" vol="80">',
                     'fxch="1" usemasterpitch="1" pitchrange="1" pitch="0" '
                     'basenote="57"><vol id="14" value="130"/>'
                     '<pan id="15" value="-40"/>')
            .replace('<track muted="0" type="0" name="Bass" solo="0">',
                     '<track type="0" name="Bass"><muted id="16" value="1"/>'
                     '<solo id="17" value="1"/>')
        )

    def test_get_mixer_channels_reports_the_automated_values(
            self, installed, tmp_path):
        srv = installed("1.2.2")
        path = TestMixerNames._write(tmp_path, None, text=self._automated_xml())
        assert "error" not in json.loads(srv.load_project(str(path)))
        drums = json.loads(srv.get_mixer_channels())["channels"][1]
        assert (drums["volume"], drums["muted"], drums["soloed"]) == (
            0.4, True, True)
        resource = json.loads(srv.resource_project_mixer())["mixer_channels"]
        assert resource[1]["volume"] == 0.4
        # the plain channel is read as before
        assert json.loads(srv.get_mixer_channels())["channels"][0][
            "volume"] == 1.2

    def test_tracks_report_the_automated_values(self, installed, tmp_path):
        srv = installed("1.2.2")
        path = TestMixerNames._write(tmp_path, None, text=self._automated_xml())
        srv.load_project(str(path))
        bass = json.loads(srv.get_track(0))
        assert (bass["volume"], bass["panning"], bass["muted"], bass["solo"],
                bass["mixer_channel"]) == (130, -40, True, True, 1)

    def test_a_volume_automated_through_the_tools_keeps_its_value(
            self, installed):
        srv = installed("1.2.2")
        srv.create_project()
        srv.add_mixer_channel("Drums")
        srv.set_mixer_channel_volume(1, 0.65)
        srv.add_instrument_track("Lead", mixer_channel=1, volume=120, panning=25)

        first = json.loads(srv.add_automation(
            "mixer", "volume",
            [{"bar": 0, "value": 1.0}, {"bar": 2, "value": 0.5}],
            target_index=1))
        assert first["current_value"] == 0.65
        track = json.loads(srv.add_automation(
            "track", "volume",
            [{"bar": 0, "value": 100}, {"bar": 2, "value": 50}],
            target_index=0))
        assert track["current_value"] == 120

        # the attribute became a child element: the readers still see it
        channel = find_mixer_channel_element(srv.get_project().root, 1)
        assert "volume" not in channel.attrib and channel.find("volume") is not None
        assert json.loads(srv.get_mixer_channels())["channels"][1][
            "volume"] == 0.65
        lead = json.loads(srv.get_track(0))
        assert (lead["volume"], lead["panning"]) == (120, 25)
        # automating the same parameter again reports the real value
        again = json.loads(srv.add_automation(
            "mixer", "volume",
            [{"bar": 3, "value": 0.1}, {"bar": 4, "value": 0.2}],
            target_index=1))
        assert again["current_value"] == 0.65
        assert again["model_id"] == first["model_id"]
        again = json.loads(srv.add_automation(
            "track", "volume",
            [{"bar": 3, "value": 10}, {"bar": 4, "value": 20}],
            target_index=0))
        assert again["current_value"] == 120

    def test_a_converted_project_keeps_the_automated_volume(
            self, installed, tmp_path):
        from lmms_mcp import xml_parser
        text = self._automated_xml().replace("fxmixer", "mixer").replace(
            "fxchannel", "mixerchannel").replace("fxch=", "mixch=")
        srv = installed("1.2.2")
        path = TestMixerNames._write(tmp_path, None, text=text)
        result = json.loads(srv.load_project(str(path)))
        assert result["converted_mixer"]["channels"] == 2
        _assert_only_spelling(srv.get_project().root, xml_parser.FX_MIXER)
        assert json.loads(srv.get_mixer_channels())["channels"][1][
            "volume"] == 0.4
        channel = find_mixer_channel_element(srv.get_project().root, 1)
        assert channel.find("volume").attrib == {
            "id": "11", "value": "0.4", "scale_type": "linear"}


class TestSettersKeepAutomatedValues:
    """LMMS reads the child element of an automated model, not an attribute
    of the same name (verified with LMMS 1.2.2: <vol value="0"/> beside
    vol="100" is silent). The setters must write the child, or they change
    nothing; the value they write is the static one, which an automation
    curve overrides where it has points."""

    # --- the shared helper -------------------------------------------

    def test_set_model_value(self):
        from lmms_mcp.xml_parser import set_model_value
        plain = ET.fromstring('<t vol="1"/>')
        assert set_model_value(plain, "vol", "50") is False
        assert plain.attrib == {"vol": "50"} and len(plain) == 0

        missing = ET.fromstring("<t/>")
        assert set_model_value(missing, "vol", "50") is False
        assert missing.attrib == {"vol": "50"}

        child = ET.fromstring('<t><vol id="3" value="100" scale_type="linear"/></t>')
        assert set_model_value(child, "vol", "50") is True
        assert child.find("vol").attrib == {
            "id": "3", "value": "50", "scale_type": "linear"}  # id kept
        assert "vol" not in child.attrib  # no attribute is added

        # both (an older version of this server wrote the attribute beside
        # the child): the one LMMS reads changes, the stale one follows
        both = ET.fromstring('<t vol="100"><vol id="3" value="0"/></t>')
        assert set_model_value(both, "vol", "50") is True
        assert both.get("vol") == "50" and both.find("vol").get("value") == "50"

        # a child without a value is not the model's value
        empty = ET.fromstring('<t vol="1"><vol id="3"/></t>')
        assert set_model_value(empty, "vol", "50") is False
        assert empty.get("vol") == "50"

    def test_model_automation(self):
        from lmms_mcp import xml_parser
        root = create_empty_project()
        add_instrument_track(root, "Lead")
        inst = find_tracks(root)[0].find("instrumenttrack")
        assert xml_parser.model_automation(root, inst, "vol") == []
        model_id, _ = xml_parser.resolve_automation_target(root, "track", 0, "volume")
        assert xml_parser.model_automation(root, inst, "vol") == []  # no curve yet
        xml_parser.add_automation_curve_track(
            root, "Swell", [(384, 100.0), (768, 50.0)], target_id=model_id)
        xml_parser.add_automation_curve_track(
            root, "Later", [(1152, 80.0), (1344, 60.0)], target_id=model_id)
        xml_parser.add_automation_curve_track(  # another model's curve
            root, "Other", [(0, 1.0), (192, 0.0)], target_id=model_id + 99)
        assert xml_parser.model_automation(root, inst, "vol") == [
            {"track_index": 1, "track": "Swell", "start": 384, "last_point": 768},
            {"track_index": 2, "track": "Later", "start": 1152, "last_point": 1344},
        ]
        assert xml_parser.model_automation(root, inst, "pan") == []

    # --- setters on a file with automated values ---------------------

    @staticmethod
    def _loaded(installed, tmp_path):
        srv = installed("1.2.2")
        path = TestMixerNames._write(
            tmp_path, None, text=TestAutomatedValues._automated_xml())
        assert "error" not in json.loads(srv.load_project(str(path)))
        return srv

    def test_track_setters_change_the_child(self, installed, tmp_path):
        srv = self._loaded(installed, tmp_path)
        inst = find_tracks(srv.get_project().root)[0].find("instrumenttrack")
        for call in (srv.set_track_volume(0, 55), srv.set_track_panning(0, 15)):
            assert "error" not in json.loads(call)
        assert inst.find("vol").attrib["value"] == "55"
        assert inst.find("vol").get("id") == "14"
        assert inst.find("pan").attrib["value"] == "15"
        assert "vol" not in inst.attrib and "pan" not in inst.attrib
        track = json.loads(srv.get_track(0))
        assert (track["volume"], track["panning"]) == (55, 15)

        srv.mute_track(0, False)
        srv.solo_track(0, False)
        outer = find_tracks(srv.get_project().root)[0]
        assert outer.find("muted").get("value") == "0"
        assert outer.find("solo").get("value") == "0"
        assert "muted" not in outer.attrib and "solo" not in outer.attrib
        track = json.loads(srv.get_track(0))
        assert (track["muted"], track["solo"]) == (False, False)

    def test_mixer_channel_volume_changes_the_child(self, installed, tmp_path):
        srv = self._loaded(installed, tmp_path)
        assert "error" not in json.loads(srv.set_mixer_channel_volume(1, 0.3))
        channel = find_mixer_channel_element(srv.get_project().root, 1)
        assert channel.find("volume").attrib["value"] == "0.3"
        assert channel.find("volume").get("id") == "11"
        assert "volume" not in channel.attrib
        assert json.loads(srv.get_mixer_channels())["channels"][1]["volume"] == 0.3
        # a plain channel still gets an attribute
        srv.set_mixer_channel_volume(0, 0.9)
        master = find_mixer_channel_element(srv.get_project().root, 0)
        assert master.get("volume") == "0.9" and master.find("volume") is None

    def test_a_stale_attribute_beside_the_child_is_brought_in_line(
            self, installed, tmp_path):
        # what an older version of this server left after set_track_volume
        # on an automated track: LMMS read the child, the attribute did nothing
        srv = self._loaded(installed, tmp_path)
        inst = find_tracks(srv.get_project().root)[0].find("instrumenttrack")
        inst.set("vol", "100")
        srv.set_track_volume(0, 40)
        assert inst.get("vol") == "40" and inst.find("vol").get("value") == "40"
        assert json.loads(srv.get_track(0))["volume"] == 40

    def test_song_settings_change_the_child(self, installed):
        from lmms_mcp import xml_parser
        srv = installed("1.2.2")
        srv.create_project()
        for param in ("tempo", "master_volume", "master_pitch"):
            srv.add_automation(
                "song", param, [{"bar": 1, "value": 100}, {"bar": 2, "value": 90}])
        srv.set_tempo(150)
        srv.set_master_volume(60)
        srv.set_master_pitch(-3)
        head = xml_parser.find_head(srv.get_project().root)
        for name, value in (("bpm", "150"), ("mastervol", "60"),
                            ("masterpitch", "-3")):
            assert head.find(name).get("value") == value
            assert name not in head.attrib
        info = json.loads(srv.get_project_info())
        assert (info["bpm"], info["master_volume"], info["master_pitch"]) == (
            150, 60, -3)
        # a project without automation keeps plain attributes
        srv.create_project()
        srv.set_tempo(150)
        head = xml_parser.find_head(srv.get_project().root)
        assert head.get("bpm") == "150" and head.find("bpm") is None

    # --- the response says the parameter is automated ----------------

    @staticmethod
    def _automated_track(installed, first_bar=2):
        srv = installed("1.2.2")
        srv.create_project()
        srv.add_mixer_channel("Drums")
        srv.add_instrument_track("Lead", mixer_channel=1, volume=100)
        srv.add_automation("track", "volume", [
            {"bar": first_bar, "value": 50}, {"bar": first_bar + 2, "value": 80}],
            target_index=0, name="Swell")
        return srv

    def test_response_notes_an_automated_volume(self, installed):
        srv = self._automated_track(installed)
        result = json.loads(srv.set_track_volume(0, 30))
        assert result["automated"] is True
        assert result["message"] == "Set track 0 volume to 30"
        note = result["note"]
        assert note.startswith("Track 0's volume is automated")
        assert "automation track 1 'Swell'" in note
        assert "30 is its static value" in note
        # the static value applies until the curve starts (the pattern
        # starts at its first point), the curve overrides it from there and
        # LMMS keeps its last value after the last point
        assert "from the start of the song until the automation begins at bar 2" in note
        assert "overrides it wherever it has points" in note
        assert "after its last point, at bar 4" in note
        assert "remove_track" in note and "add_automation" in note
        # the static value was written
        inst = find_tracks(srv.get_project().root)[0].find("instrumenttrack")
        assert inst.find("vol").get("value") == "30"
        assert json.loads(srv.get_track(0))["volume"] == 30
        # the curve itself is untouched
        pattern = find_tracks(srv.get_project().root)[1].find("automationpattern")
        assert [t.get("value") for t in pattern.findall("time")] == ["50.0", "80.0"]

    def test_the_note_names_the_bar_the_curve_starts_at(self, installed):
        srv = self._automated_track(installed, first_bar=0)
        note = json.loads(srv.set_track_panning(0, 0))
        assert "error" not in note
        assert "note" not in note  # panning is not the automated parameter
        note = json.loads(srv.set_track_volume(0, 30))["note"]
        # nothing plays before bar 0: the static value is not heard
        assert "starts at the beginning of the song" in note
        assert "30 (its static value) is not heard in playback" in note
        assert "after its last point, at bar 2" in note

    def test_other_automated_parameters(self, installed):
        srv = self._automated_track(installed)
        srv.add_automation("track", "panning", [
            {"bar": 1, "value": -50}, {"bar": 3, "value": 50}], target_index=0)
        srv.add_automation("mixer", "volume", [
            {"bar": 1.5, "value": 0.5}, {"bar": 2, "value": 1.0}], target_index=1)
        srv.add_automation("song", "tempo", [
            {"bar": 1, "value": 120}, {"bar": 2, "value": 140}])
        pan = json.loads(srv.set_track_panning(0, 25))
        assert pan["automated"] and "Track 0's panning is automated" in pan["note"]
        assert "begins at bar 1" in pan["note"]
        mixer = json.loads(srv.set_mixer_channel_volume(1, 0.75))
        assert mixer["automated"] is True
        assert "Mixer channel 1's volume is automated" in mixer["note"]
        assert "0.75 is its static value" in mixer["note"]
        assert "begins at bar 1.5" in mixer["note"]
        tempo = json.loads(srv.set_tempo(100))
        assert tempo["automated"] is True
        assert "The tempo is automated" in tempo["note"]
        assert "100 BPM is its static value" in tempo["note"]

    def test_response_has_no_note_without_automation(self, installed):
        srv = installed("1.2.2")
        srv.create_project()
        srv.add_mixer_channel("Drums")
        srv.add_instrument_track("Lead", mixer_channel=1)
        for result in (
            srv.set_track_volume(0, 30), srv.set_track_panning(0, 5),
            srv.mute_track(0), srv.solo_track(0),
            srv.set_mixer_channel_volume(1, 0.5), srv.set_tempo(100),
            srv.set_master_volume(90), srv.set_master_pitch(1),
        ):
            data = json.loads(result)
            assert set(data) == {"message"}, data
        # a model stored as a child that no curve plays is not "automated"
        inst = find_tracks(srv.get_project().root)[0].find("instrumenttrack")
        del inst.attrib["vol"]
        ET.SubElement(inst, "vol", {"id": "5", "value": "100"})
        assert set(json.loads(srv.set_track_volume(0, 20))) == {"message"}
        assert inst.find("vol").get("value") == "20"

    def test_the_mix_note_of_a_preset_reads_an_automated_volume(self, lab):
        lab.put("TripleOscillator/Loud.xpf", _xpf("Loud", vol=50, pan=0))
        inst = lab.inst_track()
        del inst.attrib["vol"]
        ET.SubElement(inst, "vol", {"id": "5", "value": "80"})
        result = lab.load(0, "TripleOscillator/Loud")
        assert "volume 50 (the track keeps 80" in result["note"]


class TestLoadProjectErrors:
    """load_project turns every problem with the file into a JSON error
    with the reason (an exception reaches the MCP client only as "Error
    executing tool load_project") and leaves the open project alone."""

    @staticmethod
    def _open_project():
        """An open project with a track, to see that failures keep it."""
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        srv.add_instrument_track("Keep me")
        return srv, proj

    @staticmethod
    def _saved(tmp_path, name):
        proj = LMMSProject()
        proj.new()
        path = tmp_path / name
        proj.save(str(path))
        return path

    def _fails(self, srv, path):
        result = json.loads(srv.load_project(str(path)))
        assert set(result) == {"error", "path"}, result
        assert result["path"] == str(path)
        assert result["error"].startswith(f"Could not load {path}: ")
        return result["error"]

    def test_missing_file(self, tmp_path):
        srv, _ = self._open_project()
        error = self._fails(srv, tmp_path / "nope.mmp")
        assert "no such file" in error
        assert "Check the path" in error and "~/" in error
        assert "does not exist either" not in error

    def test_missing_folder_is_named(self, tmp_path):
        srv, _ = self._open_project()
        error = self._fails(srv, tmp_path / "no" / "such" / "x.mmp")
        assert "no such file" in error
        assert f"The folder {tmp_path / 'no' / 'such'} does not exist" in error

    @pytest.mark.parametrize("have,ask", [
        ("song.mmpz", "song.mmp"), ("song.mmp", "song.mmpz"),
        ("Song.MMPZ", "song.mmp")])
    def test_other_extension_is_suggested(self, tmp_path, have, ask):
        srv, _ = self._open_project()
        self._saved(tmp_path, have)
        error = self._fails(srv, tmp_path / ask)
        assert f"A similar file exists: {tmp_path / have}" in error
        assert "Check the path" not in error

    def test_a_close_name_is_suggested(self, tmp_path):
        srv, _ = self._open_project()
        self._saved(tmp_path, "my_song_v2.mmpz")
        self._saved(tmp_path, "totally-different.mmp")
        (tmp_path / "notes.txt").write_text("x")
        error = self._fails(srv, tmp_path / "my_song_v3.mmpz")
        assert str(tmp_path / "my_song_v2.mmpz") in error
        assert "totally-different" not in error and "notes.txt" not in error

    def test_relative_name_with_the_other_extension_in_the_default_folder(
            self, monkeypatch, tmp_path):
        from lmms_mcp import server as srv_mod
        srv, _ = self._open_project()
        default = tmp_path / "default"
        default.mkdir()
        self._saved(default, "song.mmpz")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setattr(srv_mod, "DEFAULT_PROJECTS_DIR", str(default))
        result = json.loads(srv.load_project("song.mmp"))
        # The error names the absolute path that was looked at first, says
        # where a relative path is looked up, and suggests the similar file.
        assert result["path"] == str(default / "song.mmp")
        error = result["error"]
        assert error.startswith(f"Could not load {default / 'song.mmp'}: ")
        assert "no such file" in error
        assert f"default projects folder ({default})" in error
        assert f"working folder ({elsewhere})" in error
        assert f"A similar file exists: {default / 'song.mmpz'}" in error

    def test_relative_name_missing_everywhere(self, monkeypatch, tmp_path):
        from lmms_mcp import server as srv_mod
        srv, proj = self._open_project()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            srv_mod, "DEFAULT_PROJECTS_DIR", str(tmp_path / "default"))
        result = json.loads(srv.load_project("sub/none.mmp"))
        target = tmp_path / "default" / "sub" / "none.mmp"
        assert result["path"] == str(target)
        assert f"The folder {target.parent} does not exist" in result["error"]
        assert srv.get_project() is proj

    def test_folder(self, tmp_path):
        srv, _ = self._open_project()
        folder = tmp_path / "songs"
        folder.mkdir()
        self._saved(folder, "a.mmp")
        self._saved(folder, "b.mmpz")
        (folder / "readme.txt").write_text("x")
        error = self._fails(srv, folder)
        assert "it is a folder, not a project file" in error
        assert f"{folder / 'a.mmp'}, {folder / 'b.mmpz'}" in error
        assert "readme.txt" not in error

    def test_empty_folder(self, tmp_path):
        srv, _ = self._open_project()
        error = self._fails(srv, tmp_path)
        assert "it is a folder" in error
        assert "no .mmp or .mmpz file" in error

    def test_unreadable_file(self, tmp_path):
        import os
        srv, _ = self._open_project()
        path = self._saved(tmp_path, "locked.mmp")
        path.chmod(0)
        try:
            if os.access(path, os.R_OK):
                pytest.skip("permissions are not enforced (running as root)")
            error = self._fails(srv, path)
        finally:
            path.chmod(0o644)
        assert "Permission denied" in error
        assert "permissions" in error

    def test_junk_file(self, tmp_path):
        srv, _ = self._open_project()
        path = tmp_path / "junk.mmp"
        path.write_text("this is not a project")
        error = self._fails(srv, path)
        assert "not a valid LMMS project file" in error
        assert "Cannot parse" not in error  # the path is not named twice
        assert error.count(str(path)) == 1
        assert ".mmpz" in error and "corrupt" in error

    def test_truncated_compressed_file(self, tmp_path):
        srv, _ = self._open_project()
        good = self._saved(tmp_path, "good.mmpz").read_bytes()
        path = tmp_path / "cut.mmpz"
        path.write_bytes(good[: len(good) // 2])
        assert "not a valid LMMS project file" in self._fails(srv, path)

    @pytest.mark.parametrize("content", [b"", b"abc"])
    def test_empty_and_tiny_files(self, tmp_path, content):
        srv, _ = self._open_project()
        path = tmp_path / "tiny.mmp"
        path.write_bytes(content)
        error = self._fails(srv, path)
        assert "not a valid LMMS project file" in error
        assert ("(the file is empty)" in error) == (content == b"")

    def test_xml_that_is_not_a_project(self, tmp_path):
        srv, _ = self._open_project()
        path = tmp_path / "page.mmp"
        path.write_text("<html><body/></html>")
        error = self._fails(srv, path)
        assert "not a valid LMMS project file" in error
        assert "<html>" in error and "<lmms-project>" in error

    @pytest.mark.parametrize("make", [
        lambda tmp: tmp / "nope.mmp",
        lambda tmp: tmp,
        lambda tmp: tmp / "junk.mmp",
    ])
    def test_the_open_project_is_unchanged_after_a_failure(
            self, tmp_path, make):
        srv, proj = self._open_project()
        proj._modified = True
        (tmp_path / "junk.mmp").write_text("junk")
        before = proj.get_xml_string()
        json.loads(srv.load_project(str(make(tmp_path))))
        assert srv.get_project() is proj
        assert proj.get_xml_string() == before
        assert proj.modified is True
        assert proj.path is None or proj.path == ""
        assert json.loads(srv.get_project_info())["track_count"] == 1

    def test_the_path_of_the_open_project_is_kept(self, tmp_path):
        srv, proj = self._open_project()
        path = self._saved(tmp_path, "kept.mmp")
        proj.save(str(path))
        json.loads(srv.load_project(str(tmp_path / "nope.mmp")))
        assert srv.get_project() is proj
        assert proj.path == str(path)

    def test_loading_still_works(self, tmp_path):
        srv, _ = self._open_project()
        path = self._saved(tmp_path, "ok.mmpz")
        result = json.loads(srv.load_project(str(path)))
        assert "error" not in result
        assert result["message"] == f"Loaded project from {path}"
        assert result["track_count"] == 0
        assert srv.get_project().path == str(path)

    def test_docstring_names_the_errors(self):
        from lmms_mcp import server as srv
        doc = srv.load_project.__doc__
        assert '"error"' in doc and "stays as it was" in doc
        assert "~" in doc

    def test_parser_accepts_the_old_root_names(self, tmp_path):
        for tag in ("lmms-project", "multimediaproject", "multimedia-project"):
            path = tmp_path / f"{tag}.mmp"
            path.write_text(f"<{tag} version='1.0'><head/></{tag}>")
            assert load_project(path).tag == tag

    def test_parser_errors(self, tmp_path):
        (tmp_path / "x.mmp").write_text("<other/>")
        with pytest.raises(ValueError, match="not a valid LMMS project file"):
            load_project(tmp_path / "x.mmp")
        (tmp_path / "short.mmp").write_bytes(b"ab")
        with pytest.raises(ValueError, match="not a valid LMMS project file"):
            load_project(tmp_path / "short.mmp")
        with pytest.raises(FileNotFoundError):
            load_project(tmp_path / "missing.mmp")


class TestRelativeProjectPaths:
    """A relative project path means the default projects folder in
    save_project, load_project and render_project (output_path), so the
    name given to one works in the others."""

    @pytest.fixture
    def folders(self, monkeypatch, tmp_path):
        from lmms_mcp import server as srv
        default = tmp_path / "default"
        default.mkdir()
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        monkeypatch.setattr(srv, "DEFAULT_PROJECTS_DIR", str(default))
        return default, cwd

    @staticmethod
    def _project(*track_names):
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        for name in track_names:
            srv.add_instrument_track(name)
        return srv, proj

    @staticmethod
    def _tracks(srv):
        return [t["name"] for t in json.loads(srv.get_project_info())["tracks"]]

    def test_save_relative_then_load_relative_is_the_same_file(
            self, folders):
        default, cwd = folders
        srv, _ = self._project("Saved")
        saved = json.loads(srv.save_project("sub/song.mmp"))
        assert saved["path"] == str(default / "sub" / "song.mmp")
        self._project("Other")
        result = json.loads(srv.load_project("sub/song.mmp"))
        assert "error" not in result and "note" not in result
        assert result["path"] == saved["path"]
        assert result["message"] == f"Loaded project from {saved['path']}"
        assert self._tracks(srv) == ["Saved"]
        assert srv.get_project().path == saved["path"]
        assert not any(cwd.iterdir())

    def test_a_plain_file_name_round_trips(self, folders):
        default, _ = folders
        srv, _ = self._project("Saved")
        srv.save_project("song.mmpz")
        self._project()
        result = json.loads(srv.load_project("song.mmpz"))
        assert result["path"] == str(default / "song.mmpz")
        assert self._tracks(srv) == ["Saved"]

    def test_a_file_only_in_the_working_folder_is_loaded_with_a_note(
            self, folders):
        default, cwd = folders
        srv, proj = self._project("Local")
        proj.save(str(cwd / "local.mmp"))
        self._project("Other")
        result = json.loads(srv.load_project("local.mmp"))
        assert "error" not in result
        assert result["path"] == str(cwd / "local.mmp")
        assert result["message"] == f"Loaded project from {cwd / 'local.mmp'}"
        assert "not in the default projects folder" in result["note"]
        assert str(default) in result["note"]
        assert f"working folder instead: {cwd / 'local.mmp'}" in result["note"]
        assert self._tracks(srv) == ["Local"]
        assert srv.get_project().path == str(cwd / "local.mmp")

    def test_a_relative_path_with_folders_falls_back_to_the_working_folder(
            self, folders):
        _, cwd = folders
        (cwd / "a").mkdir()
        srv, proj = self._project("Local")
        proj.save(str(cwd / "a" / "x.mmp"))
        result = json.loads(srv.load_project("a/x.mmp"))
        assert result["path"] == str(cwd / "a" / "x.mmp")
        assert "working folder" in result["note"]

    def test_a_file_in_both_places_loads_the_default_folder_one(
            self, folders):
        default, cwd = folders
        srv, proj = self._project("InDefault")
        proj.save(str(default / "song.mmp"))
        _, proj = self._project("InWorkingFolder")
        proj.save(str(cwd / "song.mmp"))
        self._project()
        result = json.loads(srv.load_project("song.mmp"))
        assert "error" not in result
        assert result["path"] == str(default / "song.mmp")
        assert self._tracks(srv) == ["InDefault"]
        assert str(cwd / "song.mmp") in result["note"]
        assert "working folder" in result["note"]
        assert "full path" in result["note"]
        # The full path of the other file loads that one, without a note.
        result = json.loads(srv.load_project(str(cwd / "song.mmp")))
        assert result["path"] == str(cwd / "song.mmp")
        assert "note" not in result
        assert self._tracks(srv) == ["InWorkingFolder"]

    def test_the_working_folder_is_the_default_folder_has_no_note(
            self, monkeypatch, tmp_path):
        from lmms_mcp import server as srv_mod
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(srv_mod, "DEFAULT_PROJECTS_DIR", str(tmp_path))
        srv, proj = self._project("Here")
        proj.save(str(tmp_path / "song.mmp"))
        result = json.loads(srv.load_project("song.mmp"))
        assert result["path"] == str(tmp_path / "song.mmp")
        assert "note" not in result

    def test_a_relative_default_folder_gives_an_absolute_path(
            self, monkeypatch, tmp_path):
        from lmms_mcp import server as srv_mod
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(srv_mod, "DEFAULT_PROJECTS_DIR", "rel/projects")
        srv, _ = self._project("Saved")
        saved = json.loads(srv.save_project("song.mmp"))
        assert saved["path"] == str(tmp_path / "rel" / "projects" / "song.mmp")
        result = json.loads(srv.load_project("song.mmp"))
        assert result["path"] == saved["path"]

    def test_absolute_and_tilde_paths_are_not_looked_up_elsewhere(
            self, folders, monkeypatch, tmp_path):
        default, cwd = folders
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        srv, proj = self._project("Abs")
        proj.save(str(tmp_path / "abs.mmp"))
        proj.save(str(home / "home.mmp"))
        # Same names in the other places must not be picked up instead.
        proj.save(str(default / "home.mmp"))
        self._project("Default")
        srv.save_project(str(default / "abs.mmp"))
        self._project()
        result = json.loads(srv.load_project(str(tmp_path / "abs.mmp")))
        assert result["path"] == str(tmp_path / "abs.mmp")
        assert "note" not in result
        assert self._tracks(srv) == ["Abs"]
        result = json.loads(srv.load_project("~/home.mmp"))
        assert result["path"] == str(home / "home.mmp")
        assert "note" not in result
        # A missing absolute path is not retried under the default folder
        # and its error does not describe a relative lookup.
        (default / "gone.mmp").write_bytes(b"x")
        result = json.loads(srv.load_project(str(tmp_path / "gone.mmp")))
        assert result["path"] == str(tmp_path / "gone.mmp")
        assert "relative path" not in result["error"]

    def test_a_folder_name_gives_the_folder_error(self, folders):
        default, _ = folders
        (default / "songs").mkdir()
        srv, _ = self._project()
        result = json.loads(srv.load_project("songs"))
        assert result["path"] == str(default / "songs")
        assert "it is a folder, not a project file" in result["error"]

    def test_a_folder_only_in_the_working_folder_gives_the_folder_error(
            self, folders):
        _, cwd = folders
        (cwd / "songs").mkdir()
        srv, _ = self._project()
        result = json.loads(srv.load_project("songs"))
        assert result["path"] == str(cwd / "songs")
        assert "it is a folder, not a project file" in result["error"]

    def test_a_file_beats_a_folder_of_the_same_name(self, folders):
        default, cwd = folders
        (default / "song.mmp").mkdir()
        srv, proj = self._project("Local")
        proj.save(str(cwd / "song.mmp"))
        result = json.loads(srv.load_project("song.mmp"))
        assert result["path"] == str(cwd / "song.mmp")
        assert "note" in result

    def test_a_failed_load_keeps_the_open_project(self, folders):
        default, _ = folders
        (default / "junk.mmp").write_text("junk")
        srv, proj = self._project("Keep")
        result = json.loads(srv.load_project("junk.mmp"))
        assert result["path"] == str(default / "junk.mmp")
        assert "not a valid LMMS project file" in result["error"]
        assert "relative path" not in result["error"]
        assert srv.get_project() is proj
        assert self._tracks(srv) == ["Keep"]

    def test_render_relative_output_goes_under_the_default_folder(
            self, folders, monkeypatch, tmp_path):
        default, cwd = folders
        srv, calls = TestRenderProjectQuality._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(output_path="out/take1.wav"))
        target = default / "out" / "take1.wav"
        assert result["success"] is True
        assert result["output_file"] == str(target)
        assert result["created_dirs"] == [str(default / "out")]
        assert read_pcm(target)[2] == 16
        assert not (cwd / "out").exists()

    def test_render_plain_file_name_goes_to_the_default_folder(
            self, folders, monkeypatch, tmp_path):
        default, cwd = folders
        srv, calls = TestRenderProjectQuality._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(output_path="mix.wav"))
        assert result["output_file"] == str(default / "mix.wav")
        assert (default / "mix.wav").is_file()
        assert "created_dirs" not in result
        assert list(cwd.iterdir()) == []

    def test_render_absolute_output_is_unchanged(
            self, folders, monkeypatch, tmp_path):
        default, _ = folders
        srv, calls = TestRenderProjectQuality._setup(monkeypatch, tmp_path)
        out = tmp_path / "elsewhere" / "x.wav"
        result = json.loads(srv.render_project(output_path=str(out)))
        assert result["output_file"] == str(out)
        assert list(default.iterdir()) == []

    def test_render_default_output_stays_next_to_the_project(
            self, folders, monkeypatch, tmp_path):
        default, _ = folders
        srv, calls = TestRenderProjectQuality._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project())
        assert result["output_file"] == str(tmp_path / "song.wav")
        assert list(default.iterdir()) == []

    def test_render_after_loading_by_a_relative_name(
            self, folders, monkeypatch, tmp_path):
        import subprocess
        from lmms_mcp import lmms_app
        default, _ = folders
        srv, _ = self._project("Saved")
        srv.save_project("song.mmp")
        self._project()
        srv.load_project("song.mmp")
        monkeypatch.setattr(
            lmms_app, "find_lmms_exe", lambda: Path("/fake/lmms"))
        fake = FakeLmms(sine(220, 0.3, 4))
        calls = fake.commands
        monkeypatch.setattr("subprocess.run", fake)
        result = json.loads(srv.render_project())
        assert result["output_file"] == str(default / "song.wav")
        cmd = calls[0]
        assert cmd[cmd.index("render") + 1] == str(default / "song.mmp")

    def test_docstrings_and_instructions_state_the_rule(self):
        from lmms_mcp import server as srv
        for tool in (srv.load_project, srv.save_project, srv.render_project):
            doc = " ".join(tool.__doc__.split())
            assert "relative to the default projects folder" in doc, tool
            assert "LMMS_PROJECTS_DIR" in doc, tool
        text = srv.build_instructions(srv._availability, "/some/dir")
        assert "A relative path in save_project, load_project" in text


class TestTildeExpansion:
    """A path starting with ~ is in the home folder in every tool that takes
    a file or folder path (nothing expands it before the server gets it)."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        # relative paths and the default projects folder stay elsewhere
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        from lmms_mcp import server as srv
        monkeypatch.setattr(
            srv, "DEFAULT_PROJECTS_DIR", str(tmp_path / "default"))
        return home

    @staticmethod
    def _new_project():
        from lmms_mcp import server as srv
        proj = LMMSProject()
        proj.new()
        srv.set_project(proj)
        return srv, proj

    def test_helper(self, home):
        from lmms_mcp.server import _expand_user
        assert _expand_user("~/songs/x.mmp") == str(home / "songs" / "x.mmp")
        assert _expand_user("~") == str(home)
        # everything else is unchanged
        for same in ("drums/kick01.ogg", "/abs/x.wav", "x.mmp", "",
                     "a/~/b.wav", "~nosuchuser_zzz/x.wav", "C:/x.wav",
                     "basses\\bass01.ogg"):
            assert _expand_user(same) == same

    def test_save_project(self, home, tmp_path):
        srv, proj = self._new_project()
        result = json.loads(srv.save_project("~/songs/x.mmp"))
        target = home / "songs" / "x.mmp"
        assert "error" not in result
        assert result["path"] == str(target)
        assert result["created_dirs"] == [str(home / "songs")]
        assert target.is_file()
        assert proj.path == str(target)
        # not a "~" folder under the default projects folder or the cwd
        assert not (tmp_path / "default").exists()
        assert not (tmp_path / "elsewhere" / "~").exists()

    def test_save_project_again_after_a_tilde_path(self, home):
        srv, proj = self._new_project()
        srv.save_project("~/x.mmpz")
        result = json.loads(srv.save_project())
        assert result["path"] == str(home / "x.mmpz")

    def test_load_project(self, home):
        srv, proj = self._new_project()
        proj.save(str(home / "song.mmp"))
        srv.add_instrument_track("Extra")  # differs from the file
        result = json.loads(srv.load_project("~/song.mmp"))
        assert "error" not in result
        assert result["message"] == f"Loaded project from {home / 'song.mmp'}"
        assert result["track_count"] == 0
        assert srv.get_project().path == str(home / "song.mmp")

    def test_load_project_error_names_the_expanded_path(self, home):
        srv, _ = self._new_project()
        result = json.loads(srv.load_project("~/nope.mmp"))
        assert result["path"] == str(home / "nope.mmp")
        assert result["error"].startswith(
            f"Could not load {home / 'nope.mmp'}: ")

    def test_render_project_output_path(self, home, monkeypatch, tmp_path):
        srv, calls = TestRenderProjectQuality._setup(monkeypatch, tmp_path)
        result = json.loads(srv.render_project(output_path="~/out/x.wav"))
        target = home / "out" / "x.wav"
        assert result["success"] is True
        assert result["output_file"] == str(target)
        assert result["created_dirs"] == [str(home / "out")]
        assert read_pcm(target)[2] == 16
        assert not (tmp_path / "elsewhere" / "~").exists()

    def test_set_audiofile_sample(self, home):
        srv, proj = self._new_project()
        srv.add_instrument_track("Kick", instrument="audiofileprocessor")
        sample = home / "samples" / "hit.wav"
        sample.parent.mkdir()
        sample.write_bytes(b"x")
        result = json.loads(
            srv.set_audiofile_sample(0, "~/samples/hit.wav"))
        assert "error" not in result
        assert result["file"] == str(sample)
        assert "warning" not in result  # it was looked for where it is
        assert str(sample) in result["message"]
        afp = find_tracks(proj.root)[0].find(
            "instrumenttrack/instrument/audiofileprocessor")
        assert afp.get("src") == str(sample)

    def test_sample_clip_tools(self, home):
        srv, proj = self._new_project()
        srv.add_sample_track("Loop")
        sample = str(home / "loop.wav")
        result = json.loads(srv.place_sample_clip(0, "~/loop.wav"))
        assert result["file"] == sample
        clips = find_tracks(proj.root)[0].findall("sampleclip")
        assert [c.get("src") for c in clips] == [sample]
        other = str(home / "other.wav")
        result = json.loads(srv.assign_sample_file(0, "~/other.wav"))
        assert result["file"] == other
        assert [c.get("src") for c in
                find_tracks(proj.root)[0].findall("sampleclip")] == [other]

    def test_factory_relative_sample_paths_stay_relative(
            self, home, monkeypatch, tmp_path):
        from lmms_mcp import lmms_app
        samples = tmp_path / "factory"
        (samples / "drums").mkdir(parents=True)
        (samples / "drums" / "kick01.ogg").write_bytes(b"x")
        monkeypatch.setattr(lmms_app, "get_samples_dir", lambda: samples)
        # a folder of the same name in the home folder changes nothing
        (home / "drums").mkdir()
        (home / "drums" / "kick01.ogg").write_bytes(b"x")
        srv, proj = self._new_project()
        srv.add_instrument_track("Kick", instrument="audiofileprocessor")
        srv.add_sample_track("Loop")
        result = json.loads(
            srv.set_audiofile_sample(0, "drums/kick01.ogg"))
        assert result["file"] == "drums/kick01.ogg"
        assert "warning" not in result
        afp = find_tracks(proj.root)[0].find(
            "instrumenttrack/instrument/audiofileprocessor")
        assert afp.get("src") == "drums/kick01.ogg"
        assert json.loads(srv.place_sample_clip(
            1, "drums/kick01.ogg"))["file"] == "drums/kick01.ogg"
        assert json.loads(srv.assign_sample_file(
            1, "drums/kick01.ogg"))["file"] == "drums/kick01.ogg"
        sc = find_tracks(proj.root)[1].findall("sampleclip")
        assert {c.get("src") for c in sc} == {"drums/kick01.ogg"}

    def test_set_instrument_file(self, home):
        srv, proj = self._new_project()
        srv.add_instrument_track("Flute", instrument="patman")
        pat = home / "flute.pat"
        pat.write_bytes(b"")
        result = json.loads(srv.set_instrument_file(0, "~/flute.pat"))
        assert "error" not in result, result
        assert result["file"] == str(pat)

    def test_vst_tools(self, home):
        srv, proj = self._new_project()
        vst = home / "vst" / "Synth.dll"
        vst.parent.mkdir()
        vst.write_bytes(b"MZ")
        scanned = json.loads(srv.scan_vst_directory("~/vst"))
        assert "error" not in scanned
        assert scanned["directory"] == str(home / "vst")
        assert [p["path"] for p in scanned["plugins"]] == [str(vst)]
        added = json.loads(srv.add_vst_track("Lead", "~/vst/Synth.dll"))
        assert "error" not in added, added
        assert added["vst"] == str(vst)

    def test_load_zyn_preset(self, home):
        srv, proj = self._new_project()
        srv.add_instrument_track("Z", instrument="zynaddsubfx")
        (home / "my.xiz").write_bytes(
            TestLoadZynPresetReport._xiz("From Home"))
        result = json.loads(srv.load_zyn_preset(0, "~/my.xiz"))
        assert "error" not in result, result
        assert result["file"] == str(home / "my.xiz")
        assert result["requested"] == "~/my.xiz"
        assert result["internal_name"] == "From Home"

    def test_load_instrument_preset(self, lab, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        lab.put("TripleOscillator/Home.xpf", _xpf("Home"))
        result = lab.load(0, "~/presets/TripleOscillator/Home.xpf")
        assert "error" not in result, result
        assert result["file"] == str(
            tmp_path / "presets" / "TripleOscillator" / "Home.xpf")

    def test_docstrings_mention_the_tilde(self):
        from lmms_mcp import server as srv
        for tool in (
                srv.save_project, srv.load_project, srv.render_project,
                srv.set_audiofile_sample, srv.set_instrument_file,
                srv.place_sample_clip, srv.assign_sample_file,
                srv.add_vst_track, srv.scan_vst_directory,
                srv.load_zyn_preset, srv.load_instrument_preset):
            assert "~" in tool.__doc__, tool.__name__
