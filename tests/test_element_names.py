"""The names and starting values of what the server writes into a project.

LMMS ignores an attribute name it does not know, so a wrong name does
nothing and nothing says so. The tables below are what LMMS 1.2.2 itself
writes (and so reads): taken from its default template
(/usr/share/lmms/projects/templates/default.mpt) and demo projects, and
checked by loading projects in LMMS 1.2.2 (see the notes on each test).
"""

from xml.etree import ElementTree as ET

import pytest

from lmms_mcp import effects, xml_parser
from lmms_mcp.project import LMMSProject
from lmms_mcp.xml_parser import (
    ARPEGGIATOR_DEFAULTS,
    CHORDCREATOR_DEFAULTS,
    ELDATA_DEFAULTS,
    ENVELOPE_DEFAULTS,
    MIDIPORT_DEFAULTS,
    add_instrument_track,
    add_pattern_track,
    create_empty_project,
    find_tracks,
)

# --- what LMMS 1.2.2 saves for a new instrument track -----------------------

LMMS_ENVELOPE = {
    "pdel": "0", "att": "0", "hold": "0.5", "dec": "0.5", "sustain": "0.5",
    "rel": "0.1", "amt": "0",
    "lpdel": "0", "latt": "0", "lspd": "0.1", "lspd_numerator": "4",
    "lspd_denominator": "4", "lspd_syncmode": "0", "lamt": "0", "lshp": "0",
    "x100": "0", "ctlenvamt": "0", "userwavefile": "",
}
LMMS_ARPEGGIATOR = {
    "arp-enabled": "0", "arp": "0", "arprange": "1", "arpcycle": "0",
    "arpskip": "0", "arpmiss": "0", "arptime": "100",
    "arptime_numerator": "4", "arptime_denominator": "4",
    "arptime_syncmode": "0", "arpgate": "100", "arpdir": "0", "arpmode": "0",
}
LMMS_CHORDCREATOR = {"chord-enabled": "0", "chord": "0", "chordrange": "1"}
LMMS_ELDATA = {"fcut": "14000", "fres": "0.5", "ftype": "0", "fwet": "0"}
LMMS_MIDIPORT = {
    "inputcontroller": "0", "outputcontroller": "0", "inputchannel": "0",
    "outputchannel": "1", "outputprogram": "1", "fixedinputvelocity": "-1",
    "fixedoutputvelocity": "-1", "fixedoutputnote": "-1",
    "basevelocity": "63", "readable": "0", "writable": "0",
}
LMMS_INSTRUMENTTRACK = {
    "pan", "fxch", "usemasterpitch", "pitchrange", "pitch", "basenote", "vol",
}

# Names the server used to write for the envelopes and the arpeggiator;
# LMMS 1.2.2 reads none of them (verified: an envelope written with them
# has no effect, with the names above it does).
OLD_ENVELOPE_NAMES = {"attack", "decay", "amount", "syncmode"}
OLD_ARPEGGIATOR_NAMES = {"arpspeed", "arpType", "arpcenter", "syncmode"}


def fresh_track(instrument="tripleoscillator"):
    root = create_empty_project(lmms_version="1.2.2")
    add_instrument_track(root, "T", instrument=instrument)
    return find_tracks(root)[0].find("instrumenttrack")


class TestFreshInstrumentTrack:
    def test_envelopes_have_the_names_and_values_lmms_saves(self):
        eldata = fresh_track().find("eldata")
        assert [e.tag for e in eldata] == ["elvol", "elcut", "elres"]
        for envelope in eldata:
            assert envelope.attrib == LMMS_ENVELOPE, envelope.tag
            assert not OLD_ENVELOPE_NAMES & set(envelope.attrib)

    def test_envelope_attribute_names_are_lmms_122s(self):
        # The names LMMS 1.2.2 saves for an envelope (checked against the
        # demo projects it ships).
        names = {
            "amt", "att", "ctlenvamt", "dec", "hold", "lamt", "latt",
            "lpdel", "lshp", "lspd", "lspd_denominator", "lspd_numerator",
            "lspd_syncmode", "pdel", "rel", "sustain", "userwavefile", "x100",
        }
        for envelope in fresh_track().find("eldata"):
            assert set(envelope.attrib) == names

    def test_envelopes_start_off(self):
        # The same sound as before the names were fixed: an envelope with
        # amount 0 and an LFO with amount 0 change nothing.
        for envelope in fresh_track().find("eldata"):
            assert envelope.get("amt") == "0"
            assert envelope.get("lamt") == "0"

    def test_arpeggiator_has_the_names_and_values_lmms_saves(self):
        arp = fresh_track().find("arpeggiator")
        assert arp.attrib == LMMS_ARPEGGIATOR
        assert not OLD_ARPEGGIATOR_NAMES & set(arp.attrib)
        assert arp.get("arp-enabled") == "0"

    def test_arpeggiator_attribute_names_are_lmms_122s(self):
        assert set(fresh_track().find("arpeggiator").attrib) == {
            "arp", "arp-enabled", "arpcycle", "arpdir", "arpgate", "arpmiss",
            "arpmode", "arprange", "arpskip", "arptime", "arptime_denominator",
            "arptime_numerator", "arptime_syncmode",
        }

    def test_other_parts_keep_their_names_and_values(self):
        track = fresh_track()
        assert track.find("eldata").attrib == LMMS_ELDATA
        assert track.find("chordcreator").attrib == LMMS_CHORDCREATOR
        assert track.find("chordcreator").get("chord-enabled") == "0"
        assert track.find("midiport").attrib == LMMS_MIDIPORT
        assert set(track.attrib) == LMMS_INSTRUMENTTRACK
        assert track.find("fxchain").attrib == {
            "numofeffects": "0", "enabled": "0"}

    def test_tables_are_the_values_written(self):
        assert ELDATA_DEFAULTS == LMMS_ELDATA
        assert ENVELOPE_DEFAULTS == LMMS_ENVELOPE
        assert CHORDCREATOR_DEFAULTS == LMMS_CHORDCREATOR
        assert ARPEGGIATOR_DEFAULTS == LMMS_ARPEGGIATOR
        assert MIDIPORT_DEFAULTS == LMMS_MIDIPORT

    @pytest.mark.parametrize("instrument", [
        "tripleoscillator", "kicker", "audiofileprocessor", "zynaddsubfx",
        "vibedstrings", "sf2player",
    ])
    def test_every_instrument_gets_the_same_parts(self, instrument):
        track = fresh_track(instrument)
        assert track.find("eldata/elvol").attrib == LMMS_ENVELOPE
        assert track.find("arpeggiator").attrib == LMMS_ARPEGGIATOR

    def test_pattern_track_instrument_uses_the_same_parts(self):
        root = create_empty_project(lmms_version="1.2.2")
        add_pattern_track(root, "Beat")
        inner = find_tracks(root)[0].find("bbtrack/trackcontainer/track")
        assert inner.find("instrumenttrack/eldata/elres").attrib == LMMS_ENVELOPE
        assert inner.find("instrumenttrack/arpeggiator").attrib == (
            LMMS_ARPEGGIATOR)

    def test_the_elements_do_not_share_the_tables(self):
        # Editing one track's element changes neither the tables nor the
        # next track.
        track = fresh_track()
        track.find("eldata/elvol").set("amt", "1")
        track.find("arpeggiator").set("arp-enabled", "1")
        track.find("eldata").set("fcut", "500")
        assert ENVELOPE_DEFAULTS["amt"] == "0"
        assert ARPEGGIATOR_DEFAULTS["arp-enabled"] == "0"
        assert ELDATA_DEFAULTS["fcut"] == "14000"
        other = fresh_track()
        assert other.find("eldata/elvol").get("amt") == "0"
        assert other.find("eldata/elcut").get("amt") == "0"
        assert other.find("arpeggiator").get("arp-enabled") == "0"


class TestOldProjectsAreNotRewritten:
    """A project made by an older version of the server keeps the attributes
    it has: nothing converts them on load, edit or save."""

    OLD = (
        '<lmms-project version="1.0" creator="LMMS MCP Server" '
        'creatorversion="1.2.0" type="song">'
        '<head bpm="120" mastervol="100" masterpitch="0" '
        'timesig_numerator="4" timesig_denominator="4"/>'
        '<song><trackcontainer type="song">'
        '<track muted="0" type="0" name="Old" solo="0">'
        '<instrumenttrack pan="0" fxch="0" usemasterpitch="1" pitchrange="1" '
        'pitch="0" basenote="57" vol="100">'
        '<instrument name="tripleoscillator"><tripleoscillator/></instrument>'
        '<eldata fres="0.5" ftype="0" fcut="14000" fwet="0">'
        '<elvol lspd_denominator="4" sustain="0.5" lspd_numerator="4" '
        'attack="0.5" decay="0.5" hold="0" amount="1"/>'
        '<elcut lspd_denominator="4" sustain="0.5" lspd_numerator="4" '
        'attack="0" decay="0.5" hold="0" amount="0"/>'
        '<elres lspd_denominator="4" sustain="0.5" lspd_numerator="4" '
        'attack="0" decay="0.5" hold="0" amount="0"/>'
        '</eldata>'
        '<chordcreator chord="0" chordrange="1" chord-enabled="0"/>'
        '<arpeggiator arp="0" arp-enabled="1" arpcenter="0" arpdir="0" '
        'arprange="1" arpspeed="4" arpType="0"/>'
        '<midiport inputcontroller="0" readable="0" writable="0"/>'
        '<fxchain numofeffects="0" enabled="0"/>'
        '</instrumenttrack></track></trackcontainer></song></lmms-project>'
    )

    @staticmethod
    def snapshot(root):
        return {
            tag: [dict(e.attrib) for e in root.iter(tag)]
            for tag in ("eldata", "elvol", "elcut", "elres", "arpeggiator",
                        "chordcreator", "midiport")
        }

    def test_load_edit_save_keeps_the_old_attributes(self, tmp_path):
        source = tmp_path / "old.mmp"
        source.write_text(self.OLD, encoding="utf-8")
        before = self.snapshot(ET.fromstring(self.OLD))
        assert before["elvol"][0]["amount"] == "1"  # the fixture is old

        proj = LMMSProject()
        proj.load(source)
        proj.add_note(0, key=60, pos=0, length=48)
        proj.set_tempo(100)
        proj.add_track("instrument", "New")
        proj.save(tmp_path / "out.mmp")

        saved = LMMSProject()
        saved.load(tmp_path / "out.mmp")
        after = self.snapshot(saved.root)
        # the old track is exactly as it was ...
        for tag, old_elements in before.items():
            assert after[tag][0] == old_elements[0], tag
        # ... and only the new track has the names LMMS reads
        assert after["elvol"][1] == LMMS_ENVELOPE
        assert after["arpeggiator"][1] == LMMS_ARPEGGIATOR


class TestFreshEffect:
    def test_effect_names_are_the_ones_lmms_reads(self):
        # LMMS 1.2.2 reads the tempo sync mode of the auto-quit time as
        # autoquit_syncmode: with a plain syncmode (which only a project
        # from an older LMMS may have) the echoes of a delay with
        # autoquit_syncmode=2 were kept, with syncmode=2 they were cut.
        root = create_empty_project(lmms_version="1.2.2")
        add_instrument_track(root, "T")
        track = find_tracks(root)[0].find("instrumenttrack")
        effects.add_effect(track, "delay")
        effect = track.find("fxchain/effect")
        assert set(effect.attrib) == {
            "on", "wet", "autoquit", "autoquit_numerator",
            "autoquit_denominator", "autoquit_syncmode", "gate", "name",
        }
        assert effect.get("autoquit_syncmode") == "0"
        assert effect.get("gate") == "0"
        assert "syncmode" not in effect.attrib


class TestOtherElementNames:
    """Every other element the server creates has the names LMMS 1.2.2
    saves (the ones it reads); extra names are allowed where listed."""

    @staticmethod
    def build():
        root = create_empty_project(lmms_version="1.2.2")
        xml_parser.add_instrument_track(root, "Lead")
        xml_parser.add_note_to_track(root, 0, key=60, pos=0, length=48)
        xml_parser.add_sample_track(root, "Samples")
        xml_parser.add_sample_clip(root, 1, "drums/kick01.ogg", 0, 96)
        xml_parser.add_pattern_track(root, "Beat")
        xml_parser.add_note_to_track(root, 2, key=60, pos=0, length=12)
        xml_parser.add_bb_clip(root, 2, 0, 192)
        xml_parser.add_mixer_channel(root, "Bass")
        xml_parser.add_automation_curve_track(
            root, "Swell", [(0, 20.0), (192, 100.0)], target_id=4242)
        xml_parser.add_automation_track(root, "Empty")
        return root

    # element path -> (names LMMS 1.2.2 saves, extra names that are fine)
    EXPECTED = {
        "lmms-project": ({"version", "creator", "creatorversion", "type"}, set()),
        "head": ({"bpm", "masterpitch", "mastervol", "timesig_numerator",
                  "timesig_denominator"}, set()),
        "song/trackcontainer": ({"type", "visible", "width", "height", "x",
                                 "y", "maximized", "minimized"}, set()),
        "song/trackcontainer/track": ({"muted", "name", "solo", "type"}, set()),
        "song/trackcontainer/track/instrumenttrack": (
            LMMS_INSTRUMENTTRACK, set()),
        "song/trackcontainer/track/sampletrack": ({"pan", "vol"}, set()),
        "song/trackcontainer/track/pattern": (
            {"len", "muted", "name", "pos", "steps", "type"}, set()),
        "song/trackcontainer/track/pattern/note": (
            {"key", "len", "pan", "pos", "vol"}, set()),
        "song/trackcontainer/track/bbtco": (
            {"color", "len", "muted", "name", "pos", "usestyle"}, set()),
        # LMMS 1.2.2 reads the clip of a sample track whatever its tag is
        # (a track makes a clip of every child that is not its settings) and
        # from it pos, len, muted and src; "off" is the 1.3 start offset.
        "song/trackcontainer/track/sampleclip": (
            {"pos", "len", "muted", "src"}, {"off"}),
        "song/trackcontainer/track/bbtrack/trackcontainer": (
            {"type", "visible", "width", "height", "x", "y", "maximized",
             "minimized"}, set()),
        # outValue, inTan, outTan, lockedTan, off and autoresize are the
        # names of LMMS 1.3; LMMS 1.2.2 ignores them.
        "song/trackcontainer/track/automationpattern": (
            {"pos", "len", "name", "prog", "tens", "mute"},
            {"off", "autoresize"}),
        "song/trackcontainer/track/automationpattern/time": (
            {"pos", "value"}, {"outValue", "inTan", "outTan", "lockedTan"}),
        "song/trackcontainer/track/automationpattern/object": ({"id"}, set()),
        "song/fxmixer": ({"visible", "width", "height", "x", "y", "maximized",
                          "minimized"}, set()),
        "song/fxmixer/fxchannel": (
            {"num", "muted", "volume", "name", "soloed"}, set()),
        "song/fxmixer/fxchannel/send": ({"channel", "amount"}, set()),
        "song/fxmixer/fxchannel/fxchain": ({"numofeffects", "enabled"}, set()),
        "song/timeline": ({"lp0pos", "lp1pos", "lpstate"}, set()),
        "song/ControllerRackView": ({"visible", "width", "height", "x", "y",
                                     "maximized", "minimized"}, set()),
        "song/pianoroll": ({"visible", "width", "height", "x", "y",
                            "maximized", "minimized"}, set()),
        "song/automationeditor": ({"visible", "width", "height", "x", "y",
                                   "maximized", "minimized"}, set()),
        "song/projectnotes": ({"visible", "width", "height", "x", "y",
                               "maximized", "minimized"}, set()),
    }

    @pytest.mark.parametrize("path", sorted(EXPECTED))
    def test_names(self, path):
        root = self.build()
        found = [root] if path == "lmms-project" else root.findall(path)
        assert found, f"the project has no {path}"
        saved, extra = self.EXPECTED[path]
        for element in found:
            names = set(element.attrib)
            assert saved <= names, (path, saved - names)
            assert names <= saved | extra, (path, names - saved - extra)

    def test_an_automated_value_is_a_child_with_lmms_names(self):
        # LMMS saves a model that is automated as <vol id= value=
        # scale_type=/> instead of the attribute.
        root = create_empty_project(lmms_version="1.2.2")
        add_instrument_track(root, "T")
        model_id, _ = xml_parser.resolve_automation_target(
            root, "track", 0, "volume")
        settings = find_tracks(root)[0].find("instrumenttrack")
        assert "vol" not in settings.attrib
        assert settings.find("vol").attrib == {
            "id": str(model_id), "value": "100", "scale_type": "linear"}

    def test_the_instrument_in_a_pattern_track_is_written_like_any_other(self):
        # the instrument track inside a pattern track is written like any
        # other instrument track
        root = self.build()
        inner = root.find(
            "song/trackcontainer/track/bbtrack/trackcontainer/track")
        assert set(inner.attrib) == {"muted", "name", "solo", "type"}
        assert set(inner.find("instrumenttrack").attrib) == LMMS_INSTRUMENTTRACK
