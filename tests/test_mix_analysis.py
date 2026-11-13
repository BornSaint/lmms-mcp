"""mix_analysis: sections, matching the files of rendertracks to tracks, and
the balance report (levels, flags and the suggestions' math) on synthetic
per-bar statistics. No audio and no LMMS."""

import math

import pytest

from lmms_mcp import audio_analysis as aa
from lmms_mcp import mix_analysis as ma
from lmms_mcp.audio_analysis import BarStats
from lmms_mcp.mix_analysis import (
    PeakProbe, RenderedTrack, Section, SectionError, TrackStem)

COUNT = 1000  # samples of every synthetic bar


def db(x):
    return 20 * math.log10(x)


def bar(rms, peak=None, count=COUNT):
    """A BarStats with the given RMS (linear) and a peak (default: 2 x RMS)."""
    return BarStats(2 * rms if peak is None else peak, rms * rms * count, count)


def level(rms_db):
    return 10 ** (rms_db / 20)


def steady(rms_db, bars, on=None, peak=None):
    """`bars` bars at rms_db dBFS RMS where `on` (a set of bar numbers,
    default all) plays, digital silence elsewhere."""
    return [bar(level(rms_db), peak) if on is None or i in on else bar(0.0, 0.0)
            for i in range(bars)]


def mix_of(tracks):
    """The mix as the energy sum of the stems, bar by bar (like uncorrelated
    signals), with the largest peak."""
    return [
        BarStats(max(b.peak for b in column),
                 sum(b.squares for b in column), COUNT)
        for column in zip(*[t.bars for t in tracks])]


def stem(name, rms_db, bars, index=0, volume=100.0, **kw):
    return TrackStem(name, steady(rms_db, bars, kw.pop("on", None)),
                     index, "tripleoscillator", volume, **kw)


def report(tracks, sections, probes=None, master=1.0):
    return ma.analyze_balance(mix_of(tracks), tracks, sections, probes, master)


def flags_of(result, kind):
    return [f for f in result["flags"] if f["type"] == kind]


class TestParseSections:
    def test_valid_sections(self):
        result = ma.parse_sections([
            {"name": "verse", "bars": [4, 12]},
            {"name": " hook ", "bars": (12, 20)},
            {"name": "all", "bars": [0.0, 44]},
        ])
        assert result == [Section("verse", 4, 12), Section("hook", 12, 20),
                          Section("all", 0, 44)]

    def test_sections_may_overlap_or_leave_gaps(self):
        result = ma.parse_sections([
            {"name": "a", "bars": [0, 8]}, {"name": "b", "bars": [4, 6]},
            {"name": "c", "bars": [20, 22]}])
        assert [(s.start, s.end) for s in result] == [(0, 8), (4, 6), (20, 22)]

    @pytest.mark.parametrize("raw, text", [
        (None, "non-empty list"),
        ([], "non-empty list"),
        ("hook", "non-empty list"),
        ({"name": "a", "bars": [0, 4]}, "non-empty list"),
        (["hook"], "sections[0] must be an object"),
        ([{"bars": [0, 4]}], "needs a \"name\""),
        ([{"name": "  ", "bars": [0, 4]}], "needs a \"name\""),
        ([{"name": 3, "bars": [0, 4]}], "needs a \"name\""),
        ([{"name": "a"}], "\"bars\" must be [start, end]"),
        ([{"name": "a", "bars": [1]}], "\"bars\" must be [start, end]"),
        ([{"name": "a", "bars": [1, 2, 3]}], "\"bars\" must be [start, end]"),
        ([{"name": "a", "bars": "0-4"}], "\"bars\" must be [start, end]"),
        ([{"name": "a", "bars": [0, 4.5]}], "whole numbers"),
        ([{"name": "a", "bars": ["0", 4]}], "whole numbers"),
        ([{"name": "a", "bars": [True, 4]}], "whole numbers"),
        ([{"name": "a", "bars": [-1, 4]}], "negative"),
        ([{"name": "a", "bars": [4, 4]}], "is empty"),
        ([{"name": "a", "bars": [8, 4]}], "is empty"),
        ([{"name": "a", "bars": [0, 4], "start": 1}], "unknown keys ['start']"),
        ([{"name": "a", "bars": [0, 4]}, {"name": "a", "bars": [4, 8]}],
         "used twice"),
    ])
    def test_bad_sections_are_refused_with_the_format(self, raw, text):
        with pytest.raises(SectionError) as info:
            ma.parse_sections(raw)
        assert text in str(info.value)

    def test_the_error_names_the_section_and_explains_exclusive_ends(self):
        with pytest.raises(SectionError) as info:
            ma.parse_sections([{"name": "hook", "bars": [12, 12]}])
        message = str(info.value)
        assert "sections[0] 'hook'" in message
        assert "exclusive" in message

    def test_too_many_sections(self):
        raw = [{"name": f"s{i}", "bars": [i, i + 1]}
               for i in range(ma.MAX_SECTIONS + 1)]
        with pytest.raises(SectionError, match="Too many"):
            ma.parse_sections(raw)

    def test_the_format_is_in_the_message(self):
        with pytest.raises(SectionError) as info:
            ma.parse_sections([{"name": "a", "bars": [0, 4.5]}])
        assert "0-based" in str(info.value) and "exclusive" in str(info.value)


class TestDefaultAndFittedSections:
    def test_blocks_of_four_bars_with_a_short_last_block(self):
        sections = ma.default_sections(11)
        assert [(s.start, s.end) for s in sections] == [(0, 4), (4, 8), (8, 11)]
        assert [s.name for s in sections] == [
            "bars 0-4", "bars 4-8", "bars 8-11"]

    def test_blocks_double_when_there_are_more_than_the_maximum(self):
        assert len(ma.default_sections(64)) == 16
        sections = ma.default_sections(65)
        assert sections[0].end == 8 and len(sections) == 9
        sections = ma.default_sections(200)
        assert sections[0].end == 16 and len(sections) == 13

    def test_one_bar_song(self):
        assert [(s.start, s.end) for s in ma.default_sections(1)] == [(0, 1)]

    def test_fit_cuts_an_end_past_the_song(self):
        fitted, notes = ma.fit_sections(
            [Section("a", 0, 4), Section("outro", 40, 48)], 44)
        assert [(s.start, s.end) for s in fitted] == [(0, 4), (40, 44)]
        assert len(notes) == 1 and "'outro'" in notes[0] and "[40, 44]" in notes[0]

    def test_fit_leaves_out_a_section_after_the_song(self):
        fitted, notes = ma.fit_sections(
            [Section("a", 0, 4), Section("late", 44, 48)], 44)
        assert [s.name for s in fitted] == ["a"]
        assert "'late'" in notes[0] and "left out" in notes[0]


class TestMatchStems:
    def tracks(self, *names):
        return [RenderedTrack(i * 2, n) for i, n in enumerate(names)]

    def files(self, *names):
        return [(i + 1, n, f"/x/{i + 1}_{n}.wav") for i, n in enumerate(names)]

    def test_files_match_the_tracks_by_position(self):
        rendered = self.tracks("Lead", "Bass", "Lead")
        matched, notes = ma.match_stems(
            rendered, self.files("Lead", "Bass", "Lead"))
        assert [m[0] for m in matched] == rendered   # two tracks named Lead
        assert [m[2] for m in matched] == [
            "/x/1_Lead.wav", "/x/2_Bass.wav", "/x/3_Lead.wav"]
        assert notes == []

    def test_files_are_taken_in_number_order(self):
        rendered = self.tracks("A", "B", "C")
        files = list(reversed(self.files("A", "B", "C")))
        matched, _ = ma.match_stems(rendered, files)
        assert [m[1] for m in matched] == ["A", "B", "C"]

    def test_characters_lmms_leaves_out_of_a_file_name_still_match(self):
        rendered = self.tracks("Ba/ss é+1")
        matched, notes = ma.match_stems(rendered, self.files("Bass é+1"))
        assert matched[0][0] is rendered[0] and notes == []

    def test_instruments_of_pattern_tracks_have_no_track(self):
        rendered = self.tracks("Lead")
        matched, notes = ma.match_stems(
            rendered, self.files("Lead", "Beat", "Hat"))
        assert [m[0] for m in matched] == [rendered[0], None, None]
        assert [m[1] for m in matched] == ["Lead", "Beat", "Hat"]
        assert notes == []

    def test_a_name_that_is_not_at_its_position_is_found_by_name(self):
        rendered = self.tracks("A", "B")
        files = [(1, "B", "/x/1_B.wav"), (2, "A", "/x/2_A.wav")]
        matched, notes = ma.match_stems(rendered, files)
        assert [m[0].name for m in matched] == ["B", "A"]
        assert notes == []

    def test_an_unknown_file_is_reported_without_a_track_and_noted(self):
        rendered = self.tracks("A", "B")
        matched, notes = ma.match_stems(
            rendered, [(1, "A", "/x/1_A.wav"), (2, "Zed", "/x/2_Zed.wav")])
        assert matched[1][0] is None and matched[1][1] == "Zed"
        assert any("2_Zed.wav does not match track 'B'" in n for n in notes)
        assert any("Track 2 'B' has no file" in n for n in notes)

    def test_a_track_without_a_file_is_noted(self):
        rendered = self.tracks("A", "B")
        _, notes = ma.match_stems(rendered, self.files("A"))
        assert notes == [
            "Track 2 'B' has no file in the render output and is not in the "
            "report."]

    def test_stem_file_name_pattern(self):
        m = ma.STEM_NAME.match("10_Open Hat.wav")
        assert m.groups() == ("10", "Open Hat")
        assert ma.STEM_NAME.match("mix.wav") is None
        assert ma.STEM_NAME.match("3_a_b.wav").group(2) == "a_b"


class TestLevelsAndActiveBars:
    def test_levels_sections_and_share(self):
        loud = stem("Loud", -20, 8, index=0)
        soft = stem("Soft", -30, 8, index=1)
        result = report([loud, soft], [Section("a", 0, 4), Section("b", 4, 8)])
        assert result["mix"]["sections"] == [
            {"name": "a", "bars": [0, 4], "peak_dbfs": -14.0,
             "rms_dbfs": -19.6},
            {"name": "b", "bars": [4, 8], "peak_dbfs": -14.0,
             "rms_dbfs": -19.6}]
        t_loud, t_soft = result["tracks"]
        assert (t_loud["index"], t_loud["name"]) == (0, "Loud")
        assert t_loud["instrument"] == "tripleoscillator"
        assert t_loud["volume"] == 100.0
        assert t_loud["peak_dbfs"] == -14.0 and t_loud["rms_dbfs"] == -20.0
        assert t_soft["rms_dbfs"] == -30.0
        # energy share: 10^-2 / (10^-2 + 10^-3)
        assert t_loud["sections"]["a"] == {
            "rms_dbfs": -20.0, "peak_dbfs": -14.0, "share_pct": 90.9}
        assert t_soft["sections"]["a"]["share_pct"] == 9.1
        assert t_loud["active_bars"] == [[0, 8]]
        assert "silent_in" not in t_loud

    def test_active_bars_use_the_minus_60_dbfs_threshold(self):
        quiet = 0.00099   # -60.09 dBFS: not active
        audible = 0.00101  # -59.9 dBFS: active
        bars = [bar(quiet), bar(audible), bar(0.0, 0.0), bar(audible),
                bar(audible), bar(quiet)]
        track = TrackStem("T", bars, 0, "x", 100.0)
        result = ma.analyze_balance(
            mix_of([track]), [track], [Section("all", 0, 6)])
        entry = result["tracks"][0]
        assert entry["active_bars"] == [[1, 2], [3, 5]]
        # RMS over the 3 active bars only, not over the silence between
        assert entry["rms_dbfs"] == round(db(audible), 1)
        assert entry["sections"]["all"]["active_bars"] == 3
        assert ma.ACTIVE_BAR_RMS == pytest.approx(10 ** (-60 / 20))

    def test_a_track_silent_in_a_section_is_listed_there(self):
        late = stem("Late", -20, 8, on={4, 5, 6, 7})
        other = stem("Other", -25, 8, index=1)
        result = report([late, other], [Section("a", 0, 4), Section("b", 4, 8)])
        entry = result["tracks"][0]
        assert entry["silent_in"] == ["a"]
        assert list(entry["sections"]) == ["b"]
        assert entry["active_bars"] == [[4, 8]]
        # a silent track has no share in the section
        assert "a" not in entry["sections"]

    def test_a_track_that_is_silent_everywhere_is_noted(self):
        silent = TrackStem("Ghost", steady(-20, 4, on=set()), 3, "x", 100.0)
        other = stem("Other", -25, 4)
        result = report([silent, other], [Section("a", 0, 4)])
        entry = result["tracks"][0]
        assert entry["rms_dbfs"] is None and entry["active_bars"] == []
        assert entry["silent_in"] == ["a"]
        assert any("Track 3 'Ghost' is silent in the whole render" in n
                   for n in result["notes"])

    def test_active_bar_ranges_are_capped(self):
        on = set(range(0, 2 * ma.MAX_RANGES + 6, 2))
        track = stem("Choppy", -20, 2 * ma.MAX_RANGES + 6, on=on)
        result = ma.analyze_balance(
            mix_of([track]), [track], [Section("all", 0, len(track.bars))])
        entry = result["tracks"][0]
        assert len(entry["active_bars"]) == ma.MAX_RANGES
        assert entry["active_bars_more_ranges"] == len(on) - ma.MAX_RANGES

    def test_a_pattern_instrument_has_no_index_or_volume(self):
        track = TrackStem("Beat", steady(-20, 4), None, "pattern instrument")
        result = ma.analyze_balance(
            mix_of([track]), [track], [Section("all", 0, 4)])
        entry = result["tracks"][0]
        assert entry["index"] is None and "volume" not in entry

    def test_an_automated_volume_is_marked(self):
        track = stem("Auto", -20, 4, volume_automated=True)
        result = ma.analyze_balance(
            mix_of([track]), [track], [Section("all", 0, 4)])
        assert result["tracks"][0]["volume_automated"] is True


class TestDominant:
    SECTIONS = [Section("all", 0, 8)]

    def tracks(self, kick_db, volume=100.0):
        return [stem("Kick", kick_db, 8, index=0, volume=volume),
                stem("Pad", -30, 8, index=1), stem("Bass", -30, 8, index=2)]

    def test_a_track_far_above_the_others_is_dominant(self):
        result = report(self.tracks(-20), self.SECTIONS)
        [flag] = flags_of(result, "dominant")
        assert (flag["track"], flag["name"]) == (0, "Kick")
        assert flag["sections"] == ["all"]
        assert flag["worst"] == {
            "section": "all", "rms_dbfs": -20.0, "others_mean_dbfs": -30.0,
            "above_others_db": 10.0}
        # to 3 dB above the others: -7 dB; 100 * 10^(-7/20) = 44.7
        assert flag["gain_db"] == -7.0
        assert flag["suggestion"] == "set_track_volume(0, 45): -7.0 dB (from 100)"
        assert "Kick" in flag["summary"] and "10.0 dB" in flag["summary"]

    def test_the_suggestion_scales_the_current_volume(self):
        result = report(self.tracks(-20, volume=80.0), self.SECTIONS)
        [flag] = flags_of(result, "dominant")
        # 80 * 10^(-7/20) = 35.9
        assert flag["suggestion"] == "set_track_volume(0, 36): -7.0 dB (from 80)"

    def test_6_db_is_the_threshold(self):
        assert ma.DOMINANT_DB == 6.0
        assert flags_of(report(self.tracks(-24.0), self.SECTIONS), "dominant") == []
        assert flags_of(report(self.tracks(-23.9), self.SECTIONS), "dominant")

    def test_the_reference_is_the_mean_rms_of_the_others(self):
        # others at -26 and -40: linear mean (0.0501 + 0.01) / 2 = -30.4 dB
        tracks = [stem("Kick", -20, 8, index=0), stem("A", -26, 8, index=1),
                  stem("B", -40, 8, index=2)]
        [flag] = flags_of(report(tracks, self.SECTIONS), "dominant")
        mean = (level(-26) + level(-40)) / 2
        assert flag["worst"]["others_mean_dbfs"] == round(db(mean), 1)
        assert flag["worst"]["above_others_db"] == round(-20 - db(mean), 1)

    def test_two_other_tracks_are_needed(self):
        tracks = [stem("Kick", -10, 8, index=0), stem("Pad", -40, 8, index=1)]
        assert flags_of(report(tracks, self.SECTIONS), "dominant") == []

    def test_a_track_that_barely_plays_in_a_section_is_not_judged(self):
        # Kick plays 3 of 8 bars: a fill, not the groove.
        tracks = [stem("Kick", -10, 8, index=0, on={0, 1, 2}),
                  stem("Pad", -40, 8, index=1), stem("Bass", -40, 8, index=2)]
        assert flags_of(report(tracks, self.SECTIONS), "dominant") == []
        tracks[0] = stem("Kick", -10, 8, index=0, on={0, 1, 2, 3})
        assert flags_of(report(tracks, self.SECTIONS), "dominant")

    def test_tracks_that_do_not_play_are_not_part_of_the_reference(self):
        # Two quiet tracks that are silent in the section must not lower
        # the reference there.
        tracks = [stem("Kick", -20, 8, index=0), stem("Pad", -22, 8, index=1),
                  stem("Bass", -22, 8, index=2),
                  stem("Mute1", -50, 8, index=3, on=set()),
                  stem("Mute2", -50, 8, index=4, on=set())]
        assert flags_of(report(tracks, self.SECTIONS), "dominant") == []

    def test_the_worst_section_sizes_the_suggestion(self):
        # dominant by 8 dB in a and by 12 dB in b, fine in c
        kick = TrackStem("Kick", (steady(-22, 4) + steady(-18, 4)
                                  + steady(-30, 4)), 0, "x", 100.0)
        others = [stem("Pad", -30, 12, index=1), stem("Bass", -30, 12, index=2)]
        result = report([kick] + others, [
            Section("a", 0, 4), Section("b", 4, 8), Section("c", 8, 12)])
        [flag] = flags_of(result, "dominant")
        assert flag["sections"] == ["a", "b"]
        assert flag["worst"]["section"] == "b"
        assert flag["worst"]["above_others_db"] == 12.0
        assert flag["gain_db"] == -9.0

    def test_an_automated_volume_is_said_in_the_suggestion(self):
        tracks = self.tracks(-20)
        tracks[0].volume_automated = True
        [flag] = flags_of(report(tracks, self.SECTIONS), "dominant")
        assert flag["suggestion"] == (
            "set_track_volume(0, 45): -7.0 dB (from 100). The volume is "
            "automated: this changes only its static value; edit the "
            "automation instead.")


class TestBuried:
    SECTIONS = [Section("all", 0, 8)]

    def tracks(self, lead_db, volume=100.0, mix_db=-20, index=1):
        """A lead plus a loud track, which together make the mix about
        mix_db RMS (the loud one at mix_db - 0.2 dB)."""
        return [stem("Drums", mix_db - 0.2, 8, index=0),
                stem("Lead", lead_db, 8, index=index, volume=volume)]

    def below(self, tracks):
        result = report(tracks, self.SECTIONS)
        [flag] = flags_of(result, "buried")
        return flag

    def test_a_quiet_track_is_buried_and_the_suggestion_raises_it(self):
        # lead 25 dB below the mix: needs +13 dB to be 12 below, but a
        # volume of 200 only gives +6 dB
        flag = self.below(self.tracks(-45))
        mix = flag["worst"]["mix_rms_dbfs"]
        assert flag["worst"]["rms_dbfs"] == -45.0
        assert flag["worst"]["below_mix_db"] == pytest.approx(
            mix + 45.0, abs=0.11)
        assert (flag["track"], flag["name"], flag["sections"]) == (
            1, "Lead", ["all"])

    def test_the_gain_brings_the_track_to_12_db_below_the_mix(self):
        flag = self.below(self.tracks(-37))
        gap = flag["worst"]["below_mix_db"]
        assert flag["gain_db"] == pytest.approx(gap - 12.0, abs=0.11)
        # 100 * 10^(gain/20), rounded, and below the 200 limit here
        want = round(100 * 10 ** (flag["gain_db"] / 20))
        assert 100 < want < 200
        assert flag["suggestion"].startswith(f"set_track_volume(1, {want}):")

    def test_exact_numbers(self):
        # one track at -20 and a lead at -38 over the same bars: the mix is
        # 10*log10(1e-2 + 10^-3.8) = -19.93 dB, the lead 18.1 below it.
        flag = self.below(self.tracks(-38, mix_db=-20 + 0.2))
        assert flag["worst"] == {
            "section": "all", "rms_dbfs": -38.0, "mix_rms_dbfs": -19.9,
            "below_mix_db": 18.1}
        assert flag["gain_db"] == 6.1
        # wanted volume 100 * 10^(6.1/20) = 202 > 200: clamped
        assert flag["suggestion"].startswith("set_track_volume(1, 200): +6.0 dB")

    def test_when_200_is_not_enough_it_says_so_and_what_else_to_do(self):
        flag = self.below(self.tracks(-48))
        text = flag["suggestion"]
        assert text.startswith("set_track_volume(1, 200): +6.0 dB, the most")
        assert "would be volume" in text
        assert "lower the louder tracks" in text and "set_track_volume" in text

    def test_a_track_below_200_gets_the_exact_volume(self):
        flag = self.below(self.tracks(-37, volume=60.0))
        want = round(60 * 10 ** (flag["gain_db"] / 20))
        assert f"set_track_volume(1, {want}): " in flag["suggestion"]
        assert "(from 60)" in flag["suggestion"]
        assert "the most" not in flag["suggestion"]

    def test_a_track_already_at_200_cannot_get_more(self):
        flag = self.below(self.tracks(-45, volume=200.0))
        assert flag["suggestion"].startswith(
            "set_track_volume(1, 200): +0.0 dB, the most")

    def test_15_db_below_the_mix_is_the_threshold(self):
        assert ma.BURIED_DB == 15.0
        drums = stem("Drums", -20.0, 8, index=0)

        def buried(lead_db):
            tracks = [drums, stem("Lead", lead_db, 8, index=1)]
            return flags_of(report(tracks, self.SECTIONS), "buried")

        # the mix is the drums plus the lead: -19.86 dBFS with the lead at -35
        assert buried(-34.5) == []     # 14.6 dB below the mix
        [flag] = buried(-35.5)         # 15.6 dB below
        assert flag["worst"]["below_mix_db"] == 15.6

    def test_the_level_is_taken_over_the_bars_where_the_track_plays(self):
        # A lead that plays 4 of 8 bars at -30 is compared with the mix over
        # those 4 bars, not with its own average over the silence.
        drums = stem("Drums", -20, 8, index=0)
        lead = stem("Lead", -38, 8, index=1, on={0, 1, 2, 3})
        flag = self.below([drums, lead])
        assert flag["worst"]["rms_dbfs"] == -38.0
        assert flag["worst"]["mix_rms_dbfs"] == -19.9

    def test_a_tail_in_a_section_is_not_judged(self):
        drums = stem("Drums", -20, 8, index=0)
        # one bar of release tail of a track that ended: 1 of 8 bars
        pad = stem("Pad", -50, 8, index=1, on={0})
        assert flags_of(report([drums, pad], self.SECTIONS), "buried") == []

    def test_a_pattern_instrument_gets_a_hint_instead_of_a_call(self):
        drums = stem("Drums", -20, 8, index=0)
        hat = TrackStem("Hat", steady(-45, 8), None, "pattern instrument")
        [flag] = flags_of(report([drums, hat], self.SECTIONS), "buried")
        assert flag["track"] is None
        assert "pattern (BB) track" in flag["suggestion"]
        assert "set_track_volume(" not in flag["suggestion"].split(":")[0]

    def test_a_gain_below_one_volume_step_is_left(self):
        track = stem("Lead", -20, 8, index=1, volume=2.0)
        text = ma.volume_suggestion(track, 0.1, "")
        assert text == "+0.1 dB is below one volume step: leave it."

    def test_flags_are_ordered_by_kind_and_seriousness(self):
        tracks = [stem("Kick", -15, 8, index=0), stem("Pad", -40, 8, index=1),
                  stem("Bass", -33, 8, index=2), stem("Hat", -45, 8, index=3)]
        result = report(tracks, self.SECTIONS)
        assert [f["name"] for f in flags_of(result, "buried")] == [
            "Hat", "Pad", "Bass"]
        # a dominant track sets the mix level the others are measured
        # against: it comes first
        assert [f["type"] for f in result["flags"]][0] == "dominant"


class TestPeakContributors:
    def build(self, mix_peak=1.2, peaks=(0.5, 0.4, 0.05), master=1.0,
              volumes=(100.0, 100.0, 100.0)):
        names = ("Kick", "Snare", "Hat")
        tracks = []
        for i, (name, peak) in enumerate(zip(names, peaks)):
            t = TrackStem(name, steady(-25, 4), i, "x", volumes[i],
                          probe_peaks=[peak])
            tracks.append(t)
        mix = [BarStats(mix_peak if i == 1 else 0.3, 0.01 * COUNT, COUNT)
               for i in range(4)]
        probe = PeakProbe(0, 4000, 0.5, mix_peak)
        return ma.analyze_balance(
            mix, tracks, [Section("hook", 0, 4)], [probe], master)

    def test_the_tracks_that_peak_with_the_mix_are_listed_with_a_fix(self):
        result = self.build()
        [flag] = flags_of(result, "peak_contributors")
        assert flag["section"] == "hook" and flag["bars"] == [0, 4]
        assert flag["mix_peak_dbfs"] == round(db(1.2), 1)
        assert flag["at_s"] == 0.5
        # Hat (0.05) is more than 12 dB below the peak and left out
        assert [c["name"] for c in flag["contributors"]] == ["Kick", "Snare"]
        kick = flag["contributors"][0]
        target = 10 ** (ma.PEAK_TARGET_DBFS / 20)
        factor = 1 - (1.2 - target) / 0.5
        assert factor > ma.MIN_FACTOR
        assert kick["peak_dbfs"] == round(db(0.5), 1)
        assert kick["gain_db"] == round(db(factor), 1)
        assert kick["suggestion"] == (
            f"set_track_volume(0, {round(100 * factor)}): "
            f"{db(factor):+.1f} dB (from 100)")

    def test_a_track_that_cannot_fix_the_peak_alone_says_so(self):
        result = self.build()
        snare = flags_of(result, "peak_contributors")[0]["contributors"][1]
        # removing all of 0.4 still leaves 0.8 > the target
        assert snare["gain_db"] is None
        assert "cannot bring the peak to -2 dBFS" in snare["suggestion"]
        assert "master_suggestion" in snare["suggestion"]

    def test_the_master_volume_alternative(self):
        result = self.build(master=0.9)
        [flag] = flags_of(result, "peak_contributors")
        gain = ma.PEAK_TARGET_DBFS - db(1.2)
        volume = round(0.9 * 10 ** (gain / 20), 4)
        assert flag["master_suggestion"] == (
            f"to bring the peak to -2 dBFS: "
            f"set_mixer_channel_volume(0, {volume:g}): {gain:+.1f} dB")

    def test_the_volume_scales_with_the_current_track_volume(self):
        result = self.build(volumes=(50.0, 100.0, 100.0))
        kick = flags_of(result, "peak_contributors")[0]["contributors"][0]
        factor = 1 - (1.2 - 10 ** (ma.PEAK_TARGET_DBFS / 20)) / 0.5
        assert kick["suggestion"].startswith(
            f"set_track_volume(0, {round(50 * factor)}): ")
        assert "(from 50)" in kick["suggestion"]

    def test_at_most_four_contributors(self):
        names = [f"T{i}" for i in range(6)]
        tracks = [TrackStem(n, steady(-25, 4), i, "x", 100.0,
                            probe_peaks=[0.5 - i * 0.01])
                  for i, n in enumerate(names)]
        mix = [BarStats(1.1, 0.01 * COUNT, COUNT) for _ in range(4)]
        result = ma.analyze_balance(
            mix, tracks, [Section("hook", 0, 4)],
            [PeakProbe(0, 10, 0.1, 1.1)])
        [flag] = flags_of(result, "peak_contributors")
        assert [c["name"] for c in flag["contributors"]] == names[:4]

    def test_find_probes_only_for_sections_near_full_scale(self):
        near = 10 ** (ma.PEAK_NEAR_DBFS / 20)
        audio = aa.Audio(1000, 1, __import__("array").array(
            "f", [0.1] * 2000 + [0.0] * 1000))
        audio.samples[1500] = near * 1.001  # in the second section
        audio.samples[100] = near * 0.99    # below the threshold
        mix_bars = aa.bar_stats(audio, 1.0)
        assert len(mix_bars) == 3
        sections = [Section("a", 0, 1), Section("b", 1, 2), Section("c", 2, 3)]
        probes = ma.find_probes(audio, 1.0, sections, mix_bars)
        assert [(p.section, p.frame) for p in probes] == [(1, 1500)]
        assert probes[0].seconds == 1.5
        assert probes[0].peak == pytest.approx(near, rel=0.01)

    def test_find_probes_ignores_a_mix_fixed_to_minus_one(self):
        # render_project's auto_fix_clipping leaves a peak of -1 dBFS (+-0.1)
        for peak_db, probes in ((-1.1, 0), (-1.0, 0), (-0.9, 0), (-0.45, 1),
                                (0.5, 1)):
            audio = aa.Audio(1000, 1, __import__("array").array(
                "f", [0.1] * 1000))
            audio.samples[500] = 10 ** (peak_db / 20)
            mix_bars = aa.bar_stats(audio, 1.0)
            found = ma.find_probes(audio, 1.0, [Section("a", 0, 1)], mix_bars)
            assert len(found) == probes, peak_db

    def test_no_probe_no_flag(self):
        tracks = [stem("Kick", -20, 4, index=0)]
        result = ma.analyze_balance(
            mix_of(tracks), tracks, [Section("a", 0, 4)], [])
        assert flags_of(result, "peak_contributors") == []


class TestFlagLimits:
    def test_flags_per_kind_are_capped_and_a_note_says_how_many(self):
        count = ma.MAX_FLAGS_PER_KIND + 3
        tracks = [stem("Loud", -10, 8, index=0)]
        tracks += [stem(f"Quiet{i}", -35 - i / 2, 8, index=i + 1)
                   for i in range(count)]
        result = report(tracks, [Section("all", 0, 8)])
        buried = flags_of(result, "buried")
        assert len(buried) == ma.MAX_FLAGS_PER_KIND
        # the most serious (quietest) first
        assert buried[0]["name"] == f"Quiet{count - 1}"
        assert any("3 more \"buried\" flags" in n for n in result["notes"])

    def test_every_flag_has_a_summary_and_a_suggestion_in_server_units(self):
        tracks = [stem("Kick", -15, 8, index=0), stem("Pad", -40, 8, index=1),
                  stem("Bass", -40, 8, index=2)]
        result = report(tracks, [Section("all", 0, 8)])
        assert result["flags"]
        for flag in result["flags"]:
            assert flag["summary"] and flag["suggestion"]
            assert flag["suggestion"].startswith("set_track_volume(")
            assert not any(key.startswith("_") for key in flag)
