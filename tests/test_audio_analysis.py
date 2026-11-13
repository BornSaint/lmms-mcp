"""Tests for lmms_mcp.audio_analysis: reading WAV, measuring levels, writing
PCM. Signals are synthetic with known peak/RMS; the references are computed
here with plain Python (and numpy where it is installed)."""

import array
import math
import struct
import wave

import pytest

from audio_helpers import (
    SAMPLERATE, float_wav_bytes, pcm_wav_bytes, read_pcm, sine)
from lmms_mcp import audio_analysis as aa


def db(x):
    return 20 * math.log10(x)


def audio_of(samples, samplerate=SAMPLERATE, channels=2):
    return aa.parse_wav(float_wav_bytes(samples, samplerate, channels))


class TestBarSeconds:
    @pytest.mark.parametrize("bpm, num, den, expected", [
        (120, 4, 4, 2.0),
        (60, 4, 4, 4.0),
        (120, 3, 4, 1.5),
        (120, 6, 8, 1.5),
        (120, 7, 8, 1.75),
        (140, 4, 4, 60 / 140 * 4),
        (90, 2, 2, 60 / 90 * 4),
        (100, 5, 4, 3.0),
    ])
    def test_bar_length(self, bpm, num, den, expected):
        assert aa.bar_seconds(bpm, num, den) == pytest.approx(expected)

    @pytest.mark.parametrize("args", [(0, 4, 4), (120, 0, 4), (120, 4, 0)])
    def test_invalid_values_raise(self, args):
        with pytest.raises(ValueError):
            aa.bar_seconds(*args)


class TestReadWav:
    def test_float32_stereo(self):
        samples = [0.0, 0.5, -0.25, 1.5, -2.0, 0.125]
        audio = aa.parse_wav(float_wav_bytes(samples, 44100, 2))
        assert (audio.samplerate, audio.channels, audio.frames) == (44100, 2, 3)
        assert list(audio.samples) == samples
        assert audio.duration == pytest.approx(3 / 44100)

    def test_float32_mono(self):
        audio = aa.parse_wav(float_wav_bytes([0.5, -0.5], 8000, 1))
        assert (audio.channels, audio.frames) == (1, 2)

    @pytest.mark.parametrize("bits, full", [
        (16, 32768), (24, 8388608), (32, 2147483648)])
    def test_pcm_depths(self, bits, full):
        ints = [0, full // 2, -full // 2, full - 1, -full, 1]
        audio = aa.parse_wav(pcm_wav_bytes(ints, bits, 22050, 2))
        assert audio.samplerate == 22050
        assert list(audio.samples) == pytest.approx(
            [v / full for v in ints], abs=1e-7)

    def test_chunks_around_data_are_skipped(self):
        # LMMS writes a LIST chunk (odd size, padded) before the data.
        wav = float_wav_bytes([0.25, 0.5], 8000, 2)
        riff, fmt_and_data = wav[:12], wav[12:]
        fmt = fmt_and_data[:24]
        data = fmt_and_data[24:]
        extra = b"LIST" + struct.pack("<I", 3) + b"abc" + b"\0"
        audio = aa.parse_wav(riff + fmt + extra + data + b"fact\x04\0\0\0abcd")
        assert list(audio.samples) == [0.25, 0.5]

    def test_extensible_format(self):
        samples = [0.5, -0.5]
        data = array.array("f", samples).tobytes()
        fmt = struct.pack("<HHIIHH", 0xFFFE, 2, 8000, 8000 * 8, 8, 32)
        fmt += struct.pack("<HHI", 22, 32, 3)
        fmt += struct.pack("<H", 3) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        wav = (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data))
               + b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
               + b"data" + struct.pack("<I", len(data)) + data)
        audio = aa.parse_wav(wav)
        assert list(audio.samples) == samples

    def test_data_size_larger_than_the_file_reads_to_the_end(self):
        wav = bytearray(float_wav_bytes([0.5, 0.25, 0.125, 0.0625], 8000, 2))
        wav[40:44] = struct.pack("<I", 0xFFFFFFFF)
        audio = aa.parse_wav(bytes(wav))
        assert audio.frames == 2

    def test_partial_last_frame_is_dropped(self):
        wav = float_wav_bytes([0.5, 0.25, 0.125], 8000, 2)
        assert aa.parse_wav(wav[:-4] + b"\0\0").frames == 1

    @pytest.mark.parametrize("data, message", [
        (b"", "not a RIFF/WAVE"),
        (b"RIFF\0\0\0\0AVI LIST", "not a RIFF/WAVE"),
        (b"RIFF\0\0\0\0WAVE", "no fmt chunk"),
        (b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0" + b"\x01\0\x02\0" + b"\0" * 12,
         "no data chunk"),
    ])
    def test_not_a_wav(self, data, message):
        with pytest.raises(aa.AudioError, match=message):
            aa.parse_wav(data)

    def test_unsupported_formats(self):
        wav = bytearray(float_wav_bytes([0.5, 0.5], 8000, 2))
        wav[20:22] = struct.pack("<H", 85)  # MPEG layer 3
        with pytest.raises(aa.AudioError, match="format tag 85"):
            aa.parse_wav(bytes(wav))
        eight = pcm_wav_bytes([0, 0], 16, 8000, 2)
        eight = eight[:34] + struct.pack("<H", 8) + eight[36:]
        with pytest.raises(aa.AudioError, match="8 bits"):
            aa.parse_wav(eight)

    def test_read_wav_reads_a_file(self, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(float_wav_bytes([0.5, -0.5], 8000, 1))
        assert list(aa.read_wav(path).samples) == [0.5, -0.5]
        with pytest.raises(OSError):
            aa.read_wav(tmp_path / "missing.wav")


class TestAnalyze:
    def test_sine_peak_and_rms(self):
        # 250 Hz at 8 kHz: whole periods, so the RMS is amplitude / sqrt(2).
        levels = aa.analyze(audio_of(sine(250, 0.5, 8)), aa.bar_seconds(120))
        assert db(levels.peak) == pytest.approx(-6.02, abs=0.01)
        assert db(levels.rms) == pytest.approx(-9.03, abs=0.01)
        assert levels.clipped == 0
        assert levels.duration == pytest.approx(8.0)

    def test_mono_and_stereo_agree(self):
        stereo = aa.analyze(audio_of(sine(250, 0.5, 8, channels=2)), 2.0)
        mono = aa.analyze(
            audio_of(sine(250, 0.5, 8, channels=1), channels=1), 2.0)
        assert mono.peak == pytest.approx(stereo.peak)
        assert mono.rms == pytest.approx(stereo.rms)

    def test_rms_covers_every_channel(self):
        # Left at 0.5, right silent: the RMS is that of the two together.
        samples = array.array("f")
        for _ in range(8000 * 4):
            samples.extend([0.5, 0.0])
        levels = aa.analyze(audio_of(samples), 2.0)
        assert levels.rms == pytest.approx(0.5 / math.sqrt(2))
        assert levels.peak == pytest.approx(0.5)

    def test_negative_peak_counts(self):
        samples = [0.1, -0.9, 0.2, 0.3]
        levels = aa.analyze(audio_of(samples, channels=2), 2.0)
        assert levels.peak == pytest.approx(0.9)

    def test_clipped_samples_are_counted_above_and_below(self):
        signal = sine(250, 2.0, 8)
        expected = sum(1 for v in signal if abs(v) > 1.0)
        assert expected > 1000
        levels = aa.analyze(audio_of(signal), aa.bar_seconds(120))
        assert levels.clipped == expected
        assert db(levels.peak) == pytest.approx(6.02, abs=0.01)

    def test_samples_at_exactly_full_scale_are_not_clipped(self):
        levels = aa.analyze(audio_of([1.0, -1.0, 0.5, 1.0]), 2.0)
        assert levels.clipped == 0
        levels = aa.analyze(audio_of([1.0, -1.0000001, 0.5, 1.5]), 2.0)
        assert levels.clipped == 2

    def test_byte_level_count_matches_comparing_every_sample(self):
        values = [0.0, -0.0, 1.0, -1.0, 0.99999994, -0.99999994, 1.0000001,
                  -1.0000001, 1.5, -1.5, 1.9999999, 2.0, -2.0, 3.0, 123.0,
                  -1e30, 1e-45, 0.5, 1e38, 1.0, -1.0, 1.0, 1.0000002]
        random_values = array.array("f")
        import random
        rng = random.Random(5)
        for _ in range(3000):
            random_values.append(rng.uniform(-3, 3))
        for samples in (array.array("f", values), random_values,
                        array.array("f", values * 7 + list(random_values))):
            expected = sum(1 for v in samples if abs(v) > 1.0)
            assert aa.count_float32_over_one(samples) == expected
            assert aa.count_over(samples) == expected
            # the generic path (a list, or another limit) agrees
            assert aa.count_over(list(samples)) == expected
        assert aa.count_float32_over_one(array.array("f")) == 0

    def test_count_over_with_another_limit(self):
        samples = array.array("f", [0.2, -0.6, 0.5, 0.75, -0.25])
        assert aa.count_over(samples, 0.5) == 2
        assert aa.count_over(samples, 0.1) == 5
        assert aa.count_over(samples, 1.0) == 0

    def test_clipped_counted_in_blocks_that_clip(self):
        signal = sine(250, 0.5, 8) + sine(250, 3.0, 8)
        expected = sum(1 for v in signal if abs(v) > 1.0)
        levels = aa.analyze(audio_of(signal), 2.0, bars_per_block=4)
        assert levels.clipped == expected

    def test_digital_silence(self):
        levels = aa.analyze(audio_of([0.0] * 2000), 2.0)
        assert levels.peak == 0.0 and levels.rms == 0.0
        assert aa.to_db(levels.peak) is None
        assert levels.blocks[0].silent

    def test_blocks_below_minus_60_dbfs_rms_are_silent(self):
        for amplitude, silent in ((0.0, True), (0.0005, True),
                                  (0.01, False)):
            levels = aa.analyze(audio_of(sine(250, amplitude, 8)), 2.0)
            assert levels.blocks[0].silent is silent

    def test_empty_audio(self):
        levels = aa.analyze(audio_of([]), 2.0)
        assert levels.peak == 0.0 and levels.clipped == 0
        assert len(levels.blocks) == 1

    def test_blocks_are_four_bars_with_their_own_levels(self):
        # 120 bpm 4/4: a bar is 2 s. 12 bars = 3 blocks of 4 bars.
        signal = (sine(250, 0.5, 8) + sine(250, 0.1, 8) + sine(250, 0.25, 8))
        levels = aa.analyze(audio_of(signal), aa.bar_seconds(120))
        assert [(b.first_bar, b.last_bar) for b in levels.blocks] == [
            (1, 4), (5, 8), (9, 12)]
        peaks = [db(b.peak) for b in levels.blocks]
        assert peaks == pytest.approx([-6.02, -20.0, -12.04], abs=0.01)
        rms = [db(b.rms) for b in levels.blocks]
        assert rms == pytest.approx([-9.03, -23.01, -15.05], abs=0.01)

    @pytest.mark.parametrize("bpm, num, den, bars", [
        (120, 4, 4, 8), (120, 3, 4, 8), (90, 6, 8, 5), (140, 4, 4, 9),
        (75, 7, 8, 3),
    ])
    def test_blocks_follow_tempo_and_time_signature(
            self, bpm, num, den, bars):
        length = aa.bar_seconds(bpm, num, den)
        frames = round(bars * length * SAMPLERATE)
        audio = audio_of([0.1, 0.1] * frames)
        levels = aa.analyze(audio, length)
        assert levels.blocks[-1].last_bar == bars
        assert len(levels.blocks) == math.ceil(bars / 4)

    def test_block_boundary_follows_the_bar_length(self):
        # 3/4 at 120 bpm: bars of 1.5 s; a loud second bar is in block 1,
        # the loud 5th bar (6.0 s-7.5 s) starts block 2.
        quiet, loud = 0.1, 0.8
        parts = []
        for bar in range(8):
            amp = loud if bar in (1, 4) else quiet
            parts.append(sine(250, amp, 1.5))
        signal = array.array("f")
        for part in parts:
            signal.extend(part)
        levels = aa.analyze(audio_of(signal), aa.bar_seconds(120, 3, 4))
        assert [(b.first_bar, b.last_bar) for b in levels.blocks] == [
            (1, 4), (5, 8)]
        assert levels.blocks[0].peak == pytest.approx(loud, rel=1e-3)
        assert levels.blocks[1].peak == pytest.approx(loud, rel=1e-3)
        # Move the loud bar across the boundary: bar 5 -> bar 4's end.
        signal = array.array("f")
        for bar in range(8):
            signal.extend(sine(250, loud if bar == 3 else quiet, 1.5))
        levels = aa.analyze(audio_of(signal), aa.bar_seconds(120, 3, 4))
        assert levels.blocks[0].peak == pytest.approx(loud, rel=1e-3)
        assert levels.blocks[1].peak == pytest.approx(quiet, rel=1e-3)

    def test_more_than_32_blocks_use_longer_blocks(self):
        # 1-channel, 100 Hz "audio": a bar of 2 s is 200 frames.
        def levels_for(bars):
            audio = aa.Audio(100, 1, array.array("f", [0.1] * (bars * 200)))
            return aa.analyze(audio, 2.0)

        assert len(levels_for(128).blocks) == 32          # 4 bars each
        assert levels_for(128).blocks[1].first_bar == 5
        wide = levels_for(132)                            # 33 blocks -> 8 bars
        assert len(wide.blocks) == 17
        assert (wide.blocks[0].first_bar, wide.blocks[0].last_bar) == (1, 8)
        assert wide.blocks[-1].last_bar == 132
        wider = levels_for(520)                           # 130 blocks -> 32 bars
        assert (wider.blocks[0].first_bar, wider.blocks[0].last_bar) == (1, 32)
        assert len(wider.blocks) == 17

    def test_bars_per_block_can_be_forced(self):
        audio = audio_of(sine(250, 0.5, 16))
        levels = aa.analyze(audio, 2.0, bars_per_block=1)
        assert len(levels.blocks) == 8

    def test_short_tail_joins_the_last_bar_and_a_long_one_is_a_bar(self):
        def bars_of(seconds):
            audio = audio_of(sine(250, 0.5, seconds), channels=2)
            return [(b.first_bar, b.last_bar)
                    for b in aa.analyze(audio, 2.0).blocks]

        assert bars_of(16.6) == [(1, 4), (5, 8)]        # 8.3 bars
        assert bars_of(17.2) == [(1, 4), (5, 8), (9, 9)]  # 8.6 bars
        assert bars_of(0.5) == [(1, 1)]
        # The tail's samples are part of the last block.
        samples = array.array("f", sine(250, 0.1, 16.0))
        samples.extend(sine(250, 0.9, 0.6))
        last = aa.analyze(audio_of(samples), 2.0).blocks[-1]
        assert last.peak == pytest.approx(0.9, rel=1e-3)

    def test_numpy_agrees(self):
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(7)
        data = (rng.standard_normal(SAMPLERATE * 20 * 2) * 0.7).astype("<f4")
        audio = audio_of(array.array("f", data.tolist()))
        levels = aa.analyze(audio, 2.0)
        assert levels.peak == pytest.approx(float(np.abs(data).max()))
        assert levels.clipped == int((np.abs(data) > 1.0).sum())
        assert levels.rms == pytest.approx(
            float(np.sqrt(np.mean(data.astype("f8") ** 2))), rel=1e-6)
        assert len(levels.blocks) == 3
        first = data[:SAMPLERATE * 8 * 2]
        assert levels.blocks[0].rms == pytest.approx(
            float(np.sqrt(np.mean(first.astype("f8") ** 2))), rel=1e-6)
        assert levels.blocks[0].peak == pytest.approx(
            float(np.abs(first).max()))


class TestDb:
    def test_round_trip(self):
        assert aa.to_db(1.0) == 0.0
        assert aa.to_db(0.5) == pytest.approx(-6.0206, abs=1e-4)
        assert aa.to_db(0.0) is None
        assert aa.from_db(-6.0206) == pytest.approx(0.5, abs=1e-4)
        assert aa.from_db(aa.to_db(0.3)) == pytest.approx(0.3)


class TestWriteWav:
    def test_16_bit_with_gain(self, tmp_path):
        path = tmp_path / "out.wav"
        samples = [0.0, 0.5, -0.5, 1.0, -1.0, 0.25]
        aa.write_wav(path, audio_of(samples, 8000, 2), bits=16, gain=0.5)
        rate, channels, bits, ints = read_pcm(path)
        assert (rate, channels, bits) == (8000, 2, 16)
        assert ints == [0, 8192, -8192, 16384, -16384, 4096]  # round(x*16383.5)

    def test_toward_zero_is_what_lmms_does(self, tmp_path):
        path = tmp_path / "out.wav"
        samples = [0.4999, 0.5001, -0.4999, -0.5001, 0.99999, -0.99999]
        audio = audio_of(samples, 8000, 2)
        aa.write_wav(path, audio, bits=16, rounding="toward_zero")
        assert read_pcm(path)[3] == [int(v * 32767) for v in audio.samples]
        aa.write_wav(path, audio, bits=16, rounding="nearest")
        assert read_pcm(path)[3] == [round(v * 32767) for v in audio.samples]
        assert read_pcm(path)[3] != [int(v * 32767) for v in audio.samples]

    def test_unknown_rounding(self, tmp_path):
        with pytest.raises(aa.AudioError, match="rounding"):
            aa.write_wav(tmp_path / "x.wav", audio_of([0.0, 0.0]),
                         rounding="up")

    def test_24_bit(self, tmp_path):
        path = tmp_path / "out.wav"
        aa.write_wav(path, audio_of([0.5, -0.5, 0.25, -1.0], 8000, 2),
                     bits=24, gain=1.0)
        rate, channels, bits, ints = read_pcm(path)
        assert bits == 24 and channels == 2
        assert ints == [round(v * 8388607) for v in (0.5, -0.5, 0.25, -1.0)]

    def test_rounding_is_to_nearest(self, tmp_path):
        path = tmp_path / "out.wav"
        step = 1 / 32767
        samples = [0.4 * step, 0.6 * step, -0.4 * step, -0.6 * step]
        aa.write_wav(path, audio_of(samples, 8000, 2), bits=16)
        assert read_pcm(path)[3] == [0, 1, 0, -1]

    def test_gain_that_overshoots_clips_at_full_scale(self, tmp_path):
        path = tmp_path / "out.wav"
        aa.write_wav(path, audio_of([0.4, -0.4, 0.9, -0.9], 8000, 2),
                     bits=16, gain=3.0)
        assert read_pcm(path)[3] == [32767, -32767, 32767, -32767]

    def test_louder_than_full_scale_is_limited_like_lmms(self, tmp_path):
        path = tmp_path / "out.wav"
        aa.write_wav(path, audio_of([15.0, -15.0, 0.5, 1.2], 8000, 2))
        assert read_pcm(path)[3] == [32767, -32767, 16384, 32767]

    def test_round_trip_through_the_reader(self, tmp_path):
        path = tmp_path / "out.wav"
        signal = sine(250, 0.7, 1)
        aa.write_wav(path, audio_of(signal), bits=16)
        back = aa.read_wav(path)
        assert back.frames == len(signal) // 2
        assert max(abs(a - b) for a, b in zip(signal, back.samples)) < 1e-4
        aa.write_wav(path, audio_of(signal), bits=24)
        back = aa.read_wav(path)
        assert max(abs(a - b) for a, b in zip(signal, back.samples)) < 1e-6

    def test_standard_library_can_read_the_header(self, tmp_path):
        path = tmp_path / "out.wav"
        aa.write_wav(path, audio_of(sine(250, 0.5, 1), 22050, 2), bits=16)
        with wave.open(str(path)) as wav:
            assert wav.getnchannels() == 2
            assert wav.getframerate() == 22050
            assert wav.getsampwidth() == 2
            assert wav.getnframes() == 8000

    def test_odd_sized_data_is_padded(self, tmp_path):
        path = tmp_path / "odd.wav"
        aa.write_wav(path, audio_of([0.5], 8000, 1), bits=24)
        assert path.stat().st_size % 2 == 0
        assert read_pcm(path)[3] == [round(0.5 * 8388607)]

    def test_several_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aa, "_CHUNK", 1000)
        path = tmp_path / "out.wav"
        signal = sine(250, 0.5, 2.0) + sine(250, 2.0, 0.5)
        aa.write_wav(path, audio_of(signal), bits=16, gain=0.9)
        ints = read_pcm(path)[3]
        assert len(ints) == len(signal)
        expected = [max(-32767, min(32767, round(v * 0.9 * 32767)))
                    for v in signal]
        assert ints == expected

    def test_invalid_depth_and_unwritable_path(self, tmp_path):
        with pytest.raises(aa.AudioError):
            aa.write_wav(tmp_path / "x.wav", audio_of([0.0, 0.0]), bits=8)
        with pytest.raises(OSError):
            aa.write_wav(tmp_path / "missing" / "x.wav", audio_of([0.0, 0.0]))

    def test_numpy_agrees(self, tmp_path):
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(3)
        data = (rng.standard_normal(4000) * 0.3).astype("<f4")
        audio = audio_of(array.array("f", data.tolist()))
        aa.write_wav(tmp_path / "a.wav", audio, bits=16, gain=1.7)
        expected = np.clip(
            np.rint(data.astype("f8") * 1.7 * 32767), -32767, 32767)
        assert read_pcm(tmp_path / "a.wav")[3] == expected.astype(int).tolist()


class TestSpeed:
    def test_a_long_stereo_file_is_analyzed_in_seconds(self):
        # 1 minute of stereo at 44.1 kHz (a third of a 3-minute song): the
        # measured time is ~0.6 s on a slow VM; the limit only catches a
        # per-sample Python loop slipping in.
        import time
        frames = 44100 * 60
        samples = array.array("f", [0.5, -0.25] * frames)
        audio = aa.Audio(44100, 2, samples)
        start = time.perf_counter()
        levels = aa.analyze(audio, 2.0)
        assert time.perf_counter() - start < 15
        assert levels.peak == 0.5
