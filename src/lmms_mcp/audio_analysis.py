"""Measure and write WAV audio in pure Python (no numpy).

render_project renders the project as a 32-bit float WAV first, so levels
above 0 dBFS are still there to measure, and uses this module to read that
file, report its peak, clipped samples and per-bar-range loudness, and to
write the final PCM file with the gain that keeps it from clipping.

The work is done with C-level array operations (max, min, math.sumprod, map
over operator functions) and not with a Python loop per sample, so a
3-minute stereo file at 44.1 kHz is analyzed in about a second.
"""

from __future__ import annotations

import array
import math
import operator
import struct
import sys
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path

# A render whose peak is below this is silent (-80 dBFS).
SILENCE_PEAK = 1e-4

# A block of bars whose RMS is below this (-60 dBFS) counts as silence when
# looking for the quietest part of a song: the reverb or release tail after
# the last note is not a quiet section.
SILENT_BLOCK_RMS = 1e-3

# Samples converted or written per step (bounds the memory of the temporary
# lists the C-level maps build).
_CHUNK = 1 << 20

_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_FLOAT = 3
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_BIG_ENDIAN = sys.byteorder == "big"


class AudioError(ValueError):
    """The file is not a WAV this module can read, or cannot be written."""


@dataclass
class Audio:
    """Interleaved samples as 32-bit floats, full scale = +-1.0."""

    samplerate: int
    channels: int
    samples: array.array = field(repr=False)

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate


@dataclass
class BlockLevels:
    """Levels of a range of bars (1-based, inclusive). Linear values;
    rms is 0.0 for digital silence."""

    first_bar: int
    last_bar: int
    peak: float
    rms: float

    @property
    def silent(self) -> bool:
        return self.rms < SILENT_BLOCK_RMS


@dataclass
class Levels:
    """What analyze() measured. peak and rms are linear (1.0 = 0 dBFS)."""

    duration: float
    peak: float
    clipped: int
    rms: float
    blocks: list[BlockLevels]


def to_db(value: float) -> float | None:
    """A linear level as dBFS, None for exact silence."""
    if value <= 0.0:
        return None
    return 20.0 * math.log10(value)


def from_db(db: float) -> float:
    """dB as a linear factor."""
    return 10.0 ** (db / 20.0)


def bar_seconds(bpm: float, numerator: int = 4, denominator: int = 4) -> float:
    """Seconds of one bar: 60 / bpm seconds per quarter note times the
    quarter notes per bar (numerator * 4 / denominator)."""
    if bpm <= 0 or numerator <= 0 or denominator <= 0:
        raise ValueError("bpm, numerator and denominator must be positive")
    return 60.0 / bpm * numerator * 4.0 / denominator


# ──────────────────────────────────────────────────────────────────
# READING
# ──────────────────────────────────────────────────────────────────


def _swap_if_big_endian(arr: array.array) -> array.array:
    """WAV data is little-endian; arrays use the machine's order."""
    if _BIG_ENDIAN and arr.itemsize > 1:
        arr.byteswap()
    return arr


def _int_typecode(size: int) -> str:
    """The array typecode of a signed integer `size` bytes wide."""
    for code in "hilq":
        if array.array(code).itemsize == size:
            return code
    raise AudioError(f"no {size * 8}-bit integer type on this machine")


def _scaled(source, factor: float) -> array.array:
    """array('f') of source * factor, converted chunk by chunk."""
    out = array.array("f")
    for start in range(0, len(source), _CHUNK):
        out.extend(map(operator.mul, source[start:start + _CHUNK],
                       repeat(factor)))
    return out


def _decode(raw: bytes, tag: int, bits: int) -> array.array:
    """The samples of raw WAV data as float32 with full scale 1.0."""
    if tag == _WAVE_FORMAT_FLOAT:
        if bits == 32:
            out = array.array("f")
            out.frombytes(raw)
            return _swap_if_big_endian(out)
        if bits == 64:
            wide = array.array("d")
            wide.frombytes(raw)
            return array.array("f", _swap_if_big_endian(wide))
        raise AudioError(f"unsupported float WAV with {bits} bits")
    if tag != _WAVE_FORMAT_PCM:
        raise AudioError(f"unsupported WAV format tag {tag} (need PCM or float)")
    if bits == 16:
        ints = array.array("h")
        ints.frombytes(raw)
        return _scaled(_swap_if_big_endian(ints), 1.0 / 32768.0)
    if bits == 24:
        # Left-align every 3-byte sample in a 32-bit one (low byte zero) by
        # slice assignment, then read them as signed 32-bit integers.
        count = len(raw) // 3
        raw = bytes(raw)
        padded = bytearray(4 * count)
        padded[1::4] = raw[0:3 * count:3]
        padded[2::4] = raw[1:3 * count:3]
        padded[3::4] = raw[2:3 * count:3]
        ints = array.array(_int_typecode(4))
        ints.frombytes(bytes(padded))
        return _scaled(_swap_if_big_endian(ints), 1.0 / 2147483648.0)
    if bits == 32:
        ints = array.array(_int_typecode(4))
        ints.frombytes(raw)
        return _scaled(_swap_if_big_endian(ints), 1.0 / 2147483648.0)
    raise AudioError(f"unsupported PCM WAV with {bits} bits")


def parse_wav(data: bytes) -> Audio:
    """Read a RIFF/WAVE file: PCM 16/24/32-bit or IEEE float 32/64-bit,
    any number of channels (also the WAVE_FORMAT_EXTENSIBLE variants).

    A data chunk that claims more bytes than the file has (a file that is
    still being written, or whose size field was never filled in) is read
    up to the end of the file.
    """
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RF64") \
            or data[8:12] != b"WAVE":
        raise AudioError("not a RIFF/WAVE file")
    view = memoryview(data)  # slices of it are not copies
    fmt = None
    raw = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            fmt = bytes(view[body:body + size])
        elif chunk_id == b"data":
            raw = view[body:body + size] if size != 0xFFFFFFFF \
                else view[body:]
            break
        pos = body + size + (size & 1)
    if fmt is None or len(fmt) < 16:
        raise AudioError("WAV file has no fmt chunk")
    if raw is None:
        raise AudioError("WAV file has no data chunk")
    tag, channels, samplerate, _, align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == _WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 26:
            raise AudioError("WAV file has a truncated extensible fmt chunk")
        tag = struct.unpack("<H", fmt[24:26])[0]
    if channels < 1 or samplerate < 1 or align < 1:
        raise AudioError("WAV file has an invalid fmt chunk")
    raw = raw[:len(raw) - len(raw) % align]
    samples = _decode(raw, tag, bits)
    if len(samples) % channels:
        samples = samples[:len(samples) - len(samples) % channels]
    return Audio(samplerate, channels, samples)


def read_wav(path: str | Path) -> Audio:
    """parse_wav of a file; OSError when it cannot be read."""
    return parse_wav(Path(path).read_bytes())


# ──────────────────────────────────────────────────────────────────
# MEASURING
# ──────────────────────────────────────────────────────────────────


def _sum_squares(samples) -> float:
    sumprod = getattr(math, "sumprod", None)  # Python 3.12+
    if sumprod is not None:
        return sumprod(samples, samples)
    return math.fsum(map(operator.mul, samples, samples))


def extremes(samples) -> tuple[float, float]:
    """(lowest, highest) sample value, (0.0, 0.0) for no samples."""
    if not len(samples):
        return 0.0, 0.0
    return min(samples), max(samples)


def peak_of(samples) -> float:
    """The largest absolute sample value (0.0 for no samples)."""
    lowest, highest = extremes(samples)
    return max(highest, -lowest, 0.0)


def _table(test) -> bytes:
    """A bytes.translate table: 1 for the byte values `test` accepts."""
    return bytes(1 if test(value) else 0 for value in range(256))


# Tables for count_float32_over_one.
_EXPONENT_AT_LEAST_128 = _table(lambda b: b & 0x7F >= 64)   # |x| >= 2.0
_EXPONENT_HIGH_BITS_63 = _table(lambda b: b & 0x7F == 63)   # |x| in [0.5, 2)
_TOP_BIT_SET = _table(lambda b: b >= 128)
_IS_128 = _table(lambda b: b == 128)
_IS_ZERO = _table(lambda b: b == 0)


def count_float32_over_one(samples: array.array) -> int:
    """How many float32 samples have |x| > 1.0, counted on their bytes.

    The sign and exponent of a float sit in its top bytes, so which
    samples are 2.0 or more, 1.0 or more, or exactly 1.0 is decided with
    bytes.translate and bit operations on whole-array integers: about 5
    times faster than comparing every sample in Python. (A NaN or infinity
    counts as over.)
    """
    raw = samples.tobytes()
    if _BIG_ENDIAN:
        b3, b2, b1, b0 = raw[0::4], raw[1::4], raw[2::4], raw[3::4]
    else:
        b3, b2, b1, b0 = raw[3::4], raw[2::4], raw[1::4], raw[0::4]
    count = b3.translate(_EXPONENT_AT_LEAST_128).count(1)
    # Exponent 127 (|x| in [1.0, 2.0)): top 7 exponent bits are 63 and the
    # lowest exponent bit (the top bit of the next byte) is set.
    from_bytes = int.from_bytes
    between = (from_bytes(b3.translate(_EXPONENT_HIGH_BITS_63), "little")
               & from_bytes(b2.translate(_TOP_BIT_SET), "little"))
    # ... except exactly +-1.0: no mantissa bits set.
    exactly_one = (between
                   & from_bytes(b2.translate(_IS_128), "little")
                   & from_bytes(b1.translate(_IS_ZERO), "little")
                   & from_bytes(b0.translate(_IS_ZERO), "little"))
    return count + between.bit_count() - exactly_one.bit_count()


def count_over(
    samples,
    limit: float = 1.0,
    extreme_values: tuple[float, float] | None = None,
) -> int:
    """How many samples are above +limit or below -limit.

    extreme_values is (lowest, highest) of the samples when the caller
    has them (see extremes): it saves the passes over the data that find
    out whether any sample is beyond the limit.
    """
    lowest, highest = extreme_values or extremes(samples)
    if highest <= limit and lowest >= -limit:
        return 0
    if limit == 1.0 and isinstance(samples, array.array) \
            and samples.typecode == "f":
        return count_float32_over_one(samples)
    return (sum(map(operator.gt, samples, repeat(limit)))
            + sum(map(operator.lt, samples, repeat(-limit))))


def analyze(
    audio: Audio,
    bar_length: float,
    bars_per_block: int | None = None,
    max_blocks: int = 32,
) -> Levels:
    """Peak, clipped samples, RMS and the levels of each block of bars.

    bar_length is the length of a bar in seconds (see bar_seconds). Blocks
    are 4 bars long; if that gives more than max_blocks blocks the block
    length doubles (8, 16, ...) until it does not (bars_per_block forces a
    length). The song is cut into whole bars: audio after the last whole
    bar (a reverb tail) counts for the last bar when it is shorter than half
    a bar, else it is a bar of its own. RMS is taken over every sample of
    every channel: a full-scale sine is -3.01 dBFS.
    """
    channels = audio.channels
    samples = audio.samples
    frames = audio.frames
    bar_frames = bar_length * audio.samplerate
    total_bars = max(1, int(frames / bar_frames + 0.5)) if frames else 1
    if bars_per_block is None:
        bars_per_block = 4
        while math.ceil(total_bars / bars_per_block) > max_blocks:
            bars_per_block *= 2

    blocks: list[BlockLevels] = []
    peak = 0.0
    clipped = 0
    total_squares = 0.0
    for first in range(0, total_bars, bars_per_block):
        last = min(first + bars_per_block, total_bars)
        start = round(first * bar_frames) * channels
        stop = (len(samples) if last == total_bars
                else round(last * bar_frames) * channels)
        chunk = samples[start:min(stop, len(samples))]
        lowest, highest = extremes(chunk)
        block_peak = max(highest, -lowest, 0.0)
        squares = _sum_squares(chunk) if len(chunk) else 0.0
        clipped += count_over(chunk, 1.0, (lowest, highest))
        total_squares += squares
        peak = max(peak, block_peak)
        blocks.append(BlockLevels(
            first + 1, last, block_peak,
            math.sqrt(squares / len(chunk)) if len(chunk) else 0.0))
    return Levels(
        duration=audio.duration,
        peak=peak,
        clipped=clipped,
        rms=math.sqrt(total_squares / len(samples)) if len(samples) else 0.0,
        blocks=blocks,
    )


# ──────────────────────────────────────────────────────────────────
# WRITING
# ──────────────────────────────────────────────────────────────────


def _pcm_chunk(
    chunk, scale: float, limit: int, bits: int, clamp: bool, to_int,
) -> bytes:
    """Little-endian PCM bytes of chunk * scale (made integers by `to_int`,
    and limited to +-limit when clamp)."""
    values = map(operator.mul, chunk, repeat(scale))
    if clamp:
        values = map(max, map(min, values, repeat(float(limit))),
                     repeat(float(-limit)))
    if bits == 16:
        ints = array.array("h", map(to_int, values))
        _swap_if_big_endian(ints)
        return ints.tobytes()
    ints = array.array(_int_typecode(4), map(to_int, values))
    _swap_if_big_endian(ints)
    data = ints.tobytes()
    out = bytearray(3 * len(ints))
    out[0::3] = data[0::4]
    out[1::3] = data[1::4]
    out[2::3] = data[2::4]
    return bytes(out)


# float -> int conversions of write_wav's `rounding`. The unbound float
# methods are faster than the round() and int() builtins, which look the
# method up on every call.
_ROUNDING = {
    "nearest": float.__round__,      # half to even, like round()
    "toward_zero": float.__trunc__,  # like int() and a C cast
}


def write_wav(
    path: str | Path,
    audio: Audio,
    bits: int = 16,
    gain: float = 1.0,
    peak: float | None = None,
    rounding: str = "nearest",
) -> None:
    """Write audio * gain as a PCM WAV of 16 or 24 bits.

    Samples are made integers (`rounding`: "nearest", or "toward_zero",
    which is what LMMS does for its 16-bit files) and limited to the
    full-scale range (+-32767 / +-8388607: LMMS's own 16-bit output is
    symmetric too), so a gain that leaves peaks above 1.0 clips them.
    `peak` (the largest absolute sample of the audio, from analyze) saves a
    pass over the data when the caller has it. Raises OSError if the file
    cannot be written.
    """
    if bits not in (16, 24):
        raise AudioError("only 16 and 24-bit WAV files can be written")
    if rounding not in _ROUNDING:
        raise AudioError(f"unknown rounding {rounding!r}")
    to_int = _ROUNDING[rounding]
    limit = 32767 if bits == 16 else 8388607
    if peak is None:
        peak = peak_of(audio.samples)
    may_clip = peak * gain > 1.0
    scale = gain * limit
    samples = audio.samples
    width = bits // 8
    data_size = len(samples) * width
    if data_size + 36 > 0xFFFFFFFF:
        raise AudioError("audio too long for a WAV file")
    header = b"".join((
        b"RIFF", struct.pack("<I", 36 + data_size + (data_size & 1)),
        b"WAVEfmt ",
        struct.pack("<IHHIIHH", 16, _WAVE_FORMAT_PCM, audio.channels,
                    audio.samplerate,
                    audio.samplerate * audio.channels * width,
                    audio.channels * width, bits),
        b"data", struct.pack("<I", data_size),
    ))
    with open(path, "wb") as out:
        out.write(header)
        for start in range(0, len(samples), _CHUNK):
            chunk = samples[start:start + _CHUNK]
            # Limiting costs a pass of its own: only where the gain
            # pushes this part of the audio past full scale.
            clamp = may_clip and peak_of(chunk) * gain > 1.0
            out.write(_pcm_chunk(chunk, scale, limit, bits, clamp, to_int))
        if data_size & 1:
            out.write(b"\0")
