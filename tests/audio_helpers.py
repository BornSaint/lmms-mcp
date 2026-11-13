"""Synthetic audio and a fake LMMS for the render_project tests.

No real LMMS runs: FakeLmms stands in for subprocess.run and writes the
WAV that `lmms render ... -a` would. Reading and writing here uses only the
standard library (struct, wave), independent of lmms_mcp.audio_analysis.
"""

import array
import math
import struct
import subprocess
import wave
from pathlib import Path

from lmms_mcp import xml_parser

SAMPLERATE = 8000  # low, so a few seconds of audio stay small and fast


def sine(freq, amplitude, seconds, samplerate=SAMPLERATE, channels=2):
    """Interleaved float samples of a sine (the same in every channel)."""
    frames = int(seconds * samplerate)
    out = array.array("f")
    for i in range(frames):
        value = amplitude * math.sin(2 * math.pi * freq * i / samplerate)
        out.extend([value] * channels)
    return out


def float_wav_bytes(samples, samplerate=SAMPLERATE, channels=2):
    """A WAV file (IEEE float 32-bit) holding interleaved samples."""
    data = array.array("f", samples).tobytes()
    return b"".join((
        b"RIFF", struct.pack("<I", 36 + len(data)), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 3, channels, samplerate,
                             samplerate * channels * 4, channels * 4, 32),
        b"data", struct.pack("<I", len(data)), data,
    ))


def pcm_wav_bytes(ints, bits, samplerate=SAMPLERATE, channels=2):
    """A PCM WAV file of 16, 24 or 32 bit integer interleaved samples."""
    if bits == 16:
        data = array.array("h", ints).tobytes()
    elif bits == 24:
        data = b"".join(int(v).to_bytes(3, "little", signed=True) for v in ints)
    else:
        data = array.array("i", ints).tobytes()
    width = bits // 8
    return b"".join((
        b"RIFF", struct.pack("<I", 36 + len(data)), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, channels, samplerate,
                             samplerate * channels * width,
                             channels * width, bits),
        b"data", struct.pack("<I", len(data)), data,
    ))


def read_pcm(path):
    """(samplerate, channels, bits, list of ints) of a PCM WAV file, read
    with the standard library."""
    with wave.open(str(path), "rb") as wav:
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
        if width == 2:
            values = list(array.array("h", raw))
        elif width == 3:
            values = [int.from_bytes(raw[i:i + 3], "little", signed=True)
                      for i in range(0, len(raw), 3)]
        else:
            values = list(array.array("i", raw))
        return wav.getframerate(), wav.getnchannels(), width * 8, values


class FakeLmms:
    """A stand-in for subprocess.run that "renders" a fixed signal.

    `signal` is what the project plays at Master volume 1.0 (interleaved
    floats). Like LMMS it reads the saved project it is asked to render
    and scales the signal by the volume of the Master mixer channel, so a
    render after the Master volume was lowered is quieter. With -a it writes
    a float WAV; otherwise it writes b"ENCODED:<format>" (a stand-in for an
    encoded file). Every call is recorded: `commands` has the command
    lines, `calls` {"cmd", "float", "master_volume", "config", ...}.
    """

    def __init__(self, signal, samplerate=SAMPLERATE, channels=2, on_run=None):
        self.signal = array.array("f", signal)
        self.samplerate = samplerate
        self.channels = channels
        self.on_run = on_run
        self.calls = []
        self.commands = []  # the command line of every call

    def __call__(self, cmd, **kwargs):
        project = Path(cmd[cmd.index("render") + 1])
        volume = xml_parser.master_volume(xml_parser.load_project(project))
        out = Path(cmd[cmd.index("-o") + 1])
        config = Path(cmd[cmd.index("--config") + 1])
        self.commands.append(cmd)
        self.calls.append({
            "cmd": cmd, "float": "-a" in cmd, "master_volume": volume,
            "config": config, "config_text": config.read_text("utf-8"),
        })
        if self.on_run is not None:
            self.on_run(cmd)
        scaled = [v * volume for v in self.signal]
        if "-a" in cmd:
            out.write_bytes(float_wav_bytes(
                scaled, self.samplerate, self.channels))
        else:
            fmt = cmd[cmd.index("-f") + 1]
            out.write_bytes(b"ENCODED:" + fmt.encode())
        return subprocess.CompletedProcess(cmd, 0, "", "")
