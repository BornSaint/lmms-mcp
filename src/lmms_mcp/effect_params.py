"""Parameters of the built-in LMMS effects (as LMMS 1.2 stores them).

An effect's <effect> element holds a controls element (named per plugin,
see effects.KNOWN_EFFECTS) whose attributes are the plugin's knobs. When an
attribute is missing LMMS uses the plugin default. This module knows, per
effect, every parameter: the friendly name the tools use, the real XML
attribute, type, range, default, unit and meaning. All of it was checked
against the LMMS 1.2 plugin sources, the demo projects shipped with LMMS and
renders with LMMS 1.2.2 (changing an attribute changes the sound as listed).

Three effects keep a curve in the controls (waveshaper, dynamicsprocessor,
multitapecho). LMMS reads those curves unconditionally, so an empty controls
element makes it play whatever was in memory: the curves are always written
(`ensure_graph_defaults`). Their curves are set through friendly parameters
(`shape`, `threshold_db` + `ratio`, `tap_levels_db` + `tap_cutoff_hz`).

Also here, per effect, the "starting settings" add_effect applies so that a
plain add_effect does what an agent expects (see EFFECT_START).
"""

from __future__ import annotations

import base64
import math
import struct
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


class ParamError(ValueError):
    """A parameter name or value was refused. `valid` lists the names that
    would have been accepted (set for unknown names)."""

    def __init__(self, message: str, valid: list[str] | None = None):
        super().__init__(message)
        self.valid = valid or []


@dataclass(frozen=True)
class Param:
    """One parameter of an effect.

    name: friendly name used by the tools. attr: attribute on the controls
    element (also accepted as a name). kind: "float", "int", "bool",
    "choice" (stored as the index into `choices`) or "curve" (stored as
    base64 floats in `attr`, set through special handling).
    """

    name: str
    attr: str
    kind: str
    default: float | int | bool | str
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    description: str = ""
    choices: tuple = ()
    # Extra attributes to write for a choice: index -> {attr: value}.
    also: Callable[[int], dict[str, str]] | None = field(
        default=None, compare=False)


def _f(name, attr, default, lo, hi, unit="", description=""):
    return Param(name, attr, "float", default, lo, hi, unit, description)


def _i(name, attr, default, lo, hi, unit="", description=""):
    return Param(name, attr, "int", default, lo, hi, unit, description)


def _b(name, attr, default, description=""):
    return Param(name, attr, "bool", default, description=description)


def _c(name, attr, default, choices, description="", also=None):
    return Param(name, attr, "choice", default, choices=tuple(choices),
                 description=description, also=also)


def _curve(name, attr, default, unit="", description="", lo=None, hi=None):
    return Param(name, attr, "curve", default, lo, hi, unit, description)


# --- eq -----------------------------------------------------------------

_EQ_SLOPES = (12, 24, 48)


def _eq_slope(prefix: str) -> Callable[[int], dict[str, str]]:
    """The LMMS eq reads the slope from three flags (the index attribute
    alone does nothing); write them the way LMMS itself saves them."""
    def flags(index: int) -> dict[str, str]:
        return {f"{prefix}{s}": "1" if i == index else "0"
                for i, s in enumerate(_EQ_SLOPES)}
    return flags


def _eq_params() -> list[Param]:
    out = [
        _f("input_gain_db", "Inputgain", 0.0, -60, 20, "dB", "Gain before the bands"),
        _f("output_gain_db", "Outputgain", 0.0, -60, 20, "dB", "Gain after the bands"),
        _b("hp_on", "HPactive", False, "High-pass filter on"),
        _f("hp_freq_hz", "HPfreq", 31.0, 20, 20000, "Hz", "High-pass cutoff"),
        _f("hp_res", "HPres", 0.707, 0.003, 10, "", "High-pass resonance (Q)"),
        _c("hp_slope_db", "HP", 12, _EQ_SLOPES, "High-pass slope in dB per octave",
           also=_eq_slope("HP")),
        _b("lowshelf_on", "Lowshelfactive", False, "Low shelf on"),
        _f("lowshelf_gain_db", "Lowshelfgain", 0.0, -18, 18, "dB", "Low shelf boost/cut"),
        _f("lowshelf_freq_hz", "LowShelffreq", 80.0, 20, 20000, "Hz", "Low shelf corner"),
        _f("lowshelf_res", "LowShelfres", 1.4, 0.55, 10, "", "Low shelf resonance"),
    ]
    for n, freq in ((1, 120.0), (2, 250.0), (3, 2000.0), (4, 4000.0)):
        out += [
            _b(f"peak{n}_on", f"Peak{n}active", False, f"Peak band {n} on"),
            _f(f"peak{n}_gain_db", f"Peak{n}gain", 0.0, -18, 18, "dB",
               f"Peak band {n} boost/cut"),
            _f(f"peak{n}_freq_hz", f"Peak{n}freq", freq, 20, 20000, "Hz",
               f"Peak band {n} centre"),
            _f(f"peak{n}_bw", f"Peak{n}bw", 0.3, 0.1, 4, "octaves",
               f"Peak band {n} bandwidth"),
        ]
    out += [
        _b("highshelf_on", "Highshelfactive", False, "High shelf on"),
        _f("highshelf_gain_db", "HighShelfgain", 0.0, -18, 18, "dB", "High shelf boost/cut"),
        _f("highshelf_freq_hz", "Highshelffreq", 12000.0, 20, 20000, "Hz", "High shelf corner"),
        _f("highshelf_res", "HighShelfres", 1.4, 0.55, 10, "", "High shelf resonance"),
        _b("lp_on", "LPactive", False, "Low-pass filter on"),
        _f("lp_freq_hz", "LPfreq", 18000.0, 20, 20000, "Hz", "Low-pass cutoff"),
        _f("lp_res", "LPres", 0.707, 0.003, 10, "", "Low-pass resonance (Q)"),
        _c("lp_slope_db", "LP", 12, _EQ_SLOPES, "Low-pass slope in dB per octave",
           also=_eq_slope("LP")),
    ]
    return out


# --- dualfilter -----------------------------------------------------------

_FILTER_TYPES = (
    "lowpass", "highpass", "bandpass_csg", "bandpass_czpg", "notch",
    "allpass", "moog", "2x_lowpass", "rc_lowpass_12", "rc_bandpass_12",
    "rc_highpass_12", "rc_lowpass_24", "rc_bandpass_24", "rc_highpass_24",
    "vocal_formant", "2x_moog", "sv_lowpass", "sv_bandpass", "sv_highpass",
    "sv_notch", "fast_formant", "tripole",
)


def _dualfilter_params() -> list[Param]:
    out = []
    for n in (1, 2):
        out += [
            _b(f"filter{n}_on", f"enabled{n}", True,
               f"Filter {n} on (an off filter outputs silence into the mix)"),
            _c(f"filter{n}_type", f"filter{n}", "lowpass", _FILTER_TYPES,
               f"Filter {n} type"),
            _f(f"cutoff{n}_hz", f"cut{n}", 7000.0, 1, 20000, "Hz",
               f"Filter {n} cutoff"),
            _f(f"resonance{n}", f"res{n}", 0.5, 0.01, 10, "",
               f"Filter {n} resonance (Q)"),
            _f(f"gain{n}_pct", f"gain{n}", 100.0, 0, 200, "%",
               f"Filter {n} output gain"),
        ]
        if n == 1:
            out.append(_f(
                "mix", "mix", 0.0, -1, 1, "",
                "Blend of the two filters in parallel: -1 = filter 1 only, "
                "0 = half of each, 1 = filter 2 only"))
    return out


# --- the table --------------------------------------------------------------

EFFECT_PARAMS: dict[str, tuple[Param, ...]] = {
    "amplifier": (
        _f("volume", "volume", 100.0, 0, 200, "%", "Overall gain; 100 = unchanged"),
        _f("pan", "pan", 0.0, -100, 100, "",
           "Balance: -100 = left only, 0 = centre, 100 = right only"),
        _f("left_gain", "left", 100.0, 0, 200, "%", "Left channel gain"),
        _f("right_gain", "right", 100.0, 0, 200, "%", "Right channel gain"),
    ),
    "bassbooster": (
        _f("freq_hz", "freq", 100.0, 50, 200, "Hz", "Frequency below which bass is boosted"),
        _f("gain", "gain", 1.0, 0.1, 5, "", "Input gain of the booster (more = more bass)"),
        _f("ratio", "ratio", 2.0, 0.1, 10, "", "Strength of the boost curve (more = more bass)"),
    ),
    "bitcrush": (
        _f("input_gain_db", "ingain", 0.0, -20, 20, "dB", "Gain before crushing (drives the bit reduction)"),
        _f("input_noise_pct", "innoise", 0.0, 0, 100, "%", "Noise added before crushing"),
        _f("output_gain_db", "outgain", 0.0, -20, 20, "dB", "Gain after crushing"),
        _f("output_clip_db", "outclip", 0.0, -20, 20, "dB", "Hard-clip level of the output"),
        _i("sample_rate_hz", "rate", 44100, 20, 44100, "Hz",
           "Sample-rate reduction: lower = more aliasing, 44100 = off"),
        _f("stereo_diff_pct", "stereodiff", 0.0, 0, 50, "%",
           "Different rate on left and right"),
        _i("bit_levels", "levels", 256, 1, 256, "",
           "Quantisation levels: 256 is clean, 16 is 4-bit, 2 is harsh"),
        _b("rate_on", "rateon", True, "Sample-rate reduction on"),
        _b("depth_on", "depthon", True, "Bit-depth reduction on"),
    ),
    "crossovereq": (
        _f("xover_1_2_hz", "xover12", 125.0, 50, 10000, "Hz", "Crossover between bands 1 and 2"),
        _f("xover_2_3_hz", "xover23", 1250.0, 50, 20000, "Hz", "Crossover between bands 2 and 3"),
        _f("xover_3_4_hz", "xover34", 5000.0, 50, 20000, "Hz", "Crossover between bands 3 and 4"),
        _f("band1_gain_db", "gain1", 0.0, -60, 30, "dB", "Band 1 (lows) gain"),
        _f("band2_gain_db", "gain2", 0.0, -60, 30, "dB", "Band 2 gain"),
        _f("band3_gain_db", "gain3", 0.0, -60, 30, "dB", "Band 3 gain"),
        _f("band4_gain_db", "gain4", 0.0, -60, 30, "dB", "Band 4 (highs) gain"),
        _b("band1_on", "mute1", True, "Band 1 passes (False silences it; LMMS names the switch 'mute')"),
        _b("band2_on", "mute2", True, "Band 2 passes"),
        _b("band3_on", "mute3", True, "Band 3 passes"),
        _b("band4_on", "mute4", True, "Band 4 passes"),
    ),
    "delay": (
        _f("time_s", "DelayTimeSamples", 0.5, 0.01, 5, "s",
           "Delay time (the XML name says samples, the value is seconds)"),
        _f("feedback", "FeebackAmount", 0.0, 0, 1, "",
           "Share of the echo fed back in: 0 = one echo, 0.5 = repeats halving"),
        _f("lfo_period_s", "LfoFrequency", 2.0, 0.01, 5, "s",
           "Period of the delay-time wobble (the XML name says frequency)"),
        _f("lfo_amount_s", "LfoAmount", 0.0, 0, 0.5, "s",
           "Depth of the delay-time wobble; 0 = steady"),
        _f("out_gain_db", "OutGain", 0.0, -60, 20, "dB", "Gain of the echoes"),
    ),
    "dualfilter": tuple(_dualfilter_params()),
    "dynamicsprocessor": (
        _f("input_gain", "inputGain", 1.0, 0, 5, "", "Gain before the curve (1 = unchanged)"),
        _f("output_gain", "outputGain", 1.0, 0, 5, "", "Gain after the curve (make-up gain)"),
        _f("attack_ms", "attack", 10.0, 1, 500, "ms", "How fast the level follows a rise"),
        _f("release_ms", "release", 100.0, 1, 500, "ms", "How fast the level follows a fall"),
        _c("stereo_mode", "stereoMode", "maximum", ("maximum", "average", "unlinked"),
           "How the two channels share one level: the louder one, the mean, or each alone"),
        _curve("threshold_db", "waveShape", 0.0, "dB",
               "RMS level above which the level is reduced (0 = never)", -60, 0),
        _curve("ratio", "waveShape", 1.0, "",
               "Reduction above the threshold, N:1 (1 = off, 100 = limiter)", 1, 100),
    ),
    "eq": tuple(_eq_params()),
    "flanger": (
        _f("delay_s", "DelayTimeSamples", 0.001, 0.0001, 0.05, "s", "Base delay time"),
        _f("lfo_period_s", "LfoFrequency", 0.25, 0.01, 60, "s",
           "Period of the sweep (the XML name says frequency): 4 = slow"),
        _f("lfo_amount_s", "LfoAmount", 0.0, 0, 0.0025, "s",
           "Sweep depth; 0 = a static comb filter, no flanging"),
        _f("feedback", "Feedback", 0.0, 0, 1, "", "Regeneration: more = more metallic ring"),
        _f("noise", "WhiteNoise", 0.0, 0, 0.05, "", "White noise added to the input"),
        _b("invert_feedback", "Invert", False, "Cross the feedback between the channels"),
    ),
    "multitapecho": (
        _i("steps", "steps", 16, 4, 32, "", "Number of echo taps"),
        _f("step_length_ms", "steplength", 100.0, 1, 500, "ms", "Time between taps"),
        _f("dry_gain_db", "drygain", 0.0, -80, 20, "dB",
           "Level of the direct sound inside the effect (-80 = echoes only)"),
        _b("swap_inputs", "swapinputs", False, "Echoes swap left and right"),
        _curve("tap_levels_db", "ampsteps", 0.0, "dB",
               "Level of each tap: a number (all taps), a list of `steps` "
               "numbers, or \"decay\"", -60, 0),
        _curve("tap_cutoff_hz", "lpsteps", 20000.0, "Hz",
               "Low-pass cutoff of each tap: a number or a list of `steps` numbers",
               20, 20000),
    ),
    "reverbsc": (
        _f("input_gain_db", "input_gain", 0.0, -60, 15, "dB", "Gain into the reverb"),
        _f("size", "size", 0.89, 0, 1, "", "Reverb length; 1 never decays"),
        _f("color_hz", "color", 10000.0, 100, 15000, "Hz", "Low-pass of the tail (lower = darker)"),
        _f("output_gain_db", "output_gain", 0.0, -60, 15, "dB", "Gain of the reverb"),
    ),
    "stereoenhancer": (
        _f("width", "width", 0.0, 0, 180, "samples",
           "Right channel delayed against the left by this many samples "
           "(1 ms is about 44); 0 = off"),
    ),
    "stereomatrix": (
        _f("left_to_left", "l-l", 1.0, -1, 1, "", "How much of the left input goes to the left output"),
        _f("left_to_right", "l-r", 0.0, -1, 1, "", "Left input to the right output"),
        _f("right_to_left", "r-l", 0.0, -1, 1, "", "Right input to the left output"),
        _f("right_to_right", "r-r", 1.0, -1, 1, "", "Right input to the right output"),
    ),
    "waveshaper": (
        _f("input_gain", "inputGain", 1.0, 0, 5, "", "Gain before the shaper (drive)"),
        _f("output_gain", "outputGain", 1.0, 0, 5, "", "Gain after the shaper"),
        _b("clip_input", "clipInput", False, "Clip the input at full scale before shaping"),
        _curve("shape", "waveShape", "linear", "",
               "Transfer curve: a name (see WAVESHAPES) or a list of 200 "
               "numbers from 0 to 1 (output for input 1/200 ... 200/200)"),
    ),
}

# What the effect curves look like.
_GRAPH_POINTS = 200


def _ramp() -> list[float]:
    return [(i + 1) / _GRAPH_POINTS for i in range(_GRAPH_POINTS)]


WAVESHAPES: dict[str, Callable[[float], float]] = {
    "linear": lambda x: x,
    "soft_clip": lambda x: math.tanh(3 * x) / math.tanh(3),
    "hard_clip": lambda x: min(1.0, 3 * x),
    "fuzz": lambda x: min(1.0, 20 * x),
    "fold": lambda x: 1 - abs(1 - 2 * x),
    "crush": lambda x: math.ceil(x * 8) / 8,
}

WAVESHAPE_DOC = {
    "linear": "no change (the LMMS default)",
    "soft_clip": "tanh saturation, adds odd harmonics, quiet parts get louder",
    "hard_clip": "clips at one third of full scale",
    "fuzz": "almost a square wave",
    "fold": "wavefolder, loud parts fold back",
    "crush": "8-step staircase",
}

TAP_MIN_DB, TAP_MAX_DB = -60.0, 0.0
TAP_MIN_HZ, TAP_MAX_HZ = 20.0, 20000.0


def decay_levels(steps: int) -> list[float]:
    """Tap levels that fade by 4 dB per tap, starting at -9 dB."""
    return [max(TAP_MIN_DB, -9.0 - 4.0 * i) for i in range(steps)]


# --- settings every effect has (stored on the <effect> element) -----------------

EFFECT_LEVEL: tuple[Param, ...] = (
    _f("wet", "wet", 1.0, 0, 1, "", "Wet/dry mix: 0 = original sound only, 1 = effect only"),
    _i("decay_ms", "autoquit", 1, 1, 8000, "ms",
       "How long LMMS keeps the effect running after its input went "
       "silent. At 1 it cuts echoes after a gap and reverb tails"),
)


def validate_effect_level(effect: str, name: str, raw):
    """The value of "wet" or "decay_ms" (ParamError when refused)."""
    p = next(q for q in EFFECT_LEVEL if q.name == name)
    return _validate(effect, p, raw)


def split_effect_level(effect: str, params: dict) -> tuple[dict, dict]:
    """Split params into the effect-level settings (wet, decay_ms; spelled
    by name or by attribute, validated) and the plugin's own."""
    level: dict = {}
    rest: dict = {}
    for key, raw in params.items():
        spelled = str(key).strip().lower()
        p = next((q for q in EFFECT_LEVEL
                  if spelled in (q.name, q.attr)), None)
        if p is None:
            rest[key] = raw
        elif p.name in level:
            raise ParamError(f"{effect}: '{p.name}' was given twice.")
        else:
            level[p.name] = _validate(effect, p, raw)
    return level, rest


# --- starting settings --------------------------------------------------------

@dataclass(frozen=True)
class Start:
    """What add_effect applies when the caller does not say otherwise.

    wet: the wet/dry mix. Effects whose output holds only the processed
    signal (delay, reverbsc, flanger) start below 1.0, otherwise the dry
    sound would vanish. params: friendly name -> value. tempo: bpm ->
    params, for settings synced to the project tempo. note: shown when
    the caller passed no params, for effects that do nothing audible at
    their plugin defaults and have no sensible audible start. decay_ms: how
    long LMMS keeps the effect running through silence (its "Decay"; LMMS
    stops an effect whose output has been silent for that long, which cuts
    echoes and tails unless it is raised; 1 is LMMS's default).
    """

    wet: float = 1.0
    params: dict = field(default_factory=dict)
    tempo: Callable[[float], dict] | None = None
    tempo_text: str = ""
    why_wet: str = ""
    note: str = ""
    decay_ms: int = 1


def _eighth_note_s(bpm: float) -> float:
    return min(5.0, max(0.01, 30.0 / bpm))


def _sixteenth_note_ms(bpm: float) -> float:
    return min(500.0, max(1.0, 15000.0 / bpm))


EFFECT_START: dict[str, Start] = {
    "amplifier": Start(note=(
        "Neutral at its defaults (volume 100 = unchanged): set volume "
        "(0-200), pan, left_gain or right_gain with set_effect_params.")),
    "bassbooster": Start(),
    "bitcrush": Start(params={"sample_rate_hz": 8000, "bit_levels": 16}),
    "crossovereq": Start(note=(
        "Neutral at its defaults (every band at 0 dB): set band1_gain_db "
        "... band4_gain_db (and the crossovers) with set_effect_params.")),
    "delay": Start(
        wet=0.4, params={"feedback": 0.3}, decay_ms=8000,
        tempo=lambda bpm: {"time_s": round(_eighth_note_s(bpm), 4)},
        tempo_text="time_s = a 1/8 note at the project tempo (30/bpm s)",
        why_wet=("delay outputs only the delayed signal, so at wet=1.0 the "
                 "dry sound would disappear and only the echo be heard")),
    "dualfilter": Start(params={"cutoff1_hz": 1200, "cutoff2_hz": 1200}),
    "dynamicsprocessor": Start(params={"threshold_db": -12, "ratio": 4}),
    "eq": Start(note=(
        "Neutral at its defaults (every band is off): switch bands on and "
        "shape them with set_effect_params, e.g. peak1_on, peak1_gain_db, "
        "peak1_freq_hz, lowshelf_on, hp_on.")),
    "flanger": Start(
        wet=0.5, decay_ms=8000,
        params={"delay_s": 0.002, "lfo_period_s": 3, "lfo_amount_s": 0.0015,
                "feedback": 0.3},
        why_wet=("flanger outputs only the delayed signal and gets its "
                 "sound from mixing it with the dry sound")),
    "multitapecho": Start(
        params={"steps": 8, "tap_levels_db": "decay"}, decay_ms=8000,
        tempo=lambda bpm: {"step_length_ms": round(_sixteenth_note_ms(bpm), 1)},
        tempo_text="step_length_ms = a 1/16 note at the project tempo"),
    "reverbsc": Start(
        wet=0.3, decay_ms=8000,
        why_wet=("reverbsc outputs only the reverb, so at wet=1.0 the dry "
                 "sound would disappear")),
    "stereoenhancer": Start(params={"width": 60}),
    "stereomatrix": Start(note=(
        "Neutral at its defaults (left stays left, right stays right): set "
        "left_to_left, left_to_right, right_to_left, right_to_right with "
        "set_effect_params; left_to_right = right_to_left = 1 with "
        "left_to_left = right_to_right = 0 swaps the channels.")),
    "waveshaper": Start(params={"shape": "soft_clip", "input_gain": 1.5,
                                "output_gain": 0.6}),
}


def start_params(effect: str, bpm: float) -> dict:
    """The starting params of an effect at a tempo (friendly names)."""
    start = EFFECT_START.get(effect)
    if start is None:
        return {}
    out = dict(start.params)
    if start.tempo is not None:
        out.update(start.tempo(bpm))
    return out


def describe_start(effect: str) -> str:
    """One line for tool docstrings: what add_effect starts the effect with."""
    start = EFFECT_START.get(effect)
    if start is None:
        return "plugin defaults"
    parts = []
    if start.wet != 1.0:
        parts.append(f"wet {start.wet:g}")
    parts += [f"{k} {_show(v)}" for k, v in start.params.items()]
    if start.tempo_text:
        parts.append(start.tempo_text)
    if start.decay_ms != 1:
        parts.append(f"decay_ms {start.decay_ms}")
    if not parts:
        return ("neutral at the plugin defaults: set parameters" if start.note
                else "plugin defaults")
    return ", ".join(parts)


# --- lookup -----------------------------------------------------------------

def params_of(effect: str) -> tuple[Param, ...]:
    return EFFECT_PARAMS.get(effect, ())


def has_params(effect: str) -> bool:
    return effect in EFFECT_PARAMS


def _lookup(effect: str) -> dict[str, Param]:
    """Every accepted spelling (friendly name or XML attribute, any case)."""
    table = {}
    for p in EFFECT_PARAMS[effect]:
        table.setdefault(p.name.lower(), p)
    for p in EFFECT_PARAMS[effect]:
        if p.kind != "curve":  # a curve's attribute holds base64 data
            table.setdefault(p.attr.lower(), p)
    return table


def resolve_names(effect: str, params: dict) -> dict[str, object]:
    """Map the keys of `params` to friendly names; refuse unknown names."""
    if effect not in EFFECT_PARAMS:
        raise ParamError(
            f"'{effect}' has no settable parameters here; it runs with the "
            f"plugin defaults.")
    table = _lookup(effect)
    out: dict[str, object] = {}
    for key, value in params.items():
        p = table.get(str(key).strip().lower())
        if p is None:
            names = [q.name for q in EFFECT_PARAMS[effect]]
            raise ParamError(
                f"{effect} has no parameter '{key}'. Valid parameters: "
                f"{', '.join(names)}.", valid=names)
        if p.name in out:
            raise ParamError(
                f"{effect}: parameter '{p.name}' was given twice.")
        out[p.name] = value
    return out


# --- values ---------------------------------------------------------------------

def _show(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def format_number(v: float) -> str:
    text = f"{v:.8f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _number(effect: str, p: Param, raw) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ParamError(
            f"{effect}: '{p.name}' must be a number (got {raw!r}).")
    v = float(raw)
    if not math.isfinite(v):
        raise ParamError(f"{effect}: '{p.name}' must be a finite number.")
    return v


def _range_text(p: Param) -> str:
    unit = f" {p.unit}" if p.unit else ""
    return f"{_show(float(p.minimum))} to {_show(float(p.maximum))}{unit}"


def _check_range(effect: str, p: Param, v: float) -> None:
    eps = 1e-9 * max(1.0, abs(v))
    if not (p.minimum - eps <= v <= p.maximum + eps):
        raise ParamError(
            f"{effect}: '{p.name}' = {_show(v)} is out of range; valid "
            f"range is {_range_text(p)}.")


def _validate(effect: str, p: Param, raw):
    """The value of a scalar parameter as a Python value, or ParamError."""
    if p.kind == "float":
        v = _number(effect, p, raw)
        _check_range(effect, p, v)
        return min(max(v, p.minimum), p.maximum)
    if p.kind == "int":
        v = _number(effect, p, raw)
        if v != int(v):
            raise ParamError(
                f"{effect}: '{p.name}' must be a whole number "
                f"(got {_show(v)}).")
        _check_range(effect, p, v)
        return int(v)
    if p.kind == "bool":
        if isinstance(raw, bool) or (
                isinstance(raw, (int, float)) and raw in (0, 1)):
            return bool(raw)
        raise ParamError(
            f"{effect}: '{p.name}' must be true or false (got {raw!r}).")
    if p.kind == "choice":
        return p.choices[_choice_index(effect, p, raw)]
    raise AssertionError(p.kind)


def _choice_index(effect: str, p: Param, raw) -> int:
    texts = [str(c).lower() for c in p.choices]
    if not isinstance(raw, bool):
        if str(raw).strip().lower() in texts:
            return texts.index(str(raw).strip().lower())
        if isinstance(raw, int) and 0 <= raw < len(p.choices):
            return raw
    raise ParamError(
        f"{effect}: '{p.name}' must be one of "
        f"{', '.join(str(c) for c in p.choices)} (got {raw!r}).")


def _xml_value(p: Param, v) -> str:
    if p.kind == "bool":
        return "1" if v else "0"
    if p.kind == "choice":
        return str(p.choices.index(v))
    if p.kind == "int":
        return str(int(v))
    return format_number(v)


# --- curves --------------------------------------------------------------------

def pack_floats(values: Iterable[float]) -> str:
    vals = list(values)
    return base64.b64encode(struct.pack(f"<{len(vals)}f", *vals)).decode("ascii")


def unpack_floats(text: str | None) -> list[float] | None:
    if not text:
        return None
    try:
        raw = base64.b64decode(text, validate=True)
    except ValueError:
        return None
    if len(raw) % 4:
        return None
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def waveshape_curve(name: str) -> list[float]:
    f = WAVESHAPES[name]
    return [min(1.0, max(0.0, f(x))) for x in _ramp()]


def dynamics_curve(threshold_db: float, ratio: float) -> list[float]:
    """Transfer curve of a compressor: unchanged up to the threshold, above
    it the level in dB rises 1/ratio dB per dB."""
    t = 10 ** (threshold_db / 20)
    return [x if x <= t else t * (x / t) ** (1 / ratio) for x in _ramp()]


def fit_dynamics_curve(samples: list[float]) -> tuple[float, float] | None:
    """(threshold_db, ratio) of a curve made by dynamics_curve, else None."""
    if len(samples) != _GRAPH_POINTS:
        return None
    ramp = _ramp()
    if all(abs(s - x) < 1e-5 for s, x in zip(samples, ramp)):
        return 0.0, 1.0
    x1, x2 = ramp[-2], ramp[-1]
    y1, y2 = samples[-2], samples[-1]
    if min(y1, y2) <= 0 or y2 == y1:
        return None
    slope = (math.log(y2) - math.log(y1)) / (math.log(x2) - math.log(x1))
    if not 0 < slope < 1:
        return None
    ratio = 1 / slope
    # ln y = ln t * (1 - slope) + slope * ln x  ->  ln t
    ln_t = (math.log(y2) - slope * math.log(x2)) / (1 - slope)
    threshold_db = 20 * ln_t / math.log(10)
    if not -60.5 <= threshold_db <= 0.5 or not 1 < ratio <= 100.5:
        return None
    fit = dynamics_curve(threshold_db, ratio)
    if max(abs(a - b) for a, b in zip(fit, samples)) > 1e-4:
        return None
    return round(threshold_db, 2), round(ratio, 2)


def _close(a: list[float], b: list[float], tol: float = 1e-4) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def _get_attr(controls: ET.Element, attr: str) -> str | None:
    """A knob's value: the attribute, or the child element LMMS writes for
    knobs on a logarithmic scale or linked to automation."""
    value = controls.get(attr)
    if value is not None:
        return value
    child = controls.find(attr)
    return child.get("value") if child is not None else None


def _set_attr(controls: ET.Element, attr: str, value: str) -> None:
    controls.set(attr, value)
    child = controls.find(attr)
    if child is not None:
        child.set("value", value)


def _steps_of(controls: ET.Element) -> int:
    raw = _get_attr(controls, "steps")
    try:
        return int(float(raw)) if raw is not None else 16
    except ValueError:
        return 16


def ensure_graph_defaults(effect: str, controls: ET.Element) -> None:
    """Write the curves of waveshaper / dynamicsprocessor / multitapecho
    when the controls lack them (LMMS would play uninitialised memory)."""
    if effect in ("waveshaper", "dynamicsprocessor"):
        if unpack_floats(controls.get("waveShape")) is None:
            controls.set("waveShape", pack_floats(_ramp()))
    elif effect == "multitapecho":
        n = _steps_of(controls)
        if unpack_floats(controls.get("ampsteps")) is None:
            controls.set("ampsteps", pack_floats([0.0] * n))
        if unpack_floats(controls.get("lpsteps")) is None:
            controls.set("lpsteps", pack_floats([3.0] * n))


def _list_of_numbers(effect: str, p: Param, raw, count: int) -> list[float]:
    if isinstance(raw, (list, tuple)):
        if len(raw) != count:
            raise ParamError(
                f"{effect}: '{p.name}' needs {count} values (one per step), "
                f"got {len(raw)}.")
        values = [_number(effect, p, v) for v in raw]
    else:
        values = [_number(effect, p, raw)] * count
    for v in values:
        _check_range(effect, p, v)
    return values


def _levels_label(amp: list[float]):
    """Tap levels as shown: "decay" for the decay shape, else the list."""
    if _close(amp, decay_levels(len(amp)), 1e-3):
        return "decay"
    return [round(v, 2) for v in amp]


def _cutoffs_hz(lp: list[float]) -> list[float]:
    return [round(20.0 * 10 ** v, 1) for v in lp]


def _resize(values: list[float], count: int, fill: float) -> list[float]:
    values = values[:count]
    return values + [values[-1] if values else fill] * (count - len(values))


def _curve_writes(effect: str, controls: ET.Element, given: dict,
                  writes: dict[str, str], applied: dict, notes: list[str]):
    """Validate the curve parameters in `given` and add their attributes to
    `writes` (and what was set to `applied`)."""
    by_name = {p.name: p for p in EFFECT_PARAMS[effect]}
    if effect == "waveshaper":
        p = by_name["shape"]
        raw = given["shape"]
        if isinstance(raw, str):
            name = raw.strip().lower()
            if name not in WAVESHAPES:
                raise ParamError(
                    f"{effect}: shape '{raw}' is unknown; use one of "
                    f"{', '.join(WAVESHAPES)} or a list of {_GRAPH_POINTS} "
                    f"numbers from 0 to 1.")
            curve, shown = waveshape_curve(name), name
        elif (isinstance(raw, (list, tuple)) and len(raw) == _GRAPH_POINTS):
            curve = [_number(effect, p, v) for v in raw]
            if any(not 0.0 <= v <= 1.0 for v in curve):
                raise ParamError(
                    f"{effect}: every number of 'shape' must be between 0 "
                    f"and 1.")
            shown = "custom"
        else:
            raise ParamError(
                f"{effect}: 'shape' must be a name ({', '.join(WAVESHAPES)}) "
                f"or a list of {_GRAPH_POINTS} numbers from 0 to 1.")
        writes["waveShape"] = pack_floats(curve)
        applied["shape"] = shown
    elif effect == "dynamicsprocessor":
        current = fit_dynamics_curve(
            unpack_floats(controls.get("waveShape")) or _ramp()) or (0.0, 1.0)
        values = list(current)
        for i, key in enumerate(("threshold_db", "ratio")):
            if key in given:
                p = by_name[key]
                v = _number(effect, p, given[key])
                _check_range(effect, p, v)
                values[i] = v
        writes["waveShape"] = pack_floats(dynamics_curve(*values))
        applied["threshold_db"], applied["ratio"] = values
        if values[1] == 1.0 and "ratio" not in given:
            notes.append("ratio is 1, so nothing is reduced yet: set ratio "
                         "(e.g. 4).")
        if values[0] == 0.0 and "threshold_db" not in given:
            notes.append("threshold_db is 0, so nothing is reduced yet: "
                         "set threshold_db (e.g. -12).")
    elif effect == "multitapecho":
        steps = int(writes["steps"]) if "steps" in writes else _steps_of(controls)
        amp = unpack_floats(controls.get("ampsteps")) or [0.0] * steps
        lp = unpack_floats(controls.get("lpsteps")) or [3.0] * steps
        amp, lp = _resize(amp, steps, 0.0), _resize(lp, steps, 3.0)
        touched = "steps" in writes
        if "tap_levels_db" in given:
            p = by_name["tap_levels_db"]
            raw = given["tap_levels_db"]
            if isinstance(raw, str):
                if raw.strip().lower() != "decay":
                    raise ParamError(
                        f"{effect}: tap_levels_db '{raw}' is unknown; use "
                        f"\"decay\", a number or a list of {steps} numbers.")
                amp = decay_levels(steps)
            else:
                amp = _list_of_numbers(effect, p, raw, steps)
            touched = True
        if "tap_cutoff_hz" in given:
            p = by_name["tap_cutoff_hz"]
            hz = _list_of_numbers(effect, p, given["tap_cutoff_hz"], steps)
            lp = [math.log10(f / 20.0) for f in hz]
            touched = True
        if touched:
            writes["ampsteps"] = pack_floats(amp)
            writes["lpsteps"] = pack_floats(lp)
            if "tap_levels_db" in given or any(abs(v) > 1e-4 for v in amp):
                applied["tap_levels_db"] = _levels_label(amp)
            if "tap_cutoff_hz" in given or any(abs(v - 3.0) > 1e-4 for v in lp):
                applied["tap_cutoff_hz"] = _cutoffs_hz(lp)
    else:
        raise AssertionError(effect)


# --- apply / read -------------------------------------------------------------------

@dataclass
class Applied:
    params: dict
    notes: list[str]


def _crossover_check(effect: str, controls: ET.Element,
                     values: dict[str, float]) -> None:
    """The three crossovers must ascend (LMMS would silently push others)."""
    names = ("xover_1_2_hz", "xover_2_3_hz", "xover_3_4_hz")
    merged = []
    for p in EFFECT_PARAMS[effect]:
        if p.name in names:
            raw = _get_attr(controls, p.attr)
            merged.append(values.get(
                p.name, float(raw) if raw is not None else p.default))
    if merged != sorted(merged):
        raise ParamError(
            f"{effect}: the crossovers must ascend (xover_1_2_hz <= "
            f"xover_2_3_hz <= xover_3_4_hz); the result would be "
            f"{merged[0]:g}, {merged[1]:g}, {merged[2]:g} Hz. Set all that "
            f"must move together.")


def apply_params(effect: str, controls: ET.Element, params: dict) -> Applied:
    """Validate `params` (friendly names or XML attributes) and write them
    on the controls element. Nothing is written when anything is refused.

    Returns what was set (friendly names, values as stored) plus notes.
    """
    given = resolve_names(effect, params)
    by_name = {p.name: p for p in EFFECT_PARAMS[effect]}
    given = {n: given[n] for n in by_name if n in given}  # table order
    writes: dict[str, str] = {}
    applied: dict = {}
    notes: list[str] = []
    scalars: dict[str, object] = {}
    curves: dict[str, object] = {}
    for name, raw in given.items():
        p = by_name[name]
        if p.kind == "curve":
            curves[name] = raw
            continue
        value = _validate(effect, p, raw)
        scalars[name] = value
        writes[p.attr] = _xml_value(p, value)
        if p.also is not None:
            writes.update(p.also(p.choices.index(value)))
        applied[name] = value
    if effect == "crossovereq":
        _crossover_check(effect, controls, {
            k: v for k, v in scalars.items() if k.startswith("xover_")})
    if curves or ("steps" in writes and effect == "multitapecho"):
        _curve_writes(effect, controls, curves, writes, applied, notes)
    for attr, text in writes.items():
        _set_attr(controls, attr, text)
    return Applied(applied, notes)


def _parse(p: Param, text: str):
    try:
        if p.kind == "bool":
            return float(text) != 0
        if p.kind == "int":
            return int(float(text))
        if p.kind == "choice":
            return p.choices[int(float(text))]
        return float(text)
    except (ValueError, IndexError):
        return None


def read_params(effect: str, controls: ET.Element,
                changed_only: bool = True) -> dict:
    """The parameters of an effect as set in its controls element.

    By default only those that differ from the plugin default (friendly
    names). Curves are described: waveshaper `shape` by name or "custom",
    dynamicsprocessor `threshold_db` + `ratio` (or `curve: "custom"`),
    multitapecho `tap_levels_db` / `tap_cutoff_hz` as lists.
    """
    out: dict = {}
    if effect not in EFFECT_PARAMS:
        return out
    for p in EFFECT_PARAMS[effect]:
        if p.kind == "curve":
            continue
        text = _get_attr(controls, p.attr)
        value = _parse(p, text) if text is not None else None
        if value is None:
            if changed_only:
                continue
            value = p.default
        elif changed_only and _same(p, value, p.default):
            continue
        out[p.name] = round(value, 6) if isinstance(value, float) else value
    out.update(_read_curves(effect, controls, changed_only))
    return out


def _same(p: Param, a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    return a == b


def _read_curves(effect: str, controls: ET.Element, changed_only: bool) -> dict:
    out: dict = {}
    if effect == "waveshaper":
        samples = unpack_floats(controls.get("waveShape"))
        if samples is None:
            name = "linear"
        else:
            name = next((n for n in WAVESHAPES
                         if _close(samples, waveshape_curve(n), 1e-5)), "custom")
        if name != "linear" or not changed_only:
            out["shape"] = name
    elif effect == "dynamicsprocessor":
        samples = unpack_floats(controls.get("waveShape"))
        fit = (0.0, 1.0) if samples is None else fit_dynamics_curve(samples)
        if fit is None:
            out["curve"] = "custom"
        elif fit != (0.0, 1.0) or not changed_only:
            out["threshold_db"], out["ratio"] = fit
    elif effect == "multitapecho":
        steps = _steps_of(controls)
        amp = unpack_floats(controls.get("ampsteps"))
        lp = unpack_floats(controls.get("lpsteps"))
        if amp is not None and (not changed_only or any(abs(v) > 1e-4 for v in amp)):
            out["tap_levels_db"] = _levels_label(amp)
        elif amp is None and not changed_only:
            out["tap_levels_db"] = [0.0] * steps
        if lp is not None and (not changed_only or any(abs(v - 3.0) > 1e-4 for v in lp)):
            out["tap_cutoff_hz"] = _cutoffs_hz(lp)
        elif lp is None and not changed_only:
            out["tap_cutoff_hz"] = [20000.0] * steps
    return out


# --- documentation ---------------------------------------------------------------------

def param_dict(p: Param) -> dict:
    """A parameter as shown by the effects resource."""
    out: dict = {
        "name": p.name, "xml_attribute": p.attr, "type": p.kind,
        "default": p.default, "description": p.description,
    }
    if p.unit:
        out["unit"] = p.unit
    if p.minimum is not None:
        out["min"] = p.minimum
        out["max"] = p.maximum
    if p.choices:
        out["choices"] = list(p.choices)
    return out


def _brief(p: Param) -> str:
    if p.kind == "bool":
        return f"{p.name} (bool, {str(p.default).lower()})"
    if p.kind == "choice":
        if len(p.choices) <= 4:
            return f"{p.name} ({'|'.join(str(c) for c in p.choices)}, {p.default})"
        return f"{p.name} (choice, {p.default})"
    if p.kind == "curve" and p.minimum is None:
        return f"{p.name} (name or list, {p.default})"
    unit = f" {p.unit}" if p.unit else ""
    lo, hi = _show(p.minimum), _show(p.maximum)
    return f"{p.name} ({lo}..{hi}{unit}, {_show(p.default)})"


_COMPACT = {
    "eq": (
        "input_gain_db/output_gain_db (-60..20 dB, 0); bands hp, lowshelf, "
        "peak1, peak2, peak3, peak4, highshelf, lp, each with <band>_on "
        "(bool, false), <band>_freq_hz (20..20000 Hz; defaults hp 31, "
        "lowshelf 80, peak1 120, peak2 250, peak3 2000, peak4 4000, "
        "highshelf 12000, lp 18000), <band>_gain_db (-18..18 dB, 0; shelves "
        "and peaks), <band>_bw (peaks, 0.1..4 octaves, 0.3) or <band>_res "
        "(hp, lp, shelves; 0.003..10, 0.707, shelves 0.55..10, 1.4); "
        "hp_slope_db/lp_slope_db (12|24|48, 12)"
    ),
}

# Extra explanation appended to an effect's line in tool docstrings.
_EXTRA = {
    "delay": "lfo_period_s is the period of the wobble, not a frequency",
    "dynamicsprocessor": (
        "threshold_db and ratio build the curve: above the RMS level "
        "threshold_db the level rises 1/ratio dB per dB; ratio 1 or "
        "threshold_db 0 = no reduction, so set both together"),
    "multitapecho": (
        "tap_levels_db: a number (all taps), a list of `steps` numbers or "
        "\"decay\"; tap_cutoff_hz: a number or a list of `steps` numbers"),
    "waveshaper": (
        "shape: " + ", ".join(WAVESHAPES) + ", or a list of 200 numbers "
        "0..1 (output for input 1/200 ... 200/200)"),
}


def describe_params(effect: str) -> str:
    """Compact parameter list of one effect (name, range unit, default)."""
    text = _COMPACT.get(effect) or ", ".join(
        _brief(p) for p in EFFECT_PARAMS[effect])
    if effect in _EXTRA:
        text += ". " + _EXTRA[effect]
    return text


def describe_choices(effects: Iterable[str]) -> list[str]:
    """Lines naming the choices of parameters with long choice lists."""
    seen: list[tuple] = []
    lines = []
    for effect in effects:
        for p in EFFECT_PARAMS.get(effect, ()):
            if p.kind == "choice" and len(p.choices) > 4 and p.choices not in seen:
                seen.append(p.choices)
                lines.append(
                    f"{effect} filter types ({_filter_names(effect, p)}): "
                    f"{', '.join(str(c) for c in p.choices)}")
    return lines


def _filter_names(effect: str, p: Param) -> str:
    sisters = [q.name for q in EFFECT_PARAMS[effect] if q.choices == p.choices]
    return "/".join(sisters)
