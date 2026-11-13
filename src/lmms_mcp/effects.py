"""Effect chain (fxchain) handling for LMMS projects.

Provides the list of built-in LMMS effect plugins and functions to
manipulate fxchain elements on tracks and mixer channels.
"""

import xml.etree.ElementTree as ET
from collections.abc import Collection

from . import effect_params as params_mod
from .effect_params import ParamError

# Built-in LMMS effects: plugin name -> (description, controls node name).
# Names verified against LMMS source (plugin descriptors + nodeName()).
KNOWN_EFFECTS = {
    "amplifier": ("Volume/gain control", "AmplifierControls"),
    "bassbooster": ("Low frequency booster", "bassboostercontrols"),
    "bitcrush": ("Bit crusher for lo-fi sound", "bitcrushcontrols"),
    "compressor": ("Dynamic range compressor", "CompressorControls"),
    "crossovereq": ("3-band crossover EQ", "crossoevereqcontrols"),
    "delay": ("Tempo-synced delay/echo", "Delay"),
    "dispersion": ("Dispersion/phaser effect", "DispersionControls"),
    "dualfilter": ("Dual filter (2 filters in parallel/serial)", "DualFilterControls"),
    "dynamicsprocessor": ("Dynamics processor (compressor/expander)", "dynamicsprocessor_controls"),
    "eq": ("Parametric equalizer", "Eq"),
    "flanger": ("Flanger modulation effect", "Flanger"),
    "frequencyshifter": ("Frequency shifter", "FrequencyShifterControls"),
    "multitapecho": ("Multi-tap echo", "multitapechocontrols"),
    "reverbsc": ("Studio-quality reverb (Schroeder)", "ReverbSCControls"),
    "slewdistortion": ("Slew-rate distortion", "SlewDistortionControls"),
    "stereoenhancer": ("Stereo width enhancer", "stereoenhancercontrols"),
    "stereomatrix": ("Stereo channel matrix", "stereomatrixcontrols"),
    "waveshaper": ("Waveshaping distortion", "waveshapercontrols"),
}

# Effects that require external hosts/libraries - listed separately so
# agents know they may not work on every system.
EXTERNAL_EFFECTS = {
    "ladspaeffect": "LADSPA plugin host (depends on installed LADSPA libs)",
    "lv2effect": "LV2 plugin host (depends on installed LV2 plugins)",
    "vsteffect": "VST plugin host (Windows/macOS, requires VST DLLs)",
}

# Recommended effects per use case
EFFECT_RECOMMENDATIONS = {
    "vocals": ["eq", "compressor", "reverbsc", "delay"],
    "drums": ["compressor", "eq", "waveshaper"],
    "bass": ["compressor", "eq", "bassbooster"],
    "synth": ["delay", "reverbsc", "flanger", "bitcrush"],
    "master": ["eq", "compressor", "stereoenhancer", "amplifier"],
}


def get_fxchain(parent: ET.Element, create: bool = True) -> ET.Element:
    """Get or create the fxchain element under a track or mixer channel."""
    fxchain = parent.find("fxchain")
    if fxchain is None and create:
        fxchain = ET.SubElement(parent, "fxchain", {
            "numofeffects": "0", "enabled": "0",
        })
    return fxchain


def _number(text: str) -> float | int:
    """A stored number as int when it is whole."""
    value = float(text)
    return int(value) if value.is_integer() else value


def controls_of(effect: ET.Element, create: bool = False) -> ET.Element | None:
    """The controls element of an <effect> (where the knobs are stored).

    None when the effect has none (create=True adds an empty one)."""
    name = effect.get("name", "")
    known = KNOWN_EFFECTS.get(name)
    if known is not None:
        found = effect.find(known[1])
        if found is not None:
            return found
    for child in effect:
        if child.tag != "key":
            return child
    if create and known is not None:
        return ET.SubElement(effect, known[1])
    return None


def add_effect(
    parent: ET.Element,
    effect_name: str,
    wet: float | None = None,
    enabled: bool = True,
    position: int | None = None,
    available: Collection[str] | None = None,
    params: dict | None = None,
    bpm: float = 120.0,
) -> dict:
    """Add an effect to a fxchain element.

    Args:
        parent: The track or mixer channel element containing fxchain.
        effect_name: Built-in LMMS effect plugin name.
        wet: Wet/dry mix (0.0-1.0, 1.0=full effect). None=the effect's
            starting value (EFFECT_START): 1.0, except for effects whose
            output holds only the processed signal (delay, reverbsc,
            flanger), which would otherwise lose the dry sound.
        enabled: Whether the effect is active.
        position: Insert position in chain (None=append at end).
        available: Effect names to list in the "unknown effect" error
            (None=all known effects). Does not restrict what is added;
            callers refuse unavailable effects before calling.
        params: Parameters (friendly names or XML attributes, see
            effect_params.EFFECT_PARAMS) given on top of the starting
            settings; they win over them. "decay_ms" (and "wet", when the
            wet argument is not used) may be given here too.
        bpm: Project tempo, for the settings synced to it (delay time).

    Returns:
        Dict with result info; "params" lists every parameter written.

    Raises:
        ValueError: If effect name is unknown, a parameter or wet is
            refused (ParamError). Nothing is added then.
    """
    normalized = effect_name.strip().lower()
    if normalized not in KNOWN_EFFECTS:
        if normalized in EXTERNAL_EFFECTS:
            raise ValueError(
                f"Effect '{effect_name}' requires external plugins "
                f"({EXTERNAL_EFFECTS[normalized]}). Use a built-in effect instead."
            )
        valid = ", ".join(sorted(KNOWN_EFFECTS if available is None else available))
        raise ValueError(f"Unknown effect '{effect_name}'. Valid effects: {valid}")

    fxchain = get_fxchain(parent)
    existing = [
        e.get("name") for e in fxchain.findall("effect")
    ]
    if normalized in existing:
        raise ValueError(
            f"Effect '{normalized}' already exists on this chain "
            f"(position {existing.index(normalized)})."
        )

    start = params_mod.EFFECT_START.get(normalized, params_mod.Start())
    # "wet" and "decay_ms" may also come in params; the rest is the plugin's
    level, given = params_mod.split_effect_level(normalized, dict(params or {}))
    if wet is not None and "wet" in level:
        raise ParamError("Give wet once: as the wet argument or in params.")
    if wet is not None:
        level["wet"] = params_mod.validate_effect_level(normalized, "wet", wet)
    if given and not params_mod.has_params(normalized):
        raise ParamError(
            f"'{normalized}' has no settable parameters here; it runs with "
            f"the plugin defaults.")
    if params_mod.has_params(normalized):
        given = params_mod.resolve_names(normalized, given)
    notes = []
    if "wet" not in level:
        level["wet"] = start.wet
        if start.why_wet:
            notes.append(
                f"wet defaulted to {start.wet:g} because {start.why_wet}.")
    if "decay_ms" not in level:
        level["decay_ms"] = start.decay_ms
        if start.decay_ms != 1:
            notes.append(
                f"decay_ms defaulted to {start.decay_ms}: LMMS stops an "
                f"effect whose output was silent for that long, and at its "
                f"default of 1 ms that cuts the echoes or tail of a short "
                f"sound.")
    chosen = {**params_mod.start_params(normalized, bpm), **given}

    # Build effect element with standard attributes; plugin-specific
    # controls come from the plugin defaults plus the chosen parameters.
    effect = ET.Element("effect", {
        "on": "1" if enabled else "0",
        "wet": params_mod.format_number(level["wet"]),
        "autoquit": str(level["decay_ms"]),
        "autoquit_denominator": "4",
        "autoquit_numerator": "4",
        # the tempo sync mode of autoquit is read as autoquit_syncmode (a
        # plain "syncmode" is ignored by LMMS 1.2.2 in a current project)
        "autoquit_syncmode": "0",
        "gate": "0",
        "name": normalized,
    })
    controls = ET.SubElement(effect, KNOWN_EFFECTS[normalized][1])
    applied: dict = {}
    if params_mod.has_params(normalized):
        params_mod.ensure_graph_defaults(normalized, controls)
        result = params_mod.apply_params(normalized, controls, chosen)
        applied = result.params
        notes += result.notes
        if not given and start.note:
            notes.append(start.note)

    if position is None or position >= len(fxchain):
        fxchain.append(effect)
        final_pos = len(fxchain) - 1
    else:
        fxchain.insert(position, effect)
        final_pos = max(0, position)

    num = len(fxchain.findall("effect"))
    fxchain.set("numofeffects", str(num))
    fxchain.set("enabled", "1")

    out = {
        "effect": normalized,
        "position": final_pos,
        "wet": level["wet"],
        "decay_ms": level["decay_ms"],
        "enabled": enabled,
        "chain_size": num,
        "params": applied,
        "message": f"Added {normalized} at position {final_pos} "
                   f"(chain now has {num} effects)",
    }
    if notes:
        out["note"] = " ".join(notes)
    return out


def find_effect(parent: ET.Element, identifier: str | int) -> ET.Element:
    """The <effect> element of a chain, by name or position."""
    fxchain = parent.find("fxchain")
    if fxchain is None:
        raise ValueError("This target has no effect chain.")
    effects = fxchain.findall("effect")
    if isinstance(identifier, int) and not isinstance(identifier, bool):
        if 0 <= identifier < len(effects):
            return effects[identifier]
    else:
        for eff in effects:
            if eff.get("name", "").lower() == str(identifier).lower():
                return eff
    names = [e.get("name", "?") for e in effects]
    raise ValueError(
        f"Effect '{identifier}' not found. Existing effects: {names}")


def set_effect_params(
    parent: ET.Element, identifier: str | int, params: dict
) -> dict:
    """Change parameters of an effect already in a chain.

    Args:
        parent: Track or mixer channel element containing fxchain.
        identifier: Effect name (str) or chain position (int).
        params: Friendly names (or XML attributes) -> values; "wet" sets the
            effect's wet/dry mix (0.0-1.0) and "decay_ms" how long LMMS
            keeps it running through silence. Other parameters keep their
            values.

    Raises:
        ValueError: If the effect is not found or anything is refused
            (ParamError); nothing changes then.
    """
    if not params:
        raise ParamError("Pass at least one parameter to set.")
    effect = find_effect(parent, identifier)
    name = effect.get("name", "")
    level, given = params_mod.split_effect_level(name, dict(params))
    applied: dict = {}
    notes: list[str] = []
    if given:
        controls = controls_of(effect, create=True)
        if params_mod.has_params(name):
            params_mod.ensure_graph_defaults(name, controls)
        result = params_mod.apply_params(name, controls, given)
        applied, notes = result.params, result.notes
    for key, attr in (("wet", "wet"), ("decay_ms", "autoquit")):
        if key in level:
            effect.set(attr, params_mod.format_number(level[key]))
    applied = {**level, **applied}
    out = {
        "effect": name,
        "position": _position(parent, effect),
        "params": applied,
        "message": f"Set {', '.join(applied)} on {name}",
    }
    if notes:
        out["note"] = " ".join(notes)
    return out


def _position(parent: ET.Element, effect: ET.Element) -> int:
    return parent.find("fxchain").findall("effect").index(effect)


def remove_effect(parent: ET.Element, identifier: str | int) -> dict:
    """Remove an effect from a fxchain by name or position.

    Args:
        parent: Track or mixer channel element containing fxchain.
        identifier: Effect name (str) or chain position (int).

    Returns:
        Dict with result info.

    Raises:
        ValueError: If effect not found.
    """
    target = find_effect(parent, identifier)
    fxchain = parent.find("fxchain")
    removed_name = target.get("name")
    fxchain.remove(target)
    remaining = fxchain.findall("effect")
    fxchain.set("numofeffects", str(len(remaining)))
    if len(remaining) == 0:
        fxchain.set("enabled", "0")

    return {
        "removed": removed_name,
        "chain_size": len(remaining),
        "message": f"Removed {removed_name} ({len(remaining)} effects remain)",
    }


def list_effects(parent: ET.Element) -> list[dict]:
    """List all effects in a fxchain element.

    Each entry has the position, name, enabled flag, wet mix and decay_ms
    (how long LMMS keeps the effect running through silence), and for
    effects with a parameter table "params": the parameters that differ
    from the plugin defaults (friendly names).
    """
    fxchain = parent.find("fxchain")
    if fxchain is None:
        return []
    result = []
    for pos, eff in enumerate(fxchain.findall("effect")):
        name = eff.get("name", "unknown")
        entry = {
            "position": pos,
            "name": name,
            "enabled": eff.get("on", "1") == "1",
            "wet": float(eff.get("wet", "1")),
            "decay_ms": _number(eff.get("autoquit", "1")),
        }
        controls = controls_of(eff)
        if params_mod.has_params(name) and controls is not None:
            entry["params"] = params_mod.read_params(name, controls)
        result.append(entry)
    return result


def set_effect_enabled(parent: ET.Element, identifier: str | int, enabled: bool) -> dict:
    """Enable/disable an effect in a fxchain by name or position."""
    target = find_effect(parent, identifier)
    target.set("on", "1" if enabled else "0")
    return {
        "effect": target.get("name"),
        "enabled": enabled,
        "message": f"{target.get('name')} {'enabled' if enabled else 'disabled'}",
    }
