# LMMS MCP Server

An MCP (Model Context Protocol) server for [LMMS](https://lmms.io/) - the free, open-source digital audio workstation. Lets AI agents create, modify, and save LMMS music projects programmatically.

## Features

- **Create & save** LMMS projects (`.mmpz` compressed, `.mmp` XML)
- **Add tracks**: Instrument, Sample, Pattern (Beat/Bassline), Automation
- **Add notes** by LMMS key number or note name, with position, velocity, and panning
- **Tuned samples and tracks**: set a track's base note and a fine tune in cents (`set_track_tuning`, or in the same call as `set_audiofile_sample`) so a piano sample recorded at C4 sounds at its note names
- **Effects**: built-in effects on tracks and mixer channels (delay, reverb, EQ, dynamics...)
- **Adapts to your LMMS**: only plugins your installed LMMS version really has are offered or accepted (see [Installed LMMS version](#installed-lmms-version))
- **ZynAddSubFX presets**: Load any of ~950 factory instruments (.xiz), tune parameters
- **Instrument presets**: Load LMMS's factory presets and your own saved ones (.xpf, e.g. a tuned piano sample) into an instrument track with `load_instrument_preset` (see [Instrument presets](#instrument-presets-xpf))
- **Arrangement**: Place patterns, BB clips and audio clips on the song timeline
- **Automation**: Tempo ramps, volume swells, panning curves - linked to real LMMS models
- **Mixer control**: Create channels, set volume, name channels
- **Song settings**: Tempo (BPM), time signature, master volume/pitch
- **Musical utilities**: Note name conversion, scale generation, tick/bar conversion
- **Full project inspection**: Read tracks, patterns, notes, mixer channels

## Installation

```bash
pip install lmms-mcp
```

Or from source:

```bash
git clone https://github.com/TypeWolf/lmms-mcp.git
cd lmms-mcp
pip install -e .
```

### Requirements

- Python 3.10+
- An MCP host (opencode, Claude Desktop, Cursor, etc.)

## Quick Start

### With opencode

Add to your `opencode.json`:

```json
{
  "mcp": {
    "lmms": {
      "type": "local",
      "command": ["python", "-m", "lmms_mcp"],
      "environment": {
        "LMMS_PROJECTS_DIR": "/path/to/your/lmms/projects"
      }
    }
  }
}
```

### With Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lmms": {
      "command": "python",
      "args": ["-m", "lmms_mcp"],
      "env": {
        "LMMS_PROJECTS_DIR": "/path/to/your/lmms/projects"
      }
    }
  }
}
```

### Run directly

```bash
python -m lmms_mcp
```

## Tools

### Project Management

| Tool | Description |
|------|-------------|
| `create_project` | Create a new empty LMMS project |
| `load_project` | Load an existing `.mmpz` or `.mmp` file |
| `save_project` | Save the current project |
| `get_project_info` | Get project overview (tempo, tracks, mixer) |
| `get_project_xml` | Get raw XML of the project |

### Track Operations

| Tool | Description |
|------|-------------|
| `add_instrument_track` | Add a synthesizer/sampler track |
| `add_sample_track` | Add an audio sample track |
| `set_audiofile_sample` | Set the sample (and amp, reverse, loop, start/end) of an `audiofileprocessor` track; optional `base_note` / `pitch_cents` tune it in the same call |
| `set_track_tuning` | Tune an instrument track: `base_note` (the note a sample was recorded at, e.g. `"C4"`) and a `pitch_cents` fine tune; without arguments it returns the current tuning |
| `add_automation_track` | Add a parameter automation track |
| `add_pattern_track` | Add a beat/bassline pattern track |
| `remove_track` | Remove a track by index |
| `get_track` | Get detailed track information (instrument tracks with a base note other than A4 or a fine tune also show `base_note`, `base_note_name`, `pitch_cents`) |
| `list_tracks` | List all tracks with summary (same tuning fields) |
| `set_track_volume` | Set track volume (0-200) |
| `set_track_panning` | Set track panning (-100 to +100) |
| `mute_track` | Mute/unmute a track |
| `solo_track` | Solo/unsolo a track |

### Notes & Patterns

| Tool | Description |
|------|-------------|
| `add_note` | Add a note by LMMS key number (0-108; C4 = 48, A4 = 57) |
| `add_note_by_name` | Add a note by name (e.g. "C4", "A#3", "Bb2"; C0 to C9) |
| `add_notes_batch` | Add multiple notes at once |

### Mixer

| Tool | Description |
|------|-------------|
| `add_mixer_channel` | Create a new mixer channel |
| `get_mixer_channels` | List all mixer channels |
| `set_mixer_channel_volume` | Set channel volume |
| `set_mixer_channel_name` | Rename a channel |

### Song Settings

| Tool | Description |
|------|-------------|
| `set_tempo` | Set BPM (10-999) |
| `set_time_signature` | Set time signature (e.g. 4/4, 3/4) |
| `set_master_volume` | Set master volume (0-200) |
| `set_master_pitch` | Set master pitch (-12 to +12 semitones) |

### Effects (FX Chain)

| Tool | Description |
|------|-------------|
| `add_effect` | Add a built-in effect to a track or mixer channel |
| `remove_effect` | Remove an effect by name or chain position |
| `toggle_effect` | Enable/bypass an effect without removing it |
| `get_effect_chain` | List all effects on a track or mixer channel |

### ZynAddSubFX Presets & Parameters

Only offered when the installed LMMS has the `zynaddsubfx` plugin.

| Tool | Description |
|------|-------------|
| `list_zyn_presets` | Browse ~950 factory presets (.xiz) by category |
| `load_zyn_preset` | Load a preset into a zynaddsubfx track |
| `set_zyn_params` | Set portamento, filter, FM gain, resonance etc. |

### Instrument Presets (.xpf)

| Tool | Description |
|------|-------------|
| `list_instrument_presets` | Overview of the available presets per instrument (count and a few examples), or all presets of one instrument |
| `load_instrument_preset` | Load a preset into an instrument track (or create the track from it with `track_index=null`) |

### Arrangement (Song Editor Timeline)

| Tool | Description |
|------|-------------|
| `place_pattern` | Place an empty pattern clip on an instrument track |
| `place_bb_clip` | Trigger a BB pattern at a given time |
| `place_sample_clip` | Place an audio file clip on a sample track |
| `assign_sample_file` | Assign/replace the audio file on sample clips |
| `move_clip` | Move a clip to a new position |
| `delete_clip` | Delete a clip at a position |
| `get_arrangement` | Full timeline overview of all clips |

### Automation

| Tool | Description |
|------|-------------|
| `add_automation` | Create automation curves for tempo, master volume/pitch, track volume/panning and mixer channel volume |

### LMMS App Integration

| Tool | Description |
|------|-------------|
| `get_lmms_info` | Detect installed LMMS version + available plugins |
| `render_project` | Export to WAV/FLAC/OGG/MP3 via headless LMMS render. `oversampling` (1/2/4/8, default 1 = what LMMS plays in its GUI; higher changes the sound of some instruments such as ZynAddSubFX) and `interpolation` (linear/sincfastest/sincmedium/sincbest, default sincfastest) |

The server reads and writes project files directly - it never launches
the LMMS GUI. The installed LMMS is only used for: preset discovery,
plugin availability checks (see [Installed LMMS version](#installed-lmms-version))
and audio rendering.

### Installed LMMS version

LMMS 1.2.x and 1.3+ ship different plugins, and LMMS silently ignores a
plugin it does not have when loading a project (no error, just no sound).
To keep agents out of that dead end, the server inspects the installed LMMS
once at startup and adapts to it:

- **Not offered**: the server instructions, the `add_instrument_track` /
  `add_effect` tool descriptions, the `lmms://reference/*` resources,
  `list_available_plugins`, the prompts and the recommended instrument/effect
  chains list only plugins your LMMS has. A recommended plugin that is missing
  is replaced by its closest available one (e.g. `compressor` becomes
  `dynamicsprocessor` on LMMS 1.2.x) or dropped.
- **Refused**: asking for a known plugin your LMMS lacks (e.g. `compressor`
  or `slicert` on LMMS 1.2.2) returns an error that names the installed
  version, the reason (e.g. "requires LMMS 1.3+"), an `alternative` and the
  list of available plugins. The project is left unchanged.
- **Tools hidden**: tools that are useless without a plugin are not
  registered when it is missing - the ZynAddSubFX tools without
  `zynaddsubfx`, `set_audiofile_sample` without `audiofileprocessor`, and
  `add_vst_track` / `scan_vst_directory` without the `vestige` VST host.
- **Custom plugins**: instrument plugins that are installed but not in the
  built-in list can still be used by name with `add_instrument_track`
  (importers, exporters and analyzers are not instruments and are refused).
- **No LMMS found**: availability cannot be verified, so every known plugin
  is offered and allowed. The server instructions say so, and plugins that
  need LMMS 1.3+ carry a warning when added. Set `LMMS_EXECUTABLE` to enable
  the checks.
- **Mixer names in the project file**: LMMS 1.2.x reads `<fxmixer>`,
  `<fxchannel>` and the track attribute `fxch`; LMMS 1.3+ reads `<mixer>`,
  `<mixerchannel>` and `mixch`. LMMS silently ignores the names it does not
  know (the track then plays on Master, and the volume and effects of the
  mixer channel do nothing), so a new project is written with the names of
  the installed LMMS. Without a detected LMMS the 1.2 names are used: LMMS
  1.3+ upgrades them when it opens the file, so they load everywhere. A
  loaded project is read in either spelling and every edit keeps the
  spelling the project already has (the two are never mixed). LMMS 1.2.x
  does not route sample tracks to a mixer channel (they always play on
  Master), so `add_sample_track` refuses a `mixer_channel` other than 0
  there.

`list_available_plugins` also shows what is missing and why
(`unavailable_in_installed_lmms`). Restart the server after installing or
removing plugins or upgrading LMMS: the check runs once at startup.

### Custom Plugins & VST

| Tool | Description |
|------|-------------|
| `list_available_plugins` | List the plugins usable with the installed LMMS (incl. custom ones) and what is missing |
| `scan_vst_directory` | Find VST .dll files in a folder (only offered if LMMS has the `vestige` VST host) |
| `add_vst_track` | Add a track hosting a VST plugin (Vestige; only offered if LMMS has the `vestige` VST host) |

Custom LMMS instrument plugins dropped into the plugins folder are
detected automatically and can be used directly by name - no server
update needed.

### Utilities

| Tool | Description |
|------|-------------|
| `note_name_to_key` | Convert note name to LMMS key number (also returns the MIDI note number) |
| `key_to_note_name` | Convert LMMS key number to note name (also returns the MIDI note number) |
| `bars_to_ticks_converter` | Convert bars to ticks |
| `ticks_to_bars_converter` | Convert ticks to bars |
| `generate_scale` | Generate a musical scale |

## Resources

| URI | Description |
|-----|-------------|
| `lmms://project/info` | Current project information |
| `lmms://project/tracks` | All tracks in the project |
| `lmms://project/mixer` | All mixer channels |
| `lmms://project/xml` | Raw project XML |
| `lmms://reference/instruments` | Instruments available in the installed LMMS |
| `lmms://reference/effects` | Effects available in the installed LMMS |
| `lmms://reference/note_names` | LMMS key number to note name mapping (with MIDI note numbers) |
| `lmms://reference/scales` | Available musical scales |

## Prompts

| Name | Description |
|------|-------------|
| `create_basic_song` | Create a song structure with drums, bass, melody |
| `add_drum_pattern` | Generate a drum pattern (four-on-the-floor, breakbeat, etc.) |
| `create_melody` | Generate a melody in a given scale |
| `mix_and_arrange` | Mix and arrange the current project |
| `export_project` | Export/save the project |

## LMMS Concepts

Note numbers are LMMS keys, not MIDI note numbers: middle C (C4) = 48,
A4 = 57 = 440 Hz. A MIDI note number is key + 12. Note names ("C4",
"Bb2") follow the same numbering, so `add_note_by_name("A4")` is key 57,
LMMS's 440 Hz reference pitch. Valid keys are 0-108 (C0 to C9). For the
General MIDI drum kit of `sf2player` (bank 128) this means key 24 is the
kick (MIDI 36), 26 the snare (38), 30 the closed hi-hat (42), 34 the open
hi-hat (46) and 37 the crash (49).

| Concept | Value |
|---------|-------|
| Ticks per bar | 192 (in 4/4 time) |
| Default tempo | 140 BPM |
| Key 48 | C4 (middle C, 261.63 Hz) |
| Key 57 | A4 (440 Hz on every offered instrument; the default base note of an instrument track) |
| Key range | 0-108 (C0 to C9); LMMS clamps higher keys to 108 |
| Volume range | 0-200 (100 = normal) |
| Panning range | -100 (left) to +100 (right) |
| Track type 0 | Instrument |
| Track type 1 | Pattern (Beat/Bassline) |
| Track type 2 | Sample |
| Track type 5 | Automation |

### Pitch: key 57 = A4 = 440 Hz on every instrument

All offered instruments are tuned so that key 57 = A4 = 440 Hz. Some LMMS
plugins sound off the key they are played at, so the server compensates
through the instrument track's base note (LMMS transposes a track by
`57 - basenote` semitones): it writes `basenote = base note - offset`, e.g.
45 instead of 57 for a `zynaddsubfx` track. From the agent's side the base
note stays 57 and it never has to transpose notes by hand. The offsets (measured with
LMMS 1.2.2, rendering A3, A4 and A5 and estimating the fundamental) live in
`PITCH_OFFSET_SEMITONES` in `src/lmms_mcp/pitch.py`:

| Instrument | Without compensation | Fix |
|------------|----------------------|-----|
| `zynaddsubfx` | 12 semitones low (LMMS hands its key to ZynAddSubFX as the MIDI note, A4 = 69) | base note 45 |
| `opulenz` (`OPL2` in LMMS 1.2.x) | fundamental 12 semitones low on the default patch (its modulator runs at half the carrier frequency); other patches differ, see [Instrument presets](#instrument-presets-xpf) | base note 45 (default patch) |
| every other pitched instrument | within 1 % of the key | none |

ZynAddSubFX presets also carry an octave transposition of their own (about
15 % of the factory presets: basses are often one octave down, a few pads
one up). `load_zyn_preset` compensates it as well, so key 57 is A4 with
the preset's main sound (the response reports `preset_octave` when it is
not 0). A few presets still read an octave above the key because their
strongest partial is the second harmonic (measured on e.g. `Guitar/Dist
Guitar 3` and `Collection/Full Strings`); that is part of their timbre and
is not corrected.

Exceptions: `kicker` is a drum and ignores the key; `sfxr` is a
sound-effect synth that follows the key but plays about 5.5 semitones flat
and is silent from key 80; `audiofileprocessor` plays its sample at its
original pitch at the track's base note (key 57 until `set_track_tuning` or
`set_audiofile_sample` says at which note the sample was recorded, see
[Tuning a track](#tuning-a-track-base-note-and-fine-tune)). The default patches of `tripleoscillator` (its
second and third oscillator are one and two octaves down) and `organic` (a
partial one octave down) layer sub-octave partials on purpose: their main
oscillator is on the key. `bitinvader` and `watsyn` start with a random
waveform (two renders of the same project differ), so their timbre changes
from render to render and their pitch cannot be measured reliably.

Some instruments do not play all of keys 0-108 at the right pitch.
`add_instrument_track` reports `playable_keys` for them and `add_note` /
`add_notes_batch` add the note anyway with a `warning`:

| Instrument | Playable keys | Outside |
|------------|---------------|---------|
| `zynaddsubfx` | 0-95 (up to 107 for a preset with +1 octave, down to 83 for -1) | silent (ZynAddSubFX drops MIDI notes from 108 up) |
| `opulenz` | 0-90 | silent |
| `freeboy` | 24-95 | silent |
| `sid` | 2-94 | wrong pitch (16-bit frequency register wraps) |
| `nes` | 0-72 | sharp by up to 1.75 semitones (waveform period is a whole number of samples) |
| `vibedstrings` | 0-78 | too quiet to hear |
| `sfxr` | 0-78 | silent from 80 |
| `sf2player` | depends on the soundfont (the default GM soundfont's piano is silent from about key 100) | silent |

### Tuning a track: base note and fine tune

`set_track_tuning(track_index, base_note, pitch_cents)` sets the two things
LMMS stores on an instrument track to tune it (`basenote` and `pitch`
on `<instrumenttrack>`). `set_audiofile_sample` takes the same
`base_note` and `pitch_cents`, to set a sample and its tuning in one call.

- `base_note` is the note (a name like `"C4"` or an LMMS key 0-108) at
  which the instrument or sample sounds at its natural, recorded pitch.
  A new track's is key 57 (A4). A piano sample recorded at C4 gets
  `base_note="C4"`: key 48 plays it unchanged, other keys transpose it and
  every key sounds at its note name, so no note has to be shifted by hand.
  For synths and soundfonts (already tuned: key 57 = 440 Hz) a changed base
  note only transposes the track: `base_note="C4"` makes key 48 sound as
  A4. The base note you give is the user-facing one: plugins that are
  compensated (see above) keep their compensation, so a `zynaddsubfx` track
  at base note C4 is stored as basenote 36.
- `pitch_cents` is a fine tune for every note (positive raises, negative
  lowers; 100 cents = 1 semitone; -1200 to +1200). `+18` fixes a sample
  that is 18 cents flat. LMMS limits the fine tune to 100 cents times the
  track's `pitchrange`, so the server raises `pitchrange` to the smallest
  whole number of semitones that covers the value (it never lowers it).
  Measured with LMMS 1.2.2 by rendering at `-x 1`: the instruments follow
  the fine tune within a few cents (a `tripleoscillator` sine at +50 cents
  is 452.9 Hz, at -100 cents 415.3 Hz, at +150 cents 479.8 Hz; the same
  holds for `audiofileprocessor`, `monstro`, `lb302`, `organic`, `sid`,
  `sf2player` and others), except the ones in the table below, whose calls
  are refused with a message saying what to do instead.

| Instrument | Base note | Fine tune |
|------------|-----------|-----------|
| `zynaddsubfx` | works (keeps its compensation; the lowest base note is C1 on a default track) | only +-100 cents (LMMS sends it as a pitch bend that ZynAddSubFX always scales to +-100 cents; the pitch range stays 1) |
| `opulenz` (`OPL2` in LMMS 1.2.x) | works | ignored |
| `kicker` | ignores the key, so tuning is refused | refused |
| every other instrument | works | follows it up to +-1200 cents |

The response of `set_track_tuning` has `base_note` and `base_note_name`, the
`written_basenote` stored in the project, `pitch_cents`, `pitchrange`,
`playable_keys` when the instrument cannot play every key with this tuning,
and a `hint` saying what the keys play now. A tuned sample keeps its tuning
when you set another sample on the track; `set_audiofile_sample` says so
in a `note` when it happens. Example (LMMS factory samples, pitch measured
at `-x 1`): `instruments/piano01.ogg` is a C4, `instruments/cello01.ogg` a D2
(fundamental 72.65 Hz) that is about 17-18 cents flat and
`instruments/violin_pizzicato01.ogg` a G4. With the call below, key 45 (A3)
plays the cello at 220.05 Hz and key 57 (A4) at 440.0 Hz. Measure a sample's
pitch with autocorrelation or by checking the harmonic series; the strongest
spectral peak is often a harmonic (as with this cello).

```
set_audiofile_sample(0, "instruments/cello01.ogg", base_note="D2", pitch_cents=18)
add_note_by_name(0, "A3")      # key 45: sounds at 220 Hz
```

## Instrument presets (.xpf)

An `.xpf` file is what LMMS writes for "Save as preset" on an instrument
track: plain XML with the instrument, all its settings, envelopes/filter,
chords/arpeggio, effects and the track's tuning. `list_instrument_presets`
lists the ones available for the instruments your LMMS offers, and
`load_instrument_preset(track_index, "Folder/Name")` loads one, for
example `"TripleOscillator/AmazingBubbles"`, a plain `"Amazing Bubbles"`
(case, spaces, `-` and `_` do not matter) or an absolute path. Pass
`track_index=null` to create the instrument track from the preset in one
call.

Where presets are looked for, in this order: the folders in
`LMMS_PRESET_DIRS`, your LMMS presets folder (`presets/` in the LMMS
working directory from `~/.lmmsrc.xml`, where "Save as preset" puts them)
and the factory presets (`/usr/share/lmms/presets` on Linux,
`data/presets` on Windows). Every `.xpf` below those folders counts, so a
folder of your own tuned sample presets works too.

What loading into an existing track replaces and keeps:

| Replaced by the preset | Kept on the track |
|------------------------|-------------------|
| the instrument and its settings (the track changes instrument if the preset uses another one) | notes and patterns |
| envelopes and filter, chords, arpeggio | track name (`rename=True` or `name` sets it; a new track gets the preset's) |
| the effect chain, only when the preset has effects | volume, panning (the response notes it when the preset was saved with other values), mute, solo |
| the tuning: base note, fine tune and pitch range, stored as the preset's author heard them | mixer channel, MIDI port |

A call is refused, and nothing changes, when the preset needs an instrument
your LMMS does not have, when a sample or soundfont it plays is missing
(the track would be silent), or when the track is not an instrument track.
LMMS finds a relative sample path in your LMMS samples folder or its
factory samples, never next to the preset or project (checked with LMMS
1.2.2: such a sample renders silent): a sample lying beside the
preset is set by its absolute path, and the response says so. Effects whose
plugin is not installed are left out with a warning. Presets in LMMS's
pre-1.0 channel format cannot be read.

The tuning comes back in the response (`tuning` with a `hint`). A preset of
a tuned sample keeps the base note it was saved with: key 48 plays a
sample saved with base note 48 at its original pitch. Instruments LMMS
plays off pitch (see [Pitch](#pitch-key-57--a4--440-hz-on-every-instrument) above) are compensated when the
preset has the default base note 57, so key 57 is A4 = 440 Hz as on a new
track. The OPL2 pitch depends on the patch: the fundamental is the greatest
common divisor of the two operators' frequency multipliers (a quiet x0.5
operator, op1 level below 50, does not count). Measured with LMMS 1.2.2
at `-x 1` on all 15 factory OpulenZ presets: 13 sound at the key (key 57 =
441 Hz) as stored, `Funky` and `Combo_organ` an octave low and are
compensated (base note 45), and `Bubbly_days` has a ×0.5 modulator that is
too quiet to be heard as the pitch.

## Available Instruments

All built-in LMMS instruments (verified against LMMS source). LMMS has no
plugin download mechanism - only these can be used. The server offers only
the ones your installed LMMS has; plugins marked **LMMS 1.3+** do not exist
in LMMS 1.2.x and are refused there:

| Plugin ID | Name |
|-----------|------|
| `tripleoscillator` | Three-oscillator subtractive synth (default) |
| `kicker` | Kick drum synth |
| `audiofileprocessor` | Audio file player/sampler (set its sample with `set_audiofile_sample`) |
| `organic` | Additive organ synth |
| `malletsstk` | Physical modeling mallets (STK) |
| `lb302` | TB-303 style acid bass |
| `monstro` | Powerful 3-oscillator polyphonic synth |
| `freeboy` | Game Boy sound chip emulator |
| `nes` | NES 8-bit sound chip emulator |
| `sid` | Commodore 64 SID chip emulator |
| `sfxr` | Retro sound effect generator |
| `opulenz` | OPL3 FM synthesizer |
| `watsyn` | 4-oscillator wavetable-style synth |
| `xpressive` | Expressive mono lead synth (**LMMS 1.3+**) |
| `zynaddsubfx` | ZynAddSubFX powerful feature-rich synth |
| `sf2player` | SoundFont (.sf2) sample player |
| `vibedstrings` | Vibrating string physical model |
| `bitinvader` | Bit-crushed wavetable synth |
| `patman` | GUS patch sampler |
| `gigplayer` | GIG sample library player |
| `slicert` | Beat slicer for audio loops (**LMMS 1.3+**) |
| `vestige` | VST plugin host (only if your LMMS was built with VST support; use `add_vst_track`) |

## Available Effects

Built-in LMMS effects for `add_effect`: `amplifier`, `bassbooster`,
`bitcrush`, `compressor` (**LMMS 1.3+**), `crossovereq`, `delay`,
`dispersion` (**LMMS 1.3+**), `dualfilter`, `dynamicsprocessor`, `eq`,
`flanger`, `frequencyshifter` (**LMMS 1.3+**), `multitapecho`, `reverbsc`,
`slewdistortion` (**LMMS 1.3+**), `stereoenhancer`, `stereomatrix`,
`waveshaper`. As with instruments, only the ones your installed LMMS has
are offered; on LMMS 1.2.x use `dynamicsprocessor` where `compressor` is
listed below.

Typical chains:
- Lead synth: `delay` -> `reverbsc`
- Vocals: `eq` -> `compressor` -> `reverbsc`
- Master bus: `eq` -> `compressor` -> `stereoenhancer`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LMMS_PROJECTS_DIR` | `~/Desktop/Media/lmms/AI-Projects` | Default directory for saving projects |
| `LMMS_PRESETS_DIR` | auto-detected | Path to the ZynAddSubFX presets folder (`data/presets/ZynAddSubFX`) or its parent presets folder |
| `LMMS_PRESET_DIRS` | none | Extra folders (separated by `:` on Linux, `;` on Windows) with instrument presets (`.xpf`), searched before your LMMS presets folder and the factory presets |
| `LMMS_EXECUTABLE` | auto-detected | Path to the LMMS executable (`lmms.exe` / `lmms`) (for version/plugin checks and rendering) |
| `LMMS_PLUGINS_DIR` | auto-detected | LMMS plugins folder (overrides detection; `*.dll` on Windows, `lib<name>.so` on Linux) |

**Linux:** `lmms` is found on `PATH` (or `/usr/bin`, `/usr/local/bin`), and plugins are auto-detected in `<prefix>/lib/lmms`, `/usr/lib/<arch>/lmms`, `/usr/lib/lmms`, `/usr/lib64/lmms` or `/usr/local/lib/lmms`. ZynAddSubFX presets are auto-detected in `<prefix>/share/lmms/presets/ZynAddSubFX`, `/usr/share/lmms/presets/ZynAddSubFX`, `/usr/local/share/lmms/presets/ZynAddSubFX` or `~/lmms/presets/ZynAddSubFX`. Set the variables above only if your install is elsewhere.

## Configuration

### opencode.json

```json
{
  "mcp": {
    "lmms": {
      "type": "local",
      "command": ["python", "-m", "lmms_mcp"],
      "cwd": ".",
      "enabled": true,
      "environment": {
        "LMMS_PROJECTS_DIR": "C:\\Users\\you\\Music\\LMMS\\Projects"
      }
    }
  }
}
```

### claude_desktop_config.json

```json
{
  "mcpServers": {
    "lmms": {
      "command": "python",
      "args": ["-m", "lmms_mcp"],
      "env": {
        "LMMS_PROJECTS_DIR": "/home/you/music/lmms/projects"
      }
    }
  }
}
```

## Development

```bash
# Clone and install
git clone https://github.com/TypeWolf/lmms-mcp.git
cd lmms-mcp
pip install -e ".[dev]"

# Run tests
pytest

# Run in development mode
mcp dev src/lmms_mcp/server.py
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Links

- [LMMS](https://lmms.io/) - The DAW this server controls
- [MCP Protocol](https://modelcontextprotocol.io/) - Model Context Protocol specification
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Official Python SDK
- [opencode](https://opencode.ai/) - AI coding assistant with MCP support
