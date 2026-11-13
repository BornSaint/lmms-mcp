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
| `load_project` | Load an existing `.mmpz` or `.mmp` file. A relative path is relative to the default projects folder (`LMMS_PROJECTS_DIR`), like in `save_project`, so the name you saved with loads that file again; if it is not there but exists in the server's working folder, that file is loaded and the response has a `note` (if it exists in both, the default folder's file is loaded and the `note` names the other). The response has the absolute `path` that was loaded |
| `save_project` | Save the current project. A relative path is relative to the default projects folder (`LMMS_PROJECTS_DIR`). Missing folders in the path are created (listed as `created_dirs`); if the file cannot be written the response is a JSON error with the reason |
| `get_project_info` | Get project overview (tempo, tracks, mixer) and an estimate of the mix peak before any render: `estimated_mix_peak_dbfs` with its `estimate_range_dbfs`, the `loudest_bar` and the `unknown_level_tracks` (see [Loudness before the first render](#loudness-before-the-first-render)) |
| `get_project_xml` | Get raw XML of the project |

### Track Operations

| Tool | Description |
|------|-------------|
| `add_instrument_track` | Add a synthesizer/sampler track; the response has `level_at_vol_100`, how loud the instrument is, and, when you pass no `volume`, the `volume` the server chose from it with `volume_chosen` saying why (see [Loudness before the first render](#loudness-before-the-first-render)) |
| `add_sample_track` | Add an audio sample track |
| `set_audiofile_sample` | Set the sample (and amp, reverse, loop, start/end) of an `audiofileprocessor` track; optional `base_note` / `pitch_cents` tune it in the same call |
| `set_track_tuning` | Tune an instrument track: `base_note` (the note a sample was recorded at, e.g. `"C4"`) and a `pitch_cents` fine tune; without arguments it returns the current tuning |
| `add_automation_track` | Add a parameter automation track |
| `add_pattern_track` | Add a beat/bassline pattern track |
| `remove_track` | Remove a track by index |
| `get_track` | Get detailed track information (instrument tracks with a base note other than A4 or a fine tune also show `base_note`, `base_note_name`, `pitch_cents`) |
| `list_tracks` | List all tracks with summary (same tuning fields) |
| `set_track_volume` | Set track volume (0-200); the server no longer chooses that track's volume afterwards |
| `set_track_panning` | Set track panning (-100 to +100) |
| `set_track_mixer_channel` | Route an existing track to another mixer channel (the channel must exist) |
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
| `set_mixer_channel_volume` | Set channel volume (the static value, if the volume is automated) |
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
| `add_effect` | Add a built-in effect to a track or mixer channel. Without `wet` and `params` it starts with audible, musical settings (a delay synced to the tempo that keeps the dry sound), see [Available Effects](#available-effects) |
| `set_effect_params` | Set parameters of an effect already in a chain (delay time, feedback, reverb size, EQ bands, bit depth...), validated against each parameter's range |
| `remove_effect` | Remove an effect by name or chain position |
| `toggle_effect` | Enable/bypass an effect without removing it |
| `get_effect_chain` | List all effects on a track or mixer channel with their wet mix and the parameters that differ from the plugin defaults |

### ZynAddSubFX Presets & Parameters

Only offered when the installed LMMS has the `zynaddsubfx` plugin.

| Tool | Description |
|------|-------------|
| `list_zyn_presets` | Browse ~950 factory presets (.xiz) by category |
| `load_zyn_preset` | Load a preset into a zynaddsubfx track (the response has `level_at_vol_100`) |
| `set_zyn_params` | Set portamento, filter, FM gain, resonance etc. |

### Instrument Presets (.xpf)

| Tool | Description |
|------|-------------|
| `list_instrument_presets` | Overview of the available presets per instrument (count and a few examples), or all presets of one instrument |
| `load_instrument_preset` | Load a preset into an instrument track (or create the track from it with `track_index=null`); the response has `level_at_vol_100` |

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
| `render_project` | Export to WAV/FLAC/OGG/MP3 via headless LMMS render. `oversampling` (1/2/4/8, default 1 = what LMMS plays in its GUI; higher changes the sound of some instruments such as ZynAddSubFX) and `interpolation` (linear/sincfastest/sincmedium/sincbest, default sincfastest). A relative `output_path` is relative to the default projects folder (`LMMS_PROJECTS_DIR`). Missing folders of `output_path` are created (listed as `created_dirs`); a path LMMS cannot write gives an error naming the path. The response measures the render (`levels`: peak, clipped samples, RMS, loudness per 4 bars) and, with `auto_fix_clipping` (default true), lowers the Master mixer volume of a clipping mix to a -1 dBFS peak and saves the project (`auto_gain`, `levels_after`) |
| `analyze_mix` | Per-track balance of the mix in one call, instead of scripting over a render: renders every unmuted instrument/sample track and the mix as float (one LMMS render per track plus one, about 9 s each for a 2-minute song; no tokens) and reports, per track, peak, RMS where it plays, active bar ranges and per-section RMS/peak/share of the energy, plus the mix per section. `sections` (`[{"name": "hook", "bars": [12, 20]}]`, bars 0-based, end exclusive; default blocks of 4 bars) and `oversampling` (default 1). `flags` with ready-to-use suggestions: `dominant` (more than 6 dB above the other tracks), `buried` (more than 15 dB below the mix while it plays), `peak_contributors` (the tracks that peak with a mix peak at or above -0.5 dBFS: a mix `render_project` just fixed to -1 dBFS is not flagged); e.g. `set_track_volume(3, 197): +5.9 dB`, with the limit of volume 200 said when it is not enough. Does not change the project |

The server reads and writes project files directly - it never launches
the LMMS GUI. The installed LMMS is only used for: preset discovery,
plugin availability checks (see [Installed LMMS version](#installed-lmms-version))
and audio rendering.

`render_project` runs LMMS with its own temporary config instead of your
`~/.lmmsrc.xml` (every LMMS process rewrites the config it started with), so
renders never change your LMMS settings, cannot clash with an open LMMS or
another render, and do not depend on your GUI options. It takes over only your
sample, soundfont and plugin folders (so files resolve as they do in LMMS) and
keeps effects running through silence, so echoes and reverb tails are not cut
off. The temporary config is deleted after the render. Only one LMMS process of
the server runs at a time, so renders asked for together run one after the
other.

LMMS does not limit its output: a mix whose peak is above 0 dBFS is clipped
in the file (LMMS's default instrument and kicker levels often are). So
`render_project` first renders the project as a lossless 32-bit float file
(into a temporary folder, removed afterwards) and measures it in pure Python
(no numpy needed): `levels` has the peak and RMS in dBFS, the number of
samples above 0 dBFS, and the peak/RMS of every 4 bars (8, 16, ... for long
songs) with the loudest and quietest of them. A clipping mix is fixed by
lowering the Master mixer volume (and its automation, if any) to a peak of
-1 dBFS; the project is saved with the new volume, so later renders are clean
too (`auto_gain` says what changed, `auto_fix_clipping=false` turns it off).
A `wav` is then written from the float render with that gain, without
another LMMS render; flac, ogg and mp3 (and a clipping mix that is not
fixed) are rendered by LMMS once more, so they take about twice as long. A
render whose peak is below -12 dBFS is not changed, but a `note` says what
gain would bring it to -1 dBFS. Measuring and writing a 3-minute stereo song
takes a few seconds on top of the LMMS render.

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
  Master), so `add_sample_track` and `set_track_mixer_channel` refuse a
  `mixer_channel` other than 0 there.
- **Mixer routing**: a mixer channel must exist before a track is routed to
  it (`add_mixer_channel`); LMMS plays a track on a missing channel on
  another one, so the tools refuse that. `load_project` under LMMS 1.2
  converts a project that uses the 1.3 spelling (an older version of this
  server wrote it for 1.2, where its mixer and routing did nothing) and
  reports it as `converted_mixer`. A loaded track that is routed to a
  missing channel is listed in `warnings`; `set_track_mixer_channel(track,
  channel)` fixes it without removing the track.
- **Automated values**: a parameter with an automation curve
  (`add_automation`) is stored as a child element, which LMMS reads instead
  of the plain attribute. `set_track_volume`, `set_track_panning`,
  `set_mixer_channel_volume`, `mute_track`, `solo_track`, `set_tempo` and
  the master settings change that stored value and keep the curve, and the
  response then has `automated: true` and a `note`. The value is the static
  one: it applies from the start of the song until the curve begins (a
  curve starts at its first point), then the curve overrides it wherever it
  has points, and after the last point LMMS keeps the last curve value. To
  change the curve itself, `remove_track` its automation track and call
  `add_automation` again.

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
| the effect chain, only when the preset has effects | volume (unless the server chose it: see [Loudness before the first render](#loudness-before-the-first-render)), panning (the response notes it when the preset was saved with other values), mute, solo |
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

## Loudness before the first render

Instruments differ a lot in loudness: at track volume 100 a plain
`tripleoscillator` track peaks at -2.7 dBFS (and a chord of it at +8.8, LMMS
does not limit its output), a plain `zynaddsubfx` one at -28 dBFS. An agent
that does not know this gets a clipping or buried mix and has to render,
measure and fix it. So the tools that set a sound (`add_instrument_track`,
`load_zyn_preset`, `load_instrument_preset`, `set_audiofile_sample`,
`set_instrument_file`) report how loud it is, at track volume 100:

```json
"level_at_vol_100": {"note_peak_dbfs": -2.7, "chord_peak_dbfs": 8.8,
                     "rms_dbfs": -7.9, "source": "seed"}
```

`note_peak_dbfs` is the peak of one note (key 57, velocity 100, one beat),
`chord_peak_dbfs` that of a four-note chord (keys 48, 52, 55, 60) and
`rms_dbfs` the RMS while the note is held; all in dBFS of a 32-bit float
render with no effects and Master at 100. The peaks are the loudest of three
tries at different places of a song (some presets change by a few dB with the
song position, ZynAddSubFX starts its notes at random phases). Track volume
V changes a level by 20 * log10(V / 100) dB.

`source` says where the level comes from: `seed` (measured beforehand with LMMS
1.2.2 for every instrument at its default settings, `levels.SEED_LEVELS`: a
plain `add_instrument_track` never renders), `cache` (measured before,
here or in an earlier session) or `measured` (rendered now, about 1 s, the
first time a preset, sample or soundfont patch is used). What identifies a
sound is its instrument and settings, preset data, sample / soundfont file
(size and modification time), bank and patch, envelope, chord and arpeggio
settings and the LMMS version; the cache is
`${XDG_CACHE_HOME:-~/.cache}/lmms-mcp/levels.json` and a damaged file is
rebuilt. Nothing is written into the project. If a level cannot be measured
(no LMMS, a timeout) the field is left out and a `note` says so; the tool does
not fail. `LMMS_MCP_MEASURE_LEVELS=0` forbids the measuring renders (seed and
cache are still used). `bitinvader` and `watsyn` tracks without settings
(what `add_instrument_track` writes) play garbage in LMMS 1.2.2, so they have
no level.

`get_project_info` adds the levels up for the project, bar by bar: each track
with notes in the bar counts with its single-note peak (its chord peak when 2
or more notes sound together) plus its track volume, the velocity of its
loudest note, its mixer channel's volume and sends, and the song and Master
volumes. The tracks are added two ways: as power (peaks that never meet) and as
amplitude (every peak at the same instant). `estimate_range_dbfs` is
`[power, amplitude]` for the loudest bar and `estimated_mix_peak_dbfs` its
midpoint in dB. The real peak is in between, nearer the middle than the power
sum, because tracks that play on the same beat (kick and hat) add almost as
amplitude: on 4 test mixes of 6 tracks the midpoint was within +2.1 / -1.1 dB
of the rendered peak, while the power sum alone read up to 4.3 dB too low. The response also has the
`loudest_bar` (0-based) with the peak of each of its tracks, and the
`unknown_level_tracks` (sample tracks and sounds whose level is not known) that
are left out. Effects and automation are not included: it is an estimate, and
`render_project` measures the real level.

### Track volumes you do not set

A track added without `volume` (`add_instrument_track`, or created by
`load_instrument_preset` with `track_index=null`) gets a volume chosen from
its level, so that a mix of default volumes lands near 0 dBFS instead of 7 to
12 dB over, which is what volume 100 on every track gives for 6 to 8 tracks
(this replaces the plain 100 the tracks had before: a behaviour change). The rule
(`levels.auto_volume`): the volume that puts the sound's `rms_dbfs` at
-30 dBFS, but not louder than one note peaking at -5 dBFS, within 10 to 200.
`tripleoscillator`, `monstro` and `kicker` come out at the minimum 10 (still
2, 8 and 0.6 dB above the target); the drum samples tried at 13 (a kick) to
104 (a quiet hat); `lb302` 18, `opulenz` 93, `zynaddsubfx` 126; the default
soundfont piano needs more than the maximum 200 (4 dB short). The response
says so:

```json
"volume": 18,
"volume_chosen": "auto: RMS target -30 dBFS (level_at_vol_100 rms -15 dB -> 18)"
```

and, when 200 is not enough, by how much the sound stays under the target
(raise the note velocities, up to 200, or lower the other tracks).

- A track whose sound is not known yet (an `audiofileprocessor` without a
  sample, a level that could not be measured) has volume 100; when
  `set_audiofile_sample`, `set_instrument_file`, `load_zyn_preset` or
  `load_instrument_preset` then gives it a sound, the volume is chosen
  again and the response has the new `volume` and `volume_chosen` (with
  "(was N)").
- A volume you pass to `add_instrument_track`, or set with
  `set_track_volume`, is never changed (`volume_chosen` is `"explicit"`
  on the first); so is the volume of a track whose volume is automated.
  Sample tracks, VST tracks and pattern tracks keep 100 (their level is not
  known).
- The state ("chosen by the server" or not) lives in the server's memory
  only, never in the project file: a loaded project keeps every volume as it
  is.

Why -30 dBFS, and not the -20 dBFS a track has in a finished mix: the RMS of a
held note is far below the RMS of a track that plays it (eighth notes and
3-note chords are 3 to 8 dB louder than one note), and the peaks of N tracks
that sound together add up (about +11.7 dB for 6 equal ones). The numbers come
from rendering 7 test songs of 3 to 10 tracks (13 drum samples and 16 synth,
soundfont and ZynAddSubFX sounds, 16 bars) one track at a time and mixing them
again with each candidate rule; the mix peak and the `analyze_mix` flags were
read each time (the table is in the comment of `levels.AUTO_RMS_TARGET_DBFS`).
At -20 dBFS the mixes of 6 to 8 tracks peaked at +7 to +12 dBFS; at -30 dBFS
with the -5 dBFS peak limit they peak at -1 to +3.2 dBFS (+4.8 for 10 tracks)
with one borderline `buried` flag in 7 songs. The peak limit matters for
samples: a hat sample is -29 dB RMS at a 0 dBFS peak, so the RMS target alone
leaves its peak at -1.3 dBFS, and the kick, snare, hat and clap that play
together then set the mix peak (without the limit the 7 songs peaked up to 4
dB higher and had 6 flags).

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

### What `add_effect` does by default

The bare plugin defaults of several LMMS effects do nothing audible, or do
the wrong thing: LMMS's delay at its defaults plays *only* an echo half a
second late (the original note disappears and there is no repeat), and a
bitcrusher at its defaults is practically transparent. So a plain
`add_effect("track", 0, "delay")` starts the effect with the settings below
and the response lists the `params` used, the `wet` mix and a `note` where the
effect needs your input. Pass `wet` and/or `params` to choose yourself; the
parameters you give win over the starting ones.

Three effects (`delay`, `reverbsc`, `flanger`) output **only** the processed
signal (verified by rendering with LMMS 1.2.2): at `wet=1.0` the original sound
is gone, so they start below 1.0. Every other effect starts at `wet=1.0`
(they process the sound itself, or mix the original in on their own, like the
multi-tap echo's `dry_gain_db`). `wet` goes from 0.0 (original sound only) to
1.0 (effect only); `set_effect_params` accepts `"wet"` and `"decay_ms"` (and `add_effect` "params" accepts `"decay_ms"`).

Four effects with echoes or tails (`delay`, `reverbsc`, `multitapecho`,
`flanger`) also start with `decay_ms` 8000 (LMMS's "Decay", 1 to 8000). LMMS
stops an effect once its output has been silent for that long: at its default
of 1 ms the delay's echo is lost whenever there is a gap before it (a short
note, or `time_s` longer than the note) and short reverb tails are cut. This
depends on the LMMS setting "disable auto-quit" (off by default), so the
server does not rely on it. `decay_ms` can be changed with `set_effect_params`
like `wet`.

| Effect | `wet` | Starting settings |
|--------|-------|-------------------|
| `delay` | 0.4 | `time_s` = a 1/8 note at the project tempo (`30 / bpm` s), `feedback` 0.3 (repeats fading to 30 % each), `decay_ms` 8000 |
| `reverbsc` | 0.3 | plugin defaults (`size` 0.89), `decay_ms` 8000 |
| `flanger` | 0.5 | `delay_s` 0.002, `lfo_period_s` 3, `lfo_amount_s` 0.0015, `feedback` 0.3, `decay_ms` 8000 |
| `multitapecho` | 1.0 | `steps` 8, `step_length_ms` = a 1/16 note at the project tempo, `tap_levels_db` `"decay"` (-9 dB, falling 4 dB per tap), `decay_ms` 8000 |
| `bitcrush` | 1.0 | `sample_rate_hz` 8000, `bit_levels` 16 |
| `dualfilter` | 1.0 | both filters low-pass at 1200 Hz |
| `dynamicsprocessor` | 1.0 | `threshold_db` -12, `ratio` 4 |
| `waveshaper` | 1.0 | `shape` `soft_clip`, `input_gain` 1.5, `output_gain` 0.6 |
| `stereoenhancer` | 1.0 | `width` 60 |
| `bassbooster` | 1.0 | plugin defaults (audible) |
| `amplifier`, `crossovereq`, `eq`, `stereomatrix` | 1.0 | neutral at the plugin defaults; the response says so and which parameters to set |

### Setting effect parameters

```
add_effect("track", 0, "delay", params={"time_s": 0.375, "feedback": 0.5})
set_effect_params("track", 0, "delay", {"feedback": 0.2, "wet": 0.3, "decay_ms": 4000})
set_effect_params("mixer", 2, "eq", {"lp_on": True, "lp_freq_hz": 8000, "lp_slope_db": 24})
get_effect_chain("track", 0)   # shows the parameters that are not at the plugin default
```

Every parameter has a friendly name (the tables below); the real LMMS
attribute name works too. Names and values are checked first: an unknown name,
a wrong type or a value outside the range changes nothing and the error lists
the valid names or the valid range (values are never silently clamped).
`lmms://reference/effects` returns the same tables as JSON. Things worth
knowing:

- Delay and flanger `lfo_period_s` is the *period* of the wobble in seconds,
  although LMMS calls the attribute `LfoFrequency`; the delay's `time_s` is in
  seconds although the attribute is `DelayTimeSamples`.
- `crossovereq` `bandN_on` is LMMS's `muteN` switch: on = the band is heard.
  The three crossover frequencies must ascend.
- `eq` slopes: `hp_slope_db` / `lp_slope_db` take 12, 24 or 48; the server
  writes the three flags LMMS actually reads.
- Curves: `waveshaper` `shape` is a name or a list of 200 numbers (0 to 1);
  `dynamicsprocessor` builds its curve from `threshold_db` and `ratio`
  (set both together; ratio 1 or threshold 0 dB reduces nothing);
  `multitapecho` `tap_levels_db` / `tap_cutoff_hz` take a number (all taps) or
  a list with one number per step (`tap_levels_db` also `"decay"`).
  LMMS reads these curves unconditionally, so the server always writes them;
  an effect created by an older version of this server (no curve) is repaired
  by the next `set_effect_params`.
- `multitapecho`'s "low-pass stages" knob is not offered: LMMS 1.2 ignores it
  (every stage re-filters the unfiltered input).
- `compressor`, `dispersion`, `frequencyshifter` and `slewdistortion`
  (LMMS 1.3+) have no parameter tables yet: they are added with the plugin
  defaults, and `params` for them is refused.


### Parameter tables

One table per effect (click to open): friendly name, the LMMS attribute, range,
unit, plugin default and meaning. The heading shows what `add_effect` starts
the effect with.

<details>
<summary><code>amplifier</code> &mdash; starts with: neutral at the plugin defaults: set parameters</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `volume` | `volume` | 0 to 200 | % | 100 | Overall gain; 100 = unchanged |
| `pan` | `pan` | -100 to 100 |  | 0 | Balance: -100 = left only, 0 = centre, 100 = right only |
| `left_gain` | `left` | 0 to 200 | % | 100 | Left channel gain |
| `right_gain` | `right` | 0 to 200 | % | 100 | Right channel gain |

</details>

<details>
<summary><code>bassbooster</code> &mdash; starts with: plugin defaults</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `freq_hz` | `freq` | 50 to 200 | Hz | 100 | Frequency below which bass is boosted |
| `gain` | `gain` | 0.1 to 5 |  | 1 | Input gain of the booster (more = more bass) |
| `ratio` | `ratio` | 0.1 to 10 |  | 2 | Strength of the boost curve (more = more bass) |

</details>

<details>
<summary><code>bitcrush</code> &mdash; starts with: sample_rate_hz 8000, bit_levels 16</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `input_gain_db` | `ingain` | -20 to 20 | dB | 0 | Gain before crushing (drives the bit reduction) |
| `input_noise_pct` | `innoise` | 0 to 100 | % | 0 | Noise added before crushing |
| `output_gain_db` | `outgain` | -20 to 20 | dB | 0 | Gain after crushing |
| `output_clip_db` | `outclip` | -20 to 20 | dB | 0 | Hard-clip level of the output |
| `sample_rate_hz` | `rate` | 20 to 44100 | Hz | 44100 | Sample-rate reduction: lower = more aliasing, 44100 = off |
| `stereo_diff_pct` | `stereodiff` | 0 to 50 | % | 0 | Different rate on left and right |
| `bit_levels` | `levels` | 1 to 256 |  | 256 | Quantisation levels: 256 is clean, 16 is 4-bit, 2 is harsh |
| `rate_on` | `rateon` | on/off |  | on | Sample-rate reduction on |
| `depth_on` | `depthon` | on/off |  | on | Bit-depth reduction on |

</details>

<details>
<summary><code>crossovereq</code> &mdash; starts with: neutral at the plugin defaults: set parameters</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `xover_1_2_hz` | `xover12` | 50 to 10000 | Hz | 125 | Crossover between bands 1 and 2 |
| `xover_2_3_hz` | `xover23` | 50 to 20000 | Hz | 1250 | Crossover between bands 2 and 3 |
| `xover_3_4_hz` | `xover34` | 50 to 20000 | Hz | 5000 | Crossover between bands 3 and 4 |
| `band1_gain_db` | `gain1` | -60 to 30 | dB | 0 | Band 1 (lows) gain |
| `band2_gain_db` | `gain2` | -60 to 30 | dB | 0 | Band 2 gain |
| `band3_gain_db` | `gain3` | -60 to 30 | dB | 0 | Band 3 gain |
| `band4_gain_db` | `gain4` | -60 to 30 | dB | 0 | Band 4 (highs) gain |
| `band1_on` | `mute1` | on/off |  | on | Band 1 passes (False silences it; LMMS names the switch 'mute') |
| `band2_on` | `mute2` | on/off |  | on | Band 2 passes |
| `band3_on` | `mute3` | on/off |  | on | Band 3 passes |
| `band4_on` | `mute4` | on/off |  | on | Band 4 passes |

</details>

<details>
<summary><code>delay</code> &mdash; starts with: wet 0.4, feedback 0.3, time_s = a 1/8 note at the project tempo (30/bpm s)</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `time_s` | `DelayTimeSamples` | 0.01 to 5 | s | 0.5 | Delay time (the XML name says samples, the value is seconds) |
| `feedback` | `FeebackAmount` | 0 to 1 |  | 0 | Share of the echo fed back in: 0 = one echo, 0.5 = repeats halving |
| `lfo_period_s` | `LfoFrequency` | 0.01 to 5 | s | 2 | Period of the delay-time wobble (the XML name says frequency) |
| `lfo_amount_s` | `LfoAmount` | 0 to 0.5 | s | 0 | Depth of the delay-time wobble; 0 = steady |
| `out_gain_db` | `OutGain` | -60 to 20 | dB | 0 | Gain of the echoes |

</details>

<details>
<summary><code>dualfilter</code> &mdash; starts with: cutoff1_hz 1200, cutoff2_hz 1200</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `filter1_on` | `enabled1` | on/off |  | on | Filter 1 on (an off filter outputs silence into the mix) |
| `filter1_type` | `filter1` | 22 choices (below) |  | lowpass | Filter 1 type |
| `cutoff1_hz` | `cut1` | 1 to 20000 | Hz | 7000 | Filter 1 cutoff |
| `resonance1` | `res1` | 0.01 to 10 |  | 0.5 | Filter 1 resonance (Q) |
| `gain1_pct` | `gain1` | 0 to 200 | % | 100 | Filter 1 output gain |
| `mix` | `mix` | -1 to 1 |  | 0 | Blend of the two filters in parallel: -1 = filter 1 only, 0 = half of each, 1 = filter 2 only |
| `filter2_on` | `enabled2` | on/off |  | on | Filter 2 on (an off filter outputs silence into the mix) |
| `filter2_type` | `filter2` | 22 choices (below) |  | lowpass | Filter 2 type |
| `cutoff2_hz` | `cut2` | 1 to 20000 | Hz | 7000 | Filter 2 cutoff |
| `resonance2` | `res2` | 0.01 to 10 |  | 0.5 | Filter 2 resonance (Q) |
| `gain2_pct` | `gain2` | 0 to 200 | % | 100 | Filter 2 output gain |

Filter types: `lowpass`, `highpass`, `bandpass_csg`, `bandpass_czpg`, `notch`, `allpass`, `moog`, `2x_lowpass`, `rc_lowpass_12`, `rc_bandpass_12`, `rc_highpass_12`, `rc_lowpass_24`, `rc_bandpass_24`, `rc_highpass_24`, `vocal_formant`, `2x_moog`, `sv_lowpass`, `sv_bandpass`, `sv_highpass`, `sv_notch`, `fast_formant`, `tripole`.

</details>

<details>
<summary><code>dynamicsprocessor</code> &mdash; starts with: threshold_db -12, ratio 4</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `input_gain` | `inputGain` | 0 to 5 |  | 1 | Gain before the curve (1 = unchanged) |
| `output_gain` | `outputGain` | 0 to 5 |  | 1 | Gain after the curve (make-up gain) |
| `attack_ms` | `attack` | 1 to 500 | ms | 10 | How fast the level follows a rise |
| `release_ms` | `release` | 1 to 500 | ms | 100 | How fast the level follows a fall |
| `stereo_mode` | `stereoMode` | maximum / average / unlinked |  | maximum | How the two channels share one level: the louder one, the mean, or each alone |
| `threshold_db` | `waveShape (curve)` | -60 to 0 | dB | 0 | RMS level above which the level is reduced (0 = never) |
| `ratio` | `waveShape (curve)` | 1 to 100 |  | 1 | Reduction above the threshold, N:1 (1 = off, 100 = limiter) |

</details>

<details>
<summary><code>eq</code> &mdash; starts with: neutral at the plugin defaults: set parameters</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `input_gain_db` | `Inputgain` | -60 to 20 | dB | 0 | Gain before the bands |
| `output_gain_db` | `Outputgain` | -60 to 20 | dB | 0 | Gain after the bands |
| `hp_on` | `HPactive` | on/off |  | off | High-pass filter on |
| `hp_freq_hz` | `HPfreq` | 20 to 20000 | Hz | 31 | High-pass cutoff |
| `hp_res` | `HPres` | 0.003 to 10 |  | 0.707 | High-pass resonance (Q) |
| `hp_slope_db` | `HP` | 12 / 24 / 48 |  | 12 | High-pass slope in dB per octave |
| `lowshelf_on` | `Lowshelfactive` | on/off |  | off | Low shelf on |
| `lowshelf_gain_db` | `Lowshelfgain` | -18 to 18 | dB | 0 | Low shelf boost/cut |
| `lowshelf_freq_hz` | `LowShelffreq` | 20 to 20000 | Hz | 80 | Low shelf corner |
| `lowshelf_res` | `LowShelfres` | 0.55 to 10 |  | 1.4 | Low shelf resonance |
| `peak1_on` | `Peak1active` | on/off |  | off | Peak band 1 on |
| `peak1_gain_db` | `Peak1gain` | -18 to 18 | dB | 0 | Peak band 1 boost/cut |
| `peak1_freq_hz` | `Peak1freq` | 20 to 20000 | Hz | 120 | Peak band 1 centre |
| `peak1_bw` | `Peak1bw` | 0.1 to 4 | octaves | 0.3 | Peak band 1 bandwidth |
| `peak2_on` | `Peak2active` | on/off |  | off | Peak band 2 on |
| `peak2_gain_db` | `Peak2gain` | -18 to 18 | dB | 0 | Peak band 2 boost/cut |
| `peak2_freq_hz` | `Peak2freq` | 20 to 20000 | Hz | 250 | Peak band 2 centre |
| `peak2_bw` | `Peak2bw` | 0.1 to 4 | octaves | 0.3 | Peak band 2 bandwidth |
| `peak3_on` | `Peak3active` | on/off |  | off | Peak band 3 on |
| `peak3_gain_db` | `Peak3gain` | -18 to 18 | dB | 0 | Peak band 3 boost/cut |
| `peak3_freq_hz` | `Peak3freq` | 20 to 20000 | Hz | 2000 | Peak band 3 centre |
| `peak3_bw` | `Peak3bw` | 0.1 to 4 | octaves | 0.3 | Peak band 3 bandwidth |
| `peak4_on` | `Peak4active` | on/off |  | off | Peak band 4 on |
| `peak4_gain_db` | `Peak4gain` | -18 to 18 | dB | 0 | Peak band 4 boost/cut |
| `peak4_freq_hz` | `Peak4freq` | 20 to 20000 | Hz | 4000 | Peak band 4 centre |
| `peak4_bw` | `Peak4bw` | 0.1 to 4 | octaves | 0.3 | Peak band 4 bandwidth |
| `highshelf_on` | `Highshelfactive` | on/off |  | off | High shelf on |
| `highshelf_gain_db` | `HighShelfgain` | -18 to 18 | dB | 0 | High shelf boost/cut |
| `highshelf_freq_hz` | `Highshelffreq` | 20 to 20000 | Hz | 12000 | High shelf corner |
| `highshelf_res` | `HighShelfres` | 0.55 to 10 |  | 1.4 | High shelf resonance |
| `lp_on` | `LPactive` | on/off |  | off | Low-pass filter on |
| `lp_freq_hz` | `LPfreq` | 20 to 20000 | Hz | 18000 | Low-pass cutoff |
| `lp_res` | `LPres` | 0.003 to 10 |  | 0.707 | Low-pass resonance (Q) |
| `lp_slope_db` | `LP` | 12 / 24 / 48 |  | 12 | Low-pass slope in dB per octave |

</details>

<details>
<summary><code>flanger</code> &mdash; starts with: wet 0.5, delay_s 0.002, lfo_period_s 3, lfo_amount_s 0.0015, feedback 0.3</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `delay_s` | `DelayTimeSamples` | 0.0001 to 0.05 | s | 0.001 | Base delay time |
| `lfo_period_s` | `LfoFrequency` | 0.01 to 60 | s | 0.25 | Period of the sweep (the XML name says frequency): 4 = slow |
| `lfo_amount_s` | `LfoAmount` | 0 to 0.0025 | s | 0 | Sweep depth; 0 = a static comb filter, no flanging |
| `feedback` | `Feedback` | 0 to 1 |  | 0 | Regeneration: more = more metallic ring |
| `noise` | `WhiteNoise` | 0 to 0.05 |  | 0 | White noise added to the input |
| `invert_feedback` | `Invert` | on/off |  | off | Cross the feedback between the channels |

</details>

<details>
<summary><code>multitapecho</code> &mdash; starts with: steps 8, tap_levels_db decay, step_length_ms = a 1/16 note at the project tempo</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `steps` | `steps` | 4 to 32 |  | 16 | Number of echo taps |
| `step_length_ms` | `steplength` | 1 to 500 | ms | 100 | Time between taps |
| `dry_gain_db` | `drygain` | -80 to 20 | dB | 0 | Level of the direct sound inside the effect (-80 = echoes only) |
| `swap_inputs` | `swapinputs` | on/off |  | off | Echoes swap left and right |
| `tap_levels_db` | `ampsteps (curve)` | -60 to 0 | dB | 0 | Level of each tap: a number (all taps), a list of `steps` numbers, or "decay" |
| `tap_cutoff_hz` | `lpsteps (curve)` | 20 to 20000 | Hz | 20000 | Low-pass cutoff of each tap: a number or a list of `steps` numbers |

</details>

<details>
<summary><code>reverbsc</code> &mdash; starts with: wet 0.3</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `input_gain_db` | `input_gain` | -60 to 15 | dB | 0 | Gain into the reverb |
| `size` | `size` | 0 to 1 |  | 0.89 | Reverb length; 1 never decays |
| `color_hz` | `color` | 100 to 15000 | Hz | 10000 | Low-pass of the tail (lower = darker) |
| `output_gain_db` | `output_gain` | -60 to 15 | dB | 0 | Gain of the reverb |

</details>

<details>
<summary><code>stereoenhancer</code> &mdash; starts with: width 60</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `width` | `width` | 0 to 180 | samples | 0 | Right channel delayed against the left by this many samples (1 ms is about 44); 0 = off |

</details>

<details>
<summary><code>stereomatrix</code> &mdash; starts with: neutral at the plugin defaults: set parameters</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `left_to_left` | `l-l` | -1 to 1 |  | 1 | How much of the left input goes to the left output |
| `left_to_right` | `l-r` | -1 to 1 |  | 0 | Left input to the right output |
| `right_to_left` | `r-l` | -1 to 1 |  | 0 | Right input to the left output |
| `right_to_right` | `r-r` | -1 to 1 |  | 1 | Right input to the right output |

</details>

<details>
<summary><code>waveshaper</code> &mdash; starts with: shape soft_clip, input_gain 1.5, output_gain 0.6</summary>

| Parameter | XML attribute | Range | Unit | Default | Meaning |
|-----------|---------------|-------|------|---------|---------|
| `input_gain` | `inputGain` | 0 to 5 |  | 1 | Gain before the shaper (drive) |
| `output_gain` | `outputGain` | 0 to 5 |  | 1 | Gain after the shaper |
| `clip_input` | `clipInput` | on/off |  | off | Clip the input at full scale before shaping |
| `shape` | `waveShape (curve)` | name or list |  | linear | Transfer curve: a name (see WAVESHAPES) or a list of 200 numbers from 0 to 1 (output for input 1/200 ... 200/200) |

Shapes: `linear` (no change (the LMMS default)); `soft_clip` (tanh saturation, adds odd harmonics, quiet parts get louder); `hard_clip` (clips at one third of full scale); `fuzz` (almost a square wave); `fold` (wavefolder, loud parts fold back); `crush` (8-step staircase).

</details>



## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LMMS_PROJECTS_DIR` | `~/Desktop/Media/lmms/AI-Projects` | Default directory for projects: where `save_project` saves when no path is given, and what relative paths of `save_project`, `load_project` and `render_project`'s `output_path` are relative to |
| `LMMS_PRESETS_DIR` | auto-detected | Path to the ZynAddSubFX presets folder (`data/presets/ZynAddSubFX`) or its parent presets folder |
| `LMMS_PRESET_DIRS` | none | Extra folders (separated by `:` on Linux, `;` on Windows) with instrument presets (`.xpf`), searched before your LMMS presets folder and the factory presets |
| `LMMS_MCP_MEASURE_LEVELS` | `1` | `0` forbids the short LMMS renders that measure how loud a preset, sample or soundfont is (see [Loudness before the first render](#loudness-before-the-first-render)); the built-in levels and the cache are still used |
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
