# saywhat

Live speech-to-text that runs entirely on your machine, as a command-line tool and a Rust library.

Speak and the words appear while you talk. Pause, and the line settles. Audio comes from a microphone, a
WAV file or raw PCM on stdin; text goes out as plain lines or JSON. No network, no account, no audio leaves
the device.

```console
$ saywhat --partials
parakeet-unified-en-0.6b-Q8_0.gguf loaded in 0.3s (streaming); listening to MacBook Pro Microphone, Ctrl-C to stop
[00:02] Book the table for Friday at eight.
[00:07] And remind me to call the garage in the morning.
```

## Features

- **Live partial results.** With a streaming model, text is shown and revised while the speaker is still
  talking, then finalised at the pause.
- **Utterance segmentation.** A voice-activity detector splits speech at natural pauses, keeps a short
  pre-roll so first syllables are not clipped, and caps very long utterances.
- **Any input.** Microphone, WAV at any rate or channel count, or 16 kHz PCM on stdin for audio that arrives
  from elsewhere (a phone, a socket, ffmpeg).
- **Scriptable output.** One line per utterance, or JSON lines with timings; optionally appended to a file.
  `SIGINT`/`SIGTERM` finish the utterance in progress before exiting.
- **Library first.** The binary is a thin front end over a small API that any Rust program can embed.

## Installation

Requires a Rust toolchain, a C++17 compiler and CMake (the speech engine is built from source).

```sh
git clone https://github.com/aesrael/saywhat && cd saywhat
cargo install --path .
saywhat download parakeet-unified-en-0.6b
```

On Apple Silicon, build with `cargo install --path . --features metal` for GPU decoding. On macOS the
terminal that runs `saywhat` needs microphone access; the first run asks for it.

## Usage

```sh
saywhat                                   # microphone, until Ctrl-C
saywhat --partials                        # show text while it is being spoken
saywhat --file meeting.wav                # a file, fed at real time
saywhat --file meeting.wav --fast         # a file, as fast as it decodes
ffmpeg -i call.m4a -f s16le -ar 16000 -ac 1 - | saywhat --stdin
saywhat --json --out transcript.txt       # JSON lines on stdout, final lines also saved
saywhat -m parakeet-tdt-0.6b-v3-Q8_0.gguf --language de
```

| Command | |
|---|---|
| `saywhat models` | known models, and which are downloaded |
| `saywhat download <id \| url>` | fetch a model by id, or any `.gguf` URL (resumable) |
| `saywhat devices` | microphones |

| Option | Default | |
|---|---|---|
| `-m, --model` | first downloaded | model file or name in the models folder |
| `--device` | system default | microphone, by part of its name |
| `--partials` | off | live text on stderr (terminal only) |
| `--json` | off | JSON lines on stdout |
| `-o, --out` | none | append final lines to a file |
| `--silence-ms` | 600 | pause that ends an utterance |
| `--max-ms` | 20000 | longest utterance before a forced cut |
| `--threshold` | 0.5 | voice score, 0 to 1 |
| `--threads` | engine default | decode threads |
| `--language` | auto | language hint for multilingual models |

Models are stored in `~/.local/share/saywhat/models`, or in `$SAYWHAT_MODELS` if set.

### JSON output

```json
{"type":"partial","t":12.31,"text":"book the table for"}
{"type":"final","t":12.31,"dur":2.84,"text":"Book the table for Friday at eight."}
```

`t` is the utterance's start in seconds from the beginning of the input; `dur` is its length. Partials
are emitted only by streaming models.

## Library

```toml
[dependencies]
saywhat = { git = "https://github.com/aesrael/saywhat", default-features = false }   # add "mic" for microphone capture
```

```rust
use saywhat::{Config, Transcriber, Update};

let t = Transcriber::start(Config::new("parakeet-unified-en-0.6b-Q8_0.gguf"), |u| match u {
    Update::Partial { text, .. } => eprint!("\r{text}"),
    Update::Final { t, text, .. } => println!("[{t:.1}s] {text}"),
})?;

t.feed(&samples);   // 16 kHz mono f32; never blocks
t.finish()?;        // decodes the utterance in progress, delivers every update, then returns
```

`Transcriber` decodes on its own thread, so `feed` (or a sender from `feeder()`) is safe to call from an
audio callback. `saywhat::to_mono_16k` converts other rates and channel layouts. With the `mic` feature,
`saywhat::mic::open` streams a microphone into a callback.

| Feature | Default | |
|---|---|---|
| `cli` | yes | the `saywhat` binary |
| `mic` | yes | microphone capture |
| `metal` | no | Apple Silicon GPU |

## How it works

Audio is resampled to 16 kHz mono and scored for voice in 16 ms frames. An utterance opens on the first
voiced frame, together with the 300 ms before it, and closes after the configured pause. A streaming model
is fed while the utterance is open, which is what produces partial results, and is finalised when it closes;
other models decode the whole utterance at the pause. Each utterance is decoded independently, so a session
can run indefinitely.

## Models

`saywhat` runs speech models in GGUF format through transcribe.cpp. Ready-made conversions:

| Id | Languages | Mode | Size | Licence |
|---|---|---|---|---|
| `parakeet-unified-en-0.6b` | English | streaming | 731 MB | CC-BY-4.0 |
| `parakeet-tdt-0.6b-v3` | 25 European | per utterance | 740 MB | CC-BY-4.0 |
| `parakeet-tdt_ctc-110m` | English | per utterance | small | CC-BY-4.0 |

Any other model transcribe.cpp supports (Moonshine, Whisper and more) works with `--model path/to/file.gguf`
or `saywhat download <url>`.

## Credits

saywhat is a small layer over the work of others:

- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) and its Rust bindings
  ([transcribe-cpp](https://crates.io/crates/transcribe-cpp)), by handy-computer: the speech engine, built on
  [ggml](https://github.com/ggml-org/ggml). MIT.
- The GGUF model conversions listed above, published by
  [handy-computer on Hugging Face](https://huggingface.co/handy-computer). The underlying Parakeet models are
  by NVIDIA and licensed CC-BY-4.0; other models carry their own licences.
- [earshot](https://github.com/pykeio/earshot), by pyke: voice-activity detection. MIT or Apache-2.0.
- [cpal](https://github.com/RustAudio/cpal), by RustAudio: cross-platform audio input. Apache-2.0.
- [hound](https://github.com/ruuda/hound) (WAV), [clap](https://github.com/clap-rs/clap) (arguments),
  [signal-hook](https://github.com/vorner/signal-hook) and [serde_json](https://github.com/serde-rs/json).

## Licence

MIT. See [LICENSE](LICENSE). Model files are not part of this repository and are subject to their own
licences.
