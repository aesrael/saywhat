//! saywhat: live, local speech-to-text on the command line. A thin front end over the library.

use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use saywhat::{Config, SegmentConfig, Transcriber, Update, RATE};

/// Known models: id, URL, licence. GGUF conversions published by handy-computer on Hugging Face.
const MODELS: &[(&str, &str, &str)] = &[
    ("parakeet-unified-en-0.6b",
     "https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/7e948f21b7bdbac698d3318db9d350f1096f3b6c/parakeet-unified-en-0.6b-Q8_0.gguf",
     "English, streaming. CC-BY-4.0 (NVIDIA)"),
    ("parakeet-tdt-0.6b-v3",
     "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/85ac09ea12fc4b1112fa76810059364bc6adc9de/parakeet-tdt-0.6b-v3-Q8_0.gguf",
     "25 European languages. CC-BY-4.0 (NVIDIA)"),
    ("parakeet-tdt_ctc-110m",
     "https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/9d66d34f9e1594075c5dd72c90c0f4c321b29f21/parakeet-tdt_ctc-110m-Q8_0.gguf",
     "English, small and fast. CC-BY-4.0 (NVIDIA)"),
];

#[derive(Parser)]
#[command(version, about = "Live local speech-to-text: mic, WAV or stdin PCM in, text out")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// GGUF model: a path, or a file name in the models folder (default: the only/first one there).
    #[arg(long, short, global = true)]
    model: Option<String>,
    /// Transcribe a WAV file, fed at real time unless --fast.
    #[arg(long, conflicts_with = "stdin")]
    file: Option<PathBuf>,
    /// Read raw 16 kHz mono signed 16-bit little-endian PCM from stdin.
    #[arg(long)]
    stdin: bool,
    /// With --file: decode as fast as possible instead of at real time.
    #[arg(long)]
    fast: bool,
    /// Microphone (substring of its name, see `devices`); default input otherwise.
    #[arg(long)]
    device: Option<String>,
    /// One JSON object per line: {"type":"partial"|"final","t":s,"dur":s,"text":...}.
    #[arg(long)]
    json: bool,
    /// Show text while the speaker is still talking (always on with --json and a streaming model).
    #[arg(long)]
    partials: bool,
    /// Also append final lines to this file.
    #[arg(long, short)]
    out: Option<PathBuf>,
    /// Pause (ms) that ends an utterance.
    #[arg(long, default_value_t = 600)]
    silence_ms: u32,
    /// Longest utterance (ms) before a forced cut.
    #[arg(long, default_value_t = 20_000)]
    max_ms: u32,
    /// Voice threshold 0..1.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    /// Decode threads (0 = library default).
    #[arg(long, default_value_t = 0)]
    threads: i32,
    /// Language hint (ISO code) for multilingual models.
    #[arg(long)]
    language: Option<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// List known and downloaded models.
    Models,
    /// Download a model by id (see `models`) or from a URL to a .gguf file (resumable).
    Download { model: String },
    /// List microphones.
    Devices,
}

fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("SAYWHAT_MODELS") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".local/share/saywhat/models")
}

fn local_models() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(models_dir())
        .map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "gguf")).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn resolve_model(arg: Option<&str>) -> Result<PathBuf> {
    match arg {
        Some(a) => {
            let p = PathBuf::from(a);
            if p.exists() {
                return Ok(p);
            }
            let in_dir = models_dir().join(a);
            if in_dir.exists() {
                return Ok(in_dir);
            }
            bail!("no model {a} (neither a file nor in {})", models_dir().display())
        }
        None => {
            // The first known model that is present, then any other .gguf in the folder.
            let have = local_models();
            MODELS
                .iter()
                .find_map(|m| have.iter().find(|p| m.1.ends_with(&*p.file_name().unwrap_or_default().to_string_lossy())).cloned())
                .or_else(|| have.into_iter().next())
                .ok_or_else(|| anyhow::anyhow!("no model in {}: run `saywhat download {}`", models_dir().display(), MODELS[0].0))
        }
    }
}

fn download(url: &str) -> Result<()> {
    let name = url.rsplit('/').next().filter(|n| n.ends_with(".gguf")).context("the URL must end in a .gguf file name")?;
    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(name);
    if dest.exists() {
        eprintln!("{} already present", dest.display());
        return Ok(());
    }
    let part = dir.join(format!("{name}.part"));
    let ok = std::process::Command::new("curl")
        .args(["-L", "--fail", "-C", "-", "--progress-bar", "-o"])
        .arg(&part)
        .arg(url)
        .status()
        .context("running curl")?
        .success();
    if !ok {
        bail!("download failed (rerun to resume)");
    }
    std::fs::rename(&part, &dest)?;
    eprintln!("saved {}", dest.display());
    Ok(())
}

fn read_wav(path: &PathBuf) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = r.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    Ok(saywhat::to_mono_16k(&raw, spec.channels as usize, spec.sample_rate as usize))
}

fn stamp(t: f64) -> String {
    let s = t as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

struct Printer {
    json: bool,
    partials: bool,
    tty: bool,
    file: Option<std::fs::File>,
    shown: bool,
}

impl Printer {
    fn update(&mut self, u: Update) {
        match u {
            Update::Partial { t, text } => {
                if self.json {
                    println!("{}", serde_json::json!({"type": "partial", "t": round2(t), "text": text}));
                } else if self.partials && self.tty {
                    eprint!("\r\x1b[2K\x1b[2m[{}] {}\x1b[0m", stamp(t), text);
                    let _ = std::io::stderr().flush();
                    self.shown = true;
                }
            }
            Update::Final { t, dur, text } => {
                if self.shown {
                    eprint!("\r\x1b[2K");
                    self.shown = false;
                }
                let line = if self.json {
                    serde_json::json!({"type": "final", "t": round2(t), "dur": round2(dur), "text": text}).to_string()
                } else {
                    format!("[{}] {}", stamp(t), text)
                };
                println!("{line}");
                let _ = std::io::stdout().flush();
                if let Some(f) = self.file.as_mut() {
                    let _ = writeln!(f, "{line}");
                    let _ = f.flush();
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.cmd {
        Some(Cmd::Models) => {
            let have: Vec<String> =
                local_models().iter().map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned()).collect();
            for (id, url, about) in MODELS {
                let mark = if have.iter().any(|h| url.ends_with(h.as_str())) { "*" } else { " " };
                println!("{mark} {id:<26} {about}");
            }
            for h in have.iter().filter(|h| !MODELS.iter().any(|m| m.1.ends_with(h.as_str()))) {
                println!("* {h}");
            }
            println!("(* downloaded) folder: {}", models_dir().display());
            return Ok(());
        }
        Some(Cmd::Download { model }) => {
            let url = MODELS.iter().find(|m| m.0 == model).map(|m| m.1).unwrap_or(model.as_str());
            if !url.starts_with("http") {
                bail!("unknown model {model:?}: see `saywhat models`, or pass a URL");
            }
            return download(url);
        }
        Some(Cmd::Devices) => {
            for d in saywhat::mic::devices()? {
                println!("{d}");
            }
            return Ok(());
        }
        None => {}
    }

    let path = resolve_model(cli.model.as_deref())?;
    let file = match &cli.out {
        Some(p) => Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?),
        None => None,
    };
    let mut printer = Printer { json: cli.json, partials: cli.partials, tty: std::io::stderr().is_terminal(), file, shown: false };
    let cfg = Config {
        language: cli.language.clone(),
        threads: cli.threads,
        segment: SegmentConfig {
            threshold: cli.threshold, silence_ms: cli.silence_ms, max_ms: cli.max_ms, ..Default::default()
        },
        ..Config::new(&path)
    };
    let t = Transcriber::start(cfg, move |u| printer.update(u))?;

    let stop = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, stop.clone())?;
    signal_hook::flag::register(signal_hook::consts::SIGTERM, stop.clone())?;

    // Every source ends by closing `done` (end of input) or by a stop signal.
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let feed = t.feeder();
    let mut _mic = None;
    let what = if let Some(f) = cli.file.clone() {
        let pcm = read_wav(&f)?;
        let fast = cli.fast;
        std::thread::spawn(move || {
            for c in pcm.chunks(RATE / 10) {
                if feed.send(c.to_vec()).is_err() {
                    break;
                }
                if !fast {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
            drop(done_tx);
        });
        format!("{}", f.display())
    } else if cli.stdin {
        std::thread::spawn(move || {
            let mut stdin = std::io::stdin().lock();
            let mut buf = vec![0u8; 3200];
            let mut carry: Option<u8> = None;
            while let Ok(n) = stdin.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let mut bytes: Vec<u8> = carry.take().into_iter().collect();
                bytes.extend_from_slice(&buf[..n]);
                if bytes.len() % 2 == 1 {
                    carry = bytes.pop();
                }
                let s = bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();
                if feed.send(s).is_err() {
                    break;
                }
            }
            drop(done_tx);
        });
        "stdin".into()
    } else {
        let mic = saywhat::mic::open(cli.device.as_deref(), move |s| {
            let _ = feed.send(s);
        })
        .context("opening the microphone (macOS: allow microphone access for this terminal)")?;
        let name = format!("{}, Ctrl-C to stop", mic.name);
        _mic = Some((mic, done_tx));
        name
    };
    eprintln!(
        "{} loaded in {:.1}s ({}); listening to {what}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        t.loaded.load_time.as_secs_f32(),
        if t.loaded.streaming { "streaming" } else { "per utterance" },
    );

    while !stop.load(Ordering::Relaxed) {
        if let Err(mpsc::RecvTimeoutError::Disconnected) = done_rx.recv_timeout(Duration::from_millis(200)) {
            break;
        }
    }
    drop(_mic); // the mic stops; what is still being said is decoded below
    t.finish()
}
