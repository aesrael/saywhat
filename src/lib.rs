//! saywhat: live, local speech-to-text as a library.
//!
//! Feed 16 kHz mono `f32` audio from anywhere (a microphone callback, a socket, a file); get text back
//! through a callback: [`Update::Partial`] while someone is talking (streaming models) and
//! [`Update::Final`] when they pause. Decoding runs on its own thread, so [`Transcriber::feed`] never
//! blocks an audio callback.
//!
//! ```no_run
//! let t = saywhat::Transcriber::start(saywhat::Config::new("model.gguf"), |u| match u {
//!     saywhat::Update::Partial { text, .. } => eprint!("\r{text}"),
//!     saywhat::Update::Final { text, .. } => println!("{text}"),
//! })?;
//! t.feed(&[0.0f32; 1600]);
//! t.finish()?; // decodes what is still being said, then returns
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod segment;
#[cfg(feature = "mic")]
pub mod mic;

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
pub use segment::{SegmentConfig, FRAME, RATE};
use segment::{Event, Segmenter};
use transcribe_cpp::{Model, RunOptions, SessionOptions, StreamOptions};

/// What the callback receives. `t` is seconds from the start of the input to the utterance's start.
#[derive(Debug, Clone, PartialEq)]
pub enum Update {
    /// The utterance so far, revised as the speaker continues (streaming models only).
    Partial { t: f64, text: String },
    /// The settled text of an utterance, after the pause that ended it.
    Final { t: f64, dur: f64, text: String },
}

#[derive(Debug, Clone)]
pub struct Config {
    /// A transcribe-cpp GGUF model file.
    pub model: PathBuf,
    /// Language hint (ISO code) for multilingual models.
    pub language: Option<String>,
    /// Decode threads (0 = library default).
    pub threads: i32,
    pub segment: SegmentConfig,
    /// Minimum gap between partials.
    pub partial_every: Duration,
}

impl Config {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            language: None,
            threads: 0,
            segment: SegmentConfig::default(),
            partial_every: Duration::from_millis(250),
        }
    }
}

/// Facts about the loaded model, known once [`Transcriber::start`] returns.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub streaming: bool,
    pub load_time: Duration,
}

pub struct Transcriber {
    tx: Option<mpsc::Sender<Vec<f32>>>,
    worker: Option<JoinHandle<Result<()>>>,
    pub loaded: Loaded,
}

impl Transcriber {
    /// Loads the model (on the worker thread) and returns once it is ready, or with the load error.
    pub fn start<F>(cfg: Config, on_update: F) -> Result<Self>
    where
        F: FnMut(Update) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<Vec<f32>>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<Loaded>>();
        let worker = std::thread::Builder::new()
            .name("saywhat-decode".into())
            .spawn(move || run(cfg, rx, ready_tx, on_update))?;
        match ready_rx.recv() {
            Ok(Ok(loaded)) => Ok(Self { tx: Some(tx), worker: Some(worker), loaded }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(worker.join().map_err(|_| anyhow!("decoder panicked"))?.err().unwrap_or_else(|| anyhow!("decoder exited"))),
        }
    }

    /// Queues audio: 16 kHz mono, samples in -1..1. Never blocks.
    pub fn feed(&self, pcm: &[f32]) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(pcm.to_vec());
        }
    }

    /// A sender for feeding from another thread (an audio callback, say).
    pub fn feeder(&self) -> mpsc::Sender<Vec<f32>> {
        self.tx.clone().expect("transcriber running")
    }

    /// Ends the input: the open utterance is decoded, every queued update delivered, then this returns.
    pub fn finish(mut self) -> Result<()> {
        self.stop()
    }

    fn stop(&mut self) -> Result<()> {
        self.tx.take();
        match self.worker.take() {
            Some(w) => w.join().map_err(|_| anyhow!("decoder panicked"))?,
            None => Ok(()),
        }
    }
}

impl Drop for Transcriber {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn run<F: FnMut(Update)>(cfg: Config, rx: mpsc::Receiver<Vec<f32>>, ready: mpsc::Sender<Result<Loaded>>, mut emit: F) -> Result<()> {
    let t0 = Instant::now();
    let setup = (|| -> Result<_> {
        transcribe_cpp::disable_logging();
        transcribe_cpp::init_backends_default().map_err(|e| anyhow!("{e}"))?;
        let model = Model::load(&cfg.model).map_err(|e| anyhow!("loading {}: {e}", cfg.model.display()))?;
        let streaming = model.capabilities().map(|c| c.supports_streaming).unwrap_or(false);
        let session = model
            .session_with(&SessionOptions { n_threads: cfg.threads, ..Default::default() })
            .map_err(|e| anyhow!("{e}"))?;
        Ok((model, session, streaming))
    })();
    let (_model, mut session, streaming) = match setup {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e));
            return Ok(());
        }
    };
    let _ = ready.send(Ok(Loaded { streaming, load_time: t0.elapsed() }));
    let run = RunOptions { language: cfg.language.clone(), ..Default::default() };

    let mut vad = earshot::Detector::default_boxed();
    let seg = Segmenter::new(cfg.segment, move |f: &[f32]| {
        let mut c = [0f32; FRAME];
        for (d, s) in c.iter_mut().zip(f) {
            *d = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
        }
        vad.predict_f32(&c)
    });
    let mut pump = Pump { rx, seg, queue: Default::default(), done: false };

    // One utterance at a time. A streaming model gets a fresh stream per utterance (a finalised stream
    // cannot restart) and is fed while the speaker talks; other models decode the utterance at the pause.
    while let Some(ev) = pump.next() {
        let Event::Start { start } = ev else { continue };
        let t = start as f64 / RATE as f64;
        let mut samples = 0usize;
        if streaming {
            let mut st = session.stream(&run, &StreamOptions::default()).map_err(|e| anyhow!("{e}"))?;
            let mut shown = Instant::now();
            let voiced = loop {
                match pump.next() {
                    Some(Event::Audio(a)) => {
                        samples += a.len();
                        let u = st.feed(&a).map_err(|e| anyhow!("{e}"))?;
                        if (u.committed_changed || u.tentative_changed) && shown.elapsed() >= cfg.partial_every {
                            let text = st.text().display().trim().to_string();
                            if !text.is_empty() {
                                emit(Update::Partial { t, text });
                            }
                            shown = Instant::now();
                        }
                    }
                    Some(Event::End { voiced }) => break voiced,
                    Some(Event::Start { .. }) | None => break true,
                }
            };
            st.finalize().map_err(|e| anyhow!("{e}"))?;
            let text = st.text().full.trim().to_string();
            if voiced && !text.is_empty() {
                emit(Update::Final { t, dur: samples as f64 / RATE as f64, text });
            }
        } else {
            let mut utt = Vec::new();
            let voiced = loop {
                match pump.next() {
                    Some(Event::Audio(a)) => utt.extend_from_slice(&a),
                    Some(Event::End { voiced }) => break voiced,
                    Some(Event::Start { .. }) | None => break true,
                }
            };
            if voiced {
                let r = session.run(&utt, &run).map_err(|e| anyhow!("{e}"))?;
                let text = r.text.trim().to_string();
                if !text.is_empty() {
                    emit(Update::Final { t, dur: utt.len() as f64 / RATE as f64, text });
                }
            }
        }
    }
    Ok(())
}

/// Audio in, segment events out, in order. Once the input ends the open utterance is closed, so the speech
/// in progress is still decoded.
struct Pump<F: FnMut(&[f32]) -> f32> {
    rx: mpsc::Receiver<Vec<f32>>,
    seg: Segmenter<F>,
    queue: std::collections::VecDeque<Event>,
    done: bool,
}

impl<F: FnMut(&[f32]) -> f32> Pump<F> {
    fn next(&mut self) -> Option<Event> {
        loop {
            if let Some(e) = self.queue.pop_front() {
                return Some(e);
            }
            if self.done {
                return None;
            }
            match self.rx.recv() {
                Ok(chunk) => self.queue.extend(self.seg.push(&chunk)),
                Err(_) => {
                    self.queue.extend(self.seg.finish());
                    self.done = true;
                }
            }
        }
    }
}

/// Mono 16 kHz from interleaved audio at any rate: channels averaged, rate converted linearly (fine for
/// speech). For callers whose source is not already 16 kHz mono.
pub fn to_mono_16k(interleaved: &[f32], channels: usize, rate: usize) -> Vec<f32> {
    let ch = channels.max(1);
    let mono: Vec<f32> = interleaved.chunks(ch).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect();
    if rate == RATE || mono.is_empty() {
        return mono;
    }
    let n = mono.len() * RATE / rate;
    (0..n)
        .map(|i| {
            let pos = i as f64 * rate as f64 / RATE as f64;
            let j = pos as usize;
            let frac = (pos - j as f64) as f32;
            let a = mono[j.min(mono.len() - 1)];
            let b = mono[(j + 1).min(mono.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_keeps_duration_and_averages_channels() {
        let stereo: Vec<f32> = (0..48_000).flat_map(|_| [0.2f32, 0.4]).collect();
        let out = to_mono_16k(&stereo, 2, 48_000);
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|&s| (s - 0.3).abs() < 1e-6));
    }

    #[test]
    fn a_missing_model_fails_at_start() {
        let e = Transcriber::start(Config::new("/nonexistent.gguf"), |_| {}).err().expect("must fail");
        assert!(e.to_string().contains("nonexistent"), "{e}");
    }
}
