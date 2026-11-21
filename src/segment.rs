//! Utterance segmentation: 16 ms frames scored for voice -> utterances that start a little before the first
//! voiced frame and end after a pause. Pure: the scorer is injected, so tests drive it with plain numbers.

pub const RATE: usize = 16_000;
pub const FRAME: usize = 256; // earshot's frame: 16 ms at 16 kHz

#[derive(Debug, Clone, Copy)]
pub struct SegmentConfig {
    /// Voice score at or above which a frame counts as speech.
    pub threshold: f32,
    /// Silence that ends an utterance.
    pub silence_ms: u32,
    /// Audio kept from before the first voiced frame, so the first syllable is not clipped.
    pub preroll_ms: u32,
    /// Longest utterance before a forced cut.
    pub max_ms: u32,
    /// Shorter voiced runs are dropped as clicks and coughs.
    pub min_speech_ms: u32,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self { threshold: 0.5, silence_ms: 600, preroll_ms: 300, max_ms: 20_000, min_speech_ms: 200 }
    }
}

#[derive(Debug, PartialEq)]
pub enum Event {
    /// An utterance began; `start` is its first sample (pre-roll included) since input start.
    Start { start: u64 },
    /// Audio belonging to the open utterance (the pre-roll arrives with the first one).
    Audio(Vec<f32>),
    /// The open utterance ended. `voiced` = it had enough speech to be worth decoding.
    End { voiced: bool },
}

fn frames(ms: u32) -> usize {
    (ms as usize * RATE / 1000).div_ceil(FRAME)
}

pub struct Segmenter<F: FnMut(&[f32]) -> f32> {
    cfg: SegmentConfig,
    score: F,
    pending: Vec<f32>,
    preroll: std::collections::VecDeque<Vec<f32>>,
    open: bool,
    quiet: usize,
    voiced: usize,
    len: usize,
    consumed: u64,
}

impl<F: FnMut(&[f32]) -> f32> Segmenter<F> {
    pub fn new(cfg: SegmentConfig, score: F) -> Self {
        Self {
            cfg, score, pending: Vec::new(), preroll: Default::default(),
            open: false, quiet: 0, voiced: 0, len: 0, consumed: 0,
        }
    }

    /// Feeds any amount of 16 kHz mono audio; returns what happened, in order.
    pub fn push(&mut self, pcm: &[f32]) -> Vec<Event> {
        self.pending.extend_from_slice(pcm);
        let mut out = Vec::new();
        let whole = self.pending.len() / FRAME * FRAME;
        let mut buf = std::mem::take(&mut self.pending);
        self.pending = buf.split_off(whole);
        for f in buf.chunks_exact(FRAME) {
            self.frame(f, &mut out);
        }
        out
    }

    /// Ends the input: an open utterance is closed.
    pub fn finish(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        if self.open {
            if !self.pending.is_empty() {
                out.push(Event::Audio(std::mem::take(&mut self.pending)));
            }
            out.push(Event::End { voiced: self.voiced >= frames(self.cfg.min_speech_ms) });
            self.open = false;
        }
        out
    }

    fn frame(&mut self, f: &[f32], out: &mut Vec<Event>) {
        let speech = (self.score)(f) >= self.cfg.threshold;
        let at = self.consumed;
        self.consumed += FRAME as u64;
        if !self.open {
            if !speech {
                self.preroll.push_back(f.to_vec());
                while self.preroll.len() > frames(self.cfg.preroll_ms) {
                    self.preroll.pop_front();
                }
                return;
            }
            let pre: Vec<f32> = self.preroll.drain(..).flatten().collect();
            out.push(Event::Start { start: at - pre.len() as u64 });
            let mut first = pre;
            first.extend_from_slice(f);
            self.len = first.len() / FRAME;
            out.push(Event::Audio(first));
            self.open = true;
            self.quiet = 0;
            self.voiced = 1;
            return;
        }
        out.push(Event::Audio(f.to_vec()));
        self.len += 1;
        if speech {
            self.quiet = 0;
            self.voiced += 1;
        } else {
            self.quiet += 1;
        }
        if self.quiet >= frames(self.cfg.silence_ms) || self.len >= frames(self.cfg.max_ms) {
            out.push(Event::End { voiced: self.voiced >= frames(self.cfg.min_speech_ms) });
            self.open = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames carry their own score in sample 0, so a test writes the voice pattern directly.
    fn run(scores: &[f32], cfg: SegmentConfig) -> Vec<Event> {
        let mut s = Segmenter::new(cfg, |f: &[f32]| f[0]);
        let mut out = Vec::new();
        for &v in scores {
            out.extend(s.push(&[v; FRAME]));
        }
        out.extend(s.finish());
        out
    }

    fn summary(ev: &[Event]) -> Vec<String> {
        ev.iter()
            .filter_map(|e| match e {
                Event::Start { start } => Some(format!("start@{}", start / FRAME as u64)),
                Event::End { voiced } => Some(format!("end:{voiced}")),
                Event::Audio(_) => None,
            })
            .collect()
    }

    fn cfg() -> SegmentConfig {
        // 3 frames of pre-roll, 4 of silence ends, 2 voiced minimum.
        SegmentConfig { threshold: 0.5, silence_ms: 64, preroll_ms: 48, max_ms: 20_000, min_speech_ms: 32 }
    }

    #[test]
    fn utterance_starts_with_preroll_and_ends_after_the_pause() {
        let mut s = vec![0.0; 10];
        s.extend([1.0; 5]);
        s.extend([0.0; 10]);
        let ev = run(&s, cfg());
        assert_eq!(summary(&ev), ["start@7", "end:true"]);
        let audio: usize = ev.iter().map(|e| if let Event::Audio(a) = e { a.len() } else { 0 }).sum();
        assert_eq!(audio, (3 + 5 + 4) * FRAME, "pre-roll + speech + the closing silence");
    }

    #[test]
    fn a_short_gap_does_not_split_and_a_click_is_not_voiced() {
        let mut s = vec![1.0; 3];
        s.extend([0.0; 2]);
        s.extend([1.0; 3]);
        s.extend([0.0; 6]);
        s.extend([1.0; 1]); // pre-roll can only reach back to frame 12: 8..=11 closed the first utterance
        s.extend([0.0; 6]);
        assert_eq!(summary(&run(&s, cfg())), ["start@0", "end:true", "start@12", "end:false"]);
    }

    #[test]
    fn long_speech_is_cut_and_open_speech_closes_at_the_end() {
        let c = SegmentConfig { max_ms: 160, ..cfg() };
        assert_eq!(summary(&run(&[1.0; 25], c)), ["start@0", "end:true", "start@10", "end:true", "start@20", "end:true"]);
    }

    #[test]
    fn odd_sized_pushes_are_reassembled_into_frames() {
        let mut s = Segmenter::new(cfg(), |f: &[f32]| f[0]);
        let mut ev = Vec::new();
        let audio = vec![1.0f32; FRAME * 4];
        for chunk in audio.chunks(100) {
            ev.extend(s.push(chunk));
        }
        ev.extend(s.finish());
        assert_eq!(summary(&ev), ["start@0", "end:true"]);
    }
}
