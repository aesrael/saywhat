//! Microphone input (feature `mic`): the default or a named input device, delivered as 16 kHz mono `f32`
//! chunks to a callback. Keep the returned [`Mic`] alive for as long as audio should flow.

use anyhow::{anyhow, bail, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::to_mono_16k;

pub struct Mic {
    _stream: cpal::Stream,
    pub name: String,
}

/// Names of the input devices, default first.
pub fn devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.description().ok()).map(|d| d.name().to_string());
    let mut names: Vec<String> = host
        .input_devices()?
        .filter_map(|d| d.description().ok().map(|x| x.name().to_string()))
        .collect();
    if let Some(d) = default {
        names.retain(|n| n != &d);
        names.insert(0, d);
    }
    Ok(names)
}

/// Opens `device` (substring match on its name) or the default input and starts it. `on_audio` runs on
/// the audio thread: hand the samples off (a channel send), never decode there.
pub fn open<F>(device: Option<&str>, mut on_audio: F) -> Result<Mic>
where
    F: FnMut(Vec<f32>) + Send + 'static,
{
    let host = cpal::default_host();
    let dev = match device {
        None => host.default_input_device().ok_or_else(|| anyhow!("no input device"))?,
        Some(want) => host
            .input_devices()?
            .find(|d| d.description().map(|x| x.name().contains(want)).unwrap_or(false))
            .ok_or_else(|| anyhow!("no input device matching {want:?}"))?,
    };
    let name = dev.description().map(|d| d.name().to_string()).unwrap_or_default();
    let supported = dev.default_input_config()?;
    let channels = supported.channels() as usize;
    let rate = supported.sample_rate() as usize;
    let config = supported.config();
    let err = |e| eprintln!("microphone error: {e}");
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream::<f32, _, _>(
            config,
            move |data, _| on_audio(to_mono_16k(data, channels, rate)),
            err,
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream::<i16, _, _>(
            config,
            move |data, _| {
                let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                on_audio(to_mono_16k(&f, channels, rate))
            },
            err,
            None,
        )?,
        other => bail!("unsupported microphone sample format {other:?}"),
    };
    stream.play()?;
    Ok(Mic { _stream: stream, name })
}
