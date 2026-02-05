use crate::pipeline::{AudioRawFrame, ControlFrame, Frame, Processor};
use crate::audio::resampler::Resampler;
use anyhow::Result;
use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

#[allow(dead_code)]
pub struct PlaybackProcessor {
    audio_tx: Option<mpsc::Sender<Vec<f32>>>,
    stop_tx: Option<mpsc::Sender<()>>,
    resampler: Option<Arc<Mutex<Resampler>>>,
}

#[allow(dead_code)]
impl PlaybackProcessor {
    pub fn new() -> Self {
        Self {
            audio_tx: None,
            stop_tx: None,
            resampler: None,
        }
    }

    fn start_playback(&mut self) -> Result<()> {
        let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(100);
        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);
        
        std::thread::spawn(move || {
            let host = cpal::default_host();
            let device = match host.default_output_device() {
                Some(d) => d,
                None => return,
            };
            let config: cpal::StreamConfig = match device.default_output_config() {
                Ok(c) => c.into(),
                Err(_) => return,
            };
            
            let mut internal_queue = std::collections::VecDeque::new();
            
            let stream = match device.build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Try to fill from rx
                    while let Ok(samples) = audio_rx.try_recv() {
                        internal_queue.extend(samples);
                    }
                    
                    for sample in data.iter_mut() {
                        *sample = internal_queue.pop_front().unwrap_or(0.0);
                    }
                },
                |err| eprintln!("Playback error: {}", err),
                None
            ) {
                Ok(s) => s,
                Err(_) => return,
            };
            
            let _ = stream.play();
            
            // Wait for stop or rx closed
            let _ = stop_rx.blocking_recv();
            drop(stream);
        });

        // Initialize resampler for 24kHz -> device rate
        let host = cpal::default_host();
        let device = host.default_output_device().unwrap();
        let config: cpal::StreamConfig = device.default_output_config().unwrap().into();
        let target_rate = config.sample_rate.0;
        let channels = config.channels as usize;
        
        self.resampler = Some(Arc::new(Mutex::new(Resampler::new(24000, target_rate, 480, channels)?)));
        self.audio_tx = Some(audio_tx);
        self.stop_tx = Some(stop_tx);
        
        Ok(())
    }

    fn stop_playback(&mut self) {
        self.audio_tx = None;
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.try_send(());
        }
    }
}

#[async_trait]
impl Processor for PlaybackProcessor {
    async fn process(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        match frame {
            Frame::Audio(AudioRawFrame { samples, .. }) => {
                if self.audio_tx.is_none() {
                    self.start_playback()?;
                }
                
                if let (Some(resampler), Some(tx)) = (&self.resampler, &self.audio_tx) {
                    let mut r = resampler.lock().unwrap();
                    if let Ok(resampled) = r.process(&samples) {
                        let _ = tx.try_send(resampled);
                    }
                }
                Ok(vec![])
            }
            Frame::Control(ControlFrame::Cancel) | Frame::Control(ControlFrame::Stop) => {
                self.stop_playback();
                Ok(vec![])
            }
            _ => Ok(vec![frame]),
        }
    }
}
