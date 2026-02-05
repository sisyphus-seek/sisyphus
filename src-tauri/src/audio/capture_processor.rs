use crate::pipeline::{AudioRawFrame, ControlFrame, Frame, Processor};
use crate::audio::resampler::Resampler;
use anyhow::Result;
use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokio::sync::mpsc;

// Wrapper to make CaptureProcessor Send/Sync by moving the !Send stream out
#[allow(dead_code)]
pub struct CaptureProcessor {
    stream_handle: Option<mpsc::Sender<()>>, // Sending to this triggers stop
    tx: mpsc::Sender<Frame>,
}

#[allow(dead_code)]
impl CaptureProcessor {
    pub fn new(tx: mpsc::Sender<Frame>) -> Self {
        Self {
            stream_handle: None,
            tx,
        }
    }

    fn start_capture(&mut self) -> Result<()> {
        let tx = self.tx.clone();
        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);
        
        // Spawn a thread for the !Send cpal stream
        std::thread::spawn(move || {
            let host = cpal::default_host();
            let device = match host.default_input_device() {
                Some(d) => d,
                None => {
                    eprintln!("No input device found");
                    return;
                }
            };
            
            let config: cpal::StreamConfig = match device.default_input_config() {
                Ok(c) => c.into(),
                Err(e) => {
                    eprintln!("Default input config error: {}", e);
                    return;
                }
            };
            
            let source_rate = config.sample_rate.0;
            let channels = config.channels as usize;
            let chunk_size = (source_rate / 50) as usize;
            
            let mut r24 = match Resampler::new(source_rate, 24000, chunk_size, channels) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Resampler init error: {}", e);
                    return;
                }
            };
            
            let tx_clone = tx.clone();
            let stream = match device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(samples_24k) = r24.process(data) {
                        let frame = Frame::Audio(AudioRawFrame {
                            samples: samples_24k,
                            sample_rate: 24000,
                            channels: 1,
                        });
                        let _ = tx_clone.try_send(frame);
                    }
                },
                |err| eprintln!("Capture error: {}", err),
                None
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Stream build error: {}", e);
                    return;
                }
            };
            
            if let Err(e) = stream.play() {
                eprintln!("Stream play error: {}", e);
                return;
            }
            
            // Wait for stop signal
            let _ = stop_rx.blocking_recv();
            drop(stream);
        });

        self.stream_handle = Some(stop_tx);
        Ok(())
    }

    fn stop_capture(&mut self) {
        if let Some(handle) = self.stream_handle.take() {
            let _ = handle.try_send(());
        }
    }
}

#[async_trait]
impl Processor for CaptureProcessor {
    async fn process(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        match frame {
            Frame::Control(ControlFrame::Start) => {
                self.start_capture()?;
                Ok(vec![])
            }
            Frame::Control(ControlFrame::Stop) => {
                self.stop_capture();
                Ok(vec![])
            }
            _ => Ok(vec![frame]),
        }
    }
}
