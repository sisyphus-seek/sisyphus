use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate, SupportedStreamConfigRange};
use futures_util::{SinkExt, StreamExt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tauri::Emitter;
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const TARGET_SAMPLE_RATE: u32 = 16000;
const ASR_HOST: &str = "ws://127.0.0.1:8765";
const AUDIO_FRAME_SIZE: usize = 640; // 20ms at 16kHz mono = 320 samples * 2 bytes

#[derive(Clone, serde::Serialize)]
pub struct VadEvent {
    pub status: String,
}

#[derive(Clone, serde::Serialize)]
pub struct AudioLevel {
    pub level: f32,
}

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug)]
pub struct AsrTranscript {
    pub partial: String,
    #[serde(rename = "final")]
    pub final_text: Option<String>,
    pub confidence: f32,
}

static RECORDING: AtomicBool = AtomicBool::new(false);
static AUDIO_TX: OnceLock<Arc<Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>>> = OnceLock::new();
static DONE_TX: OnceLock<Arc<Mutex<Option<oneshot::Sender<()>>>>> = OnceLock::new();

fn get_audio_tx() -> &'static Arc<Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>> {
    AUDIO_TX.get_or_init(|| Arc::new(Mutex::new(None)))
}

fn get_done_tx() -> &'static Arc<Mutex<Option<oneshot::Sender<()>>>> {
    DONE_TX.get_or_init(|| Arc::new(Mutex::new(None)))
}

pub struct AudioCapture;

/// Audio configuration for capture
struct CaptureConfig {
    sample_rate: u32,
    channels: u16,
}

impl AudioCapture {
    fn get_default_input_device() -> Result<Device> {
        let host = cpal::default_host();

        let default_device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device found"))?;

        Ok(default_device)
    }

    /// Find the best supported configuration for the device
    fn get_supported_config(device: &Device) -> Result<CaptureConfig> {
        let supported_configs: Vec<SupportedStreamConfigRange> = device
            .supported_input_configs()
            .map_err(|e| anyhow::anyhow!("Failed to get supported configs: {}", e))?
            .collect();

        if supported_configs.is_empty() {
            return Err(anyhow::anyhow!("No supported input configurations"));
        }

        // Try to find a config that supports our target sample rate
        // Prefer mono, but accept stereo if needed
        let target_rate = SampleRate(TARGET_SAMPLE_RATE);

        // First, try to find exact match with mono
        for config in &supported_configs {
            if config.channels() == 1
                && config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(CaptureConfig {
                    sample_rate: TARGET_SAMPLE_RATE,
                    channels: 1,
                });
            }
        }

        // Try stereo with target sample rate
        for config in &supported_configs {
            if config.channels() == 2
                && config.min_sample_rate() <= target_rate
                && config.max_sample_rate() >= target_rate
            {
                return Ok(CaptureConfig {
                    sample_rate: TARGET_SAMPLE_RATE,
                    channels: 2,
                });
            }
        }

        // Fall back to any supported config (prefer lower sample rates and mono)
        let best_config = supported_configs
            .iter()
            .min_by_key(|c| (c.channels(), c.min_sample_rate().0))
            .unwrap();

        let sample_rate = if best_config.min_sample_rate().0 <= 48000 && best_config.max_sample_rate().0 >= 48000 {
            48000
        } else if best_config.min_sample_rate().0 <= 44100 && best_config.max_sample_rate().0 >= 44100 {
            44100
        } else {
            best_config.min_sample_rate().0
        };

        Ok(CaptureConfig {
            sample_rate,
            channels: best_config.channels(),
        })
    }

    fn calculate_audio_level(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let rms: f32 = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        rms.sqrt()
    }

    pub fn is_recording() -> bool {
        RECORDING.load(Ordering::SeqCst)
    }
}

/// Simple linear resampling from source rate to target rate (16kHz)
fn resample_to_16k(samples: &[f32], source_rate: u32, channels: u16) -> Vec<f32> {
    // First, convert stereo to mono if needed
    let mono_samples: Vec<f32> = if channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    (chunk[0] + chunk[1]) / 2.0
                } else {
                    chunk[0]
                }
            })
            .collect()
    } else {
        samples.to_vec()
    };

    // If already at target rate, return as-is
    if source_rate == TARGET_SAMPLE_RATE {
        return mono_samples;
    }

    // Linear interpolation resampling
    let ratio = source_rate as f64 / TARGET_SAMPLE_RATE as f64;
    let output_len = (mono_samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let frac = (src_idx - src_idx_floor as f64) as f32;

        let sample = if src_idx_floor + 1 < mono_samples.len() {
            mono_samples[src_idx_floor] * (1.0 - frac) + mono_samples[src_idx_floor + 1] * frac
        } else if src_idx_floor < mono_samples.len() {
            mono_samples[src_idx_floor]
        } else {
            0.0
        };

        output.push(sample);
    }

    output
}

async fn run_asr_session(
    app: tauri::AppHandle,
    mut audio_rx: mpsc::UnboundedReceiver<Vec<u8>>,
) {
    println!("Starting ASR session loop...");
    // Connect to ASR WebSocket
    let ws_result = connect_async(ASR_HOST).await;

    let (mut ws_stream, _) = match ws_result {
        Ok(conn) => conn,
        Err(e) => {
            eprintln!("Failed to connect to ASR: {}", e);
            let _ = app.emit(
                "voice_assistant:error",
                serde_json::json!({ "code": "ASR_CONNECTION_FAILED", "message": format!("{}", e) }),
            );
            // Signal done even on failure
            if let Some(tx) = get_done_tx().lock().unwrap().take() {
                let _ = tx.send(());
            }
            return;
        }
    };

    let mut audio_buffer = Vec::with_capacity(AUDIO_FRAME_SIZE);

    loop {
        tokio::select! {
            // Receive audio from capture and send to ASR
            res = audio_rx.recv() => {
                match res {
                    Some(audio_data) => {
                        audio_buffer.extend_from_slice(&audio_data);

                        // Send complete frames to ASR
                        while audio_buffer.len() >= AUDIO_FRAME_SIZE {
                            let frame: Vec<u8> = audio_buffer.drain(..AUDIO_FRAME_SIZE).collect();
                            if let Err(e) = ws_stream.send(Message::Binary(frame)).await {
                                eprintln!("Failed to send audio to ASR: {}", e);
                                break;
                            }
                        }
                    }
                    None => {
                        // Channel closed, process stop
                        println!("Audio channel closed in ASR task. Finalizing...");
                        
                        // Signal ASR to flush and finish
                        let _ = ws_stream.send(Message::Text(serde_json::json!({ "type": "stop" }).to_string())).await;
                        
                        // Send any remaining buffered audio
                        if !audio_buffer.is_empty() {
                            println!("Sending remaining buffer ({} bytes) to ASR...", audio_buffer.len());
                            let _ = ws_stream.send(Message::Binary(audio_buffer.clone())).await;
                        }
                        
                        // Wait a bit for the final transcript
                        println!("Waiting up to 1000ms for final ASR result...");
                        tokio::select! {
                            msg = ws_stream.next() => {
                                if let Some(Ok(Message::Text(text))) = msg {
                                    println!("ASR received final text during stop: {}", text);
                                    if let Ok(result) = serde_json::from_str::<serde_json::Value>(&text) {
                                        if let Some(final_text) = result.get("final").and_then(|v| v.as_str()) {
                                            let transcript = AsrTranscript {
                                                partial: "".to_string(),
                                                final_text: Some(final_text.to_string()),
                                                confidence: 0.95,
                                            };
                                            let _ = app.emit("voice_assistant:user_transcript", &transcript);
                                        }
                                    }
                                }
                            }
                            _ = tokio::time::sleep(std::time::Duration::from_millis(1000)) => {
                                println!("Timeout waiting for final ASR result");
                            }
                        }

                        // Close the WebSocket gracefully
                        let _ = ws_stream.close(None).await;
                        break;
                    }
                }
            }

            // Receive results from ASR
            res = ws_stream.next() => {
                match res {
                    Some(Ok(Message::Text(text))) => {
                        println!("ASR received text: {}", text);
                        // Parse ASR result
                        if let Ok(result) = serde_json::from_str::<serde_json::Value>(&text) {
                            let partial = result.get("partial")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();

                            let final_text = result.get("final")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());
                            
                            if let Some(ref f) = final_text {
                                println!("ASR FINAL identified: {}", f);
                            }

                            let confidence = result.get("confidence")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32;

                            let transcript = AsrTranscript {
                                partial,
                                final_text: final_text.clone(),
                                confidence,
                            };

                            let _ = app.emit("voice_assistant:user_transcript", &transcript);
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        println!("ASR connection closed");
                        break;
                    }
                    Some(Err(e)) => {
                        eprintln!("ASR WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    // Signal completion
    if let Some(tx) = get_done_tx().lock().unwrap().take() {
        let _ = tx.send(());
    }
    println!("ASR session task finished.");
}

#[tauri::command]
pub fn start_recording(app: tauri::AppHandle) -> Result<(), String> {
    if RECORDING.load(Ordering::SeqCst) {
        return Err("Already recording".to_string());
    }

    RECORDING.store(true, Ordering::SeqCst);

    let device =
        AudioCapture::get_default_input_device().map_err(|e| format!("Device error: {}", e))?;

    // Get supported configuration
    let capture_config = AudioCapture::get_supported_config(&device)
        .map_err(|e| format!("Config error: {}", e))?;

    println!(
        "Audio capture config: {}Hz, {} channels",
        capture_config.sample_rate, capture_config.channels
    );

    let config = cpal::StreamConfig {
        channels: capture_config.channels,
        sample_rate: cpal::SampleRate(capture_config.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let source_rate = capture_config.sample_rate;
    let source_channels = capture_config.channels;

    // Create channel for audio data
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<u8>>();
    let (done_tx, _done_rx) = oneshot::channel::<()>();

    // Store the sender for the capture callback
    {
        let mut tx_guard = get_audio_tx().lock().unwrap();
        *tx_guard = Some(audio_tx);
    }
    
    // Store the done sender
    {
        let mut done_guard = get_done_tx().lock().unwrap();
        *done_guard = Some(done_tx);
    }

    let app_handle = app.clone();
    let app_handle_for_stream = app.clone();

    // Spawn ASR session task
    tauri::async_runtime::spawn(async move {
        run_asr_session(app_handle_for_stream, audio_rx).await;
    });

    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !RECORDING.load(Ordering::SeqCst) {
                    return;
                }

                // Calculate and emit audio level
                let audio_level = AudioCapture::calculate_audio_level(data);
                let _ = app_handle.emit(
                    "voice_assistant:audio_level",
                    AudioLevel { level: audio_level },
                );

                // Resample to 16kHz mono if needed
                let resampled = resample_to_16k(data, source_rate, source_channels);

                // Convert f32 samples to i16 PCM bytes and send to ASR
                let pcm_bytes: Vec<u8> = resampled
                    .iter()
                    .flat_map(|&sample| {
                        let clamped = sample.max(-1.0).min(1.0);
                        let i16_sample = (clamped * 32767.0) as i16;
                        i16_sample.to_le_bytes()
                    })
                    .collect();

                // Send audio to ASR task
                let tx_ref = get_audio_tx().lock().unwrap();
                if let Some(tx) = tx_ref.as_ref() {
                    let _ = tx.send(pcm_bytes);
                }
            },
            move |err| {
                eprintln!("Audio capture error: {}", err);
            },
            None,
        )
        .map_err(|e| format!("Failed to build stream: {}", e))?;

    // Keep reference to stream to prevent drop
    Box::leak(Box::new(stream.play().map_err(|e| format!("Failed to play stream: {}", e))?));
    // Actually stream.play() returns Result<(), PlayStreamError>, so we need to call it and then leak the stream
    // Correcting:
    let _ = stream.play();
    Box::leak(Box::new(stream));

    app.emit(
        "voice_assistant:state_changed",
        serde_json::json!({ "state": "Listening" }),
    )
    .map_err(|e| format!("Failed to emit state: {}", e))?;

    app.emit(
        "voice_assistant:vad_status",
        VadEvent {
            status: "speech_start".to_string(),
        },
    )
    .map_err(|e| format!("Failed to emit event: {}", e))?;

    Ok(())
}

#[tauri::command]
pub async fn stop_recording(app: tauri::AppHandle) -> Result<(), String> {
    println!("stop_recording command received.");
    RECORDING.store(false, Ordering::SeqCst);
    println!("Recording marked as stopped in atomic flag.");

    // Close the audio channel to signal ASR task to finish
    {
        let mut tx_guard = get_audio_tx().lock().unwrap();
        *tx_guard = None;
    }
    println!("Audio channel closed. Waiting for ASR task to finalize...");

    app.emit(
        "voice_assistant:state_changed",
        serde_json::json!({ "state": "FinalizingASR" }),
    )
    .map_err(|e| format!("Failed to emit state: {}", e))?;
    println!("Emitted FinalizingASR state.");

    // Simple sleep to allow ASR task to process final results
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;
    println!("Stop recording wait finished.");

    app.emit(
        "voice_assistant:vad_status",
        VadEvent {
            status: "speech_end".to_string(),
        },
    )
    .map_err(|e| format!("Failed to emit event: {}", e))?;

    Ok(())
}

#[tauri::command]
pub fn is_recording() -> bool {
    AudioCapture::is_recording()
}
