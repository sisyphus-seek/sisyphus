use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate, SupportedStreamConfigRange};
use std::collections::VecDeque;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tauri::Emitter;

fn get_source_sample_rate() -> u32 {
    env::var("AUDIO_SAMPLE_RATE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(16000)
}
const JITTER_BUFFER_FRAMES: usize = 5;
const DRAIN_TIMEOUT_CALLBACKS: u32 = 50; // Wait ~50 callbacks (~1 sec) before stopping

static PLAYING: AtomicBool = AtomicBool::new(false);
static STREAM_ACTIVE: AtomicBool = AtomicBool::new(false);
static AUDIO_QUEUE: OnceLock<Arc<Mutex<VecDeque<Vec<u8>>>>> = OnceLock::new();
static APP_HANDLE: OnceLock<Arc<Mutex<Option<tauri::AppHandle>>>> = OnceLock::new();
static PLAYBACK_COMPLETE_FLAG: AtomicBool = AtomicBool::new(false);
static DRAIN_COUNTER: OnceLock<Arc<Mutex<u32>>> = OnceLock::new();

fn get_drain_counter() -> &'static Arc<Mutex<u32>> {
    DRAIN_COUNTER.get_or_init(|| Arc::new(Mutex::new(0)))
}

fn get_queue() -> &'static Arc<Mutex<VecDeque<Vec<u8>>>> {
    AUDIO_QUEUE.get_or_init(|| Arc::new(Mutex::new(VecDeque::new())))
}

fn get_app_handle() -> &'static Arc<Mutex<Option<tauri::AppHandle>>> {
    APP_HANDLE.get_or_init(|| Arc::new(Mutex::new(None)))
}

pub struct AudioPlayback;

/// Playback configuration
struct PlaybackConfig {
    sample_rate: u32,
    channels: u16,
}

impl AudioPlayback {
    fn get_default_output_device() -> Result<Device> {
        let host = cpal::default_host();

        let default_device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No default output device found"))?;

        Ok(default_device)
    }

    /// Find the best supported output configuration
    fn get_supported_config(device: &Device) -> Result<PlaybackConfig> {
        let supported_configs: Vec<SupportedStreamConfigRange> = device
            .supported_output_configs()
            .map_err(|e| anyhow::anyhow!("Failed to get supported configs: {}", e))?
            .collect();

        if supported_configs.is_empty() {
            return Err(anyhow::anyhow!("No supported output configurations"));
        }

        // Try to find config supporting common sample rates
        // Prefer stereo for output (most devices support it)
        let target_rates = [48000u32, 44100, 22050, 16000];

        for &rate in &target_rates {
            let target_rate = SampleRate(rate);

            // Try stereo first (more commonly supported)
            for config in &supported_configs {
                if config.channels() == 2
                    && config.min_sample_rate() <= target_rate
                    && config.max_sample_rate() >= target_rate
                {
                    return Ok(PlaybackConfig {
                        sample_rate: rate,
                        channels: 2,
                    });
                }
            }

            // Try mono
            for config in &supported_configs {
                if config.channels() == 1
                    && config.min_sample_rate() <= target_rate
                    && config.max_sample_rate() >= target_rate
                {
                    return Ok(PlaybackConfig {
                        sample_rate: rate,
                        channels: 1,
                    });
                }
            }
        }

        // Fall back to any supported config
        let best_config = supported_configs
            .iter()
            .max_by_key(|c| c.max_sample_rate().0)
            .unwrap();

        let sample_rate = if best_config.max_sample_rate().0 >= 48000 {
            48000
        } else if best_config.max_sample_rate().0 >= 44100 {
            44100
        } else {
            best_config.max_sample_rate().0
        };

        Ok(PlaybackConfig {
            sample_rate,
            channels: best_config.channels(),
        })
    }

    pub fn is_playing() -> bool {
        PLAYING.load(Ordering::SeqCst)
    }
}

/// Resample from 16kHz mono to target rate and channels
fn resample_for_playback(
    input: &[i16],
    source_rate: u32,
    target_rate: u32,
    target_channels: u16,
) -> Vec<f32> {
    // First, convert to f32
    let mono_f32: Vec<f32> = input.iter().map(|&s| s as f32 / 32768.0).collect();

    // Resample if needed
    let resampled = if target_rate != source_rate {
        let ratio = target_rate as f64 / source_rate as f64;
        let output_len = (mono_f32.len() as f64 * ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 / ratio;
            let src_idx_floor = src_idx.floor() as usize;
            let frac = (src_idx - src_idx_floor as f64) as f32;

            let sample = if src_idx_floor + 1 < mono_f32.len() {
                mono_f32[src_idx_floor] * (1.0 - frac) + mono_f32[src_idx_floor + 1] * frac
            } else if src_idx_floor < mono_f32.len() {
                mono_f32[src_idx_floor]
            } else {
                0.0
            };

            output.push(sample);
        }
        output
    } else {
        mono_f32
    };

    // Convert to stereo if needed
    if target_channels == 2 {
        resampled
            .iter()
            .flat_map(|&s| [s, s])
            .collect()
    } else {
        resampled
    }
}

fn bytes_to_i16_samples(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

static DEBUG_FRAME_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[tauri::command]
pub fn queue_playback_audio(audio_data: Vec<u8>) -> Result<(), String> {
    // Debug: print info for first few frames
    let frame_num = DEBUG_FRAME_COUNT.fetch_add(1, Ordering::SeqCst);
    if frame_num < 3 {
        // Show first few samples as i16
        let samples = bytes_to_i16_samples(&audio_data[..audio_data.len().min(20)]);
        println!(
            "TTS frame {}: {} bytes, first samples: {:?}",
            frame_num,
            audio_data.len(),
            &samples[..samples.len().min(5)]
        );
    }

    // Reset drain counter when new audio arrives
    {
        let mut counter = get_drain_counter().lock().unwrap();
        *counter = 0;
    }

    let queue = get_queue();
    let mut q = queue.lock().unwrap();
    q.push_back(audio_data);

    // Only start if we have enough frames AND no stream is active yet
    let should_start = q.len() >= JITTER_BUFFER_FRAMES && !STREAM_ACTIVE.load(Ordering::SeqCst);
    drop(q);

    if should_start {
        start_playback_internal()?;
    }

    Ok(())
}

fn start_playback_internal() -> Result<(), String> {
    // Check if stream is already active (not just playing)
    if STREAM_ACTIVE.load(Ordering::SeqCst) {
        return Ok(());
    }

    STREAM_ACTIVE.store(true, Ordering::SeqCst);
    PLAYING.store(true, Ordering::SeqCst);
    PLAYBACK_COMPLETE_FLAG.store(false, Ordering::SeqCst);

    let device =
        AudioPlayback::get_default_output_device().map_err(|e| format!("Device error: {}", e))?;

    // Get supported configuration
    let playback_config = AudioPlayback::get_supported_config(&device)
        .map_err(|e| format!("Config error: {}", e))?;

    println!(
        "Audio playback config: {}Hz, {} channels (source {}Hz)",
        playback_config.sample_rate,
        playback_config.channels,
        get_source_sample_rate()
    );

    let config = cpal::StreamConfig {
        channels: playback_config.channels,
        sample_rate: cpal::SampleRate(playback_config.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let source_rate = get_source_sample_rate();
    let target_rate = playback_config.sample_rate;
    let target_channels = playback_config.channels;

    let queue = get_queue().clone();
    let drain_counter = get_drain_counter().clone();

    // Buffer for resampled audio
    let resampled_buffer: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
    let resampled_buffer_clone = resampled_buffer.clone();

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // Initialize all samples to silence
                for sample in data.iter_mut() {
                    *sample = 0.0;
                }

                if !PLAYING.load(Ordering::SeqCst) {
                    return;
                }

                // First, try to use resampled buffer
                let mut resampled = resampled_buffer_clone.lock().unwrap();
                let mut output_idx = 0;
                let mut had_audio = false;

                // Drain from resampled buffer first
                while !resampled.is_empty() && output_idx < data.len() {
                    if let Some(sample) = resampled.pop_front() {
                        data[output_idx] = sample;
                        output_idx += 1;
                        had_audio = true;
                    }
                }

                // If we need more samples, process from queue
                let mut queue = queue.lock().unwrap();
                while !queue.is_empty() && output_idx < data.len() {
                    if let Some(audio_bytes) = queue.pop_front() {
                        let samples = bytes_to_i16_samples(&audio_bytes);

                        // Resample to target format
                        let resampled_samples = resample_for_playback(
                            &samples,
                            source_rate,
                            target_rate,
                            target_channels,
                        );

                        for sample in resampled_samples {
                            if output_idx < data.len() {
                                data[output_idx] = sample;
                                output_idx += 1;
                                had_audio = true;
                            } else {
                                // Store remaining in buffer
                                resampled.push_back(sample);
                            }
                        }
                    }
                }

                // Check if we've finished playing all audio
                if queue.is_empty() && resampled.is_empty() && !had_audio {
                    // Increment drain counter when no audio is available
                    let mut counter = drain_counter.lock().unwrap();
                    *counter += 1;

                    // Only signal completion after timeout period
                    if *counter >= DRAIN_TIMEOUT_CALLBACKS && PLAYING.load(Ordering::SeqCst) {
                        if !PLAYBACK_COMPLETE_FLAG.swap(true, Ordering::SeqCst) {
                            PLAYING.store(false, Ordering::SeqCst);
                        }
                    }
                }
            },
            move |err| {
                eprintln!("Audio playback error: {}", err);
            },
            None,
        )
        .map_err(|e| format!("Failed to build stream: {}", e))?;

    stream
        .play()
        .map_err(|e| format!("Failed to play stream: {}", e))?;

    // Leak the stream to keep it alive
    Box::leak(Box::new(stream));

    // Start a monitoring task to emit events when playback completes
    if let Some(app) = get_app_handle().lock().unwrap().as_ref() {
        let app_clone = app.clone();
        tauri::async_runtime::spawn(async move {
            // Wait for playback to complete
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                if PLAYBACK_COMPLETE_FLAG.load(Ordering::SeqCst) {
                    // Reset stream active flag so new playback can start
                    STREAM_ACTIVE.store(false, Ordering::SeqCst);

                    // Emit playback ended and state change
                    let _ = app_clone.emit("voice_assistant:playback_ended", ());
                    let _ = app_clone.emit(
                        "voice_assistant:state_changed",
                        serde_json::json!({ "state": "Idle" }),
                    );
                    break;
                }

                if !PLAYING.load(Ordering::SeqCst) && !STREAM_ACTIVE.load(Ordering::SeqCst) {
                    break;
                }
            }
        });
    }

    Ok(())
}

#[tauri::command]
pub fn start_playback(app: tauri::AppHandle) -> Result<(), String> {
    // Store the app handle for later use
    {
        let mut handle = get_app_handle().lock().unwrap();
        *handle = Some(app.clone());
    }

    start_playback_internal()?;

    app.emit("voice_assistant:playback_started", ())
        .map_err(|e| format!("Failed to emit event: {}", e))?;

    Ok(())
}

#[tauri::command]
pub fn stop_playback(app: tauri::AppHandle) -> Result<(), String> {
    PLAYING.store(false, Ordering::SeqCst);
    STREAM_ACTIVE.store(false, Ordering::SeqCst);
    get_queue().lock().unwrap().clear();

    // Reset counters
    *get_drain_counter().lock().unwrap() = 0;
    DEBUG_FRAME_COUNT.store(0, Ordering::SeqCst);

    app.emit("voice_assistant:playback_ended", ())
        .map_err(|e| format!("Failed to emit event: {}", e))?;

    app.emit(
        "voice_assistant:state_changed",
        serde_json::json!({ "state": "Idle" }),
    )
    .map_err(|e| format!("Failed to emit state: {}", e))?;

    Ok(())
}

#[tauri::command]
pub fn is_playback_active() -> bool {
    AudioPlayback::is_playing()
}

/// Initialize playback with app handle for event emission
/// This should be called at app startup
pub fn init_playback(app: tauri::AppHandle) {
    let mut handle = get_app_handle().lock().unwrap();
    *handle = Some(app);
}
