use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioRawFrame {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFrame {
    pub text: String,
    pub is_final: bool,
    pub timestamp: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlFrame {
    Start,
    Stop,
    Cancel,
    Metadata { key: String, value: String },
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Frame {
    Audio(AudioRawFrame),
    Text(TextFrame),
    Control(ControlFrame),
}
