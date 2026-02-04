#![allow(dead_code)]

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_tungstenite::MaybeTlsStream;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const ASR_HOST: &str = "127.0.0.1:8765";
const TTS_HOST: &str = "127.0.0.1:8766";
const MAX_RETRIES: u32 = 5;
const RECONNECT_DELAY_MS: u64 = 1000;

#[derive(Debug, Deserialize, Clone, serde::Serialize)]
pub struct AsrResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub partial: String,
    #[serde(rename = "final")]
    pub final_text: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Serialize)]
struct TtsRequest {
    #[serde(rename = "type")]
    request_type: String,
    text: String,
    #[serde(rename = "text_id")]
    text_id: usize,
}

pub struct AsrClient {
    url: String,
}

impl AsrClient {
    pub fn new() -> Self {
        Self {
            url: format!("ws://{}", ASR_HOST),
        }
    }

    pub async fn connect(
        &self,
    ) -> Result<tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>> {
        let (ws_stream, _) = connect_async(&self.url).await?;
        Ok(ws_stream)
    }

    pub async fn send_audio_frame(
        &self,
        ws_stream: &mut tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>,
        audio_data: &[u8],
    ) -> Result<()> {
        ws_stream.send(Message::Binary(audio_data.to_vec())).await?;
        Ok(())
    }

    pub async fn receive_result(
        &self,
        ws_stream: &mut tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>,
    ) -> Result<Option<AsrResult>> {
        match ws_stream.next().await {
            Some(Ok(message)) => match message {
                Message::Text(text) => {
                    let result: AsrResult = serde_json::from_str(&text)?;
                    Ok(Some(result))
                }
                _ => Ok(None),
            },
            Some(Err(e)) => Err(anyhow::anyhow!("WebSocket error: {}", e)),
            None => Ok(None),
        }
    }
}

pub struct TtsClient {
    url: String,
}

impl TtsClient {
    pub fn new() -> Self {
        Self {
            url: format!("ws://{}", TTS_HOST),
        }
    }

    pub async fn connect(
        &self,
    ) -> Result<tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>> {
        let (ws_stream, _) = connect_async(&self.url).await?;
        Ok(ws_stream)
    }

    pub async fn send_text_chunk(
        &self,
        ws_stream: &mut tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>,
        text: &str,
        text_id: usize,
    ) -> Result<()> {
        let request = TtsRequest {
            request_type: "text_chunk".to_string(),
            text: text.to_string(),
            text_id,
        };

        let json = serde_json::to_string(&request)?;
        ws_stream.send(Message::Text(json)).await?;
        Ok(())
    }

    pub async fn receive_audio_frame(
        &self,
        ws_stream: &mut tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>,
    ) -> Result<Option<Vec<u8>>> {
        match ws_stream.next().await {
            Some(Ok(message)) => match message {
                Message::Binary(data) => Ok(Some(data)),
                _ => Ok(None),
            },
            Some(Err(e)) => Err(anyhow::anyhow!("WebSocket error: {}", e)),
            None => Ok(None),
        }
    }
}

pub async fn connect_with_retry<F, Fut>(
    connect_fn: F,
) -> Result<tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<
        Output = Result<tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>>,
    >,
{
    let mut attempts = 0;

    loop {
        match connect_fn().await {
            Ok(stream) => return Ok(stream),
            Err(e) if attempts < MAX_RETRIES => {
                eprintln!(
                    "Connection failed (attempt {}/{}): {}",
                    attempts + 1,
                    MAX_RETRIES,
                    e
                );
                attempts += 1;
                tokio::time::sleep(tokio::time::Duration::from_millis(RECONNECT_DELAY_MS)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

#[tauri::command]
pub async fn test_asr_connection() -> Result<String, String> {
    let client = AsrClient::new();

    match connect_with_retry(|| client.connect()).await {
        Ok(_) => Ok("ASR connection successful".to_string()),
        Err(e) => Err(format!("ASR connection failed: {}", e)),
    }
}

#[tauri::command]
pub async fn test_tts_connection() -> Result<String, String> {
    let client = TtsClient::new();

    match connect_with_retry(|| client.connect()).await {
        Ok(_) => Ok("TTS connection successful".to_string()),
        Err(e) => Err(format!("TTS connection failed: {}", e)),
    }
}
