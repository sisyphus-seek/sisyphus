use anyhow::Result;
use async_openai::{
    config::OpenAIConfig,
    types::{ChatCompletionRequestMessageArgs, CreateChatCompletionRequestArgs, Role as OpenAIRole},
    Client,
};
use futures::StreamExt;
use futures_util::SinkExt;
use serde::Serialize;
use std::env;
use std::sync::{Arc, Mutex};
use tauri::Emitter;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};

use crate::audio::playback::queue_playback_audio;
use crate::conversation::{ConversationState, Message, Role};

const MIN_CHUNK_TOKENS: usize = 20;
const TTS_HOST: &str = "ws://127.0.0.1:8766";

#[derive(Debug, Clone, Serialize)]
pub struct LlmChunk {
    pub content: String,
    pub is_complete: bool,
}

#[derive(Debug, Serialize)]
struct TtsRequest {
    #[serde(rename = "type")]
    request_type: String,
    text: String,
    text_id: usize,
}

pub struct LlmClient {
    client: Client<OpenAIConfig>,
    model: String,
}

impl LlmClient {
    pub fn new() -> Result<Self> {
        // 加载 API Key（支持两种环境变量名）
        let api_key = env::var("LLM_API_KEY")
            .or_else(|_| env::var("OPENAI_API_KEY"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "LLM API key not found in environment variables (LLM_API_KEY or OPENAI_API_KEY)"
                )
            })?;

        // 加载 Base URL（支持 OpenAI-compatible API）
        let base_url = env::var("LLM_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

        // 加载模型名称
        let model = env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string());

        // 配置 OpenAI 客户端
        let config = OpenAIConfig::default()
            .with_api_key(api_key)
            .with_api_base(base_url);

        let client = Client::with_config(config);

        Ok(Self { client, model })
    }

    pub async fn stream_completion(
        &self,
        messages: Vec<async_openai::types::ChatCompletionRequestMessage>,
    ) -> Result<(String, Vec<LlmChunk>)> {
        let mut full_response = String::new();
        let mut chunks: Vec<LlmChunk> = Vec::new();
        let mut current_chunk = String::new();
        let mut token_count = 0;
        let mut last_emit_time = std::time::Instant::now();

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .temperature(0.7)
            .max_tokens(1024_u16)
            .build()?;

        let mut stream = self.client.chat().create_stream(request).await?;

        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            full_response.push_str(content);
                            current_chunk.push_str(content);
                            token_count += content.split_whitespace().count().max(1);

                            let has_punctuation = content.contains('.')
                                || content.contains(',')
                                || content.contains('!')
                                || content.contains('?')
                                || content.contains('。')
                                || content.contains('，');

                            let elapsed_ms = last_emit_time.elapsed().as_millis() as u64;

                            let should_emit = token_count >= MIN_CHUNK_TOKENS
                                || (token_count >= 10 && has_punctuation)
                                || elapsed_ms >= 200;

                            if should_emit && !current_chunk.is_empty() {
                                chunks.push(LlmChunk {
                                    content: current_chunk.clone(),
                                    is_complete: false,
                                });

                                current_chunk.clear();
                                token_count = 0;
                                last_emit_time = std::time::Instant::now();
                            }
                        }
                    }
                }
                Err(e) => return Err(anyhow::anyhow!("LLM stream error: {}", e)),
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(LlmChunk {
                content: current_chunk,
                is_complete: true,
            });
        } else if let Some(last) = chunks.last_mut() {
            last.is_complete = true;
        }

        Ok((full_response, chunks))
    }
}

fn create_user_message(
    content: String,
) -> Result<async_openai::types::ChatCompletionRequestMessage> {
    Ok(ChatCompletionRequestMessageArgs::default()
        .role(OpenAIRole::User)
        .content(content)
        .build()?)
}

fn create_assistant_message(
    content: String,
) -> Result<async_openai::types::ChatCompletionRequestMessage> {
    Ok(ChatCompletionRequestMessageArgs::default()
        .role(OpenAIRole::Assistant)
        .content(content)
        .build()?)
}

fn convert_message_to_openai(
    msg: &Message,
) -> Result<async_openai::types::ChatCompletionRequestMessage> {
    match msg.role {
        Role::User => create_user_message(msg.content.clone()),
        Role::Assistant => create_assistant_message(msg.content.clone()),
    }
}

#[tauri::command]
pub async fn send_llm_request(
    state: tauri::State<'_, Arc<Mutex<LlmClient>>>,
    user_message: String,
) -> Result<String, String> {
    // Clone the client to release the lock before await
    let (client, model) = {
        let guard = state.lock().unwrap();
        (guard.client.clone(), guard.model.clone())
    };

    let user_msg =
        create_user_message(user_message).map_err(|e| format!("Failed to build message: {}", e))?;
    let messages = vec![user_msg];

    let temp_client = LlmClient { client, model };
    match temp_client.stream_completion(messages).await {
        Ok((full_response, _)) => Ok(full_response),
        Err(e) => Err(format!("LLM request failed: {}", e)),
    }
}

/// Stream LLM response with TTS integration
/// This function:
/// 1. Streams LLM response chunks in real-time
/// 2. Sends each chunk to TTS for synthesis immediately
/// 3. Queues received audio for playback
#[tauri::command]
pub async fn stream_llm_response(
    app: tauri::AppHandle,
    llm_state: tauri::State<'_, Arc<Mutex<LlmClient>>>,
    conv_state: tauri::State<'_, Arc<Mutex<ConversationState>>>,
    user_message: String,
) -> Result<(), String> {
    // Clone the client, model and history to release locks before await
    let (client, model, history) = {
        let llm_guard = llm_state.lock().unwrap();
        let conv_guard = conv_state.lock().unwrap();
        (
            llm_guard.client.clone(),
            llm_guard.model.clone(),
            conv_guard.get_history().clone(),
        )
    };

    let mut messages: Vec<async_openai::types::ChatCompletionRequestMessage> = history
        .iter()
        .map(|msg| convert_message_to_openai(msg))
        .collect::<Result<Vec<_>>>()
        .map_err(|e| format!("Failed to convert messages: {}", e))?;

    let user_msg = create_user_message(user_message)
        .map_err(|e| format!("Failed to build user message: {}", e))?;
    messages.push(user_msg);

    // Connect to TTS WebSocket
    let tts_conn = connect_async(TTS_HOST).await;
    let (mut ws_stream, _) = match tts_conn {
        Ok(conn) => conn,
        Err(e) => {
            eprintln!("Failed to connect to TTS: {}", e);
            let _ = app.emit(
                "voice_assistant:error",
                serde_json::json!({ "code": "TTS_CONNECTION_FAILED", "message": format!("{}", e) }),
            );
            return Err(format!("TTS connection failed: {}", e));
        }
    };

    // Emit state change to Thinking
    let _ = app.emit(
        "voice_assistant:state_changed",
        serde_json::json!({ "state": "Thinking" }),
    );

    // Build LLM request
    let request = CreateChatCompletionRequestArgs::default()
        .model(&model)
        .messages(messages)
        .temperature(0.7)
        .max_tokens(1024_u16)
        .build()
        .map_err(|e| format!("Failed to build request: {}", e))?;

    let mut llm_stream = client
        .chat()
        .create_stream(request)
        .await
        .map_err(|e| format!("Failed to create LLM stream: {}", e))?;

    let mut full_response = String::new();
    let mut current_chunk = String::new();
    let mut token_count = 0;
    let mut text_id = 0;
    let mut last_emit_time = std::time::Instant::now();
    let mut first_chunk = true;

    // Process LLM stream in real-time
    while let Some(result) = llm_stream.next().await {
        match result {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    if let Some(content) = &choice.delta.content {
                        // On first content, switch to Speaking state
                        if first_chunk {
                            first_chunk = false;
                            let _ = app.emit(
                                "voice_assistant:state_changed",
                                serde_json::json!({ "state": "Speaking" }),
                            );
                        }

                        full_response.push_str(content);
                        current_chunk.push_str(content);
                        token_count += content.split_whitespace().count().max(1);

                        let has_punctuation = content.contains('.')
                            || content.contains(',')
                            || content.contains('!')
                            || content.contains('?')
                            || content.contains('。')
                            || content.contains('，')
                            || content.contains('！')
                            || content.contains('？');

                        let elapsed_ms = last_emit_time.elapsed().as_millis() as u64;

                        // Emit chunk when: enough tokens, or punctuation, or time elapsed
                        let should_emit = token_count >= MIN_CHUNK_TOKENS
                            || (token_count >= 5 && has_punctuation)
                            || elapsed_ms >= 300;

                        if should_emit && !current_chunk.is_empty() {
                            let chunk = LlmChunk {
                                content: current_chunk.clone(),
                                is_complete: false,
                            };

                            // Emit to frontend immediately
                            let _ = app.emit("voice_assistant:assistant_response", &chunk);

                            // Send to TTS
                            let tts_request = TtsRequest {
                                request_type: "text_chunk".to_string(),
                                text: current_chunk.clone(),
                                text_id,
                            };

                            if let Ok(json) = serde_json::to_string(&tts_request) {
                                let _ = ws_stream.send(WsMessage::Text(json)).await;
                            }

                            text_id += 1;
                            current_chunk.clear();
                            token_count = 0;
                            last_emit_time = std::time::Instant::now();
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("LLM stream error: {}", e);
                break;
            }
        }
    }

    // Send remaining chunk
    if !current_chunk.is_empty() {
        let chunk = LlmChunk {
            content: current_chunk.clone(),
            is_complete: false,
        };
        let _ = app.emit("voice_assistant:assistant_response", &chunk);

        let tts_request = TtsRequest {
            request_type: "text_chunk".to_string(),
            text: current_chunk,
            text_id,
        };

        if let Ok(json) = serde_json::to_string(&tts_request) {
            let _ = ws_stream.send(WsMessage::Text(json)).await;
        }
    }

    // Send end signal to TTS
    let end_request = serde_json::json!({ "type": "end" });
    let _ = ws_stream
        .send(WsMessage::Text(end_request.to_string()))
        .await;

    // Receive audio from TTS and queue for playback
    while let Some(msg_result) = ws_stream.next().await {
        match msg_result {
            Ok(WsMessage::Binary(audio_data)) => {
                // Queue audio for playback
                if let Err(e) = queue_playback_audio(audio_data) {
                    eprintln!("Failed to queue audio: {}", e);
                }
            }
            Ok(WsMessage::Text(text)) => {
                // TTS might send status messages
                if let Ok(status) = serde_json::from_str::<serde_json::Value>(&text) {
                    if status.get("type").and_then(|v| v.as_str()) == Some("complete") {
                        break;
                    }
                }
            }
            Ok(WsMessage::Close(_)) => {
                break;
            }
            Err(e) => {
                eprintln!("TTS WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    // Close TTS connection
    let _ = ws_stream.close(None).await;

    // Emit final complete event (empty content, just signal completion)
    let _ = app.emit(
        "voice_assistant:assistant_response",
        LlmChunk {
            content: String::new(),
            is_complete: true,
        },
    );

    Ok(())
}
