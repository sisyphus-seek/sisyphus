use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub timestamp: u64,
}

impl Message {
    pub fn new(role: Role, content: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            role,
            content,
            timestamp,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationStatus {
    Idle,
    Listening,
    FinalizingASR,
    Thinking,
    Speaking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub id: usize,
    pub status: ConversationStatus,
    pub asr_segment_id: Option<usize>,
    pub llm_request_id: Option<String>,
    pub tts_stream_id: Option<usize>,
}

impl Turn {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            status: ConversationStatus::Idle,
            asr_segment_id: None,
            llm_request_id: None,
            tts_stream_id: None,
        }
    }
}

pub struct ConversationState {
    history: Vec<Message>,
    current_turn: Turn,
    next_turn_id: usize,
}

impl ConversationState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            current_turn: Turn::new(0),
            next_turn_id: 1,
        }
    }

    pub fn add_message(&mut self, message: Message) {
        self.history.push(message);
    }

    pub fn get_history(&self) -> &Vec<Message> {
        &self.history
    }

    pub fn start_new_turn(&mut self) {
        self.current_turn = Turn::new(self.next_turn_id);
        self.next_turn_id += 1;
    }

    #[allow(dead_code)]
    pub fn get_current_turn(&self) -> &Turn {
        &self.current_turn
    }

    pub fn set_turn_status(&mut self, status: ConversationStatus) {
        self.current_turn.status = status;
    }

    pub fn get_turn_status(&self) -> ConversationStatus {
        self.current_turn.status
    }

    #[allow(dead_code)]
    pub fn set_asr_segment_id(&mut self, id: usize) {
        self.current_turn.asr_segment_id = Some(id);
    }

    #[allow(dead_code)]
    pub fn set_llm_request_id(&mut self, id: String) {
        self.current_turn.llm_request_id = Some(id);
    }

    #[allow(dead_code)]
    pub fn set_tts_stream_id(&mut self, id: usize) {
        self.current_turn.tts_stream_id = Some(id);
    }

    pub fn transition_idle(&mut self) {
        self.set_turn_status(ConversationStatus::Idle);
    }

    pub fn transition_listening(&mut self) {
        self.start_new_turn();
        self.set_turn_status(ConversationStatus::Listening);
    }

    pub fn transition_finalizing_asr(&mut self) {
        self.set_turn_status(ConversationStatus::FinalizingASR);
    }

    pub fn transition_thinking(&mut self) {
        self.set_turn_status(ConversationStatus::Thinking);
    }

    pub fn transition_speaking(&mut self) {
        self.set_turn_status(ConversationStatus::Speaking);
    }

    #[allow(dead_code)]
    pub fn can_interrupt(&self) -> bool {
        matches!(
            self.current_turn.status,
            ConversationStatus::Speaking | ConversationStatus::Thinking
        )
    }

    pub fn is_active(&self) -> bool {
        !matches!(self.current_turn.status, ConversationStatus::Idle)
    }
}

#[tauri::command]
pub fn get_conversation_history(
    state: tauri::State<'_, Arc<Mutex<ConversationState>>>,
) -> Vec<Message> {
    state.lock().unwrap().get_history().clone()
}

#[tauri::command]
pub fn get_conversation_status(
    state: tauri::State<'_, Arc<Mutex<ConversationState>>>,
) -> ConversationStatus {
    state.lock().unwrap().get_turn_status()
}

#[tauri::command]
pub fn is_conversation_active(state: tauri::State<'_, Arc<Mutex<ConversationState>>>) -> bool {
    state.lock().unwrap().is_active()
}

#[tauri::command]
pub fn transition_conversation_status(
    state: tauri::State<'_, Arc<Mutex<ConversationState>>>,
    new_status: ConversationStatus,
) {
    let mut conv_state = state.lock().unwrap();

    match new_status {
        ConversationStatus::Idle => conv_state.transition_idle(),
        ConversationStatus::Listening => conv_state.transition_listening(),
        ConversationStatus::FinalizingASR => conv_state.transition_finalizing_asr(),
        ConversationStatus::Thinking => conv_state.transition_thinking(),
        ConversationStatus::Speaking => conv_state.transition_speaking(),
    }
}

#[tauri::command]
pub fn add_user_message(state: tauri::State<'_, Arc<Mutex<ConversationState>>>, content: String) {
    let mut conv_state = state.lock().unwrap();
    let message = Message::new(Role::User, content);
    conv_state.add_message(message);
}

#[tauri::command]
pub fn add_assistant_message(
    state: tauri::State<'_, Arc<Mutex<ConversationState>>>,
    content: String,
) {
    let mut conv_state = state.lock().unwrap();
    let message = Message::new(Role::Assistant, content);
    conv_state.add_message(message);
}
