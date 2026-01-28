# Tauri Voice Assistant - Simplified Execution Plan

## Overview

**Objective**: Build a desktop voice assistant using Tauri + Rust + React with local GLM-ASR and Qwen3-TTS models, OpenAI-compatible LLM, achieving sub-500ms response latency.

**Architecture**: Hybrid Rust/Python approach
- **Rust**: Audio capture/playback, VAD, WebSocket clients, orchestration, LLM streaming
- **Python**: Long-lived inference service for ASR + TTS via WebSocket
- **React**: UI for conversation display and status indication
- **IPC**: WebSocket with binary PCM16 audio + JSON control messages

**Scope**: Core functional implementation (MVP focus, optimization in later phases)

---

## Task List

### Phase 1: Project Setup

- [ ] 1. Initialize Git Repository

  **What to do**:
  - Create git repository: `git init` in appropriate directory
  - Create `.gitignore` for node_modules, target, __pycache__, venv, .env, .DS_Store
  - Create initial commit: `git add . && git commit -m "chore: initialize repository"`

  **Success criteria**:
  - [ ] `git status` shows clean working directory
  - [ ] `.gitignore` exists with common patterns

---

- [ ] 2. Create Tauri v2 + React Project

  **What to do**:
  - Install Tauri CLI: `npm install -g @tauri-apps/cli` or `cargo install tauri-cli`
  - Create project: `npm create tauri-app@latest` or `cargo tauri init`
  - Select React + TypeScript template
  - Verify structure: `src-tauri/` (Rust), `src/` (React), `package.json`, `tauri.conf.json`
  - Test dev server: `npm run tauri dev` (verify app opens)

  **Success criteria**:
  - [ ] Project created with Tauri v2 + React structure
  - [ ] `npm run tauri dev` starts successfully
  - [ ] Default Tauri window opens in desktop

---

### Phase 2: Python Inference Service

- [ ] 3. Setup Python Environment

  **What to do**:
  - Create `inference/` directory
  - Create `requirements.txt`:
    ```
    websockets>=12.0
    transformers>=4.30.0
    torch>=2.0.0
    numpy>=1.24.0
    librosa>=0.10.0  # For audio processing
    ```
  - Create virtual environment: `python -m venv venv`
  - Install dependencies: `source venv/bin/activate && pip install -r requirements.txt`
  - Test imports: `python -c "import transformers, torch, websockets, librosa"`

  **Success criteria**:
  - [ ] Virtual environment created
  - [ ] All dependencies installed without errors
  - [ ] Test imports succeed

---

- [ ] 4. Implement ASR WebSocket Service

  **What to do**:
  - Create `inference/asr_service.py`:
    - WebSocket server on port 8765
    - Load GLM-ASR-Nano-2512 model from HuggingFace or local path
    - Accept binary PCM16 audio frames via WebSocket (640 bytes per 20ms frame @ 16kHz)
    - Accumulate audio buffers (2-3 second windows with overlap)
    - Run inference on accumulated audio
    - Return partial and final transcripts as JSON messages
    - Message format: `{"type": "asr_result", "partial": "text", "final": "text", "confidence": 0.95}`
  - Create `inference/test_asr.py` for manual testing

  **Success criteria**:
  - [ ] ASR service starts on port 8765
  - [ ] Accepts binary audio frames and processes them
  - [ ] Returns partial and final transcripts correctly
  - [ ] Manual test produces reasonable transcripts

---

- [ ] 5. Implement TTS WebSocket Service

  **What to do**:
  - Create `inference/tts_service.py`:
    - WebSocket server on port 8766
    - Load Qwen3-TTS model from HuggingFace or local path
    - Use `qwen-tts` package or `transformers.pipeline("text-to-speech")`
    - Accept text chunks via WebSocket as JSON messages
    - Synthesize audio in chunks (streaming)
    - Return binary PCM16 audio frames via WebSocket (raw bytes, no base64)
    - Message format: `{"type": "text_chunk", "text": "hello", "text_id": 1}`
    - Output sample rate: 16kHz (resample from 24k/48k if needed)
  - Create `inference/test_tts.py` for manual testing

  **Success criteria**:
  - [ ] TTS service starts on port 8766
  - [ ] Accepts text chunks and synthesizes audio
  - [ ] Streams binary audio frames back correctly
  - [ ] Manual test produces audible speech

---

- [ ] 6. Create Inference Orchestration Script

  **What to do**:
  - Create `inference/run_inference.py`:
    - Start both ASR and TTS WebSocket servers concurrently using asyncio
    - Handle graceful shutdown (SIGTERM/SIGINT)
    - Log startup messages clearly
  - Create simple health check or startup verification

  **Success criteria**:
  - [ ] Both ASR and TTS servers start from single script
  - [ ] Shutdown handlers work (Ctrl+C stops services cleanly)
  - [ ] Logs clearly show service status

---

### Phase 3: Rust Backend

- [ ] 7. Setup Rust Dependencies

  **What to do**:
  - Update `src-tauri/Cargo.toml`:
    ```toml
    [dependencies]
    tokio = { version = "1.35", features = ["full"] }
    tokio-tungstenite = { version = "0.21", features = ["native-tls"] }
    serde = { version = "1.0", features = ["derive"] }
    serde_json = "1.0"
    async-openai = "0.14"
    silero-vad-rs = "0.1"
    cpal = "0.15"
    anyhow = "1.0"
    ```
  - Run `cargo build` to verify

  **Success criteria**:
  - [ ] `Cargo.toml` updated with all dependencies
  - [ ] `cargo build` succeeds without errors

---

- [ ] 8. Implement Audio Capture with VAD

  **What to do**:
  - Create `src-tauri/src/audio/capture.rs`:
    - Microphone capture using cpal (16kHz mono, 16-bit PCM)
    - Audio buffer management
    - VAD integration with silero-vad-rs:
      - Process audio chunks (every 20ms)
      - Emit VAD events: SpeechStart, SpeechEnd, Silence
      - Configure hangover time (150-300ms)
    - Create Tauri commands: `start_recording()`, `stop_recording()`
    - Emit Tauri events for UI: `vad_status`, `audio_level`
  - Configure macOS Info.plist for microphone permissions
  - Configure Windows tauri.conf.json for mic access

  **Success criteria**:
  - [ ] Microphone capture works (no permission errors)
  - [ ] VAD correctly detects speech start and end
  - [ ] Tauri events emitted for VAD status
  - [ ] Commands callable from frontend

---

- [ ] 9. Implement WebSocket Clients

  **What to do**:
  - Create `src-tauri/src/inference/client.rs`:
    - ASR WebSocket client:
      - Connect to `ws://127.0.0.1:8765`
      - Send binary PCM16 audio frames (640 bytes per 20ms frame)
      - Receive ASR results as JSON messages
      - Handle reconnection logic
    - TTS WebSocket client:
      - Connect to `ws://127.0.0.1:8766`
      - Send text chunks as JSON: `{"type": "text_chunk", "text": "...", "text_id": 123}`
      - Receive binary PCM16 audio frames (raw bytes)
      - Handle streaming responses
    - Error handling: Connection failures, service unavailable

  **Success criteria**:
  - [ ] ASR client connects and sends binary audio frames
  - [ ] ASR client receives and parses JSON transcript results
  - [ ] TTS client connects and sends JSON text chunks
  - [ ] TTS client receives and buffers binary audio frames
  - [ ] Reconnection logic implemented

---

- [ ] 10. Implement Audio Playback

  **What to do**:
  - Create `src-tauri/src/audio/playback.rs`:
    - Audio playback using cpal (16kHz mono, 16-bit PCM)
    - Jitter buffer (100-200ms) to smooth TTS chunk transitions
    - Queue audio chunks received from TTS WebSocket
    - Play chunks sequentially with proper timing (nextStartTime pattern)
    - Stop/cancel playback on interruption
    - Emit Tauri events: `playback_started`, `playback_ended`

  **Success criteria**:
  - [ ] Audio playback works (audible output)
  - [ ] Jitter buffer smooths chunk transitions
  - [ ] Playback can be stopped/interrupted
  - [ ] No audio gaps or overlaps

---

- [ ] 11. Implement Conversation State Machine

  **What to do**:
  - Create `src-tauri/src/conversation/state.rs`:
    - Conversation history: `Vec<Message>` (role, content, timestamp)
    - Current turn tracking: `Turn { id, status, asr_segment_id, llm_request_id, tts_stream_id }`
    - Status enum: Idle, Listening, FinalizingASR, Thinking, Speaking
    - Cancellation tokens for stopping in-flight requests
    - State transitions:
      - Idle → Listening (user interaction)
      - Listening → FinalizingASR (VAD speech_end)
      - FinalizingASR → Thinking (ASR final received)
      - Thinking → Speaking (LLM first token)
      - Speaking → Listening (playback end OR user interruption)
      - Any → Idle (stop)
    - Interruption handling: Cancel TTS, LLM on VAD speech_start during Speaking

  **Success criteria**:
  - [ ] State transitions work correctly
  - [ ] Interruption cancels in-flight requests
  - [ ] Conversation history maintained

---

- [ ] 12. Implement LLM Streaming Client

  **What to do**:
  - Create `src-tauri/src/llm/client.rs`:
    - async-openai client setup with API key from env/config
    - Streaming chat completions:
      - Send conversation history as messages
      - Enable streaming: `stream = true`
      - Iterate over SSE events (`_text.delta`)
      - Accumulate full response
    - Chunking strategy:
      - Buffer tokens into speakable chunks (20-40 tokens or punctuation)
      - Emit chunks to TTS as they accumulate
    - Error handling: API failures, rate limits
    - Emit Tauri events: `llm_token`, `llm_chunk`, `llm_complete`

  **Success criteria**:
  - [ ] LLM client connects to OpenAI-compatible API
  - [ ] Streaming responses work (tokens arrive incrementally)
  - [ ] Chunking produces speakable fragments
  - [ ] Errors handled gracefully

---

- [ ] 13. Implement Streaming Pipeline Orchestration

  **What to do**:
  - Create `src-tauri/src/orchestration/pipeline.rs`:
    - Main event loop driving state machine
    - ASR → LLM → TTS coordination:
      - On ASR final: Send to LLM, transition to Thinking
      - On LLM first chunk: Transition to Speaking, send first chunk to TTS
      - On LLM subsequent chunks: Send to TTS (if chunk boundary)
      - On TTS audio: Queue for playback
      - On playback end: If LLM complete, transition to Listening
    - Chunking logic:
      - Accumulate LLM tokens
      - Emit to TTS when: 20-40 tokens OR punctuation (`, . ? !`) OR 200ms minimum time
    - Interruption handling:
      - On VAD speech_start during Speaking:
        - Stop playback
        - Cancel TTS WebSocket stream
        - Cancel LLM request (cancellation token)
        - Transition to Listening
    - Tauri event emissions:
      - `assistant_state` (Idle/Listening/Thinking/Speaking)
      - `user_transcript` (partial/final)
      - `llm_response` (partial/final)
    - Spawn Python inference service on startup

  **Success criteria**:
  - [ ] End-to-end pipeline works (Audio → ASR → LLM → TTS → Audio)
  - [ ] Streaming flow is smooth (no large gaps)
  - [ ] Interruption works (user can stop assistant mid-sentence)
  - [ ] State transitions are correct
  - [ ] Tauri events emitted for UI updates

---

### Phase 4: React Frontend

- [ ] 14. Setup React Dependencies

  **What to do**:
  - Update `package.json`:
    ```json
    {
      "dependencies": {
        "@tauri-apps/api": "^2.0",
        "react": "^18.2",
        "react-dom": "^18.2",
        "zustand": "^4.4"
      }
    }
    ```
  - Install dependencies: `npm install`
  - Verify dev server works: `npm run dev`

  **Success criteria**:
  - [ ] `npm install` succeeds
  - [ ] `npm run dev` works

---

- [ ] 15. Implement UI Components

  **What to do**:
  - Create `src/components/VoiceAssistant.tsx`:
    - Conversation display (user + assistant messages)
    - Real-time transcript (partial ASR results)
    - Status indicator (Idle/Listening/Thinking/Speaking)
    - Controls: Start/Stop button, Microphone permission request
  - Create Zustand store:
    - `conversation`: Array of messages
    - `status`: Current assistant state
    - `transcript`: Partial user text
  - Style with basic CSS (or Tailwind if template includes it)

  **Success criteria**:
  - [ ] UI renders conversation history
  - [ ] Real-time transcript updates
  - [ ] Status indicator shows correct state
  - [ ] Start/Stop controls functional

---

- [ ] 16. Integrate Tauri Events and Commands

  **What to do**:
  - Update `src/components/VoiceAssistant.tsx`:
    - Listen to Tauri events:
      - `listen('vad_status', (status) => ...)` - Update UI state
      - `listen('user_transcript', (text) => ...)` - Update transcript display
      - `listen('llm_response', (text) => ...)` - Update assistant response
      - `listen('assistant_state', (state) => ...)` - Update status indicator
      - `listen('audio_level', (level) => ...)` - Update visualizer
    - Call Tauri commands:
      - `invoke('start_recording')` - Begin listening
      - `invoke('stop_recording')` - Stop
    - Handle errors: Permission denied, service unavailable
    - Implement error boundaries and user-friendly error messages
    - Add loading states

  **Success criteria**:
  - [ ] Tauri events update UI in real-time
  - [ ] Start/Stop commands trigger backend correctly
  - [ ] Errors display to user
  - [ ] Loading states work

---

### Phase 5: Integration & Testing

- [ ] 17. Test Full End-to-End Flow

  **What to do**:
  - Start Python inference service: `cd inference && python run_inference.py`
  - Verify both ASR and TTS services start
  - Start Tauri dev: `npm run tauri dev`
  - Test pipeline:
    - Click "Start" button
    - Speak simple phrase: "Hello, how are you?"
    - Verify VAD detects speech
    - Verify partial transcript appears in UI
    - Verify LLM responds (check conversation history)
    - Verify TTS audio plays back
    - Verify status indicator transitions
  - Test interruption:
    - While assistant is speaking, say something
    - Verify audio stops immediately
    - Verify assistant switches to Listening
    - Verify new phrase is transcribed correctly

  **Success criteria**:
  - [ ] Full pipeline works end-to-end
  - [ ] Interruption works correctly
  - [ ] No console errors or crashes

---

### Phase 6: Documentation

- [ ] 18. Create Basic Documentation

  **What to do**:
  - Create `README.md`:
    - Project overview
    - Quick start guide
    - Prerequisites (Rust, Node.js, Python)
    - Installation steps
    - Usage instructions
  - Create `docs/ARCHITECTURE.md`:
    - High-level architecture diagram
    - Data flow explanation
    - Technology stack overview
  - Create `docs/SETUP.md`:
    - Detailed setup instructions
    - Environment configuration
    - Model setup (GLM-ASR, Qwen3-TTS)
    - Troubleshooting guide

  **Success criteria**:
  - [ ] README provides project overview and quick start
  - [ ] ARCHITECTURE.md explains system design
  - [ ] SETUP.md provides complete setup guide
  - [ ] All docs are clear and helpful

---

## Summary

This simplified plan contains **18 core tasks** organized into **6 phases**:

1. Project Setup (2 tasks)
2. Python Inference Service (4 tasks)
3. Rust Backend (6 tasks)
4. React Frontend (3 tasks)
5. Integration Testing (1 task)
6. Documentation (2 tasks)

**Key Features**:
- Sub-500ms latency target
- Real-time streaming pipeline (ASR → LLM → TTS)
- Interruption support (barge-in)
- WebSocket IPC with binary audio frames
- Hybrid Rust/Python architecture
- OpenAI-compatible LLM integration

**Next Step**: Run `/start-work` to begin execution with this plan.
