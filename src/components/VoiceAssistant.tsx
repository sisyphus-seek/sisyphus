import { useEffect, useState, useCallback } from 'react';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { create } from 'zustand';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

type ConversationStatus = 'Idle' | 'Listening' | 'FinalizingASR' | 'Thinking' | 'Speaking';

interface ConversationState {
  conversation: Message[];
  status: ConversationStatus;
  transcript: string;
  audioLevel: number;
  pendingAssistantResponse: string;
  addMessage: (message: Message) => void;
  setStatus: (status: ConversationStatus) => void;
  setTranscript: (transcript: string) => void;
  setAudioLevel: (level: number) => void;
  appendAssistantResponse: (chunk: string, isComplete: boolean) => void;
  reset: () => void;
}

const useConversation = create<ConversationState>((set, get) => ({
  conversation: [],
  status: 'Idle',
  transcript: '',
  audioLevel: 0,
  pendingAssistantResponse: '',
  addMessage: (message) =>
    set((state) => ({
      conversation: [...state.conversation, message],
    })),
  setStatus: (status) => set({ status }),
  setTranscript: (transcript) => set({ transcript }),
  setAudioLevel: (level) => set({ audioLevel: level }),
  appendAssistantResponse: (chunk, isComplete) => {
    if (isComplete) {
      const pending = get().pendingAssistantResponse + chunk;
      console.log('Assistant response complete, moving to conversation:', pending.substring(0, 100));
      if (pending.trim()) {
        set((state) => ({
          conversation: [...state.conversation, { role: 'assistant', content: pending.trim() }],
          pendingAssistantResponse: '',
        }));
      }
    } else {
      set((state) => ({
        pendingAssistantResponse: state.pendingAssistantResponse + chunk,
      }));
    }
  },
  reset: () =>
    set({
      transcript: '',
      pendingAssistantResponse: '',
    }),
}));

interface StateChangedPayload {
  state: ConversationStatus;
}

interface UserTranscriptPayload {
  partial: string;
  final: string;
  confidence: number;
}

interface AssistantResponsePayload {
  content: string;
  is_complete: boolean;
}

interface AudioLevelPayload {
  level: number;
}

interface VadStatusPayload {
  status: 'speech_start' | 'speech_end' | 'silence';
}

export function VoiceAssistant() {
  const {
    conversation,
    status,
    transcript,
    audioLevel,
    pendingAssistantResponse,
    setStatus,
    setTranscript,
    setAudioLevel,
    appendAssistantResponse,
    addMessage,
    reset,
  } = useConversation();

  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const unlisteners: UnlistenFn[] = [];

    const setupListeners = async () => {
      const unlistenStateChanged = await listen<StateChangedPayload>(
        'voice_assistant:state_changed',
        (event) => {
          setStatus(event.payload.state);
        }
      );
      unlisteners.push(unlistenStateChanged);

      const unlistenTranscript = await listen<UserTranscriptPayload>(
        'voice_assistant:user_transcript',
        (event) => {
          const { partial, final } = event.payload;
          console.log('Transcript event received:', { partial, final, currentStatus: useConversation.getState().status });
          
          if (final) {
            setTranscript(final);
            // If we are in FinalizingASR, it means the user has already clicked stop
            // and is waiting for this final result to trigger the LLM.
            const currentStatus = useConversation.getState().status;
            if (currentStatus === 'FinalizingASR') {
              console.log('Final transcript received in FinalizingASR state, triggering LLM now...');
              triggerLlmRequest(final);
            }
          } else if (partial) {
            setTranscript(partial);
          }
        }
      );
      unlisteners.push(unlistenTranscript);

      const unlistenResponse = await listen<AssistantResponsePayload>(
        'voice_assistant:assistant_response',
        (event) => {
          const { content, is_complete } = event.payload;
          console.log('Assistant response event:', { content: content.substring(0, 50), is_complete });
          appendAssistantResponse(content, is_complete);
        }
      );
      unlisteners.push(unlistenResponse);

      const unlistenAudioLevel = await listen<AudioLevelPayload>(
        'voice_assistant:audio_level',
        (event) => {
          setAudioLevel(event.payload.level);
        }
      );
      unlisteners.push(unlistenAudioLevel);

      const unlistenVadStatus = await listen<VadStatusPayload>(
        'voice_assistant:vad_status',
        (event) => {
          console.log('VAD status:', event.payload.status);
        }
      );
      unlisteners.push(unlistenVadStatus);
    };

    setupListeners();

    return () => {
      unlisteners.forEach((unlisten) => unlisten());
    };
  }, [setStatus, setTranscript, setAudioLevel, appendAssistantResponse, addMessage]);

  const handleStartRecording = useCallback(async () => {
    console.log('Start recording button clicked');
    try {
      reset();
      console.log('Invoking start_recording...');
      const result = await invoke('start_recording');
      console.log('start_recording result:', result);
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert(`Recording failed: ${error}`);
    }
  }, [reset]);

  const triggerLlmRequest = useCallback(async (text: string) => {
    if (!text.trim()) {
      setStatus('Idle');
      return;
    }

    try {
      addMessage({ role: 'user', content: text.trim() });
      setTranscript('');
      console.log('Sending to LLM:', text);
      setStatus('Thinking');
      await invoke('stream_llm_response', { userMessage: text.trim() });
    } catch (error) {
      console.error('LLM error:', error);
      setStatus('Idle');
    }
  }, [setStatus, addMessage, setTranscript]);

  const handleStopRecording = useCallback(async () => {
    // Only process if we're in an active recording/finalizing state
    const currentStatus = useConversation.getState().status;
    if (currentStatus !== 'Listening' && currentStatus !== 'FinalizingASR') {
      console.log('Ignoring stop - not in stoppable state:', currentStatus);
      return;
    }

    try {
      setStatus('FinalizingASR');
      console.log('Stopping recording, waiting for backend to finalize...');
      await invoke('stop_recording');
      
      // We don't trigger LLM here anymore if it's already triggered by the 'final' transcript event.
      // But if no 'final' transcript comes within a timeout, we should fall back to the current partial.
      setTimeout(() => {
          const statusAfterWait = useConversation.getState().status;
          if (statusAfterWait === 'FinalizingASR') {
              console.log('FinalizingASR timed out, using current transcript as fallback');
              const currentTranscript = useConversation.getState().transcript;
              triggerLlmRequest(currentTranscript);
          }
      }, 1500);

    } catch (error) {
      console.error('Failed to stop recording:', error);
      setStatus('Idle');
    }
  }, [setStatus, triggerLlmRequest]);

  const handleSubmitTranscript = useCallback(() => {
    if (transcript.trim()) {
      addMessage({ role: 'user', content: transcript.trim() });
      setTranscript('');
    }
  }, [transcript, addMessage, setTranscript]);

  if (!mounted) return null;

  const isRecordingDisabled = status !== 'Idle';
  const isStopDisabled = status !== 'Listening' && status !== 'FinalizingASR';

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ marginBottom: '20px' }}>Voice Assistant</h1>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            marginBottom: '20px',
            padding: '15px',
            backgroundColor: '#f0f0f0',
            borderRadius: '8px',
          }}
        >
          <button
            onClick={handleStartRecording}
            disabled={isRecordingDisabled}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              backgroundColor: !isRecordingDisabled ? '#007AFF' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: !isRecordingDisabled ? 'pointer' : 'not-allowed',
            }}
          >
            {status === 'Idle' ? 'Start Recording' : 'Recording...'}
          </button>

          <button
            onClick={handleStopRecording}
            disabled={isStopDisabled}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              backgroundColor: !isStopDisabled ? '#007AFF' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: !isStopDisabled ? 'pointer' : 'not-allowed',
            }}
          >
            Stop
          </button>

          {status === 'Listening' && (
            <div
              style={{
                width: '100px',
                height: '10px',
                backgroundColor: '#ddd',
                borderRadius: '5px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${Math.min(audioLevel * 100, 100)}%`,
                  height: '100%',
                  backgroundColor: '#007AFF',
                  transition: 'width 0.1s ease-out',
                }}
              />
            </div>
          )}
        </div>

        <div
          style={{
            marginBottom: '20px',
            padding: '10px',
            backgroundColor: '#e8e8e8',
            borderRadius: '4px',
          }}
        >
          <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>
            Status: {status}
          </div>
          {transcript && (
            <div style={{ fontSize: '14px', color: '#666' }}>Partial Transcript: {transcript}</div>
          )}
          {pendingAssistantResponse && (
            <div style={{ fontSize: '14px', color: '#28a745', marginTop: '5px' }}>
              Assistant: {pendingAssistantResponse}
            </div>
          )}
        </div>

        <div
          style={{
            maxHeight: '400px',
            overflowY: 'auto',
            padding: '15px',
            backgroundColor: '#f8f9fa',
            borderRadius: '4px',
            marginBottom: '20px',
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: '15px' }}>Conversation</h3>
          {conversation.length === 0 ? (
            <div style={{ color: '#999', fontStyle: 'italic' }}>
              No conversation yet. Start recording to begin.
            </div>
          ) : (
            conversation.map((msg, index) => (
              <div
                key={index}
                style={{
                  marginBottom: '15px',
                  padding: '10px',
                  backgroundColor: msg.role === 'user' ? '#e3f2fd' : '#ffffff',
                  borderRadius: '8px',
                  borderLeft: `4px solid ${msg.role === 'user' ? '#007AFF' : '#28a745'}`,
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    fontWeight: 'bold',
                    marginBottom: '5px',
                    color: msg.role === 'user' ? '#007AFF' : '#28a745',
                  }}
                >
                  {msg.role === 'user' ? 'You' : 'Assistant'}
                </div>
                <div style={{ fontSize: '14px', lineHeight: '1.5' }}>{msg.content}</div>
              </div>
            ))
          )}
        </div>

        {transcript.trim() && status === 'Listening' && (
          <div
            style={{
              marginTop: '20px',
              padding: '10px',
              backgroundColor: '#fff3cd',
              borderRadius: '4px',
              border: '1px solid #ffc107',
              textAlign: 'center',
            }}
          >
            <span style={{ fontSize: '14px', marginRight: '10px' }}>Transcript: "{transcript}"</span>
            <button
              onClick={handleSubmitTranscript}
              style={{
                padding: '8px 16px',
                fontSize: '14px',
                backgroundColor: '#007AFF',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Submit
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
