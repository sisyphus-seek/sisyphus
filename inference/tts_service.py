import asyncio
import json
import os
from typing import Optional

import numpy as np
import websockets
import yaml

class TTSService:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766):
        self.host = host
        self.port = port
        self.base_model = None
        self.custom_model = None
        self.device = "cpu"
        self.use_fp16 = False
        self.default_voice = "custom_voice"
        self.default_language = "Auto"
        self.default_speaker = "Vivian"
        self.attn_implementation = None
        self.base_model_path = "Qwen/Qwen-Audio-TTS"
        self.custom_model_path = "Qwen/Qwen-Audio-TTS"
        self.sample_rate = 24000  # TTS model native sample rate
        self.target_sample_rate = 16000  # Target for Rust playback
        self.frame_size = 640  # 640 bytes = 320 samples @ 16kHz = 20ms

    def load_config(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "models.yaml")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        tts_config = config.get("tts", {})
        self.base_model_path = tts_config.get("base_model_path", self.base_model_path)
        self.custom_model_path = tts_config.get("custom_model_path", self.custom_model_path)
        self.default_voice = tts_config.get("default_voice", self.default_voice)
        self.default_language = tts_config.get("default_language", self.default_language)
        self.default_speaker = tts_config.get("default_speaker", self.default_speaker)
        self.attn_implementation = tts_config.get(
            "attn_implementation", self.attn_implementation
        )
        requested_device = tts_config.get("device", "auto")
        self.use_fp16 = bool(tts_config.get("fp16", True))

        import torch

        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device

        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        if self.device != "cuda":
            self.use_fp16 = False
        
    async def load_model(self):
        try:
            self.load_config()

            import torch

            from qwen_tts import Qwen3TTSModel

            dtype = torch.float16 if self.use_fp16 else torch.float32
            device_map = "cuda:0" if self.device == "cuda" else "cpu"

            print(f"Loading TTS base model: {self.base_model_path}")
            self.base_model = Qwen3TTSModel.from_pretrained(
                self.base_model_path,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=self.attn_implementation,
            )

            print(f"Loading TTS custom model: {self.custom_model_path}")
            self.custom_model = Qwen3TTSModel.from_pretrained(
                self.custom_model_path,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=self.attn_implementation,
            )

            print(
                "TTS models loaded successfully on "
                f"{self.device} (fp16={self.use_fp16})"
            )
            return True
        except Exception as e:
            print(f"Error loading TTS model: {e}")
            print("Using fallback mock TTS generation")
            return False
    
    def float32_to_pcm16(self, audio_array: np.ndarray) -> bytes:
        audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str],
        language: Optional[str],
        speaker: Optional[str],
        instruct: Optional[str],
        ref_audio: Optional[str],
        ref_text: Optional[str],
        voice_mode: Optional[str],
    ) -> np.ndarray:
        if self.base_model is None or self.custom_model is None:
            return np.zeros(int(self.sample_rate * 0.5))

        try:
            selected_voice = voice or self.default_voice
            language = language or self.default_language
            speaker = speaker or self.default_speaker
            voice_mode = voice_mode or "custom"

            if selected_voice == "custom_voice" or voice_mode == "custom":
                wavs, sr = self.custom_model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    instruct=instruct or "",
                )
                audio_waveform = wavs[0]
                output_sample_rate = sr
            else:
                if not ref_audio or not ref_text:
                    print("Voice clone requires ref_audio and ref_text")
                    return np.zeros(int(self.sample_rate * 0.5))

                wavs, sr = self.base_model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                audio_waveform = wavs[0]
                output_sample_rate = sr
            
            print(f"TTS model output sample rate: {output_sample_rate}Hz, target: {self.target_sample_rate}Hz")
            print(f"Audio waveform shape before resample: {audio_waveform.shape}, min: {audio_waveform.min():.4f}, max: {audio_waveform.max():.4f}")

            # Always resample to target rate for consistency
            if output_sample_rate != self.target_sample_rate:
                import librosa
                print(f"Resampling from {output_sample_rate}Hz to {self.target_sample_rate}Hz")
                audio_waveform = librosa.resample(
                    audio_waveform,
                    orig_sr=output_sample_rate,
                    target_sr=self.target_sample_rate
                )
                print(f"Audio waveform shape after resample: {audio_waveform.shape}")

            # Normalize audio to prevent clipping
            max_val = np.abs(audio_waveform).max()
            if max_val > 1.0:
                print(f"Normalizing audio (max was {max_val})")
                audio_waveform = audio_waveform / max_val * 0.95

            return audio_waveform
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return np.zeros(int(self.sample_rate * 0.5))
    
    async def process_text_chunk(
        self,
        text: str,
        text_id: int,
        voice: Optional[str],
        language: Optional[str],
        speaker: Optional[str],
        instruct: Optional[str],
        ref_audio: Optional[str],
        ref_text: Optional[str],
        voice_mode: Optional[str],
    ) -> list[bytes]:
        audio_waveform = await self.synthesize(
            text, voice, language, speaker, instruct, ref_audio, ref_text, voice_mode
        )

        # Debug: show audio stats
        duration_sec = len(audio_waveform) / self.target_sample_rate
        print(f"Audio duration: {duration_sec:.2f}s ({len(audio_waveform)} samples @ {self.target_sample_rate}Hz)")

        audio_pcm16 = self.float32_to_pcm16(audio_waveform)

        # Debug: show first few PCM16 samples
        first_samples = np.frombuffer(audio_pcm16[:20], dtype=np.int16)
        print(f"First PCM16 samples: {first_samples}")

        frames = []
        for i in range(0, len(audio_pcm16), self.frame_size):
            frame = audio_pcm16[i:i+self.frame_size]
            if len(frame) < self.frame_size:
                frame += b'\x00' * (self.frame_size - len(frame))
            frames.append(frame)

        return frames
    
    async def handle_connection(self, websocket):
        print(f"New TTS connection from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    try:
                        control = json.loads(message)
                        
                        if control.get("type") == "text_chunk":
                            text = control.get("text", "")
                            text_id = control.get("text_id", 0)
                            voice = control.get("voice")
                            language = control.get("language")
                            speaker = control.get("speaker")
                            instruct = control.get("instruct")
                            ref_audio = control.get("ref_audio")
                            ref_text = control.get("ref_text")
                            voice_mode = control.get("voice_mode")
                            
                            print(f"Synthesizing text: {text}")
                            frames = await self.process_text_chunk(
                                text,
                                text_id,
                                voice,
                                language,
                                speaker,
                                instruct,
                                ref_audio,
                                ref_text,
                                voice_mode,
                            )
                            
                            if frames:
                                print(f"Sending {len(frames)} audio frames for text_id={text_id}")
                                for i, frame in enumerate(frames):
                                    await websocket.send(frame)
                            else:
                                print(f"No frames generated for text_id={text_id}")
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON message: {e}")
                    except Exception as e:
                        print(f"Error processing text_chunk: {e}")
                        
                elif isinstance(message, bytes):
                    print(f"Received audio frame control message (should not happen)")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"TTS connection closed: {websocket.remote_address}")
        except Exception as e:
            print(f"TTS connection error: {e}")
    
    async def start(self):
        print(f"Starting TTS WebSocket server on {self.host}:{self.port}")
        
        await self.load_model()
        
        async with websockets.serve(self.handle_connection, self.host, self.port):
            print(f"TTS server is running on ws://{self.host}:{self.port}")
            await asyncio.Future()

async def main():
    service = TTSService()
    await service.start()

if __name__ == "__main__":
    asyncio.run(main())
