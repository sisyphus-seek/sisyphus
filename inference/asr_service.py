import asyncio
import json
import os
from typing import Optional

import numpy as np
import websockets
import yaml

class ASRService:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.use_fp16 = False
        self.use_kv_cache = True
        self.model_path = "THUDM/glm-asr-nano-2512"
        self.sample_rate = 16000
        self.frame_size = 640
        self.accumulated_audio = []
        self.window_duration = 2.5
        self.overlap_duration = 0.5
        self.overlap_samples = int(self.sample_rate * self.overlap_duration)

    def load_config(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "models.yaml")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        asr_config = config.get("asr", {})
        self.model_path = asr_config.get("model_path", self.model_path)
        requested_device = asr_config.get("device", "auto")
        self.use_fp16 = bool(asr_config.get("fp16", True))
        self.use_kv_cache = bool(asr_config.get("kv_cache", True))

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
            from transformers import AutoModelForSeq2SeqLM, AutoProcessor

            self.load_config()

            print(f"Loading ASR model: {self.model_path}")

            import torch

            dtype = torch.float16 if self.use_fp16 else torch.float32

            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = self.use_kv_cache
            if hasattr(self.model, "generation_config") and hasattr(
                self.model.generation_config, "use_cache"
            ):
                self.model.generation_config.use_cache = self.use_kv_cache
            if hasattr(self.model, "generation_config"):
                generation_config = self.model.generation_config
                # Ensure pad_token_id is a single integer
                if isinstance(getattr(generation_config, "pad_token_id", None), (list, tuple)):
                    generation_config.pad_token_id = generation_config.pad_token_id[0]
                
                if getattr(generation_config, "pad_token_id", None) is None:
                    eos_token_id = getattr(generation_config, "eos_token_id", None)
                    if isinstance(eos_token_id, (list, tuple)):
                        eos_token_id = eos_token_id[0]
                    if eos_token_id is not None:
                        generation_config.pad_token_id = eos_token_id

            self.model.to(self.device)

            print(
                "ASR model loaded successfully on "
                f"{self.device} (fp16={self.use_fp16}, kv_cache={self.use_kv_cache})"
            )
            return True
        except Exception as e:
            print(f"Error loading ASR model: {e}")
            print("Using fallback mock transcription")
            return False
    
    def pcm16_to_float32(self, pcm_data: bytes) -> np.ndarray:
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
        return pcm_array.astype(np.float32) / 32768.0
    
    async def transcribe(self, audio_array: np.ndarray, is_final: bool = False) -> dict:
        if self.model is None or self.processor is None:
            return {
                "partial": "",
                "final": "[Mock transcription: model not loaded]",
                "confidence": 0.0
            }
        
        try:
            import torch

            # For very short audio, pad it to at least 0.1s to avoid processor errors
            if len(audio_array) < self.sample_rate * 0.1:
                padding = np.zeros(int(self.sample_rate * 0.1) - len(audio_array))
                audio_array = np.concatenate([audio_array, padding])

            inputs = self.processor.apply_transcription_request(audio_array)
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)

            pad_id = getattr(self.model.generation_config, "pad_token_id", None)
            if isinstance(pad_id, (list, tuple)):
                pad_id = pad_id[0]
            
            # Explicitly force EOS as PAD if still list or None to avoid comparison errors in transformers
            if pad_id is None or isinstance(pad_id, (list, tuple)):
                pad_id = getattr(self.model.generation_config, "eos_token_id", 0)
                if isinstance(pad_id, (list, tuple)):
                    pad_id = pad_id[0]

            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    **inputs,
                    use_cache=self.use_kv_cache,
                    do_sample=False,
                    max_new_tokens=500,
                    pad_token_id=pad_id,
                )

            decoded = self.processor.batch_decode(
                predicted_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            transcription = decoded[0]
            
            return {
                "partial": "" if is_final else transcription,
                "final": transcription if is_final else None,
                "confidence": 0.95
            }
        except Exception as e:
            print(f"Transcription error: {e}")
            return {
                "partial": "",
                "final": "[Transcription error]" if is_final else None,
                "confidence": 0.0
            }
    
    async def process_audio_frame(self, audio_data: bytes) -> Optional[dict]:
        audio_float = self.pcm16_to_float32(audio_data)
        self.accumulated_audio.extend(audio_float)
        
        window_samples = int(self.sample_rate * self.window_duration)
        
        if len(self.accumulated_audio) >= window_samples:
            audio_window = np.array(self.accumulated_audio[:window_samples])
            
            if self.overlap_samples > 0:
                self.accumulated_audio = self.accumulated_audio[window_samples - self.overlap_samples:]
            else:
                self.accumulated_audio = []
            
            result = await self.transcribe(audio_window, is_final=False)
            result["type"] = "asr_result"
            return result
        
        return None

    async def flush(self) -> Optional[dict]:
        if not self.accumulated_audio:
            return None
        
        print(f"Flushing remaining {len(self.accumulated_audio)} samples")
        audio_window = np.array(self.accumulated_audio)
        self.accumulated_audio = []
        
        result = await self.transcribe(audio_window, is_final=True)
        result["type"] = "asr_result"
        return result
    
    async def handle_connection(self, websocket):
        print(f"New ASR connection from {websocket.remote_address}")
        self.accumulated_audio = []
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # print(f"Received {len(message)} bytes of audio")
                    result = await self.process_audio_frame(message)
                    if result:
                        print(f"Sending partial/window result: {result.get('partial', '')}")
                        await websocket.send(json.dumps(result))
                elif isinstance(message, str):
                    print(f"Received control message: {message}")
                    control = json.loads(message)
                    if control.get("type") == "reset":
                        self.accumulated_audio = []
                        print("Audio buffer reset")
                    elif control.get("type") == "stop":
                        print("Stop command received, flushing buffer...")
                        result = await self.flush()
                        if result:
                            print(f"Sending final flush result: {result.get('final', '')}")
                            await websocket.send(json.dumps(result))
                        else:
                            print("Flush returned no result (empty buffer)")
                        print("Audio buffer flushed due to stop command")
        except websockets.exceptions.ConnectionClosed:
            print(f"ASR connection closed: {websocket.remote_address}")
        except Exception as e:
            print(f"ASR connection error: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        print(f"Starting ASR WebSocket server on {self.host}:{self.port}")
        
        await self.load_model()
        
        async with websockets.serve(self.handle_connection, self.host, self.port):
            print(f"ASR server is running on ws://{self.host}:{self.port}")
            await asyncio.Future()

async def main():
    service = ASRService()
    await service.start()

if __name__ == "__main__":
    asyncio.run(main())
