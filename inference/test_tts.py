import asyncio
import websockets
import json
import numpy as np
import wave
import os

async def test_tts_service():
    uri = "ws://127.0.0.1:8766"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to TTS service at {uri}")
            
            test_texts = [
                "Hello, how are you?",
                "This is a test of the text to speech system.",
                "The quick brown fox jumps over the lazy dog."
            ]
            
            for i, text in enumerate(test_texts, 1):
                print(f"\nTest {i}: {text}")
                
                message = {
                    "type": "text_chunk",
                    "text": text,
                    "text_id": i
                }
                
                await websocket.send(json.dumps(message))
                print("Message sent, receiving audio frames...")
                
                frames = []
                timeout_seconds = 30
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < timeout_seconds:
                    try:
                        frame = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        if isinstance(frame, bytes):
                            frames.append(frame)
                            print(f"Received frame {len(frames)}/{len(frame)}")
                    except asyncio.TimeoutError:
                        continue
                
                if frames:
                    total_audio = b''.join(frames)
                    print(f"Received {len(frames)} frames ({len(total_audio)} bytes)")
                    
                    output_file = f"tts_test_output_{i}.wav"
                    save_wav(output_file, total_audio, 16000)
                    print(f"Saved audio to {output_file}")
                else:
                    print("No audio received within timeout")
                
                await asyncio.sleep(0.5)
            
    except ConnectionRefusedError:
        print("Error: Could not connect to TTS service. Is it running?")
        print("Start it with: python inference/tts_service.py")
    except Exception as e:
        print(f"Error: {e}")

def save_wav(filename: str, audio_data: bytes, sample_rate: int):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

if __name__ == "__main__":
    print("TTS Service Test")
    print("Make sure TTS service is running on ws://127.0.0.1:8766")
    print()
    
    asyncio.run(test_tts_service())
