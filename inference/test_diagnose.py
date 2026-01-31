import asyncio
import websockets
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from tts_service import TTSService

async def diagnose_connection():
    uri = "ws://127.0.0.1:8766"
    print(f"=== WebSocket Diagnostic Test ===")
    print(f"Connecting to {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✓ Connected to {uri}")
            print(f"Remote address: {websocket.remote_address}")
            
            # 测试1: 等待5秒看连接是否保持
            print(f"\n[Test 1] Waiting 5 seconds to check connection stability...")
            for i in range(5):
                await asyncio.sleep(1)
            
            # 测试2: 发送简单消息
            print(f"\n[Test 2] Sending simple text message...")
            simple_message = {
                "type": "text_chunk",
                "text": "Test",
                "text_id": 0
            }
            await websocket.send(json.dumps(simple_message))
            print(f"✓ Message sent")
            
            # 测试3: 等待响应
            print(f"\n[Test 3] Waiting for response (30s timeout)...")
            response_received = False
            response_frames = []
            
            timeout = 30
            start = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start < timeout:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    print(f"✓ Received message type: {type(msg).__name__}, size: {len(msg) if isinstance(msg, bytes) else 'N/A'}")
                    response_received = True
                    
                    if isinstance(msg, bytes) and len(msg) > 0:
                        response_frames.append(msg)
                        print(f"  Audio frame {len(response_frames)}/{len(msg)} received")
                        
                except asyncio.TimeoutError:
                    print(f"✓ Timeout after {(asyncio.get_event_loop().time() - start):.1f}s")
                    break
            
            if response_received and response_frames:
                total_audio = b''.join(response_frames)
                print(f"\n[Test 3 Results]")
                print(f"  Total messages received: True")
                print(f"  Total audio data: {len(total_audio)} bytes")
                print(f"  Estimated duration: {len(total_audio) / (2 * 16000):.2f}s")
                
                import wave
                output_file = "diagnostic_output.wav"
                with wave.open(output_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(total_audio)
                print(f"  ✓ Saved to {output_file}")
            else:
                print(f"\n[Test 3 Results]")
                print(f"  Total messages received: False")
                print(f"  No audio data received")
            
            # 测试4: 检查连接状态
            print(f"\n[Test 4] Checking if connection still alive...")
            try:
                ping = await asyncio.wait_for(websocket.ping(), timeout=2.0)
                print(f"✓ Connection is alive (ping successful)")
            except:
                print(f"✗ Connection may have closed (ping failed)")
            
            print(f"\n=== Diagnostic Test Complete ===")
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_connection())
