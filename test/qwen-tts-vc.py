import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "F:\\GitRepository\\Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.float16,
    attn_implementation="sdpa",
)

ref_audio = "output_custom_voice_1.wav"
ref_text  = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"

wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)
