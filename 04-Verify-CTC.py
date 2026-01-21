#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import librosa
import base64
import time

def load_tokens(filename):
    id2token = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            
            # 处理空格 token 的特殊情况
            if len(parts) == 1:
                t, i = " ", parts[0]
            else:
                t, i = parts
            id2token[int(i)] = t
    return id2token

def decode_ctc(logits, id2token, blank_id):
    indices = np.argmax(logits[0], axis=-1)
    
    # 官方偏移逻辑：(frame * 60 - 30) / 1000
    frame_shift_ms = 60
    offset_ms = -30
    
    results = []
    last_idx = -1
    for i, idx in enumerate(indices):
        # 卫语句：跳过空白符和重复帧
        if idx == blank_id or idx == last_idx:
            last_idx = idx
            continue
        
        last_idx = idx
        token_b64 = id2token.get(idx, "")
        if not token_b64: continue
        
        token_text = base64.b64decode(token_b64).decode("utf-8")
        is_timestamp = False
        display_text = token_text
        
        # 识别内置时间戳 Token
        if token_text.startswith("<|") and token_text.endswith("|>"):
            try:
                ts_val = float(token_text[2:-2])
                display_text = f" #⚓{ts_val:.2f}s# "
                is_timestamp = True
            except ValueError:
                pass
        
        results.append({
            "text": display_text,
            "time": max((i * frame_shift_ms + offset_ms) / 1000.0, 0.0),
            "is_ts_token": is_timestamp
        })
                
    full_text = "".join([r["text"] for r in results])
    
    # 构造易读的带时间戳输出
    detailed_lines = []
    for r in results:
        if r["is_ts_token"]:
            detailed_lines.append(f"\n[{r['time']:.2f}s] {r['text']}")
            continue
        detailed_lines.append(f"({r['time']:.2f}s){r['text']}")
            
    return full_text, "".join(detailed_lines)

def main():
    model_path = "sense_voice_all.onnx"
    audio_path = "input.mp3"
    tokens_path = "tokens.txt"
    
    print(f"Loading tokens from {tokens_path}...")
    id2token = load_tokens(tokens_path)
    blank_id = max(id2token.keys())
    
    print(f"Loading model {model_path}...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    print(f"Loading audio {audio_path}...")
    audio, _ = librosa.load(audio_path, sr=16000)
    audio = audio[np.newaxis, :].astype(np.float32) 
    
    print(f"Inference starting (input length: {audio.shape[1]} samples)...")
    start_time = time.time()
    logits = session.run(None, {"audio": audio})[0]
    
    print(f"Inference finished in {time.time() - start_time:.2f} seconds.")
    full_text, detailed_text = decode_ctc(logits, id2token, blank_id)
    
    print("\n" + "="*20)
    print("Transcription Result (Refactored Logic):")
    print("-" * 20)
    print(f"Full Text:\n{full_text}")
    print("-" * 20)
    print(f"Detailed with Timestamps:\n{detailed_text}")
    print("="*20)

if __name__ == "__main__":
    main()
