#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
import sys
from pathlib import Path

# 添加 rknn 到搜索路径以便导入模型定义
sys.path.insert(0, str(Path(__file__).parent / "rknn"))

import torch_model
import adaptor
import base64

def generate_sensevoice_tokens(tiktoken_path):
    """
    根据 SenseVoice 官方逻辑动态生成 60515 个 Token。
    包含：Tiktoken基础词、语种标签、语音事件、情感标签、控制符、占位符、时间戳及 CTC 空白符。
    """
    print(f"Generating tokens from {tiktoken_path}...")
    tokens = []
    # 1. 加载基础 Tiktoken (58836 tokens)
    with open(tiktoken_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tokens.append(line.split()[0])
    
    # 2. 补全 ASR 特殊 Token (58836 开始)
    special_labels = [
        "<|endoftext|>",              # 58836
        "<|startoftranscript|>",      # 58837
        # Languages (105)
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", 
        "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", 
        "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", 
        "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", 
        "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", 
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", 
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", 
        "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
        # Events (11)
        "ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause",
        # Emotions (4)
        "HAPPY", "SAD", "ANGRY", "NEUTRAL",
        # Control (6)
        "translate", "transcribe", "startoflm", "startofprev", "nospeech", "notimestamps"
    ]
    
    for label in special_labels:
        if not label.startswith("<|"): label = f"<|{label}|>"
        tokens.append(base64.b64encode(label.encode()).decode())
        
    # 3. 预留占位符 (50)
    for i in range(1, 51):
        label = f"<|SPECIAL_TOKEN_{i}|>"
        tokens.append(base64.b64encode(label.encode()).decode())
        
    # 4. 时间戳 (1500 entries: 0.00 to 29.98)
    for i in range(1500):
        label = f"<|{i * 0.02:.2f}|>"
        tokens.append(base64.b64encode(label.encode()).decode())
        
    # 5. CTC Blank (60514)
    tokens.append(base64.b64encode("<blk>".encode()).decode())
    
    return tokens

class AudioPreprocessor(nn.Module):
    def __init__(self, n_fft=512, win_len=400, hop_len=160, n_mels=80, sample_rate=16000, lfr_m=7, lfr_n=6):
        super().__init__()
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_length = hop_len
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        
        # 1. Window: Hamming
        window = torch.hamming_window(win_len)
        self.register_buffer("window", window)
        
        # 2. Mel FBanks: HTK style
        # 我们使用 torchaudio 生成与 Kaldi 兼容的 Mel 矩阵
        mel_fbanks = torchaudio.functional.melscale_fbanks(
            n_fft // 2 + 1, 
            n_mels=n_mels, 
            f_min=20, 
            f_max=sample_rate // 2, 
            sample_rate=sample_rate, 
            mel_scale='htk'
        )
        # mel_fbanks shape: [bins, n_fft//2+1]
        self.register_buffer("mel_fbanks", mel_fbanks) # [n_fft//2+1, 80]

    def forward(self, audio):
        # audio: [1, N]
        
        # DC Offset Removal (模仿 reference.py)
        audio = audio - torch.mean(audio, dim=1, keepdim=True)
        
        # Pre-emphasis
        # audio = [batch, samples]
        # x[n] = x[n] - 0.97 * x[n-1]
        # 这里用简单的切片实现
        audio_p = torch.cat([audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1]], dim=1)
        
        # STFT
        # 注意：为了匹配 kaldi-native-fbank，我们需要设置 center=False 或进行特定的 padding
        # Kaldi 默认不使用 center padding
        stft = torch.stft(
            audio_p, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_len, 
            window=self.window, 
            center=False, 
            return_complex=True
        )
        # stft shape: [1, n_fft//2+1, T]
        
        # Power Spectrum
        pow_spec = torch.abs(stft) ** 2
        
        # Mel Spectrum
        # [1, n_fft//2+1, T] -> [1, T, n_fft//2+1]
        pow_spec = pow_spec.transpose(1, 2)
        mel_spec = torch.matmul(pow_spec, self.mel_fbanks) # [1, T, 80]
        
        # Log-Mel
        log_mel = torch.log(mel_spec + 1e-7)
        
        # LFR Stacking
        # T 是动态的。我们使用 unfold 进行滑动窗口拼接。
        # log_mel: [1, T, 80]
        # 我们需要在维度 1 上展开。
        # size=lfr_m (7), step=lfr_n (6)
        if log_mel.shape[1] < self.lfr_m:
            # 长度不足时进行补齐
            pad_len = self.lfr_m - log_mel.shape[1]
            log_mel = F.pad(log_mel, (0, 0, 0, pad_len), "constant", 0)
            
        lfr_feat = log_mel.unfold(1, self.lfr_m, self.lfr_n) # [1, T_lfr, 80, 7]
        # 转置并重塑为 [1, T_lfr, 560]
        T_lfr = lfr_feat.shape[1]
        lfr_feat = lfr_feat.transpose(2, 3).reshape(1, T_lfr, -1)
        
        return lfr_feat

class AllInOneSenseVoice(nn.Module):
    def __init__(self, core_model, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        self.core_model = core_model
        
    def forward(self, audio):
        # audio: [1, N] raw waveform
        features = self.preprocessor(audio)
        logits = self.core_model(features)
        return logits

def load_core_model():
    # 模仿 rknn/nano.py 的初始化过程
    import nano
    model = nano.Nano()
    
    state_dict = torch.load("Fun-ASR-Nano-2512/model.pt", map_location="cpu")
    # 移除无用权重
    to_delete = [k for k in state_dict if "llm" in k or "audio_adaptor" in k]
    for k in to_delete:
        del state_dict[k]
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

@torch.no_grad()
def main():
    print("Loading core model...")
    core_model = load_core_model()
    
    print("Initializing preprocessor...")
    preprocessor = AudioPreprocessor()
    
    model = AllInOneSenseVoice(core_model, preprocessor)
    model.eval()
    
    print(f"Core model loaded. Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Dummy input: 1 second of audio at 16kHz
    dummy_input = torch.randn(1, 16000)
    
    # 准备词表
    tiktoken_path = "Fun-ASR-Nano-2512/multilingual.tiktoken"
    all_tokens = generate_sensevoice_tokens(tiktoken_path)
    vocab_size = len(all_tokens)
    blank_id = vocab_size - 1
    
    # 保存词表文件
    tokens_output_path = "tokens.txt"
    with open(tokens_output_path, "w", encoding="utf-8") as f:
        for i, t in enumerate(all_tokens):
            f.write(f"{t} {i}\n")
    print(f"Generated {vocab_size} tokens and saved to {tokens_output_path}")

    print("Exporting to ONNX...")
    output_path = "sense_voice_all.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13, # 降低到 13 以获得更好的稳定性
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={
            "audio": {1: "samples"},
            "logits": {1: "frames"}
        }
    )
    
    # 注入元数据 (模仿 sherpa-onnx)
    import onnx
    onnx_model = onnx.load(output_path)
    meta_data = {
        "model_type": "sense_voice_all_in_one",
        "sample_rate": 16000,
        "lfr_window_size": 7,
        "lfr_window_shift": 6,
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "comment": "SenseVoice-CTC All-in-One with Internal Vocab Generation",
        "author": "Haujet (Integrated by Antigravity)"
    }
    
    # 清理旧元数据
    while len(onnx_model.metadata_props):
        onnx_model.metadata_props.pop()
        
    for key, value in meta_data.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
        
    onnx.save(onnx_model, output_path)
    print(f"Successfully exported to {output_path}")

if __name__ == "__main__":
    main()
