import gc
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
import os
from pydub import AudioSegment
from funasr import AutoModel
from transformers import AutoTokenizer
from STFT_Process import STFT_Process

# =========================================================================
# 配置部分
# =========================================================================

# 输出目录
OUTPUT_DIR = r'./model-gguf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 源模型路径
model_path = r'./Fun-ASR-Nano-2512'
tokenizer_path = r'./Fun-ASR-Nano-2512/Qwen3-0.6B'

# 输出 ONNX 路径
onnx_model_A = f'{OUTPUT_DIR}/FunASR_Nano_Encoder.onnx'
onnx_model_B = f'{OUTPUT_DIR}/FunASR_Nano_Decoder_Embed.onnx'
onnx_model_C = f'{OUTPUT_DIR}/FunASR_Nano_Decoder_Main.onnx'
onnx_model_D = f'{OUTPUT_DIR}/FunASR_Nano_Greedy_Search.onnx'
onnx_model_E = f'{OUTPUT_DIR}/FunASR_Nano_First_Beam_Search.onnx'
onnx_model_F = f'{OUTPUT_DIR}/FunASR_Nano_Second_Beam_Search.onnx'
onnx_model_G = f'{OUTPUT_DIR}/FunASR_Nano_Reset_Penality.onnx'

# 参数配置
SAMPLE_RATE = 16000
WINDOW_TYPE = 'hamming'
N_MELS = 80
NFFT_STFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
PRE_EMPHASIZE = 0.97
USE_NORMALIZER = True
LFR_M = 7
LFR_N = 6
STOP_TOKEN = [151643, 151645]
MAX_SEQ_LEN = 1024
MAX_INPUT_AUDIO_LENGTH = 320000
SLIDING_WINDOW = 0
DYNAMIC_AXES = True
BEAM_SIZE = 3
MAX_BEAM_SIZE = 10
REPEAT_PENALITY = 0.9
PENALITY_RANGE = 10
MAX_THREADS = 0
DEVICE_ID = 0
OPSET = 17

MAX_STFT_SIGNAL_LENGTH = MAX_INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
LFR_LENGTH = (MAX_STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if HOP_LENGTH > MAX_INPUT_AUDIO_LENGTH:
    HOP_LENGTH = MAX_INPUT_AUDIO_LENGTH

# =========================================================================
# 模型类定义 (直接复制自导出脚本)
# =========================================================================

class GREEDY_SEARCH(torch.nn.Module):
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, logits, repeat_penality, penality_value, batch_size):
        max_logits_idx = torch.argmax(logits * repeat_penality, dim=-1, keepdim=True)
        batch_indices = self.batch_indices[:batch_size].long()
        repeat_penality[batch_indices, max_logits_idx.squeeze(-1)] *= penality_value
        return max_logits_idx.int(), repeat_penality


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        repeat_penality = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        batch_indices = self.batch_indices[:beam_size].long()
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.transpose(0, 1), batch_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-8]
        save_id = all_inputs[-7]
        repeat_penality = all_inputs[-6]
        previous_prob = all_inputs[-5]
        batch_indices = all_inputs[-4]
        penality_value = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        logits = torch.log_softmax(logits * repeat_penality, dim=-1)
        top_k_prob, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=False)
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, top_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=False)
        beam_index = top_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[top_beam_indices]
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i][beam_index]
        repeat_penality = repeat_penality[beam_index]
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        max_logits_idx = top_beam_indices[[0]]
        top_beam_indices = top_beam_indices.unsqueeze(-1)
        save_id = torch.cat([save_id[beam_index], top_beam_indices], dim=-1)
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.unsqueeze(-1), max_logits_idx


class RESET_PENALITY(torch.nn.Module):
    def __init__(self):
        super(RESET_PENALITY, self).__init__()
        pass

    def forward(self, save_id, repeat_penality, penality_reset_count, batch_indices):
        repeat_penality[batch_indices, save_id[batch_indices, penality_reset_count[batch_indices]]] = 1.0
        penality_reset_count += 1
        return save_id, repeat_penality, penality_reset_count


class FUNASR_NANO_ENCODER(torch.nn.Module):
    def __init__(self, funasr_nano, stft_model, nfft_stft, max_stft_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, _tokenizer):
        super(FUNASR_NANO_ENCODER, self).__init__()
        self.funasr_nano = funasr_nano.float()
        self.stft_model = stft_model
        self.T_lfr = lfr_len
        self.lfr_n = lfr_n
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32).view(1, 1, -1)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=max_stft_len + self.lfr_m_factor - 1).to(torch.int16)
        self.output_size_factor = self.funasr_nano.audio_encoder.output_size() ** 0.5
        self.position_encoding = self.funasr_nano.audio_encoder.embed(torch.zeros([1, max_stft_len, 560], dtype=torch.float32))
        num_head = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.h
        head_dim = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.d_k
        self.pad_zeros = torch.zeros((1, num_head * head_dim, 5), dtype=torch.float32)
        factor = float(head_dim ** (-0.25))
        self.total_encoders = list(self.funasr_nano.audio_encoder.encoders0) + list(self.funasr_nano.audio_encoder.encoders) + list(self.funasr_nano.audio_encoder.tp_encoders)
        in_size = self.funasr_nano.audio_encoder.encoders._modules["0"].in_size
        for encoder_layer in self.total_encoders:
            encoder_layer.self_attn.linear_q_k_v.weight.data[:-in_size] *= factor
            encoder_layer.self_attn.linear_q_k_v.bias.data[:-in_size] *= factor

        num_head = self.funasr_nano.audio_adaptor.blocks._modules["0"].self_attn.h
        head_dim = self.funasr_nano.audio_adaptor.blocks._modules["0"].self_attn.d_k
        factor = float(head_dim ** (-0.25))
        for block in self.funasr_nano.audio_adaptor.blocks:
            block.self_attn.linear_q.weight.data *= factor
            block.self_attn.linear_q.bias.data *= factor
            block.self_attn.linear_k.weight.data *= factor
            block.self_attn.linear_k.bias.data *= factor

        head_ids = _tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n", return_tensors="pt")
        tail_ids = _tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
        self.head_embed = self.funasr_nano.llm.model.embed_tokens(head_ids)
        self.tail_embed = self.funasr_nano.llm.model.embed_tokens(tail_ids)
        self.fake_token = torch.zeros(max_stft_len + 1, dtype=torch.int16)
        for i in range(self.fake_token.shape[0]):
            self.fake_token[i] = (((i - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1

    def forward(self, audio, query_embed):
        audio = audio.float()
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[..., :1], audio[..., 1:] - self.pre_emphasis * audio[..., :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = (torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2) + 1e-7).log()
        features_len = mel_features.shape[1].unsqueeze(0)
        left_padding = mel_features[:, [0]]
        padded_inputs = torch.cat([left_padding] * self.lfr_m_factor + [mel_features], dim=1)
        _len = features_len // self.lfr_n - 1
        mel_features = padded_inputs[:, self.indices_mel[:_len].int()].reshape(1, _len, -1)
        x = mel_features * self.output_size_factor + self.position_encoding[:, :_len].float()
        for encoder_layer in self.funasr_nano.audio_encoder.encoders0 + self.funasr_nano.audio_encoder.encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            q_h, k_h, v = torch.split(qkv, encoder_layer.size, dim=-1)
            q_h = q_h.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            k_h = k_h.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0)
            v_h = v.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            fsmn_memory = encoder_layer.self_attn.fsmn_block(torch.cat([self.pad_zeros, v.transpose(1, 2), self.pad_zeros], dim=-1)).transpose(1, 2) + v
            attn = torch.matmul(torch.softmax(torch.matmul(q_h, k_h), dim=-1), v_h).transpose(0, 1).contiguous().view(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            if encoder_layer.in_size == encoder_layer.size:
                x += attn
            else:
                x = attn
            x = x + encoder_layer.feed_forward(encoder_layer.norm2(x))
        x = self.funasr_nano.audio_encoder.after_norm(x)
        for encoder_layer in self.funasr_nano.audio_encoder.tp_encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            q_h, k_h, v = torch.split(qkv, encoder_layer.size, dim=-1)
            q_h = q_h.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            k_h = k_h.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0)
            v_h = v.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            fsmn_memory = encoder_layer.self_attn.fsmn_block(torch.cat([self.pad_zeros, v.transpose(1, 2), self.pad_zeros], dim=-1)).transpose(1, 2) + v
            attn = torch.matmul(torch.softmax(torch.matmul(q_h, k_h), dim=-1), v_h).transpose(0, 1).contiguous().view(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            x += attn
            x = x + encoder_layer.feed_forward(encoder_layer.norm2(x))
        x = self.funasr_nano.audio_encoder.tp_norm(x)
        x = self.funasr_nano.audio_adaptor.linear1(x)
        x = self.funasr_nano.audio_adaptor.relu(x)
        x = self.funasr_nano.audio_adaptor.linear2(x)
        for block in self.funasr_nano.audio_adaptor.blocks:
            x1 = block.norm1(x)
            q = block.self_attn.linear_q(x1).view(-1, block.self_attn.h, block.self_attn.d_k).transpose(0, 1)
            k = block.self_attn.linear_k(x1).view(-1, block.self_attn.h, block.self_attn.d_k).permute(1, 2, 0)
            v = block.self_attn.linear_v(x1).view(-1, block.self_attn.h, block.self_attn.d_k).transpose(0, 1)
            attn = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v).transpose(0, 1).contiguous().view(1, -1, block.self_attn.linear_out.in_features)
            attn = block.self_attn.linear_out(attn)
            x += attn
            x = x + block.feed_forward(block.norm2(x))
        x = x[:, :self.fake_token[features_len].to(torch.int64)]
        concat_embed = torch.cat([self.head_embed, query_embed, x, self.tail_embed], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


class FUNASR_NANO_DECODER_EMBED(torch.nn.Module):
    def __init__(self, funasr_nano):
        super(FUNASR_NANO_DECODER_EMBED, self).__init__()
        self.funasr_nano_decoder_embed = funasr_nano.llm.model.embed_tokens.float()
        
    def forward(self, input_ids):
        return self.funasr_nano_decoder_embed(input_ids)


class FUNASR_NANO_DECODER_MAIN(torch.nn.Module):
    def __init__(self, funasr_nano, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers):
        super(FUNASR_NANO_DECODER_MAIN, self).__init__()
        self.funasr_nano_decoder_main = funasr_nano.llm.float()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)
        self.scale_factor = float(head_dim ** -0.25)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        idx_theta = (position_ids * self.funasr_nano_decoder_main.model.rotary_emb.inv_freq).unsqueeze(0).unsqueeze(0)
        cos_rotary_pos_emb = torch.cos(idx_theta) * self.scale_factor
        sin_rotary_pos_emb = torch.sin(idx_theta) * self.scale_factor
        self.cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).half()
        self.sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).half()

        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

    def rotate_half(self, x, head_dim_half, dim):
        x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
        return torch.cat((-x2, x1), dim=dim)

    def repeat_k(self, kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
        return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, head_dim, -1)

    def repeat_v(self, kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
        return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, -1, head_dim)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-4]
        history_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2).unsqueeze(0)
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for i, layer in enumerate(self.funasr_nano_decoder_main.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(batch_size, -1, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim)
            v = layer.self_attn.v_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 3)
            q = (layer.self_attn.q_norm.weight * (q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).transpose(1, 2)
            k = (layer.self_attn.k_norm.weight * (k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).permute(0, 3, 2, 4, 1)
            q = q * rotary_pos_emb_cos_q + self.rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + self.rotate_half(k, self.head_dim_half, -2) * rotary_pos_emb_sin_k
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            k = self.repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
            v = self.repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, layer.self_attn.o_proj.in_features))
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        hidden_states = self.funasr_nano_decoder_main.model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        logits = self.funasr_nano_decoder_main.lm_head(hidden_states)
        return *self.save_key, *self.save_value, logits, kv_seq_len


# =========================================================================
# 主流程
# =========================================================================

def export():
    print('\nExport start ...\n')
    
    # 强制在 CPU 上运行
    device = "cpu"
    
    with torch.inference_mode():
        custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
        
        # 加载完整模型
        # 注意: FunASR AutoModel 会加载 model.pt (包含 LLM 和 Encoder)
        model = AutoModel(
            model=model_path,
            trust_remote_code=True,
            remote_code="./Fun-ASR/model.py", 
            device=device,
            disable_update=True
        )

        num_heads = model.model.llm.config.num_attention_heads
        num_key_value_heads = model.model.llm.config.num_key_value_heads
        head_dim = model.model.llm.config.head_dim
        num_layers = model.model.llm.config.num_hidden_layers
        vocab_size = model.model.llm.model.vocab_size
        hidden_size = model.model.llm.model.embed_tokens.embedding_dim
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # 1. Export Audio Encoder
        print(f"Exporting Encoder to {onnx_model_A} ...")
        funasr_nano_encoder = FUNASR_NANO_ENCODER(model.model, custom_stft, NFFT_STFT, MAX_STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, tokenizer)
        audio = torch.ones((1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=torch.int16)
        query_embed = torch.ones((1, 10, hidden_size), dtype=torch.float32)
        torch.onnx.export(
            funasr_nano_encoder,
            (audio, query_embed),
            onnx_model_A,
            input_names=['audio', 'query_embed'],
            output_names=['concat_embed', 'ids_len'],
            do_constant_folding=True,
            dynamic_axes={
                'audio': {2: 'audio_len'},
                'query_embed': {1: 'num_token'},
                'concat_embed': {1: 'num_token'}
            } if DYNAMIC_AXES else None,
            opset_version=OPSET,
            dynamo=False
        )
        del funasr_nano_encoder, audio, custom_stft
        gc.collect()

        # 2. Export Decoder Embedder (for Prompt)
        print(f"Exporting Decoder Embedder to {onnx_model_B} ...")
        batch_size = 3 # 示例e
        ids_len = torch.tensor([10], dtype=torch.long)      
        history_len = torch.tensor([0], dtype=torch.long)
        input_ids = torch.ones((1, ids_len), dtype=torch.int32)
        hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
        attention_mask = torch.tensor([1], dtype=torch.int8)
        past_keys = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, 0), dtype=torch.float32)
        past_values = torch.zeros((batch_size, num_key_value_heads, 1, 0, head_dim), dtype=torch.float32)
        kv_seq_len = history_len + ids_len

        model_B = FUNASR_NANO_DECODER_EMBED(model.model)
        torch.onnx.export(
            model_B,
            (input_ids,),
            onnx_model_B,
            input_names=['input_ids'],
            output_names=['hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'ids_len'},
                'hidden_states': {0: 'batch', 1: 'ids_len'}
            },
            do_constant_folding=True,
            opset_version=OPSET,
            dynamo=False
        )
        del model_B, input_ids
        
        # # 3. Export Decoder Main
        # print(f"Exporting Decoder Main to {onnx_model_C} ...")
        # all_inputs = []
        # input_names = []
        # output_names = []
        # dynamic_axes = {'hidden_states': {0: 'batch', 1: 'ids_len'}}
        # for i in range(num_layers):
        #     name = f'in_key_{i}'
        #     input_names.append(name)
        #     all_inputs.append(past_keys)
        #     dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        #     name = f'out_key_{i}'
        #     output_names.append(name)
        #     dynamic_axes[name] = {0: 'batch', 4: 'ks_seq_len'}
        # for i in range(num_layers):
        #     name = f'in_value_{i}'
        #     input_names.append(name)
        #     all_inputs.append(past_values)
        #     dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        #     name = f'out_value_{i}'
        #     output_names.append(name)
        #     dynamic_axes[name] = {0: 'batch', 3: 'ks_seq_len'}
        # input_names.append('hidden_states')
        # all_inputs.append(hidden_states)
        # input_names.append('history_len')
        # all_inputs.append(history_len)
        # input_names.append('ids_len')
        # all_inputs.append(ids_len)
        # input_names.append('attention_mask')
        # all_inputs.append(attention_mask)
        # output_names.append('logits')
        # output_names.append('kv_seq_len')
        # dynamic_axes['logits'] = {0: 'batch'}

        # model_C = FUNASR_NANO_DECODER_MAIN(model.model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers)
        # torch.onnx.export(
        #     model_C,
        #     tuple(all_inputs),
        #     onnx_model_C,
        #     input_names=input_names,
        #     output_names=output_names,
        #     dynamic_axes=dynamic_axes,
        #     do_constant_folding=True,
        #     opset_version=OPSET,
        #     dynamo=False
        # )
        # del model_C, input_names, output_names, dynamic_axes, all_inputs
        # gc.collect()
        
        # # 4. Export Greedy Search
        # print(f"Exporting Greedy Search to {onnx_model_D} ...")
        # greedy = GREEDY_SEARCH()
        # beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        # repeat_penality = torch.ones((beam_size, vocab_size), dtype=torch.float32)
        # logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)
        # penality_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)

        # torch.onnx.export(
        #     greedy,
        #     (logits, repeat_penality, penality_value, beam_size),
        #     onnx_model_D,
        #     input_names=['logits', 'repeat_penality_in', 'penality_value', 'batch_size'],
        #     output_names=['max_logits_idx', 'repeat_penality_out'],
        #     dynamic_axes={
        #         'logits': {0: 'batch'},
        #         'repeat_penality_in': {0: 'batch'},
        #         'repeat_penality_out': {0: 'batch'},
        #         'max_logits_idx': {0: 'batch'}
        #     },
        #     do_constant_folding=True,
        #     opset_version=OPSET,
        #     dynamo=False
        # )
        # del greedy

        # print("Exporting Beam Search modules (Optional) ...")
        # # (简单起见，这里也导出，反正脚本都写了)
        # pass 
        
        print('\nAll ONNX models exported successfully to:', OUTPUT_DIR)
        
        # # 5. Clean up External Data (Optional but recommended)
        # try:
        #     import onnx
        #     print("\nConsolidating external data for large models...")
            
        #     # 对 Decoder Main 进行整理
        #     if os.path.exists(onnx_model_C):
        #         print(f"Processing {onnx_model_C} ...")
        #         model_c = onnx.load(onnx_model_C)
        #         onnx.save_model(
        #             model_c,
        #             onnx_model_C,
        #             save_as_external_data=True,
        #             all_tensors_to_one_file=True,
        #             location="FunASR_Nano_Decoder_Main.data",
        #             size_threshold=1024,
        #             convert_attribute=False
        #         )
        #         print(f"Consolidated external data to FunASR_Nano_Decoder_Main.data")
                
        #     # 对 Encoder 进行整理 (如果大于 2GB 也会有这个问题，虽然这里可能刚好没超，但整理一下更好)
        #     if os.path.exists(onnx_model_A):
        #          # 检查一下文件大小，如果很大才处理，或者统一处理
        #          file_size = os.path.getsize(onnx_model_A)
        #          if file_size > 100 * 1024 * 1024: # > 100MB
        #             print(f"Processing {onnx_model_A} ...")
        #             model_a = onnx.load(onnx_model_A)
        #             onnx.save_model(
        #                 model_a,
        #                 onnx_model_A,
        #                 save_as_external_data=True,
        #                 all_tensors_to_one_file=True,
        #                 location="FunASR_Nano_Encoder.data",
        #                 size_threshold=1024,
        #                 convert_attribute=False
        #             )
        #             print(f"Consolidated external data to FunASR_Nano_Encoder.data")

        # except ImportError:
        #     print("Warning: 'onnx' library not found. Skipping external data consolidation.")
        #     print("To enable cleaner output, please install onnx: pip install onnx")
        # except Exception as e:
        #     import traceback
        #     traceback.print_exc()
        #     print(f"Error during consolidation: {e}")

        # # 6. Delete Residual Files (The ones created by torch.onnx.export)
        # # 只有在上面的整理步骤成功后，或者确认存在 .data 文件时才执行删除，避免误删
        # print("\nCleaning up residual external data files...")
        # cleanup_count = 0
        # for filename in os.listdir(OUTPUT_DIR):
        #     if filename.startswith("onnx__MatMul_"):
        #         file_path = os.path.join(OUTPUT_DIR, filename)
        #         try:
        #             os.remove(file_path)
        #             cleanup_count += 1
        #         except OSError as e:
        #             print(f"Error deleting {filename}: {e}")
        
        # if cleanup_count > 0:
        #     print(f"Deleted {cleanup_count} residual external weight files.")
        # else:
        #     print("No residual files found or deleted.")

    print('\nExport flow finished.')

if __name__ == "__main__":
    export()
