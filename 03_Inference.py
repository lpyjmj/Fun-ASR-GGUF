import ctypes
import numpy as np
import logging
import time
import onnxruntime
import torch
import torchaudio
from pydub import AudioSegment
from transformers import AutoTokenizer
from llama_cpp import (
    Llama,
    llama_batch_init,
    llama_batch_free,
    llama_decode,
    llama_get_logits,
    llama_kv_self_clear,  # 新版 API：清理缓存
)

# =========================================================================
# 配置部分
# =========================================================================

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_refine.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型路径
model_dir = r'./model-gguf'
tokenizer_path = f'{model_dir}/Qwen3-0.6B'

# ONNX 模型
onnx_encoder = f'{model_dir}/FunASR_Nano_Encoder.onnx'       # Audio Encoder
onnx_embed = f'{model_dir}/FunASR_Nano_Decoder_Embed.onnx'  # Text/Prompt Embedder

# GGUF 模型 (用于解码)
gguf_model_path = f'{model_dir}/qwen3-0.6b-asr.gguf'

# 输入音频
test_audio = r'./input.mp3'
task_prompt = ["将语音转写成中文："]  # 任务提示词 (在 Encoder 阶段融合)

# 音频处理参数 (需与模型训练时一致)
SAMPLE_RATE = 16000
USE_NORMALIZER = True
MAX_INPUT_AUDIO_LENGTH = 320000 
SLIDING_WINDOW = 0 # 0 表示根据音频长度自动分段 (这里简化为一次性处理或整段)

# 模型参数
MAX_SEQ_LEN = 1024
STOP_TOKEN = [151643, 151645] # Qwen 的特殊停止 Token
MAX_THREADS = 0 # 0 = Auto

# =========================================================================
# 辅助函数
# =========================================================================

def normalizer(_audio, target_value=8192.0):
    """音频归一化处理"""
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)

def decode_with_pure_embeddings(llm_obj, audio_embeddings, max_new_tokens=200):
    """
    纯 Embedding 解码函数 (Pure Embedding Decoding)
    
    原理：
    FunASR 的 Encoder 输出的 Embedding 已经融合了音频特征和任务 Prompt 的语义。
    因此，我们直接将这个 "Fused Embedding" 注入到 LLM 中，不需要额外拼接 Text Prefix (如 <|im_start|>)。
    这避免了 "Double Prompt" 导致的分布不匹配问题。
    
    参数:
    llm_obj: Llama 对象
    audio_embeddings: Numpy 数组 (Shape: [seq_len, 1024])
    max_new_tokens: 最大生成长度
    """
    
    # 1. 准备数据
    embeds = audio_embeddings.squeeze()
    if len(embeds.shape) == 1:
        embeds = embeds.reshape(1, -1)
    
    n_audio_tokens, n_dim = embeds.shape
    logger.info(f"注入 Audio Embeddings Shape: {embeds.shape}")

    # 2. 初始化 Batch
    # batch_embd: 用于存放 Audio Embeddings (embd=n_dim, token=NULL)
    batch_embd = llama_batch_init(n_audio_tokens, n_dim, 1)        
    
    # batch_text: 用于存放生成的 Token IDs (embd=0, token=分配内存)
    batch_text = llama_batch_init(1, 0, 1)

    ctx = llm_obj.ctx
    
    # 3. 清理上下文缓存 (KV Cache)
    llama_kv_self_clear(llm_obj.ctx) 
    
    try:
        # ---------------------------------------------------------------------
        # 阶段 A: 注入融合 Embedding (Audio + Prompt fused)
        # ---------------------------------------------------------------------
        logger.info("正在注入融合 Embedding...")
        
        batch_embd.n_tokens = n_audio_tokens
        llm_obj.n_tokens = 0 # 重置 LLM 内部计数器
        
        for i in range(n_audio_tokens):
            batch_embd.pos[i] = i
            batch_embd.n_seq_id[i] = 1
            batch_embd.seq_id[i][0] = 0
            
            # 只在最后一个 Token 开启 Logits 计算，用于预测第一个生成的文本 Token
            batch_embd.logits[i] = 1 if i == n_audio_tokens - 1 else 0

        # 使用 ctypes.memmove 高效拷贝 Numpy 数据到 C 指针
        if not embeds.flags['C_CONTIGUOUS']:
            embeds = np.ascontiguousarray(embeds)
        
        ctypes.memmove(batch_embd.embd, embeds.ctypes.data, embeds.nbytes)
        
        # 执行解码
        if llama_decode(ctx, batch_embd) != 0:
             raise RuntimeError("Audio embedding decoding failed")
        
        llm_obj.n_tokens += n_audio_tokens

        # ---------------------------------------------------------------------
        # 阶段 B: 文本生成 (Greedy Search)
        # ---------------------------------------------------------------------
        generated_text = ""
        logger.info(f"开始生成文本 (最大 {max_new_tokens} tokens)...\n")
        
        eos_token = llm_obj.token_eos()
        vocab_size = llm_obj.n_vocab()
        
        batch_text.n_tokens = 1
        
        gen_start_time = time.time()
        tokens_generated = 0
        
        for step in range(max_new_tokens):
            # 1. 获取 Logits
            logits_ptr = llama_get_logits(ctx)
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            
            # 2. 贪婪采样 (Argmax)
            token_id = int(np.argmax(logits_arr))
            
            # 3. 检查停止条件
            if token_id == eos_token or token_id in STOP_TOKEN:
                break
                
            # 4. 解码 token 为文本
            try:
                text_piece = llm_obj.detokenize([token_id]).decode('utf-8', errors='ignore')
                print(text_piece, end="", flush=True)
                generated_text += text_piece
                tokens_generated += 1
            except Exception:
                pass
                
            # 5. 把生成的 Token 喂回去 (Autoregressive)
            batch_text.token[0] = token_id
            batch_text.pos[0] = llm_obj.n_tokens
            # 必须显式初始化 seq_id，否则会导致随机内存访问错误 (access violation)
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1
            
            if llama_decode(ctx, batch_text) != 0:
                break
            
            llm_obj.n_tokens += 1
            
        print('\n\n')
        gen_duration = time.time() - gen_start_time
        tps = tokens_generated / gen_duration if gen_duration > 0 else 0

        logger.info(f"解码速度: {tps:.2f} tokens/s ({tokens_generated} tokens in {gen_duration:.2f}s)\n\n")
        
    finally:
        # 释放资源
        llama_batch_free(batch_embd)
        llama_batch_free(batch_text)

    return generated_text

# =========================================================================
# 主程序
# =========================================================================

def main():
    print('\nLoading ONNX models...')
    
    # 1. 初始化 ONNX Runtime
    session_opts = onnxruntime.SessionOptions()
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.intra_op_num_threads = MAX_THREADS
    
    # 加载 Audio Encoder 和 Text Embedder
    # 注意：这里我们只需要这两个模型来生成 "Fused Embedding"
    ort_session_A = onnxruntime.InferenceSession(onnx_encoder, sess_options=session_opts, providers=['CPUExecutionProvider'])
    ort_session_B = onnxruntime.InferenceSession(onnx_embed, sess_options=session_opts, providers=['CPUExecutionProvider'])
    
    in_name_A = [x.name for x in ort_session_A.get_inputs()]
    out_name_A = [x.name for x in ort_session_A.get_outputs()]
    in_name_B = ort_session_B.get_inputs()[0].name
    out_name_B = [x.name for x in ort_session_B.get_outputs()]

    shape_value_in_A = ort_session_A._inputs_meta[0].shape[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 2. 加载 GGUF 模型
    print(f'\nLoading GGUF model: {gguf_model_path}')
    llm = Llama(
        model_path=gguf_model_path,
        n_ctx=MAX_SEQ_LEN,
        n_threads=MAX_THREADS,
        embedding=True, # 必须开启，才能分配 embedding 内存
        verbose=False
    )
    print('GGUF model loaded successfully!')

    # 3. 预处理 Prompt
    # 将文本提示词转换为 Embedding
    init_all_outputs_B = []
    for t in task_prompt:
        tokens = tokenizer(t, return_tensors='np')['input_ids'].astype(np.int32)
        input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, 'cpu', 0)
        input_feed_B = {in_name_B: input_ids}
        # 运行 Text Embedder
        init_all_outputs_B.append(ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0])

    # 4. 处理音频并推理
    for prompt_embed, audio_file in zip(init_all_outputs_B, [test_audio]):
        print("-" * 80)
        print(f"Test Input Audio: {audio_file}")
        
        # 加载和归一化音频
        audio = np.array(AudioSegment.from_file(audio_file).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
        if USE_NORMALIZER:
            audio = normalizer(audio, 8192.0)
            
        audio_len = len(audio)
        audio = audio.reshape(1, 1, -1)
        
        # 定义输入长度
        if isinstance(shape_value_in_A, str):
             INPUT_AUDIO_LENGTH = min(MAX_INPUT_AUDIO_LENGTH, audio_len)
        else:
             INPUT_AUDIO_LENGTH = shape_value_in_A
             
        stride_step = INPUT_AUDIO_LENGTH if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
        
        # Padding 逻辑 (保持简单，针对短音频填充静音)
        if audio_len < INPUT_AUDIO_LENGTH:
             pad_len = INPUT_AUDIO_LENGTH - audio_len
             pad_samples = np.zeros((1, 1, pad_len), dtype=audio.dtype) # 简单补零，或者补白噪声
             audio = np.concatenate((audio, pad_samples), axis=-1)
             
        aligned_len = audio.shape[-1]
        
        asr_result = ""
        slice_start = 0
        slice_end = INPUT_AUDIO_LENGTH
        rtf_time = time.time()
        
        # 循环处理音频切片
        while slice_end <= aligned_len:
            # 4.1 运行 ONNX Audio Encoder
            input_feed_A = {}
            input_feed_A[in_name_A[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(audio[..., slice_start: slice_end], 'cpu', 0)
            input_feed_A[in_name_A[1]] = prompt_embed # 注入 Task Prompt Embedding
            
            all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)
            
            # 获取融合 Embedding (Batch, Seq, Dim)
            comprehensive_embedding = all_outputs_A[0].numpy() 
            
            print("\n=== 使用 GGUF 模型解码 (Low-Level API) ===")
            
            try:
                # 4.2 调用 Llama 模型进行解码
                result_text = decode_with_pure_embeddings(
                    llm, 
                    comprehensive_embedding,
                    max_new_tokens=MAX_SEQ_LEN
                )
                asr_result += result_text
                
            except Exception as e:
                logger.error(f"解码发生错误: {e}")
                import traceback
                traceback.print_exc()
            
            # 4.3 更新切片窗口 (防止死循环!)
            slice_start += stride_step
            slice_end = slice_start + INPUT_AUDIO_LENGTH

        # 打印最终统计
        # print(f"\n\nRTF: {((time.time() - rtf_time) / (audio_len / SAMPLE_RATE)):.3f}")

if __name__ == "__main__":
    main()
