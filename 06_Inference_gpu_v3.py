
import pickle
import numpy as np
import logging
import ctypes
import time
import os
import sys
from llama_cpp import (
    Llama,
    llama_batch_init,
    llama_batch_free,
    llama_decode,
    llama_get_logits,
    llama_kv_self_clear,
)

# =========================================================================
# 最终配置 (基于测试结果调优)
# =========================================================================

# 黄金数值：8
# 经测试，Vulkan 后端在此硬件上一次注入 > 8 个 Embedding 会导致 NaN
CHUNK_SIZE = 8 

# 数据类型
# 虽然测试 False 也能跑，但为了防止潜在的内存错位，标准工程实现建议 True
# 这里设为 True 以保证长期稳定性，你可以随时改回 False
FORCE_FLOAT32 = True 

GGUF_MODEL_PATH = r'./model-gguf/qwen3-0.6b-asr.gguf'
PICKLE_PATH = r'./pickles/embedding_slice_0_160000.pkl' 

MAX_SEQ_LEN = 1024
STOP_TOKEN = [151643, 151645]

# =========================================================================
# 主程序
# =========================================================================

def main():
    # 1. 自动寻找最新的 pickle
    target_pickle = PICKLE_PATH
    if not os.path.exists(target_pickle):
        if os.path.exists("pickles"):
            files = [os.path.join("pickles", f) for f in os.listdir("pickles") if f.endswith(".pkl")]
            if files: target_pickle = max(files, key=os.path.getctime)
    
    print(f'\n📂 Loading Pickle: {target_pickle}')
    with open(target_pickle, 'rb') as f:
        embeddings = pickle.load(f)
    
    # 数据预处理
    embeds = embeddings.squeeze()
    if len(embeds.shape) == 1: embeds = embeds.reshape(1, -1)
    
    if FORCE_FLOAT32:
        embeds = embeds.astype(np.float32)
        dtype_str = "float32 (Converted)"
    else:
        dtype_str = f"{embeds.dtype} (Raw)"

    n_tokens, n_dim = embeds.shape
    print(f"📊 Data Shape: {embeds.shape} | Dtype: {dtype_str}")

    # 2. 加载模型
    print(f'🤖 Loading Model: {GGUF_MODEL_PATH}')
    # 注意：n_ubatch 设为 64 是安全的，因为我们手动控制了 batch 大小为 8
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=MAX_SEQ_LEN + 2048,
        n_batch=2048,
        n_ubatch=64, 
        n_gpu_layers=-1,
        embedding=True,
        verbose=False 
    )
    print("✅ Model Loaded (Vulkan Backend Active)")

    # 3. 推理过程
    ctx = llm.ctx
    llama_kv_self_clear(ctx)
    llm.n_tokens = 0
    
    # 初始化 Batch
    batch_embd = llama_batch_init(2048, n_dim, 1)
    batch_embd.token = ctypes.cast(None, ctypes.POINTER(ctypes.c_int32)) # 标记为 embedding
    
    batch_text = llama_batch_init(1, 0, 1)

    try:
        print(f"\n🚀 Start Injection (Chunk Size: {CHUNK_SIZE})...")
        inject_start = time.time()
        
        # --- 核心循环：分块注入 ---
        for i in range(0, n_tokens, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, n_tokens)
            current_len = end - i
            
            # 准备数据切片
            chunk_data = embeds[i:end]
            if not chunk_data.flags['C_CONTIGUOUS']:
                chunk_data = np.ascontiguousarray(chunk_data)
                
            # 设置 Batch
            batch_embd.n_tokens = current_len
            for k in range(current_len):
                batch_embd.pos[k] = i + k
                batch_embd.n_seq_id[k] = 1
                batch_embd.seq_id[k][0] = 0
                # 仅在整个序列的最后一个 token 开启 logits
                is_last = (i + k == n_tokens - 1)
                batch_embd.logits[k] = 1 if is_last else 0
            
            # 内存拷贝
            ctypes.memmove(batch_embd.embd, chunk_data.ctypes.data, chunk_data.nbytes)
            
            # 解码
            if llama_decode(ctx, batch_embd) != 0:
                print(f"❌ Error during injection at index {i}")
                return
            
            llm_obj = llm # 别名
            llm_obj.n_tokens += current_len
            
            # 简易进度条
            if i % 32 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
                
        inject_time = time.time() - inject_start
        print(f"\n✅ Injection Done. Time: {inject_time:.4f}s (Avg: {n_tokens/inject_time:.1f} t/s)")

        # --- 文本生成 ---
        print("\n📝 Generating Text:")
        print("-" * 40)
        
        vocab_size = llm.n_vocab()
        gen_start = time.time()
        gen_count = 0
        eos_token = llm.token_eos()
        
        full_text = ""
        
        for _ in range(MAX_SEQ_LEN):
            # 获取 Logits
            logits = np.ctypeslib.as_array(llama_get_logits(ctx), shape=(vocab_size,))
            token_id = int(np.argmax(logits))
            
            if token_id == eos_token or token_id in STOP_TOKEN:
                break
            
            # 打印字符
            try:
                txt = llm.detokenize([token_id]).decode('utf-8', errors='ignore')
                print(txt, end="", flush=True)
                full_text += txt
                gen_count += 1
            except:
                pass
            
            # 下一步
            batch_text.n_tokens = 1
            batch_text.token[0] = token_id
            batch_text.pos[0] = llm.n_tokens
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1
            
            if llama_decode(ctx, batch_text) != 0:
                break
            llm.n_tokens += 1
            
        print("\n" + "-" * 40)
        gen_time = time.time() - gen_start
        print(f"⚡ Generation Speed: {gen_count/gen_time:.2f} tokens/s")
        
    finally:
        llama_batch_free(batch_embd)
        llama_batch_free(batch_text)

if __name__ == "__main__":
    main()