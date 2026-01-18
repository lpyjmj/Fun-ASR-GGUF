import torch
import os
import json
import shutil
import subprocess
import sys

# =========================================================================
# 配置部分
# =========================================================================

# 源模型路径
SOURCE_MODEL_PATH = './Fun-ASR-Nano-2512'
CONFIG_PATH = f'{SOURCE_MODEL_PATH}/Qwen3-0.6B/config.json'

# 中间产物 (HF 格式) 输出路径
OUTPUT_HF_DIR = './model-gguf/Qwen3-0.6B'
# Tokenizer 输出路径
OUTPUT_TOKENIZER_DIR = './model-gguf/Qwen3-0.6B'

# 最终 GGUF 输出文件
OUTPUT_GGUF_FILE = './model-gguf/qwen3-0.6b-asr.gguf'

# Llama.cpp 路径 (自动寻找 convert_hf_to_gguf.py)
# 优先尝试 llama.cpp 目录，如果不存在尝试 llama.cpp-master
if os.path.exists('./llama.cpp/convert_hf_to_gguf.py'):
    LLAMA_CPP_PATH = './llama.cpp'
elif os.path.exists('./llama.cpp-master/convert_hf_to_gguf.py'):
    LLAMA_CPP_PATH = './llama.cpp-master'
else:
    # Fallback to current directory or error later
    LLAMA_CPP_PATH = './llama.cpp' 

CONVERT_SCRIPT = f'{LLAMA_CPP_PATH}/convert_hf_to_gguf.py'


def main():
    # ---------------------------------------------------------------------
    # 1. 提取 LLM 并保存为 Hugging Face 格式
    # ---------------------------------------------------------------------
    print("\n[Stage 1] Extracting LLM Decoder to Hugging Face format...")
    
    # 尝试导入 Qwen3 类 (参考 save_standard_hf_model.py)
    try:
        from transformers import Qwen3ForCausalLM, Qwen3Config
        print("Successfully imported Qwen3ForCausalLM and Qwen3Config")
    except ImportError:
        print("Warning: Qwen3 classes not found in transformers, falling back to Qwen2 or AutoClasses.")
        try:
             from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM
             from transformers import Qwen2Config as Qwen3Config
        except ImportError:
             from transformers import AutoModelForCausalLM as Qwen3ForCausalLM
             from transformers import AutoConfig as Qwen3Config

    # 加载完整 PyTorch 模型 (FunASR 格式)
    model_pt_path = f'{SOURCE_MODEL_PATH}/model.pt'
    print(f"Loading full model from {model_pt_path} ...")
    full_model = torch.load(model_pt_path, map_location='cpu')

    # 提取 LLM 权重
    llm_weights = {}
    print("Extracting LLM weights...")
    for key in full_model.keys():
        if key.startswith('llm.'):
            # 将键名从 llm.model.xxx 转换为 model.xxx (HF 标准格式)
            hf_key = key.replace('llm.', '')
            llm_weights[hf_key] = full_model[key]
    
    print(f"Extracted {len(llm_weights)} weight keys.")
    del full_model
    
    # 加载配置
    print(f"Loading config from {CONFIG_PATH} ...")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
        
    config = Qwen3Config(**config_dict)
    
    # 初始化空模型
    print("Initializing empty Qwen3ForCausalLM...")
    qwen_model = Qwen3ForCausalLM(config)

    # 加载权重
    print("Loading state dict into LLM...")
    qwen_model.load_state_dict(llm_weights, strict=True)
    
    # 保存 HF 模型 (Safetensors)
    os.makedirs(OUTPUT_HF_DIR, exist_ok=True)
    print(f"Saving HF model to {OUTPUT_HF_DIR} ...")
    qwen_model.save_pretrained(OUTPUT_HF_DIR, safe_serialization=True)
    
    # 复制 tokenizer 文件到单独目录
    print(f"Copying tokenizer files to {OUTPUT_TOKENIZER_DIR} ...")
    os.makedirs(OUTPUT_TOKENIZER_DIR, exist_ok=True)
    original_tokenizer_dir = f'{SOURCE_MODEL_PATH}/Qwen3-0.6B'
    files_to_copy = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json']
    for file in files_to_copy:
        src = os.path.join(original_tokenizer_dir, file)
        dst = os.path.join(OUTPUT_TOKENIZER_DIR, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  Copied {file}")
    
    print("HF Model and Tokenizer saved successfully.")

    # ---------------------------------------------------------------------
    # 2. 转换为 GGUF 格式
    # ---------------------------------------------------------------------
    print("\n[Stage 2] Converting HF model to GGUF...")
    
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"Error: Llama.cpp conversion script not found at {CONVERT_SCRIPT}")
        # Try finding it
        print("Attempting to locate convert_hf_to_gguf.py ...")
        # simple search
        possible_paths = [
            './llama.cpp/convert_hf_to_gguf.py',
            './llama.cpp-master/convert_hf_to_gguf.py',
            './llama.cpp/llama.cpp/convert_hf_to_gguf.py'
        ]
        for p in possible_paths:
            if os.path.exists(p):
                print(f"Found at {p}")
                con_script = p
                break
        else:
             print("Could not find conversion script. Please check your llama.cpp installation.")
             return
    else:
        con_script = CONVERT_SCRIPT

    # 构建命令
    # python convert.py model_dir --outfile ... --outtype f16 --vocab-dir tokenizer_dir
    
    cmd = [
        sys.executable,
        con_script,
        OUTPUT_HF_DIR,
        '--outfile', OUTPUT_GGUF_FILE,
        '--outtype', 'f16',
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ GGUF conversion successful! Output: {OUTPUT_GGUF_FILE}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ GGUF conversion failed with error: {e}")

if __name__ == "__main__":
    main()
