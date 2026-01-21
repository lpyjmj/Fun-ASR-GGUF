"""检查 token ID 对应的文本 - 使用 llama-cpp-python"""
from llama_cpp import Llama
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.gguf')

# 加载模型（只需要 vocab）
llm = Llama(model_path=MODEL_PATH, n_ctx=32, verbose=False)

# 检查生成的 token IDs
generated_tokens = [6606, 114, 3837, 102600, 8903, 3837, 100437, 50009, 50930, 108035, 99822]

print("Token ID -> Token Text")
print("-" * 50)
for tok_id in generated_tokens:
    try:
        text = llm.detokenize([tok_id]).decode('utf-8', errors='replace')
        raw_bytes = llm.detokenize([tok_id])
        print(f"{tok_id:6d} -> bytes: {raw_bytes.hex():20s} text: {repr(text)}")
    except Exception as e:
        print(f"{tok_id:6d} -> error: {e}")

# 尝试组合前两个 token
print("\n前两个 token 组合:")
try:
    t1_bytes = llm.detokenize([6606])
    t2_bytes = llm.detokenize([114])
    combined = t1_bytes + t2_bytes
    print(f"Token 6606: bytes={t1_bytes.hex()}, text={repr(t1_bytes.decode('utf-8', errors='replace'))}")
    print(f"Token 114:  bytes={t2_bytes.hex()}, text={repr(t2_bytes.decode('utf-8', errors='replace'))}")
    print(f"Combined:   bytes={combined.hex()}, text={repr(combined.decode('utf-8', errors='replace'))}")
except Exception as e:
    print(f"Error: {e}")
