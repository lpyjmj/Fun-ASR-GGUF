"""
ASR 演示脚本 - 简单直接的使用示例
"""

import os
import time
import threading
import psutil
from pydub import AudioSegment
from fun_asr_gguf import create_asr_engine


# ==================== Vulkan 选项 ====================

# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"   # 禁止 Vulkan 用独显（强制用集显）
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）


# ==================== 配置区域 ====================

# 音频文件路径
audio_file = "vad_example.wav"

# 语言设置（None=自动检测, "中文", "英文", "日文" 等）
language = None

# 上下文信息（留空则不使用）
context = "这是1004期睡前消息节目，主持人叫督工，助理叫静静"

# 是否启用 CTC 辅助（True=提供时间戳和热词, False=仅LLM）
enable_ctc = True

# 是否打印详细信息
verbose = True

# 是否以 JSON 格式输出结果
json_output = False

# 模型文件路径
model_dir = "./model"
encoder_onnx_path = f"{model_dir}/Fun-ASR-Nano-Encoder-Adaptor.fp16.onnx"
ctc_onnx_path = f"{model_dir}/Fun-ASR-Nano-CTC.fp16.onnx"
decoder_gguf_path = f"{model_dir}/Fun-ASR-Nano-Decoder.q8_0.gguf"
tokens_path = f"{model_dir}/tokens.txt"
hotwords_path = "./hot.txt"  # 可选，留空则不使用热词

# ==================== 语言说明 ====================

"""
Fun-ASR-Nano-2512
    中文、英文、日文
     
Fun-ASR-MLT-Nano-2512
    中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
    印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
    匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
    斯洛伐克语、斯洛文尼亚语、瑞典语 
"""

# ==================== 监控工具 ====================

class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.peak_memory = 0
        self.thread = None
        self.process = psutil.Process(os.getpid())

    def start(self):
        self.running = True
        self.peak_memory = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        return self.peak_memory

    def _monitor(self):
        while self.running:
            try:
                current_memory = self.process.memory_info().rss
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                time.sleep(self.interval)
            except Exception:
                break

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def get_device_info():
    """推断当前的计算设备"""
    # 检查是否显式禁用了 Vulkan
    if os.environ.get("VK_ICD_FILENAMES") == "none":
        return "CPU"
    
    # 检查是否强制指定了设备（通常也是启用 Vulkan）
    if os.environ.get("GGML_VK_VISIBLE_DEVICES") == "0":
        return "GPU (Vulkan - Device 0)"

    # 默认情况
    return "GPU (Vulkan)"

# ==================== 执行区域 ====================

def main():

    print("="*70)
    print("ASR 语音识别")
    print("="*70)

    # 创建 ASR 引擎
    engine = create_asr_engine(
        encoder_onnx_path=encoder_onnx_path,
        ctc_onnx_path=ctc_onnx_path,
        decoder_gguf_path=decoder_gguf_path,
        tokens_path=tokens_path,
        hotwords_path=hotwords_path, 
        similar_threshold=0.6, 
        max_hotwords=10, 
        enable_ctc=enable_ctc,
        verbose=verbose,
    )

    print(f'\n预跑一遍，分配内存......\n')
    result = engine.transcribe(
        audio_file, 
        language=language, 
        context=context, 
        verbose=False, 
        duration=5.0,
    )

    print(f'\n开始正式转录...\n')
    
    # 启动内存监控
    monitor = MemoryMonitor()
    monitor.start()
    
    start_time = time.time()
    
    result = engine.transcribe(
        audio_file, 
        language=language, 
        context=context, 
        verbose=verbose,
        segment_size=60.0,
        overlap=4.0,
        start_second=0.0,
        # duration=60.0,  # 暂时不要限制音频时长60s
        srt=True, 
        temperature=0.4
    )
    
    end_time = time.time()
    peak_memory = monitor.stop()
    
    # 计算统计信息
    total_time = end_time - start_time
    audio_duration = 0.0
    
    # 尝试从结果中获取音频时长（如果有），或者使用预设的60s（如果转录的是整段）
    # 注意：engine.transcribe 返回的 result 是一个 dict，通常包含 text, timestamps 等
    # 这里我们简单地使用 time.time 差值作为处理时间，
    # 音频时长我们可以尝试从 result 中推断，或者如果 result 没包含总时长，就只能手动指定或解析音频文件
    # FunASR-GGUF 的 transcribe 可能会在日志里打印音频长度，但 result 结构需要确认
    # 我们这里直接使用 60.0 或者尝试读取 result
    
    # 获取音频实际时长用于计算 RTF
    audio_duration = AudioSegment.from_file(audio_file).duration_seconds
    
    rtf = total_time / audio_duration if audio_duration > 0 else 0
    device_info = get_device_info()
    
    print("\n" + "="*70)
    print("性能监控报告")
    print("="*70)
    print(f"音频时长:     {audio_duration:.2f} s")
    print(f"转录耗时:     {total_time:.2f} s")
    print(f"RTF (实时率): {rtf:.4f}")
    print(f"峰值内存:     {format_size(peak_memory)}")
    print(f"推理设备:     {device_info}")
    print("="*70 + "\n")

    # 输出结果
    if json_output:
        import json
        print("\n" + "="*70)
        print("识别结果 (JSON)")
        print("="*70)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 清理资源
    engine.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
