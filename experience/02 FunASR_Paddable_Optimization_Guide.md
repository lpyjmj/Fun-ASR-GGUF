# FunASR (SenseVoice) 模型长度感知与 Padding 优化指南

## 1. 背景与目标
在 Windows 环境下使用 **DirectML** 进行推理时，由于算子编译（Operator Fusion/Specialization）的巨大开销，频繁变动的输入长度会导致严重的性能抖动。因此，必须采用 **Bucketing (分桶)** 策略，将音频填充到固定长度（如 30s 或 60s）。

然而，传统的 Padding（在末尾补零）会导致以下问题：
- **均值偏移 (Mean Shift)**：归一化时，静音区域会拉低全局均值。
- **边界泄露 (Boundary Seepage)**：卷积（LFR）或 STFT 窗口在有效信号末尾会“看到”静音，导致特征突变。
- **注意稀释 (Attention Dilution)**：Transformer 会将权重浪费在无效的零向量上。

**目标**：实现“逻辑长度感知”，确保 5s 音频在 30s 载体中的输出，与 5s 原始音频的输出实现**数值级一致（Cosine Similarity > 0.99999）**。

---

## 2. 核心优化策略

### 2.1 音频域“硬清零” (Audio Hard-Zeroing)
这是解决 STFT 阶段误差的第一步，也是最重要的一步。
- **现象**：当 `audio = audio - mean(valid_audio)` 后，原本是 0 的 Padding 区域变成了 `-mean`。这会导致 STFT 窗口在边界处产生非预期的能量。
- **策略**：在 Normalization 和 Pre-emphasis 之后，立即强制执行：
  ```python
  audio[:, :, valid_samples:] = 0.0
  ```
- **效果**：确保 STFT Real/Imag 阶段在有效区间内实现近乎 0 误差。

### 2.2 LFR 复读覆盖策略 (Replicate Padding)
- **挑战**：LFR 拼接时通常需要向右看几帧补齐窗口。如果右边是静音（Log-Mel 约为 -16），会产生巨大的梯度跳变。
- **策略**：将有效长度以外的物理 Tensor 空间，全部填充为有效帧的**最后一帧**内容。
  ```python
  last_frame = mel[:, [valid_end]]
  padding_fill = last_frame.repeat(1, T_phys - T_valid, 1)
  mel_consistent = torch.cat([valid_part, padding_fill], dim=1)
  ```
- **效果**：模拟了短音频原生推理时的边界条件，消除了滑窗引入的由于“断崖式静音”带来的特征污染。

### 2.3 全链路掩码透传 (Mask Propagation)
- **残差隔离**：在每个 Transformer Block 结束时，强制执行 `x = x * mask`。防止浮点微差通过残差连接（Residual Connection）扩散到 Padding 区域并反向污染。
- **SOFTMAX 适配 (Additive Masking)**：
  DirectML 对 `-inf` 支持存在风险。推荐使用加性掩码：
  ```python
  m_addon = (mask - 1.0) * 10000.0  # 有效区为0，无效区为-10000
  scores = scores + m_addon
  ```

### 2.4 位置编码的一致性 (Positional Invariance)
- **策略**：位置编码的索引生成应基于 Tensor 本身的 rank，而不是总长度。确保无论载体多长，第 N 帧永远拿到的是第 N 个位置的编码值。

---

## 3. 调试方法论：差分打桩分析 (Differential Probing)

当 5s vs 30s 出现误差时，不要盲目猜测。按以下步骤定位：
1. **逐级打桩**：在 `Normalization`, `STFT`, `Mel`, `LFR`, `Encoder Layer 0`, `Encoder Final` 插入统计函数（Mean/Max/Min）。
2. **前缀统计**：**关键点！** 统计时必须只对 `[:valid_len]` 区域进行 Slicing 统计。
3. **定位首个分叉点**：寻找第一个出现 Delta 变化的阶段。
   - 若 STFT 分叉 -> 检查音频域清零。
   - 若 LFR 分叉 -> 检查复读策略。
   - 若 Layer 20+ 分叉 -> 检查 Mask 隔离是否彻底。

## 4. 结论
通过上述“逻辑隔离、物理一致”的改造，模型可以在保持固定显存分配和算子执行逻辑的同时，提供极其稳定的转写精度。这对于 DirectML、ONNXRuntime 以及其它对动态形状敏感的后端是至关重要的。
