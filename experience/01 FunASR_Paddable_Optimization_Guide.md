# FunASR-GGUF DirectML Padding 优化技术白皮书

## 1. 背景与问题定义

在 Windows 平台上使用 ONNX Runtime 配合 **DirectML (DML)** 执行提供程序时，存在明显的 **动态形状性能开销**。当输入音频长度变化时，DirectML 驱动程序会触发大规模的着色器（Shader）重编译或重新分配 GPU 资源。

### 核心现象
*   **首次形状特化 (First-Shape Specialization)**：DML 会针对会话收到的第一个输入维度进行深度优化。
*   **重编译延迟**：后续若输入维度改变（如从 5s 变到 10s），推理时间会瞬间飙升（从百毫秒级变为秒级）。
*   **资源抖动**：频繁改变形状会导致 GPU 显存频繁申请与释放，影响系统稳定性。

---

## 2. 核心优化策略：统一填充 (Uniform Padding)

为了绕过 DirectML 的动态形状限制，我们采用了 **统一填充** 策略：
1.  **固定形状输入**：将所有输入音频由后端统一填充（Padding）到一个固定最大长度（例如 60 秒的采样点 960,000）。
2.  **预热 (Warmup)**：在模型初始化时，使用 60s 的随机全长数据跑一遍推理，使 DML 驱动程序完成该固定维度的特化优化。
3.  **单会话执行**：整个推理周期只对 DML 呈现唯一的输入维度，彻底消除后续推理的重编译开销。

---

## 3. 技术挑战：数值一致性漂移

简单的“补零填充”会导致 ASR 模型（尤其是 Transformer 架构）出现严重的数值漂移，导致识别率下降：
*   **均值偏移 (Mean Shift)**：传统的 `Normalization` 在计算均值时会包含填充区的大量 0，导致有效信号区的偏置计算错误。
*   **注意力稀释 (Attention Dilution)**：Transformer 的注意力机制会将 Padding 区的 0 帧视作有效信息，从而平摊掉真实信号的权重。
*   **特征拼接污染 (LFR Contamination)**：LFR 拼接时，如果从物理末尾开始对齐，会将 Padding 产生的假边缘信息卷入计算。

---

## 4. 解决方案：长度感知模型定义 (Paddable Definition)

我们重构了模型导出定义，推出了 `fun_asr_gguf/model_definition_paddable.py`：

### A. 双输入架构
模型现在接受两个 Tensor 输入：
1.  `audio`: (1, 1, 固定最大长度) —— 填充后的音频。
2.  `ilens`: (1, ) —— 包含音频真实有效采样点数的整数 Tensor。

### B. 长度感知归一化 (Length-Aware Norm)
在前端处理阶段，模型仅对 `audio[..., :ilens[0]]` 部分计算均值。这确保了无论后面填充了多少秒的 0，有效信号的归一化偏置始终保持一致。

### C. 动态掩码 (Dynamic Attention Masking)
模型内部会根据 `ilens` 自动计算出各层对应的有效帧数，并生成 `0/1 Mask`：
*   **计算公式**：`valid_frames = (ilens[0] // 160) + 1` (对应 STFT 特征)。
*   **透传机制**：通过 `MultiHeadedAttention` 内核执行 `masked_fill`，在 Softmax 之前屏蔽填充区的影响。

### D. 正确的 LFR 与输出裁剪
*   **LFR 修正**：使用有效帧边界进行镜像/边缘填充，而非物理 Tensor 末尾。
*   **输出裁剪**：Adaptor 的输出根据有效长度进行逻辑裁剪，确保交给解码器的向量维度与音频实际长度 100% 匹配。

---

## 5. 验证与成果

### 验证脚本
*   `05-Verify-Initial-Consistency.py`: 基础数值验证。
*   `06-Verify-Paddable-Consistency.py`: 验证“填充输入”与“短音频原生输入”的输出误差。

### 关键成果
*   **性能稳定性**：DirectML 在处理变长音频时，首帧延迟降至最低，推理性能实现常态化。
*   **数值一致性**：填充 60s 后的 5s 音频输出与 5s 原始音频输出的 Max Error 保持在 **1e-5** 量级。
*   **零代价 Padding**：本方案在保证 DML 高效执行的同时，解决了模型对动态输入的敏感度问题。

---

## 6. 使用说明

在导出模型时，请使用 `01-Export-Encoder-Adaptor-Paddable.py`：
```python
# 核心逻辑演示
audio = torch.randn(1, 1, 960000) # 填充到60s
ilens = torch.tensor([80000])     # 实际只有5s
enc_output, adaptor_output = model(audio, ilens)
```

本技术文档作为 FunASR-GGUF 项目在 Windows DML 环境下的工程加速指南，建议在所有类似 DML 部署场景中参考使用。
