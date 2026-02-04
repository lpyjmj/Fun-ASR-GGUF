# PyTorch 模型 ONNX 导出与 Padding 一致性优化经验总结

本方案针对语音识别模型（以 SenseVoice 为例）在 DirectML/ONNX 部署环境下，因音频填充（Padding）导致的数值漂移和导出失败问题，总结了一套成熟的“符号化重构”方法论。

## 1. 核心矛盾：Padding 引入的数值漂移

### 1.1 均值污染 (Mean Pollution)
在进行长度归一化（Normalize）时，如果直接对整段 Tensor（含 Padding 零）计算均值，会导致有效音频段的均值被拉低。
- **后果**：同一段音频，放在 5s 容器和 30s 容器中，减去的均值不同，导致后续 STFT 和神经网络输入发生分叉。

### 1.2 边界泄露 (Boundary Leakage)
模型中的滑窗算子（如 LFR 特征拼接、FSMN 卷积）在有效音频与 Padding 区域的交界处，会同时采样到“有效信号”和“静音/归一化偏差值”。
- **后果**：卷积计算结果在边界处出现不正常的跳变，并随着 50 层 Transformer 的残差连接指数级扩散，最终导致 Max Error 爆炸。

---

## 2. 第一阶段：PyTorch 层面的逻辑对齐 (Paddable Implementation)

我们通过引入 `ilens`（输入有效长度）作为模型输入，实施了以下策略：
1. **Length-Aware Normalization**：仅对 `audio[:ilens]` 区域计算均值。
2. **全链路 Mask 向下透传**：
    - 在 Attention 阶段对 Softmax 之前的分值进行补丁（`-10000.0`），确保 Attention 不看 Padding 区。
    - 在 FSMN 阶段前进行物理清零。
3. **Replicate Padding (复读填充)**：
    - LFR 拼接逻辑改为：有效长度以外的部分，用有效长度的最后一帧进行填充，而不是默认的静音填充。这保证了卷积窗在边界处的统计稳定性。

---

## 3. 第二阶段：攻克 ONNX 导出障碍 (Symbolic Refactoring)

在尝试导出时，PyTorch 2.x 的 Export 模块抛出了 `GuardOnDataDependentSymNode` 异常。这是因为传统的物理切片（Slicing）逻辑破坏了静态计算图。

### 3.1 致命禁忌：数据依赖型切片
**错误示范：**
```python
valid_segment = audio[..., :ilens[0]] # ❌ 报错：切片边界依赖于输入数据的内容
audio[..., ilens[0]:] = 0.0           # ❌ 报错：物理张量修改边界不能由输入决定
```

### 3.2 黄金法则：符号化掩码 (Symbolic Masking & Gather)
我们通过以下替代方案实现了“形状不变、内容受控”：

#### A. 均值计算的符号化
利用生成的 `mask` (0/1) 将切片求和转换为全量求和：
```python
# 将 [..., :valid] 的 mean 改为：
mean_val = (audio * mask).sum() / ilens.float()
```

#### B. 硬清零 (Hard-Zeroing)
将物理切片赋值改为元素级乘法：
```python
audio = (audio - mean_val) * mask # 自动抹除 Padding 区产生的 -Mean 数值
```

#### C. 复读填充的符号化 (Gather 策略)
这是最关键的一步。通过 `torch.gather` 模拟“切掉尾巴并补齐最后一张”：
```python
# 生成映射索引：[0, 1, 2, ..., valid-1, valid-1, valid-1]
# 利用 clamp 强行将 valid 之后的所有索引拉回到有效边界
indices = torch.arange(T_phys)
gather_idx = torch.clamp(indices, max=valid_len - 1)
mel_consistent = torch.gather(mel, 1, gather_idx)
```

---

## 4. 验证标准 (The Golden Standards)

为了确保优化成功，我们建立了三关验收机制：
1. **数值关 (Max Error)**：5s 输入 vs 30s 填充输入的输出 Max Error 必须在 $10^{-4}$ 级别（FP32 下）。
2. **方向关 (Cosine Similarity)**：输出向量的余弦相似度必须大于 `0.999999`。
3. **导出关 (ONNX Success)**：`torch.onnx.export` 不得触发数据依赖型 Guard 报错。

## 5. 总结建议
面向 DirectML 这种追求固定 Shape 的加速后端，**“符号化（Symbolic）”**意识是模型定义的灵魂。
- **不要物理切断**：保持张量长度为容器长度（如 30s）。
- **不要物理拼接**：尽量在预分配好的 Buffer 上操作。
- **利用索引和掩码控制逻辑边界**：让模型在跑 30s 的全量算力的同时，在数值上模拟出 5s 孤立推理的边界条件。
