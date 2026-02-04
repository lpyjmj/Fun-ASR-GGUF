# DirectML Paddable 架构完整技术指南
## FunASR/SenseVoice 模型在 Windows ONNX Runtime 下的工程加速全记录

**记录日期**：2026-02-04  
**适用场景**：语音识别模型在 DirectML 后端部署时的性能优化与数值一致性保障

---

# 第一部分：问题背景与方案提出

## 1.1 DirectML 动态形状的性能陷阱

在 Windows 平台使用 ONNX Runtime + **DirectML (DML)** 执行推理时，存在严重的 **动态形状性能开销**：

| 现象 | 描述 |
|:---|:---|
| **首次形状特化** | DML 会针对会话收到的第一个输入维度进行深度优化（Shader 编译、显存布局） |
| **重编译延迟** | 后续若输入维度改变（如从 5s 变到 10s），推理时间从几十毫秒飙升至200多毫秒 |

**根因**：DirectML 驱动程序的算子特化（Operator Fusion/Specialization）机制会针对特定形状进行深度优化，形状变化意味着重走整个优化流程。

## 1.2 核心优化策略：统一填充 (Uniform Padding)

为绕过 DirectML 的动态形状限制，我们采用 **统一填充** 策略：

1. **固定形状输入**：将所有输入的短音频统一填充（Padding）到固定最大长度（如 30s）
2. **预热 (Warmup)**：模型初始化时，使用固定长度的随机数据跑一遍推理，触发 DML 完成特化优化
3. **单会话执行**：整个推理周期的短音频只对 DML 呈现唯一的输入维度，彻底消除重编译开销

---

# 第二部分：数值一致性问题与解决

## 2.1 Padding 引入的三大数值漂移

简单的"补零填充"会导致 Transformer 架构的 ASR 模型出现严重数值漂移：

### A. 均值污染 (Mean Pollution)
- **现象**：5s 音频在 30s 容器中，STFT 阶段就出现误差
- **原因**：归一化时对整段 Tensor（含 Padding 零）计算均值，导致有效音频段的均值被拉低
- **后果**：同一段音频在不同容器中减去的均值不同，导致后续特征分叉

### B. 边界泄露 (Boundary Leakage)
- **现象**：有效音频末尾的激活值在 Padding 模式下发生跳变
- **原因**：滑窗算子（LFR 特征拼接、FSMN 卷积）在有效信号与 Padding 区域交界处，同时采样到"有效信号"和"静音/归一化偏差值"
- **后果**：卷积计算结果在边界处出现跳变，并随着 50 层 Transformer 的残差连接指数级扩散

### C. 注意力稀释 (Attention Dilution)
- **现象**：Transformer 输出特征整体偏移
- **原因**：Attention 的 Softmax 会将 Padding 区的 0 帧视作有效信息，平摊真实信号的权重
- **后果**：分类/识别准确率下降

## 2.2 解决方案：长度感知模型定义 (Paddable Definition)

### A. 双输入架构
模型接受两个 Tensor 输入：
```python
audio: (1, 1, 固定最大长度)  # 填充后的音频
ilens: (1,)                   # 包含音频真实有效采样点数
```

### B. 长度感知归一化 (Length-Aware Normalization)
仅对有效区域计算均值：
```python
# 错误写法：mean = audio.mean()
# 正确写法：
mean_val = (audio * mask).sum() / ilens.float()
audio = (audio - mean_val) * mask  # 自动抹除 Padding 区产生的 -Mean
```

### C. 音频域硬清零 (Audio Hard-Zeroing)
在 Normalization 和 Pre-emphasis 之后，立即强制清零：
```python
audio = audio - mean_val
audio = audio * mask  # 确保 Padding 区绝对为 0
```
**效果**：确保 STFT Real/Imag 在有效区间实现近乎 0 误差

### D. LFR 复读覆盖策略 (Replicate Padding)
将有效长度以外的物理 Tensor 空间，填充为有效帧的**最后一帧**内容：
```python
# 生成映射索引：[0, 1, 2, ..., valid-1, valid-1, valid-1]
gather_idx = torch.clamp(indices, max=valid_len - 1)
mel_consistent = torch.gather(mel, 1, gather_idx)
```
**效果**：模拟短音频原生推理的边界条件，消除滑窗引入的"断崖式静音"特征污染

### E. 全链路掩码透传 (Mask Propagation)
- **残差隔离**：在每个 Transformer Block 结束时，强制执行 `x = x * mask`，防止浮点微差通过残差连接扩散
- **FSMN 阶段清零**：在 FSMN 卷积之前进行物理清零，确保卷积核不会采样到 Padding 区的噪声
- **关键位置**：Encoder 的 `after_norm` 和 `tp_norm` 之后必须进行 Fire-wall Sweeping

### F. 位置编码的一致性 (Positional Invariance)
- **原则**：位置编码的索引生成应基于 Tensor 本身的 rank，而不是总长度
- **效果**：确保无论载体多长，第 N 帧永远拿到的是第 N 个位置的编码值
- **实现**：使用 `ones_like.cumsum` 而非 `arange` 生成位置索引

## 2.3 调试方法论：差分打桩分析 (Differential Probing)

当 5s vs 30s 出现误差时，按以下步骤定位：

1. **逐级打桩**：在 `Normalization`, `STFT`, `Mel`, `LFR`, `Encoder Layer 0`, `Encoder Final` 插入统计函数
2. **前缀统计**：**关键！** 统计时必须只对 `[:valid_len]` 区域进行 Slicing
3. **定位首个分叉点**：
   - 若 STFT 分叉 → 检查音频域清零
   - 若 LFR 分叉 → 检查复读策略
   - 若 Layer 20+ 分叉 -> 检查 Mask 隔离是否彻底

**目标指标**：填充 30s 后的 5s 音频输出与 5s 原始音频输出的 Max Error 保持在 **1e-5** 量级。

---

# 第三部分：ONNX 导出障碍与攻克

## 3.1 致命禁忌：数据依赖型切片

PyTorch 2.x 的 Export 模块会抛出 `GuardOnDataDependentSymNode` 异常：

```python
# ❌ 错误示范：切片边界依赖于输入数据的内容
valid_segment = audio[..., :ilens[0]]
audio[..., ilens[0]:] = 0.0

# 报错：torch.export 无法静态追踪数据依赖的边界
```

## 3.2 黄金法则：符号化掩码 (Symbolic Masking)

通过以下替代方案实现"形状不变、内容受控"：

### A. 均值计算的符号化
```python
# 将 [..., :valid] 的 mean 改为：
mean_val = (audio * mask).sum() / ilens.float()
```

### B. 硬清零的符号化
```python
# 将物理切片赋值改为元素级乘法：
audio = (audio - mean_val) * mask
```

### C. 复读填充的符号化 (Gather 策略)
```python
# 利用 clamp 强行将 valid 之后的所有索引拉回到有效边界
indices = torch.arange(T_phys)
gather_idx = torch.clamp(indices, max=valid_len - 1)
mel_consistent = torch.gather(mel, 1, gather_idx)
```

## 3.3 导出验证三关

1. **数值关 (Max Error)**：5s 输入 vs 30s 填充输入的输出 Max Error 必须在 $10^{-4}$ 级别
2. **方向关 (Cosine Similarity)**：输出向量的余弦相似度必须大于 `0.999999`
3. **导出关 (ONNX Success)**：`torch.onnx.export` 不得触发数据依赖型 Guard 报错

---

# 第四部分：DirectML 兼容性问题与修复

## 4.1 动态 Reshape 崩溃 (8007023E Error)

- **现象**：模型包含 `torch.arange(T).view(1, 1, -1)` 逻辑时，DML 在 `Reshape` 节点报错
- **原因**：DML 对于从动态长度生成的 1D 张量执行 Reshape 非常脆弱

**修复方案 - 视图消除 Cumsum 策略**：
```python
# 错误写法 (引发 DML 崩溃)
idx = torch.arange(samples).view(1, 1, -1)

# 正确写法 (Shape 继承，DML 友好)
idx = torch.ones_like(audio, dtype=torch.long).cumsum(-1) - 1
```

## 4.2 Where 算子性能问题 (碎片算子)

- **现象**：使用 `masked_fill` 时性能严重下降；手动实现的 `x * mask` 产生大量微小算子
- **原因**：`masked_fill` 在 ONNX 中映射为 `Where` 算子，在 70 层深度模型中非常昂贵，阻碍 DML 的算子融合 (Fusion)
- **核心原则**：**能用 Add 不用 Where** — 在深层网络中，掩码逻辑应优先考虑加法方式

**修复方案 - 加法掩码 Additive Masking**：
```python
# 错误写法 (昂贵的 Where 算子)
scores = scores.masked_fill(mask.eq(0), -float('inf'))

# 正确写法 (DML 友好的 Add 算子)
m_addon = (mask - 1.0) * 10000.0  # 有效区为0，无效区为-10000
scores = scores + m_addon
```

**原理**：`Add` 是 DML 最擅长处理和融合的基础算子。通过加法模拟遮罩，避开了昂贵的 `Where` 条件分支。

**效果**：在 70 层 Transformer 结构下，直接夺回约 40-50ms 的性能损失

## 4.3 torch.gather 类型要求

- **常见报错根源**：`torch.gather` 的 `index` 参数必须是 `torch.long` 类型
- **修复**：确保索引张量的 dtype 正确

---

# 第五部分：性能优化成果

## 5.1 优化前后对比

以下为 30s 音频的性能参考

| 阶段/模型状态 | FP32 耗时 | FP16 耗时 | 备注 |
|:---|:---|:---|:---|
| **旧模型 (无 Padding)** | 150ms | 50ms | 长度不定，显存波动 |
| **初版 Padding** | 220ms | 80ms | 加入 ilens，但算子散乱 |
| **最终优化版** | **170ms** | **55ms** | 性能接近旧模型，极度稳定 |

## 5.2 数值一致性成果

| 指标 | 初始状态 | 优化后 |
|:---|:---|:---|
| **Max Absolute Error** | 1.48 (崩溃级) | 0.0008 (可忽略) |
| **Cosine Similarity** | < 0.95 | **1.00000000** |
| **数值一致性** | 随填充长度漂移 | 5s、30s、60s 完全对齐 |

## 5.3 预热机制的意义 (Warmup)

- **显存预分配**：在加载模型阶段，使用固定长度（如 30s）的伪音频运行一次完整的前向传播
- **意义**：这触发了 DML 的 Graph Optimizer，确保第一次真实推理时，显卡已经完成了算子链接的最优方案和显存空间的固定
- **效果**：彻底消除首次推理的"冷启动"延迟

---

# 第六部分：最佳实践 Checklist

## 模型定义层面
- [ ] **双输入架构**：模型接受 `audio` 和 `ilens` 两个输入
- [ ] **长度感知均值**：`(audio * mask).sum() / ilens`
- [ ] **音频域硬清零**：归一化、预加重后必须对 Padding 区执行 `* mask`
- [ ] **复读填充**：LFR 拼接使用 `torch.gather` + `torch.clamp` 实现符号化

## ONNX 导出层面
- [ ] **禁止物理切片**：不使用 `tensor[:valid]` 依赖输入数据的切片
- [ ] **索引生成**：严禁 `arange.view`，一律使用 `ones_like.cumsum`
- [ ] **类型正确**：`torch.gather` 的 `index` 必须是 `torch.long`

## DirectML 兼容层面
- [ ] **加法掩码**：Attention 使用 `scores + (mask - 1.0) * 10000.0` 替代 `masked_fill`
- [ ] **视图消除**：使用 Shape 继承策略避免动态 Reshape
- [ ] **预热逻辑**：加载模型后使用固定长度伪音频运行一次前向传播

## 验证层面
- [ ] **一致性脚本**：每次优化后运行 5s vs 30s 对比
- [ ] **Cosine Similarity**：必须为 1.0
- [ ] **Max Error**：必须在 $10^{-4}$ 级别

---

# 附录：核心代码示例

## 符号化位置编码
```python
def forward(self, x, mask=None):
    # Shape 继承：创建索引 (1, 2, ..., T) 不使用 arange.view
    positions = torch.ones_like(x[:, :, 0], dtype=torch.long).cumsum(1)
    position_encoding = self.encode(positions, x.size(-1), x.dtype)
    return x + position_encoding
```

## 加法掩码 Attention
```python
def forward_attention(self, v_h, scores, mask):
    if mask is not None:
        # 加法掩码：有效区为0，无效区为-10000
        m_addon = (mask - 1.0).unsqueeze(1).unsqueeze(2) * 10000.0
        scores = scores + m_addon
    attn = torch.softmax(scores, dim=-1)
    x = torch.matmul(self.dropout(attn), v_h).permute(0, 2, 1, 3).flatten(2)
    return self.linear_out(x)
```

## 符号化复读填充
```python
# 生成映射索引并 clamp 到有效边界
mel_indices = torch.ones_like(mel[..., :1], dtype=torch.long).cumsum(1) - 1
gather_idx = torch.clamp(mel_indices, max=T_mel_valid - 1).expand(-1, -1, 80)
mel_consistent = torch.gather(mel, 1, gather_idx)
```

---

# 第七部分：核心原则总结

## 设计原则

1. **逻辑隔离、物理一致**：保持张量长度为容器长度（如 30s），通过索引和掩码控制逻辑边界
2. **不要物理切断**：禁止使用依靠输入数据决定的物理切片 `tensor[:valid]`
3. **不要物理拼接**：尽量在预分配好的 Buffer 上操作
4. **能用 Add 不用 Where**：在深层网络中，掩码逻辑应优先考虑加法方式
5. **保持原子性**：Padding 逻辑应当由一个统一的 Wrapper 处理，内部计算全部符号化
6. **逻辑向后兼容**：所有的优化均在 `model_definition.py` 中完成，对外接口保持简洁稳定

## 黄金法则

> 让模型在跑 30s 的全量算力的同时，在数值上模拟出 5s 孤立推理的边界条件。

---

*本文档作为 FunASR-GGUF 项目在 Windows DirectML 环境下的工程加速完整指南，建议在所有类似 DML 部署场景中参考使用。*

*记录日期：2026-02-04*
*记录人：Antigravity*
