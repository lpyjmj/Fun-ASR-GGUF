# DirectML 下固定长度 Padding 与数值一致性优化笔记

## 1. 背景与目标
在语音模型（如 SenseVoice）的端侧部署中，频繁改变推理长度会导致 DirectML (DML) 重新进行算子特化（Specialization），产生巨大的“入门费”延迟。为了实现极致性能，我们通常将输入统一填充至固定长度（如 60s）。

**核心挑战：**
1. **数值不一致**：同一个 5s 音频，放在 5s 容器和 60s 容器中，其推理出的特征值必须在 $10^{-6}$ 级别对齐，否则会影响 ASR 准确度。
2. **DML 算子兼容性**：传统的动态切片（Slicing）和 Reshape 算子极易触发 DML 的运行时崩溃。

---

## 2. 踩坑与解决方案汇总

### A. 前端“均值污染” (Mean Pollution)
*   **现象**：5s 音频在 60s 推理时，STFT 阶段就出现了微小误差。
*   **原因**：模型执行归一化（减去有效区均值）后，原本补零的区域变成了 `-Mean`，这部分非零数值在 STFT 卷积边界处会“污染”有效信号。
*   **修复方案 [硬清零]**：
    ```python
    audio = audio - mean_val
    audio[:, :, valid_samples:] = 0.0  # 归一化后立即物理置零
    ```

### B. FSMN 边界泄露 (Convolution Seepage)
*   **现象**：有效音频末尾的激活值在 Padding 模式下发生跳变。
*   **原因**：卷积窗滑过边界时，一侧是声音，另一侧是 Padding。
*   **修复方案 [复读填充]**：利用“复读有效末帧”替代“补零”，让卷积窗看到连续的能量分布。但在全链路 Mask 加持下，回归**“补零 + 强制层层 Mask”**更为稳健。

### C. DML 动态 Reshape 崩溃 (8007023E Error)
*   **现象**：模型包含 `torch.arange(T).view(1, 1, -1)` 逻辑时，DML 在 `Reshape` 节点报错。
*   **原因**：DML 对于从动态长度生成的 1D 张量执行 Reshape 非常脆弱。
*   **修复方案 [视图消除 - Cumsum 策略]**：
    ```python
    # 错误写法 (引发 DML 崩溃)
    idx = torch.arange(samples).view(1, 1, -1)
    
    # 正确写法 (Shape 继承，DML 友好)
    idx = torch.ones_like(audio, dtype=torch.long).cumsum(-1) - 1
    ```

### D. 物理切片 vs. 符号掩码 (Slice vs. Mask)
*   **原则**：导出模型时，禁止使用依靠输入数据决定的物理切片 `tensor[:valid]`。
*   **修复方案**：全量计算 60s，但全链路伴随 `mask`。
    *   **求均值**：`(audio * mask).sum() / valid_len`
    *   **末帧复读 (符号版)**：利用 `torch.gather` 配合 `torch.clamp(indices, max=valid-1)` 实现。

---

## 3. 最终优化成效

| 指标 | 初始状态 | 优化后 |
| :--- | :--- | :--- |
| **Max Absolute Error** | 1.48 (崩溃级) | 0.0008 (可忽略) |
| **Cosine Similarity** | < 0.95 | **1.00000000** |
| **DML 60s 执行时间** | N/A (报错) | ~230ms |
| **数值一致性** | ❌ 随填充长度漂移 | ✅ 五秒、六十秒完全对齐 |

---

## 4. 最佳实践 Checklist
1. [ ] **音频域输入**：归一化、预加重后必须对 Padding 区执行 `* mask`。
2. [ ] **索引生成**：严禁 `arange.view`，一律使用 `ones_like.cumsum`。
3. [ ] **数据对位**：`torch.gather` 的 `index` 必须是 `torch.long` 类型（常见报错根源）。
4. [ ] **输出截断**：在 ONNX 内部保持物理全长，由外部调用者根据业务逻辑裁切，或在输出层使用符号化乘法 Mask。
