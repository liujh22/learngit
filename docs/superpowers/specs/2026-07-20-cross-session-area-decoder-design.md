# 双向跨学习阶段的脑区级 Decoder 设计

## 目标

在不要求学习前后神经元一一配准的前提下，检验学习前训练得到的 `leaf1` / `circle1` decoder 能否直接泛化到学习后数据，并分别评估 `V1`、`medial`、`lateral` 和 `anterior` 脑区。

核心科学问题是脑区总体表征能否跨学习阶段保持，而不是同一批神经元的 readout 权重是否保持。学习前→学习后仍是主要方向；新增学习后→学习前作为对称性检查，用于判断跨阶段泛化是否依赖训练方向。

## 数据范围

- 学习前会话：`TX105_2022_10_08_2`
- 学习后会话：`TX105_2022_10_19_2`
- 神经活动：每个会话独立保存的前 400 个 SVD 成分，以及该会话的 `U` 矩阵
- 行为标签：`leaf1` 和 `circle1`
- 脑区标签：对应日期的 `_trans.npz` 文件中的 `iarea`
- 走廊位置：每个试次 60 个位置分箱，其中 0–39 为纹理区域，40–59 为灰色区域

每个会话的活动都可以用于不依赖刺激标签的会话内归一化。在主要的学习前→学习后分析中，学习后 `leaf1` / `circle1` 标签只能用于最终评分和单独标记的学习后内部交叉验证基线。在新增的学习后→学习前分析中，学习后标签属于训练标签，而学习前标签只能用于该反方向的最终评分。两个方向的模型和统计结果必须明确分开，不能利用某一方向的测试标签调参。

## 共同特征空间

学习前后不能直接使用原始神经元作为共同特征，因为两个会话的神经元数量不同；也不能直接把两个会话各自的 400 个 SVD 成分视为同一坐标系，因为它们是分别拟合的。

共同特征定义为每个脑区的平均空间活动曲线。对脑区 `A`：

```python
area_loading = U[:, area_mask].mean(axis=1)
area_activity = np.einsum(
    "c,ctp->tp",
    area_loading,
    component_activity_trial_pos,
)
```

其中：

- `U` 的形状为 `components × neurons`；
- `component_activity_trial_pos` 的形状为 `components × trials × 60 positions`；
- `area_activity` 的形状为 `trials × 60 positions`。

该计算等价于先将 SVD 活动重建到该脑区的所有神经元，再对神经元取平均，但不需要生成庞大的 `neurons × trials × positions` 数组。

主分析分别处理：

- `V1`：`iarea == 8`
- `medial`：`iarea` 属于 `{0, 1, 2, 9}`
- `lateral`：`iarea` 属于 `{5, 6}`
- `anterior`：`iarea` 属于 `{3, 4}`

附加报告 `visual_all`，定义为以上四个脑区的并集；它不包含 `iarea == -1`、`7` 或其他未归类神经元。四个单独脑区是主分析，`visual_all` 仅作为总体参照。

## 无标签归一化

每个会话、每个脑区独立使用灰色区域 40–59 的全部活动计算一个会话级均值和标准差：

```python
gray = area_activity[:, 40:60]
gray_mean = gray.mean()
gray_std = gray.std()
normalized_activity = (area_activity - gray_mean) / gray_std
X = normalized_activity[:, 0:40]
```

归一化不使用 `leaf1` / `circle1` 标签。采用会话级而非逐试次归一化，避免某个试次灰色区域中的刺激残留效应单独改变该试次的尺度。

如果 `gray_std` 非有限值或小于 `1e-12`，分析必须停止并给出明确错误，而不是静默生成无穷值或 NaN。

## 标签

标签固定编码为：

- `circle1 = 0`
- `leaf1 = 1`

其他刺激不进入训练或测试。特征和标签必须经过断言确认具有相同试次数，并且两个类别都存在。

## Decoder

每个脑区单独训练一个固定模型：

```text
StandardScaler
→ LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=5000,
    random_state=0
)
```

学习前→学习后方向的 `StandardScaler` 和分类器只在学习前特征上拟合；训练完成后直接预测学习后特征。学习后→学习前方向使用一个独立 pipeline，只在学习后特征上拟合，然后直接预测学习前特征。任何方向都不能在目标会话上重新拟合 scaler 或分类器。

第一版固定 `C=1.0`，避免利用任一方向的目标会话结果选择超参数。若以后增加超参数搜索，每个方向只能在自己的源会话内部通过嵌套交叉验证完成。

实现保留现有的单方向 `evaluate_transfer` 接口，并交换输入完成反方向分析：

```python
forward_metrics, forward_artifacts = evaluate_transfer(
    X_before, y_before, X_after, y_after,
)
reverse_metrics, reverse_artifacts = evaluate_transfer(
    X_after, y_after, X_before, y_before,
)
```

这样两个方向共享完全相同的评估代码，同时分别保存 pipeline、预测、概率、混淆矩阵、bootstrap 分数和 permutation 分数。结果表与图中必须使用明确的 `Before → after` 和 `After → before` 标签，避免把反方向结果误认为原来的核心方向。

## 评估

每个脑区报告四项互补结果：

1. **学习前内部基线**：学习前数据的分层 5 折交叉验证。
2. **学习后内部基线**：学习后数据的分层 5 折交叉验证，仅用于判断学习后是否存在可解码信息。
3. **核心前→后 transfer 结果**：在全部学习前试次上拟合模型，不进行任何重训，直接预测全部学习后试次。
4. **反方向后→前 transfer 结果**：在全部学习后试次上拟合一个独立模型，不进行任何重训，直接预测全部学习前试次。

两个 transfer 方向使用相同的特征定义、模型超参数和统计流程。反方向是稳定性与域偏移的补充检查，不代表时间因果关系；两个训练集的试次数不同，因此方向差异不能仅凭两个点估计解释为学习导致的变化。

主要指标为：

- balanced accuracy，机会水平为 0.5；
- ROC AUC，以 `leaf1` 为正类；
- confusion matrix；
- 目标会话试次的分层 bootstrap 95% 置信区间；
- 源会话训练标签 permutation 检验。

### Bootstrap

每个方向都在目标会话的 `circle1` 和 `leaf1` 内分别有放回抽样，再合并计算 transfer balanced accuracy。前→后方向抽样学习后试次；后→前方向抽样学习前试次。每个方向执行 2,000 次，报告 2.5% 和 97.5% 分位数。分层抽样保证每次重采样都包含两个类别。

### Permutation 检验

每个方向随机打乱源会话的训练标签 1,000 次。前→后方向打乱学习前标签并在学习后评分；后→前方向打乱学习后标签并在学习前评分。每次重新拟合同一方向的 pipeline，并在未经重训的目标会话上评分。单侧 p 值定义为：

```text
p = (1 + sum(null_score >= observed_score)) / (1 + permutation 次数)
```

分类器和交叉验证使用种子 0，permutation 使用种子 1，bootstrap 使用种子 2。每个随机过程分别创建随机数生成器，避免不同分析步骤因调用顺序变化而改变结果。

## 输出

生成一张结果表，每行一个脑区，至少包含：

- 学习前和学习后的神经元数量；
- 学习前试次数和学习后试次数；
- 学习前内部 balanced accuracy；
- 学习后内部 balanced accuracy；
- 前→后 transfer balanced accuracy；
- 前→后 transfer ROC AUC、bootstrap 95% 置信区间和 permutation p 值；
- 后→前 transfer balanced accuracy；
- 后→前 transfer ROC AUC、bootstrap 95% 置信区间和 permutation p 值。

生成以下图形：

1. 各脑区前→后与后→前 transfer balanced accuracy 的并列点图，分别带 bootstrap 95% 置信区间和 0.5 机会水平线；
2. 学习前内部、学习后内部、前→后和后→前四种准确率的并列比较图；
3. 每个脑区两个 transfer 方向的 confusion matrix；
4. 各脑区两个方向的 permutation 零分布及观测值位置。

## 解释框架

| 学习前内部 | 学习后内部 | 前→后 transfer | 解释 |
|---|---|---|---|
| 高 | 高 | 高 | 脑区的 leaf/circle 表征稳定 |
| 高 | 高 | 低 | 两阶段都有信息，但表征格式发生改变 |
| 低 | 高 | 低 | leaf/circle 信息主要在学习后形成 |
| 高 | 低 | 低 | 学习后该脑区的信息减弱或消失 |

对于“高”或“低”的判断，应同时考虑交叉验证置信区间、transfer bootstrap 区间和 permutation 检验，而不是只使用单个点估计。

若两个 transfer 方向都高，说明当前脑区平均特征具有较强的双向跨阶段稳定性。若只有一个方向高，首先应考虑训练试次数、类别分布、方差和决策阈值造成的方向不对称；在没有对方向差值进行专门统计检验前，不能把这种不对称直接解释为学习的因果效应。

## Notebook 集成范围

保留已有的数据下载、SVD 加载、行为读取和“帧→试次×位置”插值部分。只重构 decoder 相关单元：

- 删除两个连续且互相覆盖的 `%%writefile decoding.py` 单元；
- 使用一个自包含的 decoder 工具函数单元，避免模块缓存和隐藏状态；
- 删除依赖未明确初始化的 `keep`、`beh` 等调试单元；
- 新增脑区共同特征、灰色区域归一化、两个会话内基线、双向 transfer、bootstrap、permutation、表格和绘图单元；
- 所有 Markdown 说明和新增代码注释使用中文；
- 不把旧的缓存输出当作新分析结果，完成后从头执行 notebook。

## 验证与失败条件

实现必须在训练前检查：

- `U.shape[0]` 等于对应会话的成分活动第一维；
- `U.shape[1]` 等于 `iarea` 长度；
- 每个脑区至少包含一个神经元；
- 特征矩阵中不存在 NaN 或无穷值；
- 特征与标签的试次数一致；
- 学习前和学习后都同时包含 `circle1` 与 `leaf1`；
- 学习前和学习后的最终特征列数都等于 40；
- 前→后 pipeline 的 scaler 与分类器只在学习前数据上调用 `fit`；
- 后→前 pipeline 的 scaler 与分类器只在学习后数据上调用 `fit`；
- 两个 transfer 方向都没有在目标会话上重新调用 `fit`。

Notebook 应在 Colab 数据环境中从头执行完成。若因本机缺少 `/content/Zhong_et_al_2025` 数据而无法本地执行，则至少要通过 notebook 结构验证、Python 语法验证和不依赖实际数据的合成数组单元测试，并明确记录完整执行所需的 Colab 命令和数据条件。

## 不在本次范围内

- 跨会话单神经元配准；
- 直接对齐两个会话独立拟合的 SVD 坐标；
- CCA、Procrustes、CORAL 等潜在空间对齐；
- 使用任一方向的目标会话标签调参、选择归一化方式或筛选脑区；
- 将两个 transfer 方向的差异直接解释为学习的因果效应；
- 将单只小鼠、两个会话的结果推广到群体水平结论。
