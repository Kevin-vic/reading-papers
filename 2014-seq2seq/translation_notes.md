# 翻译笔记：《Sequence to Sequence Learning with Neural Networks》(arXiv:1409.3215v3)

## 一、批量简化模式说明

- 本次翻译为批量简化模式：**未生成 source_map.json**，**未提取图片**，图/表以页码标注（原图见 paper.pdf 第 N 页）。
- 本机无 poppler（Read 工具无法直接渲染 PDF 页面），改用 pymupdf 提取全文文本（共 9 页），公式区域另做坐标级提取还原。
- 原文为 LaTeX 排版论文，PDF 文本层中的公式为乱码（如 `sigm W hxxt + W hhht−1`），已按原论文 LaTeX 源码还原为可编译的 LaTeX 公式（display 用 `$$...$$`，行内用 `$...$`），数学内容不译。
- 图 1 在第 2 页、表 1/表 2/图 2 在第 6 页、表 3/图 3 在第 7 页。
- 表 3 中为英法翻译任务的法语语料例句，作为数据整体保留原文不译（附译者注说明）。

## 二、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| Deep Neural Network (DNN) | 深度神经网络（DNN） | 首次出现标注全称，之后用 DNN |
| Long Short-Term Memory (LSTM) | 长短期记忆网络（LSTM） | 首次出现标注全称，之后用 LSTM |
| Recurrent Neural Network (RNN) | 循环神经网络（RNN） | 首次出现标注全称 |
| sequence to sequence | 序列到序列 | |
| end-to-end | 端到端 | |
| BLEU score | BLEU 分数 | BLEU 不译 |
| cased BLEU score | 区分大小写的 BLEU 分数（cased BLEU score） | |
| statistical machine translation (SMT) | 统计机器翻译（SMT） | 首次出现标注；phrase-based SMT system → 基于短语的 SMT 系统 |
| machine translation (MT) | 机器翻译（MT） | |
| beam search | 束搜索 | beam size → 束宽 |
| hypothesis / partial hypothesis / complete hypothesis | 假设 / 部分假设 / 完整假设 | 束搜索语境 |
| n-best list / 1000-best list | n-best 列表 / 1000-best 列表 | 保留原样 |
| rescore / rescoring | 重打分 | |
| out-of-vocabulary word | 词表外词 | 特殊标记 "UNK"、"<EOS>" 保留不译 |
| vocabulary | 词表 | |
| softmax | Softmax | 保留原文 |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| backpropagation | 反向传播 | |
| vanishing gradient problem | 梯度消失问题 | |
| exploding gradients | 梯度爆炸 | |
| long term dependencies | 长期依赖 | |
| long range temporal dependencies | 长程时间依赖 | 与"长期依赖"按原文用词区分 |
| short term dependencies | 短期依赖 | |
| minimal time lag | 最小时滞（minimal time lag） | |
| perplexity | 困惑度 | |
| ensemble | 集成 | |
| word embedding | 词嵌入 | |
| hidden state | 隐藏状态 | |
| encoder / decoder | 编码器 / 解码器 | 原文以引号形式出现，保留 |
| attention mechanism / differentiable attention mechanism | 注意力机制 / 可微注意力机制 | |
| monotonic alignment | 单调对齐 | |
| conditional probability | 条件概率 | |
| log probability | 对数概率 | |
| language model | 语言模型 | RNNLM → 循环神经网络语言模型（RNNLM）；NNLM → 前馈神经网络语言模型（NNLM） |
| feedforward neural network | 前馈神经网络 | |
| minibatch | 小批量 | |
| epoch | 轮次（epoch） | |
| learning rate | 学习率 | |
| momentum | 动量 | |
| gradient norm | 梯度范数 | |
| activations | 激活值 | |
| PCA | 主成分分析（PCA） | |
| bag-of-words model | 词袋（bag-of-words）模型 | |
| active / passive voice | 主动语态 / 被动语态 | |
| paraphrase | 改写（paraphrase） | |
| ground truth | 参考译文（ground truth） | |
| tokenized | 分词后的 | |
| timestep | 时间步 | |
| fixed-dimensional / fixed-sized vector | 固定维数向量 / 固定大小向量 | 按原文用词区分 |
| language pair | 语言对 | |
| parallelize / parallelization | 并行化 | |
| Connectionist Sequence Classification | 联结主义序列分类（Connectionist Sequence Classification） | 原文如此；通常指 CTC，已加译注 |
| Oracle Rescoring | Oracle 重打分 | Oracle 不译 |
| "establish communication" | "建立联系" | 原文带引号的形象说法 |
| even average | 等权平均 | |
| naive softmax | 朴素 Softmax | |

## 三、数字与单位处理（全文统一）

- 大数用中文计数单位：384M → 3.84 亿；12M → 1200 万；348M → 3.48 亿；304M → 3.04 亿；80k → 8 万；160,000 → 16 万；80,000 → 8 万。
- 小数字保留阿拉伯数字：8000 维状态、1000 个单元、1000 维词嵌入、128 个序列、学习率 0.7、束宽 12 等。
- 速度统一为"每秒 N 词"；"2x speedup" → "2 倍加速"。

## 四、不确定 / 存疑点

1. **"Connectionist Sequence Classification"**：原文（arXiv v3）在引言中写作 "Connectionist Sequence Classification"，但引文 [11] 是 Graves 等人著名的 "Connectionist Temporal Classification"（CTC）。译文按原文直译并加了译注说明。若严格按惯例，应指 CTC。
2. **原文笔误（均已加 [译注]）**：
   - 3 节 "and we it to rescore"（疑漏 "used"）；
   - 3.4 节 "divided it the size of the batch"（疑漏 "by"）；
   - 3.5 节 "about a ten days"（多 "a"）；
   - 3.6 节 "BELU score"（BLEU 拼写错误）；
   - 3.6 节脚注 "There several variants"（疑漏 "are"）。
3. **公式还原**：PDF 文本层的公式为乱码，已按该论文 arXiv LaTeX 源码还原。其中 RNN 递推式权重记号 $W^{hx}$、$W^{hh}$、$W^{yh}$（上标 hx/hh/yh）与常见记号 $W_{xh}$ 等不同，此为原论文写法，未改动。
4. **"minimal time lag"**：出自 Hochreiter & Schmidhuber 1997，译为"最小时滞"，加英文标注。
5. **"even average"**：指两分数各占一半的简单平均，译为"等权平均"。
6. **图 2 题中 "the phrases in the figures"**：原文如此（"figures" 为复数），按字面译为"图中短语"。
7. **"cased BLEU"**：即区分大小写评估的 BLEU（对照 mteval 的 case-sensitive 模式），译文保留英文标注。
8. **表 3 法语例句**：按数据保留原文；个别词如 "crémation"（火葬）、"antennes-relais"（基站天线）仅供读者对照，未在文中展开翻译。
9. **术语"假设"**：hypothesis 在摘要（1000 hypotheses）与束搜索（partial/complete hypotheses）中均译为"假设"并首次标注英文，全文一致。
10. **N N-bit numbers**：原文 "sort $N$ $N$-bit numbers"，即 N 个 N 位数，译为"N 个 N 位数"。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（3 处）；图注内重复的页图链接去重（3 处）；成对重复的页图引用行去重（3 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
