# 翻译笔记：《Attention Is All You Need》中英对照翻译

## 一、本次为批量简化模式

- 本次翻译按批量简化流程执行：**未生成 source_map.json，未做图片提取**。图（Figure 1–5）与表（Table 1–4）在 paper.md 中以页码标注方式注明原图位置，例如「（原图见 paper.pdf 第 4 页）」。
- 原文 PDF 共 15 页，全部读完并翻译（含正文、脚注、致谢、参考文献与附录图题）。
- 使用 PDF 版本：arXiv:1706.03762v7（2023 年 8 月修订版，NIPS 2017 发表版本）。
- 提取方式：pypdf 提取文本。提取文本中数学公式的下标/上标/分式排版有丢失，翻译时已按原文 LaTeX 语义重建公式（式 (1)(2)(3)、MultiHead 公式、位置编码公式、脚注 4 中点积方差公式）。
- 译文遵守 `nature-reader/references/cs-glossary.md` 的翻译硬规则八条。

## 二、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| Transformer | Transformer（不译） | 模型名，保留英文 |
| attention / attention mechanism | 注意力 / 注意力机制 | 全文统一 |
| self-attention | 自注意力（Self-Attention） | 首次出现标注英文 |
| intra-attention | 内部注意力（intra-attention） | 与 self-attention 同义 |
| multi-head attention | 多头注意力（multi-head attention） | 首次标注，后统一中文 |
| scaled dot-product attention | 缩放点积注意力（scaled dot-product attention） | 首次标注，后统一中文 |
| additive attention | 加性注意力（additive attention） | 首次标注 |
| dot-product attention | 点积注意力 | |
| encoder / decoder | 编码器（encoder）/ 解码器（decoder） | 首次标注，后统一中文 |
| encoder-decoder attention | 编码器-解码器注意力 | |
| sequence transduction | 序列转换（sequence transduction） | 首次标注 |
| sequence modeling | 序列建模 | |
| sequence-to-sequence | 序列到序列 | 规范译名 |
| recurrent neural network (RNN) | 循环神经网络（RNN） | |
| long short-term memory (LSTM) | 长短期记忆网络（LSTM） | |
| gated recurrent neural network | 门控循环神经网络 | |
| convolutional / convolution | 卷积 | |
| dilated convolutions | 空洞卷积（dilated convolutions） | |
| separable convolutions | 可分离卷积（separable convolutions） | |
| feed-forward network | 前馈网络（feed-forward network） | |
| position-wise / point-wise | 逐位置 | 用于前馈层描述 |
| residual connection | 残差连接 | |
| layer normalization | 层归一化 | |
| sub-layer | 子层（sub-layer） | |
| embedding | 嵌入（embedding） | |
| positional encoding | 位置编码（positional encoding） | |
| sinusoidal | 正弦（的） | 用于位置编码描述 |
| query / key / value | 查询 / 键 / 值 | 规范译名 |
| compatibility function | 兼容性函数（compatibility function） | |
| softmax | Softmax（保留） | 规范保留英文 |
| token | Token（保留） | 规范保留英文 |
| vocabulary | 词表（vocabulary） | 首次标注 |
| word-piece / byte-pair encoding | WordPiece / 字节对编码（byte-pair encoding） | WordPiece 为专有名词保留；BPE 首次标注 |
| auto-regressive | 自回归（auto-regressive） | 首次标注 |
| masking / mask out | 掩码 / 掩蔽 | 后文出现 "masked" 译「掩码」 |
| beam search | 束搜索（beam search） | 首次标注 |
| length penalty | 长度惩罚 | |
| checkpoint | 检查点（checkpoint） | 首次标注 |
| development set | 开发集 | |
| label smoothing | 标签平滑（label smoothing） | 首次标注 |
| dropout | 随机失活（Dropout） | 规范译名；Residual Dropout 译「残差随机失活」 |
| regularization | 正则化 | |
| perplexity | 困惑度 | |
| overfitting | 过拟合 | |
| generalization / generalize | 泛化 / 泛化到 | |
| state-of-the-art | 当前最优（state-of-the-art） | 规范译名；全文统一「当前最优」 |
| ensemble | 集成模型（ensemble） | 首次标注 |
| single model | 单模型（single-model） | |
| optimizer | 优化器 | |
| learning rate | 学习率 | |
| warmup steps | `warmup_steps`（保留变量名） | 代码/变量名不译 |
| batch / batching | 批次 / 分批 | |
| hidden state | 隐藏状态 | |
| long-range dependencies | 长距离依赖 | |
| path length | 路径长度 | |
| representation subspace | 表示子空间 | |
| constituency parsing | 成分句法分析（constituency parsing） | 首次标注 |
| semi-supervised | 半监督 | |
| discriminative / generative | 判别式 / 生成式 | |
| anaphora resolution | 指代消解 | 图 4 图题 |
| inference | 推理 | |
| TFLOPS / FLOPs | 保留原文 | 单位不译 |
| BLEU | BLEU 分数 | 指标保留英文 |
| Adam | Adam（不译） | 优化器名保留 |
| WMT 2014 / newstest2013 / newstest2014 | 保留原文 | 数据集名不译 |
| Penn Treebank / Wall Street Journal (WSJ) | 保留原文 | 数据集名不译 |
| P100 / K80 / K40 / M40 | 保留原文 | 硬件名不译 |
| tensor2tensor | tensor2tensor（不译） | 代码库名保留 |
| Extended Neural GPU / ByteNet / ConvS2S / GNMT / MoE / Berkeley-Parser | 保留原文 | 模型名不译 |
| Recurrent Neural Network Grammar | 循环神经网络文法（Recurrent Neural Network Grammar） | 模型名首次标注，后保留 |
| NIPS / ICLR / NAACL / ACL | 保留原文 | 会议名不译 |

## 三、不确定 / 存疑的翻译点

1. **Abstract 与正文 6.1 节 BLEU 数值不一致**：摘要称英语到法语任务取得 41.8 BLEU，正文 6.1 称 41.0（arXiv v7 摘要更新后正文未同步，v1–v6 摘要为 41.0）。已在正文译文加 `[译注]` 说明，未改动原文。
2. **"abstractive summarization"** 译作「抽取式与生成式摘要」不完全准确，应仅指「生成式摘要」；为忠实原文，建议理解为「生成式摘要（abstractive summarization）」。备注于此，正文未加译注以免过度增注。
3. **"masking" 与 "masking out"**：3.1 节中 "This masking" 译「掩码」，3.2.3 中 "masking out (setting to −∞)" 译「掩蔽掉（置为 −∞）」；两处语义一致，措辞略有差异。
4. **"sustained single-precision floating-point capacity"** 译作「持续单精度浮点运算能力」，属意译，业界亦常称「持续单精度浮点吞吐」。
5. **"dispensing with recurrence and convolutions entirely"** 译作「彻底摒弃了循环与卷积」，其中 recurrence 指循环连接/循环结构，未直译为「循环性」以免生硬。
6. **"factorization tricks"** 译作「因式分解技巧」（对应文献 [21] Factorization tricks for LSTM networks），若读者习惯亦可作「分解技巧」。
7. **结论段原文笔误**："Making generation less sequential is another research goals of ours." 原文 goals 单复数不一致，译文按原意译出并加 `[译注]`。
8. **表 3 结构**：原表中 (D) 行下还有两行（$\epsilon_{ls}$ = 0.0/0.2 的标签平滑实验），原文未标行号；paper.md 中按原表位置重建，未加标注以免改动原意。
9. **附录图 3–5 中词表**：为注意力热力图的图像内容（词与 attention 矩阵），不属正文文字，故未逐词复现翻译，仅在图题与说明中注明。
10. **"intra-attention"** 规范术语表未收录，按通行译法「内部注意力」处理并在首次出现处标注英文。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（9 处）；图注内重复的页图链接去重（0 处）；成对重复的页图引用行去重（9 对（其中 9 对在并行修复中已去除，本次补修 0 对并统一核验），修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
