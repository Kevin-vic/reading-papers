# 翻译笔记：Improving Language Understanding by Generative Pre-Training（GPT-1）

- 源文件：`paper.pdf`（OpenAI 官方 PDF，https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf ，541,036 字节，共 12 页）
- 产出：`paper.md`（中英对照全文翻译）
- 说明：本次为**批量简化模式**——未生成 source_map.json、未提取图片；图、表以页码标注（原图见 paper.pdf 第 N 页）。原文文本由 PyMuPDF（fitz）提取后人工对照校正公式与表格。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| natural language understanding | 自然语言理解 | |
| generative pre-training | 生成式预训练 | 论文标题核心术语 |
| discriminative fine-tuning | 判别式微调 | |
| language model (LM) | 语言模型 | |
| textual entailment | 文本蕴含 | |
| natural language inference (NLI) | 自然语言推理（NLI） | |
| question answering | 问答 | |
| semantic similarity | 语义相似度 | |
| text classification | 文本分类 | |
| commonsense reasoning | 常识推理 | |
| document classification | 文档分类 | |
| supervised / unsupervised / semi-supervised learning | 监督学习 / 无监督学习 / 半监督学习 | |
| pre-training / fine-tuning | 预训练 / 微调 | |
| transfer learning | 迁移学习 | |
| word embedding | 词嵌入 | |
| representation | 表示 | |
| downstream task | 下游任务 | |
| task-agnostic | 任务无关 | |
| token | 词元 | 首次「词元（token）」，之后统一「词元」 |
| delimiter token | 分隔符词元 | |
| self-attention / masked self-attention | 自注意力 / 掩码自注意力 | |
| multi-headed self-attention | 多头自注意力 | 首次出现标注英文 |
| position-wise feedforward layer | 逐位置前馈层 | |
| position embedding | 位置嵌入 | |
| long-range dependency | 长程依赖 | |
| stochastic gradient descent | 随机梯度下降（SGD） | |
| perplexity | 困惑度 | |
| byte-pair encoding (BPE) | 字节对编码（BPE） | |
| layer normalization | 层归一化 | |
| dropout | 随机失活（Dropout） | |
| auxiliary objective | 辅助目标 | |
| traversal-style approach | 遍历式方法 | |
| zero-shot | 零样本 | |
| ablation study | 消融实验 | |
| state of the art (SOTA) | 当前最优（SOTA） | 首次全称，之后用 SOTA |
| ensemble | 集成 | |
| hyperparameter | 超参数 | |
| learning rate | 学习率 | |
| minibatch | 小批量 | |
| epoch | 轮次（epoch） | |
| warmup | 预热（warmup） | |
| generalization / convergence | 泛化 / 收敛 | |
| inductive bias | 归纳偏置 | |
| thresholding | 阈值化 | |
| log-probability | 对数概率 | |
| paraphrase detection | 释义检测 | |
| story completion | 故事补全 | |
| premise / hypothesis | 前提 / 假设 | |
| entailment / contradiction / neutral | 蕴含 / 矛盾 / 中立 | NLI 三分类标签 |
| world knowledge | 世界知识 | |
| multi-task | 多任务 | |
| LSTM | 长短期记忆网络（LSTM） | 首次标注，之后用 LSTM |

保留英文不译：Transformer、GPT、ELMo、Adam、GELU、BPE、GLUE、SNLI、MultiNLI、QNLI、RTE、SciTail、RACE、Story Cloze Test、MRPC、QQP、STS-B、CoLA、SST-2、BooksCorpus、1B Word Benchmark、SQuaD、CNN、DPRD、WSC（winograd schemas）、人名、会议名（NAACL/ACL/EMNLP/ICLR/CVPR/AAAI 等）。

## 数字与单位约定

- 保留原文数字格式：2.5e-4、6.25e-5、0.1、0.01、18.4、5.7k、550k、40,000（中文里写作 40000）。
- 百分比保留百分号原文形式（8.9%、5.7%、4.2% 等）。
- 公式全部 LaTeX 原样保留：display 公式用 `$$...$$`，行内用 `$...$`；式 1–5 编号保留。

## 表格还原说明

- 表 1–5 由 PDF 文本提取还原为 Markdown 表格，数字与模型名原样保留，表头指标简写（mc/acc/F1/pc）在表注中以中文解释。
- 表 2 中 CAFE [58] 行的 QNLI、RTE 两列在提取时缺失，按原论文版式以 "-" 补齐（已在 paper.md 中注明）。
- 表 4 采用两行表头以还原原文的列分组（Classification / Semantic Similarity / GLUE）。

## 存疑与译注

1. **"DPRD [46] (winograd schemas)"**（第 8 页，Zero-shot Behaviors）：原文即写作 DPRD，疑为笔误；引用 [46] 对应 Winograd Schema Challenge（通常简称 WSC）。已在译文中加译注，未改动原文。
2. **"an score of 45.4"**（第 6 页）：原文语法笔误（应为 "a score"），已加译注。
3. **"more reasoning type questions that other datasets"**（第 6 页）：原文 "that" 疑为 "than" 之笔误，已加译注。
4. **"upto 1.5%"**（第 6 页）：原文拼写 "upto"（应为 "up to"），译文作「最高 1.5%」，未加注。
5. **"referrents"**（第 8 页）：原文拼写（应为 "referents"），译文按「指代对象」处理，未加注。
6. **"bytepair encoding"**（第 5 页）：原文连写，规范写法 byte-pair encoding，译名「字节对编码（BPE）」。
7. **"SQuaD"**（第 6 页）：原文大小写如此（通常写作 SQuAD），保留原文。
8. **"Stories Cloze Test"**（摘要与引言）：原文如此（该数据集通常写作 Story Cloze Test），保留原文。
9. **式 2 的表述**：原文写 "$h_l = \text{transformer\_block}(h_{l-1}) \forall i \in [1, n]$"，其中循环变量 $i$ 按原文保留（上下文指第 $i$ 层，原文用 $i$ 而非 $l$，疑为原文笔误，但按「公式不译」原则原样保留）。
10. **PDF 文本提取**：个别连字（ﬁ）已还原为标准字符；页眉脚注（gluebenchmark 链接、ftfy、spacy 链接）保留并译注位置。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（5 处）；图注内重复的页图链接去重（2 处）；成对重复的页图引用行去重（5 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
