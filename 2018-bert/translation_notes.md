# BERT 论文翻译笔记

- 论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（arXiv:1810.04805v2）
- 译文文件：`paper.md`（中英双语对照，段落级 **Original:** / **中文:** 格式）
- 原文 PDF：`paper.pdf`（775,166 字节，共 16 页）
- 规范依据：`C:/Users/13054/.claude/skills/nature-reader/references/cs-glossary.md`

## 批量简化模式说明

本次为批量简化模式，做了以下降级：

1. **未生成 source_map.json**：原文-译文对齐以段落级「**Original:** / **中文:**」标注承载，不做逐句 source map。
2. **未提取图片**：论文的 5 幅图（图 1–5）不单独导出，仅在译文中以块引用标注图题译文与所在页码（`（原图见 paper.pdf 第 N 页）`）。
3. **表格处理**：8 张表格以 Markdown 表格重建，数据逐格保留，表头中英对照，表题以块引用形式翻译并标注页码。
4. **PDF 文本提取方式**：本机 Read 工具缺 poppler，改用 PyMuPDF（脚本 `C:/Users/13054/mclearning/.tmp_extract/extract_bert.py`）逐页提取文本到 `extracted.txt`，再人工按原文版式重组段落（公式、图表元素在提取时顺序会乱，已按论文原版式复原）。
5. **References 整体保留原文不译**（约 50 条）。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| Bidirectional Encoder Representations from Transformers (BERT) | BERT（保留英文） | 模型名不译 |
| masked language model (MLM) | 掩码语言模型（MLM） | 首次全称，之后用 MLM |
| next sentence prediction (NSP) | 下一句预测（NSP） | |
| pre-training / fine-tuning | 预训练 / 微调 | |
| language model (LM) | 语言模型（LM） | |
| language representation model | 语言表示模型 | 首次出现标注英文 |
| unidirectional / bidirectional | 单向 / 双向 | |
| left-to-right (LTR) | 从左到右（LTR） | 首次出现标注全称 |
| self-attention | 自注意力 | 术语表钉死 |
| attention head | 注意力头 | |
| Transformer block | Transformer 块 | |
| encoder / decoder | 编码器 / 解码器 | |
| hidden state | 隐藏状态 | |
| token | Token（保留英文） | 术语表规定保留 |
| vocabulary id | 词表中的编号 | |
| word / sentence / paragraph embedding | 词 / 句子 / 段落嵌入 | |
| contextual representation | 上下文表示 | |
| position / segment embedding | 位置 / 分段嵌入 | |
| downstream task | 下游任务 | |
| feature-based approach | 基于特征的方法 | |
| cloze task | 完形填空任务（Cloze task） | |
| natural language inference (NLI) | 自然语言推理（NLI） | |
| question answering (QA) | 问答（QA） | |
| named entity recognition (NER) | 命名实体识别（NER） | |
| machine translation | 机器翻译 | |
| transfer learning | 迁移学习 | |
| state-of-the-art | 当前最优（state-of-the-art，SOTA） | 首次全称标注 |
| benchmark | 基准测试 | |
| perplexity (ppl) | 困惑度（ppl） | |
| ablation study | 消融实验 | |
| hyperparameter | 超参数 | |
| batch size | 批次大小 | |
| epoch | 轮次 | |
| learning rate warmup | 学习率预热 | |
| weight decay | 权重衰减 | |
| dropout | 随机失活（Dropout） | 术语表钉死 |
| cross entropy loss | 交叉熵损失 | |
| Softmax | Softmax（保留） | |
| span | 片段 | |
| sequence tagging | 序列标注 | |
| entailment | 蕴含 | |
| paraphrasing | 复述改写 | |
| sentiment analysis | 情感分析 | |
| commonsense inference | 常识推理 | |
| corpus | 语料库 | |
| ensembling | 集成 | |
| data augmentation | 数据增强 | |
| check point / checkpoint | 检查点 | |
| dev set / test set | Dev 集 / Test 集 | 保留 Dev/Test 原名 |
| denoising auto-encoder | 去噪自编码器 | |
| log-likelihood | 对数似然 | |
| feed-forward network | 前馈网络 | |

保留英文不译（术语表规定）：BERT、OpenAI GPT、ELMo、Transformer、LSTM、BiLSTM、CRF、WordPiece、GLUE、MNLI、QQP、QNLI、SST-2、CoLA、STS-B、MRPC、RTE、WNLI、SQuAD v1.1/v2.0、SWAG、TriviaQA、BooksCorpus、Wikipedia、ImageNet、CoNLL-2003、Billion Word Benchmark、Adam、gelu、relu、Cloud TPU、GPU、tensor2tensor、Peters/Radford/Vaswani 等人名、ACL/EMNLP/NAACL/ICLR/CoNLL/NIPS/ICML 等会议名。

## 数字与单位统一

- 大数统一用中文计数：110M→1.1 亿、340M→3.4 亿、800M words→8 亿词、2,500M→25 亿、3.3 billion→33 亿、113k→11.3 万、100k→10 万、1M steps→100 万步；3,600、65.1、91.0 等直接保留阿拉伯数字。
- 学习率科学计数法保留原文格式：5e-5、1e-4 等。
- 模型规模记号保留：L=12、H=768、A=12、128,000 tokens/batch、≤512 tokens。
- 百分比与 F1/EM 分数一律保留原文格式（80.5%、+1.5 F1 等）。

## 公式处理

全部以 LaTeX 保留：display 公式用 `$$...$$`（如 $P_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}$），行内用 `$...$`（如 $C \in \mathbb{R}^H$、$\hat{s}_{i,j} > s_{\mathrm{null}} + \tau$）。原文中 $\le$、$\ge$ 等符号按 LaTeX 规范书写。

## 不确定 / 存疑的翻译点

1. **原文笔误（已加 [译注]）**：
   - 附录 A.1「standard langauge model」→ language（原文拼写错误）；
   - 附录 A.1「the masked LM only make predictions」→ only makes（主谓一致错误）；
   - 附录 A.1「we evaluate the impact this procedure」→ 疑缺 of；
   - 附录 A.2「To speed up pretraing」→ pretraining；
   - 附录 A.4「Here we studies」→ Here we study；
   - 正文 4.2「befor fine-tuning on SQuAD」→ before（此处在正文翻译中按原意翻译，未在正文加注，在此记录）。
2. **「degenerate text-∅ pair」** 译为「退化的『文本-∅』对」——指文本分类/序列标注任务中只有一个文本、没有文本 B 的情况。
3. **「grounded commonsense inference」**（SWAG）译为「有依据的常识推理」，指推理需以给定情境为依据。
4. **「R.M. Reader」**（表 2）为系统名，保留原文不译。
5. **「Pod configuration」**（附录 A.2）保留原文，指 Cloud TPU 的 Pod 部署形态（多机互联），未找到简洁且无歧义的中文对应。
6. **「case-preserving WordPiece」** 译为「保留大小写的 WordPiece」。
7. **BERTBASE / BERTLARGE 下标**：Markdown 中以 BERT_BASE / BERT_LARGE 表示，与原文 LaTeX 下标等价。
8. **图 4** 原图题只有一句（子图 a–d 的标签为图内文字，PDF 提取不全），译注中已按原文完整翻译其说明文字。
9. **脚注**：原文 15 条脚注全部翻译，以块引用形式紧随所属段落；脚注中的 URL 保留原文。
10. **「The Annotated Transformer」**：一篇著名博客教程名，保留英文加书名号，未译。
11. **表 1 的「Pre-OpenAI SOTA」行**：系统名保留英文；「SOTA」为论文原文用词，对应术语表「当前最优」。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（9 处）；图注内重复的页图链接去重（4 处）；成对重复的页图引用行去重（9 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
