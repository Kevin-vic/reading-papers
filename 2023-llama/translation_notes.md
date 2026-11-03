# 翻译笔记：LLaMA: Open and Efficient Foundation Language Models

- 论文：arXiv:2302.13971v1（2023-02-27，Meta AI）
- 成果文件：`paper.md`（全文双语对照）、`paper.pdf`（原始 PDF，27 页）
- 规范依据：`~/.claude/skills/nature-reader/references/cs-glossary.md`

## 一、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| foundation language model | 基座语言模型 | 首次出现带原文；后续「基座模型」 |
| Large Language Models (LLMs) | 大语言模型 | 全名首次展开后缩写 LLM |
| state-of-the-art (SOTA) | 当前最优 | 按术语表 |
| benchmark | 基准测试 | 按术语表 |
| scaling laws | 扩展定律 | 全译名「扩展定律（scaling laws）」 |
| compute budget | 算力预算 | |
| inference | 推理 | 按术语表 |
| token | token（不译） | 术语表「保留英文」条目 |
| few-shot / zero-shot | 少样本 / 零样本 | 首次出现带原文 |
| instruction finetuning | 指令微调 | 全译名「指令微调（instruction finetuning）」 |
| instruction-tuned | 经指令微调的 | |
| perplexity | 困惑度 | 按术语表 |
| pre-training / fine-tuning | 预训练 / 微调 | 按术语表 |
| transformer architecture | Transformer 架构 | 模型名保留 |
| normalization | 归一化 | 按术语表 |
| pre-normalization | 预归一化 | 小节标题保留原文并加译名 |
| activation function | 激活函数 | 按术语表 |
| rotary positional embeddings (RoPE) | 旋转位置嵌入（RoPE） | |
| causal multi-head attention | 因果多头注意力 | |
| optimizer | 优化器 | 按术语表 |
| learning rate / weight decay / gradient clipping | 学习率 / 权重衰减 / 梯度裁剪 | 按术语表 |
| warmup | 预热 | 全译名「预热（warmup）步」 |
| batch size | 批次大小 | |
| checkpointing | 检查点 | 全译名「检查点（checkpointing）技术」 |
| model / sequence parallelism | 模型并行 / 序列并行 | |
| exact match | 精确匹配 | |
| closed-book question answering | 闭卷问答 | |
| common sense reasoning | 常识推理 | |
| reading comprehension | 阅读理解 | 按术语表 |
| mathematical reasoning | 数学推理 | |
| code generation | 代码生成 | |
| majority voting | 多数投票 | |
| hallucinate | 幻觉 | 按术语表 |
| co-reference resolution | 指代消解 | |
| stereotype | 刻板印象 | |
| toxicity / bias | 毒性 / 偏见 | |
| misinformation | 虚假信息 | |
| carbon footprint | 碳足迹 | |
| Power Usage Effectiveness (PUE) | 电能利用效率（PUE） | |
| greedy decoding | 贪心解码 | |
| temperature | 温度 | 按术语表（temperature 0.1/0.8） |
| prompt | 提示 | 按术语表 |
| robustness | 鲁棒性 | 术语表未收；用通行译名 |
| n-gram / smoothing | n-gram（不译）/ 平滑 | n-gram 保留；smoothing 译「平滑」 |
| power law | 幂律 | |
| open-ended generation | 开放式生成 | |
| likelihood | 似然 | |
| docstring | docstring（不译） | 代码语境专有名词 |
| grid（电力） | 电网 | |
| thermal design power | 热设计功耗 | |
| epoch | 轮次 | 按术语表 |
| tokenization | 词元化 | 按术语表（本文统一选「词元化」） |
| byte-pair encoding (BPE) | 字节对编码（BPE） | |

## 二、保留不译（按术语表）

模型/方法名：LLaMA、GPT-3、Chinchilla、PaLM、Gopher、OPT、BLOOM、GLM、GPT-Neo(X)、GPT-J、LaMDA、Minerva、Flan-PaLM、Flan-T5-XXL、OPT-IML、PaLM-Coder、BERT、GPT-2、Megatron-LM、Megatron-Turing NLG、T5、Jurassic-1、Transformer、Transformer-XL、AdamW、RMSNorm、SwiGLU、ReLU、RoPE、SentencePiece、fastText、CCNet、xformers、PyTorch。
数据集/基准：CommonCrawl、C4、Wikipedia、GitHub、Gutenberg、Books3、ThePile、ArXiv、Stack Exchange、BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC-e/ARC-c、OpenBookQA、Natural Questions、TriviaQA、RACE、MATH、GSM8k、HumanEval、MBPP、MMLU、RealToxicityPrompts、CrowS-Pairs、WinoGender、TruthfulQA、One Billion Word。
人名/机构/会议：Hoffmann、Kaplan、Brown、Vaswani、Chowdhery、Lewkowycz 等一律保留；Meta AI 保留；NeurIPS、ACL、EMNLP-CoNLL、NAACL-HLT、LREC、AAAI/IAAI、Interspeech 保留（仅在 References 中出现，References 整体不译）。

## 三、数字与单位统一规则

- 参数/数据规模统一用「中文大数 + 原文括号」：7B → 70 亿（7B）；1.4T → 1.4 万亿（1.4T）；200B → 2000 亿（200B）；38.5B → 385 亿（38.5B）；4M tokens → 400 万 token。
- 倍数保留数字：「10× smaller」→「小 10 倍」；「5-10× smaller」→「小 5-10 倍」。
- 物理/能耗单位保留原文：400W、MWh、tCO2eq、kg CO2eq/KWh、380 token/sec/GPU、GPU-hours。
- 表格内数字一律原样保留，不转换。

## 四、公式处理

- 行内公式用 $...$：如 $\frac{2}{3}4d$、$\beta_1 = 0.9$、$P(\text{completion}|\text{context})/P(\text{completion}|\text{"Answer:"})$。
- 独立公式用 $$...$$：
  - $$Wh = GPU\text{-}h \times (\text{GPU power consumption}) \times PUE$$
  - $$tCO2eq = MWh \times 0.385$$
- 论文除上述公式外无更多数学公式（本文为工程型论文）；附录生成示例中的 E = mc2 按原文保留。

## 五、图题表题页码标注

| 图表 | paper.md 位置 | PDF 页码 |
|---|---|---|
| 表 1 Pre-training data | 2.1 节 | 第 2 页 |
| 表 2 Model sizes | 2.2 节 | 第 3 页 |
| 图 1 Training loss | 2.4 节末 | 第 3 页 |
| 表 3 Common Sense Reasoning | 3.1 节 | 第 4 页 |
| 表 4 NaturalQuestions | 3.2 节 | 第 4 页 |
| 表 5 TriviaQA | 3.2 节 | 第 5 页 |
| 表 6 RACE | 3.3 节 | 第 5 页 |
| 表 7 MATH/GSM8k | 3.4 节 | 第 6 页 |
| 表 8 Code generation | 3.5 节 | 第 6 页 |
| 表 9 MMLU | 3.6 节 | 第 7 页 |
| 表 10 Instruction finetuning | 第 4 节 | 第 7 页 |
| 图 2 Evolution of performance | 3.7 节 | 第 8 页 |
| 表 11 RealToxicityPrompts | 5.1 节 | 第 8 页 |
| 表 12 CrowS-Pairs | 5.2 节 | 第 9 页 |
| 表 13 WinoGender | 5.3 节 | 第 10 页 |
| 表 14 TruthfulQA | 5.4 节 | 第 10 页 |
| 表 15 Carbon footprint | 第 6 节 | 第 11 页 |
| 图 3 问答示例 | 附录 A | 第 17 页 |
| 表 16 MMLU 详细结果 | 附录 B | 第 18 页 |

## 六、本次为批量简化模式说明

1. **未生成 source_map.json**：本次按批量简化模式执行，未做逐句源文-译文对齐映射文件。
2. **未提取图片**：图（Figure 1/2/3）以页码标注方式引用，未导出图片文件；如需看图，请打开 `paper.pdf` 对应页码。
3. **PDF 文本提取方式**：本机缺少 poppler（`pdftoppm` 不可用），Read 工具无法直接渲染 PDF，故用 `pdftotext -layout` + pypdf 双重提取文本后重建排版。原文双栏版式在提取时交错，已按论文标准内容顺序重建；所有正文段落、表格数字均经与 pypdf 逐页输出交叉核对。
4. **表格数据重建**：所有 16 张表的数据均从 PDF 逐页提取核对；表 16（附录 B，57 行 × 9 列）由脚本按行配对重建，与提取序列逐项一致。

## 七、存疑与说明

1. **CrowS-Pairs（表 12）列顺序**：pdftotext 布局模式输出有交错（GPT3/OPT 每类别疑似两值），按 pypdf 文本顺序重建为「LLaMA 65B / GPT3 175B / OPT 175B」三列，与表注「compare LLaMA-65B with OPT-175B and GPT3-175B」一致。交错输出中的第二组数值（如 Gender 的 73.3/68.6）疑为提取重影，未采用。
2. **StackExchange 采样比例**：pdftotext 布局输出在 2.5% 与 2.0% 之间断裂，取 2.0%（与官方 v1 论文一致）。
3. **Table 10 排序**：原表为左右两栏（左：未指令微调；右：经指令微调），已合并为单表按分数升序排列，并在小节文字中说明分组。
4. **附录 C 示例 2（推荐信）**：原文与提示之间无显式分界（推荐信为紧接 Fibonacci 例的连续生成），按论文版式拆为两段对照。
5. **附录 C 代码例**：论文原文即为模型生成输出的展示（solve 函数），属于"代码块不译、注释可译"范围，代码原样保留。
6. **WinoGender 表 13 的代词标签**：「her/her/she」「his/him/he」「their/them/someone」保留原文不译（为基准测试的原始代词组标签）。
7. **「perplexity」与「likelihood」**：均按术语表译为「困惑度」「似然」。
8. **原文笔误**：3.2 节 "LLaMA-65B achieve"（主谓不一致）、3.6 节 "As show in Table 8"、5.3 节 "Theses cases" 等为原文笔误，翻译时按原意处理，未逐处加译注。
9. **References 编码修复**：PDF 提取中损坏的字符（Rozière、Mazaré、Réjean、Jürgen、Dániel、Karafiát、Cernocký、Christopher Ré、Ilić、Castagné、François、Gallé、Guzmán、Łukasz 等）已按原论文修复；References 整体保留英文不译。
10. **脚注**：正文脚注 1（GitHub 链接）、2（xformers）、3（PerspectiveAPI）、4（OPT 日志）、5（CodaLab 链接）均保留并编号。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（12 处）；图注内重复的页图链接去重（7 处）；成对重复的页图引用行去重（12 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
