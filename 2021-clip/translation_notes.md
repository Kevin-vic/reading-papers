# 翻译笔记：CLIP 论文（arXiv:2103.00020v1，48 页）

- 目标文件：`C:/Users/13054/mclearning/papers/2021-clip/paper.md`（全文双语对照）
- 原文 PDF：`C:/Users/13054/mclearning/papers/2021-clip/paper.pdf`（6,813,639 字节）
- 本文本提取自 arXiv LaTeX 源码（`src/clip_paper.tex` + `clip_paper.bbl` + `linear-probe-table.tex` + `zero-shot-table.tex`），表格数据与参考文献以源码为准，正文以 PDF 文本层为准。

## 模式说明

本次为**批量简化模式**：
- 未做 source_map.json 与图片提取；图以页码标注（"原图见 paper.pdf 第 N 页"），图内元素文字（如 Figure 21 的 36 个分类器预测文本）未逐条翻译。
- 图题/表题翻译集中在"图表清单"一节（位于 References 之前），表格数据紧跟对应表题之后以 Markdown 表格原样呈现。
- 表格由 LaTeX tabular 自动转换；粗体（各数据集最高分置信区间内）与下划线（类别内最佳）语义保留为 `**...**` 与 `<u>...</u>`。
- References 共 211 条，整体保留原文（英文），仅清理了 LaTeX 断行与转义符号。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| natural language supervision | 自然语言监督 | 首次出现写「自然语言监督（natural language supervision）」 |
| zero-shot / zero-shot transfer | 零样本 / 零样本迁移（zero-shot transfer） | 全文统一"零样本" |
| few-shot / one-shot / N-shot | 少样本（few-shot）/ 单样本（one-shot）/ N 样本 | |
| pre-training / pre-trained | 预训练 | |
| fine-tuning | 微调 | |
| contrastive objective | 对比目标函数（contrastive objective） | |
| image/text encoder | 图像编码器 / 文本编码器 | |
| caption / image captioning | 图像说明文字（caption）/ 图像说明文字生成 | 未用"字幕"以避免与视频字幕混淆 |
| prompt / prompt engineering | 提示（prompt）/ 提示工程（prompt engineering） | |
| ensembling | 集成（ensembling） | |
| representation learning | 表示学习（representation learning） | |
| linear probe / linear classifier | 线性探针（linear probe）/ 线性分类器 | |
| logistic regression | 逻辑回归（logistic regression） | |
| state of the art / SOTA | 当前最优（state-of-the-art，简称 SOTA） | 首次出现标注，之后用 SOTA |
| distribution shift | 分布偏移（distribution shift） | |
| effective / relative robustness | 有效鲁棒性 / 相对鲁棒性 | |
| bag-of-words (BoW) | 词袋（bag-of-words，BoW） | |
| byte pair encoding (BPE) | 字节对编码（byte pair encoding，BPE） | |
| attention pooling | 注意力池化（attention pooling） | |
| cosine similarity | 余弦相似度（cosine similarity） | |
| cross entropy | 交叉熵 | |
| temperature parameter τ | 温度参数 τ | |
| minibatch | 小批量（minibatch） | |
| mixed-precision | 混合精度（mixed-precision） | |
| gradient checkpointing | 梯度检查点（gradient checkpointing） | |
| weight decay | 权重衰减（weight decay） | |
| backbone | 主干网络（backbone） | |
| hypernetwork | 超网络（hypernetwork） | |
| OCR | 光学字符识别（OCR） | 首次展开，之后用 OCR |
| geo-localization | 地理定位（geo-localization） | 附录 E.4 沿用术语表中"地理定位" |
| action recognition | 动作识别（action recognition） | |
| fine-grained classification | 细粒度分类（fine-grained classification） | |
| webly supervised learning | 网络监督学习（webly supervised learning） | |
| class design | 类别设计（class design） | |
| surveillance | 监控（surveillance） | |
| denigration harm / representational harm | 贬损伤害（denigration harm）/ 代表性伤害（representational harm） | |
| spurious correlations | 虚假关联（spurious correlations） | |
| masked self-attention | 掩码自注意力 | |
| layer normalization | 层归一化 | |
| multinomial logistic regression | 多项逻辑回归 | |
| hold-out performance | 保留集（hold-out）性能 | |
| top-1 / top-5 | top-1 / top-5 | 保留英文 |
| 数据增强 data augmentation | 数据增强（data augmentation） | |

**保留英文不译**：CLIP、ResNet、ViT、Transformer、EfficientNet、BiT、SimCLRv2、BYOL、MoCo、VirTex、GPT、BERT、Unicoder-VL、ImageNet、YFCC100M、WIT、MS-COCO、MNIST、SVHN、CIFAR-10/100、Kinetics-700、UCF-101、FairFace、CelebA、Adam、Softmax、GELU、InfoNCE 等（按 cs-glossary.md 规则）。

## 数字与单位

- 大数量用中文单位：400 million → 4 亿；12.8 billion → 128 亿；63M parameters → 6300 万参数；1.28 million → 128 万。
- 技术参数保留数字：batch size 32,768、vocab 49,152、温度 0.07、λ 范围 $10^{-6}$ 到 $10^{6}$。
- 原文的 `10⁻⁶`、`N × N`、`N² − N` 等数学表达式保留。

## 不确定 / 存疑的翻译点

1. **表 5 的第三行数据**：FairFace 性别分类表中每个模型组除 Male/Female 外还有一行无标签数据（tex 源码如此，如 Linear Probe CLIP 的 97.4/96.5/…）。数值恰为男、女两行的均值，判断为合并后的总体值。按"不增不减"原则原样保留，并在表题译注中说明。若需绝对精确，请对照 paper.pdf 第 21 页原表。
2. **表 14 中 "59.0$\x i$"**：LaTeX 原文为 `59.0\xxi`（作者疑似笔误，应为上标 i，对应脚注 i = Mahajan et al., 2018），转换后显示为 59.0$x$i。原文如此，未改动。
3. **"A photo of a {label}." 等提示模板**：为保持可用性，译文采用「提示模板原文（中文释义）」双写形式，例如 "A photo of a {label}."（一张 {label} 的照片。）。模板内英文为模型实际输入，未译。
4. **"state of the art"**：术语表规定首次「当前最优（state-of-the-art，简称 SOTA）」，文中个别处按语境译为"当前最优水平/当前最优"，SOTA 缩写沿用原文。
5. **t_u / t_v / I_f / T_f 等变量**：代码/变量名不译，保留（如 t_u、t_v、I_f）。
6. **"an embarrassingly simple baseline"**：译为"简单到令人尴尬的基线"（第 6 节），"令人尴尬"为字面对应，亦可读作"简单到离谱"；取字面直译。
7. **BiT-M ResNet-152x2**：正文一处写 "ResNet-50x64"，图 10 相关段落写作 "ResNet-50x64"（原文即为 "ResNet-50x64"，PDF 与 tex 一致），非 RN50x64 笔误，照录。
8. **SVHN 的 "full number SVHN"**：译为"完整数字 SVHN"，指完整门牌号（多数字）任务；此为直译，更准确的表述是"完整（门牌）号码"。
9. **pm-0 / pm-10**（Youtube-BB 与 ImageNet-Vid 评估设定）：保留原文不译，含义为按 ImageNet 类层级最大池化的两种设定。
10. **图 21 图内文字**（36 个零样本分类器的提示文本与概率条）未翻译，仅翻译图题；图内预测文本在 PDF 文本层可读，若需完整对照可后续补做。
11. **正文提取断词**：PDF 文本层存在连字符断词（如 "over-ﬁtting"），原文引用中已按原词拼写还原；正文引用与 PDF 版式断行不完全一致属正常。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（30 处）；图注内重复的页图链接去重（12 处）；成对重复的页图引用行去重（30 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
