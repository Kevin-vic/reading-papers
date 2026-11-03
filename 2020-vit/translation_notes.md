# 翻译笔记：An Image is Worth 16x16 Words (ViT)

## 本次为批量简化模式

- 未生成 `source_map.json`，未做图片提取。图/表以「原图见 paper.pdf 第 N 页」页码标注（N 为 PDF 物理页码，非论文印刷页码；论文印刷页码 = PDF 页码 − 2）。
- PDF 共 22 页，已全部读取（Read 工具无法渲染 PDF，本机缺 poppler，改用 PyMuPDF 提取文本后通读全文；提取文本存于同目录 `_extracted_text.txt`，可作校对参考）。
- 图/表页码对照：图 1→p.3；表 1→p.5；表 2→p.6；图 2→p.6；图 3/4/5→p.7；图 6→p.8；图 7→p.9；表 3→p.13；表 4→p.14；表 5/6→p.15；表 7→p.16；图 8→p.16；图 9→p.17；表 8→p.17；图 10→p.18；图 11→p.18；图 12→p.19；图 13→p.20；图 14→p.21；表 9→p.22。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| Vision Transformer (ViT) | 视觉 Transformer（ViT） | 模型名，首次出现标注中文，之后可简称 ViT |
| Transformer | Transformer | 模型名，保留英文 |
| self-attention | 自注意力（Self-Attention） | 首次出现给对照，之后统一「自注意力」 |
| multihead self-attention (MSA) | 多头自注意力（MSA） | |
| attention head | 注意力头 | |
| token | 词元（token） | 首次对照后统一「词元」 |
| [class] token | [class] 词元 / 分类词元 | BERT 的 [class] token 保留原形 |
| patch | 图像块（patch） | 首次对照后统一「图像块」 |
| patch embedding | 图像块嵌入 | |
| position embedding | 位置嵌入 | 与「位置编码」不同，本文指可学习嵌入 |
| embedding | 嵌入 | |
| linear projection | 线性投影 | |
| classification head | 分类头 | |
| MLP | MLP（多层感知机） | 首次标注后统一 MLP |
| GELU | GELU | 激活函数名，保留 |
| Layernorm (LN) | 层归一化（Layernorm，LN） | |
| residual connection | 残差连接 | |
| inductive bias | 归纳偏置（inductive bias） | 首次对照后统一「归纳偏置」 |
| translation equivariance | 平移等变性 | |
| locality | 局部性 | |
| pre-training / pre-train | 预训练 | |
| fine-tuning / fine-tune | 微调 | |
| transfer | 迁移 | |
| downstream task | 下游任务 | |
| benchmark | 基准测试（benchmark） | |
| state of the art (SOTA) | 当前最优（state-of-the-art，SOTA） | 首次对照后统一「当前最优」 |
| baseline | 基线 | |
| hybrid | 混合模型（hybrid） | 指 ResNet 主干 + ViT |
| few-shot | 少样本 | |
| linear few-shot accuracy | 线性少样本准确率 | |
| accuracy | 准确率 | |
| top-1 / top-5 | top-1 / top-5 | 保留原形 |
| regularization | 正则化 | |
| weight decay | 权重衰减 | |
| dropout | 随机失活（Dropout） | |
| label smoothing | 标签平滑 | |
| learning rate | 学习率 | |
| warmup | 预热（warmup） | |
| epoch | 轮次（epoch） | |
| batch size | 批次大小（batch size） | |
| grid search | 网格搜索 | |
| development set | 开发集（development set） | |
| early-stopping | 早停（early-stopping） | |
| gradient clipping | 梯度裁剪 | |
| receptive field | 感受野 | |
| attention distance | 「注意力距离」（attention distance） | 本文自定义概念，保留引号 |
| masked patch prediction | 掩码图像块预测 | |
| masked language modeling | 掩码语言建模 | |
| self-supervision / self-supervised | 自监督 | |
| semi-supervised | 半监督 | |
| supervised | 监督 | |
| unsupervised | 无监督 | |
| generative model | 生成模型 | |
| qkv | qkv | 查询/键/值，公式中保留 qkv 记号 |
| query / key / value | 查询（query）/ 键（key）/ 值（value） | |
| softmax | softmax | 公式保留 |
| principal component | 主成分 | |
| cosine similarity | 余弦相似度 | |
| ablation study | 消融实验 | |
| FLOPs / exaFLOPs | FLOPs / exaFLOPs | 保留原形 |
| TPUv3-core-days | TPUv3-core-days | 保留原形（Google 的算力计量单位） |
| feature map | 特征图 | |
| global average pooling (GAP) | 全局平均池化（GAP） | |
| batch normalization | 批归一化（Batch Normalization） | |
| group normalization | 组归一化（Group Normalization） | |
| standardized convolutions | 标准化卷积 | |
| Axial Attention | 轴向注意力（Axial Attention） | |
| Attention Rollout | Attention Rollout | 方法名，保留 |
| dense layer | 全连接层 | |
| BiT (Big Transfer) | Big Transfer（BiT） | 模型名，保留 |
| Noisy Student | Noisy Student | 模型名，保留 |
| iGPT (image GPT) | image GPT（iGPT） | 模型名，保留 |
| ViT-B/L/H | ViT-B/L/H | 模型名，保留 |
| R50x1, R152x2 等 | R50x1, R152x2 等 | ResNet 变体记号，保留 |

## 翻译硬规则执行情况

1. 公式原样保留：式 (1)–(8) 以 `$$...$$` 重建，行内数学用 `$...$`；下标/上标按原文 LaTeX 记号（如 $z_L^0$、$x_p^1 E$、$U_{qkv}$）。
2. 模型名/人名/数据集/会议名均保留英文（ViT、ResNet、BERT、JFT-300M、VTAB、ImageNet-21k、ICLR 等）。
3. 图题/表题已全部翻译并以块引用标注原 PDF 页码；表格数字原样保留。
4. References 整体保留原文，未翻译。
5. 首次出现术语给「中文（English）」对照，之后统一中文。
6. 全文翻译：正文 + 附录（A–D.10）+ 脚注 1/2/3 + Acknowledgements 均翻译，无删节；提取文本中的明显断行已按原文排版合并。
7. 数字与单位统一：大数用中文计数单位（100B 参数 → 1000 亿；14M-300M images → 1400 万至 3 亿张图像；1.3M → 130 万；303M → 3.03 亿；1M steps → 100 万步）；百分数保留原格式（88.55%）。
8. 无代码块；变量记号（x, z, D, N 等）保留不译。

## 不确定 / 存疑的翻译点

1. **「vanilla transformers」**（2 Related Work）：译为「普通 Transformer」。「vanilla」在原文语境指「未加视觉专用改进的标准 Transformer」，若读者习惯「原生 Transformer」亦可。
2. **「token」**：术语表允许「词元化/分词」两种语境译法。本文中 token 指图像块在 Transformer 中的抽象单位，统一译为「词元（token）」；与 BERT 的 [class] token 对应的「分类词元」在 Figure 1 图中即为 CLS 块。
3. **「stem」**（附录 D.4）：译为「起始部分」。原文 "right after the stem of the model" 指输入嵌入投影这一前端部分，不是 ResNet 的 stem 卷积层，此处存疑度低但译文用「模型的起始部分」以避开歧义。
4. **「bi-quadratic scaling」**（D.5）：译为「双二次增长」，指 ViT 注意力代价随输入分辨率四次方增长（序列长度二次 × 注意力二次），中文文献少见此说法，保留直译。
5. **「dowsampling」**（4.4 原文拼写）：PDF 原文即 "total dowsampling ratio"（应为 downsampling 的笔误）。译文按「总下采样比率」处理，不另加译注。
6. **图 13 图题原文**：原文 "inference time (left)" 与左图 FLOPs 矛盾，按规范「不增不减」原则未修正，译文如实对应并保留左/右标注。
7. **Table 3 中 ViT-L/{16,32} 的轮次列**：原文 "30/90"（ImageNet-21k 行），分别对应 L/32=90 与 L/16=30，表中如实保留。
8. **提取文本中连字与重音字符**：PDF 提取将 ﬁ 转为 fi、重音字母有误，译文与原文均已还原（Hénaff、Zürich、Łukasz、Täckström、Aäron、Lučić）。
9. **「The illustration of the Transformer encoder was inspired by Vaswani et al. (2017)」**（图 1 图题末句）：译为「Transformer 编码器的图示灵感来自 Vaswani et al. (2017)」。
10. **中文译名**：标题按任务指定译作「一张图值 16×16 个词：大规模图像识别中的 Transformer」。学界另有「一幅图胜过 16×16 个词」等译法，本稿统一用任务指定版本。

## 产出文件

- `C:/Users/13054/mclearning/papers/2020-vit/paper.md` — 全文双语对照翻译（约 108 KB）
- `C:/Users/13054/mclearning/papers/2020-vit/paper.pdf` — 原文 PDF（arXiv 2010.11929v2，3,743,814 字节）
- `C:/Users/13054/mclearning/papers/2020-vit/_extracted_text.txt` — PyMuPDF 提取的纯文本（含页码标记，供校对）

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（16 处）；图注内重复的页图链接去重（7 处）；成对重复的页图引用行去重（16 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
