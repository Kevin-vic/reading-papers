# 翻译笔记：VGG（Very Deep Convolutional Networks for Large-Scale Image Recognition）

- 论文：Very Deep Convolutional Networks for Large-Scale Image Recognition（arXiv:1409.1556v6，ICLR 2015 会议版）
- 作者：Karen Simonyan, Andrew Zisserman（Visual Geometry Group, University of Oxford）
- 译文文件：`paper.md`（全文双语对照）；原文 PDF：`paper.pdf`（共 14 页）
- 翻译依据：nature-reader 技能 `references/cs-glossary.md` 术语表与翻译硬规则八条

## 批量简化模式说明

本次为批量简化模式，相比完整 nature-reader 流程做了以下简化：

1. **未生成 source_map.json**：无逐句对齐索引，对齐通过 `paper.md` 内 **Original:** / **中文:** 段落配对实现。
2. **未做图片提取**：本论文正文**没有图**，只有 12 张表（Table 1–12）。每张表按规范以块引用给出表题译文并标注「（原表见 paper.pdf 第 N 页）」，表格内容以 Markdown 表格/代码块重建。
3. **表格内容不逐字对照**：表中数值、配置名（如 conv3-64、FC-4096）为代码/数字，按硬规则第 8 条不译；表题已译。
4. 参考文献整体保留原文（硬规则第 4 条）。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| convolutional network (ConvNet) | 卷积网络（ConvNet） | 首次全称对照，之后统一「卷积网络（ConvNet）」 |
| convolutional layer (conv. layer) | 卷积层 | |
| convolution filter | 卷积滤波器 | 全文统一「滤波器」 |
| receptive field / receptive window | 感受野 / 感受窗口 | |
| stride | 步长 | |
| spatial padding | 空间填充 | |
| spatial pooling / max-pooling | 空间池化 / 最大池化 | |
| fully-connected (FC) layer | 全连接（FC）层 | |
| soft-max | Softmax | 保留 |
| rectification non-linearity | 整流非线性 | ReLU 保留 |
| Local Response Normalisation (LRN) | 局部响应归一化（LRN） | |
| weight layer | 权重层 | |
| channels | 通道 | |
| feature map | 特征图 | |
| spatial resolution | 空间分辨率 | |
| decision function | 决策函数 | |
| regularisation | 正则化 | |
| weight decay | 权重衰减 | |
| dropout | 随机失活（Dropout） | |
| learning rate | 学习率 | |
| momentum | 动量 | |
| mini-batch gradient descent | 小批量梯度下降 | |
| back-propagation | 反向传播 | |
| batch size | 批次大小 | |
| epoch | 轮次（epoch） | |
| multinomial logistic regression objective | 多项式逻辑回归目标 | |
| pre-initialisation | 预初始化 | |
| crop | 裁剪块 / 裁剪 | 名词「裁剪块」、动词「裁剪」 |
| training scale / test scale | 训练尺度 / 测试尺度 | |
| single-scale / multi-scale training | 单尺度 / 多尺度训练 | |
| scale jittering | 尺度抖动（scale jittering） | |
| fine-tuning | 微调 | |
| dense evaluation | 密集评估 | |
| multi-crop evaluation | 多裁剪评估 | |
| fully-convolutional net | 全卷积网络 | |
| class score map | 类别得分图 | |
| class posteriors | 类别后验概率 | |
| sum-pooling / global average pooling | 求和池化 / 全局平均池化 | |
| data parallelism / model parallelism | 数据并行 / 模型并行 | |
| top-1 / top-5 error | top-1 / top-5 错误率 | top-1/top-5 按术语表保留 |
| ground truth | 真值（ground truth） | |
| localisation / classification | 定位（localisation）/ 分类 | ILSVRC 任务名 |
| track | 赛道 | |
| ensemble | 集成 | |
| fusion / late fusion | 融合 / 后期融合 | |
| state of the art | 当前最优（state of the art） | |
| bounding box | 边界框 | |
| single-class regression (SCR) | 单类回归（SCR） | |
| per-class regression (PCR) | 逐类回归（PCR） | |
| Euclidean loss | 欧氏损失 | |
| greedy merging | 贪心合并 | |
| pooling offsets | 池化偏移 | |
| intersection over union (IoU) | 交并比（IoU） | |
| object detection | 目标检测 | |
| feature extractor | 特征提取器 | |
| image descriptor | 图像描述子 | |
| L2-normalised | L2 归一化 | |
| linear SVM | 线性支持向量机（SVM） | |
| mean average precision (mAP) | 平均精度均值（mAP） | |
| mean class recall | 平均类别召回率 | |
| over-fitting | 过拟合 | |
| hand-crafted representations | 手工设计的表示 | |
| hyper-parameter | 超参数 | |
| semantic segmentation | 语义分割 | |
| image caption generation | 图像描述生成 | |
| benchmark | 基准测试 | |
| prior art | 既有方法 / 既有最优 | 按上下文取「既有方法（prior art）」或「既有最优（prior-art）」 |
| generalisation | 泛化 | |
| data augmentation | 数据增强 / 扩充 | 正文译「扩充（augment）」，指训练集扩充 |
| camera-ready | 终稿（camera-ready） | 附录 C v6 |

## 数字与单位

按硬规则第 7 条全文统一：大数译成中文数字（144M → 1.44 亿、1.3M → 130 万、370K → 37 万、50K → 5 万、100K → 10 万、22.5K → 2.25 万、10K → 1 万、9K → 9000、31K → 3.1 万、4.6K → 4600、4096-D → 4096 维）。表格内数值保留原文阿拉伯数字（单位见各表标题）。科学计数法统一为 LaTeX 形式（$10^{-2}$、$5 \cdot 10^{-4}$、$10^{-3}$）。

## 存疑 / 不确定的翻译点

1. **"commodity"（§1）**：原文 "With ConvNets becoming more of a commodity in the computer vision field"。译为「基础工具（commodity）」——commodity 在此指「人手可得的通用品」，另可译「日用品」。选「基础工具」贴合计算机视觉语境，括号保留原文。
2. **"prior-art configurations"（Abstract）**：译为「既有最优（prior-art）配置」。论文此处指之前取得最好成绩的配置，与后文 "prior art"（既有方法）略有不同，两处分别处理。
3. **原论文笔误 "mult-crop"（§4.3 正文）**：原文 "we compare dense ConvNet evaluation with mult-crop evaluation" 中 "mult-crop" 应为 "multi-crop"（§4.3 标题即为 MULTI-CROP EVALUATION）。译文按 "multi-crop" 处理并加译者注。
4. **Table 1 重建**：原表按「每列一个配置」排列且新增层以粗体显示。PDF 文本提取时六列交错、行序有轻微错位，粗体信息也无法从文本可靠恢复。译文按论文公认的 A/A-LRN/B/C/D/E 各列层序列重建为按列代码块（各列 conv 层数与权重层数可交叉验证：A/A-LRN=8 conv、B=10、C=13、D=13、E=16），**未恢复粗体标记**。如需逐像素核对，请查看 paper.pdf 第 3 页。
5. **"track"（Abstract）**：ILSVRC 的 classification/localisation 两个子任务，译「赛道（tracks）」。
6. **"camera-ready"（附录 C）**：译「终稿（camera-ready）」，保留原词。
7. **"isotropically-rescaled"**：译「各向同性缩放」，即长宽等比缩放。
8. **表 6 ILSVRC submission 行**：原文该行含 7 个网络的记号（D、C、E 各配置），按原表行序照录，未重排。
9. **LRN 参数句（§2.1）**："the parameters for the LRN layer are those of (Krizhevsky et al., 2012)" 译「LRN 层的参数与（Krizhevsky et al., 2012）相同」。
10. **"commodity" 之外的体例**：作者脚注（∗/+ current affiliation）已译；模型发布 URL 脚注保留原文并加说明。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（7 处）；图注内重复的页图链接去重（5 处）；成对重复的页图引用行去重（7 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
