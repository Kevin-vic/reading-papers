# 翻译笔记：Going deeper with convolutions (GoogLeNet, arXiv:1409.4842)

- 翻译日期：2026-11-03
- 源文件：`paper.pdf`（arXiv:1409.4842v1 [cs.CV]，2014-09-17，共 12 页，1,214,160 字节）
- 产出：`paper.md`（全文双语对照）
- 翻译规范依据：`C:/Users/13054/.claude/skills/nature-reader/references/cs-glossary.md`

## 本次为批量简化模式

按任务要求，本次采用批量简化模式：

- **未生成 source_map.json**：没有做段落级 source 映射文件。
- **未提取图片**：图（Figure 1-3）与表（Table 1-5）未导出为图片文件，在 `paper.md` 中以块引用标注页码指向原 PDF。
- **PDF 文本提取方式说明**：本机缺少 poppler-utils，Read 工具的 PDF 页面渲染不可用；改用 pypdf 6.16.2 逐页提取全文（脚本 `C:/Users/13054/mclearning/_extract_pdf_text.py`，中间产物 `fulltext.txt`），已完整覆盖全部 12 页。原文中的连字（如 "ﬁ"）已还原为正常字符。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| convolutional neural network (CNN) | 卷积神经网络（CNN） | 首现标注，后文统一用 CNN/卷积网络 |
| deep learning | 深度学习 | |
| object detection | 目标检测 | |
| state of the art | 当前最优（state-of-the-art） | 首现标注 |
| computational budget | 计算开销（computational budget） | |
| multiply-adds | 乘加运算 | |
| inference | 推理 | |
| Inception | Inception | 模型名，保留英文 |
| GoogLeNet | GoogLeNet | 模型名，保留英文 |
| Network in Network / Network-in-Network | Network in Network | 方法名，保留英文 |
| R-CNN | R-CNN | 方法名，保留英文 |
| Selective Search | Selective Search | 方法名，保留英文 |
| multi-box | multi-box | 方法名，保留英文 |
| DistBelief | DistBelief | 系统名，保留英文 |
| ILSVRC14 / ImageNet / MNIST / CIFAR | 保留英文 | 竞赛/数据集 |
| max pooling | 最大池化 | |
| average pooling | 平均池化 | |
| fully connected layer | 全连接层 | |
| dropout | 随机失活（Dropout） | |
| overfitting | 过拟合 | |
| rectified linear activation | 修正线性激活（ReLU） | 首现标注 ReLU |
| receptive field | 感受野 | |
| filter / filter bank | 滤波器 / 滤波器组 | |
| stride | 步长 | |
| patch | 图像块（patch） | |
| concatenate | 拼接 | |
| dimension reduction | 降维 | |
| embedding | 嵌入 | |
| pooling | 池化 | |
| Hebbian principle | 赫布原则（Hebbian principle） | |
| multi-scale | 多尺度 | |
| translation invariance | 平移不变性 | |
| localization | 定位 | |
| human pose estimation | 人体姿态估计 | |
| Gabor filters | Gabor 滤波器（Gabor filters） | |
| superpixel | 超像素 | |
| bounding box | 边界框 | |
| region proposal | 候选区域 | |
| recall | 召回率 | |
| ensemble | 集成 | |
| ensemble prediction | 集成预测 | |
| category-agnostic | 类别无关 | |
| mean average precision (mAP) | 平均精度均值（mAP） | |
| Jaccard index | Jaccard 指数 | |
| false positive | 假阳性 | |
| ground truth | 真值（ground truth） | |
| top-1 accuracy / top-5 error rate | top-1 准确率 / top-5 错误率 | |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| momentum | 动量 | |
| learning rate | 学习率 | |
| epoch | 轮次（epoch） | |
| Polyak averaging | Polyak 平均 | |
| convergence | 收敛 | |
| auxiliary classifier | 辅助分类器 | |
| gradient | 梯度 | |
| regularization | 正则化 | |
| discriminative | 判别力（discriminative） | 首现标注 |
| fine-tune | 微调 | |
| pre-training | 预训练 | |
| photometric distortions | 光度畸变（photometric distortions） | |
| crop | 裁剪（crop） | |
| bounding box regression | 边界框回归 | |
| contextual model | 上下文模型 | |
| region classifier | 区域分类器 | |
| memory footprint | 内存占用（memory footprint） | |
| hyperparameter | 超参数 | |
| base network | 基础网络（base network） | |
| building blocks | 构建块（building blocks） | |
| knobs and levers | 可调旋钮（knobs and levers） | 意译 |
| softmax loss | softmax 损失 | softmax 保留 |

## 不确定 / 存疑的翻译点

1. **arXiv 版本差异**：本次下载到的是 **arXiv v1**（2014-09-17 首版）。该版本与 CVPR 2015 正式版存在差异，最明显的是表 5 中 GoogLeNet 单模型检测 mAP：v1 为 38.02%，正式版为 37.1%。翻译以 v1 原文为准，未按正式版改动。
2. **原文笔误（已加译注）**：
   - 第 1 节 "so that the they do not end up to be a purely academic curiosity"——"the they" 疑为笔误，已加 `[译注]`。
   - 第 3 节 "This is as an easy and safe way"——"as" 疑为笔误，已加 `[译注]`。
   - 第 4 节 "all the included the knobs and levers"——"the included" 疑为笔误，已加 `[译注]`。
3. **"deep" 双关**：标题与正文中 "deep" 兼指"网络深"与 Inception 梗（"we need to go deeper"），中文难以完整保留双关，正文中已加括号说明。
4. **"12× fewer parameters"**：译为"参数量仅为……的 1/12"，避免中文"少 12 倍"的歧义。
5. **"naïve"**：v1 原文写作 "na¨ıve"，已还原为 naïve，译为"朴素"。
6. **图 2 跨页**：图 2 及其两个子图（(a) 第 4 页末、(b) 第 5 页）跨 paper.pdf 第 4-5 页，页码标注为"第 4-5 页"。
7. **图 3 为网络结构图**：PDF 提取出的图 3 节点文本（"Conv 7x7+2(S)"、"DepthConcat" 等，为 Caffe 风格层配置）属图内容，仅译图题，节点文本未逐一翻译，已在块引用中说明。
8. **表格处理**：Table 1-5 均以"英文原表 + 中文表"两份呈现，数值原样保留；表 4 中 "ensemble approach" 列在 v1 原文排版中错位（"Fisher vectors" 落在表头行），Markdown 中按 v1 原始行列顺序呈现，未擅自调整。
9. **参考文献**：按规范整体保留原文；其中含特殊字符的条目（[3] Ümit V. Çatalyürek 等）已从 PDF 提取的乱序字符还原为正常拼写。
10. **Inception 名称来源**："in conjunction with the famous 'we need to go deeper' internet meme" 中的 meme 指电影《盗梦空间》（Inception）衍生网络梗，正文按原文直译，未加注。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（5 处）；图注内重复的页图链接去重（2 处）；成对重复的页图引用行去重（5 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
