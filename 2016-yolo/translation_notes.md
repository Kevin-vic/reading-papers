# 翻译笔记：《You Only Look Once: Unified, Real-Time Object Detection》

- 论文：arXiv:1506.02640v5 [cs.CV]（2016-05-09 版，与 CVPR 2016 版本同文）
- 目标目录：`C:/Users/13054/mclearning/papers/2016-yolo/`
- 产出：`paper.md`（全文双语对照）、`translation_notes.md`（本文件）
- PDF 共 10 页，全文已读完（10/10 页）

## 批量简化模式说明

本次为批量简化模式，与 nature-reader 全量模式相比有以下降级，已记录在案：

1. **未生成 source_map.json**：未做段落级到 PDF 页面的精确映射文件，图片位置仅在图注中以页码标注（如"原图见 paper.pdf 第 N 页"）。
2. **未提取图片**：未将 PDF 内嵌图片导出为独立文件，图以页码标注方式引用原 PDF。
3. **正文提取方式**：本机未安装 poppler（pdftoppm），Read 工具无法直接渲染 PDF 页面。改用 pymupdf（fitz 2.0）提取全文文本（`paper_fulltext.txt`，保留在目录中供校对），再据以翻译。文本层提取对段落/公式的线性顺序忠实，但与双栏排版的视觉位置有差异（如图 3 架构图标签为散乱提取，已按提取顺序保留在图注块中）。
4. **公式还原**：PDF 文本层中公式 (1)(2)(3) 为碎片化提取，已按 arXiv 源 LaTeX（1506.02640v5）还原为标准 LaTeX 形式，display 用 `$$...$$`，行内用 `$...$`。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| object detection | 目标检测 | 全文统一 |
| bounding box | 边界框 | 全文统一 |
| grid cell | 网格单元 | 首次出现标注（grid cell），之后用中文 |
| confidence (score) | 置信度（分数） | |
| intersection over union (IOU) | 交并比（IOU） | 术语表规范拼写为 IoU；原文一律写作 IOU，按原文保留 IOU（公式符号一致） |
| mean average precision (mAP) | 平均精度均值（mAP） | 首次出现标注，之后直接写 mAP |
| average precision (AP) | 平均精度（AP） | 4.5 节首次出现标注 |
| ground truth | 真值（ground truth） | 首次标注，之后用中文 |
| region proposal | 候选区域 | 首次标注（region proposal），之后统一 |
| non-maximal suppression / non-max suppression | 非极大值抑制（NMS） | 原文两种写法，首次出现标注，之后统一中文 |
| false positive | 假阳性 | |
| localization error | 定位错误 | |
| classification error | 分类误差 | |
| recall | 召回率 | |
| sliding window | 滑动窗口 | |
| fully connected layer | 全连接层 | |
| convolutional layer | 卷积层 | |
| average-pooling layer | 平均池化层 | |
| dropout | 随机失活（Dropout） | 首次标注，之后统一"随机失活" |
| data augmentation | 数据增强 | |
| pretrain / pretraining | 预训练 | |
| inference | 推理 | |
| end-to-end | 端到端 | |
| state-of-the-art | 当前最优（state-of-the-art） | 首次标注，之后统一"当前最优" |
| epoch | 轮次（epoch） | 首次标注，之后"轮次" |
| learning rate | 学习率 | |
| momentum | 动量 | |
| decay | 权重衰减 | 原文 "a decay of 0.0005"，指 weight decay |
| batch size | 批次大小 | |
| sum-squared error | 平方和误差 | |
| leaky rectified linear activation | 带泄漏的线性整流激活 | 即 Leaky ReLU；公式 $\varphi(x)$ 保留 |
| overfitting | 过拟合 | |
| co-adaptation | 协同适应 | |
| data augmentation | 数据增强 | |
| exposure / saturation | 曝光度 / 饱和度 | |
| HSV color space | HSV 色彩空间 | |
| downsampling | 下采样 | |
| aspect ratio | 长宽比 | |
| filter | 滤波器 | 术语表按上下文二选一，此处用"滤波器"（Fast YOLO 描述） |
| deformable parts models (DPM) | 可变形部件模型（Deformable Parts Models，DPM） | 首次展开标注，之后统一用 DPM |
| SVM | 支持向量机（SVM） | 首次标注 |
| grasp detection | 抓取检测 | |
| leaderboard | 排行榜 | |
| precision-recall curve | 精确率-召回率曲线 | 图 5(a) 图注 |
| F1 score | F1 分数 | |
| qualitative results | 定性结果 | |
| webcam | 网络摄像头 | |
| latency | 延迟 | |
| streaming video | 视频流 | |
| fps | 帧/秒（fps） | 首次出现标注，之后按原文混用 fps / 每秒 N 帧 |
| real-time | 实时 | |
| generalizable / generalizability | 可泛化 / 泛化能力 | |
| ensemble | 集成 | |
| tensor | 张量 | |
| feature extraction | 特征提取 | |
| post-processing | 后处理 | |
| background errors | 背景错误 | |

### 保留英文（不译）清单（本次论文涉及）

- 模型/方法：YOLO、Fast YOLO、DPM、R-CNN、Fast R-CNN、Faster R-CNN、GoogLeNet、VGG-16、MultiBox（Deep MultiBox）、OverFeat、MultiGrasp、Poselets、Selective Search、Inception（模块）、HOG、SIFT、Haar、Edge Boxes（仅参考文献）、R-CNN Minus R、CaffeNet、HyperNet、NoC、SDS、Feature Edit、DEEP ENS COCO、MR CNN、NUS NIN、BabyLearning、UMICH FGS STRUCT、D&T、ZF（Zeiler-Fergus）、VGG-M
- 数据集/基准：PASCAL VOC（2007/2010/2012）、ImageNet、Picasso Dataset、People-Art Dataset、COCO（仅表 3 方法名）
- 框架/硬件：Darknet、Caffe（Model Zoo）、Titan X、GPU
- 人名：Joseph Redmon、Santosh Divvala、Ross Girshick、Ali Farhadi、Sadeghi、Szegedy、Sermanet、Hoiem、Ren、Lin 等一律保留
- 会议/机构：CVPR、ECCV、ICCV、University of Washington、Allen Institute for AI、Facebook AI Research、ONR、NSF、The Allen Distinguished Investigator Award

## 数字与单位格式约定

- 百分比、小数保留原文写法（如 52.7% mAP、rate = .5、λnoobj = .5、IOU > .5）。
- 帧率：正文译"每秒 45 帧"或"45 fps"（保留数字与原文单位，不换算）。
- 论文原文用 ".5" 而非 "0.5" 处（λnoobj = .5、rate = .5、IOU > .5 等），原文保留，译文行文中如"0.5 fps"按原文。

## 不确定 / 存疑的翻译点

1. **公式 (1)(2)(3) 为还原件**：PDF 文本层提取碎片化，已按 arXiv v5 源 LaTeX 还原。$\mathbb{1}^{\text{obj}}_{ij}$ 指示函数记号、$\hat{x}_i$ 等帽子记号均按 arXiv 源。
2. **原文拼写/语法错误**（均按"不增不减"原则保留原文并加译注）：
   - 表 3 图注 "the forth highest scoring method"——应为 "fourth"，译文中已加 `[译注]`。
   - 4.3 节 "we run each model seperately"——应为 "separately"，译文按"分别"译，未逐字纠错。
   - 原文多处 "it's" 作所有格（如 "13.6% of it's top detections"、"predict it's class"），译文按所有格语义翻译。
3. **Table 3 列名还原**：PDF 提取中 "person" 与 "plant" 粘连，按 PASCAL VOC 标准 20 类顺序（aero, bike, bird, boat, bottle, bus, car, cat, chair, cow, table, dog, horse, mbike, person, plant, sheep, sofa, train, tv）还原。表中数据 21 列 × 20 行已全部保留。
4. **Figure 3 架构图**：图内标签为散乱提取，图注块中按 PDF 提取顺序列出原始标签文本，不保证与图中视觉布局一一对应，以原图（第 3 页）为准。图内文字属代码/参数名，未翻译。
5. **"repurpose classifiers to perform detection"**：译"改造分类器来执行检测"。"repurpose" 也有"复用/改作他用"之意，取"改造"更贴合当时 R-CNN 系方法的做法。
6. **"1 × 1 reduction layers"**：译"1 × 1 降维层"，指 NIN（Lin et al.）风格的降维卷积。
7. **"you only look once (YOLO)"**：译"你只需看一眼（You Only Look Once，YOLO）"，双关（方法名 + 字面义）在正文首次展开处注明。
8. **"decay of 0.0005"**：按权重衰减（weight decay）理解。若严格字面可作"衰减"，此处结合训练上下文取权重衰减，台账已记录。
9. **IOU vs IoU**：术语表规范为 IoU，但原文（含公式）一律写 IOU，为忠实原文全文保留 IOU。
10. **图 1 图内文字**（"1. Resize image. ..." 与 "Dog: 0.30" 等）为图的一部分，已双语列于图注块中；Figure 4、Figure 5 图内标签同样双语列出。
11. **"Non-max suppression" 与 "Non-maximal suppression"**：原文混用两种写法，均统一译"非极大值抑制（NMS）"，首次出现（Abstract 之前的图 1 图内文字）即标注。
12. **Abstract 分段**：原文 Abstract 为 4 个自然段，其中第 1、3 段较长，按语义拆为小段并保持一一对应，未删减任何句子。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（6 处）；图注内重复的页图链接去重（3 处）；成对重复的页图引用行去重（6 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
