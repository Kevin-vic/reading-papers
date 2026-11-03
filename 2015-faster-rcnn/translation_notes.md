# 翻译笔记：Faster R-CNN（arXiv:1506.01497v3）

## 模式说明

本次为**批量简化模式**：未做 source_map.json 与图片提取，图以页码标注（如「原图见 paper.pdf 第 2 页」）。原文 PDF 共 14 页，下载自 https://arxiv.org/pdf/1506.01497（6,913,803 字节，v3 版，2016-01-06，为 NIPS 2015 会议版扩展版，含会议版之后新增的竞赛结果、表 12 等内容）。

本机未安装 poppler，Read 工具无法直接渲染 PDF，故用 PyMuPDF（pymupdf）提取全文文本后通读（提取脚本：C:/Users/13054/mclearning/scripts/extract_pdf_text.py，中间文本：paper_text.txt）。表格数据由 PDF 文本流线性重建并按 PASCAL VOC 标准类别顺序核对；图 5、图 6 的图内检测标签文字（如 "person : 0.992"）为图像内文字，未逐条收录，以图题标注代替。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| object detection | 目标检测 | 全文统一 |
| region proposal / object proposal | 候选区域 | object proposal 在相关工作语境下也统一为「候选区域」 |
| Region Proposal Network (RPN) | 候选区域网络（RPN） | 模型名，此后统一 RPN |
| objectness score | 目标性分数 | 术语表未收录；objectness 取「目标性」 |
| anchor / anchor box | 锚框 | 术语表译名 |
| reference box | 参考框 | 与锚框区分 |
| bounding box | 边界框 | |
| convolutional feature map | 卷积特征图 | feature map → 特征图 |
| fully convolutional network (FCN) | 全卷积网络（FCN） | |
| sliding window | 滑动窗口 | |
| fully-connected layer | 全连接层 | |
| receptive field | 感受野 | |
| stride | 步长 | |
| scale / aspect ratio | 尺度 / 长宽比 | |
| translation invariant | 平移不变 | |
| Intersection-over-Union (IoU) | 交并比（IoU） | |
| non-maximum suppression (NMS) | 非极大值抑制（NMS） | |
| mean Average Precision (mAP) | 平均精度均值（mAP） | |
| multi-task loss | 多任务损失 | |
| objective function | 目标函数 | |
| log loss | 对数损失 | |
| robust loss function (smooth L1) | 稳健损失函数（smooth L1） | smooth L1 保留 |
| two-class softmax | 两类 softmax | softmax 保留英文 |
| logistic regression | 逻辑回归 | |
| mini-batch | 小批量 | |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| back-propagation | 反向传播 | |
| learning rate | 学习率 | |
| momentum | 动量 | |
| weight decay | 权重衰减 | |
| fine-tuning | 微调 | |
| pre-training / pre-trained | 预训练 | |
| end-to-end | 端到端 | |
| overfitting | 过拟合 | |
| state-of-the-art | 当前最优（state-of-the-art） | |
| benchmark | 基准测试 | |
| baseline | 基线 | |
| ablation study | 消融实验 | |
| hyper-parameter | 超参数 | |
| image pyramid / filter pyramid | 图像金字塔 / 滤波器金字塔 | |
| pyramid of regression references | 回归参考的金字塔 | 意译，原文 "pyramid of regression references" |
| image-centric sampling | 以图像为中心的采样 | |
| cross-boundary anchor | 跨边界锚框 | |
| ground-truth box | 真实标注框 | ground-truth 统一译「真实标注」 |
| positive / negative sample | 正样本 / 负样本 | |
| one-stage / two-stage | 单阶段 / 两阶段 | |
| class-agnostic / class-specific | 类别无关 / 类别相关 | |
| pipeline | 流程 | |
| region-wise | 逐区域 | 表 5 计时项 |
| RoI (Region of Interest) | 感兴趣区域（RoI） | |
| RoI pooling | RoI 池化 | |
| hard-negative mining | 难负样本挖掘 | |
| SVD | 奇异值分解（SVD） | |
| ensemble | 集成 | |
| residual net | 残差网络 | ResNet-101 等模型名保留 |
| recall | 召回率 | |
| frame rate / fps | 帧率 / fps | fps 保留 |
| conv3_1 | conv3_1 | 层名保留 |
| superpixel | 超像素 | |
| low-level features | 低级特征 | |
| k-means | k-means | 保留 |
| 模型名/方法名 | 保留英文 | R-CNN, Fast R-CNN, Faster R-CNN, SPPnet, OverFeat, MultiBox, DeepMask, Selective Search (SS), EdgeBoxes (EB), DPM, CPMC, MCG, ZF, VGG-16, GoogleNet, ResNet-101, Caffe, HOG |
| 数据集/竞赛 | 保留英文 | PASCAL VOC, MS COCO, ImageNet, ILSVRC |
| 会议/期刊名 | 保留英文 | NIPS, CVPR, ICCV, ECCV, ICLR, ICML, BMVC, TPAMI, IJCV（正文引用及参考文献中均不译） |

## 不确定/存疑的翻译点

1. **"objectness" 译名**：cs-glossary.md 未收录。取「目标性」（如「目标性分数」）。中文社区另有「目标性」「物体性」等译法，本译文统一为「目标性」。
2. **"down-stream"**：原文即带连字符（非 PDF 断行连字符），译文「下游」，已按原文保留连字符。
3. **"areo" 表头**：表 6/7 原文表头 "areo" 疑为 aeroplane 的笔误（原文如此），表题处已加译注说明，表中保留原文写法。
4. **"pyramid of regression references"**：直译「回归参考的金字塔」，意译为「回归参考的金字塔」/「作为回归参考的多尺度锚框」按上下文处理。
5. **"image-centric" sampling**：译「以图像为中心的采样」，Fast R-CNN 论文术语，中文文献常用「以图像为中心」。
6. **图 4 图形内部文字**：第 10 页图 4 的图内文字在 PDF 文本层为乱码（曲线字体编码），无法提取，仅图题可读；不影响译文完整性。
7. **5fps/17fps 数值**：保留原文写法（fps），未转成「帧/秒」，全文一致。
8. **"object bounds"**：译「目标边界」（与 bounding box「边界框」区分，bounds 指坐标回归的边界）。
9. **数字格式**：原文的 10⁴/10⁶ 上标在正文译文与公式中以 $10^4$/$10^6$ 呈现；2,400 等千分位写法按原文保留（译文里写作 2,400）。
10. **"RoI warping"**：保留英文（[15] 中提出的层名），未译「变形」以免与已有中文文献译名冲突。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（11 处）；图注内重复的页图链接去重（7 处）；成对重复的页图引用行去重（11 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
