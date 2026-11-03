# 翻译笔记：Deep Residual Learning for Image Recognition

- 论文：Deep Residual Learning for Image Recognition（arXiv:1512.03385v1，2015-12-10，共 12 页）
- 译名：用于图像识别的深度残差学习
- 来源 PDF：`paper.pdf`（819,383 字节，从 https://arxiv.org/pdf/1512.03385 经代理下载）
- 译文：`paper.md`（全文双语对照）
- 规范依据：`C:/Users/13054/.claude/skills/nature-reader/references/cs-glossary.md`（术语表 + 翻译硬规则八条）

## 批量简化模式说明

本次为批量简化模式，与完整 nature-reader 流程的差异：

1. 未生成 source_map.json（原文-译文逐句映射文件）。
2. 未提取图片：文中所有图均以页码标注形式引用（"（原图见 paper.pdf 第 N 页）"）。
3. 表格以 Markdown 表格重建（表头译中文，数据原样保留），并同时在块引用中给出表题译文与页码。
4. 参考文献部分按规范整体保留英文原文，不翻译。
5. 公式按规范原样保留：编号公式用 `$$...$$`（式 (1)、(2)），行内公式用 `$...$`；公式中的花体字母按 arXiv LaTeX 源恢复为 `\mathcal{F}`、`\mathcal{H}`（PDF 文本提取无法区分字体，参见下文存疑点 6）。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| residual learning framework | 残差学习框架 | 摘要首现，全文统一 |
| residual function / residual mapping | 残差函数 / 残差映射 | 按上下文区分 |
| residual net / ResNet | 残差网络 / ResNet | ResNet 为模型名不译 |
| degradation problem | 退化问题 | 1. Introduction 首现 |
| plain nets | 普通网络（"plain" nets） | 保留原文引号；首次标注后统一"普通网络" |
| shortcut connection | 捷径连接 | 术语表无此条，按领域惯例 |
| identity mapping | 恒等映射 | |
| projection shortcut | 投影捷径 | linear projection → 线性投影 |
| zero-padding | 零填充 | |
| building block | 构建块 | |
| bottleneck | 瓶颈 | bottleneck design → 瓶颈设计 |
| feature map | 特征图 | |
| fully-connected layer | 全连接层 | 缩写 fc 保留 |
| global average pooling | 全局平均池化 | |
| downsampling / subsampling | 下采样 | 两词统一译"下采样" |
| stride | 步长 | |
| filter | 滤波器 | 卷积层语境，全文统一（不用"卷积核"） |
| element-wise addition | 逐元素相加 | |
| batch normalization (BN) | 批归一化（BN） | 首次标注后统一用 BN |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | 首次标注后统一用 SGD |
| vanishing/exploding gradients | 梯度消失/爆炸 | |
| backpropagation | 反向传播 | |
| overfitting | 过拟合 | |
| learning rate | 学习率 | |
| mini-batch | 小批量 | |
| weight decay | 权重衰减 | |
| momentum | 动量 | |
| dropout | 随机失活（Dropout） | 术语表规定；maxout 为网络名保留 |
| data augmentation | 数据增强 | scale/color augmentation → 尺度/颜色增强 |
| 10-crop testing | 10 裁剪测试 | |
| fully-convolutional | 全卷积 | |
| ensemble | 集成 | |
| state-of-the-art | 当前最优（state-of-the-art） | 按术语表；全文统一 |
| object detection | 目标检测 | |
| mAP | 平均精度均值（mAP） | 表 7 表题首现标注，之后统一 mAP |
| IoU | 交并比（IoU） | |
| non-maximum suppression (NMS) | 非极大值抑制（NMS） | |
| box refinement / box voting | 边界框精修 / 边界框投票 | |
| region proposal network (RPN) | 候选区域网络（RPN） | |
| RoI pooling | 感兴趣区域（RoI）池化 | 首次展开，之后统一 RoI 池化 |
| anchor | 锚框 | |
| ground truth | 真值（ground truth） | 附录 B 首现标注 |
| fine-tune | 微调 | |
| pre-training | 预训练 | |
| hyper-parameters | 超参数 | |
| per-class regression (PCR) | 逐类回归（PCR） | |
| category-agnostic | 类别无关 | |
| translation-invariant | 平移不变 | |
| logistic regression | 逻辑回归 | |
| binary classification | 二分类 | |
| image pyramid / feature pyramid | 图像金字塔 / 特征金字塔 | |
| multi-scale | 多尺度 | |
| spatial pyramid pooling | 空间金字塔池化 | |
| end-to-end | 端到端 | |
| warm up | 预热（warm up） | 4.2 节首现标注 |
| precondition | 预条件 | 3.1 节；"reasonable preconditioning" → 合理的预条件 |
| Multigrid | 多重网格（Multigrid） | |
| hierarchical basis preconditioning | 分层基预条件 | |
| Partial Differential Equations (PDEs) | 偏微分方程（PDE） | |
| vector quantization | 向量量化 | |
| multi-layer perceptrons (MLPs) | 多层感知机（MLP） | |
| gating functions | 门控函数 | |
| highway networks | Highway 网络 | 模型名保留 |
| VGG / GoogLeNet / Faster R-CNN / FitNet / Maxout / NIN / DSN / NoC | 不译 | 模型/方法名保留 |
| ImageNet / CIFAR-10 / MNIST / COCO / PASCAL VOC / ILSVRC | 不译 | 数据集与竞赛名保留 |
| FLOPs / GPU / Caffe / ReLU / Softmax / top-1 / top-5 | 不译 | 按术语表保留 |

数字与单位格式约定（规则 7）：million/billion 译为中文"万/亿"（1.28 million → 128 万；3.6 billion → 36 亿）；"k" 与科学计数法保留原文形式（50k 张、60 × 10⁴ 次迭代、1.8×10⁹ FLOPs）。

## 存疑 / 不确定点

1. **"this 103-layer network"（第 8 页）**：PDF 原文即如此（已用 PyMuPDF 二次核对），系 arXiv v1 版笔误，CVPR 2016 正式版为 "1202-layer"。译文按原文翻译并加译注说明。
2. **"expect that a shortcut connection is added"（4.1 节）**：原文 "expect" 应为 "except"（"除了……之外"），译文按 except 理解并加译注。
3. **"both in a per-class from"（附录 C）**：原文 "from" 应为 "form"，加译注。
4. **"negative samples being dominate" / "a single dominate object"（附录 C）**：原文 "dominate" 应为形容词 "dominant"，加译注。
5. **表 10/11 表题 "The system "baseline+++" include..."**：原文主谓不一致（include 应为 includes），保留原文并加译注。
6. **公式字体**：PDF 文本提取无法区分花体；按 arXiv 源将正文映射函数写为 $\mathcal{H}$、$\mathcal{F}$。式 (1)(2) 中 $\{W_i\}$、$W_s$ 下标按源文件恢复。
7. **"60 × 104 iterations"**：PDF 提取丢失上标，实为 60 × 10⁴，译文中按 60 × 10⁴ 处理。
8. **表 10/11 类别列缩写**（areo/mbike/tv 等）：为原表排版缩写（aeroplane/motorbike/tvmonitor），表格重建时保留原文缩写。
9. **图 4 图题中 "thin/bold curves"**：原文 thin 曲线为训练误差、bold 曲线为验证误差，译文按此对应。
10. **表 9 的列结构**：原表为 2 组训练数据（COCO train / COCO trainval）各对应一组测试数据（COCO val / COCO test-dev），每组含 mAP@.5 与 mAP@[.5, .95] 两列；重建表格时按此四列布局。
11. **图 3 中的架构 ASCII 图与图 5 内部标注文字**：属图内容，不重建，仅以块引用标注图题与页码（符合批量简化模式）。
12. **"oracle" testing（附录 C）**：保留 "oracle" 不译，指用真值类别充当分类预测的测试方式。
13. **Plain Network 的 "weighted layers"**：译为"加权层"（即含参数的层）；图 3 图题中的 "parameter layers" 译为"参数层"。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（9 处）；图注内重复的页图链接去重（12 处）；成对重复的页图引用行去重（9 对（其中 8 对在并行修复中已去除，本次补修 1 对并统一核验），修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
