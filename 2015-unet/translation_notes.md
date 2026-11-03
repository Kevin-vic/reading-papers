# 翻译笔记：U-Net: Convolutional Networks for Biomedical Image Segmentation

- 论文：arXiv:1505.04597v1 [cs.CV]，2015（MICCAI 2015）
- 译者：Claude（批量简化模式）
- 日期：2026-11-03
- 源文件：`C:/Users/13054/mclearning/papers/2015-unet/paper.pdf`（1,648,684 字节，8 页）
- 成果：`C:/Users/13054/mclearning/papers/2015-unet/paper.md`

## 批量简化模式说明

本次翻译为批量简化模式，相对完整 nature-reader 流程做了以下降级：

1. **未生成 source_map.json**：未做段落级原文溯源映射。
2. **未做图片提取**：文中图仅以页码标注（如"原图见 paper.pdf 第 2 页"），未提取 PNG 图件。
3. **PDF 读取降级**：本机 Read 工具依赖的 pdftoppm（poppler）不可用，无法直接渲染 PDF；改用 Python `pypdf` 提取文本层后翻译。文本层质量良好（arXiv LaTeX 源），但图 1 架构图内的散落标注文字（尺寸数字等）仅按图内标注方式收录翻译，未逐一还原图形布局。
4. **公式还原**：PDF 文本层中的公式排版被打散（分数、求和上下标分离），已按 arXiv LaTeX 源码样式人工还原为 `$$...$$`/`$...$`，数学内容未改动。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| deep network / deep convolutional network | 深度网络 / 深度卷积网络 | 首次出现给英文对照 |
| data augmentation | 数据增强 | 与术语表一致 |
| annotated training samples | 标注训练样本 | |
| contracting path | 收缩路径 | 本文核心术语，全文统一 |
| expansive path / expanding path | 扩张路径 | 摘要用 expanding、正文用 expansive，译文统一 |
| end-to-end | 端到端 | 与术语表一致 |
| sliding-window convolutional network | 滑动窗口卷积网络 | |
| patch | 图像块 | |
| tile | 图像块 | 与 patch 同译"图像块"；"input tile size" 译"输入图像块尺寸" |
| overlap-tile strategy | 重叠平铺（overlap-tile）策略 | 论文自造词，保留英文并加译名 |
| localization | 定位 | |
| context | 上下文 | |
| max pooling | 最大池化 | 与术语表一致 |
| fully convolutional network | 全卷积网络（fully convolutional network） | |
| pooling / upsampling operators | 池化算子 / 上采样算子 | |
| up-convolution | 上卷积（up-convolution） | |
| feature channel / feature map | 特征通道 / 特征图 | 与术语表一致 |
| fully connected layer | 全连接层 | 与术语表一致 |
| valid part of convolution | 卷积的有效部分 | |
| unpadded convolution | 无填充卷积 | |
| segmentation map | 分割图 | |
| mirroring | 镜像 | |
| GPU memory | GPU 显存 | |
| elastic deformation | 弹性形变 | |
| invariance | 不变性 | |
| weighted loss | 加权损失 | |
| ground truth | 真值（ground truth） | 后文统一用"真值" |
| rectified linear unit (ReLU) | 修正线性单元（rectified linear unit，ReLU） | 后文用 ReLU |
| stride | 步长 | 与术语表一致 |
| downsampling | 下采样 | |
| concatenation | 拼接 | |
| crop / cropping | 裁剪 | |
| batch size | 批次大小 | |
| momentum | 动量 | 与术语表一致 |
| energy function | 能量函数 | |
| pixel-wise soft-max | 逐像素 Softmax | Softmax 按术语表保留 |
| cross entropy | 交叉熵 | 与术语表一致 |
| activation | 激活值 | |
| weight map | 权重图 | |
| separation border | 分离边界 | |
| morphological operations | 形态学操作 | |
| Gaussian distribution | 高斯分布 | 与术语表一致 |
| standard deviation | 标准差 | |
| unit variance | 单位方差 | |
| incoming nodes | 输入节点 | |
| shift and rotation invariance | 平移与旋转不变性 | |
| displacement vector | 位移向量 | |
| bicubic interpolation | 双三次插值 | |
| Dropout | 随机失活（Dropout） | 与术语表一致 |
| serial section transmission electron microscopy | 连续切片透射电子显微镜 | |
| membrane / membranes | 细胞膜 | |
| membrane probability map | 膜概率图 | |
| thresholding | 阈值化 | |
| warping error | warping 误差 | 竞赛指标，保留英文；正文首次出现加引号 |
| Rand error | Rand 误差 | 同上 |
| pixel error | 像素误差 | |
| pre- / postprocessing | 前处理 / 后处理 | |
| probability map | 概率图 | |
| IOU (intersection over union) | 交并比（intersection over union，IoU） | 与术语表一致；论文写作 IOU，译文用 IoU |
| cell tracking challenge | 细胞追踪挑战赛 | |
| neuronal structures | 神经元结构 | |
| EM stacks | EM（电子显微镜）堆栈 | EM = electron microscopy |
| phase contrast microscopy | 相差显微镜 | |
| DIC (differential interference contrast) | 微分干涉相差（differential interference contrast，DIC） | |
| VNC (ventral nerve cord) | 腹神经索（ventral nerve cord，VNC） | |
| supplementary material | 补充材料 | |
| unsupervised feature learning | 无监督特征学习 | |
| state of the art | 当前最优水平（state of the art） | |

不译/保留：U-Net（正文原样写作 u-net）、Caffe、ImageNet、ISBI、EM、ReLU、Softmax、HeLa、U373、Drosophila、NVidia Titan GPU、人名（Ronneberger、Ciresan、Krizhevsky、Dosovitskiy 等）、数据集名（"PhC-U373"、"DIC-HeLa"）、表 1 中的队伍名（DIVE、IDSIA、KTH-SE、IMCB-SG、HOUS-US 等）。

## 存疑 / 不确定的翻译点

1. **"There is large consent that..."（摘要首句）**：arXiv v1 原文如此，"consent" 疑为 "consensus" 之误（v2 及后续版本已改为 "There is large consensus that..."）。已按原文翻译并加译注。
2. **"expanding path"（摘要）与 "expansive path"（正文）**：同一概念两种写法，译文统一为"扩张路径"，未逐处标注差异。
3. **warping error / Rand error**：EM 分割挑战赛的专有评估指标，业界通用做法是保留英文原文，本译文保留英文并加中文"误差"后缀；表 1 表头按原文保留。
4. **"tile" 与 "patch" 的区分**：两词在本文语境下都指输入图像子块，统一译"图像块"，未做细分（patch 偏滑动窗口语境、tile 偏平铺语境）。
5. **表 1 中间的 "..." 行**：PDF 提取仅得 "..."，为原文省略行（第 5–9 名未列出），已在表格中保留。
6. **图 1 内标注的还原**：图内尺寸序列（572x572、284²、570x570 等）是图形布局元素，未逐条列入正文；关键图例文字（max pool 2x2、up-conv 2x2、conv 3x3, ReLU、conv 1x1、copy and crop 等）已收录翻译。
7. **"Acknowlegements"**：原文章节标题拼写如此（少一个 d），保留原文拼写。
8. **"segmenation"（第 7 页脚注上方正文）**：PDF 提取出现一次 "This segmenation task"（少一个 t），按上下文译为"该分割任务"，未加译注。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（5 处）；图注内重复的页图链接去重（1 处）；成对重复的页图引用行去重（5 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
