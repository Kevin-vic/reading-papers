# 翻译笔记：ImageNet Classification with Deep Convolutional Neural Networks（NeurIPS 2012）

## 一、本次为批量简化模式

- 本次翻译为**批量简化模式**：未生成 source_map.json，未提取/分离图片文件。图、表均以页码标注，读者可对照 paper.pdf 查看原图。
- 图/表页码定位（依据 PDF 中 caption 所在页）：
  - 图 1（ReLU vs tanh 训练曲线）→ paper.pdf 第 3 页
  - 图 2（网络结构示意图）→ paper.pdf 第 5 页
  - 图 3（96 个 11×11×3 卷积核）→ paper.pdf 第 6 页
  - 图 4（定性评估：预测标签 + 最近邻检索）→ paper.pdf 第 8 页
  - 表 1（ILSVRC-2010 结果对比）→ paper.pdf 第 7 页
  - 表 2（ILSVRC-2012 结果对比）→ paper.pdf 第 7 页
- 公式按 PDF 文本恢复为标准 LaTeX（display 公式用 $$...$$，行内用 $...$），未改动数学内容。PDF 文本提取中公式有乱码，已按论文原貌恢复。
- PDF 提取时将连字 fi/fl（如 "ﬁve"、"efﬁcient"）恢复为标准拼写；正文人名 "Cire¸san" 保留 PDF 原样（正确拼写为 Cireșan，参考文献同样保留原文）。

## 二、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| convolutional neural network (CNN) | 卷积神经网络（CNN） | 首次出现标注全称，之后简称 |
| deep convolutional neural network | 深度卷积神经网络 | |
| object recognition | 目标识别 | 与 object detection（目标检测）区分 |
| state-of-the-art | 最优结果（state-of-the-art） | 摘要中出现，未展开全称式 |
| fully-connected layer | 全连接层 | |
| max-pooling | 最大池化 | |
| softmax | Softmax | 保留英文，首字母大写 |
| non-saturating / saturating nonlinearity | 非饱和/饱和非线性 | |
| dropout | 随机失活（dropout） | 首次出现标注，之后统一"随机失活" |
| overfitting | 过拟合 | |
| regularization | 正则化 | |
| label-preserving transformations | 保持标签的变换 | |
| learning capacity | 学习容量 | |
| feedforward neural network | 前馈神经网络 | |
| stationarity of statistics and locality of pixel dependencies | 统计平稳性和像素依赖的局部性 | |
| kernel | 卷积核 | 全文统一；kernel map 译"卷积核映射" |
| kernel map | 卷积核映射（kernel map） | 即后来的 feature map（特征图），2012 年论文用 kernel map |
| stride | 步长 | |
| receptive field | 感受野 | |
| rectified linear unit (ReLU) | 修正线性单元（ReLU） | 首次全称，之后 ReLU |
| gradient descent | 梯度下降 | |
| contrast normalization | 对比度归一化 | |
| local response normalization | 局部响应归一化 | 标题；正文"response normalization"译"响应归一化" |
| brightness normalization | 亮度归一化 | |
| lateral inhibition | 侧抑制 | |
| hyper-parameter | 超参数 | |
| cross-validation | 交叉验证 | |
| columnar CNN | 柱状（columnar）CNN | |
| pooling / overlapping pooling | 池化 / 重叠池化 | |
| mean activity | 平均值 | 指训练集像素均值，非神经元激活 |
| multinomial logistic regression | 多项逻辑回归 | |
| log-probability | 对数概率 | |
| PCA | 主成分分析（PCA） | |
| covariance matrix | 协方差矩阵 | |
| eigenvector / eigenvalue | 特征向量 / 特征值 | |
| Gaussian | 高斯 | |
| data augmentation | 数据增强 | |
| patch | 图像块 | |
| geometric mean | 几何平均 | |
| co-adaptation | 协同适应 | |
| forward pass | 前向传播 | |
| backpropagation | 反向传播 | |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| batch size | 批次大小（batch size） | |
| momentum | 动量 | |
| weight decay | 权重衰减 | |
| learning rate | 学习率 | |
| bias | 偏置 | |
| epoch (cycles) | 轮次（epoch） | 原文用 "cycles"，首次标注 epoch |
| sparse-coding | 稀疏编码 | |
| Fisher Vector (FV) | Fisher 向量（FV） | |
| densely-sampled features | 稠密采样特征 | |
| fine-tuning | 微调（fine-tuning） | |
| pre-trained / pre-training | 预训练 | |
| supervised learning | 监督学习 | |
| unsupervised pre-training | 无监督预训练 | |
| feature activation | 特征激活 | |
| Euclidean distance / separation | 欧氏距离 | |
| auto-encoder | 自编码器（auto-encoder） | |
| infero-temporal pathway | 下颞通路（infero-temporal pathway） | 神经科学术语，指人类视觉系统腹侧通路 |
| Mechanical Turk | Mechanical Turk | 保留英文，加"众包工具"说明 |
| ILSVRC | ILSVRC | 首次展开全称"ImageNet 大规模视觉识别挑战赛（ImageNet Large-Scale Visual Recognition Challenge，ILSVRC）"，之后用 ILSVRC |
| top-1 / top-5 error rate | top-1 / top-5 错误率 | 保留 top-1/top-5 原文 |
| 数字格式 | 6000 万参数 / 65 万神经元 / 120 万 / 1500 万 / 5 万 / 15 万 / 890 万 | 英文 million 统一转"万"；百分比、模型编号（如 5 × 5 × 48）、bit 数保留原文格式 |

## 三、保留英文（不译）清单

模型/方法名：ReLU、Softmax、SGD、SIFT、Fisher Vectors、dropout（首注后转中文译名）、PCA（首注后转中文译名）。
数据集/竞赛：ImageNet、ILSVRC、CIFAR-10、MNIST、NORB、Caltech-101/256、LabelMe、Pascal Visual Object Challenge（挑战赛名保留并附译名）。
硬件：GPU、GTX 580、NVIDIA。
人名/机构：Krizhevsky、Sutskever、Hinton、Nair、Jarrett、Cire¸san 等一律原文。
会议：CVPR 等在参考文献中保留原文。

## 四、不确定/存疑的翻译点

1. **"mean activity"**：原文两处（第 2 节预处理、3.3 节）指像素值均值而非神经元激活值，故统一译「平均值」，未译「平均活跃度」。若读者对照其他译本可能见到「平均活跃度」译法。
2. **"kernel map" vs "feature map"**：2012 年原文用 "kernel map"，是今天 "feature map" 的前身。为忠实原文，译「卷积核映射（kernel map）」并在首次出现处标注。
3. **原文笔误 "on on GPU 2"**（6.1 节）：保留原文并加 `[译注：原文此处写作 "on on GPU 2"，系笔误]`。另脚注 2 原文 "fully-conneced" 系拼写笔误，参考文献部分照抄，正文按正确拼写恢复。
4. **公式恢复**：PDF 文本层提取的公式（LRN 归一化式、PCA 颜色增强式、SGD 更新式）存在上下标错乱，已按论文标准形式恢复为 LaTeX；如与 PDF 渲染有细微出入（如求和上下限的书写方式），以 PDF 原图为准。
5. **"infero-temporal pathway"**：神经科学中多写作 inferotemporal，译「下颞通路」，指视觉腹侧通路的一部分。译文未加译注，此处列出备忘。
6. **"columnar" CNN**：Cireșan 等人的多列网络，译「柱状」。国内亦有「多列」译法（指其 Multi-column DNN 名称），本文译法按原文 columnar 直译并附英文。
7. **"10 bits of constraint"**：信息论表述（1000 类约 10 比特），直译「10 比特的约束」，保留 bit 单位。
8. **章节标题 "Details of learning"**：译「学习细节」，与常见译名「训练细节」有别；原文用 learning 而非 training，故直译。附录于此处备忘。
9. **下载过程**：任务指定的 neurips.cc URL 经代理（127.0.0.1:7890）反复失败（HTTP 000/SSL 错误），最终不走代理直连 proceedings.neurips.cc 成功，得到 1,418,820 字节、9 页、%PDF-1.3 的完整 PDF。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（5 处）；图注内重复的页图链接去重（1 处）；成对重复的页图引用行去重（5 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
