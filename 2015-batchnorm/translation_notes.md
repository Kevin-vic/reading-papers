# 翻译笔记：Batch Normalization（arXiv:1502.03167v3）

## 本次为批量简化模式

- 未生成 source_map.json，未提取图片；图以页码标注（图 1→第 6 页，图 2/3→第 7 页，图 4→第 8 页，图 5→第 11 页），图题均已翻译。
- 公式未用 OCR/截图对照，而是依据论文官方 LaTeX 源码（已下载至 `src/arx.tex`，来自 `https://export.arxiv.org/e-print/1502.03167`）精确重建，保证数学内容原样准确。
- PDF 全文共 11 页，已全部阅读（先 PyMuPDF 提取文本，再对照 LaTeX 源码逐段核对）。
- 下载降级说明：`https://arxiv.org/pdf/1502.03167` 经代理（127.0.0.1:7890）CONNECT 隧道建立成功但 TLS 握手失败（curl schannel 与 Python ssl 均为 EOF），改用镜像 `https://export.arxiv.org/pdf/1502.03167` 成功（173,548 字节，%PDF- 校验通过）。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| Batch Normalization | 批归一化（Batch Normalization，BN） | 术语表钉名 |
| internal covariate shift | 内部协变量偏移（Internal Covariate Shift） | 本文核心概念，首见于摘要 |
| covariate shift | 协变量偏移（covariate shift） | 概念先行于本文，注意区分 |
| domain adaptation | 领域自适应（domain adaptation） | |
| deep learning | 深度学习（deep learning） | |
| deep neural network | 深度神经网络（Deep Neural Network） | |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| mini-batch | 小批量（mini-batch） | |
| batch gradient descent | 批量梯度下降（batch gradient descent） | 与 mini-batch SGD 相对 |
| learning rate | 学习率 | |
| momentum | 动量（momentum） | |
| gradient | 梯度 | |
| vanishing / exploding gradient | 梯度消失 / 梯度爆炸 | |
| backpropagation | 反向传播 | |
| saturating nonlinearity | 饱和非线性（saturating nonlinearities） | |
| saturated regime | 饱和区（saturated regime） | |
| whitening | 白化（whitening） | |
| normalization | 归一化 | |
| mean / variance | 均值 / 方差 | |
| first and second moments | 一阶矩和二阶矩 | 与均值/方差呼应 |
| population statistics | 总体统计量（population statistics） | 与 mini-batch statistics 相对 |
| unbiased variance estimate | 无偏方差估计 | |
| moving average | 移动平均 | |
| activation | 激活值（activation） | |
| weight / bias | 权重 / 偏置 | |
| weight matrix / bias vector | 权重矩阵 / 偏置向量 | |
| affine transformation | 仿射变换 | |
| identity transform | 恒等变换 | |
| differentiable | 可微 | |
| Jacobian | 雅可比矩阵（Jacobian） | |
| singular value | 奇异值 | |
| regularization / regularizer | 正则化 / 正则化项 | |
| overfitting | 过拟合 | |
| Dropout | 随机失活（Dropout） | 术语表钉名 |
| Rectified Linear Unit | 修正线性单元（Rectified Linear Unit，ReLU） | 之后统一用 ReLU |
| fully-connected layer | 全连接层 | |
| convolutional layer | 卷积层 | |
| convolutional property | 卷积性质 | |
| feature map | 特征图 | |
| filter | 滤波器 | 术语表：kernel/filter 按上下文统一 |
| pooling layer | 池化层 | |
| softmax | Softmax | 保留 |
| cross-entropy loss | 交叉熵损失 | |
| inference | 推理 | |
| ensemble | 集成 | |
| state-of-the-art | 当前最优（state-of-the-art） | 术语表钉名（SOTA） |
| human rater | 人工标注者（human rater） | ImageNet 语境 |
| single crop / multicrop | 单次裁剪 / 多裁剪 | |
| top-1 / top-5 error | top-1 / top-5 错误率 | 保留 top-1/top-5 |
| machine infinity | 机器无穷（machine infinity） | 数值溢出为无穷大 |
| Local Response Normalization | 局部响应归一化（Local Response Normalization） | 自定译名，不在预置术语表 |
| within-shard shuffling | 分片内打乱（within-shard shuffling） | 自定译名，不在预置术语表 |
| photometric distortion | 光度畸变（photometric distortion） | 自定译名，不在预置术语表 |
| standardization layer | 标准化层（standardization layer） | 自定译名（Gülçehre & Bengio, 2013） |
| separable convolution | 可分离卷积（separable convolution） | 附录 |
| depth multiplier | 深度乘数（depth multiplier） | 附录 |
| percentile | 百分位数 | 图 1 题注 |
| stride | 步长 | 附录 |
| filter concatenation | 滤波器拼接 | 附录 |
| generalization | 泛化 | |
| convergence | 收敛 | |
| divergence | 发散 | |
| local minima | 局部极小值 | |

## 数字与单位格式

- 全文统一使用「万」计数：$13.6\cdot 10^6$ → 1360 万；$31\cdot 10^6$ → 3100 万；$6\cdot 10^6$ → 600 万；$2.1\cdot 10^6$ → 210 万；$13.3\cdot 10^6$ → 1330 万；$2.7\cdot 10^6$ → 270 万。
- 倍数统一：14 times fewer → 少 14 倍；5 times fewer → 少 5 倍；6 times faster → 6 倍的速度。
- 百分比、步数（50000 步、60 个样本）保持原文数字形式。

## 不确定 / 存疑的翻译点

1. **原文笔误（未改，加译注）**：第 2 节末 "We want to a preserve the information" 多了一个 "a"；4.2.2 节 sigmoid 定义 $g(t)=\frac{1}{1+\exp(-x)}$ 自变量 $t$ 与表达式 $x$ 不一致；图 4 表格中 Deep Image high-res 的 Top-1 error 为 "24.88" 缺百分号；附录 "GoogleNet archictecture" 拼写错误。以上均为 arXiv v3 官方版本原文如此，已用 `[译注：…]` 标注，未擅自改动。
2. **3.4 节语法**："the training network no longer producing deterministic values" 缺谓语（v3 版如此），已加译注，译文按通顺理解处理。
3. **"training rates"（结论）**：原文 "more tolerant to increased training rates"，结合 3.3 节语境译为「提高的训练速率（学习率）」，并括注学习率以避免歧义。这是本文少见的用词，存疑但影响小。
4. **"human raters"**：译为「人工标注者」。ImageNet 文献中常见译法有「人工标注者」「人类评估者」，此处取「人工标注者」并保留英文。
5. **"across the board pooling layers"（附录）**：译为「全局的池化层」，即横跨整个网络的池化层，与 GoogLeNet 原论文表述对照后确认。
6. **摘要 "14 times fewer training steps"**：译为「只用 1/14 的训练步数」，正文 4.2.2 同义句译为「所需步数少 14 倍」，语义一致，表述略有差异（一处强调比值、一处强调倍数），可接受。
7. **公式排版**：$\mathbf{x}$、$\mathbf{u}$、$\mathbf{z}$、$\mathbf{b}$ 等正体加粗对应原文 LaTeX 宏 `\vx=\mathrm{x}` 等（arXiv 源文件中所有向量均用正体斜体混合样式），普通标量 $x, u, b$ 保持斜体，忠实还原。

## 图/表页码对照

| 图/表 | 位置（paper.pdf 页码） |
|---|---|
| 图 1（MNIST 实验 + 分布演变） | 第 6 页 |
| 图 2（验证准确率曲线） | 第 7 页 |
| 图 3（步数与准确率表） | 第 7 页 |
| 图 4（与 SOTA 对比表） | 第 8 页 |
| 图 5（Inception 架构表） | 第 11 页 |

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（4 处）；图注内重复的页图链接去重（1 处）；成对重复的页图引用行去重（4 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
