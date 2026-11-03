# 翻译笔记：《Improving neural networks by preventing co-adaptation of feature detectors》

- 源文件：C:/Users/13054/mclearning/papers/2014-dropout/paper.pdf（arXiv:1207.0580v1，1,665,256 字节，共 18 页）
- 译文：C:/Users/13054/mclearning/papers/2014-dropout/paper.md
- 翻译日期：2026-11-03

## 一、版本说明（重要）

任务指定论文为《Dropout: A Simple Way to Prevent Neural Networks from Overfitting》（2014，JMLR 版本），但下载 URL https://arxiv.org/pdf/1207.0580 返回的是 **v1 版本（2012-07-03 投稿版）**，实际标题为《Improving neural networks by preventing co-adaptation of feature detectors》，是同一研究的 Science 风格短文（正文 6 页 + 附录 A–H 共 18 页），不含 2014 JMLR 版新增的 Motivation / Model Description / Practical Guide 等章节。

处理方式：翻译以实际下载的 PDF 全文为准，标题沿用 v1 原文标题，并在 paper.md 开头注明版本差异；如需 JMLR 完整版，可另行下载 arXiv:1207.0580v2 及以后版本。

## 二、本次为批量简化模式

- 未生成 source_map.json，未提取图片；图 1–7 仅以页码标注（页码为 caption 在 PDF 中的页号）。
- PDF 页渲染：本机未安装 poppler，Read 工具无法直接渲染 PDF 页面；改用 PyMuPDF（fitz）提取全 18 页文本后通读翻译。公式按提取文本重建为 LaTeX（display 用 `$$...$$`，行内用 `$...$`），未对照原始排版逐字核对符号位置，个别上下标如有出入以原 PDF 为准。
- 图/表页码对照：

| 图 | 内容 | 所在页（paper.pdf） |
|---|---|---|
| 图 1 | MNIST 测试集错误率曲线 | 第 3 页 |
| 图 2 | TIMIT 核心测试集帧分类错误率 | 第 4 页 |
| 图 3 | CIFAR-10 "鸟"类十样本 | 第 5 页 |
| 图 4 | ImageNet 测试样本及 top-5 概率 | 第 5 页 |
| 图 5 | MNIST 第一层特征可视化 (a) backprop (b) dropout | 第 10 页 |
| 图 6 | TIMIT 训练/验证错误与交叉熵曲线 | 第 11 页 |
| 图 7 | Reuters 训练/验证分类错误率曲线 | 第 13 页 |

## 三、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| dropout | 随机失活（Dropout） | 首次出现标注；之后正文统一用"随机失活"；标题、图中保留 Dropout。业界通行保留英文，此处遵照术语表规定 |
| overfitting | 过拟合 | 首次出现标注（overfitting） |
| feedforward neural network | 前馈神经网络 | 首次出现标注（feedforward neural network） |
| feature detector | 特征检测器 | |
| co-adaptation | 协同适应 | |
| hidden unit | 隐藏单元 | |
| visible unit | 可见单元 | 指输入层单元（RBM 语境） |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| mini-batch | 小批量 | |
| mean network | 平均网络（mean network） | 文中自带英文 |
| geometric mean | 几何平均 | |
| regression | 回归 | |
| learning rate | 学习率 | |
| momentum | 动量 | |
| weight decay | 权重衰减 | |
| epoch | 轮次 | 附录首次出现时标注（epoch） |
| cross-entropy | 交叉熵 | |
| hidden Markov model (HMM) | 隐马尔可夫模型（HMM） | |
| Gaussian mixture model | 高斯混合模型 | |
| Viterbi algorithm | Viterbi 算法 | 保留人名 |
| deep belief net / Deep Belief Network (DBN) | 深度信念网络（Deep Belief Network，DBN） | 术语表未列，按首次标注处理 |
| deep Boltzmann machine (DBM) | 深度玻尔兹曼机（Deep Boltzmann Machine，DBM） | 同上 |
| contrastive divergence | 对比散度（Contrastive Divergence） | |
| Gaussian RBM / binary RBM | 高斯受限玻尔兹曼机 / 二元 RBM | RBM 首次展开为"受限玻尔兹曼机" |
| mean-field | 平均场 | |
| convolutional neural network (CNN) | 卷积神经网络（CNN） | |
| max-pooling / average-pooling | 最大池化 / 平均池化 | |
| locally-connected layer | 局部连接层 | |
| globally-connected layer | 全局连接层 | |
| filter bank | 滤波器组 | |
| receptive field | 感受野 | |
| stride | 步长 | |
| pooling | 池化 | |
| overlapping pooling | 重叠池化 | |
| response normalization | 响应归一化 | |
| lateral inhibition | 侧向抑制 | |
| max-with-zero nonlinearity | 与零取最大（max-with-zero）非线性 | 即后来的 ReLU，但原文未用 ReLU 之名，译文中加注 |
| contrast-normalization | 对比度归一化 | |
| multinomial logistic regression | 多项逻辑斯谛回归 | |
| softmax | softmax | 保留英文 |
| Bayesian model averaging | 贝叶斯模型平均 | |
| posterior probability / posterior distribution | 后验概率 / 后验分布 | |
| Markov chain Monte Carlo | 马尔可夫链蒙特卡洛 | |
| bagging | bagging | 保留英文（方法名） |
| naive Bayes | 朴素贝叶斯（naive Bayes） | 原文小写 "naive bayes" |
| logistic classification | 逻辑斯谛分类 | |
| mixture of experts | 专家混合 | |
| mixability | 可混合性（mixability） | Livnat et al. 2008 进化理论术语 |
| generative pre-training | 生成式预训练 | |
| fine-tuning | 微调 | |
| backpropagation | 反向传播 | |
| validation set | 验证集 | |
| early stopping | 提前停止 | 附录 B.2 首次出现时标注 |
| data augmentation | 数据增强 | |
| state-of-the-art | 最高水平（state-of-the-art） | 附录 H 首次出现时标注 |
| hyperparameter | 超参数 | |
| objective function | 目标函数 | |
| regime | 工作区段（regime） | 正文第 13 段语境：输入-输出映射的不同"工作区段" |
| filter bank response | 滤波器组响应 | 语音语境，log filter bank 译为"对数滤波器组" |
| complex cell / simple cell | 复杂细胞 / 简单细胞 | 视觉皮层神经科学术语 |
| stationary statistics | 统计特性平稳 | |
| translation invariance | 平移不变性 | |
| regime | 工作区段（regime） | |

数据集/工具/人名/会议均保留英文：MNIST、TIMIT、CIFAR-10、Tiny Images、ImageNet、Reuters (RCV1-v2)、Kaldi、cuda-convnet、NVIDIA GTX 580、Hinton 等。

## 四、不确定 / 存疑的翻译点

1. **版本错位**：URL 返回 v1（2012）而非任务描述的 2014 JMLR 版标题。已按实际 PDF 翻译并在文中注明。
2. **"at URL" 占位**（正文第 6 段）：原文 "When the publically available code at URL was used to pre-train a deep Boltzmann machine" 中 "at URL" 是投稿时的链接占位，实际指向附录 A.2 脚注 1 的代码链接。已加译注。
3. **附录 B.2 句子不完整**：原文 "keeping all other hyperparameters" 后直接接下一段（Figure 6...），疑缺 "the same"。已加译注，未擅自补全。
4. **附录 A.1 "Constraining input vectors to have a fixed length"**：此处 "input vectors" 与前文 "incoming weight vector" 应指同一物（每个隐藏单元的输入权重向量）。译文按上下文译为"（输入权重）向量"并加译注。
5. **dropout 译名**：术语表规定译"随机失活（Dropout）"，但该术语作为方法名时业界几乎一律保留英文。本译文严格遵照术语表（首次标注后正文统一用"随机失活"），标题与图表名保留 Dropout。若读者偏好，可将正文"随机失活"全局替换回 Dropout。
6. **"regime"**（正文第 13 段）：语境为输入-输出映射的若干不同"区段/模态"，译为"工作区段"并保留原文括注，此译法存疑，可考虑"模态"或"分区"。
7. **公式重建**：PDF 为双栏排版，公式（附录 A.1 的更新规则、F.2 的响应归一化、F.3 的 max-with-zero、F.6 的动量更新）依据提取文本重建为 LaTeX；其中附录 A.1 的 ⟨∇wL⟩ 在原文中为小批量平均梯度符号，重建为 $\langle \nabla_w L \rangle$。个别符号（如分式横线、cases 结构）为排版重建，语义与原文一致。
8. **脚注处理**：原文脚注（附录 A.2 两个代码链接、附录 B 的 Kaldi 链接、附录 C 的语料库链接、附录 D 的 CIFAR 链接）以 [脚注：…] 形式并入对应段落后，URL 保留原文。
9. **数字格式**：按规则 7 保留原文格式（如 60,000、47.2%、10ms），未做"万"化改写；同一数量在正文（"50,000"）与附录（"60000"）写法不同，均照原文保留。
10. **图 4 无正文显式引用**：正文未出现 "figure 4" 字样，按惯例将图 4 块引用放在 ImageNet 段落（第 10 段）之后。

## 五、后续修复记录

- 2026-11-03：**块标记格式统一**。paper.md 中 49 处中文块标记原先使用全角冒号 `**中文：**`，与其余 24 篇论文统一使用的半角 `**中文:**` 不一致，导致按 `**中文:**` 模式做自动化块对校验时该篇中文块被误报为 0。已将全部 49 处 `**中文：**` 替换为 `**中文:**`，与 49 处 `**Original:**` 一一对齐；正文中中文标点的全角冒号（如译注里"Dropout：防止"）未改动。翻译内容与术语台账无其他变更。

- 2026-11-03：**图注与图片引用去重**。图注尾部重复的「见下方图，见下方图」去重（6 处）；图注内重复的页图链接去重（1 处）；成对重复的页图引用行去重（6 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
