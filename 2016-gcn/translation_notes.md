# 翻译笔记：《Semi-Supervised Classification with Graph Convolutional Networks》(ICLR 2017)

- 源文件：C:/Users/13054/mclearning/papers/2016-gcn/paper.pdf（arXiv:1609.02907v4，873,897 字节，14 页）
- 译文：C:/Users/13054/mclearning/papers/2016-gcn/paper.md
- 翻译依据规范：C:/Users/13054/.claude/skills/nature-reader/references/cs-glossary.md（术语表 + 翻译硬规则八条）

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| Graph Convolutional Network (GCN) | GCN（首次出现展开为「图卷积网络（Graph Convolutional Network，GCN）」） | 模型名，正文统一用 GCN |
| graph convolution / spectral graph convolution | 图卷积 / 谱图卷积 | 一般术语；「spectral」首现标注（spectral graph convolution） |
| semi-supervised learning / classification | 半监督学习 / 半监督分类 | 术语表 |
| graph-structured data | 图结构数据 | |
| node / edge | 节点 / 边 | 术语表 |
| graph Laplacian | 图拉普拉斯矩阵 | 含 normalized（归一化）/ unnormalized（未归一化）变体 |
| adjacency matrix | 邻接矩阵 | |
| degree matrix | 度矩阵 | |
| self-connection | 自连接 | |
| identity matrix | 单位矩阵 | |
| propagation rule | 传播规则 | |
| layer-wise | 逐层 | |
| localized / K-localized | 局部化 / K-局部化 | 「局部化」首现标注（localized） |
| first-order approximation | 一阶近似 | |
| Chebyshev polynomial | 切比雪夫多项式 | 首现标注（Chebyshev polynomial） |
| eigenvalue / eigenvector / eigendecomposition | 特征值 / 特征向量 / 特征分解 | |
| feature vector / feature map | 特征向量 / 特征图 | 术语表 feature map → 特征图 |
| hidden layer / hidden units | 隐藏层 / 隐藏单元 | |
| cross-entropy | 交叉熵 | 术语表 |
| batch gradient descent | 批量梯度下降 | 术语表 batch → 批次；此处按通行译法 |
| mini-batch stochastic gradient descent | 小批量随机梯度下降 | 术语表 mini-batch → 小批量 |
| dropout | 随机失活（Dropout） | 首现标注，之后统一「随机失活」；术语表规定 |
| L2 regularization | L2 正则化 | 术语表 regularization → 正则化 |
| renormalization trick | 重归一化技巧 | 首现标注（renormalization trick），之后统一 |
| wall-clock time | 墙钟时间 | 首现标注（wall-clock time），之后统一 |
| epoch | 轮 | 首现「200 轮（epoch）」，之后统一「轮」；术语表 epoch → 轮次（epoch），本文从简用「轮」 |
| early stopping | 早停 | 首现标注（early stopping） |
| learning rate | 学习率 | 术语表 |
| hyperparameter | 超参数 | 术语表 |
| overfitting | 过拟合 | 术语表 |
| gradient descent | 梯度下降 | 术语表 |
| state-of-the-art | 当前最优（state-of-the-art） | 术语表 |
| label propagation (LP) | 标签传播（Label Propagation，LP） | 方法名，首现标注，之后用 LP |
| citation network | 引文网络 | |
| knowledge graph | 知识图谱 | |
| bipartite graph | 二分图 | |
| entity / relation | 实体 / 关系 | |
| transductive / inductive | 转导 / 归纳 | 首现标注（transductive） |
| skip-gram | skip-gram | 模型名保留英文 |
| random walk | 随机游走 | |
| breadth-first search | 广度优先搜索 | |
| one-hot | 独热 | 术语表 one-hot encoding → 独热编码 |
| bag-of-words | 词袋 | 首现标注（bag-of-words） |
| contraction map | 收缩映射 | |
| fixed point | 不动点 | |
| Weisfeiler-Lehman (WL-1) algorithm | Weisfeiler-Lehman（WL-1）算法 | 人名保留，算法名不译 |
| residual connection | 残差连接 | 术语表 |
| modularity-based clustering | 基于模块度的聚类 | |
| out-of-memory | 内存不足 | |
| featureless (approach) | 无特征（方法） | 首现标注（featureless） |
| bootstrap | 引导 | 5.3 节「用局部分类器引导类别标签」；首现标注（bootstrap） |
| multi-layer perceptron | 多层感知机 | 表 3 |
| hash function | 哈希函数 | |
| 1D convolutional neural network | 1 维卷积神经网络 | |
| full-batch gradient descent | （全）批量梯度下降 | 与 batch gradient descent 同义，统一为「批量梯度下降」 |

## 批量简化模式说明

本次翻译为批量简化模式，与完整 nature-reader 流程的差异：

1. **未做 source_map.json**：不生成段落级溯源映射文件，正文段落溯源依赖 paper.pdf 页码与章节结构。
2. **未提取图片**：所有图（图 1–5）与算法 1 仅以页码标注位置（见 paper.md 中各块引用「原图见 paper.pdf 第 N 页」），未导出 PNG 文件。图题表题均已翻译并标注页码（图 1→第 4 页；表 1→第 5 页；表 2、表 3→第 7 页；图 2→第 8 页；算法 1→第 11 页；图 3→第 12 页；图 4→第 13 页；图 5→第 14 页）。
3. **PDF 文本提取方式**：Read 工具渲染 PDF 需要 poppler（pdftoppm 未安装），故改用 Python pypdf 提取全文到 _text.txt 后阅读；公式在提取时被拆散为上下标碎片，已依据论文原文（arXiv v4 源码格式）手工重建为 LaTeX，编号 (1)–(14) 全部保留。
4. **参考文献**：整体保留原文不译，仅修复了 pypdf 对重音字符的提取误差（如 Mart´ın → Martín、R ´emi → Rémi、Sch ¨olkopf → Schölkopf 等）。

## 不确定 / 存疑的翻译点

1. **「weighing factor」**：pypdf 提取文本为 "λ is a weighing factor"。arXiv v4 正式版原文为 "weighting factor"（加权因子），「weighing」疑为连字/提取误差。译文按「加权因子」处理，未加译者注。
2. **「epoch」译法**：术语表规定「轮次（epoch）」，本文为行文简洁统一译为「轮」，首次出现处标注「200 轮（epoch，即训练迭代）」。
3. **「renormalization trick」**：通行译法有「重归一化技巧」「再归一化技巧」两种，本文取「重归一化技巧」，并在 6.2 节表 3 中统一。
4. **「wall-clock time」**：译「墙钟时间」为通行译法；术语表无此条，已入台账。
5. **「transductive / inductive」**：译「转导式 / 归纳式」；GNN 领域亦有「直推式」译法，取「转导式」与本文「谱卷积→传播规则」的技术语境一致。
6. **「ICA」**：全称 Iterative Classification Algorithm 译为「迭代分类算法（ICA）」；首现标注，之后统一 ICA。
7. **「Planetoid*」**：为 Yang et al. (2016) 各数据集最优变体的记号，保留原文（方法名）。
8. **「Kth-order neighborhood」**：译「K 阶邻域」，与「K-localized → K-局部化」统一。
9. **表 2 中「26s / 13s / 25s / 185s」**：括号内为墙钟训练秒数，保留原格式，仅去除 pypdf 提取产生的多余空格（如 59 .5 → 59.5）。
10. **附录 A 标题**：原文 "A RELATION TO WEISFEILER-LEHMAN ALGORITHM" 中 "Relation" 为单数，译文取「与 Weisfeiler-Lehman 算法的关系」。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（8 处）；图注内重复的页图链接去重（1 处）；成对重复的页图引用行去重（8 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
