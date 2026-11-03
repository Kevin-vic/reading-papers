# 翻译笔记：Neural Machine Translation by Jointly Learning to Align and Translate

- 论文：Bahdanau, Cho & Bengio, ICLR 2015, arXiv:1409.0473v7（15 页）
- 译文文件：`paper.md`（中英逐段对照）
- 源 PDF：`paper.pdf`（444,482 字节）

## 修复记录（2026-11-03）

针对块对完整性校验（88 Original / 87 中文）做了针对性修复，未整体重翻：

1. **补缺失中文块**：References（参考文献）块原本只有 `**Original:**` 无对应 `**中文:**`。按「原文保留不译」规范补上说明块：「（参考文献按规范原文保留，不译，共 29 条。）」。修复后 Original / 中文块对为 88 / 88，完全对齐。
2. **统一块标签**：附录 C 三个样例块的标签由 `**Original（样例 N）:**` 统一为规范的 `**Original:**`（样例编号信息保留在对应中文块「样例 N 说明」中），使自动化校验可按精确模式计数。
3. **修重复文本**：删除 6 处图/表卡片中重复的「见下方图，见下方图」，统一为单次「见下方图」（图 1、图 2、图 3、表 1、表 2、表 3 卡片各 1 处）。
4. **删重复图片引用**：每张页面原图（assets/pages/page-{3,5,6,7,14,15}.png）原先被连续引用两次，各删去一次重复，现每图仅保留一张。
5. **核对原文**：用 PyMuPDF 核对附录 C 原文为「RNNenc-50 and RNNsearch-50」（非 RNNencdec-50），paper.md 照录无误，未改动。

校验结果：`^\*\*Original:\*\*` 与 `^\*\*中文:\*\*` 各 88 处，无「见下方图，见下方图」残留，无重复图片行。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| neural machine translation | 神经机器翻译 | 首次出现于摘要，全文统一 |
| statistical machine translation | 统计机器翻译 | 首次出现于摘要 |
| encoder–decoder | 编码器–解码器 | 保留连字符形式 |
| fixed-length vector | 固定长度向量 | 本文核心概念 |
| context vector | 上下文向量 | |
| annotation | 注解 | 本文特指编码器为每个输入词生成的双向状态拼接向量 $h_j$，不是常规「标注（labelling）」义 |
| (soft-)search | （软）搜索 | 保留原文括号形式 |
| soft alignment / hard alignment | 软对齐 / 硬对齐 | |
| alignment model | 对齐模型 | 指打分函数 $a(s_{i-1}, h_j)$ |
| attention | 注意力 | 论文中「attention mechanism」译「注意力机制」 |
| recurrent neural network (RNN) | 循环神经网络（RNN） | |
| long short-term memory (LSTM) | 长短期记忆网络（LSTM） | 按 cs-glossary 术语表统一为「长短期记忆网络（LSTM）」 |
| bidirectional RNN (BiRNN) | 双向循环神经网络（BiRNN） | 首次出现于 3.2 节 |
| hidden state | 隐藏状态 | |
| hidden unit | 隐藏单元 | |
| gated hidden unit | 门控隐藏单元 | 附录 A.1.1 |
| update gate / reset gate | 更新门 / 重置门 | |
| element-wise multiplication | 逐元素乘法 | 符号 $\circ$ |
| 1-of-K coding | 1-of-K 编码（即独热编码） | 原文用 1-of-K 而非 one-hot；首次出现处注明等价关系 |
| word embedding | 词嵌入 | |
| embedding matrix | 嵌入矩阵 | |
| feedforward neural network | 前馈神经网络 | |
| multilayer perceptron | 多层感知机 | 附录 A.1.2 |
| maxout | maxout | 保留英文，方法名 |
| deep output | 深度输出 | Pascanu et al. (2014) 术语，保留括号标注 |
| cost function | 代价函数 | 术语表只有 loss/objective 条目；「代价函数」为该领域惯用译法 |
| log-probability | 对数概率 | |
| backpropagate | 反向传播 | 动词化译作「反向传播（通过它）」 |
| latent variable | 隐变量 | |
| conditional probability | 条件概率 | |
| joint probability | 联合概率 | |
| phrase-based system | 基于短语的系统 | 未入术语表，本文统一此译法 |
| phrase table | 短语表 | |
| beam search | 束搜索 | |
| minibatch stochastic gradient descent (SGD) | 小批量随机梯度下降（SGD） | 原文为「minibatch stochastic gradient descent」 |
| Adadelta | Adadelta | 优化方法名，保留英文 |
| BLEU score | BLEU 分数 | |
| training corpus | 训练语料 | |
| parallel corpus / monolingual corpus | 平行语料 / 单语语料 | |
| development (validation) set | 开发集（验证集） | |
| test set | 测试集 | |
| tokenization | 分词 | 术语表「分词/词元化」二选一，本文选「分词」并附英文 |
| shortlist | 高频词表 | 原文指 30,000 最常用词构成的词表 |
| [UNK] token | [UNK] token | token 保留英文（术语表：Token 保留） |
| lowercasing / stemming | 小写化 / 词干提取 | |
| state-of-the-art | 当前最优 | 按术语表统一 |
| baseline 未出现 | — | 本文用「basic encoder–decoder」，译「基本编码器–解码器」 |
| gated RNN 激活 | 门控单元（GRU 前身） | 原文未用 GRU 一词，仅称 gated hidden unit |
| monotonic / non-monotonic | 单调 / 非单调 | 对齐方向 |
| Gaussian distribution | 高斯分布 | |
| orthogonal matrices | 正交矩阵 | |
| learning rate | 学习率 | |
| L2-norm | L2 范数 | |
| epoch | epoch | 按术语表「轮次（epoch）」，表 2 译文中保留 epoch 并作解释 |
| NLL | NLL | 表 2 中保留，译「平均条件对数概率」 |
| handwriting synthesis | 手写合成 | |
| mixture of Gaussian kernels | 混合高斯核 | |
| neural probabilistic language model | 神经概率语言模型 | |
| re-rank / rescore | 重排序 / 重新打分 | |
| gold-standard translation | 金标准译文 | 附录 C |

## 保留英文（不译）清单

- 模型/方法名：RNNencdec（RNNenc-30/50）、RNNsearch（RNNsearch-30/50、RNNsearch-50⋆）、Moses、LSTM、BiRNN、maxout、softmax、Adadelta、SGD、logistic sigmoid 函数
- 人名：Bahdanau、Cho、Bengio 及全部引用作者
- 数据集/基准：WMT '14、Europarl、UN、news commentary、news-test-2012/2013/2014、BLEU
- 会议/机构：ICLR、ACL、EMNLP、NIPS、CIFAR、NSERC 等
- 法语例句（正文 5.2.2 与附录 C 中的模型输出、参考译文、Google Translate 输出）：一律原文照录不译，仅在中文段注明「法语译文样例，原文照录」

## 本次为批量简化模式

- 未生成 `source_map.json`（原文-译文逐行溯源映射）。
- 未提取 PDF 中的图片；图 1–3、表 1–3 以页码标注，见 paper.md 中块引用：图 1（第 3 页）、图 2（第 5 页）、图 3（第 6 页）、表 1（第 7 页）、表 2（第 14 页）、表 3（第 15 页）。
- PDF 文本由 PyMuPDF（fitz）提取（Read 工具需要 poppler，本机未装，已改用 Python 提取全部 15 页），公式按论文 LaTeX 原意重建（原样保留、数学不译）。
- 图 3 单词矩阵、图 1 节点标注等图内文字未单独成段，仅在对应位置以简短说明交代图中元素。

## 存疑/不确定的翻译点

1. **annotation**：本文核心术语，指 BiRNN 编码器为每个源词输出的向量。译作「注解（annotation）」。中文社区也常译「标注」或「注释」；「标注」易与 labelling 混淆，故取「注解」。首次出现于 3.1 节。
2. **shortlist**：译「高频词表」，指训练用 30,000 高频词名单；非术语表内词条。
3. **LSTM 首次出现的处理**：原文为「RNNs with long short-term memory (LSTM) units」，按术语表统一写作「长短期记忆网络（LSTM）单元」，严格说原文是「长短期记忆单元」；两者同指，台账已注明。
4. **cost function**：术语表无此条目（只有 loss/objective）。本文译「代价函数（cost function）」，该词在该学派（Bengio 组）论文中即指负对数似然损失。
5. **原文笔误保留**：附录 A.2.1 原文「We concatenate the forward and backward states to to obtain the annotations」中「to to」重复，译文按正确意思译出，未加译注。
6. **minibatch**：正文 4.2 原文写作「minibatch stochastic gradient descent (SGD)」；术语表条目为「mini-batch | 小批量」，正文统一用「小批量」。
7. **原文 5.2.2 中 [a medical center] 下划线**：原文强调 RNNencdec-50 自此开始偏离，方括号为原文样式，译文照录。
8. **作者单位**：Dzmitry Bahdanau 单位为 Jacobs University Bremen（不莱梅雅各布大学），KyungHyun Cho 与 Yoshua Bengio 单位为蒙特利尔大学；* 号脚注「CIFAR Senior Fellow」保留英文并附中文说明。
9. **arXiv 版本**：所下载 PDF 为 arXiv:1409.0473v7（2016-05-19，ICLR 2015 相机版），与 ICLR 官方版排版一致，页码以该 PDF 为准。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（0 处）；图注内重复的页图链接去重（0 处）；成对重复的页图引用行去重（6 对（其中 6 对在并行修复中已去除，本次补修 0 对并统一核验），修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
