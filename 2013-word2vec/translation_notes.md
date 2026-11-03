# 翻译笔记：Efficient Estimation of Word Representations in Vector Space（2013）

- 论文：arXiv:1301.3781v3 [cs.CL]（Mikolov et al., 2013，共 12 页）
- 下载：代理 `127.0.0.1:7890` 下 arxiv 直链 SSL 失败（exit 35），改为不挂代理直连 `https://arxiv.org/pdf/1301.3781` 成功，228716 字节。
- 阅读方式：本机无 poppler（Read 工具无法渲染 PDF 页），改用 PyMuPDF（fitz）以 `python -I` 提取全文纯文本至 `paper_text.txt` 后阅读。PDF 共 12 页，全部读完（pages_read = 12）。

## 批量简化模式说明

本次为批量简化模式：

- 未生成 `source_map.json`；段落对应关系以「**Original:** / **中文:**」块直接呈现于 `paper.md`。
- 未提取图片；图/表以页码标注（见下），图内文字（如 Figure 1 的 INPUT/PROJECTION/SUM 标注）未单独翻译。
- 表格以 Markdown 表格按原文数据重建（数据未改动，列头翻译并附英文）。
- 公式原样保留，display 用 `$$...$$`，行内用 `$...$`；编号 (1)–(5) 保留。

图/表页码对照：

| 图/表 | 页码（paper.pdf） |
|---|---|
| Figure 1 | 第 5 页 |
| Table 1 | 第 6 页 |
| Table 2 | 第 7 页 |
| Table 3 | 第 7 页 |
| Table 4 | 第 8 页 |
| Table 5 | 第 8 页 |
| Table 6 | 第 9 页 |
| Table 7 | 第 9 页 |
| Table 8 | 第 10 页 |

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| word vector | 词向量 | 全文统一 |
| distributed representation | 分布式表示 | |
| continuous vector representation | 连续向量表示 | |
| vector space | 向量空间 | |
| subspace | 子空间 | |
| neural network | 神经网络 | |
| feedforward neural network | 前馈神经网络 | |
| recurrent neural network (RNN) | 循环神经网络（RNN） | |
| NNLM | NNLM（神经网络语言模型） | 缩写保留，首次给全称 |
| RNNLM | RNNLM（循环神经网络语言模型） | 缩写保留 |
| CBOW | CBOW（连续词袋模型） | 模型名保留英文 |
| Skip-gram | Skip-gram | 模型名保留英文 |
| bag-of-words | 词袋 | |
| log-linear model / classifier | 对数线性模型 / 对数线性分类器 | |
| hierarchical softmax | 层级 softmax | Softmax 按术语表保留英文 |
| Huffman binary tree | 哈夫曼二叉树 | |
| vocabulary | 词表 | |
| corpus / corpora | 语料 / 语料库 | |
| token | 词元 | |
| word similarity task | 词相似度任务 | |
| semantic / syntactic | 语义 / 句法 | |
| syntactic regularity | 句法规律性 | |
| linear regularity | 线性规律性 | |
| accuracy | 准确率 | |
| dimensionality | 维度 | |
| computational complexity | 计算复杂度 | |
| training epoch | 训练轮次（epoch） | 首次出现附英文 |
| learning rate | 学习率 | |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| backpropagation | 反向传播 | |
| mini-batch | 小批量 | |
| gradient | 梯度 | |
| replica | 副本 | 分布式训练语境 |
| projection layer | 投影层 | |
| projection matrix | 投影矩阵 | |
| hidden layer | 隐藏层 | |
| output layer | 输出层 | |
| weight matrix | 权重矩阵 | |
| diagonal weight matrix | 对角权重矩阵 | |
| cosine distance | 余弦距离 | |
| state-of-the-art | 当前最优（state-of-the-art） | 首次出现附英文，之后用「当前最优」 |
| benchmark | 基准（benchmark） | |
| hyperparameter | 超参数 | |
| named entity | 命名实体 | |
| sentiment analysis | 情感分析 | |
| paraphrase detection | 复述检测 | |
| knowledge base | 知识库 | |
| machine translation | 机器翻译 | |
| information retrieval | 信息检索 | |
| question answering | 问答 | |
| language model | 语言模型 | |
| statistical language model | 统计语言模型 | |
| short term memory | 短期记忆 | |
| time-delayed connection | 带时间延迟的连接 | |
| recurrent matrix | 循环矩阵 | |
| 1-of-V coding | 1-of-V 编码 | 保留原文并括注「词表大小 V 上的独热式编码」 |
| Unigram perplexity | Unigram perplexity | 保留英文不译 |
| morphology | 形态学 | |
| synonym | 同义词 | |
| Spearman's rank correlation | 斯皮尔曼秩相关 | |
| Latent Semantic Analysis (LSA) | 潜在语义分析（LSA） | |
| Latent Dirichlet Allocation (LDA) | 隐狄利克雷分配（LDA） | |
| Latent Relational Analysis | 潜在关系分析 | |
| log-bilinear model | 对数双线性模型 | |
| word offset technique | 词偏移技术 | |
| inflectional language | 屈折语言 | |
| atomic unit | 原子单位 | |
| probability distribution | 概率分布 | |
| sampling | 采样 | |
| training criterion | 训练准则 | |
| CPU cores | CPU 核心 | |
| data center | 数据中心 | |
| adaptive learning rate | 自适应学习率 | |
| word morphology | 词形态学 | |
| sentence | 句子 | |
| context | 上下文 | |

保留原文不译（按术语表规则）：模型/方法名（NNLM、RNNLM、CBOW、Skip-gram、DistBelief、Adagrad、N-gram、LSA、LDA 缩写）、机构与产品名（Google News、LDC、Microsoft、MSR）、会议名（NIPS、ICML、ACL、ICASSP、NAACL HLT、Interspeech 等，仅出现在参考文献）、人名、脚注 URL、参考文献全部条目。

## 数字与单位约定（全文统一）

- 数据量级缩写（6B、320M、783M、1.6B、50M、30k、2.5K）保留原文，紧跟单位译出（如「6B 个词元」「30k 个词」）。
- 叙述性数字译中文：1.6 billion → 16 亿；trillions → 数万亿；a few billions → 几十亿；one trillion → 一万亿；1 million → 100 万；1.4 million → 140 多万；100 billion → 1000 亿。
- 百分比、维度数值、学习率（0.025）等保留原文。

## 不确定 / 存疑的翻译点

1. **1-of-V coding**：未译作「独热编码」（one-hot encoding 的说法在当时尚未普及），保留原文并加括注。
2. **Unigram perplexity(V)**：原文即为此写法，未展开解释（与语言模型困惑度的关系未在文中展开）。
3. **Table 3 表头**：原文为两层表头（「Semantic-Syntactic Word Relationship test set」横跨 Semantic/Syntactic 两列）。Markdown 不支持合并单元格，改为单层表头并在列名中加测试集名前缀，数据未变。
4. **原文疑似笔误，按规则加注保留**：「The result are reported in Table 6」（应为 results，原文如此，未改动）；参考文献 [14] 中「higly inflective languages」的拼写为原文如此（应为 highly inflective），按规则参考文献整体保留，未改。
5. **NIPS 2013**：会议名保留原文（当年尚未更名为 NeurIPS）。
6. **France - Paris + Italy = Rome 中的连字符**：原文用「-」，译文按原文保留（原文此处语义上是向量运算而非减法公式，未擅自改为 −）。
7. **PDF 连字（ligature）**：原文 PDF 中 fi/fl 连字在文本提取时出现为 U+FB01/U+FB02，已统一还原为常规字母（如 Efficient、significant、inflectional、midfielder）。
8. **脚注**：脚注 1、2、4 内容为 URL/代码地址，保留原文并附中文说明；脚注 3（致谢）给出译文并附原文。

## 修复记录

### 2026-11-03：补标题块对

- 问题：块对不完整——`paper.md` 中 57 个 `**Original:**` 块只有 56 个 `**中文:**` 块。开头标题块「**Original:** Efficient Estimation of Word Representations in Vector Space」缺少对应的中文块（中文标题仅在 H1 标题行内给出，未形成块对）。
- 修复：在标题 Original 块后补「**中文:** 向量空间中词表示的有效估计」（译名沿用 H1 标题行已定译法），与后续作者信息之间保留空行，符合全文块对格式。
- 校验：逐块配对检查通过，57 Original / 57 中文，无孤儿块、无错序。其余内容未改动（不属整体重翻情形）。

### 2026-11-03：图注与图片引用去重

- 图注尾部重复的「见下方图，见下方图」去重（6 处）；图注内重复的页图链接去重（3 处）；成对重复的页图引用行去重（6 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
