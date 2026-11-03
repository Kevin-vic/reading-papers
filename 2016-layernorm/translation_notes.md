# 翻译笔记：Layer Normalization（arXiv:1607.06450）

## 一、本次为批量简化模式

- 未生成 source_map.json（原文-译文逐段锚定映射），未提取图片；图/表以「（原图见 paper.pdf 第 N 页）」标注页码。
- 原文（Original）段落依据 arXiv 提交的 LaTeX 源码（`src/ln_paper_arxiv.tex`）与 PDF 文本逐段核对；公式从源码原样重建为 Markdown LaTeX（display 用 `$$...$$`，行内用 `$...$`），公式编号与原文一致：正文 (1)–(14)，补充材料 (15)–(35)。
- PDF 共 14 页，已全部读完。
- 原文中由 `\cut{}` 隐藏的段落（如「Regularization through sample statistics」一节及附录权重缩放的推导）不在正式 PDF 中，未翻译。
- 参考文献整体保留原文（取自 `src/ln_paper_arxiv.bbl`）。

## 二、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| layer normalization | 层归一化 | 论文主题，首次出现标注英文 |
| batch normalization | 批归一化 | 缩写 BN |
| weight normalization | 权重归一化 | 术语表无此条，按 weight=权重 统一 |
| recurrent batch normalization | 循环批归一化 | 后文简称循环 BN |
| summed input | 加权和输入 | 即 $w^\top h$ / $W x$ 的神经元输入之和 |
| covariate shift | 协变量偏移 | 首次出现标注英文，原文带引号 |
| mini-batch | 小批量 | 与"批次大小（batch-size）"按上下文区分 |
| gain parameter | 增益参数 | 与 bias（偏置）对举 |
| normalization statistics | 归一化统计量 | |
| hidden unit | 隐藏单元 | |
| hidden state | 隐藏状态 | |
| hidden-to-hidden dynamics | 隐层到隐层动态 | |
| re-scaling | 缩放 | 与 re-centering 对举 |
| re-centering | 平移 | 字面为"重新定中心"，正文统一用"平移"，首次标注英文 |
| invariance | 不变性 | |
| Riemannian manifold / metric | 黎曼流形 / 黎曼度量 | |
| Fisher information matrix | Fisher 信息矩阵 | Fisher 保留英文 |
| generalized linear model (GLM) | 广义线性模型（GLM） | |
| exponential family | 指数族 | |
| transfer function | 传递函数 | |
| log partition function | 对数配分函数 | |
| Kronecker product / vector operator | Kronecker 积 / Kronecker 向量化算子 | |
| early stopping | 提前停止 | 原文带引号，指隐式正则效果 |
| order-embeddings | 序嵌入 | 方法名，首次标注英文，后统一"序嵌入" |
| attentive reader | 注意力阅读器 | 首次标注英文 |
| sequence to sequence | 序列到序列 | |
| online regime | 在线模式 | |
| permutation invariant | 置换不变 | |
| one-hot vector | 独热向量 | 首次标注英文 |
| bivariate Gaussian mixture | 双变量高斯混合 | |
| variational bound | 变分下界 | |
| negative log likelihood | 负对数似然 | |
| epoch | 轮次 | 术语表统一译名 |
| fine-tuning | 微调 | |
| unbiased estimator | 无偏估计量 | |
| receptive field | 感受野 | |
| glimpse | 瞥视 | DRAW 模型的注意力机制术语，首次标注英文 |
| LSTM / GRU / ReLU / Adam / SGD | 长短期记忆网络（LSTM）/ 门控循环单元（GRU）/ ReLU / Adam / SGD | 模型与算法名按术语表处理；LSTM、GRU 首次展开后只用缩写 |
| Softmax / Logit / Recall@K / R@K / nats | Softmax / Logit / Recall@K / R@K / nats | 保留英文 |
| Skip-thoughts / Skip-gram / DRAW / VGG / Theano / CNMeM / MSCOCO / BookCorpus / IAM-OnDB | 保留英文 | 模型名、框架名、数据集名不译 |
| CNN corpus | CNN 语料库 | |
| 10-crop | 10-crop | 数据增强方式，保留英文 |
| IAM Online Handwriting Database | IAM 在线手写数据库 | 首次标注英文全称 |
| state-of-the-art | 当前最优 | 术语表统一译名 |

## 三、存疑 / 不确定点

1. **"summed inputs" 的译名**：直译为"输入之和"，本译文统一为「加权和输入」并在首次出现标注英文；另有译法「求和输入」「总和输入」，未采纳。
2. **"re-centering" 的译名**：直译「重新定中心」，译文采用更简洁的「平移」，首次出现处标注英文（re-centering）；指对权重矩阵或数据集整体加常数向量。
3. **原文笔误（照抄并加译注）**：
   - 第 3 节正文 "layer normaliztion" → 应为 "normalization"；
   - 3.1 节 "we need to to compute" → "to" 重复；
   - 式 (32) 中 $\bar{F}(\text{vec}([W, b, g]^\top)$ 缺右括号（源码即如此）；
   - 式 (35) 第一个 vec 中第二项分母为 $\|w_i\|_2$，按上下文应为 $\|w_j\|_2$。
4. **"order-embeddings" 是否该译**：它既是模型名又具描述性，术语表规定模型名保留英文；此处折中为首现「序嵌入（order-embeddings）」后统一用中文，图题中保留了英文对照。
5. **表 2 中 Sym 与 OE 两行的 R@5 为空**：原论文表格即如此（该列数据在原论文中未报告），照原样保留，未补填。
6. **Figure 4 与 Figure 5 同在第 9 页**：PDF 中 Figure 5（手写序列生成）位于第 9 页顶部、Figure 4（DRAW）位于同页中部（LaTeX 浮动体所致），页码标注按 PDF 实际位置。
7. **式 (35) 左侧与右侧形式**：原文第一处 vec 与第二处 vec 内容不完全相同（第一处 $\delta_{gj} w_j/\|w_i\|_2$，第二处 $\delta_{gj} w_j/\|w_j\|_2$），疑为笔误（见 3），译文照抄原文。

## 四、产物清单

| 文件 | 说明 |
|---|---|
| `paper.pdf` | 下载的论文 PDF（arXiv:1607.06450v1，612446 字节） |
| `paper.md` | 全文双语对照翻译（本目录主产物） |
| `translation_notes.md` | 本笔记 |
| `paper.txt` | PyMuPDF 提取的 PDF 纯文本（工作中间产物） |
| `src/` | arXiv 提交的 LaTeX 源码（`ln_paper_arxiv.tex`、`macros.tex`、`ln_paper_arxiv.bbl` 等），用于公式精确重建与参考文献还原 |

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（6 处）；图注内重复的页图链接去重（3 处）；成对重复的页图引用行去重（6 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
