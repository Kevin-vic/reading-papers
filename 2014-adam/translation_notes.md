# 翻译笔记：《Adam: A Method for Stochastic Optimization》

- 来源：arXiv:1412.6980v9 [cs.LG]（ICLR 2015 会议版，共 15 页：正文 11 页 + 附录 4 页）
- 下载：https://arxiv.org/pdf/1412.6980，经本地代理 127.0.0.1:7890 下载，584,641 字节，验证通过（`%PDF-` 开头）
- 译文文件：paper.md（同目录）
- 翻译规范依据：`C:/Users/13054/.claude/skills/nature-reader/references/cs-glossary.md`（术语表 + 翻译硬规则八条）

## 一、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| stochastic optimization | 随机优化 | |
| objective function | 目标函数 | |
| gradient descent | 梯度下降 | |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | 首次出现标注英文全称 |
| first-order | 一阶 | |
| moment | 矩 | first/second moment：一阶/二阶矩 |
| raw moment | 原点矩 | |
| uncentered variance | 非中心方差 | |
| exponential moving average | 指数移动平均 | |
| decay rate | 衰减率 | |
| bias / bias correction | 偏差 / 偏差修正 | |
| stepsize | 步长 | 与 stride（步幅）区分 |
| stride | 步幅 | 术语表预置为「步长」，此处为避免与 stepsize 冲突改用「步幅」，首次出现标注英文 |
| learning rate | 学习率 | |
| hyperparameter | 超参数 | 原文拼写 hyper-parameter，译文统一「超参数」 |
| minibatch | 小批量 | |
| sparse gradient | 稀疏梯度 | |
| non-stationary | 非平稳 | |
| stationary | 平稳 | |
| convex / non-convex | 凸 / 非凸 | |
| regret / regret bound | 遗憾 / 遗憾界 | |
| online convex optimization | 在线凸优化 | |
| feasible set | 可行集 | |
| signal-to-noise ratio (SNR) | 信噪比（SNR） | 原文为「借用术语」 |
| trust region | 信任区域 | |
| annealing | 退火 | step size annealing：步长退火；automatic annealing：自动退火 |
| dropout | 随机失活 | 图例/算法标识中的 dropout 保留英文 |
| overfitting | 过拟合 | |
| logistic regression | 逻辑回归 | |
| neural network | 神经网络 | |
| convolutional neural network (CNN) | 卷积神经网络（CNN） | |
| hidden layer / hidden unit | 隐藏层 / 隐单元 | |
| pooling / max pooling | 池化 / 最大池化 | |
| convolution | 卷积 | |
| whitening | 白化 | |
| cross-entropy | 交叉熵 | |
| weight decay | 权重衰减 | |
| quasi-Newton | 拟牛顿 | |
| preconditioner | 预条件子 | |
| Fisher information matrix | Fisher 信息矩阵 | |
| natural gradient descent (NGD) | 自然梯度下降（NGD） | |
| momentum / Nesterov momentum | 动量 / Nesterov 动量 | |
| bag-of-words (BoW) | 词袋（BoW） | |
| variational auto-encoder (VAE) | 变分自编码器（VAE） | |
| latent variable | 隐变量 | |
| epoch | 轮次 | |
| convergence | 收敛 | |
| infinity norm | 无穷范数 | AdaMax 的理论基础；$L^p$ 范数写作 $L^p$ norm |
| temporal averaging | 时间平均 | |
| Polyak-Ruppert averaging | Polyak-Ruppert 平均 | 人名不译 |
| wall-clock time | 墙上时钟时间 | |
| element-wise | 逐元素 | |
| diagonal rescaling | 对角缩放 | |
| hyperplane | 超平面 | |
| tangent | 切点/切线 | |
| induction | 归纳 | |
| arithmetic-geometric series | 算术-几何级数 | |
| Young's inequality | Young 不等式 | |
| adaptive moment estimation | 自适应矩估计 | Adam 名称来源 |

保留英文（不译）：Adam、AdaGrad、RMSProp、AdaMax、AdaDelta、vSGD、SFO、MNIST、IMDB、CIFAR-10、ReLU、softplus、ICLR、ICML、ICASSP、GPU、OpenAI、Google Deepmind、SURF、所有人名。

## 二、批量简化模式说明

本次为批量简化模式，与 nature-reader 完整流程的差异：

1. **未生成 source_map.json**：未做逐句源映射文件。
2. **未提取图片**：环境缺少 poppler（`pdftoppm` 不可用），PDF 无法按页渲染为图像，Read 工具读 PDF 失败。改用 pymupdf（Python 脚本 `C:/Users/13054/mclearning/scripts/extract_pdf_text.py`）逐页提取全部文本（15 页、42,210 字符），原文阅读基于该提取文本完成。
3. **图以页码标注**：图 1（第 6 页）、图 2（第 7 页）、图 3（第 7 页）、图 4（第 8 页）、算法 1（第 2 页）、算法 2（第 9 页），均在文中以块引用标注「（原图见 paper.pdf 第 N 页）」，未附图片本体。
4. **公式复原**：pymupdf 提取的文本将公式拆散（上下标分离），全部 display 公式已按 arXiv 原文恢复为 LaTeX（`$$...$$`，编号式 (1)–(12)），行内公式用 `$...$`。附录推导链式公式（Lemma 10.4、Theorem 10.5 证明中的连续不等式）已按推导顺序复原。

## 三、不确定/存疑的翻译点

1. **引理 10.4 陈述与证明不一致**：v9 版陈述给出的界为 $\frac{2}{1-\gamma}\frac{1}{\sqrt{1-\beta_2}}\|g_{1:T,i}\|_2$（无 $G_\infty$、无平方），而证明末尾得到的界是 $\frac{2G_\infty}{(1-\gamma)^2\sqrt{1-\beta_2}}\|g_{1:T,i}\|_2$。译文如实保留两处，并加译注说明 ICLR 2015 正式版已将陈述修正为后者。
2. **定理 10.5 证明末尾的界**：最后一行的第三项分母为 $2\alpha\beta_1(1-\lambda)^2$，与定理 4.1/10.5 陈述中的 $2\alpha(1-\beta_1)(1-\lambda)^2$ 不一致，疑为 v9 版笔误。译文忠实保留原式并加译注。
3. **stride 译名**：术语表预置 stride → 步长，但本文 stepsize 也译「步长」，为区分将 stride 译作「步幅」并在台账注明，属对预置表的局部调整。
4. **theoratical / theoratically**：6.1 节原文两处拼写（应为 theoretical/theoretically），属明显拼写笔误，译文按正确含义翻译，未逐处加译注。
5. **theoretical 原文拼写 "theoratical"**：同上。
6. **AdaGrad 对应式中的下标**：原文写作 $g_t/\sqrt{\sum_{i=1}^{t} g_t^2}$（根号内下标 $t$ 与求和下标 $i$ 混用），忠实保留原文符号。
7. **$f_1(\theta), ..., , f_T(\theta)$**：原文双逗号，忠实保留。
8. **Polyak-Ruppert 平均式**：PDF 文本提取显示 $\sum_{k=1}^{n}$，据 Polyak-Ruppert 平均的标准定义与上下文（除以 $1/t$）恢复为 $\sum_{k=1}^{t}$。
9. **「遗憾（regret）」译名**：在线学习文献中文圈也常作「悔恨/后悔」，本文统一用「遗憾」并在首次出现标注英文。

## 四、修复记录（2026-11-03）

1. **块对缺失修复**：算法 1 与算法 2（AdaMax）两个伪代码 Original 块原本只有 **Original:** 无配对 **中文:**（全文件 53 个 Original / 51 个中文）。已在两块的伪代码代码块结束后补上 **中文:** 块：翻译算法描述文字，伪代码本身按公式/代码原样保留的规范不重复翻译，并在块内注明逐行中文注释见上方的「算法 1（译）」/「算法 2（译）」引用块。修复后块计数 53 / 53，无未配对块、无空中文块。
2. **格式规整**：删除「算法 1（译）」「算法 2（译）」及「图 1（译）」「图 2（译）」「图 3（译）」「图 4（译）」各引用块尾部重复的「见下方图」短语/重复图片链接，并删除重复的图片引用行（page-2.png、page-6.png、page-7.png、page-8.png、page-9.png 原各出现两次，现各保留一张）。
3. **文件末尾检查**：定理 10.5 证明最后一块（遗憾界结论）的 **中文:** 块经核查已有完整正文（翻译 + 公式 + 译注），结构规整，无需改动。
