# 翻译笔记：Generative Adversarial Nets（arXiv:1406.2661v1，2014）

## 一、模式说明

本次为**批量简化模式**：未做 source_map.json 与图片提取，图/表以页码标注（原图见 `paper.pdf` 第 N 页）。原文以 PyMuPDF 提取的 PDF 文本与 arXiv LaTeX 源码（`src/extracted/adversarial.tex`）交叉核对，公式按 LaTeX 源精确转录。

## 二、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| Generative Adversarial Nets | 生成对抗网络 | 论文标题译名；正文中 "adversarial nets" 统一译「对抗网络」，与通行中文名「生成对抗网络（GAN）」对应（原文未出现 "GAN" 一词） |
| adversarial process | 对抗过程 | |
| adversarial nets | 对抗网络 | 全文统一 |
| generative model | 生成模型 | |
| discriminative model | 判别模型 | |
| generator | 生成器 | |
| discriminator | 判别器 | |
| multilayer perceptron | 多层感知机 | 首次标注 MLP |
| backpropagation | 反向传播 | 简称 backprop 同译 |
| dropout | 随机失活 | 保留 Dropout 括注 |
| piecewise linear unit | 分段线性单元 | |
| maximum likelihood estimation | 最大似然估计 | |
| Markov chain | 马尔可夫链 | |
| approximate inference | 近似推断 | |
| minimax two-player game | 极小极大双人博弈 | |
| value function | 价值函数 | |
| latent variable | 隐变量 | |
| potential function | 势函数 | |
| partition function | 配分函数 | |
| Markov chain Monte Carlo (MCMC) | 马尔可夫链蒙特卡洛 | 缩写 MCMC 保留 |
| mixing | 混合 | MCMC 语境 |
| log-likelihood | 对数似然 | |
| score matching | 得分匹配 | |
| noise-contrastive estimation (NCE) | 噪声对比估计 | |
| denoising auto-encoder | 去噪自编码器 | |
| contractive autoencoder | 收缩自编码器 | |
| generative stochastic network (GSN) | 生成随机网络 | |
| feedback loop | 反馈回路 | |
| auto-encoding variational Bayes | 自编码变分贝叶斯 | |
| stochastic backpropagation | 随机反向传播 | |
| prior | 先验 | |
| hyperparameter | 超参数 | |
| minibatch | 小批量 | |
| stochastic gradient descent | 随机梯度下降 | 算法 1 标题 |
| momentum | 动量 | |
| overfitting | 过拟合 | |
| saturate | 饱和 | |
| fixed point | 不动点 | |
| non-parametric limit / setting | 非参数极限 / 非参数设定 | |
| probability density function | 概率密度函数 | |
| conditional probability | 条件概率 | |
| global optimum / minimum | 全局最优 / 全局最小值 | |
| virtual training criterion | 虚拟训练准则 | $C(G)$ |
| Kullback–Leibler divergence (KL) | Kullback–Leibler 散度 | 后文用 KL 散度 |
| Jensen–Shannon divergence (JSD) | Jensen–Shannon 散度 | |
| convex | 凸 | |
| subderivative | 次导数 | |
| supremum | 上确界 | |
| support (Supp) | 支撑集 | |
| critical point | 临界点 | |
| parameter space | 参数空间 | |
| rectifier linear activation | 修正线性激活 | |
| sigmoid / maxout | sigmoid / maxout | 保留英文 |
| Gaussian Parzen window | 高斯 Parzen 窗 | Parzen 为人名，保留 |
| cross validation | 交叉验证 | |
| standard error | 标准误 | |
| fold | 折 | 交叉验证语境 |
| fully connected model | 全连接模型 | |
| hidden unit | 隐单元 | |
| conditional mean | 条件均值 | |
| variational inference | 变分推断 | |
| negative chains | 负链 | 玻尔兹曼机训练语境 |
| Boltzmann machine | 玻尔兹曼机 | |
| mode | 模态 | 分布的 mode |
| degenerate distribution | 退化分布 | |
| conditional generative model | 条件生成模型 | |
| wake-sleep algorithm | wake-sleep 算法 | 保留英文 |
| semi-supervised learning | 半监督学习 | |
| labeled data | 标注数据 | |
| restricted Boltzmann machine (RBM) | 受限玻尔兹曼机 | 首次标注后简称 RBM |
| deep Boltzmann machine (DBM) | 深度玻尔兹曼机 | |
| deep belief network (DBN) | 深度信念网络 | |
| directed / undirected graphical model | 有向 / 无向图模型 | |
| hierarchical model | 层次模型 | |
| deep learning | 深度学习 | |
| forward propagation | 前向传播 | |
| inference | 推断 | 名词性语境统一「推断」 |
| data generating distribution | 数据生成分布 | |
| SML/PCD | SML/PCD | 保留（随机最大似然 / 持续对比散度，原文未展开） |
| AIS | AIS | 保留（退火重要性采样，原文未展开） |

保留英文不译：MNIST、Toronto Face Database (TFD)、CIFAR-10、DBN/Stacked CAE/Deep GSN（表 1 模型名）、MP-DBM、Pylearn2、Theano、Helvetica、CIFAR、Canada Research Chairs、Compute Canada、Calcul Québec、人名全部、会议名（NIPS/ICML/ICLR/AISTATS/ICCV/SciPy）全部。

## 三、存疑 / 翻译决策点

1. **公式 (1) 中 $V(D, G)$ 与正文 $V(G, D)$ 不一致**：原文 LaTeX 公式为 $\min_G \max_D V(D, G) = \ldots$，而正文文字写 "value function $V(G, D)$"。按原文保留，未统一，加译注说明。
2. **公式 (4) 最后一行 $P_\text{data}(\bm{x})$ 首字母大写**：与全文 $p_\text{data}$ 不一致，系原文笔误，按原文保留并加译注。
3. **算法 1 中两处 "from noise prior $p_g(\bm{z})$"**：按上下文应为噪声先验 $p_{\bm{z}}(\bm{z})$（$p_g$ 是生成器分布，不是噪声先验），疑为原文笔误。按原文保留，译注说明。
4. **原文拼写/排版笔误**（按原文保留，译注标注）："indistiguishable"（Introduction）、"ina feedback loop"（Related work）、"an a range of datasets"（Experiments）、"divising"（Conclusions）。符合规范"明显笔误可加译者注"。
5. **Table 2 中 "Helvetica."**：疑为作者遗留的占位文本（NIPS 提交版未删净），按原文保留并加译注。
6. **"the Helvetica scenario"**（第 6 节）：作者内部说法，指生成器把多个 $z$ 值坍缩到同一 $x$ 值、多样性不足的失败情形，即后来所称"模式坍缩（mode collapse）"的早期描述。译作「Helvetica 情景」保留原文并加译注，未擅自意译为「模式坍缩」。
7. **Table 1 表题重复句**："On TFD, we computed the standard error across folds..." 与 "On TFD, σ was cross validated on each fold..." 两句信息重叠，系原文如此，全文照译未合并。
8. **"adversarial nets" 的译名**：中文通行名「生成对抗网络」对应本文后续知名度；但本文正文自指用 "adversarial nets"，故正文统一译「对抗网络」，标题与首行说明给出「生成对抗网络」对应关系。
9. **$p_x$ / $p_{\bm{x}}$**：图 1 图题与命题 2 证明中出现 $p_x$、$p_{\bm{x}}$（数据分布的另一记号），保留原记号不统一。
10. **参考文献 [15]**：原文页码 "1558–1161" 明显为 "1558–1561" 之误（Science 268 卷），按规范参考文献整体保留原文，未改动。
11. **"burning in"**：译「预热」，为 MCMC 语境标准译法（burn-in）。
12. **Figure 2 图题 "Rightmost column..."**：图 2 实际布局中右列是"相邻样本对应的最近训练样本"，按原文直译，未按视觉布局改写。

## 四、公式清单（与原文编号一致）

- 式 (1)：极小极大博弈价值函数
- 式 (2)：最优判别器 $D^*_G$
- 式 (3)：$V(G, D)$ 的积分形式
- 式 (4)：$C(G)$ 改写
- 式 (5)：$C(G) = -\log 4 + KL + KL$
- 式 (6)：$C(G) = -\log 4 + 2 \cdot JSD$
- 算法 1：判别器/生成器小批量 SGD 更新式

## 五、图/表页码标注

| 对象 | paper.pdf 页码 |
|---|---|
| Figure 1（训练示意图） | 第 3 页 |
| Algorithm 1（伪代码） | 第 4 页 |
| Table 1（Parzen 对数似然） | 第 6 页 |
| Figure 2（样本可视化） | 第 6 页 |
| Figure 3（z 空间插值数字） | 第 7 页 |
| Table 2（生成建模挑战对比） | 第 7 页 |

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（4 处）；图注内重复的页图链接去重（2 处）；成对重复的页图引用行去重（4 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
