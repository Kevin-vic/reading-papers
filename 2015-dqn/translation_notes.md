# 翻译笔记：Human-level control through deep reinforcement learning

- 论文：Mnih, V. et al. *Human-level control through deep reinforcement learning*. Nature 518, 529–533 (2015). doi:10.1038/nature14236
- 版本说明：**Nature 2015 正式版**（非 arXiv 2013 workshop 版）。Nature 直链 `nature.com/articles/nature14236.pdf` 返回文章落地页 HTML，改用 `https://link.springer.com/content/pdf/10.1038/nature14236.pdf` 下载成功，文件头 `%PDF-`、大小 3,821,434 字节，共 13 页。
- 产出文件：
  - `paper.md` —— 全文双语对照翻译
  - `paper.txt` —— pdftotext 提取的原文文本（备用）
  - `paper.pdf` —— 原文 PDF（第 1 页标题/摘要；第 2 页 Fig. 1、Fig. 2；第 3 页 Fig. 3、Fig. 4 图；第 4 页 Fig. 4 图注与参考文献开头；第 5 页参考文献；第 6–7 页 Methods；第 8–13 页 Extended Data 图 1、2 与表 1–4）
  - `pages/page_*.png|jpg` —— 页面渲染图（本环境模型不支持视觉输入，Read 读图返回 Unsupported Image，未能用于校对，仅存档）

## 批量简化模式说明

本次为批量简化模式：
- 未生成 `source_map.json`（原文段落与 PDF 页码/文本位置的映射文件）；
- 未提取论文内嵌图片到单独文件，图/表位置一律以页码标注，如「（原图见 paper.pdf 第 N 页）」；
- PDF 阅读方式降级：Read 工具直接读 PDF 需要 poppler（pdftoppm），本机未安装，且当前模型不支持读图；改用 Git 自带 `pdftotext -layout` 提取全文文本（13 页全部读完）。**公式的 PDF 字符映射存在乱码**（`5`→`=`、`c`→`γ`、`h`→`θ`、`e-greedy`→`ε-greedy`、`--`→破折号等），Original 文本已按 Nature 正式版原文逐处还原，公式以 LaTeX 重写。

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| reinforcement learning | 强化学习 | 全篇统一 |
| agent | 智能体 | |
| environment | 环境 | |
| policy | 策略 | behaviour policy = 行为策略 |
| deep Q-network (DQN) | 深度 Q 网络（DQN） | DQN 依术语表保留英文 |
| Q-network | Q 网络 | 首次出现「Q 网络（Q-network）」 |
| target Q-network | 目标 Q 网络 | 原文 target network 亦统一译「目标网络」 |
| action-value function | 动作值函数 | 原文亦写作 (Q) function |
| optimal action-value function | 最优动作值函数 | |
| state value (V) | 状态值 | |
| Q-learning | Q 学习 | 算法名 Q-learning 首次带英文 |
| experience replay | 经验回放 | |
| replay memory | 回放记忆 | |
| replay algorithm | 回放算法 | |
| transition | 转移 | |
| episode | 回合 | |
| terminal state | 终止状态 | |
| ε-greedy policy | ε-贪心策略 | 原文 ε 用希腊字母 |
| greedy policy | 贪心策略 | |
| discount factor | 折扣因子 | 原文 discount 动词译「折扣」 |
| return | 回报 | discounted return = 折扣回报 |
| horizon | 视野 | |
| Markov decision process (MDP) | 马尔可夫决策过程（MDP） | |
| value iteration | 值迭代 | |
| Bellman equation | Bellman 方程 | Bellman 为人名，保留英文（不译「贝尔曼」） |
| function approximator | 函数逼近器 | linear/nonlinear function approximator = 线性/非线性函数逼近器 |
| neural fitted Q-iteration | 神经拟合 Q 迭代 | 方法名，首次带英文 |
| model-free | 无模型 | |
| off-policy / on-policy | 离策略 / 同策略 | |
| behaviour distribution | 行为分布 | |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| loss function | 损失函数 | |
| mean-squared error | 均方误差 | |
| gradient | 梯度 | |
| variance | 方差 | |
| learning rate | 学习率 | |
| minibatch | 小批量 | |
| hyperparameter | 超参数 | |
| overfitting | 过拟合 | |
| baseline | 基线 | |
| deep neural network | 深度神经网络 | |
| artificial neural network | 人工神经网络 | |
| deep convolutional network | 深度卷积网络 | |
| convolutional layer | 卷积层 | |
| filter | 滤波器 | |
| stride | 步长 | |
| rectifier nonlinearity | 整流非线性 | 原文括注 max(0,x)，即 ReLU 类 |
| fully connected layer | 全连接层 | |
| forward pass | 前向传播 | |
| receptive field | 感受野 | |
| representation | 表示 | |
| embedding | 嵌入 | |
| t-SNE | t-SNE | 依术语表保留 |
| temporal difference (TD) | 时序差分（TD） | |
| dopaminergic neurons | 多巴胺能神经元 | |
| phasic signals | 相位信号 | |
| perceptual learning | 知觉学习 | |
| hippocampus | 海马体 | |
| basal ganglia | 基底神经节 | |
| prioritized sweeping | 优先回溯 | 首次带英文 |
| salient features / events | 显著特征 / 显著事件 | |
| state-of-the-art | 当前最优 | |
| end-to-end | 端到端 | |
| generalization / generalize | 泛化 | |
| emulator | 模拟器 | Atari 模拟器 |
| preprocessing | 预处理 | |
| luminance (Y channel) | 亮度（Y 通道） | |
| frame | 帧 | |
| frame-skipping | 跳帧 | |
| sprite | 精灵 | Atari 硬件术语 |
| artefact | 伪影 | |
| held-out set | 留出集 | |
| error bars | 误差棒 | |
| s.d. | 标准差 | |
| prior knowledge | 先验知识 | |
| evaluation episode | 评估回合 | |
| annealed | 退火 | ε 线性退火 |
| experience tuple | 经验元组 | |
| memory buffer | 内存缓冲区 | |
| oscillation | 振荡 | |
| divergence / diverge | 发散 | |
| local minimum | 局部极小值 | |
| data efficiency | 数据效率 | |
| partially observed | 部分可观测 | |
| perceptually aliased | 感知上混叠的 | |
| normalized performance | 归一化表现 | |
| robust / robustness | 稳健 / 鲁棒性 | 形容词译「稳健」，名词译「鲁棒性」 |
| general artificial intelligence | 通用人工智能 | |
| norm account | 规范性说明 | normative account |
| Atari 2600 | Atari 2600 | 保留 |
| 游戏名（Space Invaders、Seaquest、Breakout、Pong、Beam Rider、River Raid、Boxing、Enduro、Montezuma's Revenge 等） | 保留英文 | 游戏名为专有名词不译 |
| Hubel、Wiesel、Hinton、Dayan、Bowling 等 | 保留英文 | 人名不译 |
| Square Enix Limited、Atari Interactive, Inc. | 保留英文 | 机构名不译 |

## 数字与单位格式约定

- 按原文保留阿拉伯数字：「50 million frames」→「5000 万帧」；「1 million most recent frames」→「最近 100 万帧」；「84 × 84 × 4」「210 × 160」「60 Hz」「10 Hz」原样保留。
- 「520 k frames」保留为「520k 帧」；「2 h」「30 min」「5 min」译为「2 小时」「30 分钟」「5 分钟」。
- 分数「75%」「5%」原样保留。

## 不确定 / 存疑点

1. **Extended Data Table 1–4 的表格数据未翻译**：表中数值在 PDF 中为图片，文本提取只得到表注。paper.md 中仅译表注并标注页码，表内数字请对照 PDF 第 10–13 页查看。若需要，可用 OCR 或人工录入补全。
2. **公式还原**：pdftotext 输出的公式乱码较严重（嵌入字体编码映射问题），所有公式按 Nature 正式版已知内容以 LaTeX 重写。正文第 2 页的损失函数下标为 $(s,a,r,s')\sim U(D)$，Methods 中的损失函数含方差项（$\mathbb{E}[V_{s'\sim\mathcal{E}}[y]]$）——两处按原文分别保留，未强行统一。
3. **正文段落划分**：Nature Letter 正文无节标题，双栏排版。paper.md 按语义段落切分，段落边界以原文语义为准（Nature 版 529–533 页）；正文中图 1–4 的图注以块引用插在对应论述位置。
4. **术语「rectifier nonlinearity」**：译「整流非线性」。2015 年原文未用 ReLU 一词，译文不引入 ReLU 缩略语，仅在括注中保留原文的 max(0,x)。
5. **「normative account」**：译「规范性说明」，学界亦有「规范解释」译法，此处取直译并保留英文括注。
6. **perceptually aliased**：译「感知上混叠的」，为 POMDP 语境下常见译法「感知混叠」的直译变体，保留英文括注。
7. **图 4 的页码标注**：图本身在 PDF 第 3 页底部、图注跨至第 4 页顶部，故标注「第 3–4 页」。Fig. 1、Fig. 2 同在第 2 页，Fig. 3 在第 3 页。
8. **致谢、作者贡献等非技术内容**：依「全文翻译不删节」原则已译；References 依硬规则整体保留原文。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（4 处）；图注内重复的页图链接去重（1 处）；成对重复的页图引用行去重（4 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
