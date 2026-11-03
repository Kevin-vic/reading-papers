# 深度学习经典论文双语翻译库（2012–2023）

本目录收录深度学习领域 25 篇经典论文（2012–2023）的中英对照全文翻译，按年份建目录。每篇论文的目录结构如下：

```
papers/
├── 20xx-名称/
│   ├── paper.pdf              英文原文 PDF（arXiv 版本）
│   ├── paper.md               中英对照全文翻译（双语块格式）
│   ├── translation_notes.md   翻译笔记：术语台账 + 翻译说明 + 存疑点
│   ├── paper_text.txt         PDF 提取的原文纯文本（部分目录有）
│   └── assets/                提取的图片（部分目录有）
└── _scripts/                  批量处理的辅助脚本（PDF 提取、图题去重等）
```

阅读方式：以 `paper.md` 为主（每段先 **Original:** 原文、后 **中文:** 译文，公式与图表页码见文内标注），配合 `paper.pdf` 看图，查 `translation_notes.md` 了解术语译名与翻译决策。

翻译规范依据：`C:/Users/13054/.claude/skills/nature-reader/references/cs-glossary.md`（术语表 + 翻译硬规则八条）。

---

## 时间线目录表

| 年份 | 目录                                    | 论文                                                                                                                                             | 一句话导读                                                                                                                         |
| ---- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| 2012 | [2012-alexnet](2012-alexnet/)           | 用于 ImageNet 的深度卷积神经网络（ImageNet Classification with Deep Convolutional Neural Networks）                                              | 用 ReLU、双 GPU 并行与 Dropout 把深度 CNN 成功训练在百万级 ImageNet 上，将 top-5 错误率从 26% 大幅压到 15.3%，开启深度学习视觉时代 |
| 2013 | [2013-word2vec](2013-word2vec/)         | 词向量空间的有效估计（Efficient Estimation of Word Representations in Vector Space）                                                             | 用 CBOW 与 Skip-gram 两个浅层模型加负采样，在十亿级词上高效训练出能捕捉语义与句法规律的稠密词向量                                  |
| 2014 | [2014-dropout](2014-dropout/)           | Dropout：防止神经网络过拟合的简单方法（Dropout: A Simple Way to Prevent Neural Networks from Overfitting）                                       | 训练时随机丢弃神经元，等效于指数多个「瘦网络」的集成，以极小代价大幅缓解全连接网络过拟合                                           |
| 2014 | [2014-adam](2014-adam/)                 | Adam：一种随机优化方法（Adam: A Method for Stochastic Optimization）                                                                             | 结合梯度一阶矩与二阶矩的自适应学习率优化器，对超参数鲁棒、适合大规模参数与稀疏梯度，成为默认优化器                                 |
| 2014 | [2014-gan](2014-gan/)                   | 生成对抗网络（Generative Adversarial Networks）                                                                                                  | 让生成器与判别器玩 minimax 对抗博弈，无需马尔可夫链即可隐式学出真实数据分布，奠定生成模型新范式                                    |
| 2014 | [2014-vgg](2014-vgg/)                   | 用于大规模图像识别的非常深卷积网络（Very Deep Convolutional Networks for Large-Scale Image Recognition）                                         | 全部用 3×3 小卷积堆叠出 16/19 层网络，证明「深度」本身是性能关键，其简洁结构成为经典主干网络                                       |
| 2014 | [2014-googlenet](2014-googlenet/)       | 用卷积走得更深（Inception 架构）（Going Deeper with Convolutions）                                                                               | 用 Inception 模块做多尺度并行卷积加 1×1 降维，22 层网络参数反而只有 AlexNet 的 1/12，拿下 ILSVRC 2014 冠军                         |
| 2014 | [2014-seq2seq](2014-seq2seq/)           | 用神经网络做序列到序列学习（Sequence to Sequence Learning with Neural Networks）                                                                 | 多层 LSTM 编码器把变长输入压成固定向量、解码器逐词生成，首次实现端到端神经机器翻译                                                 |
| 2014 | [2014-bahdanau](2014-bahdanau/)         | 联合学习对齐与翻译的神经机器翻译（Neural Machine Translation by Jointly Learning to Align and Translate）                                        | 用注意力机制让解码器每步软搜索源句相关位置，摆脱固定长度向量的瓶颈，长句翻译质量显著提升                                           |
| 2015 | [2015-batchnorm](2015-batchnorm/)       | 批归一化：通过减少内部协变量偏移加速深度网络训练（Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift） | 对每层小批量输入做归一化以抑制内部协变量偏移，允许更大学习率、收敛快一个数量级并自带正则化效果                                     |
| 2015 | [2015-resnet](2015-resnet/)             | 用于图像识别的深度残差学习（Deep Residual Learning for Image Recognition）                                                                       | 用残差学习（捷径连接让层拟合恒等映射的偏离）破解网络加深后的退化问题，152 层网络更易优化且精度更高，获 ILSVRC 2015 多项第一        |
| 2015 | [2015-faster-rcnn](2015-faster-rcnn/)   | Faster R-CNN：用候选区域网络走向实时目标检测（Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks）                   | 用与检测网络共享卷积特征的候选区域网络（RPN）替代外部区域提议，把「找框」也交给网络，实现近实时的端到端检测                        |
| 2015 | [2015-unet](2015-unet/)                 | U-Net：用于生物医学图像分割的卷积网络（U-Net: Convolutional Networks for Biomedical Image Segmentation）                                         | 对称编码-解码的 U 形架构加跳跃连接，只用极少量标注样本就能训练出精准的像素级分割                                                   |
| 2015 | [2015-dqn](2015-dqn/)                   | 通过深度强化学习达到人类水平控制（Human-level control through deep reinforcement learning）                                                      | 用深度 CNN 逼近 Q 函数，配合经验回放与目标网络稳定训练，在 49 款 Atari 游戏上达到人类玩家水平                                      |
| 2016 | [2016-yolo](2016-yolo/)                 | YOLO：统一的实时目标检测（You Only Look Once: Unified, Real-Time Object Detection）                                                              | 把目标检测重新表述为单个回归问题，一次前向同时预测所有边界框与类别，以 45 FPS 实现实时检测                                         |
| 2016 | [2016-layernorm](2016-layernorm/)       | 层归一化（Layer Normalization）                                                                                                                  | 对单个样本的所有隐层单元做归一化而非跨 batch，摆脱对批大小与序列长度的依赖，特别适合 RNN 等在线场景（后成 Transformer 标配）       |
| 2016 | [2016-gcn](2016-gcn/)                   | 图卷积网络的半监督分类（Semi-Supervised Classification with Graph Convolutional Networks）                                                       | 给出谱图卷积的一阶近似传播规则，把图结构与节点特征结合做逐层传播，仅用少量标签即可在图数据上高效分类                               |
| 2017 | [2017-transformer](2017-transformer/)   | 注意力即一切（Attention Is All You Need）                                                                                                        | 彻底摒弃 RNN 与卷积，完全靠多头自注意力建模序列，训练高度并行且捕获长距离依赖，成为后来一切大模型的基础架构                        |
| 2018 | [2018-gpt1](2018-gpt1/)                 | 通过生成式预训练改善语言理解（Improving Language Understanding by Generative Pre-Training）                                                      | 先在大规模无标注文本上做生成式预训练语言模型，再到下游任务做判别式微调，多项 NLU 任务大幅超过特定任务模型                          |
| 2018 | [2018-bert](2018-bert/)                 | BERT：用于语言理解的深度双向 Transformer 预训练（BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding）              | 用掩码语言模型与下一句预测在深度双向 Transformer 上预训练，一举刷新 11 项 NLP 任务记录，确立「预训练 + 微调」统治地位              |
| 2019 | [2019-efficientnet](2019-efficientnet/) | EfficientNet：对卷积网络模型缩放的重新思考（EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks）                           | 用复合缩放系数按固定比例同时放大网络深度、宽度与分辨率，同等精度下参数与计算量减少约一个数量级                                     |
| 2020 | [2020-ddpm](2020-ddpm/)                 | 去噪扩散概率模型（Denoising Diffusion Probabilistic Models）                                                                                     | 用马尔可夫链逐步加噪再学反向去噪，以简单的似然目标稳定训练出高质量样本，为扩散模型奠基                                             |
| 2020 | [2020-vit](2020-vit/)                   | 一张图值 16×16 个词：大规模图像识别中的 Transformer（An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale）                | 把图像切成 patch 当词序列直接喂给标准 Transformer，在大数据预训练下超越 CNN，证明视觉不再需要卷积归纳偏置                          |
| 2021 | [2021-clip](2021-clip/)                 | 从自然语言监督中学习可迁移的视觉模型（Learning Transferable Visual Models From Natural Language Supervision）                                    | 用 4 亿图文对做对比学习，把图像与文本映射到同一表示空间，零样本迁移到 30 个下游数据集即可媲美监督模型                              |
| 2023 | [2023-llama](2023-llama/)               | LLaMA：开放而高效的基座语言模型（LLaMA: Open and Efficient Foundation Language Models）                                                          | 只用公开数据、配合高效优化器与推理加速技巧训练开源模型，13B 参数即在多数基准上超过大得多的 GPT-3                                   |

---

## 初学者学习路线

按主题分组的建议阅读顺序。组内顺序即推荐顺序：每篇论文解决的是上一篇留下的问题。

### CNN 主线：alexnet → vgg → googlenet → resnet → efficientnet → vit

- **2012-alexnet** 先读：它把 ImageNet 大胜的完整配方（CNN + ReLU + GPU + 正则）第一次讲清楚，是所有视觉论文的共同背景。
- **2014-vgg** 次之：回答「网络更深会不会更好」，给出 3×3 小卷积堆叠的标准做法，结构最简单、最易读懂。
- **2014-googlenet** 承接：更深以后参数和计算爆炸怎么办，用 Inception 模块在宽度上做多尺度并控算力。
- **2015-resnet** 是这组的高潮：深度再往上走遇到退化问题，残差学习一举解决，之后几乎所有视觉网络都带捷径连接。
- **2019-efficientnet** 做收束：把「怎么放大网络」从手工试错变成系统化的复合缩放，代表 CNN 时代的工程总结。
- **2020-vit** 是转折：同一批任务换 Transformer 也能赢，为进入大模型时代做铺垫。

### NLP 主线：word2vec → seq2seq → bahdanau → transformer → gpt1 → bert → llama

- **2013-word2vec** 是起点：先解决「词怎么表示成向量」，稠密词向量是后面一切模型的地基。
- **2014-seq2seq** 建立框架：端到端序列到序列学习，但固定长度向量是明显瓶颈。
- **2014-bahdanau** 紧接着解这个瓶颈：注意力机制出场，这也是理解 Transformer 的关键前置知识。
- **2017-transformer** 是整条线的高潮：注意力全面取代 RNN，务必精读。
- **2018-gpt1** 转向预训练范式：单向生成式预训练 + 下游微调，大模型路线图的第一块。
- **2018-bert** 对比着读：双向掩码预训练为什么更强，两者差异就是后来自回归 vs 自编码两条路线的分歧点。
- **2023-llama** 收官：预训练范式进入规模化时代后，如何用公开数据与工程技巧做出开放高效的大模型。

### 生成模型：gan → ddpm

- **2014-gan** 先读：用对抗博弈定义「生成」这个目标，概念优雅、公式少，是理解生成模型视角的入口。
- **2020-ddpm** 后读：另一条与对抗完全不同的路线（逐步加噪-去噪），训练稳定，读到这能看到生成模型十几年的范式对比。

### 检测与分割：faster-rcnn → yolo → unet

- **2015-faster-rcnn** 先读：先理解两阶段检测的完整机制（找候选框 → 再分类精修），「候选区域」这个概念是理解检测速度问题的钥匙。
- **2016-yolo** 对比着读：看懂它如何把检测压成单次回归、用速度换精度，两篇对照才能体会「两阶段 vs 一阶段」的取舍。
- **2015-unet** 收尾：转向像素级任务的分割，其跳跃连接与 ResNet 思想一脉相承，作为视觉任务谱系的最后一环。

### 训练与优化：dropout → batchnorm → layernorm → adam

- **2014-dropout** 先读：从「过拟合怎么办」切入，正则化思想最直观。
- **2015-batchnorm** 次之：从正则化转向「训练为什么难、怎么加速」，归一化第一次成为网络标配组件。
- **2016-layernorm** 对比着读：同为归一化，为什么换一个归一化方向就适配了 RNN 与在线场景——这个对比能真正理解归一化的本质。
- **2014-adam** 最后读：前几篇都在改网络，这篇改的是优化器本身，可以作为这组的收尾。

### 强化学习：dqn

- 单篇 **2015-dqn**：建议在读完 CNN 主线（至少 resnet）之后读——DQN 本质是用 CNN 从像素学 Q 值，需要先有视觉表示的知识。

### 图神经网络：gcn

- 单篇 **2016-gcn**：涉及少量谱图论背景，放在 CNN 与训练基础之后读；它的逐层传播形式与 CNN 共享同一套「局部聚合」直觉。

---

## 翻译约定

本库所有译文遵循统一的翻译规范（`nature-reader/references/cs-glossary.md` 的术语表与硬规则八条）。要点如下：

### 术语与译名

- **首次出现原则**：术语第一次出现写成「中文译名（English Term）」，之后统一使用中文译名；同一术语全文译名必须一致。
- **专有名词不译**：模型名（Transformer、ResNet、GPT……）、人名（Hinton、Vaswani……）、机构、数据集（ImageNet、WMT……）、会议名（NIPS、ICML……）一律保留英文。
- **术语台账**：每篇的 `translation_notes.md` 都附有「术语台账」表格（英文术语 | 译名 | 备注），记录该文所有术语的钉定译名与依据，是查证译名一致性的入口。

### 双语块格式（paper.md）

- 全文按段落/小节组织为双语块：先 `**Original:**` 原文，换行后 `**中文:**` 译文；原文与译文按段对齐，不增不减。
- 标题统一为「英文标题（中文译名）」。
- 图表标题（Figure/Table caption）翻译，并标注原图位置，例如「（原图见 paper.pdf 第 4 页）」——正文不嵌图，看图请翻对应页码的 `paper.pdf`。
- 表格以 Markdown 表格重建（表头译中文，数据原样保留）。

### 公式、代码与文献

- **公式不译**：所有数学公式按 LaTeX 原样保留，display 公式用 `$$...$$`，行内公式用 `$...$`；PDF 文本提取导致的上下标丢失已按原文 LaTeX 语义重建。
- **代码与变量名不译**（注释可译），数据集、单位、指标名（如 BLEU、FLOPs）保留原文。
- **参考文献不译**：References 部分整体保留英文原文，方便直接检索引用条目。
- 原文明显的笔误等，以译者注 `[译注：…]` 形式标注，不擅自改动正文。

### 流程说明

本库采用批量简化流程翻译：未生成 `source_map.json`（原文-译文逐句映射文件）；部分目录保留 `paper_text.txt`（PDF 提取的原文纯文本）与 `assets/`（提取的图片）。每篇 `translation_notes.md` 的开头都记录了该文的 PDF 版本、提取方式、与标准流程的差异及存疑点。
