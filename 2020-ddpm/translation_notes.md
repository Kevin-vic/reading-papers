# 翻译笔记：Denoising Diffusion Probabilistic Models（去噪扩散概率模型）

- 论文：Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. arXiv:2006.11239v2
- 原文 PDF：`C:/Users/13054/mclearning/papers/2020-ddpm/paper.pdf`（10,267,274 字节，25 页）
- 翻译稿：`C:/Users/13054/mclearning/papers/2020-ddpm/paper.md`
- 术语规范依据：`C:/Users/13054/.claude/skills/nature-reader/references/cs-glossary.md`

## 一、批量简化模式说明

本次为批量简化模式，与完整 nature-reader 流程相比有以下降级：

1. **未生成 source_map.json**：未做段落级原文定位映射文件。图/表以页码标注（"原图见 paper.pdf 第 N 页"）。
2. **未提取图片**：未把 PDF 中的插图导出为独立图片文件；全部图（图 1–图 19）以块引用形式标注图题译文与页码。
3. **原文获取方式降级**：本机未安装 poppler（Read 工具直接读 PDF 依赖 pdftoppm，不可用），故先用 PyMuPDF（`fitz`）把 25 页全文按页提取为 `paper_text.txt`，再基于该文本逐段翻译。提取文本中数学公式呈现为错位乱码，所有公式已按论文 LaTeX 源码（论文公开版本）重建为规范 LaTeX 形式，数值与符号与原文一致。
4. 正文中的连字（ligature，如 "difﬁcult"）已还原为正常拼写（"difficult"）；换行断词已拼接。

## 二、术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| diffusion probabilistic model | 扩散概率模型 | 首次出现标注英文；后文简称"扩散模型" |
| diffusion model | 扩散模型 | glossary 收录 |
| latent variable model | 隐变量模型 | |
| nonequilibrium thermodynamics | 非平衡热力学 | |
| variational bound | 变分下界 | glossary 未收录此组合词，取通用译法 |
| variational inference | 变分推断 | |
| denoising score matching | 去噪分数匹配 | |
| score matching | 分数匹配 | |
| annealed Langevin dynamics | 退火 Langevin 动力学 | Langevin 为人名，保留 |
| annealed importance sampling | 退火重要性采样 | |
| forward process | 前向过程 | glossary 收录 |
| reverse process | 反向过程 | glossary 收录 |
| Markov chain | 马尔可夫链 | |
| variational autoencoder (VAE) | 变分自编码器（VAE） | |
| generative adversarial network (GAN) | 生成对抗网络（GAN） | 首次出现给中文全称，此后统一 GAN |
| flow / flow-based model | 流模型 | glossary 未收录；flow 学界通译"流模型" |
| energy-based model | 基于能量的模型 | 文中 EBM 简称保留 |
| autoregressive model | 自回归模型 | glossary 收录 |
| Gaussian | 高斯 | glossary 收录 |
| variance schedule | 方差调度 | |
| reparameterization | 重参数化 | |
| stochastic gradient descent | 随机梯度下降（SGD） | glossary 收录；文中未出现全称，译"随机梯度下降" |
| KL divergence | KL 散度 | glossary 收录 |
| Rao-Blackwellized | Rao-Blackwell 化 | 人名保留 |
| Monte Carlo estimate | Monte Carlo 估计 | |
| denoising autoencoder | 去噪自编码器 | 出现在第 3 节标题 |
| decoder | 解码器 | glossary 收录 |
| codelength | 码长 | 信息论术语 |
| bits per dimension (bits/dim) | 比特/维度 | 全文统一"比特/维度" |
| rate | 码率 | 率失真语境 |
| distortion | 失真 | |
| rate-distortion | 率失真 | |
| lossy / lossless compression | 有损 / 无损压缩 | |
| progressive | 渐进式 | |
| minimal random coding | 最小随机编码 | |
| stochastic encoder | 随机编码器 | |
| mutual information | 互信息 | |
| entropy | 熵 | 式 (16) 中 $H(x_0)$ |
| log likelihood | 对数似然 | glossary 收录 log-likelihood |
| negative log likelihood (NLL) | 负对数似然 | |
| Inception score | Inception 分数 | 简称 IS |
| FID score | FID 分数 | glossary 收录 |
| U-Net / PixelCNN++ / Wide ResNet / RefineNet / NCSN / BigGAN / StyleGAN / ProgressiveGAN / SNGAN | 保留英文 | glossary 硬规则：模型名不译 |
| group normalization | 组归一化 | 类比 glossary 的 batch normalization→批归一化 |
| weight normalization | 权重归一化 | |
| sinusoidal position embedding | 正弦位置嵌入 | glossary：position embedding→位置嵌入 |
| self-attention | 自注意力 | glossary 收录 |
| feature map | 特征图 | glossary 收录 |
| residual block | 残差块 | |
| dilated convolution | 空洞卷积 | |
| backbone | 主干网络 | glossary 收录 |
| dropout | 随机失活（Dropout） | glossary 收录 |
| EMA | EMA（指数滑动平均） | 原文缩写，首次出现补中文 |
| batch size | 批次大小 | |
| learning rate | 学习率 | glossary 收录 |
| hyperparameter | 超参数 | glossary 收录 |
| inductive bias | 归纳偏置 | |
| conceptual compression | 概念压缩 | Gregor et al. 论文术语 |
| infusion training | 注入训练（infusion training） | 方法名保留英文并给出译名 |
| variational walkback | variational walkback | 专有方法名，保留英文 |
| generative stochastic networks | 生成随机网络 | |
| distribution shift | 分布偏移 | |
| overfitting | 过拟合 | glossary 收录 |
| signal-to-noise ratio | 信噪比 | |
| ablations / ablation | 消融实验 | glossary 收录 ablation study |
| interpolation | 插值 | |
| number of parameters | 参数数量 | "35.7 million"→"3570 万"（数字单位规则统一用"万/亿"） |
| uncurated samples | 未经筛选的样本 | |

## 三、不确定/存疑的翻译点

1. **"variational bound" 的译法**：glossary 未收录该组合词。正文统一译"变分下界"（evidence lower bound 的语境），仅在 Abstract 首次出现时标注（variational bound）。若读者习惯"变分界"亦可接受。
2. **"flow" 的译法**：译"流模型"，glossary 未收录。首次出现已标注英文。
3. **"Rao-Blackwellized fashion"**：译为"以 Rao-Blackwell 化的方式"，保留人名不译，符合硬规则。
4. **"bits per dimension" 与 "bits/dim"**：全文统一"比特/维度"，与"每维度 0.03 比特"混用（后者为行文通顺的语序倒装，数值语义一致）。
5. **式 (16) 中 "H(x0)"**：译注未展开；H 为熵（entropy），已在术语台账注明。
6. **图 2 的图注**：原文图注仅一句 "The directed graphical model considered in this work"，图中节点/箭头在文本提取中错位，译者根据论文语义补注了图形结构描述（$x_T \to \cdots \to x_0$ 反向链在上、$x_0 \to \cdots \to x_T$ 前向链在下），该描述为译者补充而非原文，请对照 paper.pdf 第 2 页核实。
7. **图 9 的图注**：原文图注较短，图内坐标文字（Source / Rec. / λ 值 / 步数）已提取到，译者将其并入块引用以完整呈现图的内容结构。
8. **"masked out"（第 4.3 节）**：译"屏蔽"。若按"掩蔽"理解（masking noise 相关语境）亦通；全文统一用"屏蔽"。
9. **"which seems unstable to train with the larger learning rate"（附录 B）**：原文主语指向 256 × 256 图像（模型），译为"用较大学习率训练似乎不稳定"，符合上下文。
10. **参考文献 [62] 标题大小写**：原文即为 "Cnn-generated images are surprisingly easy to spot...for now."，按硬规则参考文献整体保留原文，未做大小写订正。
11. **公式重建**：全部公式（(1)–(26)）按论文 LaTeX 源码重建，未改动任何符号；$\bar\alpha$ 使用 $\bar\alpha_t$，$\epsilon$ 与原文一致。若个别上下标与 arXiv 排版有出入，以 paper.pdf 为准。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（17 处）；图注内重复的页图链接去重（6 处）；成对重复的页图引用行去重（17 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
