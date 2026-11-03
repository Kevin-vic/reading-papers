# 翻译笔记：EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

- 论文：EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks（EfficientNet：对卷积网络模型缩放的重新思考）
- 来源：arXiv:1905.11946v5 [cs.LG]，ICML 2019（PMLR 97）
- 下载：https://arxiv.org/pdf/1905.11946（经代理 127.0.0.1:7890），933,600 字节，共 11 页
- 翻译依据：`C:/Users/13054/.claude/skills/nature-reader/references/cs-glossary.md`（术语表 + 翻译硬规则八条）

## 术语台账

| 英文术语 | 译名 | 备注 |
|---|---|---|
| Convolutional Neural Networks (ConvNets) | 卷积神经网络（ConvNets） | 首次标注后常用 ConvNets |
| model scaling | 模型缩放 | |
| scale up | 放大 | |
| compound scaling | 复合缩放 | 论文核心方法 |
| compound coefficient | 复合系数 | |
| scaling coefficients | 缩放系数 | |
| baseline network | 基线网络 | |
| neural architecture search | 神经架构搜索 | |
| multi-objective neural architecture search | 多目标神经架构搜索 | |
| grid search | 网格搜索 | |
| state-of-the-art (SOTA) | 当前最优（state-of-the-art，SOTA） | |
| top-1 / top-5 accuracy | top-1 / top-5 准确率 | 保留 top-1 写法 |
| receptive field | 感受野 | |
| fine-grained | 细粒度 | |
| transfer learning | 迁移学习 | |
| finetune | 微调 | |
| pretrained | 预训练 | |
| inference | 推理 | |
| latency | 延迟 | |
| FLOPS | FLOPS | 保留英文 |
| parameters (#Params) | 参数（参数量） | |
| depth / width / resolution | 深度 / 宽度 / 分辨率 | 三个缩放维度 |
| vanishing gradient | 梯度消失 | |
| skip connection | 跳跃连接 | |
| batch normalization | 批归一化 | |
| down-sampling | 下采样 | |
| model compression | 模型压缩 | |
| over-parameterized | 过度参数化 | |
| hand-crafted | 手工设计 | |
| pipeline parallelism | 流水线并行 | |
| accelerator | 加速器 | |
| expressive power | 表达能力 | |
| tensor | 张量 | |
| operator | 算子 | |
| stage | 阶段（stage） | |
| squeeze-and-excitation | 压缩-激励 | |
| mobile inverted bottleneck (MBConv) | 移动倒置瓶颈（MBConv） | |
| stochastic depth | 随机深度 | |
| survival probability | 存活概率 | |
| Dropout | 随机失活（Dropout） | |
| regularization | 正则化 | |
| early stopping | 早停 | |
| checkpoint | 检查点 | |
| epoch | 轮次（epoch） | |
| optimizer | 优化器 | |
| momentum / decay | 动量 / 衰减 | |
| weight decay | 权重衰减 | |
| learning rate | 学习率 | |
| hyperparameter | 超参数 | |
| minival set | minival 集（小验证集） | |
| class activation map (CAM) | 类激活图（CAM） | |
| object detection | 目标检测 | |
| single-crop | 单裁剪 | |
| multi-crop | 多裁剪 | |
| ensemble | 集成 | |
| depth multiplier | 深度乘子 | 脚注 2，指宽度系数 w |
| domain adaptive transfer learning (DAT) | 域自适应迁移学习（DAT） | |
| an order of magnitude | 一个数量级 | |
| 8.4x / 6.1x | 8.4 倍 / 6.1 倍 | 数字与单位全文统一 |
| 66M / 37B / 400M | 66M / 37B / 400M | 表格与正文中的 M/B 缩写保留原格式 |
| principled | 有原则的 | |
| empirical study | 经验研究 | |
| generalize | 泛化 | |

## 批量简化模式说明

本次为批量简化模式：

- 未生成 `source_map.json`（无逐句源文件映射），双语对照直接以 **Original:** / **中文:** 段落块形式写在 `paper.md` 中。
- 未做图片提取：图 1–8、表 1–8 均以「块引用 + 页码」标注原图位置（如 `> **图 1（译）**：……（原图见 paper.pdf 第 1 页）`）。
- PDF 读取降级：本机无 poppler-utils，Read 工具无法直接渲染 PDF 页面，改用 Python `pypdf` 提取全部 11 页文本（`paper.txt` 中间文件已生成于同目录），再逐页读取翻译。图片内容未做视觉核对。

## 存疑 / 不确定的翻译点

1. **原文语法错误（未改，已加译注）**：3.3 节 "In this paper, we constraint α·β²·γ² ≈ 2"，原文应为 "constrain"。译文按 constrain 理解，并加 `[译注：原文此处为 constraint，应为 constrain]`。
2. **Introduction "four time larger"**：原文即如此（应为 four times larger），译文按"放大四倍"处理，未加译注。
3. **表 4 中 EfficientNet-B1 为 78.8%，与正文/图 1 的 79.1% 不一致**：原文如此（延迟测量模型与主实验的训练设置不同所致），按原文保留，不做改动。
4. **公式重建**：PDF 文本提取无法还原 LaTeX 源码，公式 (1)(2)(3) 按 arXiv v5 官方 LaTeX 源码重建（如 $\mathcal{N} = \bigodot_{i=1...s} \mathcal{F}_i^{L_i}(X_{\langle H_i, W_i, C_i \rangle})$），与原文排版一致。
5. **连字还原**：pypdf 提取的 `ﬁ`（U+FB01）已还原为 "fi"（如 EfficientNet、EfficientNets）；正文中 GPipe/GPIPE、AmoebaNet/AmeobaNet、CovNets/ConvNets 的提取差异均按标准拼写还原。
6. **Flowers 数据集译名**：按术语表规则数据集名不译，正文中保留英文 "Flowers"。
7. **表 5 结构复杂**（四组对照列），重建为 markdown 时合并表头为两组大列，数据与原表逐格一致。
8. **GoogleNet vs GoogLeNet**：原文 Related Work 中写作 "GoogleNet"（原文如此），保留。

## 修复记录（2026-11-03）

- 图注尾部重复的「见下方图，见下方图」去重（8 处）；图注内重复的页图链接去重（8 处）；成对重复的页图引用行去重（8 对，修复后每图仅保留一张 `![第 X 页原图](assets/pages/page-X.png)`，并在图片行后补齐空行）。翻译内容与术语台账未改动。
