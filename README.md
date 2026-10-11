# Word2Vec NLP Tutorial - 复现与学习

本仓库记录了对 Kaggle **“Bag of Words Meets Bags of Popcorn”** NLP 教程及经典 NLP 模型的学习与复现过程。

项目以 **IMDb 电影评论情感分类** 为统一任务，从传统的 Bag of Words 文本表示开始，逐步学习 Word2Vec、GloVe 等经典词向量方法，并进一步使用 CNN、LSTM、GRU、Attention 和 Transformer 进行文本情感分类。

实验结果均通过 Kaggle **“Bag of Words Meets Bags of Popcorn”** 竞赛进行测试。

---

## 项目内容

### Part 1：Bag of Words

对电影评论进行文本清洗和预处理，使用 Bag of Words（词袋模型）将文本转换为固定维度的词频向量，并使用 Random Forest 完成情感二分类。

主要流程：

`IMDb Review → Text Cleaning → Bag of Words → Random Forest → Sentiment`

Kaggle Score：**0.84432**

---

### Part 2：Word2Vec Average Vectors

使用 Word2Vec 学习词的分布式向量表示，并通过对一条评论中所有词向量求平均，将不同长度的文本转换为固定的 300 维向量，最后使用 Random Forest 进行情感分类。

主要流程：

`IMDb Review → Word2Vec → Average Word Vectors → Random Forest → Sentiment`

Kaggle Score：**0.82792**

---

### Part 3：Word2Vec Clustering / Bag of Centroids

使用 K-Means 对 Word2Vec 词向量进行聚类，将语义相近的词划分到同一 cluster，并使用 Bag of Centroids 将每条评论表示为基于语义词簇的固定维度向量，再使用 Random Forest 完成分类。

主要流程：

`IMDb Review → Word2Vec → K-Means → Bag of Centroids → Random Forest → Sentiment`

Kaggle Score：**0.84768**

---

## Part 4：GloVe + Neural Networks

在 Word2Vec 实验的基础上，进一步学习经典的预训练词向量 **GloVe** 以及神经网络文本分类模型。

实验使用 Stanford GloVe **Common Crawl 840B 300d** 预训练词向量，将 IMDb 评论中的单词映射为 300 维词向量，并分别使用 CNN、LSTM 和 GRU 完成情感二分类。

### CNN

使用冻结的 GloVe 词向量作为 Embedding，通过一维卷积提取文本中的局部模式，再经过 Global Max Pooling 和全连接层完成分类。

主要流程：

`IMDb Review → GloVe Embedding → Conv1D → ReLU → Global Max Pooling → Linear → Sentiment`

实验配置：

* GloVe Embedding：300d
* Embedding：Frozen
* Conv1D filters：128
* Kernel size：3
* Optimizer：SGD
* Learning rate：0.8
* Epochs：10
* Batch size：64

最佳 Validation Accuracy：**80.46%**

Kaggle Score：**0.78600**

---

### LSTM

使用双向 LSTM 对 GloVe 词向量序列进行建模，以学习文本中的顺序信息和较长距离的上下文依赖关系。

主要流程：

`IMDb Review → GloVe Embedding → BiLSTM → Feature Concatenation → Linear → Sentiment`

实验配置：

* GloVe Embedding：300d
* Embedding：Frozen
* Hidden size：120
* LSTM layers：2
* Bidirectional：True
* Batch size：64
* Epochs：10
* Optimizer：Adam

原始代码使用 `lr=0.05`，训练过程中出现数值不稳定（NaN）。在保持其他网络结构和训练设置不变的情况下，将学习率调整为 `0.001` 后训练恢复稳定。

最佳 Validation Accuracy：**89.36%**

Kaggle Score：**0.88660**

---

### GRU

使用双向 GRU 对 GloVe 词向量序列进行建模，并学习门控的序列表示。

主要流程：

`IMDb Review → GloVe Embedding → BiGRU → Feature Concatenation → Linear → Sentiment`

实验配置：

* GloVe Embedding：300d
* Embedding：Frozen
* Hidden size：120
* GRU layers：2
* Bidirectional：True
* Batch size：64
* Epochs：10
* Optimizer：SGD

原始代码使用 `lr=0.8`，训练从第一个 epoch 开始出现 NaN。随后进行了单变量学习率实验：

| Learning Rate | Training Status | Best Validation Accuracy |
| ------------: | --------------- | -----------------------: |
| 0.8 | NaN / 数值失稳 | 无有效结果 |
| 0.01 | Stable | **65.11%** |
| 0.001 | Stable，但 10 epochs 内收敛较慢 | 55.66% |

最终使用 `lr=0.01` 的稳定版本生成 Kaggle Submission。

Kaggle Score：**0.64148**

---

## Part 5：BiLSTM + Temporal Attention

在 BiLSTM 的基础上进一步加入 Temporal Attention，使模型能够根据当前分类任务，为不同时间步的 hidden states 学习不同的注意力权重。

主要流程：

`IMDb Review → GloVe Embedding → BiLSTM → Temporal Attention → Context Vector → Linear → Sentiment`

实验中对 padding token 使用 mask，使其 Attention weight 为 0。

实验配置：

* GloVe Embedding：300d
* Embedding：Frozen
* BiLSTM Hidden size：128
* LSTM layers：2
* Bidirectional：True
* Attention：Temporal Attention
* Context dimension：256
* Optimizer：Adam
* Learning rate：0.01
* Epochs：10
* Batch size：64

最佳 Validation Accuracy：**90.55%**

Kaggle Score：**0.88588**

---

## Part 6：Transformer Encoder

实现基于 **Transformer Encoder** 的 IMDb 情感分类模型。

通过 Multi-Head Self-Attention 直接建模不同 token 之间的关系。

主要流程：

`IMDb Review → Word Embedding + Positional Encoding → Transformer Encoder → Masked Mean Pooling → Linear → Sentiment`

最终使用的网络结构：

```text
Word-level Token IDs [B, 512]
→ Trainable Embedding(vocab_size, 300)
→ Sinusoidal Positional Encoding
→ 2-layer Transformer Encoder
→ Masked Mean Pooling
→ Linear(300, 2)
→ Sentiment
```

### Transformer 配置

* Vocabulary size：74,220
* PAD：0
* UNK：1
* Maximum sequence length：512
* Embedding dimension：300
* Embedding：Randomly initialized / Trainable
* `d_model`：300
* Attention heads：2
* Dimension per head：150
* Encoder layers：2
* Feedforward dimension：512
* Dropout：0.1
* Activation：ReLU
* Pooling：Masked Mean Pooling
* Classifier：`Linear(300, 2)`
* Loss：CrossEntropyLoss
* Batch size：64
* Epochs：10

### Learning Rate 实验

导师原始代码使用：

`Adam lr=0.05`

在保持其他模型结构和训练配置不变的情况下，仅将：

`lr=0.05 → lr=0.001`

模型恢复正常训练。

| Learning Rate | Best Validation Accuracy | Test Prediction | Training Status |
| ------------: | -----------------------: | --------------- | --------------- |
| 0.05 | 50.96% | 25,000 / 0 | 单类别退化 |
| 0.001 | **85.20%** | 11,117 / 13,883 | Stable |

最终采用：

`Adam lr=0.001`

Kaggle Score：**0.80772**

---

## Part 7：BERT Fine-tuning

在 Transformer Encoder 实验的基础上，进一步学习预训练语言模型 **BERT**，并使用 Hugging Face 提供的 `bert-base-uncased` 在 IMDb 情感分类任务上进行 Fine-tuning。

加载已经完成预训练的 `bert-base-uncased` 权重，并在 IMDb 有标签数据上进行二分类微调。

主要流程：

`IMDb Review → WordPiece Tokenizer → Pretrained BERT → [CLS] Representation → Linear → Sentiment`

模型输入经过 WordPiece Tokenizer 处理，并加入 `[CLS]` 和 `[SEP]` 等特殊 token。BERT 使用 Token Embedding、Position Embedding 和 Segment Embedding 构造输入表示，再经过 12 层双向 Transformer Encoder 获得上下文化表示。

对于情感分类任务，使用 `[CLS]` 对应的序列级表示，并通过 `Linear(768, 2)` 完成 Positive / Negative 二分类。

### BERT 配置

* Pretrained Model：`bert-base-uncased`
* Tokenizer：BertTokenizerFast / WordPiece
* Vocabulary size：30,522
* Maximum sequence length：512
* Hidden size：768
* Encoder layers：12
* Attention heads：12
* Dimension per head：64
* Feedforward dimension：3,072
* Activation：GELU
* Dropout：0.1
* Pooling：`[CLS]` Pooler
* Classifier：`Linear(768, 2)`
* Loss：CrossEntropyLoss
* Optimizer：AdamW
* Learning rate：`5e-5`
* Weight decay：0.01
* Warmup steps：500
* Scheduler：Linear
* Gradient clipping：1.0
* Train batch size：6
* Evaluation batch size：12
* Epochs：3

训练过程中整个预训练 BERT 与新加入的分类层共同进行 Fine-tuning。

| Epoch | Train Loss | Train Accuracy | Validation Accuracy |
| ----: | ---------: | -------------: | ------------------: |
| 1 | 0.395091 | 86.99% | 90.00% |
| 2 | 0.225738 | 94.85% | 92.52% |
| 3 | 0.087335 | 98.36% | **93.16%** |

最佳 Validation Accuracy：**93.16%**

Kaggle Score：**0.93268**

---

## Part 8：RoBERTa-large Fine-tuning

在 BERT 实验的基础上，进一步使用 **RoBERTa-large** 进行 IMDb 情感分类。

RoBERTa 保留了 Transformer Encoder 的主体结构，并在 BERT 的基础上进一步优化预训练策略，包括取消 NSP、采用 Dynamic Masking，并使用更多数据和更充分的预训练。

本实验加载 Hugging Face 官方预训练模型：

`FacebookAI/roberta-large`

并在 IMDb 有标签数据上进行全参数 Fine-tuning。

主要流程：

`IMDb Review → Byte-level BPE Tokenizer → Pretrained RoBERTa-large → First Token Representation → Classification Head → Sentiment`

### RoBERTa-large 配置

* Pretrained Model：`FacebookAI/roberta-large`
* Tokenizer：RobertaTokenizerFast / Byte-level BPE
* Vocabulary size：50,265
* Maximum sequence length：512
* Hidden size：1,024
* Encoder layers：24
* Attention heads：16
* Dimension per head：64
* Feedforward dimension：4,096
* Dropout：0.1
* Parameter count：约 355M
* Fine-tuning：Full Fine-tuning
* FP16：True
* Gradient checkpointing：True
* Train micro-batch：1
* Gradient accumulation：4
* Effective batch size：4
* Learning rate：`5e-6`
* Weight decay：0.01
* Warmup steps：500
* Epochs：3

最佳 Validation Accuracy：**96.18%**

Kaggle Score：**96.35%**

---

## Part 9：DeBERTa-Large Fine-tuning

在 RoBERTa 实验之后，进一步学习并实验 **DeBERTa**。

DeBERTa 在 BERT / RoBERTa 的 Encoder-only Transformer 路线上进一步引入：

* Disentangled Attention
* Relative Position Representation
* Enhanced Mask Decoder

本实验使用：

`microsoft/deberta-large`

作为 pretrained backbone，并在 IMDb 情感分类任务上进行全参数 Fine-tuning。

主要流程：

`IMDb Review → DeBERTa Tokenizer → Pretrained DeBERTa-large → Disentangled Transformer Encoder → Classification Head → Sentiment`

### DeBERTa-Large 配置

* Pretrained Model：`microsoft/deberta-large`
* Hidden size：1,024
* Encoder layers：24
* Attention heads：16
* Feedforward dimension：4,096
* Maximum sequence length：512
* Attention：Disentangled Attention
* Relative position：Enabled
* Fine-tuning：Full Fine-tuning

最佳 Validation Accuracy：**95.74%**

Kaggle Score：**96.03%**

---

## Part 10：DeBERTa-Large + LoRA

在 DeBERTa 全参数 Fine-tuning 的基础上，进一步学习 **Parameter-Efficient Fine-Tuning（PEFT）**。

首先实验 **LoRA（Low-Rank Adaptation）**。

LoRA 冻结 pretrained backbone 的原始权重，并通过低秩矩阵学习任务相关的权重增量：

\[
\Delta W = BA
\]

有效权重可表示为：

\[
W_{\text{effective}}
=
W+
\frac{\alpha}{r}BA
\]

本实验将 LoRA 挂载到 DeBERTa-large 的 Attention `in_proj` 模块，仅训练少量 LoRA 参数和分类头，而不对整个 DeBERTa backbone 进行全参数更新。

主要流程：

`IMDb Review → Pretrained DeBERTa-large + LoRA Adapter → Classification Head → Sentiment`

### LoRA 配置

* Base Model：`microsoft/deberta-large`
* PEFT Method：LoRA
* Rank `r`：16
* LoRA alpha：32
* LoRA dropout：0.05
* Target modules：24 层 Attention `in_proj`
* Trainable parameters：2,624,514
* Trainable percentage：0.6419%
* Gradient checkpointing：Enabled
* Checkpoint interval：每 500 optimizer steps

最佳 Validation Accuracy：**96.08%**

Kaggle Score：**96.396%**

---

## Part 11：DeBERTa-Large + AdaLoRA

在 LoRA 的基础上进一步实验 **AdaLoRA（Adaptive LoRA）**。

普通 LoRA 为各目标层使用固定 rank，而 AdaLoRA 根据训练过程动态评估不同低秩方向的重要性，并重新分配有限的 rank budget。

主要流程：

`IMDb Review → Pretrained DeBERTa-large + AdaLoRA Adapter → Classification Head → Sentiment`

### AdaLoRA 配置

* Base Model：`microsoft/deberta-large`
* PEFT Method：AdaLoRA
* Initial rank：16
* Target rank：8
* LoRA alpha：32
* LoRA dropout：0.05
* `tinit`：0
* `tfinal`：0
* `deltaT`：1
* `beta1`：0.85
* `beta2`：0.85
* Orthogonal regularization weight：0.5
* Target modules：24 层 Attention `in_proj`
* Trainable parameters：2,624,898
* Trainable percentage：0.6420%
* 最终活动 rank 总数：192
* Checkpoint interval：每 500 optimizer steps

训练结果：

| Epoch | Train Loss | Train Accuracy | Validation Accuracy |
| ----: | ---------: | -------------: | ------------------: |
| 1 | 0.7045 | 63.56% | 94.52% |
| 2 | 0.1957 | 94.34% | 94.94% |
| 3 | 0.1697 | 94.785% | **94.96%** |

AdaLoRA 的训练 loss 包含正交正则项，因此不能直接与普通 LoRA 的 loss 数值进行比较。

最佳 Validation Accuracy：**94.96%**

Kaggle Score：**95.568%**

---

## Part 12：DeBERTa-Large + Prefix Tuning

进一步实验 **Prefix Tuning**。

Prefix Tuning 不直接修改 DeBERTa backbone 的大规模预训练权重，而是引入一组可训练的 virtual prefix representations，使 Attention 在处理真实文本时同时利用学习得到的 prefix 信息。

主要流程：

`Learnable Prefix → DeBERTa Attention → Frozen Pretrained DeBERTa-large → Classification Head → Sentiment`

### Prefix Tuning 配置

* Base Model：`microsoft/deberta-large`
* PEFT Method：Prefix Tuning
* Virtual tokens：20
* Backbone：Frozen
* Classification head：Trainable
* Gradient checkpointing：Enabled
* Checkpoint interval：每 500 optimizer steps

最佳 Validation Accuracy：**95.26%**

Kaggle Score：**95.884%**

---

## Part 13：DeBERTa-Large + P-Tuning

进一步实验 **P-Tuning**。

P-Tuning 在输入端引入可训练的 continuous virtual prompt，并通过 Prompt Encoder 学习适合当前任务的连续 Prompt Representation，再与真实文本表示共同送入 pretrained DeBERTa。

主要流程：

`Virtual Tokens → Prompt Encoder → Continuous Prompt Representation → Pretrained DeBERTa-large → Classification Head → Sentiment`

### P-Tuning 配置

* Base Model：`microsoft/deberta-large`
* PEFT Method：P-Tuning
* Virtual tokens：20
* Prompt Encoder hidden size：128
* Backbone：Frozen
* Classification head：Trainable
* Gradient checkpointing：Enabled
* Checkpoint interval：每 500 optimizer steps

最佳 Validation Accuracy：**95.58%**

Kaggle Score：**96.032%**

---

## Part 14：BERT-base + Supervised Contrastive Learning

在普通 BERT Fine-tuning 的基础上，进一步实验 **Supervised Contrastive Learning（SCL，监督对比学习）**。

普通 Cross Entropy 主要约束最终分类结果，而 SCL 进一步约束 hidden representation space，使相同标签样本的表示更加接近，不同标签样本的表示相对分离。

本实验使用：

`bert-base-uncased`

作为 pretrained backbone，并在分类损失之外加入 Supervised Contrastive Loss。

主要训练目标：

\[
L
=
L_{\text{CE}}
+
0.2L_{\text{SCL}}
\]

导师原始 SCL Demo 直接将 `[B,768]` 的 pooled representation 输入 `SupConLoss`，而该损失函数实际期望的输入格式为：

`[Batch Size, Number of Views, Hidden Dimension]`

因此正式实验进行了必要的理论修正：

`BERT pooled representation [B,768] → 两次独立 Dropout → Stack → [B,2,768] → SupConLoss`

分类分支仍使用其中一条 Dropout representation 完成情感分类。

主要流程：

`IMDb Review → BERT → [CLS] Pooled Representation → Two Dropout Views → CE + Supervised Contrastive Loss → Sentiment`

### SCL 配置

* Base Model：`bert-base-uncased`
* Hidden size：768
* Maximum sequence length：512
* Contrastive views：2
* View generation：Independent Dropout
* SCL temperature：0.07
* SCL weight：0.2
* Loss：`CE + 0.2 × SCL`
* Optimizer：AdamW
* Learning rate：`5e-5`
* Weight decay：0.01
* Warmup steps：500
* Train batch size：2
* Evaluation batch size：4
* Epochs：3
* Fine-tuning：Full Fine-tuning

训练结果：

| Epoch | Total Loss | CE Loss | SCL Loss | Train Accuracy | Validation Accuracy |
| ----: | ---------: | ------: | -------: | -------------: | ------------------: |
| 1 | 0.867953 | 0.669077 | 0.994378 | 84.16% | 87.42% |
| 2 | 0.630850 | 0.459679 | 0.855858 | 91.28% | 89.20% |
| 3 | 0.435783 | 0.288261 | 0.737607 | 95.08% | **90.52%** |

最佳 Validation Accuracy：**90.52%**

Kaggle Score：**0.89876**

在当前实验配置下，SCL 的 Validation Accuracy 和 Kaggle Score 均低于普通 BERT Fine-tuning。该结果只代表当前模型、batch size、view 构造方式和超参数配置，不能据此认为 SCL 在所有文本分类任务中均弱于普通 Cross Entropy。

---

## Part 15：BERT-base + R-Drop

进一步实验 **R-Drop（Regularized Dropout）**。

R-Drop 对同一个输入执行两次独立的 stochastic forward。由于训练阶段 Dropout mask 不同，两次 forward 会得到不同的 prediction distribution。

R-Drop 在分别使用 Cross Entropy 保证两次预测正确的同时，通过 Symmetric KL Divergence 约束两次输出分布保持一致，从而提高模型面对 Dropout 随机扰动时的预测稳定性。

主要训练目标：

\[
L
=
L_{\text{CE1}}
+
L_{\text{CE2}}
+
\frac{
KL(P_1||P_2)
+
KL(P_2||P_1)
}{2}
\]

主要流程：

`IMDb Review → BERT Forward #1 / BERT Forward #2 → logits₁ / logits₂ → CE₁ + CE₂ + Symmetric KL → Sentiment`

### R-Drop 配置

* Base Model：`bert-base-uncased`
* Fine-tuning：Full Fine-tuning
* Hidden size：768
* Maximum sequence length：512
* Forward passes per training batch：2
* Dropout：0.1
* Classification Head：`Linear(768, 2)`
* Loss：`CE1 + CE2 + Symmetric KL`
* KL reduction：`sum`
* KL weight：1.0
* Optimizer：AdamW
* Learning rate：`5e-5`
* Weight decay：0.01
* Warmup steps：500
* Epochs：3
* Checkpoint interval：每 500 optimizer steps

训练结果：

| Epoch | Total Loss | CE1 | CE2 | Symmetric KL | Train Accuracy | Validation Accuracy |
| ----: | ---------: | ---: | ---: | -----------: | -------------: | ------------------: |
| 1 | 0.954108 | 0.424472 | 0.424696 | 0.104940 | 87.775% | 89.78% |
| 2 | 0.613707 | 0.240865 | 0.249045 | 0.123796 | 95.015% | 92.24% |
| 3 | 0.297055 | 0.103558 | 0.106203 | 0.087293 | 98.285% | **93.48%** |

最佳 Validation Accuracy：**93.48%**

Kaggle Score：**0.93508**

在 Train mode 下，两次 forward 的 logits 存在明显差异，而在 Evaluation mode 下由于 Dropout 被关闭，两次 forward 的 logits 基本完全一致，因此 KL Loss 接近 0。

与普通 BERT 的 Validation Accuracy `93.16%` 和 Kaggle Score `0.93268` 相比，本次 R-Drop 实验分别达到 `93.48%` 和 `0.93508`。由于不同实验没有严格固定完全相同的数据划分和随机状态，因此该差异不能视为严格的单变量对照结论。

---

## Part 16：DeBERTa-v2-XXLarge + Unsloth + LoRA

在 DeBERTa 与 PEFT 实验的基础上，进一步使用 **Unsloth + LoRA** 对更大的 `microsoft/deberta-v2-xxlarge` 进行参数高效微调。

此前已经验证，DeBERTa-v2-XXLarge 约包含 1.57B 参数，在 RTX 4060 Laptop 8GB GPU 上进行 Full Fine-tuning 时，即使降低 sequence length，也会在第一次 AdamW `optimizer.step()` 创建大规模 optimizer states 时发生 CUDA OOM。

本实验通过：

* BF16
* LoRA
* Gradient Checkpointing
* Gradient Accumulation
* Unsloth

共同降低训练显存需求，并成功在同一张 8GB GPU 上完成 DeBERTa-v2-XXLarge 的 3 epochs Fine-tuning。

主要流程：

`IMDb Review → DeBERTa-v2-XXLarge → Unsloth + LoRA → Classification Head → Sentiment`

### Model 配置

* Base Model：`microsoft/deberta-v2-xxlarge`
* Encoder layers：48
* Hidden size：1,536
* Attention heads：24
* Feedforward dimension：6,144
* Maximum sequence length：512
* Base parameters：1,566,913,538
* LoRA 包装后总参数：1,578,714,628
* Trainable parameters：11,801,090
* Trainable percentage：0.7475%
* Precision：BF16
* Quantization：None
* `load_in_4bit=False`

### LoRA / Unsloth 配置

* Framework：Unsloth
* PEFT Method：LoRA
* LoRA rank：16
* LoRA alpha：32
* LoRA dropout：0
* LoRA adapters：192
* Target modules：
  * `query_proj`
  * `key_proj`
  * `value_proj`
  * Attention output `dense`
* Classification head：Trainable
* Pooler：Trainable
* Gradient checkpointing：`"unsloth"`

### Training 配置

* Train samples：20,000
* Validation samples：5,000
* Test samples：25,000
* Train micro-batch：1
* Evaluation batch size：1
* Gradient accumulation steps：8
* Effective batch size：8
* Optimizer：AdamW
* Learning rate：`2e-5`
* Weight decay：0.001
* Warmup steps：10
* Scheduler：Linear
* Epochs：3
* Checkpoint interval：每 500 optimizer steps

训练结果：

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Global Step |
| ----: | ---------: | -------------: | --------------: | ------------------: | ----------: |
| 1 | 0.195246 | 93.115% | 0.165719 | 96.24% | 2,500 |
| 2 | 0.129985 | 96.660% | 0.162717 | 96.26% | 5,000 |
| 3 | 0.115264 | 97.065% | 0.151031 | **96.44%** | 7,500 |

最佳 Validation Accuracy：**96.44%**

Kaggle Score：**0.96868**

训练时间：约 **8 小时 35 分钟**

正式训练 Peak Allocated VRAM：约 **3.40 GiB**

训练过程中：

* 无 CUDA OOM
* 无 NaN
* 无 Inf
* Full forward / backward / optimizer step 均正常
* Checkpoint resume 验证通过

该结果表明，在当前硬件与实验配置下，使用 LoRA 将可训练参数降低至约 **0.7475%**，并配合 BF16、Gradient Checkpointing 与 Unsloth 后，可以在 RTX 4060 Laptop 8GB GPU 上完成此前 Full Fine-tuning 无法完成的 DeBERTa-v2-XXLarge 微调。

需要说明的是，Windows 环境下当前 Unsloth 版本的 DeBERTa Transformer 编译快速路径存在兼容性问题，因此该快速编译路径被禁用；Unsloth 模型加载、Gradient Checkpointing 与 LoRA 训练仍正常工作。因此本实验可以说明该方案在显存和训练可行性上的有效性，但不能用于证明完整 Unsloth 编译加速路径所能达到的速度提升。

---

## Kaggle 实验结果

目前所有有效模型生成的 CSV 文件均已提交至 Kaggle **“Bag of Words Meets Bags of Popcorn”** 竞赛。

| 方法 | Kaggle Score |
| --- | ---: |
| Bag of Words + Random Forest | 0.84432 |
| Word2Vec Average Vectors + Random Forest | 0.82792 |
| Word2Vec Bag of Centroids + Random Forest | 0.84768 |
| GloVe + CNN | 0.78600 |
| GloVe + LSTM | 0.88660 |
| GloVe + GRU | 0.64148 |
| GloVe + Temporal Attention-LSTM | 0.88588 |
| Transformer Encoder | 0.80772 |
| BERT-base-uncased Fine-tuning | 0.93268 |
| BERT-base + SCL | 0.89876 |
| BERT-base + R-Drop | 0.93508 |
| RoBERTa-large Full Fine-tuning | 0.9635 |
| DeBERTa-Large Full Fine-tuning | 0.9603 |
| DeBERTa-Large + LoRA | 0.96396 |
| DeBERTa-Large + AdaLoRA | 0.95568 |
| DeBERTa-Large + Prefix Tuning | 0.95884 |
| DeBERTa-Large + P-Tuning | 0.96032 |
| **DeBERTa-v2-XXLarge + Unsloth + LoRA** | **0.96868** |

在当前实验配置下，**DeBERTa-v2-XXLarge + Unsloth + LoRA** 获得了目前最高的 Kaggle Score：**0.96868**。

---

## 新增模型实验结果

在完成基础 BERT Fine-tuning 后，进一步实验了 RoBERTa、DeBERTa、Supervised Contrastive Learning、R-Drop，以及基于 DeBERTa 的多种 Parameter-Efficient Fine-Tuning 方法。

| 方法 | Best Validation Accuracy | Kaggle Score |
| --- | ---: | ---: |
| BERT-base + SCL | 90.52% | 0.89876 |
| BERT-base + R-Drop | 93.48% | 0.93508 |
| RoBERTa-large Full Fine-tuning | 96.18% | 0.9635 |
| DeBERTa-Large Full Fine-tuning | 95.74% | 0.9603 |
| DeBERTa-Large + LoRA | 96.08% | 0.96396 |
| DeBERTa-Large + AdaLoRA | 94.96% | 0.95568 |
| DeBERTa-Large + Prefix Tuning | 95.26% | 0.95884 |
| DeBERTa-Large + P-Tuning | 95.58% | 0.96032 |
| **DeBERTa-v2-XXLarge + Unsloth + LoRA** | **96.44%** | **0.96868** |

### 当前实验观察

在 BERT-base 系列实验中：

* 普通 BERT Fine-tuning：Validation `93.16%`，Kaggle `0.93268`
* BERT + SCL：Validation `90.52%`，Kaggle `0.89876`
* BERT + R-Drop：Validation `93.48%`，Kaggle `0.93508`

在当前配置下，R-Drop 的结果高于普通 BERT，而当前 SCL 配置低于普通 BERT。由于这些实验没有严格固定完全相同的数据划分和随机状态，因此不能将差异视为严格的单变量方法比较。

在 PEFT 实验中，DeBERTa-Large + LoRA 仅训练约 **0.6419%** 的模型参数，即获得了 `0.96396` 的 Kaggle Score，体现了 Parameter-Efficient Fine-Tuning 在大型预训练模型下游适配中的参数效率。

进一步将 Base Model 扩展到约 1.57B 参数的 DeBERTa-v2-XXLarge 后，Full Fine-tuning 在 RTX 4060 Laptop 8GB 上会因 optimizer states 导致 CUDA OOM；而使用 **Unsloth + LoRA + BF16 + Gradient Checkpointing** 后，仅训练约 **0.7475%** 的参数即可完成完整 3 epochs Fine-tuning，并取得：

* Best Validation Accuracy：**96.44%**
* Kaggle Score：**0.96868**
* Peak Allocated VRAM：约 **3.40 GiB**

因此，在当前所有实验配置下：

**DeBERTa-v2-XXLarge + Unsloth + LoRA 获得了目前最高的 Kaggle Score：0.96868。**

需要注意，DeBERTa-Large + LoRA 与 DeBERTa-v2-XXLarge + Unsloth + LoRA 使用的 Base Model 不同，因此不能将二者之间的性能差异单独归因于 Unsloth。Unsloth 在本实验中的主要意义是与 LoRA、Gradient Checkpointing 等技术共同实现对更大模型的显存高效训练。
