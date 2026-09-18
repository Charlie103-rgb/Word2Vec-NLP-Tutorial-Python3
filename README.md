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

| Learning Rate | Training Status          | Best Validation Accuracy |
| ------------: | ------------------------ | -----------------------: |
|           0.8 | NaN / 数值失稳               |                    无有效结果 |
|          0.01 | Stable                   |               **65.11%** |
|         0.001 | Stable，但 10 epochs 内收敛较慢 |                   55.66% |

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
|          0.05 |                   50.96% | 25,000 / 0      | 单类别退化           |
|         0.001 |               **85.20%** | 11,117 / 13,883 | Stable          |

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
|     1 |   0.395091 |         86.99% |              90.00% |
|     2 |   0.225738 |         94.85% |              92.52% |
|     3 |   0.087335 |         98.36% |          **93.16%** |

最佳 Validation Accuracy：**93.16%**

Kaggle Score：**0.93268**

---

## Kaggle 实验结果

目前所有有效模型生成的 CSV 文件均已提交至 Kaggle **“Bag of Words Meets Bags of Popcorn”** 竞赛。

| 方法                                        | Kaggle Score |
| ----------------------------------------- | -----------: |
| Bag of Words + Random Forest              |      0.84432 |
| Word2Vec Average Vectors + Random Forest  |      0.82792 |
| Word2Vec Bag of Centroids + Random Forest |      0.84768 |
| GloVe + CNN                               |      0.78600 |
| GloVe + LSTM                              |      0.88660 |
| GloVe + GRU                               |      0.64148 |
| GloVe + Temporal Attention-LSTM           |      0.88588 |
| Transformer Encoder                       |      0.80772 |
| BERT-base-uncased Fine-tuning             |  **0.93268** |

在当前实验配置下，**BERT-base-uncased Fine-tuning** 获得了最高的 Kaggle Score：**0.93268**。
