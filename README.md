# Word2Vec NLP Tutorial - 复现与学习

本仓库记录了对 Kaggle **“Bag of Words Meets Bags of Popcorn”** NLP 教程及经典深度学习文本分类方法的学习与复现过程。

项目以 **IMDb 电影评论情感分类** 为统一任务，从传统的 Bag of Words 文本表示开始，逐步学习 Word2Vec、GloVe 等经典词向量方法，并进一步使用 CNN、LSTM 和 GRU 进行文本情感分类。

实验结果均通过 Kaggle **“Bag of Words Meets Bags of Popcorn”** 竞赛进行测试。

---

## 项目内容

### Part 1：Bag of Words

对电影评论进行文本清洗和预处理，使用 Bag of Words（词袋模型）将文本转换为固定维度的词频向量，并使用 Random Forest 完成情感二分类。

主要流程：

`IMDb Review → Text Cleaning → Bag of Words → Random Forest → Sentiment`

---

### Part 2：Word2Vec Average Vectors

使用 Word2Vec 学习词的分布式向量表示，并通过对一条评论中所有词向量求平均，将不同长度的文本转换为固定的 300 维向量，最后使用 Random Forest 进行情感分类。

主要流程：

`IMDb Review → Word2Vec → Average Word Vectors → Random Forest → Sentiment`

---

### Part 3：Word2Vec Clustering / Bag of Centroids

使用 K-Means 对 Word2Vec 词向量进行聚类，将语义相近的词划分到同一 cluster，并使用 Bag of Centroids 将每条评论表示为基于语义词簇的固定维度向量，再使用 Random Forest 完成分类。

主要流程：

`IMDb Review → Word2Vec → K-Means → Bag of Centroids → Random Forest → Sentiment`

---

### Part 4：GloVe + Neural Networks

在 Word2Vec 实验的基础上，进一步学习经典的预训练词向量 **GloVe（Global Vectors for Word Representation）** 以及神经网络文本分类模型。

实验使用 Stanford GloVe **Common Crawl 840B 300d** 预训练词向量，将 IMDb 评论中的单词映射为 300 维词向量，并分别使用 CNN、LSTM 和 GRU 完成情感二分类。

由于 GloVe Common Crawl 840B 文件体积较大，本仓库不上传原始 GloVe 模型文件。

#### CNN

使用冻结的 GloVe 词向量作为 Embedding，通过一维卷积提取文本中的局部模式，再经过 Global Max Pooling 和全连接层完成分类。

主要流程：

`IMDb Review → GloVe Embedding → Conv1D → ReLU → Global Max Pooling → Linear → Sentiment`

实验配置：

- GloVe Embedding：300d
- Embedding：Frozen
- Conv1D filters：128
- Kernel size：3
- Optimizer：SGD
- Learning rate：0.8
- Epochs：10
- Batch size：64

最佳 Validation Accuracy：**80.46%**

Kaggle Score：**0.78600**

---

#### LSTM

使用双向 LSTM 对 GloVe 词向量序列进行建模，以学习文本中的顺序信息和较长距离的上下文依赖关系。

主要流程：

`IMDb Review → GloVe Embedding → BiLSTM → Feature Concatenation → Linear → Sentiment`

实验配置：

- GloVe Embedding：300d
- Embedding：Frozen
- Hidden size：120
- LSTM layers：2
- Bidirectional：True
- Batch size：64
- Epochs：10
- Optimizer：Adam

原始代码使用 `lr=0.05`，训练过程中出现数值不稳定（NaN）。在保持其他网络结构和训练设置不变的情况下，将学习率调整为 `0.001` 后训练恢复稳定。

最佳 Validation Accuracy：**89.36%**

Kaggle Score：**0.88660**

---

#### GRU

使用双向 GRU 对 GloVe 词向量序列进行建模，并学习门控的序列表示。

主要流程：

`IMDb Review → GloVe Embedding → BiGRU → Feature Concatenation → Linear → Sentiment`

实验配置：

- GloVe Embedding：300d
- Embedding：Frozen
- Hidden size：120
- GRU layers：2
- Bidirectional：True
- Batch size：64
- Epochs：10
- Optimizer：SGD

原始代码使用 `lr=0.8`，训练从第一个 epoch 开始出现 NaN。随后进行了单变量学习率实验：

| Learning Rate | Training Status | Best Validation Accuracy |
| ---: | --- | ---: |
| 0.8 | NaN / 数值失稳 | 无有效结果 |
| 0.01 | Stable | **65.11%** |
| 0.001 | Stable，但 10 epochs 内收敛较慢 | 55.66% |

最终使用 `lr=0.01` 的稳定版本生成 Kaggle Submission。

Kaggle Score：**0.64148**

---

## Kaggle 实验结果

目前所有有效模型生成的 CSV 文件均已提交至 Kaggle **“Bag of Words Meets Bags of Popcorn”** 竞赛。

| 方法 | Kaggle Score |
| --- | ---: |
| Bag of Words + Random Forest | 0.84432 |
| Word2Vec Average Vectors + Random Forest | 0.82792 |
| Word2Vec Bag of Centroids + Random Forest | 0.84768 |
| GloVe + CNN | 0.78600 |
| GloVe + LSTM | **0.88660** |
| GloVe + GRU | 0.64148 |

在当前实验配置下，**GloVe + LSTM** 获得了最高的 Kaggle Score：**0.88660**。
