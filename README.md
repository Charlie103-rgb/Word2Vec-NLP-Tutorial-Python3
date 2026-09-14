# Word2Vec NLP Tutorial - 复现与学习

本仓库记录了对 Kaggle **“Bag of Words Meets Bags of Popcorn”** NLP 教程的学习与复现过程。该教程以 IMDb 电影评论情感分类为任务，介绍了自然语言处理（NLP）中 Bag of Words、Word2Vec 以及基于词向量聚类的文本表示方法。本项目的实验均在 Kaggle Notebook 中完成，并成功完成了对应的 Kaggle Submission。

## 项目内容

本项目主要包含三个部分：

- **Part 1：Bag of Words**
  
  对电影评论进行文本清洗和预处理，使用 Bag of Words（词袋模型）将文本转换为固定维度的词频向量，并使用 Random Forest 完成情感二分类。

- **Part 2：Word2Vec Average Vectors**
  
  使用 Word2Vec 学习词的分布式向量表示，并通过对一条评论中所有词向量求平均，将不同长度的文本转换为固定的 300 维向量，最后使用 Random Forest 进行情感分类。

- **Part 3：Word2Vec Clustering / Bag of Centroids**
  
  使用 K-Means 对 Word2Vec 词向量进行聚类，将语义相近的词划分到同一 cluster，并使用 Bag of Centroids 将每条评论表示为基于语义词簇的固定维度向量，再使用 Random Forest 完成分类。

## Kaggle 实验结果

三个模型生成的 CSV 文件均提交至 Kaggle "Bag of Words Meets Bags of Popcorn" 竞赛进行测试，结果如下：

| 方法 | Kaggle Score |
| --- | ---: |
| Bag of Words + Random Forest | 0.84432 |
| Word2Vec Average Vectors + Random Forest | 0.82792 |
| Word2Vec Bag of Centroids + Random Forest | 0.84768 |

在三个实验中，基于 Word2Vec 聚类得到的 Bag of Centroids 方法取得了最高的 Kaggle Score。
