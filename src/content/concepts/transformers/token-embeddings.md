---
title: 'Token Embeddings'
description: 'How tokens are mapped into dense numerical vectors.'
pubDate: 'Aug 1 2026'
order: 2
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Embeddings give tokens meaning by mapping them into dense vectors. Higher dimensions can capture richer relationships, but at a computational cost. Modern GPT models typically use embedding sizes from 768 to 12,288 dimensions.

## Word2Vec

Early embedding methods used Word2Vec. Embeddings were learned from a classification task: train a neural network to predict whether two words commonly appear together. The network takes two words and predicts `1` if they co-occur, or `0` if they do not.

**Example sentence:** *i want to learn machine learning and human ethics.*

### Training samples

| Word 1 | Word 2 | Target |
| ------ | ------ | ------ |
| i | want | 1 |
| want | to | 1 |
| to | learn | 1 |
| learn | machine | 1 |
| i | to | 1 |
| want | learn | 1 |
| to | machine | 1 |
| i | machine | 0 |
| want | machine | 0 |
| i | learn | 0 |

It helps to include random negative examples. Each word starts with random token weights. Over training, the network updates those weights into useful embeddings.

![Word2Vec](image-3.png)

## Embedding techniques

There are several families of embedding methods beyond Word2Vec. The diagram below compares common approaches.

![Embedding Techniques](image-4.png)
