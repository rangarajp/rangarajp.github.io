---
title: 'Transformers'
description: 'From raw text to contextual representations — tokenization through the transformer block.'
pubDate: 'Aug 1 2026'
seriesOrder: 1
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Transformers process text as a sequence of tokens. This book walks that pipeline step by step — from raw characters to the layered block that stacks attention and feed-forward networks.

Read the chapters in order. Each one assumes the previous.

## Contents

1. [Tokenization](./tokenization) — break raw text into tokens the model can consume
2. [Token embeddings](./token-embeddings) — map each token to a dense numerical vector
3. [Positional encoding](./positional-encoding) — inject sequence order into those vectors
4. [Attention](./attention) — let tokens weigh how relevant other tokens are in context
5. [Query, key, and value](./attention_detailed) — the QKV machinery behind attention scores
6. [Multi-head attention](./multi-head-attention) — run multiple attention patterns in parallel
7. [Transformer block](./transformer-block) — residuals, normalization, and the feed-forward layer

Together, these chapters turn a sentence into context-aware representations that downstream layers can reason over.

## Implement it

Ready to code the full encoder–decoder? See [Transformer from Scratch](/build-from-scratch/transformers) — PyTorch modules for config, dataset/masks, model, and training, with a dimension diagram matching this book.
