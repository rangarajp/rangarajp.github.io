---
title: 'Tokenization'
description: 'How raw text is converted into model-friendly tokens.'
pubDate: 'Aug 1 2026'
order: 1
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Tokens are the units a model uses for both inputs and outputs. When you send text to a model, it is first broken into tokens.

You can try this interactively with the [OpenAI tokenizer](https://platform.openai.com/tokenizer).

![Tokenization from OpenAI](./images/tokenization-openai-example.png)

As a rough rule of thumb:

- 1 token ≈ 4 characters ≈ ¾ of a word
- 100 tokens ≈ 75 words

Tokens can be:

1. Complete words — `want`, `learn`
2. Parts of words — `token`, `ization`
3. Punctuation — often its own token

## Tokenizer approaches

### 1. Word tokens

The earliest approach: one token per word.

**Challenge:** new or rare words cannot be represented well. Related forms like `problem` and `problematic` get entirely different tokens, which drives vocabulary size up.

```
Input:  i want to learn ML
Tokens: "i", "want", "to", "learn", "ML"
```

### 2. Subword tokens

Subword tokenization can represent unseen words by composing known pieces. Frequent words stay intact; rare words break into recognizable roots. This also reduces unknown-token problems.

```
Input:  i want to learn ML nicely
Tokens: "i", "want", "to", "learn", "ML", "nic", "ely"
```

In practice:

- Frequent words stay as a single token
- Rare words split into recognizable roots
- Unknown words become less of a problem

### 3. Character tokens

Every word is split into letters. The vocabulary is tiny (just the alphabet and symbols), but you lose a lot of word-level context.

```
Input:  i want to learn ML
Tokens: "i", "w", "a", "n", "t", ...
```

![Tokenizer Algorithms](./images/tokenization-approaches-algorithms.png)

### 4. Byte Pair Encoding (BPE)

BPE is a subword method inspired by data compression. It iteratively merges the most frequent pairs of consecutive characters in a corpus until a target vocabulary size is reached.

#### Steps

**Sample corpus:** `"ab"`, `"bc"`, `"bcd"`, `"cde"`

1. **Initialize** — split into individual characters:

   ```
   {"a", "b", "c", "d", "e"}
   ```

2. **Count character frequencies:**

   ```
   {"a": 1, "b": 3, "c": 3, "d": 2, "e": 1}
   ```

3. **Count adjacent pairs:**

   ```
   {"ab": 1, "bc": 2, "cd": 2, "de": 1}
   ```

   `"bc"` and `"cd"` both appear twice. Merge `"bc"` first (tie-break; either pair is valid).

4. **Merge into a new subword:**

   ```
   Vocabulary: {"a", "b", "c", "d", "e", "bc"}
   Counts:     {"a": 1, "b": 1, "c": 1, "d": 2, "e": 1, "bc": 2}
   ```

   Frequencies of `b` and `c` drop because those occurrences were absorbed into `"bc"`.

5. **Repeat** until the vocabulary reaches the desired size

6. **Final subword units** might look like:

   ```
   {"a", "b", "c", "d", "e", "bc", "cd", "de", "ab", "bcd", "cde"}
   ```

7. **Encode the original corpus:**

   | Text | Tokens |
   | ---- | ------ |
   | `ab` | `"a"` + `"b"` |
   | `bc` | `"bc"` |
   | `bcd` | `"bc"` + `"d"` |
   | `cde` | `"c"` + `"de"` |

#### Encoding new text

When new text arrives, the tokenizer applies the learned merge rules in order. It prefers the longest matching tokens already in the vocabulary. Anything left unmatched is broken down into fundamental byte representations.
