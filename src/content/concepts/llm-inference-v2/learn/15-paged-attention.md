---
title: "PagedAttention"
description: "Logical token blocks, physical allocation, sharing, and fragmentation."
section: learn
pubDate: 2026-10-29
part: 3
chapter: 15
order: 15
tags: ["llm-inference", "systems", "hands-on"]
---

# PagedAttention

A contiguous reservation sized for every request's maximum answer wastes memory when answers stop early. PagedAttention separates a sequence's logical token positions from the physical blocks holding its KV tensors.

## 1. Logical and physical blocks

Divide the sequence into fixed-size token blocks. A block table maps logical block indices to physical blocks. An attention kernel uses that table to find the correct keys and values even when blocks are scattered in memory.

```python
# CPU-only address translation; not an attention implementation.
block_size = 16
block_table = [7, 2, 11]
for token in (0, 15, 16, 34):
    logical, offset = divmod(token, block_size)
    physical = block_table[logical]
    print(token, '-> physical block', physical, 'offset', offset)
length = 35
allocated = ((length + block_size - 1) // block_size) * block_size
print('unused token slots:', allocated - length)  # 13
```

Only the final block needs unused slots in this simplified layout. More sequences create more partially filled tails. Smaller blocks reduce tail waste but increase tables and allocation work.

## 2. Sharing and copy-on-write

A shared prefix can reference the same physical blocks with reference counts. If a branch needs to modify a shared partially filled block, copy-on-write preserves the other branch's state. Prefix reuse, branching generation, and eviction all need correct ownership; freeing a block still used by another request corrupts outputs.

## 3. What paging changes

Paging improves allocation and sharing. It does not reduce the mathematical cost of dense attention or compress each stored element. FlashAttention addresses attention IO; quantization addresses representation size. These techniques can complement one another.

Primary reading: [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180).
