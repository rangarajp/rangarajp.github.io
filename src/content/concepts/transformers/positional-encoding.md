---
title: 'Positional Encoding'
description: 'How sequence order information is injected into token representations.'
pubDate: 'Aug 1 2026'
order: 3
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

## Positional Encoding

In english language, order of words are important. Ex: Dog chased the cat Vs Cat chased the dog. Completely different sentences but transformer architecture processesing them all token in parallel could consider same. So there is a need to include positional awareness. 

In short, self attention knows relationship between tokens but not order. Attention asks which token is relevant to other token and not where is that token
Attention answers what
Position Encoding anwers where

Positional Embeddings - inform word order and contextual relationships. Now with that let us think of some what ideally we want in them
1. different positions get different representation
2. nearby ones are similar and meaningful
3. representation stay numerically bound even for large position
4. recovery - if i know PE(k) i should get PE(k+10)

There are few variations

### 1. Naive Representation

Why cant we add absolute position number. 

    I       → embedding + 1
    love    → embedding + 2
    AI      → embedding + 3

Few problems, the numbers can get large and no relative relationship captured. Model should ideally capture like previous token, 5 token earlier, etc


### 2. Sinusoidal Position Encoding

2.1 Before sine, why not polynomial or exponential. They get quickly out of bounds, we want something that changes with position without increasing in magnitude.

2.2 Sine & cos always stay bounded between -1 and 1. But one sine or cos cant distinguish all because sine curve repeats after 360. Basically it cant represent or uniquely distinguish all positions.
Let us say, we think

    [sin(pos),sin(pos/10),sin(pos/100)] analgous to [ second hand, minute hand, hour hand]
    Second hand alone is not enough because it repeats every 60 seconds

So, let us combine it all three - seconds + minutes + hours = unique and richer representation of time and doesnt repeat. Now sinusoidal encoding creates hundreds of these clock hands.

2.3 Now, let us bring the equation

![Sinusoidal Encoding](position_encoding_base_formula.png)

| Variable    | Meaning                       |
| ----------- | ----------------------------- |
| (pos)       | token position: 0, 1, 2, 3... |
| (i)         | embedding dimension index     |
| (d_{model}) | embedding dimension : 768 etc |

for simplicity, let us assume
    pe (pos,  2i) = sin (pos * wi)
    pe (pos, 2i + 1) = cos (pos * wi)

Ex : assume dimension = 4, 
for i = 0,  
    wi = 1000^(0/4) = 1
    dim 0, dim 1 = sin(pos), cos(pos)

for i = 1,
    wi = 100
    dim 2, dim 3 = sin(pos/100), cos(pos/100)

    PE(pos)=[sin(pos),cos(pos),sin(pos/100),cos(pos/100)]
    PE(1) = [0.841,0.540,0.010,0.999]
    PE(2) = [0.909,−0.416,0.020,0.9998]
    PE(3) = [0.141,−0.990,0.030,0.9995]

first dimension change rapidly, but later dimension change slowly [0.01, 0.02, 0.03]
Sine curve causes different dimension, i to oscillate differently. Like dim1 chage rapidly, dim 2 less and so on.

2.4 Sine and cos create the linear transformation in going from k to p+k.
    sin(p + k) can be represented only from sin(p) sin(k)
    sin(p+k) = sin(p)cos(k)+cos(p)sin(k)
This also explains we need both cos and sin to represent the linear relationship

2.5 Take example, "Cat ate my fish", elsewhere "Yesterday, the cat ate my fish". In both different sentences, the cat and ate have similar relative relationship, with sin representation this give same positional information rather giving distinct unique ID

#### In short
        Position
        │
        ▼
        Represent it as many clocks
        │
        ├── very fast clock
        ├── fast clock
        ├── medium clock
        ├── slow clock
        └── very slow clock
                │
                ▼
        Each clock represented by
            [sin(angle), cos(angle)]
                │
                ▼
        Combined values give a
        positional fingerprint

    Why a sin/cos pair?

        Because it represents rotation, stays bounded, and shifting position by k becomes a simple linear transformation depending on k.

    Why many frequencies?

        Because one clock repeats; many clocks capture position across different distance scales.

    Why 10000?

        It spreads those frequencies over a wide range of wavelengths; 10000 itself isn't sacred.

    Why efficient?

        No learned parameters, cheap to precompute, simple addition to embeddings, same dimensionality, and mathematically structured relative-position information.
    

