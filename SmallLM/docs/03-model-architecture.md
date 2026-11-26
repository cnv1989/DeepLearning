# 3. The model: a transformer

**Code:** [`slm/model.py`](../slm/model.py) · **Tests:** [`tests/test_model.py`](../tests/test_model.py)

## What it is

The model is a **decoder-only transformer**, the same family as GPT and Llama.
Given a sequence of tokens, it outputs, *for every position*, a probability for
each possible next token.

```
 token ids        [ROM][EO][:][\n]
     │
 embedding        look up a learned vector for each id         (batch, seq, width)
     │
 Block 1  ┐
 Block 2  │       each block mixes information between positions,
   ...    │       then transforms each position separately
 Block N  ┘
     │
 RMSNorm
     │
 lm_head          one score ("logit") per vocabulary entry      (batch, seq, vocab)
     │
 softmax          probabilities for the next token at every position
```

## Shapes, with the `tiny` config

`n_embd`=128, `n_head`=4, `n_layer`=4, `block_size`=128, vocabulary 66.

| Tensor | Shape | Meaning |
|---|---|---|
| input ids | (32, 128) | 32 sequences of 128 tokens |
| after embedding | (32, 128, 128) | a 128-number vector per token |
| inside attention | (32, 4, 128, 32) | 4 heads, each with 32-number vectors |
| logits | (32, 128, 66) | 66 scores at each of 128 positions |

## The block

Every block does two things, each wrapped in a **residual connection** (the
`x = x + ...`):

```python
x = x + Attention(RMSNorm(x))     # positions exchange information
x = x + FeedForward(RMSNorm(x))   # each position is processed on its own
```

The residual connection means each block only adds a *correction* to a running
"residual stream". Gradients can flow straight through the additions, which is
what makes deep stacks trainable. Normalizing *before* each sub-layer
("pre-norm") keeps training stable.

## Attention: how positions talk to each other

For each position, attention computes three vectors from its hidden state:

* **Query**: what am I looking for?
* **Key**: what do I contain?
* **Value**: what will I pass on if someone attends to me?

Each position compares its query with every key (a dot product). The scores go
through a softmax to become weights, and the output is the weighted average of
the values:

```
attention(Q, K, V) = softmax(Q·Kᵀ / √head_dim) · V
```

**Causal mask.** A language model must not peek at the future: position 5 may
only look at positions 0–5. `is_causal=True` sets the scores for later
positions to −∞ before the softmax. A test (`test_model_is_causal`) checks that
changing a later token never changes earlier predictions.

**Multiple heads.** The width is split into `n_head` independent heads (4 × 32
here). Each can learn a different pattern: one tracks the previous word,
another matches quotation marks, and so on.

`F.scaled_dot_product_attention` is PyTorch's fused version of the formula. On
GPUs it uses FlashAttention, which is much faster and never stores the full
(seq × seq) score matrix.

## The modern components

The original GPT-2 design used LayerNorm, learned position vectors and a GELU
MLP. This model uses the replacements found in Llama, Mistral and most
current open models. On Shakespeare at the same size and number of steps,
switching lowered validation loss from 1.91 to 1.59.

### RMSNorm (`RMSNorm`)

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · gain
```

Rescales each vector to a standard size. Unlike LayerNorm it doesn't subtract
the mean or add a bias: it's cheaper and works just as well. The calculation is
done in float32 for numerical safety, even under mixed precision.

### Rotary position embeddings, RoPE (`rotary_tables`, `apply_rotary`)

Attention by itself has no idea of order: "dog bites man" and "man bites dog"
look the same. Something must encode position.

RoPE **rotates** each query and key by an angle proportional to its position.
Features are treated as pairs, each pair a point on a 2D plane, and each pair
rotates at a different speed (fast pairs capture nearby detail, slow pairs
long-range position). The key property: the dot product of a query at position
*m* with a key at position *n* depends only on **m − n**, the relative
distance. That's what attention needs ("the word 2 back"), and it adds no
parameters. The angle tables are precomputed once for all `block_size`
positions.

### SwiGLU feed-forward (`FeedForward`)

```
FeedForward(x) = W_out · ( silu(W_gate · x) ⊙ (W_up · x) )
```

A *gated* layer: one projection decides how much of the other to let through,
element by element. It consistently trains better than a plain GELU MLP. The
hidden size is 8/3 × width, rounded up to a multiple of 64 (384 for width
128). That keeps the parameter count equal to the classic 4 × width MLP,
because there are three matrices instead of two.

### Grouped-query attention, GQA (`n_kv_head`)

Normally each query head has its own key and value head. With GQA, several
query heads **share** one key/value head:

```
n_head = 6, n_kv_head = 2:   queries  Q1 Q2 Q3 | Q4 Q5 Q6
                             keys     K1       | K2
```

Quality barely changes, but the keys and values that must be stored during
generation (the KV cache, see [Generation](04-generation.md)) shrink by
`n_head / n_kv_head`, 3× here. Large models rely on this to serve long
contexts. `n_kv_head = 0` means no grouping. In the code, the shared heads are
copied with `repeat_interleave` so the attention call stays simple.

## Weight tying

The embedding (id → vector) and `lm_head` (vector → score per id) are the same
matrix, transposed. The intuition: a token's input meaning and "how much this
vector looks like that token" are closely related. It saves `vocab × width`
parameters and helps small models.

## Initialization (`_init_weights`)

Weights start as small random numbers (normal, std 0.02). The output
projections of each sub-layer (`out_proj`) are made smaller still, by
1/√(2·n_layer). Every block adds two contributions to the residual stream, so
this keeps the stream's size steady however many blocks there are, and
training starts smoothly. A well-initialized model starts with loss ≈
ln(vocab_size): for 66 tokens, ln 66 ≈ 4.19 (the training log shows 4.25),
meaning "every token equally likely".

## Parameter count

For `tiny` with characters, 0.86M parameters in total:

| Part | Calculation | Parameters |
|---|---|---|
| Embedding (shared with `lm_head`) | 66 × 128 | 8,448 |
| Attention, per block | 4 × 128 × 128 | 65,536 |
| Feed-forward, per block | 3 × 128 × 384 | 147,456 |
| Norms, per block | 2 × 128 | 256 |
| × 4 blocks + final norm | | 852,992 + 128 |

## Activation checkpointing

Training stores every layer's intermediate results ("activations") for the
backward pass. With `--activation-checkpointing`, each block's activations are
thrown away and recomputed during backward. That costs roughly 30% more compute
but much less memory, so you can fit a bigger batch or model. A test checks
the gradients are identical either way.

## Settings (`GPTConfig`)

| Setting | Effect of raising it |
|---|---|
| `n_layer` | deeper: more steps of reasoning per token; slower |
| `n_embd` | wider: more capacity per token; parameters grow with the square |
| `n_head` | more, smaller heads (must divide `n_embd`) |
| `n_kv_head` | less key/value sharing (must divide `n_head`) |
| `block_size` | longer context; attention cost grows with the square |
| `dropout` | more regularization; useful when data is small |

A balanced shape matters more than any single setting: double the width and
depth together, rather than one by 8×.

## Try it

* Set `--n-kv-head 1` (all query heads share one key/value head) and compare
  validation loss with the default.
* Read `test_kv_cache_matches_full_forward` and `test_model_is_causal` in
  `tests/test_model.py`: they state exactly what the model must guarantee.
