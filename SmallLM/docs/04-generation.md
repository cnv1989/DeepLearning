# 4. Generation: turning predictions into text

**Code:** `GPT.generate`, `sample_next_token` and `KVCache` in [`slm/model.py`](../slm/model.py); [`sample.py`](../sample.py)

## What it is

The model only ever predicts **one** next token. To write a paragraph, it runs
in a loop: predict a token, append it, predict the next.

```
"ROMEO:"        → model → probabilities → pick "\n"
"ROMEO:\n"      → model → probabilities → pick "W"
"ROMEO:\nW"     → model → probabilities → pick "h"   ... and so on
```

This is called **autoregressive** generation. Two questions decide how it
behaves: *how do we pick* a token from the probabilities, and *how do we avoid
redoing work* on every step?

## Picking the next token (`sample_next_token`)

The model outputs a score per vocabulary entry (logits). Softmax turns them
into probabilities, for example:

```
" the" 0.40   " a" 0.25   " my" 0.15   " thy" 0.10   ...   " zebra" 0.0001
```

| Method | Setting | Behaviour |
|---|---|---|
| Greedy | `temperature 0` | always the most likely token. Repeatable; good for answers with one right reply; often repetitive for creative text |
| Sampling | `temperature 1` | pick at random according to the probabilities. Varied |
| Temperature | `< 1` / `> 1` | divide logits by T before softmax. Below 1 sharpens toward the likeliest tokens (safer); above 1 flattens (more surprising, more mistakes) |
| Top-k | `top_k 50` | only the 50 likeliest tokens are allowed. Cuts off the long tail of nonsense |

`sample.py` defaults to temperature 0.8 and top-k 50 for open text.
`evaluate.py` uses greedy decoding so its scores are repeatable. GRPO samples
at temperature 1 with no top-k, because RL needs the model's true
probabilities.

## Stopping (`eos_id`)

Fine-tuned models end each answer with the end-of-text token. With `eos_id`
set, a sequence stops when it produces EOS. In a batch, sequences that have
stopped are padded with EOS until all are done; later code knows to ignore that
padding.

## The KV cache: avoiding repeated work

Without a cache, generating token 100 means running the model over all 100
tokens, even though tokens 1–99 were processed on the previous step. That's
quadratic work.

The trick: in attention, earlier tokens' **keys and values never change**
(causal masking means they can't see later tokens). So keep them:

```
first pass ("prefill"):  run the whole prompt; store each layer's keys and values
each later step:         run only the newest token; its query attends to
                         [stored keys + its own key]; append its key and value
```

Each new token now costs a single position. The `KVCache` class holds one list
of keys and one of values per layer. `test_kv_cache_matches_full_forward`
checks that feeding tokens one at a time with the cache gives exactly the same
predictions as processing everything at once.

### Details that matter

* **Causal masking with a cache.** In the prefill, positions must not see later
  ones, so the mask is on. When one new token arrives, it may see everything
  cached, so the mask is off. Each layer checks *its own* cache entry to decide.
  Checking the shared cache length was a real bug the tests caught: by the time
  layer 2 ran, layer 1 had already written to the cache.
* **Rotary positions** continue from the cache length, so the new token gets
  the right position (`hidden_states` uses `cache.length` as the start).
* **Grouped-query attention** shrinks the cache: with 6 query heads sharing 2
  key/value heads, it is 3× smaller.

### Running past the context window

The model can only read `block_size` tokens. When generation runs longer,
`generate` slides the window: it keeps the most recent `block_size` tokens and
rebuilds the cache from them. This is slower than the fast path, but correct.

## Chat generation

For fine-tuned models, `generate_reply` in [`slm/sft.py`](../slm/sft.py) wraps
the conversation in the chat template, generates until EOS, and returns just
the reply text:

```bash
python sample.py --ckpt out/sft/ckpt.pt --chat --prompt "Reverse: flower" --temperature 0
rewolf
python sample.py --ckpt out/sft/ckpt.pt --chat          # interactive; empty line quits
```

## Common problems

* **Repetition loops** ("the the the"): greedy decoding on a weak model. Use
  temperature 0.7–0.9 with top-k.
* **Gibberish**: temperature too high, or the model is undertrained.
* **The model never stops**: it was never trained with EOS at the end of
  answers. Fine-tune it (see [SFT](07-supervised-fine-tuning.md)).

## Try it

```bash
python sample.py --prompt "ROMEO:" --temperature 0.3 --seed 1
python sample.py --prompt "ROMEO:" --temperature 1.5 --seed 1
python sample.py --prompt "ROMEO:" --temperature 0
```
