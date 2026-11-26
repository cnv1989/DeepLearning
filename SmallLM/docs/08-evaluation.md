# 8. Evaluation

**Code:** [`evaluate.py`](../evaluate.py)

## What it is

Training loss tells you the model is learning *something*. Evaluation tells you
whether it's learning what you want, on data it has never seen. This repo
measures two things:

```bash
python evaluate.py --ckpt out/ckpt.pt --data data/shakespeare_char        # perplexity
python evaluate.py --ckpt out/sft/ckpt.pt --tasks data/tasks/test.jsonl   # task accuracy
```

## Perplexity (`perplexity`)

How surprised the model is by held-out text. Lower is better.

Unlike the quick estimate printed during training (a few *random* windows),
`evaluate.py` reads the **whole** validation set in order, in non-overlapping
windows of `block_size`, and weights each by its token count. The result is
exact and repeatable, and good for comparing checkpoints.

```
validation: loss 1.5803 | perplexity 4.86 | 111,488 tokens
```

**Comparing perplexities.** Only compare models that use the **same
tokenizer**. A character model predicts ~3× as many, much easier tokens as a
BPE model, so its perplexity is naturally lower. To compare across tokenizers,
convert to **bits per character**: total loss over the text ÷ ln 2 ÷ number of
characters.

## Task accuracy (`task_accuracy`)

For prompts with one correct answer, generate a reply with **greedy decoding**
(always the most likely token, so the score is repeatable) and count exact
matches, overall and per task:

```
  ✓ 'Reverse: brains' -> 'sniarb' (expected 'sniarb')
  ✗ 'Uppercase: rocky' -> 'the heart of the stand of the st' (expected 'ROCKY')
    first letter: 100.0%  (44/44)
          repeat: 100.0%  (53/53)
         reverse: 100.0%  (52/52)
       uppercase: 100.0%  (51/51)
accuracy: 100.0% on 200 examples
```

Always look at the printed examples (`--show`), not just the number: the
failures tell you *how* the model fails (wrong format, never stopping, nearly
right).

## What makes an evaluation trustworthy

* **Held-out data.** Never evaluate on training data. `make_task_data.py` keeps
  test *words* out of training entirely, not just test examples.
* **Enough examples.** With 200 examples, one example is 0.5%, and differences
  of a few percent can be noise. The 23.5% → 22.5% change in the GRPO test
  run is two examples: noise, not a real decline.
* **Fixed decoding.** Greedy decoding gives the same answer every time.
  Sampling would change scores between runs.
* **Test what you care about.** Low perplexity doesn't guarantee good answers,
  and high task accuracy doesn't guarantee fluent text. Use both, and read
  samples.

## Beyond this repo

Real LLMs are measured on standard benchmarks:

| Kind | Examples | How it's scored |
|---|---|---|
| Multiple choice | HellaSwag, MMLU, ARC | pick the option the model gives the highest probability |
| Exact answers | GSM8K (maths), HumanEval (code) | check the final answer, or run the code's tests |
| Open-ended | MT-Bench, chatbot arenas | a strong model or people judge the replies |

Multiple-choice scoring is a good next addition here: for each option, compute
the total log-probability of its tokens given the question (the
`completion_logprob_sums` function in `slm/rl/logprobs.py` already does this),
and see whether the right option scores highest.

## Try it

```bash
python evaluate.py --ckpt out/sft/ckpt.pt --tasks data/tasks/test.jsonl --show 20
```

Run it on checkpoints at different stages (pretrained, short SFT, full SFT) to
see where each skill appears.
