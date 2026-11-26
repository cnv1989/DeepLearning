import torch

from slm.model import GPT, GPTConfig, KVCache


def tiny_model(**overrides):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                       **overrides)
    return GPT(config).eval()


def test_output_shapes_and_loss():
    model = tiny_model()
    tokens = torch.randint(0, 50, (3, 16))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (3, 16, 50)
    assert loss.item() > 0


def test_ignored_targets_do_not_count():
    model = tiny_model()
    tokens = torch.randint(0, 50, (2, 10))
    targets = tokens.clone()
    targets[:, :5] = -100
    _, masked_loss = model(tokens, targets)
    logits, _ = model(tokens)
    expected = torch.nn.functional.cross_entropy(logits[:, 5:].reshape(-1, 50), tokens[:, 5:].reshape(-1))
    assert torch.allclose(masked_loss, expected, atol=1e-5)


def test_kv_cache_matches_full_forward():
    """Feeding tokens one at a time with a cache must give the same logits as all at once."""
    model = tiny_model()
    tokens = torch.randint(0, 50, (2, 12))
    full_logits, _ = model(tokens)

    cache = KVCache(model.config.n_layer)
    prefix_logits, _ = model(tokens[:, :5], cache=cache)
    step_logits = [prefix_logits]
    for i in range(5, 12):
        logits, _ = model(tokens[:, i : i + 1], cache=cache)
        step_logits.append(logits)
    assert torch.allclose(full_logits, torch.cat(step_logits, dim=1), atol=1e-5)


def test_model_is_causal():
    """Changing a later token must not change predictions at earlier positions."""
    model = tiny_model()
    tokens = torch.randint(0, 50, (1, 10))
    changed = tokens.clone()
    changed[0, 7] = (changed[0, 7] + 1) % 50
    a, _ = model(tokens)
    b, _ = model(changed)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-6)
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_generate_stops_at_eos_and_slides_window():
    model = tiny_model()
    prompt = torch.randint(0, 50, (2, 4))
    long_output = model.generate(prompt, max_new_tokens=60, temperature=1.0)  # beyond block_size
    assert long_output.shape == (2, 64)

    greedy = model.generate(prompt, max_new_tokens=5, temperature=0)
    eos = greedy[0, 4].item()  # make the first greedy token the "EOS"
    stopped = model.generate(prompt[:1], max_new_tokens=5, temperature=0, eos_id=eos)
    assert stopped.shape == (1, 5)


def test_activation_checkpointing_gives_same_gradients():
    tokens = torch.randint(0, 50, (2, 8))
    grads = []
    for use_checkpointing in (False, True):
        model = tiny_model().train()
        model.activation_checkpointing = use_checkpointing
        _, loss = model(tokens, tokens)
        loss.backward()
        grads.append(model.token_embedding.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
