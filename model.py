"""A small Transformer in pure JAX. Bidirectional -> masked-diffusion denoiser;
causal -> autoregressive baseline. Params are a plain dict pytree."""
import jax, jax.numpy as jnp

def init_params(key, vocab, d=96, n_layers=3, n_heads=4, d_ff=192, max_len=48):
    ks = jax.random.split(key, 4 + 6 * n_layers)
    s = lambda *sh: 0.02
    p = {
        "emb":  jax.random.normal(ks[0], (vocab, d)) * s(),
        "pos":  jax.random.normal(ks[1], (max_len, d)) * s(),
        "lnf_g": jnp.ones((d,)), "lnf_b": jnp.zeros((d,)),
        "head": jax.random.normal(ks[2], (d, vocab)) * s(),
        "head_b": jnp.zeros((vocab,)),
        "layers": [],
    }
    for i in range(n_layers):
        k = ks[4 + 6 * i: 4 + 6 * (i + 1)]
        p["layers"].append({
            "ln1_g": jnp.ones((d,)), "ln1_b": jnp.zeros((d,)),
            "wqkv": jax.random.normal(k[0], (d, 3 * d)) * s(),
            "wo":   jax.random.normal(k[1], (d, d)) * s(),
            "ln2_g": jnp.ones((d,)), "ln2_b": jnp.zeros((d,)),
            "w1": jax.random.normal(k[2], (d, d_ff)) * s(), "b1": jnp.zeros((d_ff,)),
            "w2": jax.random.normal(k[3], (d_ff, d)) * s(), "b2": jnp.zeros((d,)),
        })
    return p

def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(v + 1e-5) * g + b

N_HEADS = 4

def forward(params, tokens, causal: bool):
    """tokens: (B, L) int32 -> logits (B, L, V)."""
    d = params["emb"].shape[1]; h = N_HEADS; hd = d // h
    B, Lq = tokens.shape
    x = params["emb"][tokens] + params["pos"][:Lq][None]
    if causal:
        attn_bias = jnp.where(jnp.tril(jnp.ones((Lq, Lq), bool)), 0.0, -1e9)
    else:
        attn_bias = jnp.zeros((Lq, Lq))
    for lyr in params["layers"]:
        hx = _ln(x, lyr["ln1_g"], lyr["ln1_b"])
        qkv = hx @ lyr["wqkv"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        def sp(t):  # (B,L,d)->(B,h,L,hd)
            return t.reshape(B, Lq, h, hd).transpose(0, 2, 1, 3)
        q, k, v = sp(q), sp(k), sp(v)
        att = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(hd) + attn_bias
        att = jax.nn.softmax(att, axis=-1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, Lq, d)
        x = x + o @ lyr["wo"]
        hx = _ln(x, lyr["ln2_g"], lyr["ln2_b"])
        x = x + jax.nn.gelu(hx @ lyr["w1"] + lyr["b1"]) @ lyr["w2"] + lyr["b2"]
    x = _ln(x, params["lnf_g"], params["lnf_b"])
    return x @ params["head"] + params["head_b"]

def masked_ce(params, noisy_tokens, targets, loss_mask, causal=False):
    """Cross-entropy on positions where loss_mask==1. For the denoiser, logits at
    position p predict the clean token at p. For the AR model, logits at p-1
    predict p (callers pass pre-shifted arrays)."""
    logits = forward(params, noisy_tokens, causal)
    logp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(logp, targets[..., None], -1)[..., 0]
    return (nll * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)
