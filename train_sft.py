"""SFT: masked-diffusion denoiser + AR baseline on expert snapshots.
Usage: python3 train_sft.py <seed> <outdir> [steps] [batch]"""
import sys, os, time, pickle
import numpy as np
import jax, jax.numpy as jnp
import optax
import toolflow as tf
from model import init_params, masked_ce
from data import build_dataset

def make_train_state(seed, lr, steps):
    import os
    MD = int(os.environ.get("MODEL_D", "96"))
    ML = int(os.environ.get("MODEL_L", "3"))
    params = init_params(jax.random.PRNGKey(seed), tf.V, d=MD, n_layers=ML, d_ff=2 * MD)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(100, max(1, steps // 10)), steps + 1, lr * 0.05)
    opt = optax.adamw(sched, weight_decay=1e-4)
    return params, opt, opt.init(jax.tree.map(jnp.asarray, params))

def diffusion_batch(rng, seqs, maskable, idx):
    x = seqs[idx].copy()
    u = rng.uniform(0.15, 1.0, size=(len(idx), 1))
    m = maskable[idx] & (rng.random(x.shape) < u)
    for r in range(len(idx)):                      # guarantee >=1 masked token
        if not m[r].any():
            opts = np.where(maskable[idx[r]])[0]
            m[r, rng.choice(opts)] = True
    noisy = x.copy()
    noisy[m] = tf.MASK
    return noisy, x, m.astype(np.float32)

def train(seed, outdir, steps=1500, batch=64, lr=3e-3, kinds=("diffusion", "ar"), tag="", cov=0.1, n_dose=0, init_tag=None):
    os.makedirs(outdir, exist_ok=True)
    import os as _os
    n_inst = int(_os.environ.get("N_INST", "4000"))
    seqs, maskable = build_dataset(seed, n_inst=n_inst, p_repair_cov=cov, n_dose=n_dose)
    _md = int(_os.environ.get("MODEL_D", "96")); _ml = int(_os.environ.get("MODEL_L", "3"))
    n_par = sum(x.size for x in jax.tree.leaves(
        init_params(jax.random.PRNGKey(0), tf.V, d=_md, n_layers=_ml, d_ff=2 * _md)))
    print(f"  config d={_os.environ.get('MODEL_D','96')} L={_os.environ.get('MODEL_L','3')} "
          f"n_inst={n_inst} params={n_par}", flush=True)
    n = len(seqs)
    print(f"[seed {seed}] dataset: {n} snapshots")
    rng = np.random.default_rng(seed)

    for kind in kinds:
        causal = kind == "ar"
        params, opt, opt_state = make_train_state(seed + (1000 if causal else 0), lr, steps)
        if init_tag is not None:
            with open(f"{outdir}/{kind}{init_tag}_seed{seed}.pkl", "rb") as f:
                params = jax.tree.map(jnp.asarray, pickle.load(f))
            print(f"  {kind} warm-started from {init_tag}", flush=True)
        ck_path = f"{outdir}/_ck_{kind}{tag}_seed{seed}.pkl"
        start_it = 0
        if os.path.exists(ck_path):
            with open(ck_path, "rb") as f:
                ck = pickle.load(f)
            params = jax.tree.map(jnp.asarray, ck["params"])
            opt_state = jax.tree.map(jnp.asarray, ck["opt_state"])
            start_it = ck["it"]
            rng = np.random.default_rng(seed * 31 + start_it)
            print(f"  {kind} resuming at it {start_it}", flush=True)
        @jax.jit
        def step(params, opt_state, noisy, tgt, lm):
            loss, g = jax.value_and_grad(masked_ce)(params, noisy, tgt, lm, causal)
            upd, opt_state = opt.update(g, opt_state, params)
            return optax.apply_updates(params, upd), opt_state, loss
        t0, losses = time.time(), []
        for it in range(start_it, steps):
            idx = rng.integers(0, n, batch)
            if causal:
                x = seqs[idx]
                noisy, tgt = x[:, :-1], x[:, 1:]
                lm = maskable[idx][:, 1:].astype(np.float32)
            else:
                noisy, tgt, lm = diffusion_batch(rng, seqs, maskable, idx)
            params, opt_state, loss = step(params, opt_state,
                                           jnp.asarray(noisy), jnp.asarray(tgt),
                                           jnp.asarray(lm))
            losses.append(float(loss))
            if it % 250 == 0 or it == steps - 1:
                print(f"  {kind} it {it:5d} loss {np.mean(losses[-100:]):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if it % 400 == 399:
                with open(ck_path, "wb") as f:
                    pickle.dump({"params": jax.tree.map(np.asarray, params),
                                 "opt_state": jax.tree.map(np.asarray, opt_state),
                                 "it": it + 1}, f)
        params = jax.tree.map(np.asarray, params)
        with open(f"{outdir}/{kind}{tag}_seed{seed}.pkl", "wb") as f:
            pickle.dump(params, f)
        if os.path.exists(ck_path):
            os.remove(ck_path)
    print(f"[seed {seed}] done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    seed, outdir = int(sys.argv[1]), sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    kinds = tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else ("diffusion", "ar")
    tag = sys.argv[6] if len(sys.argv) > 6 else ""
    cov = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
    n_dose = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    init_tag = sys.argv[9] if len(sys.argv) > 9 else None
    train(seed, outdir, steps, batch, kinds=kinds, tag=tag, cov=cov,
          n_dose=n_dose, init_tag=init_tag)
