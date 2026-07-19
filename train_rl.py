"""GRPO-lite on top of the diffusion SFT checkpoint, with the sel_conf agent as
the rollout policy. Reward = 2*success - 0.2*invalid_calls - 0.3*(regen/15).
Group-relative advantages (G rollouts per task); policy gradient uses the
one-step masked log-likelihood estimator (as in diffu-GRPO-style methods):
re-mask a random 50% of the final plan tokens and weight their NLL by the
advantage. A small replay (SFT) anchor stabilizes training.

Usage: python3 train_rl.py <seed> <ckpt_dir> [iters]
"""
import sys, time, pickle, json
import numpy as np
import jax, jax.numpy as jnp
import optax
import toolflow as tf
from model import forward, masked_ce
from data import build_dataset, eval_instances
from agent import run_episodes

T_TASKS, G = 8, 6
BETA_REPLAY = 1.0
LAMBDA_INV, LAMBDA_REGEN = 0.2, 0.3

def reward(m, b):
    return 2.0 * m["success"][b] - LAMBDA_INV * m["invalid"][b] \
           - LAMBDA_REGEN * (m["regen"][b] / 24.0)

EPS_CLIP = 0.2

def token_logp(params, noisy, tgt):
    logits = forward(params, noisy, False)
    logp = jax.nn.log_softmax(logits, -1)
    return jnp.take_along_axis(logp, tgt[..., None], -1)[..., 0]

def rl_loss(params, noisy, tgt, lm, adv, old_lp, rp_noisy, rp_tgt, rp_lm):
    """GRPO-lite: per-token importance ratios vs. rollout-time logprobs under the
    same random masking, PPO-clipped, advantage-weighted (one-step estimator)."""
    new_lp = token_logp(params, noisy, tgt)
    ratio = jnp.exp(new_lp - old_lp)
    a = adv[:, None]
    obj = jnp.minimum(ratio * a, jnp.clip(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * a)
    pg = -(obj * lm).sum() / jnp.maximum(lm.sum(), 1.0)
    anchor = masked_ce(params, rp_noisy, rp_tgt, rp_lm, False)
    return pg + BETA_REPLAY * anchor

def probe(params, insts, rng):
    m, _ = run_episodes(params, insts, "sel_conf", rng, temp=0.7)
    return float(m["success"].mean()), float(m["regen"].mean())

def main(seed, ckpt_dir, iters=150, lr=1e-4, mode="filtered"):
    """mode selects the advantage transform (all else identical):
       filtered: clip(A,0,1.5)  -- reward-filtered imitation (main recipe)
       signed:   clip(A,-1.5,1.5) -- signed group-relative advantages
       negonly:  clip(A,-1.5,0)  -- destructive direction isolated
       randsign: random +/-1 per episode -- pure variance control
       replay:   A=0 (anchor-only; continued SFT control)
       raftbc:   A=1[success] (unnormalized success-only BC)"""
    import os
    tag = "" if mode == "filtered" else f"arm_{mode}_"
    rl_path = f"{ckpt_dir}/diffusion_rl{('_' + tag[:-1]) if tag else ''}_seed{seed}.pkl".replace("__", "_")
    curve_path = f"results/rl_{tag}curve_seed{seed}.json" if tag else f"results/rl_curve_seed{seed}.json"
    start_it, curve = 0, []
    if os.path.exists(rl_path) and os.path.exists(curve_path):
        with open(rl_path, "rb") as f:
            params = pickle.load(f)
        curve = json.load(open(curve_path))
        start_it = curve[-1]["it"]
        print(f"[rl seed {seed}] resuming at it {start_it}", flush=True)
    else:
        with open(f"{ckpt_dir}/diffusion_lite_seed{seed}.pkl", "rb") as f:
            params = pickle.load(f)
    params = jax.tree.map(jnp.asarray, params)
    opt = optax.adamw(lr, weight_decay=1e-4)
    ost = opt.init(params)
    step_fn = jax.jit(lambda p, o, *a: _step(p, o, opt, *a))
    seqs, maskable = build_dataset(seed + 500, n_inst=1200)
    rng = np.random.default_rng(seed + 77)
    probe_insts = eval_instances(seed + 900, 96, 1)
    if start_it == 0:
        s0, r0 = probe(params, probe_insts, np.random.default_rng(1))
        curve.append({"it": 0, "probe_success": s0, "probe_regen": r0, "reward": None})
        print(f"[rl seed {seed}] probe@0 success {s0:.3f} regen {r0:.2f}", flush=True)
    t0 = time.time()
    for it in range(start_it + 1, iters + 1):
        insts = []
        for _ in range(T_TASKS):
            base = eval_instances(int(rng.integers(1e9)), 1, 0.75)[0]
            insts += [base] * G
        m, final_seqs = run_episodes(params, insts, "sel_conf", rng, temp=1.0)
        R = np.array([reward(m, b) for b in range(len(insts))])
        adv = np.zeros_like(R)
        for t in range(T_TASKS):
            g = R[t * G:(t + 1) * G]
            adv[t * G:(t + 1) * G] = (g - g.mean()) / (g.std() + 1e-4)
        if mode in ("filtered",):
            adv = np.clip(adv, 0.0, 1.5)
        elif mode == "signed":
            adv = np.clip(adv, -1.5, 1.5)
        elif mode == "negonly":
            adv = np.clip(adv, -1.5, 0.0)
        elif mode == "randsign":
            adv = rng.choice([-1.0, 1.0], size=adv.shape)
        elif mode == "replay":
            adv = np.zeros_like(adv)
        elif mode == "raftbc":
            adv = np.array([1.0 if m["success"][b] > 0.5 else 0.0
                            for b in range(len(insts))])
        # (filtered-mode rationale) non-negative advantages only:
        # under the final context correct tokens saturate at p~1, so positive
        # gradients vanish while negative ones keep eroding them; at this scale
        # the stable estimator imitates above-mean rollouts (RAFT-style) with
        # the replay anchor preventing drift.
        lm = (rng.random(final_seqs.shape) < 0.5)
        lm[:, :tf.PLAN_START] = False
        noisy = final_seqs.copy(); noisy[lm] = tf.MASK
        old_lp = np.asarray(token_logp(params, jnp.asarray(noisy),
                                       jnp.asarray(final_seqs)))
        ridx = rng.integers(0, len(seqs), 64)
        rp = seqs[ridx].copy()
        u = rng.uniform(0.15, 1.0, (64, 1))
        rlm = maskable[ridx] & (rng.random(rp.shape) < u)
        rp_noisy = rp.copy(); rp_noisy[rlm] = tf.MASK
        params, ost, loss = step_fn(params, ost, jnp.asarray(noisy),
                                    jnp.asarray(final_seqs), jnp.asarray(lm, jnp.float32),
                                    jnp.asarray(adv), jnp.asarray(old_lp),
                                    jnp.asarray(rp_noisy),
                                    jnp.asarray(rp), jnp.asarray(rlm, jnp.float32))
        if it % 15 == 0 or it == iters:
            s, rg = probe(params, probe_insts, np.random.default_rng(1))
            curve.append({"it": it, "probe_success": s, "probe_regen": rg,
                          "reward": float(R.mean())})
            print(f"[rl seed {seed}] it {it:4d} R {R.mean():+.3f} "
                  f"probe {s:.3f} regen {rg:.2f} ({time.time()-t0:.0f}s)", flush=True)
            np_params = jax.tree.map(np.asarray, params)
            with open(rl_path, "wb") as f:
                pickle.dump(np_params, f)
            with open(curve_path, "w") as f:
                json.dump(curve, f)

def _step(params, ost, opt, noisy, tgt, lm, adv, old_lp, rp_noisy, rp_tgt, rp_lm):
    loss, g = jax.value_and_grad(rl_loss)(params, noisy, tgt, lm, adv, old_lp,
                                          rp_noisy, rp_tgt, rp_lm)
    upd, ost = opt.update(g, ost, params)
    return optax.apply_updates(params, upd), ost, loss

if __name__ == "__main__":
    seed, ckpt = int(sys.argv[1]), sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    mode = sys.argv[4] if len(sys.argv) > 4 else "filtered"
    main(seed, ckpt, iters, mode=mode)
