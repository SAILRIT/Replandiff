"""Diagnostics for the one-step estimator at the lite checkpoint (seed 0).
Collects, over rollout batches identical to training: per-token one-step
probability p of each sampled plan token under the final context and 50% masking,
the episode's group-relative advantage A (signed, clip +/-1.5), and the exact
per-token logit-gradient norm ||p_vec - e_y|| * |A| / |S_b|.
Outputs results/rl_diag.json with histograms and binned gradient mass."""
import json
import numpy as np
import pickle, jax, jax.numpy as jnp
import toolflow as tf
from model import forward
from data import eval_instances
from agent import run_episodes
from train_rl import reward, T_TASKS, G

params = jax.tree.map(jnp.asarray, pickle.load(open("runs/diffusion_lite_seed0.pkl", "rb")))
rng = np.random.default_rng(123)
recs = {"succ_p": [], "fail_p": [], "mass": {}}   # mass[(sign,bin)] = summed grad-norm
bins = [0.0, 0.5, 0.9, 0.99, 1.0]

for it in range(10):
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
    adv = np.clip(adv, -1.5, 1.5)
    lm = (rng.random(final_seqs.shape) < 0.5)
    lm[:, :tf.PLAN_START] = False
    noisy = final_seqs.copy(); noisy[lm] = tf.MASK
    logits = np.asarray(forward(params, jnp.asarray(noisy), False))
    ex = np.exp(logits - logits.max(-1, keepdims=True))
    probs = ex / ex.sum(-1, keepdims=True)
    for b in range(len(insts)):
        idx = np.where(lm[b])[0]
        if idx.size == 0 or abs(adv[b]) < 1e-6:
            tgt_kind = None
        p_tok = np.array([probs[b, i, final_seqs[b, i]] for i in idx])
        (recs["succ_p"] if m["success"][b] > 0.5 else recs["fail_p"]).extend(p_tok.tolist())
        # exact logit-grad norm per token: |A|/|S| * ||p_vec - onehot||
        for j, i in enumerate(idx):
            pv = probs[b, i].copy()
            gy = np.sqrt(((pv - np.eye(len(pv))[final_seqs[b, i]]) ** 2).sum())
            mass = abs(adv[b]) / idx.size * gy
            sgn = "pos" if adv[b] > 0 else "neg"
            k = f"{sgn}|{np.digitize(p_tok[j], bins[1:-1])}"
            recs["mass"][k] = recs["mass"].get(k, 0.0) + float(mass)

out = {
    "succ_p_hist": np.histogram(recs["succ_p"], bins=20, range=(0, 1))[0].tolist(),
    "fail_p_hist": np.histogram(recs["fail_p"], bins=20, range=(0, 1))[0].tolist(),
    "succ_p_median": float(np.median(recs["succ_p"])),
    "fail_p_median": float(np.median(recs["fail_p"])),
    "succ_p_frac_below_0.9": float(np.mean(np.array(recs["succ_p"]) < 0.9)),
    "fail_p_frac_below_0.9": float(np.mean(np.array(recs["fail_p"]) < 0.9)),
    "n_succ_tokens": len(recs["succ_p"]), "n_fail_tokens": len(recs["fail_p"]),
    "grad_mass": recs["mass"], "bins": bins,
}
json.dump(out, open("results/rl_diag.json", "w"))
print(json.dumps({k: v for k, v in out.items() if "hist" not in k}, indent=1))
