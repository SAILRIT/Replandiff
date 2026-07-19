"""Evaluate the dominance-supervision dose grid (cov=1.0; doses 0/8/64; seeds 0-2)
plus the base cov=0.1 five-seed models for reference. For each model: famA/famC
held-out doubles (400 eps), n=1 and n=0 sanity (240 eps), and the direct
composed-conditional probe p(escalate | SUSPENDED + INV_OPEN, step-6 masked).
Writes results/dose.json incrementally (resumable)."""
import os, json, random
import numpy as np, pickle, jax, jax.numpy as jnp
import toolflow as tf
from data import eval_instances, expert_rollout
from model import forward
from agent import run_episodes

load = lambda p: jax.tree.map(jnp.asarray, pickle.load(open(p, "rb")))
OUT = "results/dose.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def composed_probe(d, n=64):
    """p(escalate) at step 6 under the composed A1+A2 context, plan otherwise
    the dominance-correct repaired plan, step-6 slot masked."""
    rng = random.Random(11)
    seqs = []
    for _ in range(n):
        name, contact, dis = rng.sample(tf.NAMES, 3)
        inst = tf.Instance("A", name, contact, dis, frozenset({"A1", "A2"}),
                           rng.choice(["STD", "PREMIUM"]))
        obs, _ = expert_rollout(inst)
        s = tf.initial_context(inst)
        plan = tf.expert_plan_given(inst, 4)
        for p in tf.PLAN_POS:
            s[p] = plan[p - tf.PLAN_START]
        for i in range(4):
            s[tf.OBS_POS[i]] = obs[i]
        seqs.append(s)
    seqs = np.array(seqs, np.int32)
    sl = tf.step_slots(6)
    seqs[:, sl] = tf.MASK
    logits = np.asarray(forward(d, jnp.asarray(seqs), False))
    ex = np.exp(logits - logits.max(-1, keepdims=True))
    pr = ex / ex.sum(-1, keepdims=True)
    return (float(pr[:, sl[0], tf.TOK["escalate"]].mean()),
            float(pr[:, sl[0], tf.TOK["apply_credit"]].mean()))

def eval_model(key, path):
    if key in res:
        return
    d = load(path)
    i2 = eval_instances(70_000, 400, 2)
    fam = np.array([i.family for i in i2])
    m2, _ = run_episodes(d, i2, "sel_conf", np.random.default_rng(3))
    m1, _ = run_episodes(d, eval_instances(70_001, 240, 1), "sel_conf",
                         np.random.default_rng(4))
    m0, _ = run_episodes(d, eval_instances(70_002, 240, 0), "sel_conf",
                         np.random.default_rng(5))
    pe, pc = composed_probe(d)
    res[key] = {
        "famA": float(m2["success"][fam == "A"].mean()),
        "famC": float(m2["success"][fam == "C"].mean()),
        "all2": float(m2["success"].mean()),
        "n1": float(m1["success"].mean()), "n0": float(m0["success"].mean()),
        "n1_inval": float(m1["invalid"].mean()),
        "p_escalate_composed": pe, "p_credit_composed": pc,
    }
    json.dump(res, open(OUT, "w"))
    r = res[key]
    print(f"{key:14s} famA {r['famA']:.3f} famC {r['famC']:.3f} "
          f"n1 {r['n1']:.3f} n0 {r['n0']:.3f} "
          f"p(esc|comp) {pe:.3f} p(credit|comp) {pc:.3f}", flush=True)

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = []
    if which in ("all", "dose"):
        for s in [0, 1, 2]:
            for k in [0, 8, 64]:
                jobs.append((f"c1d{k}|s{s}", f"runs/diffusion_c1d{k}_seed{s}.pkl"))
        for s in [3, 4]:
            for k in [0, 8]:
                jobs.append((f"c1d{k}|s{s}", f"runs/diffusion_c1d{k}_seed{s}.pkl"))
    if which in ("all", "scale"):
        for s in [0, 1, 2]:
            for k in [0, 8]:
                jobs.append((f"w48d{k}|s{s}", f"runs/diffusion_w48d{k}_seed{s}.pkl"))
        for s in [0, 1]:
            for k in [0, 8]:
                jobs.append((f"w144d{k}|s{s}", f"runs/diffusion_w144d{k}_seed{s}.pkl"))
        jobs.append(("w144d8|s2", "runs/diffusion_w144d8_seed2.pkl"))
        for s in [0, 1, 2]:
            jobs.append((f"n16d0|s{s}", f"runs/diffusion_n16d0_seed{s}.pkl"))
    if which in ("all", "ext"):
        for s in [1, 4]:
            jobs.append((f"ext|s{s}", f"runs/diffusion_v2x_seed{s}.pkl"))
    if which in ("all", "base"):
        for s in [0, 1, 2, 3, 4]:
            jobs.append((f"base|s{s}", f"runs/diffusion_v2_seed{s}.pkl"))
    for key, path in jobs:
        eval_model(key, path)
