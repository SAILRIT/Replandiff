"""tau_inv operating curve for the confidence masker on the 5 base models:
success, regenerated tokens, and masker precision/recall vs threshold, at n=1
and n=2. Also the calibration spot-check: seed-2 dose-0 famC at tau=0.15.
Writes results/tau.json incrementally."""
import os, sys, json
import numpy as np, pickle, jax, jax.numpy as jnp
from data import eval_instances
from agent import run_episodes

load = lambda p: jax.tree.map(jnp.asarray, pickle.load(open(p, "rb")))
OUT = "results/tau.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
TAUS = [0.10, 0.20, 0.35, 0.50, 0.65]

def cell(key, d, insts, tau):
    if key in res:
        return
    m, _ = run_episodes(d, insts, "sel_conf", np.random.default_rng(13), tau_inv=tau)
    tp, fp, fn = m["mask_tp"].sum(), m["mask_fp"].sum(), m["mask_fn"].sum()
    res[key] = {"success": float(m["success"].mean()),
                "regen": float(m["regen"].mean()),
                "invalid": float(m["invalid"].mean()),
                "P": float(tp / max(tp + fp, 1)), "R": float(tp / max(tp + fn, 1))}
    json.dump(res, open(OUT, "w"))
    r = res[key]
    print(f"{key:22s} succ {r['success']:.3f} regen {r['regen']:5.2f} "
          f"P {r['P']:.2f} R {r['R']:.2f}", flush=True)

if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
    for s in seeds:
        d = load(f"runs/diffusion_v2_seed{s}.pkl")
        for n in [1, 2]:
            insts = eval_instances(80_000 + n, 300, n)
            for t in TAUS:
                cell(f"tau{t}|n{n}|s{s}", d, insts, t)
    # calibration spot-check: the over-repair pathology is threshold-fixable
    if "calib_s2d0_famC" not in res:
        d = load("runs/diffusion_c1d0_seed2.pkl")
        insts = eval_instances(70_002, 240, 0)
        fam = np.array([i.family for i in insts])
        out = {}
        for t in [0.35, 0.15]:
            m, _ = run_episodes(d, insts, "sel_conf", np.random.default_rng(5), tau_inv=t)
            out[str(t)] = {"famC_n0": float(m["success"][fam == "C"].mean()),
                           "famA_n0": float(m["success"][fam == "A"].mean())}
            print(f"calib s2-d0 tau={t}: famC n0 {out[str(t)]['famC_n0']:.3f} "
                  f"famA n0 {out[str(t)]['famA_n0']:.3f}", flush=True)
        res["calib_s2d0_famC"] = out
        json.dump(res, open(OUT, "w"))
