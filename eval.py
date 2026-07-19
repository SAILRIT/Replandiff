"""Evaluate all methods across perturbation rates and seeds. Results merge into
results/results.json so the evaluation can run in resumable slices.
Usage: python3 eval.py <ckpt_dir> [n_episodes_per_cell] [seeds_csv] [ps_csv]"""
import sys, os, json, time, pickle
import numpy as np
import jax, jax.numpy as jnp
from data import eval_instances
from agent import run_episodes

METHODS = ["fixed", "full", "sel_oracle", "sel_conf", "ar", "ar_patch", "sel_conf_lite", "sel_conf_rl",
           "sel_oracle_super", "sel_random", "full_otrig", "full_ltrig", "ar_otrig"]
P_RATES = [0, 1, 2]
SEEDS = [0, 1, 2, 3, 4]
CHUNK = 120

def load(path):
    with open(path, "rb") as f:
        return jax.tree.map(jnp.asarray, pickle.load(f))

def eval_cell(dparams, arparams, method, insts, rng, **kw):
    agg = None
    for c0 in range(0, len(insts), CHUNK):
        chunk = insts[c0:c0 + CHUNK]
        real = method
        if method in ("sel_conf_rl", "sel_conf_lite"):
            real = "sel_conf"
        m, _ = run_episodes(dparams, chunk, real if real != "ar" else "ar",
                            rng, ar_params=arparams, **kw)
        if agg is None:
            agg = {k: list(v) for k, v in m.items()}
        else:
            for k in m: agg[k] += list(m[k])
    return {k: np.array(v) for k, v in agg.items()}

def main(ckpt_dir, n_eps=240, seeds=None, ps=None):
    seeds = seeds if seeds is not None else SEEDS
    ps = ps if ps is not None else P_RATES
    results = {}
    if os.path.exists("results/results.json"):
        results = json.load(open("results/results.json"))
    t0 = time.time()
    for seed in seeds:
        d = load(f"{ckpt_dir}/diffusion_v2_seed{seed}.pkl")
        a = load(f"{ckpt_dir}/ar_v2_seed{seed}.pkl")
        try:
            dlite = load(f"{ckpt_dir}/diffusion_lite_seed{seed}.pkl")
            drl = load(f"{ckpt_dir}/diffusion_rl_seed{seed}.pkl")
        except FileNotFoundError:
            dlite = drl = None
        for p in ps:
            insts = eval_instances(10_000 + seed * 100 + int(p) * 10, n_eps, p)
            for method in METHODS:
                key = f"{method}|p{p}|s{seed}"
                if key in results:
                    continue
                if method in ("sel_conf_lite", "sel_conf_rl") and drl is None:
                    continue
                rng = np.random.default_rng(seed * 7 + 3)
                params = {"sel_conf_rl": drl, "sel_conf_lite": dlite}.get(method, d)
                m = eval_cell(params, a, method, insts, rng)
                results[key] = {k: float(v.mean()) for k, v in m.items()}
                results[key]["success_list"] = [int(x) for x in m["success"]]
                print(f"{key:28s} succ {m['success'].mean():.3f} "
                      f"regen {m['regen'].mean():5.1f} edits {m['edits'].mean():4.1f} "
                      f"inval {m['invalid'].mean():4.2f} nfe {m['nfe'].mean():5.1f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                with open("results/results.json", "w") as f:
                    json.dump(results, f)
        # temperature robustness sweep at n=1 (in-distribution repairs)
        for T in [0.5, 0.8, 1.1]:
            insts = eval_instances(40_000 + seed, n_eps, 1)
            for method in ["full", "ar", "ar_patch", "sel_conf", "sel_oracle"]:
                key = f"U{T}|{method}|s{seed}"
                if key in results:
                    continue
                rng = np.random.default_rng(seed * 7 + 13)
                m = eval_cell(d, a, method, insts, rng, temp=T)
                results[key] = {k: float(v.mean()) for k, v in m.items()}
                results[key]["success_list"] = [int(x) for x in m["success"]]
                print(f"{key:28s} succ {m['success'].mean():.3f} "
                      f"regen {m['regen'].mean():5.1f}", flush=True)
                with open("results/results.json", "w") as f:
                    json.dump(results, f)
        # temperature robustness sweep at n=2 (held-out doubles)
        for T in [0.5, 0.8, 1.1]:
            insts = eval_instances(30_000 + seed, n_eps, 2)
            for method in ["full", "ar", "ar_patch", "sel_conf", "sel_oracle"]:
                key = f"T{T}|{method}|s{seed}"
                if key in results:
                    continue
                rng = np.random.default_rng(seed * 7 + 11)
                m = eval_cell(d, a, method, insts, rng, temp=T)
                results[key] = {k: float(v.mean()) for k, v in m.items()}
                results[key]["success_list"] = [int(x) for x in m["success"]]
                print(f"{key:28s} succ {m['success'].mean():.3f} "
                      f"regen {m['regen'].mean():5.1f}", flush=True)
                with open("results/results.json", "w") as f:
                    json.dump(results, f)
        # hole ablation: fixed-plan agent with holes on/off, unperturbed tasks
        insts = eval_instances(20_000 + seed, n_eps, 0)
        for tag, th in [("hole_on", 0.62), ("hole_off", 0.0)]:
            if f"{tag}|s{seed}" in results:
                continue
            rng = np.random.default_rng(seed * 7 + 5)
            m = eval_cell(d, a, "fixed", insts, rng, tau_hole=th)
            key = f"{tag}|s{seed}"
            results[key] = {k: float(v.mean()) for k, v in m.items()}
            results[key]["success_list"] = [int(x) for x in m["success"]]
            print(f"{key:28s} succ {m['success'].mean():.3f} "
                  f"inval {m['invalid'].mean():4.2f}", flush=True)
            with open("results/results.json", "w") as f:
                json.dump(results, f)
    print(f"done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    sds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    prs = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    main(ckpt, n, sds, prs)
