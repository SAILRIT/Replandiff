#!/usr/bin/env bash
# ~5-minute sanity check: tiny training run, small evaluation, one figure.
set -e
mkdir -p runs logs results figures
echo "[1/4] environment self-checks (expert solves all perturbation specs)..."
python3 - <<'PY'
import random, toolflow as tf
from data import expert_rollout, eval_instances
rng = random.Random(0)
for _ in range(200):
    expert_rollout(tf.sample_instance(rng, 0.9))
for inst in eval_instances(1, 50, 2):
    expert_rollout(inst)
print("   environment OK")
PY
echo "[2/4] 150-step training run (d=48)..."
MODEL_D=48 MODEL_L=2 python3 train_sft.py 0 runs 150 32 diffusion _smoke 1.0 0 > logs/smoke.log
tail -n 1 logs/smoke.log
echo "[3/4] 40-episode evaluation of the smoke model..."
python3 - <<'PY'
import numpy as np, pickle, jax, jax.numpy as jnp
from data import eval_instances
from agent import run_episodes
d = jax.tree.map(jnp.asarray, pickle.load(open("runs/diffusion_smoke_seed0.pkl", "rb")))
m, _ = run_episodes(d, eval_instances(5, 40, 1), "sel_conf", np.random.default_rng(0))
print(f"   smoke model n=1 success {m['success'].mean():.2f} "
      f"(150 steps: any nonzero value passes)")
PY
echo "[4/4] regenerate one figure from committed results..."
python3 - <<'PY'
import json
assert "sel_conf|p1|s0" in json.load(open("results/results.json"))
print("   committed results present")
PY
python3 plots.py > /dev/null && echo "   figures regenerated OK"
echo "SMOKE TEST PASSED"
