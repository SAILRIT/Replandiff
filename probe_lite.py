import sys
import numpy as np, pickle, jax, jax.numpy as jnp
from data import eval_instances
from agent import run_episodes

tag = sys.argv[1] if len(sys.argv) > 1 else "steps=?"
d = jax.tree.map(jnp.asarray, pickle.load(open("runs/diffusion_lite_seed0.pkl", "rb")))
m1, _ = run_episodes(d, eval_instances(900, 96, 1.0), "sel_conf", np.random.default_rng(1))
m0, _ = run_episodes(d, eval_instances(901, 96, 0.0), "sel_conf", np.random.default_rng(1))
print(f"{tag}  p=1 succ {m1['success'].mean():.3f} regen {m1['regen'].mean():.2f} "
      f"inval {m1['invalid'].mean():.2f} | p=0 succ {m0['success'].mean():.3f} "
      f"inval {m0['invalid'].mean():.2f}", flush=True)
