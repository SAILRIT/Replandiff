import numpy as np, jax, jax.numpy as jnp, optax, time
import toolflow as tf
from model import masked_ce
from train_sft import make_train_state, diffusion_batch
from data import build_dataset, eval_instances
from agent import run_episodes
seqs, mk = build_dataset(0)
print("snapshots:", len(seqs), flush=True)
params, opt, ost = make_train_state(0, 3e-3, 2000)
@jax.jit
def step(params, ost, noisy, tgt, lm):
    loss, g = jax.value_and_grad(masked_ce)(params, noisy, tgt, lm, False)
    upd, ost = opt.update(g, ost, params)
    return optax.apply_updates(params, upd), ost, loss
r = np.random.default_rng(0); t0=time.time()
for it in range(2000):
    idx = r.integers(0, len(seqs), 64)
    noisy, tgt, lm = diffusion_batch(r, seqs, mk, idx)
    params, ost, l = step(params, ost, jnp.asarray(noisy), jnp.asarray(tgt), jnp.asarray(lm))
print(f"trained in {time.time()-t0:.0f}s loss {float(l):.4f}", flush=True)
instsB = [i for i in eval_instances(555, 400, 1.0) if i.family == "B"][:100]
m, _ = run_episodes(params, instsB, "sel_conf", np.random.default_rng(1))
print(f"famB perturbed sel_conf success {m['success'].mean():.3f} regen {m['regen'].mean():.1f}", flush=True)
m0, _ = run_episodes(params, eval_instances(556, 100, 0.0), "sel_conf", np.random.default_rng(1))
print(f"unperturbed success {m0['success'].mean():.3f}", flush=True)
