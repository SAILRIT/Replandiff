"""Episode runners. All diffusion methods share one denoiser; they differ only in
the *invalidation masker* applied after each observation:

  fixed      : never revise committed tokens
  full       : remask every future token after each observation (replan)
  sel_oracle : remask exactly the tokens contradicted by revealed state (upper bound)
  sel_conf   : slot-wise posterior rescoring under the updated context; remask
               committed tokens whose posterior prob drops below tau_inv
  ar         : causal baseline; regenerates the remaining suffix after each obs

Metrics per episode: success, invalid_calls, regen_tokens (re-committed after an
initial commitment), edits (tokens whose value changed on re-commit), nfe (model
forward passes), deferred (hole fills that were never prematurely committed).
"""
import numpy as np
import functools
import jax, jax.numpy as jnp
import toolflow as tf
from model import forward

fwd_bi = jax.jit(functools.partial(forward, causal=False))
fwd_ar = jax.jit(functools.partial(forward, causal=True))

def _softmax_np(x, temp):
    x = x / max(temp, 1e-6)
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

# --------------------------------------------------------------------- denoising
def denoise(params, seqs, mask, rng, rounds, temp, tau_hole, nfe_counter):
    """Iteratively commit masked positions, most-confident first. On the last
    round, positions with max-prob < tau_hole stay masked (typed holes)."""
    seqs, mask = seqs.copy(), mask.copy()
    for r in range(rounds):
        if not mask.any():
            break
        logits = np.asarray(fwd_bi(params, jnp.asarray(seqs)))
        nfe_counter[0] += 1
        probs = _softmax_np(logits, temp)
        B = seqs.shape[0]
        flat = probs.reshape(-1, probs.shape[-1])
        samp = (flat.cumsum(-1) > rng.random((flat.shape[0], 1))).argmax(-1)
        samp = samp.reshape(B, tf.L)
        conf = np.take_along_axis(probs, samp[..., None], -1)[..., 0]
        maxp = probs.max(-1)
        last = (r == rounds - 1)
        for b in range(B):
            idx = np.where(mask[b])[0]
            if idx.size == 0:
                continue
            if tau_hole > 0:                 # holes: never commit uncertain tokens
                idx = idx[maxp[b, idx] >= tau_hole]
                if idx.size == 0:
                    continue
            if last:
                commit = idx
            else:
                rem_rounds = rounds - r
                q = int(np.ceil(idx.size / rem_rounds))
                commit = idx[np.argsort(-conf[b, idx])[:q]]
            seqs[b, commit] = samp[b, commit]
            mask[b, commit] = False
    return seqs, mask

def rescore_future(params, seqs, cur_step, nfe_counter):
    """Posterior prob of each *committed* future token when its step-slot is
    jointly masked, under the updated context. Returns probs (B, L) with NaN at
    non-scored positions. One batched forward over (episode x future-slot)."""
    B = seqs.shape[0]
    fut = list(range(cur_step + 1, tf.N_STEPS + 1))
    if not fut:
        return None
    variants, meta = [], []
    for j in fut:
        v = seqs.copy()
        sl = tf.step_slots(j)
        v[:, sl] = tf.MASK
        variants.append(v)
        meta.append(sl)
    big = np.concatenate(variants, 0)
    logits = np.asarray(fwd_bi(params, jnp.asarray(big)))
    nfe_counter[0] += 1
    probs = _softmax_np(logits, 1.0)
    out = np.full((B, tf.L), np.nan)
    for fi, sl in enumerate(meta):
        blk = probs[fi * B:(fi + 1) * B]
        for p in sl:
            out[:, p] = np.take_along_axis(blk[:, p], seqs[:, p][:, None], -1)[:, 0]
    return out

# ---------------------------------------------------------------- episode runner
def run_episodes(params, instances, method, rng, temp=0.7, rounds=6, r_rounds=4,
                 tau_hole=0.62, tau_inv=0.35, ar_params=None):
    B = len(instances)
    seqs = np.array([tf.initial_context(i) for i in instances], np.int32)
    mask = np.zeros((B, tf.L), bool)
    mask[:, tf.PLAN_POS] = True
    ever_committed = np.zeros((B, tf.L), bool)
    prev_val = np.zeros((B, tf.L), np.int32)
    m = {k: np.zeros(B) for k in
         ["success", "invalid", "regen", "edits", "deferred", "nfe",
          "mask_tp", "mask_fp", "mask_fn"]}
    nfe = [0]
    states = [tf.EnvState(i) for i in instances]

    post_initial = [False]

    def note_commits(pre_mask, new_seqs, new_mask):
        newly = pre_mask & ~new_mask
        re_commit = newly & ever_committed
        m["regen"] += re_commit.sum(1)
        m["edits"] += (re_commit & (new_seqs != prev_val)).sum(1)
        if post_initial[0]:
            m["deferred"] += (newly & ~ever_committed).sum(1)
        ever_committed[newly] = True
        prev_val[newly] = new_seqs[newly]

    if method in ("ar", "ar_patch", "ar_otrig"):
        _ar_generate(ar_params, seqs, tf.PLAN_POS, rng, temp, nfe)
        ever_committed[:, tf.PLAN_POS] = True
        prev_val[:] = seqs
        if method == "ar_patch":
            m["nfe"] += len(tf.PLAN_POS)
    else:
        ns, nm = denoise(params, seqs, mask, rng, rounds, temp, tau_hole, nfe)
        note_commits(mask, ns, nm)
        seqs, mask = ns, nm
    post_initial[0] = True

    for i in range(1, tf.N_STEPS + 1):
        sl = tf.step_slots(i)
        if method not in ("ar", "ar_patch", "ar_otrig"):  # just-in-time hole fill
            need = mask[:, sl].any(1)
            if need.any():
                jm = mask & _only(sl)
                ns, nm = denoise(params, seqs, jm, rng, 2, temp, 0.0, nfe)
                note_commits(jm, ns, nm)
                seqs[:, sl] = ns[:, sl]
                mask[:, sl] = False
        for b in range(B):                              # execute (host loop)
            obs = tf.execute_step(states[b], i, *seqs[b, sl])
            seqs[b, tf.OBS_POS[i - 1]] = obs
        if i == tf.N_STEPS:
            break
        # ---- revision phase ----
        if method == "fixed":
            pass
        elif method == "ar":
            fut = [p for j in range(i + 1, tf.N_STEPS + 1) for p in tf.step_slots(j)]
            old = seqs[:, fut].copy()
            _ar_generate(ar_params, seqs, fut, rng, temp, nfe)
            m["regen"] += len(fut)
            m["edits"] += (seqs[:, fut] != old).sum(1)
        elif method == "ar_otrig":
            trig = np.array([len(_committed_diff(instances[b], seqs[b], i)) > 0
                             for b in range(B)])
            if trig.any():
                fut = [p for j in range(i + 1, tf.N_STEPS + 1) for p in tf.step_slots(j)]
                old = seqs.copy()
                _ar_generate(ar_params, seqs, fut, rng, temp, nfe)
                keep = ~trig
                seqs[keep] = old[keep]                    # untriggered rows unchanged
                m["regen"][trig] += len(fut)
                m["edits"][trig] += (seqs[trig][:, fut] != old[trig][:, fut]).sum(1)
                m["nfe"][trig] += len(fut)                # idealized per-episode count
        elif method == "ar_patch":
            fut = [p for j in range(i + 1, tf.N_STEPS + 1) for p in tf.step_slots(j)]
            logits = np.asarray(fwd_ar(ar_params, jnp.asarray(seqs)))
            nfe[0] += 1
            m["nfe"] += 1
            probs = _softmax_np(logits, 1.0)
            flagged = np.zeros((B, tf.L), bool)
            for p in fut:
                pr = np.take_along_axis(probs[:, p - 1], seqs[:, p][:, None], -1)[:, 0]
                flagged[:, p] = pr < tau_inv
            _masker_stats(m, flagged, instances, seqs, i)
            old = seqs.copy()
            for p in fut:                       # left-to-right local edits
                rows = flagged[:, p]
                if not rows.any():
                    continue
                lg = np.asarray(fwd_ar(ar_params, jnp.asarray(seqs)))[:, p - 1]
                nfe[0] += 1
                pr = _softmax_np(lg, temp)
                samp = (pr.cumsum(-1) > rng.random((B, 1))).argmax(-1)
                seqs[rows, p] = samp[rows]
            m["nfe"] += flagged.sum(1)
            m["regen"] += flagged.sum(1)
            m["edits"] += (flagged & (seqs != old)).sum(1)
        else:
            remask = np.zeros((B, tf.L), bool)
            if method == "full":
                for j in range(i + 1, tf.N_STEPS + 1):
                    remask[:, tf.step_slots(j)] = True
            elif method == "sel_oracle":
                for b in range(B):
                    plan_toks = list(seqs[b, tf.PLAN_POS])
                    for p in tf.oracle_invalid_positions(instances[b], plan_toks, i):
                        remask[b, p] = True
            elif method == "sel_oracle_super":
                for b in range(B):
                    for p in tf.superset_positions(instances[b], i):
                        remask[b, p] = True
            elif method == "sel_random":
                # control: size-matched to the oracle set, positions random
                fut = [p for j in range(i + 1, tf.N_STEPS + 1)
                       for p in tf.step_slots(j)]
                for b in range(B):
                    ksz = len(_committed_diff(instances[b], seqs[b], i))
                    if ksz == 0:
                        continue
                    cand = [p for p in fut if ever_committed[b, p] and not mask[b, p]]
                    if not cand:
                        continue
                    pick = rng.choice(len(cand), size=min(ksz, len(cand)),
                                      replace=False)
                    for c in np.atleast_1d(pick):
                        remask[b, cand[int(c)]] = True
            elif method in ("full_otrig", "full_ltrig"):
                if method == "full_otrig":
                    trig = np.array([len(_committed_diff(instances[b], seqs[b], i)) > 0
                                     for b in range(B)])
                else:
                    sc = rescore_future(params, seqs, i, nfe)
                    m["nfe"] += 1
                    fl = (np.nan_to_num(sc, nan=1.0) < tau_inv) if sc is not None                         else np.zeros((B, tf.L), bool)
                    trig = (fl & ever_committed & ~mask).any(1)
                if trig.any():
                    fut = [p for j in range(i + 1, tf.N_STEPS + 1)
                           for p in tf.step_slots(j)]
                    remask[np.ix_(np.where(trig)[0], fut)] = True
                    m["nfe"][trig] += r_rounds
            elif method == "sel_conf":
                sc = rescore_future(params, seqs, i, nfe)
                if sc is not None:
                    remask = np.nan_to_num(sc, nan=1.0) < tau_inv
                    _masker_stats(m, remask & ever_committed & ~mask,
                                  instances, seqs, i)
            remask &= ever_committed & ~mask
            if remask.any() or mask.any():
                mask2 = mask | remask
                seqs2 = seqs.copy()
                seqs2[mask2] = tf.MASK
                ns, nm = denoise(params, seqs2, mask2, rng, r_rounds, temp,
                                 tau_hole, nfe)
                note_commits(mask2, ns, nm)
                seqs, mask = ns, nm
    for b in range(B):
        m["success"][b] = float(tf.success(states[b]))
        m["invalid"][b] = states[b].invalid_calls
    if method == "ar_otrig":
        m["nfe"] += len(tf.PLAN_POS)                      # initial generation
    elif method in ("full_otrig", "full_ltrig"):
        m["nfe"] += 6 + 2                                 # initial rounds + JIT budget
    elif method != "ar_patch":
        m["nfe"][:] = nfe[0]
    return m, seqs

def _committed_diff(inst, seqs_row, k):
    """Oracle-diff positions restricted to committed tokens (open holes excluded)."""
    return [p for p in tf.oracle_invalid_positions(inst, list(seqs_row[tf.PLAN_POS]), k)
            if seqs_row[p] != tf.MASK]

def _masker_stats(m, flagged, instances, seqs, k):
    for b in range(len(instances)):
        plan_toks = list(seqs[b, tf.PLAN_POS])
        oracle = {p for p in tf.oracle_invalid_positions(instances[b], plan_toks, k)
                  if seqs[b, p] != tf.MASK}
        pred = set(np.where(flagged[b])[0])
        m["mask_tp"][b] += len(pred & oracle)
        m["mask_fp"][b] += len(pred - oracle)
        m["mask_fn"][b] += len(oracle - pred)

def _only(slots):
    v = np.zeros(tf.L, bool)
    v[slots] = True
    return v

def _ar_generate(ar_params, seqs, positions, rng, temp, nfe_counter):
    for p in sorted(positions):
        logits = np.asarray(fwd_ar(ar_params, jnp.asarray(seqs)))[:, p - 1]
        nfe_counter[0] += 1
        probs = _softmax_np(logits, temp)
        c = probs.cumsum(-1)
        seqs[:, p] = (c > rng.random((seqs.shape[0], 1))).argmax(-1)
