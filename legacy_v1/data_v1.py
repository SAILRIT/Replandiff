"""Builds SFT snapshots. Each snapshot is (sequence@k, maskable-positions):
the context with observations revealed up to step k, executed steps 1..k visible,
and the expert-correct plan (under knowledge k) as the clean target for steps >k.

Cold-start gap: post-revelation repair snapshots for family-B perturbations are
included only with probability `p_repair_cov` (default 0.1), so SFT alone
under-repairs family B; the RL stage must close this gap from outcome reward.
"""
import random
import numpy as np
import toolflow as tf

def expert_rollout(inst):
    """Execute the expert (who repairs at revelation). Returns obs token per step
    and the plan tokens as actually executed."""
    st = tf.EnvState(inst)
    executed = tf.expert_default_plan(inst)[:]
    obs = []
    for i in range(1, tf.N_STEPS + 1):
        plan_now = tf.expert_plan_given(inst, i - 1)
        for p in tf.step_slots(i):
            executed[p - tf.PLAN_START] = plan_now[p - tf.PLAN_START]
        s0, s1, s2 = [executed[p - tf.PLAN_START] for p in tf.step_slots(i)]
        obs.append(tf.execute_step(st, i, s0, s1, s2))
    assert tf.success(st), "expert must always succeed"
    return obs, executed

def snapshot(inst, obs, executed, k):
    seq = tf.initial_context(inst)
    for i in range(k):
        seq[tf.OBS_POS[i]] = obs[i]
    target = tf.expert_plan_given(inst, k)
    for i in range(1, tf.N_STEPS + 1):
        for p in tf.step_slots(i):
            seq[p] = executed[p - tf.PLAN_START] if i <= k else target[p - tf.PLAN_START]
    maskable = [False] * tf.L
    for i in range(k + 1, tf.N_STEPS + 1):
        for p in tf.step_slots(i):
            maskable[p] = True
    return seq, maskable

def build_dataset(seed, n_inst=3000, p_perturb=0.5, p_repair_cov=0.1):
    rng = random.Random(seed)
    seqs, maskables = [], []
    for _ in range(n_inst):
        inst = tf.sample_instance(rng, p_perturb)
        obs, executed = expert_rollout(inst)
        ks = [0, 2, 3] + ([rng.choice([1, 4])] if rng.random() < 0.25 else [])
        for k in ks:
            if (inst.family == "B" and inst.perturbed and k >= 2
                    and rng.random() > p_repair_cov):
                continue  # withheld repair supervision (cold-start gap)
            s, m = snapshot(inst, obs, executed, k)
            seqs.append(s); maskables.append(m)
    return (np.array(seqs, np.int32), np.array(maskables, bool))

def eval_instances(seed, n, p_perturb):
    rng = random.Random(seed)
    return [tf.sample_instance(rng, p_perturb) for _ in range(n)]
