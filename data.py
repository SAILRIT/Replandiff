"""SFT snapshots for ToolFlow-v2. Snapshots are taken at k=0, at each active
perturbation's reveal step, and (with prob 0.5) at one random stability step.
Training instances carry AT MOST ONE perturbation; double-perturbation episodes
exist only at evaluation time (compositional generalization split).
`p_repair_cov` < 1 withholds a fraction of post-reveal repair snapshots
(used only for the supervision-scarce RL condition)."""
import random
import numpy as np
import toolflow as tf

def expert_rollout(inst):
    st = tf.EnvState(inst)
    executed = tf.expert_default_plan(inst)[:]
    obs = []
    for i in range(1, tf.N_STEPS + 1):
        plan_now = tf.expert_plan_given(inst, i - 1)
        for p in tf.step_slots(i):
            executed[p - tf.PLAN_START] = plan_now[p - tf.PLAN_START]
        s0, s1, s2 = [executed[p - tf.PLAN_START] for p in tf.step_slots(i)]
        obs.append(tf.execute_step(st, i, s0, s1, s2))
    assert tf.success(st), f"expert failed: {inst}"
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

def build_dataset(seed, n_inst=4000, p_perturb=0.55, p_repair_cov=1.0, n_dose=0):
    """n_dose > 0 adds that many family-A double-perturbation (A1+A2) instances
    with expert dominance-consistent repairs -- the controlled supervision
    intervention. Family-C doubles remain fully held out."""
    rng = random.Random(seed)
    seqs, maskables = [], []
    specs = [None] * n_inst + ["doseA"] * n_dose
    rng.shuffle(specs)
    for spec in specs:
        if spec == "doseA":
            name, contact, distractor = rng.sample(tf.NAMES, 3)
            inst = tf.Instance("A", name, contact, distractor,
                               frozenset({"A1", "A2"}), rng.choice(["STD", "PREMIUM"]))
        else:
            inst = tf.sample_instance(rng, p_perturb)
        obs, executed = expert_rollout(inst)
        ks = {0}
        reveals = sorted(tf.REVEAL[p] for p in inst.perturbs)
        ks.update(reveals)
        if rng.random() < 0.5:
            ks.add(rng.choice([k for k in range(1, tf.N_STEPS) if k not in ks]))
        for k in sorted(ks):
            if (inst.perturbs and reveals and k >= reveals[0]
                    and rng.random() > p_repair_cov):
                continue
            s, m = snapshot(inst, obs, executed, k)
            seqs.append(s); maskables.append(m)
    return (np.array(seqs, np.int32), np.array(maskables, bool))

def eval_instances(seed, n, n_perturb):
    """n_perturb: 0, 1, or 2 (2 = compositionally novel doubles, families A/C)."""
    rng = random.Random(seed)
    spec = int(n_perturb) if float(n_perturb) in (0.0, 1.0, 2.0) else float(n_perturb)
    return [tf.sample_instance(rng, spec) for _ in range(n)]
