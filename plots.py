"""Publication figures, v3: collision-proof layout. Multi-series figures use
figure-level legends OUTSIDE the axes (constrained layout); no text annotations
sit on data; inside legends only where a region is verifiably empty."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from data import eval_instances

W_FULL, W_HALF = 5.5, 2.65
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 6.6, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.linewidth": 0.8,
    "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
})
R = json.load(open("results/results.json"))
S5 = [0, 1, 2, 3, 4]
C = {"fixed": "#999999", "full": "#E69F00", "sel_oracle": "#009E73",
     "sel_conf": "#0072B2", "sel_conf_rl": "#D55E00", "ar": "#CC79A7",
     "sel_conf_lite": "#56B4E9", "ar_patch": "#E8D44D"}
EC = {"ar_patch": "#7a731c"}
MK = {"fixed": "s", "full": "^", "sel_oracle": "D", "sel_conf": "o",
      "sel_conf_rl": "P", "ar": "v", "sel_conf_lite": "X", "ar_patch": "h"}
NAME = {"fixed": "Fixed plan", "full": "Full replan", "sel_oracle": "Selective (oracle)",
        "sel_conf": "Selective (conf.)", "sel_conf_rl": "lite-SFT + reward-filtered",
        "ar": "AR replan", "sel_conf_lite": "lite-SFT", "ar_patch": "AR patch"}
MAIN = ["fixed", "full", "ar", "ar_patch", "sel_conf", "sel_oracle"]

def cells(m, p, key, seeds=S5):
    return np.array([R[f"{m}|p{p}|s{s}"][key] for s in seeds])

def handle(m, ms=4.2):
    return Line2D([], [], marker=MK[m], color=C[m], ms=ms, lw=1.2,
                  mfc="white" if m in ("full", "ar") else C[m],
                  mec=EC.get(m, C[m]), mew=0.9)

def save(fig, name):
    fig.savefig(f"figures/{name}.pdf")
    fig.savefig(f"figures/{name}.png")
    plt.close(fig)
    print("wrote", name)

# ---------------- fig1: success vs number of perturbations --------------------
fig, ax = plt.subplots(figsize=(W_HALF, 2.55), layout="constrained")
JIT = dict(zip(MAIN, np.linspace(-0.07, 0.07, len(MAIN))))
for m in MAIN:
    mu = [cells(m, p, "success").mean() for p in [0, 1, 2]]
    se = [cells(m, p, "success").std() / np.sqrt(5) for p in [0, 1, 2]]
    mfc = "white" if m in ("full", "ar") else C[m]
    ax.errorbar(np.array([0, 1, 2]) + JIT[m], mu, yerr=se, marker=MK[m],
                ms=4 if m in ("full", "ar") else 3.6, color=C[m], lw=1.2,
                capsize=1.5, elinewidth=0.7, mfc=mfc, mec=EC.get(m, C[m]), mew=0.9)
ax.set_xlabel("Perturbations per episode ($n{=}2$: novel pairs)")
ax.set_ylabel("Task success")
ax.set_xticks([0, 1, 2])
ax.set_ylim(-0.03, 1.05)
fig.legend([handle(m) for m in MAIN], [NAME[m] for m in MAIN],
           loc="outside upper center", ncol=2, frameon=False, fontsize=6.4)
save(fig, "fig1_success_vs_nperturb")

# ---------------- fig2: efficiency Pareto at n=1 (headline) -------------------
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.5), layout="constrained")
for ax, xkey, xlab in [(axes[0], "regen", "Regenerated tokens / episode (log)"),
                       (axes[1], "nfe", "Model calls (NFE) / episode (log)")]:
    for m in MAIN:
        x = cells(m, 1, xkey); y = cells(m, 1, "success")
        xm = max(x.mean(), 0.05)
        mfc = "white" if m in ("full", "ar") else C[m]
        ax.errorbar(xm, y.mean(), xerr=x.std() / np.sqrt(5), yerr=y.std() / np.sqrt(5),
                    marker=MK[m], ms=5.5 if m == "ar_patch" else 4.8, color=C[m],
                    capsize=1.5, elinewidth=0.7, mfc=mfc, mec=EC.get(m, C[m]),
                    mew=1.0, zorder=3 if m == "sel_oracle" else 4)
    ax.set_xscale("log")
    ax.set_xlabel(xlab)
    ax.set_ylim(-0.06, 1.10)
    ax.annotate("", xy=(0.03, 0.965), xytext=(0.30, 0.965),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#666666"))
    ax.text(0.165, 0.905, "better", transform=ax.transAxes, fontsize=6.5,
            color="#666666", ha="center")
axes[0].set_ylabel("Task success ($n{=}1$)")
axes[0].set_xlim(0.03, 300)
axes[1].set_xlim(4, 300)
fig.legend([handle(m) for m in MAIN], [NAME[m] for m in MAIN],
           loc="outside upper center", ncol=6, frameon=False, fontsize=6.3)
save(fig, "fig2_pareto_n1")

# ------------- fig3: compositional doubles, per family, per seed --------------
def family_split(seed):
    insts = eval_instances(10_000 + seed * 100 + 20, 400, 2)
    return np.array([i.family for i in insts])

fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.25), sharey=True,
                         layout="constrained")
panels = [(None, "All doubles"), ("A", "Family A: conflicting (dominance)"),
          ("C", "Family C: independent")]
METH3 = ["full", "ar", "ar_patch", "sel_conf", "sel_oracle"]
SHORT = {"full": "Full", "ar": "AR", "ar_patch": "AR\npatch",
         "sel_conf": "Sel.\n(conf.)", "sel_oracle": "Sel.\n(oracle)"}
for ax, (fam, title) in zip(axes, panels):
    for xi, m in enumerate(METH3):
        vals = []
        for s in S5:
            sl = np.array(R[f"{m}|p2|s{s}"]["success_list"])
            vals.append(sl.mean() if fam is None else sl[family_split(s) == fam].mean())
        xs = np.full(5, xi) + np.linspace(-0.14, 0.14, 5)
        ax.scatter(xs, vals, s=13, color=C[m], marker=MK[m], zorder=3,
                   edgecolors=EC.get(m, "none"), linewidths=0.6)
        ax.hlines(np.mean(vals), xi - 0.3, xi + 0.3, color=C[m], lw=1.7, zorder=2)
    ax.set_xticks(range(len(METH3)))
    ax.set_xticklabels([SHORT[m] for m in METH3], fontsize=6.8)
    ax.set_xlim(-0.55, len(METH3) - 0.45)
    ax.set_title(title, fontsize=7.4)
    ax.set_ylim(-0.06, 1.08)
axes[0].set_ylabel("Task success (5 seeds)")
save(fig, "fig3_composition_by_family")

# ---------------- fig4: post-training (5 seeds) -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.4), layout="constrained")
ax = axes[0]
curves = [json.load(open(f"results/rl_curve_seed{s}.json")) for s in S5]
for c in curves:
    ax.plot([d["it"] for d in c], [d["probe_success"] for d in c],
            color="#D55E00", alpha=0.32, lw=0.9)
its = [d["it"] for d in curves[0]]
mean = np.mean([[d["probe_success"] for d in c] for c in curves], axis=0)
mean_h, = ax.plot(its, mean, color="#D55E00", lw=2.0)
ax.set_xlabel("Post-training iteration")
ax.set_ylabel("Probe success ($n{=}1$, perturbed)")
ax.set_ylim(0.55, 1.04)
ax = axes[1]
groups = [("Success\n($n{=}1$)", "success", 1), ("Success\n($n{=}2$, novel)", "success", 2),
          ("Invalid calls\n($n{=}1$)", "invalid", 1)]
w = 0.36
for gi, (lab, key, p) in enumerate(groups):
    for oi, m in enumerate(["sel_conf_lite", "sel_conf_rl"]):
        v = cells(m, p, key, seeds=S5)
        ax.bar(gi + (oi - 0.5) * w, v.mean(), w * 0.9, color=C[m],
               edgecolor="none")
        ax.errorbar(gi + (oi - 0.5) * w, v.mean(), v.std() / np.sqrt(5),
                    color="k", capsize=2, elinewidth=0.8, lw=0)
        ax.scatter(np.full(5, gi + (oi - 0.5) * w) + np.linspace(-0.06, 0.06, 5),
                   v, s=7, color="k", zorder=3, alpha=0.55)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=6.9)
ax.set_ylabel("Success / invalid calls per ep.", fontsize=7.2)
ax.set_ylim(0, 1.10)
from matplotlib.patches import Patch
fig.legend([mean_h, Line2D([], [], color="#D55E00", alpha=0.32, lw=0.9),
            Patch(color=C["sel_conf_lite"]), Patch(color=C["sel_conf_rl"])],
           ["mean of 5 seeds", "individual seeds",
            NAME["sel_conf_lite"], NAME["sel_conf_rl"]],
           loc="outside upper center", ncol=4, frameon=False, fontsize=6.3)
save(fig, "fig4_rl")

# ---------------- fig5: typed-hole ablation (5 seeds) -------------------------
fig, ax = plt.subplots(figsize=(W_HALF, 2.4), layout="constrained")
for gi, tag in enumerate(["hole_on", "hole_off"]):
    for oi, (key, col, klab) in enumerate([("success", "#0072B2", "Task success"),
                                           ("invalid", "#E69F00", "Invalid calls / ep.")]):
        v = np.array([R[f"{tag}|s{s}"][key] for s in S5])
        x = gi + (oi - 0.5) * 0.36
        ax.bar(x, v.mean(), 0.33, color=col, label=klab if gi == 0 else None)
        ax.scatter(np.full(5, x) + np.linspace(-0.05, 0.05, 5), v, s=7,
                   color="k", zorder=3, alpha=0.55)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Defer holes\n" + r"($\tau_{hole}{=}0.62$)",
                    "Commit early\n" + r"($\tau_{hole}{=}0$)"], fontsize=7)
ax.set_ylim(0, 1.16)
ax.set_title("Unperturbed tasks, fixed-plan agent", fontsize=7.6)
ax.legend(frameon=False, fontsize=6.6, loc="upper center", ncol=1,
          bbox_to_anchor=(0.5, 1.0))
save(fig, "fig5_hole_ablation")

# ---------------- fig6: decode-noise robustness -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.35), sharey=True,
                         layout="constrained")
M6 = ["full", "ar", "ar_patch", "sel_conf", "sel_oracle"]
for ax, pref, title in [(axes[0], "U", "In-distribution repairs ($n{=}1$)"),
                        (axes[1], "T", "Novel doubles ($n{=}2$)")]:
    for m in M6:
        mu = [np.mean([R[f"{pref}{T}|{m}|s{s}"]["success"] for s in S5])
              for T in [0.5, 0.8, 1.1]]
        se = [np.std([R[f"{pref}{T}|{m}|s{s}"]["success"] for s in S5]) / np.sqrt(5)
              for T in [0.5, 0.8, 1.1]]
        mfc = "white" if m in ("full", "ar") else C[m]
        ax.errorbar([0.5, 0.8, 1.1], mu, yerr=se, marker=MK[m], ms=3.6, lw=1.2,
                    color=C[m], capsize=1.5, elinewidth=0.7, mfc=mfc,
                    mec=EC.get(m, C[m]), mew=0.9)
    ax.set_xlabel("Decode temperature")
    ax.set_title(title, fontsize=7.4)
    ax.set_xticks([0.5, 0.8, 1.1])
axes[0].set_ylabel("Task success")
axes[0].set_ylim(0.42, 1.03)
fig.legend([handle(m) for m in M6], [NAME[m] for m in M6],
           loc="outside upper center", ncol=5, frameon=False, fontsize=6.3)
save(fig, "fig6_temperature")

# -------- fig7: post-training decomposition (arms) + gradient diagnostics -----
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.5), layout="constrained")
ax = axes[0]
ARMS = [("curve", "reward-filtered (main)", "#D55E00", "-"),
        ("arm_replay_curve", "anchor only (cont.\\ SFT)", "#0072B2", "--"),
        ("arm_signed_curve", "signed advantages", "#009E73", "-"),
        ("arm_randsign_curve", "random-sign control", "#999999", ":"),
        ("arm_negonly_curve", "negative-only", "#CC79A7", "-")]
hnd, lbl = [], []
for key, lab, col, ls in ARMS:
    c = json.load(open(f"results/rl_{key}_seed0.json"))
    c = [d for d in c if d["it"] <= 100]
    ax.plot([d["it"] for d in c], [d["probe_success"] for d in c],
            color=col, ls=ls, lw=1.5)
    hnd.append(Line2D([], [], color=col, ls=ls, lw=1.5))
    lbl.append(lab.replace("\\ ", " "))
ax.set_xlabel("Post-training iteration (seed 0)")
ax.set_ylabel("Probe success ($n{=}1$, perturbed)")
ax.set_ylim(0, 1.05)
ax = axes[1]
D = json.load(open("results/rl_diag.json"))
labels = ["$p{<}0.5$", "$0.5$–$0.9$", "$0.9$–$0.99$", "$p{\\geq}0.99$"]
pos = [D["grad_mass"].get(f"pos|{i}", 0) for i in range(4)]
neg = [D["grad_mass"].get(f"neg|{i}", 0) for i in range(4)]
x = np.arange(4)
b1 = ax.bar(x - 0.18, pos, 0.34, color="#009E73")
b2 = ax.bar(x + 0.18, neg, 0.34, color="#CC79A7")
ax.set_yscale("log")
ax.set_ylim(0.15, 30)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_xlabel("One-step token confidence bin")
ax.set_ylabel("Logit-gradient mass (log)", fontsize=7.4)
hnd += [b1, b2]
lbl += ["positive advantage", "negative advantage"]
fig.legend(hnd, lbl, loc="outside upper center", ncol=4, frameon=False,
           fontsize=6.2)
save(fig, "fig7_rl_decomposition")
print("all figures regenerated")
