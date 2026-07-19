"""fig8: dominance-supervision dose-response. fig9: masker operating curve.
Shares the style/save conventions of plots.py; audited by qa_overlap."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W_FULL = 5.5
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 6.6, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})

def save(fig, name):
    fig.savefig(f"figures/{name}.pdf")
    fig.savefig(f"figures/{name}.png")
    plt.close(fig)
    print("wrote", name)

DOSE = json.load(open("results/dose.json"))
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.4), layout="constrained")
doses, xpos = [0, 8, 64], [0, 1, 2]
for ax, key, ylab in [(axes[0], "p_escalate_composed",
                       r"$p(\mathrm{escalate} \mid \mathrm{composed\ ctx})$"),
                      (axes[1], "famA", "Family-A doubles success")]:
    means = []
    for xi, k in zip(xpos, doses):
        seeds = range(5) if k < 64 else range(3)
        vals = [DOSE[f"c1d{k}|s{s}"][key] for s in seeds]
        ax.scatter(np.full(len(vals), xi) + np.linspace(-0.10, 0.10, len(vals)),
                   vals, s=15, color="#0072B2", zorder=3)
        ax.hlines(np.mean(vals), xi - 0.24, xi + 0.24, color="#0072B2",
                  lw=1.8, zorder=2)
        means.append(np.mean(vals))
    ax.plot(xpos, means, color="#0072B2", lw=1.0, alpha=0.4, zorder=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["0\n(5 seeds)", "8\n(5 seeds)", "64\n(3 seeds)"],
                       fontsize=7)
    ax.set_xlabel("Dominance demonstrations in training")
    ax.set_ylabel(ylab, fontsize=7.6)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.5, 2.5)
save(fig, "fig8_dose_response")

TAU = json.load(open("results/tau.json"))
TAUS = [0.10, 0.20, 0.35, 0.50, 0.65]
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.45), layout="constrained")
CN = {1: "#0072B2", 2: "#D55E00"}
h_s, l_s = [], []
ax = axes[0]
for n in [1, 2]:
    mu = [np.mean([TAU[f"tau{t}|n{n}|s{s}"]["success"] for s in range(5)])
          for t in TAUS]
    se = [np.std([TAU[f"tau{t}|n{n}|s{s}"]["success"] for s in range(5)]) / np.sqrt(5)
          for t in TAUS]
    ax.errorbar(TAUS, mu, yerr=se, marker="o", ms=3.4, lw=1.3, color=CN[n],
                capsize=1.5, elinewidth=0.7)
    h_s.append(Line2D([], [], color=CN[n], marker="o", ms=3.4, lw=1.3))
    l_s.append(f"$n{{=}}{n}$" + (" (novel doubles)" if n == 2 else " (in-distribution)"))
ax.set_xlabel(r"Invalidation threshold $\tau_{\mathrm{inv}}$")
ax.set_ylabel("Task success")
ax.set_ylim(0.4, 1.03)
ax = axes[1]
for n in [1, 2]:
    P = [np.mean([TAU[f"tau{t}|n{n}|s{s}"]["P"] for s in range(5)]) for t in TAUS]
    Rr = [np.mean([TAU[f"tau{t}|n{n}|s{s}"]["R"] for s in range(5)]) for t in TAUS]
    ax.plot(TAUS, P, color=CN[n], lw=1.3, marker="s", ms=3.0)
    ax.plot(TAUS, Rr, color=CN[n], lw=1.3, ls="--", marker="^", ms=3.2)
ax.set_xlabel(r"Invalidation threshold $\tau_{\mathrm{inv}}$")
ax.set_ylabel("Masker precision / recall")
ax.set_ylim(0.4, 1.03)
h_s += [Line2D([], [], color="k", lw=1.1, marker="s", ms=3.0),
        Line2D([], [], color="k", lw=1.1, ls="--", marker="^", ms=3.2)]
l_s += ["precision", "recall"]
fig.legend(h_s, l_s, loc="outside upper center", ncol=4, frameon=False,
           fontsize=6.4)
save(fig, "fig9_tau_curve")

from scipy.stats import fisher_exact
a = sum(DOSE[f"c1d8|s{s}"]["p_escalate_composed"] > 0.9 for s in range(5))
b = sum(DOSE[f"c1d0|s{s}"]["p_escalate_composed"] > 0.9 for s in range(5))
_, p = fisher_exact([[a, 5 - a], [b, 5 - b]], alternative="greater")
print(f"Fisher exact (probe>0.9): dose8 {a}/5 vs dose0 {b}/5 -> one-sided p={p:.4f}")
pe0 = [DOSE[f"c1d0|s{s}"]["p_escalate_composed"] for s in range(5)]
pe8 = [DOSE[f"c1d8|s{s}"]["p_escalate_composed"] for s in range(5)]
print(f"probe: dose0 {np.mean(pe0):.3f}+/-{np.std(pe0):.3f}  dose8 {np.mean(pe8):.3f}+/-{np.std(pe8):.3f}")

# ---------------- fig10: scale and data robustness ----------------------------
PAR = {"w48": 46144, "c1": 240352, "w144": 526384}
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.5), layout="constrained")
CD = {"d0": "#E69F00", "d8": "#0072B2"}
def cellv(pref, k, seeds, key):
    return [(DOSE[f"{pref}d{k}|s{s}"][key], s) for s in seeds]
series = [  # (x, dosekey, seeds, prefix)
    (PAR["w48"], 0, [0, 1, 2], "w48"), (PAR["w48"], 8, [0, 1, 2], "w48"),
    (PAR["c1"], 0, [0, 1, 2, 3, 4], "c1"), (PAR["c1"], 8, [0, 1, 2, 3, 4], "c1"),
    (PAR["w144"], 0, [0, 1], "w144"), (PAR["w144"], 8, [0, 2], "w144"),
]
for ax, key, ylab in [(axes[0], "p_escalate_composed",
                       r"$p(\mathrm{escalate} \mid \mathrm{composed\ ctx})$"),
                      (axes[1], "famA", "Family-A doubles success")]:
    for x, k, seeds, pref in series:
        vals = [DOSE[f"{pref}d{k}|s{s}"][key] for s in seeds]
        xs = x * (1.10 ** np.linspace(-1, 1, len(vals))) * (0.86 if k == 0 else 1.16)
        ax.scatter(xs, vals, s=15, marker="o", zorder=3,
                   facecolors="none" if k == 0 else CD["d8"],
                   edgecolors=CD["d0"] if k == 0 else CD["d8"], linewidths=1.0)
    # 4x-data arm (dose 0, base width), x-dodged left
    dv = [DOSE[f"n16d0|s{s}"][key] for s in [0, 1, 2]]
    ax.scatter(PAR["c1"] * 0.62 * (1.07 ** np.linspace(-1, 1, 3)), dv, s=16,
               marker="s", facecolors="none", edgecolors="#666666",
               linewidths=1.0, zorder=3)
    # the excluded failed-training draw
    ax.scatter([PAR["w144"] * 1.16], [DOSE["w144d8|s1"][key]], s=26, marker="x",
               color="#CC3311", linewidths=1.2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (log)")
    ax.set_ylabel(ylab, fontsize=7.6)
    ax.set_ylim(-0.06, 1.09)
    ax.set_xticks([46144, 240352, 526384])
    ax.set_xticklabels(["46k", "240k", "526k"])
    ax.tick_params(axis="x", which="minor", bottom=False)
from matplotlib.lines import Line2D as L2
fig.legend([L2([], [], marker="o", mfc="none", mec=CD["d0"], ls="", ms=4.5),
            L2([], [], marker="o", color=CD["d8"], ls="", ms=4.5),
            L2([], [], marker="s", mfc="none", mec="#666666", ls="", ms=4.5),
            L2([], [], marker="x", color="#CC3311", ls="", ms=5)],
           ["dose 0", "dose 8", r"dose 0, $4\times$ data",
            "excluded draw (training spike)"],
           loc="outside upper center", ncol=4, frameon=False, fontsize=6.3)
save(fig, "fig10_scale")

# pooled exploratory Fisher across all healthy draws
cons0 = [DOSE[f"{p}d0|s{s}"]["p_escalate_composed"] > 0.9
         for p, ss in [("w48", [0,1,2]), ("c1", [0,1,2,3,4]), ("w144", [0,1])]
         for s in ss] + [DOSE[f"n16d0|s{s}"]["p_escalate_composed"] > 0.9 for s in [0,1,2]]
cons8 = [DOSE[f"{p}d8|s{s}"]["p_escalate_composed"] > 0.9
         for p, ss in [("w48", [0,1,2]), ("c1", [0,1,2,3,4]), ("w144", [0,2])]
         for s in ss]
a, b = sum(cons8), sum(cons0)
_, pp = fisher_exact([[a, len(cons8)-a], [b, len(cons0)-b]], alternative="greater")
print(f"pooled healthy draws: dose8 {a}/{len(cons8)}  dose0 {b}/{len(cons0)}  "
      f"exploratory one-sided Fisher p={pp:.2e}")
