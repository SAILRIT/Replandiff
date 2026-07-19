"""Emit paper tables straight from results/results.json (no hand-typed numbers).
Writes paper/tab_main.tex, tab_family.tex, and prints a summary."""
import json
import numpy as np
from data import eval_instances

R = json.load(open("results/results.json"))
S5, S3 = [0, 1, 2, 3, 4], [0, 1, 2]
OUT = "paper"

NAME = {"fixed": "Fixed plan (no repair)", "full": "Full replan",
        "ar": "AR replan (suffix)", "ar_patch": "AR patch",
        "sel_conf": "\\textbf{Selective (conf.)}", "sel_oracle": "Selective (oracle)",
        "sel_conf_lite": "\\quad lite-SFT", "sel_conf_rl": "\\quad lite-SFT + RL"}

def cell(m, p, k, seeds):
    return np.array([R[f"{m}|p{p}|s{s}"][k] for s in seeds])

def ms(m, p, k, seeds, prec=3):
    v = cell(m, p, k, seeds)
    return f"${v.mean():.{prec}f}${{\\scriptsize$\\pm${v.std():.{prec}f}}}".replace(
        "{\\scriptsize$\\pm$", "{\\,\\scriptsize$\\pm$\\,")

def pr(m):
    tp, fp = cell(m, 2, "mask_tp", S5), cell(m, 2, "mask_fp", S5)
    fn = cell(m, 2, "mask_fn", S5)
    return f"${np.mean(tp/(tp+fp)):.2f}$ / ${np.mean(tp/(tp+fn)):.2f}$"

rows = []
for m in ["fixed", "full", "ar", "ar_patch", "sel_conf", "sel_oracle"]:
    rows.append(" & ".join([
        NAME[m], ms(m, 0, "success", S5), ms(m, 1, "success", S5),
        ms(m, 2, "success", S5), ms(m, 1, "regen", S5, 1), ms(m, 1, "nfe", S5, 0),
        ms(m, 1, "invalid", S5, 2),
        pr(m) if m in ("sel_conf", "ar_patch") else "--",
    ]) + r" \\")
rl_rows = []
for m in ["sel_conf_lite", "sel_conf_rl"]:
    rl_rows.append(" & ".join([
        NAME[m], ms(m, 0, "success", S5), ms(m, 1, "success", S5),
        ms(m, 2, "success", S5), ms(m, 1, "regen", S5, 1), ms(m, 1, "nfe", S5, 0),
        ms(m, 1, "invalid", S5, 2), "--",
    ]) + r" \\")

tab = r"""\begin{tabular}{@{}lccccccc@{}}
\toprule
& \multicolumn{3}{c}{Task success $\uparrow$} & \multicolumn{3}{c}{Cost / errors at $n{=}1$ $\downarrow$} & Masker \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Agent & $n{=}0$ & $n{=}1$ & $n{=}2$ (novel) & Regen. & NFE & Invalid & P / R ($n{=}2$) \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{8}{@{}l}{\emph{Supervision-scarce condition (5 seeds): selective (conf.) agent, 425-step SFT}}\\
""" + "\n".join(rl_rows) + r"""
\bottomrule
\end{tabular}"""
open(f"{OUT}/tab_main.tex", "w").write(tab + "\n")

# ---- per-family composition table -------------------------------------------
def fam(seed):
    return np.array([i.family for i in eval_instances(10_000 + seed * 100 + 20, 400, 2)])

frows = []
for m in ["full", "ar", "ar_patch", "sel_conf", "sel_oracle"]:
    a, c = [], []
    for s in S5:
        sl = np.array(R[f"{m}|p2|s{s}"]["success_list"]); f = fam(s)
        a.append(sl[f == "A"].mean()); c.append(sl[f == "C"].mean())
    a, c = np.array(a), np.array(c)
    frows.append(f"{NAME[m]} & ${a.mean():.3f}$\\,{{\\scriptsize$\\pm$\\,{a.std():.3f}}} & "
                 f"[{', '.join(f'{x:.2f}' for x in a)}] & "
                 f"${c.mean():.3f}$\\,{{\\scriptsize$\\pm$\\,{c.std():.3f}}} \\\\")
ftab = r"""\begin{tabular}{@{}lccc@{}}
\toprule
& \multicolumn{2}{c}{Family A: \emph{conflicting} (dominance rule)} & Family C: \emph{independent} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}
Agent & Success & per-seed & Success \\
\midrule
""" + "\n".join(frows) + r"""
\bottomrule
\end{tabular}"""
open(f"{OUT}/tab_family.tex", "w").write(ftab + "\n")

# ---- numbers quoted in prose ------------------------------------------------
print("hole ablation (5 seeds):")
for tag in ["hole_on", "hole_off"]:
    su = np.array([R[f"{tag}|s{s}"]["success"] for s in S5])
    iv = np.array([R[f"{tag}|s{s}"]["invalid"] for s in S5])
    print(f"  {tag}: succ {su.mean():.3f}±{su.std():.3f}  invalid {iv.mean():.3f}±{iv.std():.3f}")
print("RL curves:")
for s in S5:
    c = json.load(open(f"results/rl_curve_seed{s}.json"))
    print(f"  seed {s}: {c[0]['probe_success']:.3f} -> {c[-1]['probe_success']:.3f}")
print("temp sweeps (mean over 5 seeds):")
for pref, lab in [("U", "n=1"), ("T", "n=2")]:
    for T in [0.5, 0.8, 1.1]:
        r = {m: np.mean([R[f"{pref}{T}|{m}|s{s}"]["success"] for s in S5])
             for m in ["full", "ar", "ar_patch", "sel_conf", "sel_oracle"]}
        print(f"  {lab} T={T}: " + " ".join(f"{k} {v:.3f}" for k, v in r.items()))
print("wrote tab_main.tex, tab_family.tex")
