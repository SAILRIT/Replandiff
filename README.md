# Repair, Don't Replan

**A controlled study of observation-conditioned remasking in masked-diffusion planners.**

This repository contains everything needed to reproduce the paper end-to-end:
environment, models, training, evaluation, statistics, figures, and the LaTeX
source. All experiments run on a **single CPU core** (~6–7 hours total, fully
checkpointed) — or regenerate every figure and table **in minutes** from the
committed `results/` without retraining.

> **Scope, stated up front.** This is a mechanism study on a synthetic tool-use
> environment with 46k–526k-parameter models trained from scratch. It makes no
> claim about 7–8B diffusion LLMs; what it establishes is that the mechanism is
> implementable, its efficiency gains are real at matched success in this
> setting, and its compositional failure mode is structured, diagnosable, and
> fixable by a targeted supervision intervention. The paper's §1 (Scope) and §6
> (Limitations) hold this line explicitly.

## Findings at a glance

| Agent | Success n=1 | Success n=2 (novel) | Regen. tokens | NFE |
|---|---|---|---|---|
| Fixed plan (no repair) | 0.000 | 0.000 | 0.0 | 7 |
| Full replan | 0.966 ± 0.067 | 0.764 ± 0.210 | 83.3 | 33 |
| AR replan (suffix) | 0.998 ± 0.004 | 0.754 ± 0.155 | 84.0 | 108 |
| AR patch | 1.000 ± 0.001 | 0.884 ± 0.211 | 2.0 | 33 |
| **Selective (conf.)** | 0.974 ± 0.052 | 0.785 ± 0.239 | **1.9** | **27** |
| Selective (oracle) | 0.977 ± 0.046 | **0.996 ± 0.006** | 1.9 | 20 |

*(5 seeds, 400 episodes/cell; n = perturbations per episode; n=2 combinations
are held out from training.)*

1. **Selective remasking is ~44× cheaper at matched success.** All
   repair-capable agents solve single-perturbation episodes; they differ by
   1.5 orders of magnitude in regenerated tokens and 4× in network calls. The
   learned masker is oracle-grade in-distribution (precision 0.92 / recall
   0.90 against a computable ground-truth invalidation set).
2. **Localization, not repair synthesis, is the compositional bottleneck.**
   On never-demonstrated perturbation *pairs*, every learned agent drops while
   the same denoisers with oracle masking hold 0.996. The failure concentrates
   entirely in a family whose two perturbations conflict under a dominance
   rule absent from training — and a single conditional,
   p(escalate | composed context), predicts each seed's end-to-end outcome by
   whether it clears the invalidation threshold.
3. **Eight demonstrations of the priority rule fix it; nothing else does.**
   Dose–response: 0/8/64 dominance demonstrations move the composed
   conditional from 0.408 ± 0.341 (1/5 seeds consistent) to 0.997 ± 0.002
   (5/5; Fisher exact p = 0.024), while 10× more single-perturbation
   supervision, 2× compute (which *entrenches* the wrong draw), 2.1× model
   capacity, and 4× data all fail. Scale-stable across a 46k–526k parameter
   sweep: healthy dose-8 draws are dominance-consistent 10/10 vs 3/13 at
   dose 0 (exploratory pooled p = 2.5e-4).
4. **Post-training decomposition (with a public self-correction).** From a
   supervision-scarce start, reward-filtered imitation restores repair on
   every seed — but an anchor-only control matches it (the gains are mostly
   continued SFT), isolated negative advantages collapse the policy, and the
   paper retracts its own earlier "saturation asymmetry" explanation with the
   correct derivation and measurements (Appendix C).

All anomalies encountered along the way (a training-lottery repair-degenerate
draw, conditional-corruption pathologies, one loss-spike draw excluded with a
replacement seed) are disclosed in the paper's appendices, not smoothed over.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 5-minute smoke test: tiny training run + evaluation + one figure
bash smoke_test.sh

# regenerate every figure & table from the committed results (no training)
python3 plots.py && python3 plots_extra.py && python3 gen_paper_tables.py
python3 qa_overlap.py        # geometric figure-quality audit (expects 10/10)
make -C paper                # build the paper PDF
```

## Full reproduction

```bash
bash run_all.sh              # ~6-7 h on one CPU core, fully resumable
```

`run_all.sh` is organized into 12 commented stages (base SFT → post-training →
ablation arms → main eval → τ sweep → dose grid → compute control → scale
sweep → dose/probe evals → figures/audit → paper). Every training script
checkpoints every ~400 steps and every evaluation script writes results
incrementally and skips completed cells, so interrupting and re-running any
stage is safe. Determinism: environment instances, datasets, and evaluation
sets are seeded; training is deterministic given (seed, config) on CPU JAX.

## Where every number comes from

Tables are **generated, never hand-typed**: `gen_paper_tables.py` emits
`paper/tab_main.tex` and `paper/tab_family.tex` directly from
`results/results.json`, so the paper cannot drift from the data.

| Paper claim | Data | Regenerated by |
|---|---|---|
| Table 1, Figs 1–6 (main grid, temp sweeps, holes) | `results/results.json` | `eval.py` → `plots.py`, `gen_paper_tables.py` |
| Fig 7, Appendix C (post-training decomposition) | `results/rl_curve_seed*.json`, `results/rl_arm_*_seed0.json`, `results/rl_diag.json` | `train_rl.py <s> runs 200 [mode]`, `rl_diag.py` |
| Fig 8, §4 & Appendix D (dose–response, Fisher p=0.024) | `results/dose.json` (`c1d*` keys) | `eval_dose.py dose` → `plots_extra.py` |
| §4 probe (base models) & compute control | `results/dose.json` (`base\|*`, `ext\|*` keys) | `eval_dose.py base`, `eval_dose.py ext` |
| Fig 9 (τ operating curve) | `results/tau.json` | `eval_tau.py` |
| Fig 10, Appendix D (scale & data sweep, pooled p=2.5e-4) | `results/dose.json` (`w48*`, `w144*`, `n16*` keys) | `eval_dose.py scale` → `plots_extra.py` |

## Repository layout

```
toolflow.py            ToolFlow-v2 environment: 8-step plans, 22 tools, dataflow
                       variables, state-diff success, 5 observation-revealed
                       perturbations, oracle invalidation set, compositional split
model.py               ~250k-param JAX transformer (bidirectional denoiser / causal AR);
                       width/depth read from checkpoints at inference
data.py                Expert rollouts, reveal-aware training snapshots, dose injection
agent.py               Denoise loop, typed holes, posterior-rescoring masker, all six
                       agents (fixed / full / oracle / conf / AR replan / AR patch),
                       masker precision-recall instrumentation
train_sft.py           SFT with resume; env knobs MODEL_D / MODEL_L / N_INST; warm-start
train_rl.py            Post-training: reward-filtered imitation + ablation arms
                       (replay / signed / negonly / randsign / raftbc)
eval.py                Main grid: 8 methods x n∈{0,1,2} x 5 seeds + temp sweeps + holes
eval_dose.py           Dose grid, base/ext/scale probes (composed-conditional endpoint)
eval_tau.py            Masker operating curve; calibration spot-checks
rl_diag.py             Confidence histograms + exact per-token gradient-mass binning
plots.py / plots_extra.py   All 10 figures (Times-metric fonts, outside legends)
qa_overlap.py          Geometric audit: every legend/text bbox vs every data artist
gen_paper_tables.py    Emits paper tables from results.json
run_all.sh             The 12-stage pipeline;  smoke_test.sh  a 5-minute sanity check
results/               Committed evaluation outputs (regenerate figures without training)
figures/               Committed figure PDFs/PNGs
paper/                 LaTeX source, style, bib, Makefile, built PDF (see paper/README.md
                       for the NeurIPS style-file swap and bibliography notes)
legacy_v1/             The v1 pilot environment and results referenced historically
```

## Notes for reviewers / users

- **Style file:** `paper/neuripsstyle.sty` is a layout-compatible
  reimplementation (the official file was not fetchable in the build sandbox).
  Swap one line for the official workshop style before submission — see
  `paper/README.md`.
- **Placeholders to fill:** the author block in `paper/main.tex`, `LICENSE`
  copyright holder, and `CITATION.cff` authors.
- **Compute:** developed and fully reproduced on 1 CPU core / 3 GB RAM
  (JAX CPU). No GPU is required; none was used.

## License

MIT — see `LICENSE`.
