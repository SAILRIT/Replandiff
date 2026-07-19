#!/usr/bin/env bash
# =============================================================================
# RePlanDiff -- full reproduction pipeline
#
# Reproduces every number, table, and figure in the paper from scratch on a
# single CPU core in ~6-7 hours total. Every stage checkpoints and resumes:
# re-running this script (or any stage) after an interruption continues from
# the last checkpoint and skips completed evaluation cells.
#
# Quick alternative: all evaluation outputs are committed under results/, so
#   python3 plots.py && python3 plots_extra.py && python3 gen_paper_tables.py
# regenerates every figure and table in minutes without retraining anything.
# =============================================================================
set -e
mkdir -p runs logs results figures

echo "=== Stage 1: base SFT (5 seeds x {diffusion, AR}, 2000 steps) ~80 min ==="
for s in 0 1 2 3 4; do
  python3 train_sft.py $s runs 2000 64 diffusion _v2 | tee -a logs/sft.log
  python3 train_sft.py $s runs 2000 64 ar        _v2 | tee -a logs/sft.log
done

echo "=== Stage 2: supervision-scarce SFT (lite, 425 steps, 5 seeds) ~8 min ==="
for s in 0 1 2 3 4; do
  python3 train_sft.py $s runs 425 64 diffusion _lite | tee -a logs/lite.log
done

echo "=== Stage 3: post-training, reward-filtered imitation (5 seeds) ~35 min ==="
for s in 0 1 2 3 4; do
  python3 train_rl.py $s runs 200 | tee -a logs/rl.log
done

echo "=== Stage 4: advantage-transform ablation arms (seed 0) ~12 min ==="
for mode in replay signed negonly randsign; do
  python3 train_rl.py 0 runs 100 $mode | tee -a logs/rl_arms.log
done
python3 rl_diag.py | tee logs/rl_diag.log     # confidence/gradient-mass diagnostics

echo "=== Stage 5: main evaluation grid (Table 1, Figs 1-6) ~35 min ==="
for s in 0 1 2 3 4; do python3 eval.py runs 400 $s; done

echo "=== Stage 6: masker operating curve (Fig 9) ~12 min ==="
python3 eval_tau.py 0,1,2,3,4

echo "=== Stage 7: supervision-dose grid (Fig 8, Appendix D) ~75 min ==="
# cov=1.0, 1500 steps; doses {0,8} x seeds 0-4, dose 64 x seeds 0-2
for s in 0 1 2 3 4; do
  python3 train_sft.py $s runs 1500 64 diffusion _c1d0 1.0 0  | tee -a logs/dose.log
  python3 train_sft.py $s runs 1500 64 diffusion _c1d8 1.0 8  | tee -a logs/dose.log
done
for s in 0 1 2; do
  python3 train_sft.py $s runs 1500 64 diffusion _c1d64 1.0 64 | tee -a logs/dose.log
done

echo "=== Stage 8: compute-only control (warm-start failing seeds) ~18 min ==="
for s in 1 4; do
  python3 train_sft.py $s runs 2000 64 diffusion _v2x 0.1 0 _v2 | tee -a logs/ext.log
done

echo "=== Stage 9: scale & data sweep (Fig 10) ~70 min ==="
for s in 0 1 2; do
  MODEL_D=48 MODEL_L=2 python3 train_sft.py $s runs 1500 64 diffusion _w48d0 1.0 0 | tee -a logs/scale.log
  MODEL_D=48 MODEL_L=2 python3 train_sft.py $s runs 1500 64 diffusion _w48d8 1.0 8 | tee -a logs/scale.log
  N_INST=16000        python3 train_sft.py $s runs 1500 64 diffusion _n16d0 1.0 0 | tee -a logs/scale.log
done
for s in 0 1; do
  MODEL_D=144 MODEL_L=3 python3 train_sft.py $s runs 1500 64 diffusion _w144d0 1.0 0 | tee -a logs/scale.log
  MODEL_D=144 MODEL_L=3 python3 train_sft.py $s runs 1500 64 diffusion _w144d8 1.0 8 | tee -a logs/scale.log
done
# replacement for the documented failed-training draw (see Appendix D)
MODEL_D=144 MODEL_L=3 python3 train_sft.py 2 runs 1500 64 diffusion _w144d8 1.0 8 | tee -a logs/scale.log

echo "=== Stage 10: dose/probe/scale evaluations ~25 min ==="
python3 eval_dose.py dose
python3 eval_dose.py base
python3 eval_dose.py ext
python3 eval_dose.py scale

echo "=== Stage 11: figures, tables, overlap audit ~5 min ==="
python3 plots.py
python3 plots_extra.py
python3 qa_overlap.py            # geometric legend/text-vs-data audit (10/10 expected)
python3 gen_paper_tables.py      # writes paper/tab_main.tex, paper/tab_family.tex

echo "=== Stage 12: build the paper ==="
make -C paper

echo "All done. Paper at paper/main.pdf; figures in figures/; results in results/."
