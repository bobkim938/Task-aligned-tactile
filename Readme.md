# Phase-Aware Tactile Representation Learning

Code and data for **Phase-Aware Tactile Representation Learning**.

We learn a tactile encoder that compensates for the object-pose information lost
to visual occlusion during contact-rich manipulation. A robot cannot see an
object once its gripper closes around it; the encoder turns the raw contact
forces into a representation the downstream policy can act on. The method has two
components:

1. **Data-driven auxiliary-head selection** — auxiliary supervision targets are
   *derived from expert rollouts* by a discriminability-and-learnability
   criterion, rather than hand-picked. Five factors pass (contact, bilateral
   force symmetry, grasp stability, manipulation phase, force magnitude); slip
   prediction is rejected as not learnable from force history.
2. **A hinge temporal-consistency loss** — a margin-based penalty that compresses
   within-phase latent oscillation while the auxiliary-head objectives preserve
   phase-transition structure, keeping the representation stable enough for RL
   without flattening the phase information the policy needs.

The encoder is pre-trained, then **frozen** as an observation module for a PPO
policy on a 62-object YCB pick-and-place task (ManiSkill3 / SAPIEN) under a
proxy occlusion model.

## Key results

Across four observation conditions (10 seeds, 6,200 episodes each):

| Condition       | Success (%)      | Gap to Oracle |
| --------------- | ---------------- | ------------- |
| Oracle          | 95.08 ± 1.01     | ---           |
| Encoder (ours)  | 92.10 ± 1.31     | 2.98          |
| Raw Tactile     | 79.50 ± 2.33     | 15.58         |
| Visual-only     | 70.84 ± 1.61     | 24.24         |

The encoder recovers **87.7%** of the Visual-to-Oracle gap. The headline is the
**12.6-point** gap between Encoder and Raw Tactile (Welch's *t* on per-seed
means, *t* = 14.91): both receive identical 6D contact forces, differing only in
whether the policy consumes the raw vector or the pre-trained latent — so the
benefit comes from the *representation*, not from additional information.

## Repository structure

```
env/                     Simulation environment (PickAndPlaceYCBToTray task)
tactile_encoder/         Encoder architecture + pre-training
training/                PPO training script
evaluation/              Multi-seed evaluation + seed-level statistics
analysis/                Figure generation + probes (all analysis scripts)
data/encoder_pretraining/ Expert-rollout pre-training data (.h5; Hugging Face)
data/                    Unified deployment records (.npz) read by analysis (v1, v2, 4_4_heldout)
checkpoints/             Final encoder + policy checkpoints
results/                 Raw and summary evaluation CSVs, analysis results
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ManiSkill3 / SAPIEN must be installed separately per their instructions.
```

## Pre-training Data

**Pre-training data (Hugging Face).** The expert-rollout dataset used to
pre-train the encoder (`clean.h5`, `low_friction.h5`, `noisy.h5`; ~8.5M
timesteps across 62 YCB objects) is hosted on Hugging Face. Clone it into
`data/encoder_pretraining/`:

```bash
git clone https://huggingface.co/datasets/kingIronMan777/phase-aware-tact-pretraining-data \
    data/encoder_pretraining
```

This data is only needed to **retrain the encoder from scratch** (stage 1). To
reproduce the figures and numbers, the pre-trained checkpoints below are
sufficient and the pre-training data is not required.


## Reproducing the paper

The pipeline has four stages. To reproduce **the figures and numbers only** (no
retraining), skip to stage 4 — the shipped checkpoints and deployment `.npz`
files are sufficient.

### 1. Pre-train the tactile encoder *(optional; checkpoints shipped)*

Requires the pre-training data from Hugging Face (see *Data*
above), placed in `data/encoder_pretraining/`.

```bash
python tactile_encoder/train_tactile_encoder.py \
        --data-dir data/encoder_pretraining \
        --output-dir runs/tactile_encoder_v2 \
        --epochs 30 \
        --batch-size 512 \
        --lr 3e-4 \
        --w-temporal 0.5
# produces the hinge encoder; set --w-temporal 0 for the no-hinge ablation
```

Optionally verify the data reproduces the head-selection analysis
(symmetry separation, slip non-learnability, class balance):

```bash
python tactile_encoder/analyze_encoder_data.py \
    --data-dir data/encoder_pretraining --window-size 16
```

### 2. Train the four ablation policies *(optional; checkpoints shipped)*

```bash
python training/ppo.py --env_id=PickAndPlaceYCBToTray-v1 --exp-name policy_oracle --seed 4796 --num-envs 4096 --num-steps 16 --num-eval-envs 8 --num-eval-steps 100 --update-epochs 8 --num-minibatches 32 --total-timesteps 35000000 --gamma 0.93 --track --capture-video --save-model

python training/ppo.py --env_id=PickAndPlaceYCBToTray-v1 --exp-name policy_encoder --seed 4796 --num-envs 4096 --num-steps 16 --num-eval-envs 8 --num-eval-steps 100 --update-epochs 8 --num-minibatches 32 --total-timesteps 35000000 --gamma 0.93 --track --capture-video --save-model

python training/ppo.py --env_id=PickAndPlaceYCBToTray-v1 --exp-name policy_raw_tactile --seed 4796 --num-envs 4096 --num-steps 16 --num-eval-envs 8 --num-eval-steps 100 --update-epochs 8 --num-minibatches 32 --total-timesteps 40000000 --gamma 0.93 --track --capture-video --save-model

python training/ppo.py --env_id=PickAndPlaceYCBToTray-v1 --exp-name policy_visual_only --seed 4796 --num-envs 4096 --num-steps 16 --num-eval-envs 8 --num-eval-steps 100 --update-epochs 8 --num-minibatches 32 --total-timesteps 100000000 --gamma 0.93 --track --capture-video --save-model
```

### 3. Evaluate + collect deployment data *(optional; .npz shipped)*

```bash
python evaluation/eval_multiseed.py                    # -> results/multiseed_eval_raw.csv
python analysis/collect_.py --tag v2                   # -> data/v2/
```

### 4. Generate numbers and statistics *(uses shipped checkpoints + data)*

```bash
# Downstream policy performance -- per-variant / per-category seed-level tests
python evaluation/recompute_seedlevel_pvalues.py \
      --csv results/multiseed_eval_raw.csv \
      --cond-a visual_encoder --cond-b visual_raw_tact
python analysis/per_category_analysis.py

# Latent-space analysis figure (phase probe, category distances, correlations)
python analysis/latent_space_analysis.py

# Hinge / transport compression figure + numbers
python analysis/plot_transport_dz.py
python analysis/verify_hinge_margin_claim.py

# Grasp-phase asymmetry separation figure
python analysis/plot_grasp_asymmetry.py

# Phase-functional probes (grasp outcome + transport outcome + slip mechanism)
python analysis/probe_grasp_outcome.py
python analysis/probe_transport_functional.py
```

Every figure script also writes its underlying data as `.csv` under `results/`, so
plots can be regenerated in any tool without rerunning the pipeline.

## Method notes

- **Occlusion model.** The observed object position is blended toward the tool
  center point as the gripper closes, `p̂ = (1−α)·p_obj + α·p_tcp`, degrading
  rather than removing the privileged signal so that cross-condition differences
  isolate the tactile representation.
- **Encoder.** A temporal CNN over a 16-frame force window → 16-dimensional
  latent `z_t`. Auxiliary heads are used only in pre-training and discarded at
  deployment; the policy consumes only `z_t`.
- **Hinge loss.** The penalty `max(0, ‖z_t − z_{t−1}‖ − m)` is zero below the
  margin `m`, so once oscillation is small the temporal penalty switches off and
  the auxiliary-head objectives shape the representation unopposed. Within-phase
  oscillation compresses ~16×; phase transitions stay proportionally distinct
  (transition/within-phase step-distance ratio preserved, 8.8× → 10.7×) because
  the auxiliary heads require phase-distinct latents — structure an always-on L2
  penalty would erode.
- **Statistics.** All multi-seed, per-variant, and per-category significance
  tests use per-seed success rates (Welch's *t*, Mann-Whitney *U* confirming),
  never episode-level tests, which would treat clustered episodes as independent.

## Checkpoint naming

- `checkpoints/tactile_encoder_hinge/best_encoder.pt` — full encoder with the hinge loss
  (used for all downstream policies).
- `checkpoints/tactile_encoder_no_hinge/best_encoder.pt` — ablation encoder without the
  temporal loss.
- `checkpoints/policy_{oracle,encoder,raw_tactile,visual_only}.pt` — the four
  ablation policies.

## Acknowledgments

The simulation environment and PPO training build on
[ManiSkill3](https://github.com/haosulab/ManiSkill) (Tao et al., RSS 2025),
running on the [SAPIEN](https://sapien.ucsd.edu/) physics engine (Xiang et al.,
CVPR 2020). Our policies adapt the ManiSkill PPO baseline
([examples/baselines/ppo](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo)),
which is itself based on [CleanRL](https://github.com/vwxyzjn/cleanrl)
(Huang et al., 2022) and implements PPO (Schulman et al., 2017); we modified it
to consume the frozen tactile latent `z_t` as an observation. ManiSkill is
released under the Apache-2.0 license; the adapted `training/ppo.py` retains its
original notices.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{TODO_bibkey,
  title     = {Phase-Aware Tactile Representation Learning},
  author    = {Min Joong Kim, Zhengrong Ling, Yajing Shen},
  booktitle = {TODO},
  year      = {2026}
}
```

## License
This project is released under [LICENSE](LICENSE).