# NudgeRL

Official code for **Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR**.

NudgeRL is a reinforcement learning with verifiable rewards (RLVR) framework for improving exploration in mathematical reasoning. Instead of increasing the rollout budget by brute force, NudgeRL samples lightweight strategy-level contexts, uses them to induce diverse reasoning trajectories, and transfers the discovered behavior back to the base policy.

This repository includes:

- `NudgeRLTrainer`, implementing strategy-conditioned rollouts, inter-intra group advantage estimation, and a distillation term.
- GRPO and POPE-style oracle-prefix baselines.
- Dataset builders for DAPO-Math-17k strategy contexts and POPE-style oracle prefixes.
- vLLM-based evaluation on AIME, AMC23, MATH500, and Apex Shortlist.

## Repository Layout

```text
configs/
  context_config.yaml       # API/model settings for generating contexts and oracle prefixes
  train_config.yaml         # training hyperparameters
scripts/
  build_dataset.sh          # build strategy-context dataset
  build_pope_st_dataset.sh  # build POPE-style oracle-prefix dataset
  train_grpo.sh             # GRPO baseline
  train_grpope.sh           # POPE-style baseline
  train_nudgerl.sh          # NudgeRL training
  eval_model.sh             # LoRA adapter evaluation wrapper
src/
  data/                     # dataset downloaders and benchmark loaders
  train/                    # GRPO, POPE, and NudgeRL training code
  eval_model.py             # evaluate a trained LoRA adapter
  eval_model_base.py        # evaluate a base Hugging Face model
  verify.py                 # math-verify reward wrapper
```

## Installation

Training and evaluation require CUDA-compatible PyTorch and vLLM. The commands below assume a Linux CUDA environment.

```bash
conda create -n nudgerl python=3.11
conda activate nudgerl
pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

If the pinned `torch`, `vllm`, `xformers`, or `flashinfer-python` wheels do not match your CUDA setup, install matching wheels for your machine before running training.

## Data Preparation

Download the base DAPO-Math-17k dataset:

```bash
python -m src.data.download_dapo17k
```

The checked-in training scripts expect these processed files:

- `data/dapo17k_contexts_5_samples_500.jsonl` for NudgeRL.
- `data/dapo17k_pope_500.jsonl` for the POPE-style baseline.

To regenerate the strategy-context dataset, edit `configs/context_config.yaml` and set your OpenAI API key and context-generation model:

```bash
bash scripts/build_dataset.sh
```

To regenerate the POPE-style oracle-prefix dataset, set the DeepSeek API key under `pope_style` in `configs/context_config.yaml`:

```bash
bash scripts/build_pope_st_dataset.sh
```

If you change `num_contexts` or `num_samples`, update the corresponding `data_files` path in the training script, since the current release uses the 500-sample files above.

## Training

Edit `configs/train_config.yaml` before launching a run:

- `model_name`: base model, for example `Qwen/Qwen3-4B-Instruct-2507`.
- `WANDB_API_KEY` and `wandb_project`: Weights & Biases logging.
- `max_steps`, `save_steps`, batch sizes, sequence lengths, and sampling parameters.
- `nudgerl`: NudgeRL-specific hyperparameters.

Run NudgeRL:

```bash
bash scripts/train_nudgerl.sh
```

Run GRPO and POPE-style baselines:

```bash
bash scripts/train_grpo.sh
bash scripts/train_grpope.sh
```

The main training entry points also accept useful overrides:

```bash
python -m src.train.train_nudgerl --config configs/train_config.yaml --eps_high 0.2
python -m src.train.train_grpo --config configs/train_config.yaml --num_rollouts 8 --eps_high 0.2
python -m src.train.train_grpope --config configs/train_config.yaml --num_rollouts 8 --eps_high 0.2
```

Adapters are saved under:

```text
outputs/models/<base-model-id>/<method-name>/
```

For the default Qwen config, NudgeRL saves to a path like:

```text
outputs/models/Qwen3-4B-Instruct-2507/NudgeRL_2x4_eps20/
```

## Evaluation

Evaluate a trained LoRA adapter:

```bash
bash scripts/eval_model.sh \
  outputs/models/Qwen3-4B-Instruct-2507/NudgeRL_2x4_eps20 \
  --datasets AIME,AMC23,MATH500,APEX_SHORTLIST \
  --num-samples 128 \
  --estimate-k 16 \
  --batch-size 128
```

Evaluate a base model without LoRA:

```bash
python -m src.eval_model_base \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --datasets AIME,AMC23,MATH500,APEX_SHORTLIST \
  --num-samples 128 \
  --estimate-k 16 \
  --batch-size 128
```

Supported dataset names are:

```text
AIME, AMC23, MATH500, APEX_SHORTLIST
```

Benchmark data is downloaded and cached under `data/` on first use. Evaluation JSON files are written under `outputs/evals/` unless `--output-json` is provided.

## Important Configuration Values

NudgeRL defaults in `configs/train_config.yaml`:

| Key | Meaning |
| --- | --- |
| `nudge_grpo.num_hints` | Number of strategy contexts sampled per problem ($\vert\mathcal{C}(x)\vert$).|
| `nudge_grpo.rollouts_per_hint` | Rollout budget allocated per strategy context ($N/\vert\mathcal{C}(x)\vert$). |
| `nudge_grpo.p_dropout` | Probability of dropping the context and sampling from the original prompt ($p_\text{drop}$). |
| `nudge_grpo.adv_lbd` | Inter-context advantage weight ($\lambda$). |
| `nudge_grpo.distill_coeff` | Distillation weight from context-conditioned rollouts to the base prompt. ($\lambda_\text{distill}$) |

Paper-style defaults use LoRA rank 32, max prompt length 2048, max completion length 6144, 500 RL steps, AdamW 8-bit, learning rate `2e-5`, and `2 x 4` NudgeRL rollouts.
