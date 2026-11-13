# Script recipes

All runnable recipes now call the canonical module entry point:

```bash
PYTHONPATH=src .venv/bin/python -m federated_mcts ...
```

The installed equivalents are `fmcts-run` (search) and `fmcts-exp` (the
historical sampling sweep in `src/federated_mcts/experiments/sweep.py`).

The recipes run from the repository root. They inherit the caller's model
configuration and credentials; none of them select or substitute a model on
their own.

## Converted recipes

| Recipe | What changed |
| --- | --- |
| `scripts/exp_strategies.sh` | `python scripts/merged_run.py` → `.venv/bin/python -m federated_mcts`; all arguments unchanged. |
| `scripts/game24/eval.sh` | `python merged_run.py` → `.venv/bin/python -m federated_mcts`; all arguments unchanged. |

## Deleted recipes

These were tied to entry points that no longer exist (root `run.py`,
`merged_run.py`, or `scripts/merged_run.py`). They were deleted rather than
silently rewritten because a faithful conversion would have changed the model
or the algorithm:

| Recipe | Why it cannot be migrated faithfully |
| --- | --- |
| `scripts/game24/bfs.sh`, `cot_sampling.sh`, `standard_sampling.sh` | Used the deleted root `run.py`, which selected the model via `--backend`. There is no equivalent canonical flag; choosing a `--model_config` would change the model. |
| `scripts/text/*.sh`, `scripts/crosswords/*.sh` | Same deleted `run.py` / `--backend` model selection. |
| `scripts/gsm8k/baseline.sh` | Passed `--model_config` to the deleted `run.py`, whose parser never accepted that flag. |
| `scripts/game24/eval_deepseek.sh`, `eval_deepseek_pre.sh`, `mid.sh`, `mid_baseline.sh`, `simple.sh`, `simple_base.sh` | Pinned `deepseek-v3`, which the current strict rate card cannot price, and some ran non-`tot` Game24 experiments over more than 20 tasks, which the current runner safety gate rejects. |
| `scripts/game24/eval_qwen.sh` | Pinned `Qwen-2.5-32B-Instruct`, which has no strict rate-card entry. |
| `scripts/gsm8k/eval.sh` | Pinned `deepseek-v3`, which the current strict rate card cannot price. |
| `scripts/game24/baseline.sh` | Empty placeholder; carried no recipe. |

The historical logs and result files remain untouched. Re-create a deleted
recipe only together with an explicit, priced model configuration and a
decision configuration that the current safety gates accept.
