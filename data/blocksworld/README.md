# PlanBench Blocksworld (generated_basic) — normalized dataset

This directory stores a normalized, attribution-preserving copy of the official
PlanBench Blocksworld `generated_basic` instance set used by this repository's
`blocksworld` task.

## Source

- Repository: https://github.com/karthikv792/LLMs-Planning (MIT License)
- Upstream path: `plan-bench/instances/blocksworld/generated_basic/`
- Domain: `blocksworld-4ops` (actions `pick-up`, `put-down`, `stack`, `unstack`)
- Scope: **all 501 instances** (`instance-1.pddl` … `instance-501.pddl`),
  retrieved on 2026-08-01 from the `main` branch. No subsetting was applied.
- Upstream license text: see `LICENSE.txt`.

## Format

`blocksworld.jsonl` contains one JSON object per line. Each record is a
deterministic normalization of the corresponding upstream PDDL file and carries
its provenance:

| Field          | Meaning                                                              |
| -------------- | -------------------------------------------------------------------- |
| `id`           | stable identifier `planbench-blocksworld-{instance_id}`              |
| `instance_id`  | upstream instance number (1..501)                                    |
| `domain`       | PDDL domain name (`blocksworld-4ops`)                                |
| `source`       | upstream file URL for attribution                                     |
| `num_blocks`   | number of block objects                                               |
| `max_steps`    | per-instance depth bound, deterministic `4 * (num_blocks - 1)`        |
| `blocks`       | ordered list of block names                                           |
| `init`         | initial predicates `["handempty"]`, `["ontable", b]`, `["on", a, b]`, `["clear", b]` |
| `goal`         | goal predicates (all `["on", a, b]`)                                  |

`max_steps` is a safe deterministic upper bound: a worst-case arrangement may
force a block that must be relocated to be dismantled from a tower to the table
(`unstack` + `put-down`, 2 actions) and later rebuilt onto the goal tower
(`pick-up` + `stack`, 2 actions), and at most `num_blocks - 1` blocks ever need
relocating.  Hence `4 * (num_blocks - 1)` actions always admit a solution while
capping search depth (4 blocks -> 12, 5 blocks -> 16).

## Splits

`../splits/blocksworld.json` provides deterministic 70/15/15 train/val/test
index splits over the 501 records, generated with `random.Random(42)`. The
three parts are pairwise disjoint and their union covers every record.

## Runtime behavior

The task loads this file at construction time. There is **no runtime network
dependency**.
