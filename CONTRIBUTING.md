# Contributing to TeleBoost

Thanks for considering a contribution.  This codebase is a research
recipe on top of upstream `volcengine/verl`; it is intentionally small
and surgical, so contributions that follow the same discipline are
easier to merge.

## Ground rules

1. **Paper-faithfulness over local cleverness.**  Every algorithm in
   `recipe/teleboost/algorithms/` is pinned to a specific paper
   equation by a unit test.  If you add a variant that diverges from
   the paper, document it in the module docstring as an *in-house
   extension* and mark it clearly in the yaml comment.
2. **No silent walkarounds for upstream bugs.**  If `verl`,
   `transformers`, `vllm`, `hpsv2`, etc. are buggy, fix at the install
   layer (Dockerfile, pin in `requirements*.txt`, copy the missing
   data file) and add a row to `INSTALL.md` "Known gotchas".  Do not
   sprinkle defensive try / except in algorithm code.
3. **Tests pin paper equations, not implementation details.**  When
   you add a formula, add a test that pins the *paper's* numerical
   form (with explicit constants and references), not the call shape.

## What to test

The full local check (no GPU needed):

```bash
pytest tests/ -v
```

Key suites:

* `tests/test_sigma_schedule.py` — σ_t SDE-step formulas (DanceGRPO /
  Flow-GRPO) + σ=1 edge case
* `tests/test_dancegrpo_algorithms.py` — BGPO Eq. 4 (CRT) + Eq. 2 (RAS)
  + joint reward task weights
* `tests/test_grpo_advantage.py` — per-prompt z-score advantage
* `tests/test_grpo_guard.py` — RatioNorm (Eq. 8) + grad-reweight δ (Eq. 12)
* `tests/test_uid_broadcast.py` — UUID alignment with `repeat_interleave`

GPU smokes (8×H800 verified):

```bash
TRAIN_FILE=... TEST_FILE=... WAN_MODEL_PATH=... REWARD_MODEL_PATH=... \
  bash recipe/teleboost/run_teleboost_smoke.sh
```

`TELEBOOST_METHOD=baseline|bgpo|vipo|bgpo_vipo|joint` for the matrix.
Set `actor_rollout_ref.actor.sigma_form=flow_grpo` etc. via `"$@"`
(the smoke script forwards trailing args to Hydra).

## Pull request etiquette

* Run `pytest tests/` before pushing.  All pin tests must stay green.
* Commit messages: conventional-ish (`feat(scope):`, `fix(scope):`,
  `refactor(scope):`, `chore(scope):`, `docs(scope):`).
* If your PR touches an algorithm formula, include the paper +
  equation reference in the commit body.
* Run at least one smoke against the change if it touches the rollout
  / actor / loss path.  Mention the smoke result in the PR body.
* Don't push commits that contain absolute paths, internal cluster
  hostnames, or credentials — `INSTALL.md` and the smoke env are the
  only place such things should live (and even there as placeholders).

## Reporting issues

If you hit something that's *not* covered by `INSTALL.md` "Known
gotchas", open an issue with:

* Hardware (GPU model + count)
* `pip freeze | grep -E "torch|verl|transformers|vllm|flash-attn|hpsv2"`
* Full Hydra command (smoke shell or your own)
* The first traceback line (not the Ray-wrapped one) and the
  worker stdout near the error
