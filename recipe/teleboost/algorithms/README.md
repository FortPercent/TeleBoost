# Supported algorithms

| name        | file                                  | enable flag                                       | role                                                                |
|-------------|---------------------------------------|---------------------------------------------------|---------------------------------------------------------------------|
| baseline    | (no module — vanilla GRPO loop)       | `algorithm.adv_estimator=grpo`                    | reference policy-gradient baseline                                  |
| BGPO        | [`bgpo.py`](bgpo.py)                  | `algorithm.bgpo.enable=true`                      | Bayesian-prior group optimization (CRT reward rerange + RAS adv scale) — paper arxiv 2511.18919 |
| VIPO        | [`vipo.py`](vipo.py)                  | `actor_rollout_ref.pixel_weight.enable=true`      | DINOv2 per-pixel advantage broadcast — paper arxiv 2511.18719       |
| joint reward| [`joint.py`](joint.py)                | `reward_model.type=joint`                         | multi-head joint reward orchestration                               |
| multi-reward agg | [`multi_reward_aggregation.py`](multi_reward_aggregation.py) | (always on under `joint`)        | in-house convex zero-bracket weights (NOT from BGPO paper)         |
| GRPO-Guard  | [`grpo_guard.py`](grpo_guard.py)      | `actor_rollout_ref.actor.grpo_guard.enable=true` (+ `ratio_norm` / `grad_reweight` separable per paper §4.3) | flow-matching ratio-norm (Eq. 8) + grad-reweight δ (Eq. 12, ``flow_grpo`` / ``dancegrpo`` forms) — paper arxiv 2510.22319 |
| σ_t schedule| [`sigma_schedule.py`](sigma_schedule.py) | `actor_rollout_ref.actor.sigma_form=dancegrpo\|flow_grpo` | SDE-step σ_t formula registry: DanceGRPO constant-η (arxiv 2505.07818) vs Flow-GRPO η·√(t/(1−t)) (arxiv 2505.05470). Pairs with ``grpo_guard.grad_reweight_form`` (same key set). |
| flow-grpo   | (in `dp_actor.py` / `teleboost/workers/rollout/diffusion_rollout.py`) | `actor_rollout_ref.flow_grpo.enable=true`         | SDE-window subsampling for fast flow-grpo — paper arxiv 2505.05470  |

## Layout convention

Each algorithm module exposes:

1. **Pure-function compute / helpers at module level.** These are unit-testable
   without spinning up a trainer (see `compute_joint_task_weights`,
   `rerange_group_rewards`, `compute_batch_pixel_weight_maps`).
2. **A `*Mixin` class** that `RayDanceGRPOTrainer` inherits from. Mixins read
   `self.config`, `self.global_steps`, etc. from the trainer.
3. **A no-op fallback** when the enable flag is False — every algorithm must
   degrade cleanly to baseline GRPO so smokes are comparable.

## Adding a new algorithm

1. Create `algorithms/<name>.py` with a docstring at the top describing the
   paper / motivation / enable flag.
2. Add a `<Name>Mixin` class with `_is_<name>_enabled()` and the trainer
   hooks. Use the same structure as `bgpo.py` / `vipo.py`.
3. Re-export both helpers and the mixin from `algorithms/__init__.py`.
4. Add a row to the table above.
5. Wire the trainer to inherit the mixin (in `teleboost_ray_trainer.py`).
6. Add an entry in `run_teleboost_smoke.sh` so the smoke launcher exposes
   `TELEBOOST_METHOD=<name>`.
