# Supported algorithms

| name        | file                                  | enable flag                                       | role                                                                |
|-------------|---------------------------------------|---------------------------------------------------|---------------------------------------------------------------------|
| baseline    | (no module — vanilla GRPO loop)       | `algorithm.adv_estimator=grpo`                    | reference policy-gradient baseline                                  |
| BGPO        | [`bgpo.py`](bgpo.py)                  | `algorithm.bgpo.enable=true`                      | Bayesian-prior group optimization (CRT reward rerange + RAS adv scale) |
| VIPO        | [`vipo.py`](vipo.py)                  | `actor_rollout_ref.pixel_weight.enable=true`      | DINOv2 per-pixel advantage broadcast                                |
| GRPO-Guard  | (in `dp_actor.py`)                    | `actor_rollout_ref.actor.grpo_guard.enable=true`  | flow-matching ratio-norm + grad-reweight                            |
| flow-grpo   | (in `dp_actor.py` / rollout)          | `actor_rollout_ref.flow_grpo.enable=true`         | SDE-window subsampling for fast flow-grpo                           |
| joint reward| `unified_reward_worker.py` + `reward_models/dynamic_joint.py` | `reward_model.type=joint` | weighted-sum / mean / max / min over multiple reward heads          |

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
5. Wire the trainer to inherit the mixin (in `dancegrpo_ray_trainer.py`).
6. Add an entry in `run_dancegrpo_smoke.sh` so the smoke launcher exposes
   `TELEBOOST_METHOD=<name>`.
