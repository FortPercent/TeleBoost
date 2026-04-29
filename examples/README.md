# Examples

## Layout

```
examples/
├── teleai/                  # TeleAI's branded model training & inference
│   ├── train_dpo.sh         # canonical DPO launcher (env-driven)
│   ├── pretrain_dpo_i2v.py  # DPO entry point
│   ├── pretrain_*.py        # other entry points (i2v / multimask / sr / t2v)
│   ├── config/              # per-experiment config dicts
│   ├── infer/               # inference pipelines & demo prompts
│   └── *.py                 # data analysis / conversion utilities
└── wan/                     # Wan-Video reference model integration
    ├── pretrain_*.py
    └── config/
```

## Running DPO training

`examples/teleai/train_dpo.sh` is the canonical launcher. Override anything via
env vars:

```bash
export MEGATRON_LM_DIR=/path/to/Megatron-LM       # required: custom Megatron fork
export TELEAI_DATA_TOOL_DIR=/path/to/teleai_data_tool  # required: data utilities
export EXPR_NAME=my_dpo_run
export CHECKPOINT_PATH_SAVE=/path/to/ckpt          # default: ./checkpoints/${EXPR_NAME}
export CP=8 TP=1 N_VAE=2 N_MOE=1                   # parallelism (defaults shown)

bash examples/teleai/train_dpo.sh
```

For multi-node, additionally set `NNODES`, `NODE_RANK`, `MASTER_ADDR`.

The default config path is `config.wan_dpo.config` which resolves to
`examples/teleai/config/wan_dpo.py:config`. Override as the second positional
argument:

```bash
bash examples/teleai/train_dpo.sh examples/teleai/pretrain_dpo_i2v.py config.wan_dpo.config
```

## Editing configs

Files under `examples/teleai/config/` and `examples/wan/config/` contain
**dataset / model-checkpoint paths from the original development environment**.
Replace these with your own paths before running.
