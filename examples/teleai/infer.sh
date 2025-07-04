# python infer.py \
# --models /nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/vast_4_moe/node_0/mp_rank_00/model_optim_rng.pt \
#  /nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/vast_4_moe/node_1/mp_rank_00/model_optim_rng.pt \
#  /nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/vast_4_moe/node_2/mp_rank_00/model_optim_rng.pt \
#  /nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/vast_4_moe/node_3/mp_rank_00/model_optim_rng.pt \
# --timesteps 1000 750 500 250 0

# python infer.py \
# --models /nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/vast_4_moe/node_0/mp_rank_00/model_optim_rng.pt \
#  /nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/vast_4_moe/node_1/mp_rank_00/model_optim_rng.pt \
# --timesteps 1000 500 0

python examples/teleai/infer.py \
 --models \
  /nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/expr1/node_0/iter_0001000/mp_rank_00/model_optim_rng.pt \
  /nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/expr1/node_1/iter_0001000/mp_rank_00/model_optim_rng.pt \
  --timestep 1000 500 0
