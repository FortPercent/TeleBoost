# Teletron 使用文档

Teletron 是一个专为训练长上下文多模态Transformer模型而设计的分布式训练框架，支持多种视频生成模型的高效训练。

## QuickStart

### 环境设置

一般可以直接使用basemodel的最新镜像。主要依赖包括torch、flash_attn、teleai_data_tool、yunchang、
deepspeed等，见requirements.txt

### shell脚本和配置文件py设置方法

#### 1. Shell脚本配置

在examples文件夹中，Teletron 为每个模型提供了shell脚本来启动训练流程，如examples/teleai/run.sh. 
以下是主要的配置参数：

请根据实际硬件资源（如GPU数量、节点数等）配置以下参数：

| 参数名               | 说明                                   |
|-------------------|--------------------------------------|
| `CP`              | 序列并行组大小，CP组内的GPU看到的是同一份数据的不同分片       |
| `TP`              | 张量并行长度，TP组内的GPU对模型线性层权重做分片           |
| `N_GPU_FOR_TRAIN` | 用于模型训练的 GPU 总数                       |
| `N_GPU_FOR_DATA`  | 用于数据服务的 GPU 数量                       |
| `N_LAYERS`        | 模型层数，默认为 25；调试时可设为 1（需与加载权重匹配）       |
| `N_MOE`           | MoE 模块数量，目前支持 1/2/4；为 1 时使用普通非 MoE 模型 |

注意，在P2P分布式编码器实现下，要求 N_GPU_FOR_TRAIN/CP/TP % N_GPU_FOR_DATA = 0。
在数据服务的分布式编码器实现下不强制要求整除，但我们依然有推荐的[最优配置比例](#训练效率和最优配置一览)。

配置示例：
```commandline
# Parallel config 
CP=2
TP=1

# Multi-node config 
N_MOE=1
N_GPU_FOR_TRAIN=16
N_GPU_FOR_DATA=8

# Single-node config 
N_MOE=1
N_GPU_FOR_TRAIN=1
N_GPU_FOR_DATA=1
```

**启动训练：**

推荐先在开发环境上使用单节点配置（N_GPU_FOR_TRAIN=4，N_GPU_FOR_DATA=1）启动单节点训练
```bash
# Teleai模型训练
bash examples/teleai/run.sh

# 根据任务自定义使用训练脚本和配置文件，如i2v任务则使用pretrain_i2v.py训练脚本和teleai_i2v配置文件
bash examples/teleai/run.sh examples/teleai/pretrain_i2v.py config.teleai_i2v.config

# Wan2.1模型训练
bash examples/wan/run_wan.sh

# 自回归Wan2.1模型训练
bash examples/wan/run_causal.sh
```

**训练配置参数：**
```bash
EXPR_NAME=f1fn2v_1.3B           # 实验名称，会从这个路径加载和保存权重
TRAIN_SCRIPT=${1:-"examples/teleai/pretrain_i2v.py"}  # 训练脚本，默认为examples/teleai/pretrain_i2v.py，可以从命令行传入
CONFIG_PATH=${2:-"config.teleai_i2v.config"}  # 配置文件路径，默认为config.teleai_i2v.config，可以从命令行传入
```
配置文件在examples各个模型文件夹下的config文件夹，如examples/teleai/config/teleai_i2v.py，
注意传入时要用config.{文件名}.config这样的格式。


#### 2. Python配置文件设置

配置文件采用Python字典格式，主要包含数据集配置、DiT模型配置和encoder模型配置，其整体结构如下，
配置文件的详细说明见[config_guide.md](config_guide.md)：

```python
# 基础参数
dst_size = (832, 480)    # 目标分辨率
dst_fps = 16             # 目标帧率
dst_num_frames = 81      # 目标帧数

config = dict(
    # 数据集配置
    dataset=dict(
        type="ClipDataset",
        data_path_list=["/path/to/dataset.json"],
        filter_cfg=dict(
            dst_size=dst_size,
            dst_num_frames=dst_num_frames,
            dst_fps=dst_fps,
            # 更多过滤参数...
        ),
        transforms=[...]
    ),
    
    # 模型配置
    model_config=dict(
        dit=dict(
            type="ParallelTeleaiModel",
            config=dict(
                has_image_input=False,
                patch_size=[1, 2, 2],
                dim=3072,
                num_heads=24,
                num_layers=30,
                # 其他模型参数...
            )
        ),
        encoder=dict(
            type="teleai_encoder",
            # 编码器配置...
        )
    ),
    sampler=dict(
        type="DefaultSampler",    
        # dataloader sampler 配置...
    )
)
```

### 模型支持列表

| 模型名称          | 类型      | 描述                | 支持任务                    |
|---------------|---------|-------------------|-------------------------|
| **Teleai**    | 视频生成    | TeleAI系列视频生成模型    | T2V, I2V, SR, multimask |
| **Wan2.1**    | 视频生成    | Wan2.1系列大规模视频生成模型 | I2V                     |
| **CausalWan** | 自回归视频生成 | 支持自回归Wan变体        | T2V                     |

#### 模型规格对比

| 模型          | 参数量 | 输入维度 | 隐藏维度 | 注意力头数 | 层数 |
|-------------|--------|----------|----------|------------|----|
| Teleai-1.3B | 1.3B | 36/16 | 1536 | 12 | 30 |
| Teleai-5B   | 5B | 48 | 3072 | 24 | 30 |
| Teleai-10B  | 10B | - | 5120 | 40 | 30 |
| Teleai-14B  | 14B | - | 5120 | 40 | 40 |
| Wan2.1-14B  | 14B | - | 5120 | 40 | 40 |

### 训练效率和最优配置一览

<!-- 待补充 -->

## 常用特性

### 分布式多模态编码器

分布式多模态编码器是Teletron的核心组件，支持将视频、图像和文本编码任务在独立的GPU上与DiT并行执行。

#### 使用方法

在shell脚本中启用分布式编码器：

```bash
MODEL_PARALLEL_ARGS=(
    --distributed-vae
    --distributed-vae-world-size $N_GPU_FOR_DATA
)
```

编码器配置示例：
```python
encoder=dict(
    type="teleai_encoder",  # 或 "wan_encoder"
    encoder_schema=['context', 'latents'],
    vae=dict(
        type="TeleaiVideoVAE_2_1",
        path="/path/to/vae.pth",
        tiler_kwargs=dict(
            tiled=False, # 2K以下分辨率推荐tiled=False
            tile_size=(34, 34),
            tile_stride=(18, 16),
        ),
    ),
    text_encoder=dict(
        path="/path/to/text_encoder.pth",
        tokenizer_path="/path/to/tokenizer",
    )
)
```

#### 适配方法


### ContextParallel（上下文并行）

ContextParallel 是专门为长序列训练设计的并行策略，将长序列分割到不同GPU上并行处理。

#### 使用方法

启用上下文并行，注意CP-size要能被模型的attention num head整除。
```bash
--context-parallel-size 2  # 设置CP大小
```

```commandline
Note：现在CausalWan还没有实现CP
```
#### 适配方法

1. 改造DiT前向以切分和聚合序列。
```python
class ParallelTeleaiModel(ContextParallelMixin, TensorParallelMixin, TransformerGeneralMixin, TeleaiModel):
    def forward(...):
        ...
        x = self.split_input(x, dim=1)
        freqs = self.split_input(freqs, dim=0)
        x = self.blocks(x, context_emb, t_mod, freqs)
        x = self.gather_output(x, dim=1)
        ...
        return x 
```
2. 改造DiTBlock Attention层
调用ContextParallelMixin的enable_context_parallel方法使得self attention并行计算。
（一般cross attention不需要额外操作，因为kv没有切分，kv分别和切分的q计算attention即可）
```python
class ParallelTeleaiDitBlock(TensorParallelMixin, ContextParallelMixin, DiTBlock):
    def __init__(self, *args, **kwargs):
        # from ContextParallelMixin
        self.enable_context_parallel(self.self_attn.attn)
```

3. 改造DiT block的modulate和gate层，在ContextParallelMixin中有适配了CP的modulate和gate层。
这两个层计算梯度后需要对shift、scale和gate的梯度额外在cp group做一次reduce sum，因为reduce前他们
只与当前cp rank的部分token做了梯度计算。
```python
class ParallelTeleaiDitBlock(TensorParallelMixin, ContextParallelMixin, DiTBlock):
    def forward(self, x, context, t_mod, freqs):
        ...
        modulated_x1 = self.modulate_with_cp_grad_reduce(normed_x1, shift_msa, scale_msa)
        ...
        gated_x1 = self.gate_with_cp_grad_reduce(x, gate_msa, attn_output)
        ...
        modulated_x2 = self.modulate_with_cp_grad_reduce(normed_x2, shift_mlp, scale_mlp)
        ...
        x = self.gate_with_cp_grad_reduce(x, gate_mlp, ffn_output)

        return x
```

4. 在DiT上应用反向hook。由于input sequence做了切分，大部分层是与切分的sequence做计算，因此他们的权重梯度也是部分结果，
需要在cp group做reduce sum。而部分层（如patch_emb、head）是对完全序列做的计算，不需要cp grad reduce，所以必须要在这里做特殊处理（不能合并到DP reduce）。

另外，modulation和time 相关的权重梯度，已经在modulate和gate中做了处理，所以这里也不需要额外reduce。适配新模型时需要注意。
（TODO：补充使用tensorwatch工具观测梯度以指导实现grad reduce的方法和案例，联系李天催更）

```python
    def register_cp_grad_reduce_hook(self):

        # layers with parallel input sequence need to reduce its param gradient.
        # list the parameters that needs grad reduce and register tensor grad hook

        for name, param in self.named_parameters():
            if name.startswith("patch_emb") or \
                name.startswith("time") or \
                    name.startswith("head") or \
                    "modulation" in name:
                continue

            param.register_hook(self.cp_grad_reduce)

```


### zero2分布式优化器

基于ZeRO-2的分布式优化器，将优化器状态分片以节省内存。

#### 使用方法

启用分布式优化器：
```bash
--use-zero2
```

### EMA

EMA用于模型权重的平滑更新，提高最终模型质量。在每次保存模型权重时额外保存一份ema权重，断点续训时也会加载。
基本不影响训练速度，但是会额外占用一点显存。目前的实现是把ema权重在所有训练的GPU上切分，所以训练GPU越多显存影响越小。

使用方法：
```bash
--with-ema
--ema-decay 0.999 # 一般设置0.999到0.9999
```

### 断点续训

支持训练中断后的恢复。

使用方法：
```bash
--save-interval 500          # 每500步保存一次
--save $CHECKPOINT_PATH_SAVE # 保存路径
--load $CHECKPOINT_PATH_LOAD # 加载路径
```

支持使用--override-opt_param-scheduler来用当前指定的超参（如lr、wd）覆盖上一次训练的优化器超参，
也可以使用--no-load-optim和--no-load-rng来跳过加载优化器状态或者rng state。

推荐使用--data-parallel-random-init训练，因为这样可以让不同rank随机采样的timestep不同，有利于稳定训练。
（开启后模型checkpoint中会额外存每份dp的rng state）


### TensorParallel（张量并行）

张量并行将模型参数分布到多个GPU上，实现模型级别的并行训练。

#### 使用方法

启用张量并行，要求TP-size*CP-size要能被模型的attention num head整除。
```bash
--tensor-model-parallel-size 2  # 设置TP大小
```
```commandline
Note：因为CP用的是ulysses实现，是在head维度切分，而TP切分hidden-size维度，
体现到attention上也会导致head维度被切分，因此要求TPxCP能被head数整除。
Note：现在CausalWan和Hunyuan还没有实现TP
```


#### 适配方法

1. **模型张量并行改造：**
使用TensorParallelMixin中的enable_tensor_parallel系列方法来将模型中的线性层改为列并行线性层或行并行线性层。
关于这些接口的使用方式详见(TensorParallelMixin)[TODO:TensorParallelMixin接口文档]

```python
class ParallelTeleaiDitBlock(TensorParallelMixin, ContextParallelMixin, DiTBlock):
    def __init__(...):
        DiTBlock.__init__(...)
        ...
        # from TensorParallelMixin
        self.enable_ffn_tensor_parallel(self.ffn, config)
        self.enable_self_attn_tensor_parallel(self.self_attn, config)
        self.enable_cross_attn_tensor_parallel(self.cross_attn, config)
```

2. 检查其他层是否受TP影响
我们给Wan适配TP时发现Wan的qk norm是在整个hidden dim上取平均（而不是在head dim上取平均），
这意味着TP情况下，qknorm层收到的hidden dim是切分后的，必须要做一次reduce同步TP group内的rms。 
qk norm的反向也要做处理且更复杂，详见(TensorParallelMixin.TeleParallelRMSNorm)[TODO:TensorParallelMixin接口文档.RMSNorm]。
