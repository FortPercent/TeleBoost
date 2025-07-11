
---

# Teletron 模型训练与推理简单介绍

本项目提供图像到视频生成模型（I2V）的训练和推理流程说明，适用于多机多卡环境下模型训练和数据处理分离的的大规模分布式训练。

---

## 🧰 环境准备

### 1. 复制模型权重
将预训练模型权重复制到本地工作目录：

```bash
cp -r /nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/ /workspace/
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install /nvfile-heatstorage/yxy/code/teleai_data_tool
pip install /nvfile-heatstorage/yxy/code/Megatron_060
pip install -e .
```

---

## 🚀 训练流程

### 启动训练脚本

```bash
bash examples/teleai/run_i2v.sh
```

### 参数配置说明

请根据实际硬件资源（如GPU数量、节点数等）配置以下参数：

| 参数名                | 说明 |
|----------------------|------|
| `CP`                 | 序列并行长度，N个CP内的GPU看到的是同一份数据 |
| `N_GPU_FOR_TRAIN`    | 用于模型训练的 GPU 总数 |
| `N_GPU_FOR_DATA`     | 用于数据服务的 GPU 数量 |
| `N_LAYERS`           | 模型层数，默认为 25；调试时可设为 1（需与加载权重匹配） |
| `N_MOE`              | MoE 模块数量，目前支持 1/2/4；为 1 时使用普通非 MoE 模型 |

#### 配置建议

- **必须满足：**
  ```bash
  N_GPU_FOR_TRAIN = N_MOE * CP * N
  ```

- **推荐设置：**
  ```bash
  N_GPU_FOR_TRAIN / N_MOE / CP < N_GPU_FOR_DATA
  ```

### 权重路径配置

通过以下参数指定模型权重路径：

- `--save`: 指定保存模型权重的路径
- `--load`: 指定加载模型权重的路径

### 权重格式转换

若需进行模型权重格式转换，请使用如下脚本：

```bash
bash examples/teleai/convert_ckpt_temp.sh
```

> ⚠️ 注意：
>
> - 权重folder-name名字必须为'release'或者'iter-0000100'格式，否则读取会报错
> - 权重目录中需包含一个 `latest_checkpointed_iteration.txt` 文件，用于指示当前读取的权重文件夹。(release or 100)

---

## 🧪 推理流程

执行推理脚本：

```bash
bash examples/teleai/infer.sh
```

---


