### 环境
1. 复制权重到本地
'''
cp -r /nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/ /workspace/
'''
安装依赖
'''
pip install -e .
'''

### 训练
'''
bash examples/teleai/run_i2v.sh
'''
1. 参数
根据机器数量配置 CP / N_NOE / N_GPU_FOR_TRAIN / N_GPU_FOR_DATA / N_LAYERS=25 参数
CP：序列并行的长度
N_NOE： MOE 模型个数，目前仅支持自动配置1/2/4时的moe-step-factor-list， 为1时为普通非moe模型
N_GPU_FOR_TRAIN：训练使用的gpu数量
N_GPU_FOR_DATA：数据服务使用的gpu数量
N_LAYERS：模型层数，可设置为1用于DEBUG，但是层数要和加载权重一致


必须配置：
N_GPU_FOR_TRAIN = N_MOE * CP * N
推荐配置：
N_GPU_FOR_TRAIN / N_MOE / CP < N_GPU_FOR_DATA 

2. 权重
--save / --load 参数配置权重读取和存储路径

权重转换
CHECKPOINT_PATH: 权重路径
TARGET_CKPT_PATH: 转换后的权重路径
folder-name: node_0 / node_1 / node_2 / node_3 ...
需要注意权重读取路径需要有一个latest_checkpointed_iteration.txt文件，标注读取文件夹名字

'''
examples/teleai/convert_ckpt_temp.sh
'''

### 推理
'''
bash examples/teleai/infer.sh
'''