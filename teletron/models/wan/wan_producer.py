import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import collections
from teletron.utils import get_args
from teletron.core.parallel_state import get_world_group
from teletron.models.wan.wan_encoder_utils import get_encoder_features
from vast.models.dit.wan_dit import ModelManager
from vast.models.dit.wan_dit import WanModel
from vast.models.dit.wan_dit import WanTextEncoder
from vast.models.dit.wan_dit import WanVideoVAE
from vast.models.dit.wan_dit import WanImageEncoder
from vast.models.dit.wan_dit import WanPrompter
from vast.models.dit.wan_dit import ModelManager
from vast.pipelines.wan.wan_video import WanVideoPipeline

# --- 配置参数 ---
# 定义每个消费者将处理的Tensor数量
NUM_ITEMS_PER_CONSUMER = 100000


# 生产者为每个消费者维护的GPU端队列的最大容量
MAX_QUEUE_PER_CONSUMER_ON_PRODUCER = 2 # 用户要求的队列大小
# 生产者为每个消费者同时进行的isend操作的最大数量 (提供更精细的发送流控)
MAX_OUTSTANDING_SENDS_PER_CONSUMER = 1 # 允许最多isend在途

def cleanup_dist():
    # pass
    if dist.is_initialized():
        rank = dist.get_rank()
        dist.destroy_process_group()
        print(f"Rank {rank}: 进程组已销毁。")
    else:
        print("进程组未初始化或已被销毁。")

def pack_tensors(tensors_to_flatten):
    context_tensor = torch.flatten(tensors_to_flatten[0])
    clip_tensor = torch.flatten(tensors_to_flatten[1])
    img_tensor = torch.flatten(tensors_to_flatten[2])
    latents_tensor = torch.flatten(tensors_to_flatten[3])
    result = torch.cat((context_tensor, clip_tensor, img_tensor, latents_tensor), dim=0)
    return result

def get_tensors_size(tensor_list:list, device):
    size_info = ()
    for item in tensor_list:
        size_info += item.size()
    return torch.tensor(size_info, dtype=torch.int32, device=device)

def producer_process(
        rank, 
        world_size,
        build_train_valid_test_data_iterators, 
        # train_valid_test_dataset_provider, 
        train_ds=None, 
    ):
    
    model_manager = ModelManager(torch_dtype=torch.float32, device="cpu")
    model_path = [
        '/workspace/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth', 
        '/workspace/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth', 
        '/workspace/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
    ]

    model_manager.load_models(model_path)
    pipe = WanVideoPipeline.from_model_manager(model_manager)
    text_encoder=pipe.text_encoder.to(device=torch.cuda.current_device())
    image_encoder=pipe.image_encoder.to(device=torch.cuda.current_device())
    vae = pipe.vae.to(device=torch.cuda.current_device(),dtype=torch.bfloat16)
    del pipe

    prompter = WanPrompter()
    prompter.fetch_models(text_encoder)
    prompter.fetch_tokenizer("/workspace/Wan2___1-I2V-14B-480P/google/umt5-xxl")
    
    tiler_kwargs = {
        "tiled": True, 
        "tile_size":  (34, 34), 
        "tile_stride": (18, 16)
    }

    args = get_args()
    dit_world_size = args.dit_world_size
    producer_size = world_size - dit_world_size
    torch.manual_seed(1234)

    from teletron.core.parallel_state import get_comm_pair
    comm_pairs = get_comm_pair()
    consumers_data = torch.zeros(
        (len(comm_pairs), 1), dtype=int, device=torch.cuda.current_device()
    )
    consumers_queue = []
    idx=0
    for ccs in comm_pairs:
        req = dist.irecv(tensor=consumers_data[idx], src=ccs.consumer, tag=0)
        consumers_queue.append(req)
        idx+=1

    while len(consumers_queue) > 0:
        for ccs_req in  consumers_queue:
            if ccs_req.is_completed():
                consumers_queue.remove(ccs_req)
        time.sleep(0.1)

    args.iteration = consumers_data[0][0]

    train_data_iterators = {comm_pair.consumer: None for comm_pair in comm_pairs}
    valid_data_iterators = {comm_pair.consumer: None for comm_pair in comm_pairs}
    test_data_iterators = {comm_pair.consumer: None for comm_pair in comm_pairs}

    for comm_pair in comm_pairs:
        train_data_iterators[comm_pair.consumer], valid_data_iterators[comm_pair.consumer], test_data_iterators[comm_pair.consumer] \
                = build_train_valid_test_data_iterators(
                    # train_valid_test_dataset_provider, 
                    is_tp_first = True, 
                    dp_rank = comm_pair.dp_rank, 
                    dp_size = comm_pair.dp_size, 
                    train_ds_prev =train_ds )
        
    consumers_data_queues = {
        crank.consumer: collections.deque() for crank in comm_pairs
    }
    consumers_size_queues = {
        crank.consumer: collections.deque() for crank in comm_pairs
    }

    items_initiated_send_for_consumer = {crank.consumer: 0 for crank in comm_pairs}

    send_size_in_flight = []
    send_features_in_flight = []

    try:
        while any(items_initiated_send_for_consumer[crank.consumer] < NUM_ITEMS_PER_CONSUMER for crank in comm_pairs):
            new_send_size_in_flight = []
            for req, cr, item_idx_req, size2send in send_size_in_flight:
                if req.is_completed() is True:
                    del size2send
                    tensor2send = consumers_data_queues[cr].popleft()
                    req_data = dist.isend(tensor=tensor2send, dst=cr, tag=1)
                    send_features_in_flight.append((req_data, cr, item_idx_req, tensor2send))
                else:
                    new_send_size_in_flight.append((req, cr, item_idx_req, size2send))
            send_size_in_flight = new_send_size_in_flight

            for current_comm_pair in comm_pairs:
     
                items_in_queue = len(consumers_data_queues[current_comm_pair.consumer])
                
                total_managed_for_consumer = items_initiated_send_for_consumer[current_comm_pair.consumer] + items_in_queue

                if items_in_queue < MAX_QUEUE_PER_CONSUMER_ON_PRODUCER:
                    
                    data = next(train_data_iterators[current_comm_pair.consumer]) 
                    batch = dict(data)
                    prompt_emb, image_emb, latents = get_encoder_features(batch, prompter, vae, tiler_kwargs, image_encoder)
                    context = prompt_emb['context']
                    img_clip_feature = image_emb["clip_feature"]
                    img_emb_y = image_emb["y"]
                    consumer_size_info = get_tensors_size((context, img_clip_feature, img_emb_y, latents), device=torch.cuda.current_device())
                    tensor_to_send = pack_tensors([context, img_clip_feature, img_emb_y, latents])
                    consumers_size_queues[current_comm_pair.consumer].append(consumer_size_info)
                    consumers_data_queues[current_comm_pair.consumer].append(tensor_to_send)
 
            for current_comm_pair in comm_pairs:
                outstanding_sends_for_this_consumer = sum(
                    1 for _, cr, _, _ in send_size_in_flight if cr == current_comm_pair.consumer
                )

                if consumers_size_queues[current_comm_pair.consumer] and \
                    outstanding_sends_for_this_consumer < MAX_OUTSTANDING_SENDS_PER_CONSUMER:
                    
                    size_to_send = consumers_size_queues[current_comm_pair.consumer].popleft()
                    current_item_idx_to_send = items_initiated_send_for_consumer[current_comm_pair.consumer]
                    #print(f"Rank{current_comm_pair.producer} send data to Rank{current_comm_pair.consumer}, tensor shape: {size_to_send}")
                    req = dist.isend(tensor=size_to_send, dst=current_comm_pair.consumer, tag=1)

                    send_size_in_flight.append((req, current_comm_pair.consumer, current_item_idx_to_send, size_to_send))
                    items_initiated_send_for_consumer[current_comm_pair.consumer] += 1

            new_send_features_in_flight = []
            for req, cr, item_idx_req, tensor2send in send_features_in_flight:
                if req.is_completed() is True:
                    del tensor2send
                else:
                    new_send_features_in_flight.append((req, cr, item_idx_req, tensor2send))
            send_features_in_flight = new_send_features_in_flight

            time.sleep(0.05)


        for req_obj, cr, item_idx_req,_ in send_size_in_flight:
            req_obj.wait()

        dist.barrier(group=get_world_group())

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        cleanup_dist()


