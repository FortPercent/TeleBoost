dst_size = (720, 480)
dst_fps = 15
dst_num_frames = 81
NUM_WORKERS = 1

config = dict(
            dataset=dict(
                type="TensorDataset",
                pth_paths=[
                    "/nvfile-heatstorage/teleai-infra/kaikai/HumanData_subset_500/merged_videos_latents",
                    # "/nvfile-heatstorage/Text2Video/data/huggingface/dataset/OpenVid-1M/OpenVidVideoData",
                ],
                metadata_paths=[
                    "/nvfile-heatstorage/teleai-infra/kaikai/HumanData_subset_500/filtered_500.csv",
                    # "/nvfile-heatstorage/Text2Video/data/huggingface/dataset/OpenVid-1M/data/train/OpenVid-1M.csv",
                ],
                # only using the above 3 items
                filter_cfg=dict(
                    dst_size=dst_size,
                    dst_num_frames=dst_num_frames,
                    dst_fps=dst_fps,
                    multiple=16,
                    min_area=dst_size[0] * dst_size[1],
                    optical_flow_th=4,
                    aesthetic_th=4.5,
                    bucket_size_th=4,
                    motion_th=0,
                    clearity_th=0.9,
                    laplacian_th=200,
                    training_suitability_th=5.0,
                    area_th=1280 * 720,
                ),
            ),
            batch_size_per_gpu=1,
            num_workers=NUM_WORKERS,
            sampler=dict(
                type="DefaultSampler",
            ),
            collator=dict(
                is_equal=True,
            ),
    
)
# import os
# import math

# dst_size = (720, 480)
# dst_fps = 16
# dst_num_frames = 81

# # Temporary code for quick debugging
# debug = False # open
# if debug:
#     GPU_IDS = [0]
#     NUM_WORKERS = 1
#     import logging

#     logging.basicConfig(level=logging.DEBUG)
# else:
#     GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
#     NUM_WORKERS = 2

# TRAIN_ON_LOW_NOISE = None

# if TRAIN_ON_LOW_NOISE is not None:
#     save_name = f"work_dirs/wanvideo_i2v/moe_8b_480p_lownoise={TRAIN_ON_LOW_NOISE}"
# else:
#     save_name = f"work_dirs/wanvideo_i2v/finetune_8b_480p"

# config = dict(
#     runners=["projects.wan.adaptors.WanI2VTrainer"],
#     project_dir = os.path.join(
#         os.getcwd(), save_name
#     ),
#     launch=dict(
#         gpu_ids=GPU_IDS,
#         distributed_type="DEEPSPEED",
#         deepspeed_config=dict(
#             deepspeed_config_file=os.path.join(
#                 os.getcwd(), "configs/accelerate_configs/zero3_offload.json"
#             ),
#         ),
#         num_machines=os.environ.get("WORLD_SIZE", 1),
#         until_completion=True,
#     ),

#     dataset=dict(
#         type="ClipDataset",
#         serialize_data=False,
#         data_path_list=[
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_1.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_2.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_3.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_4.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_5.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_6.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_7.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_8.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_9.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_10.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_11.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_12.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_13.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_14.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_15.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_16.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_17.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_18.json",
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_19.json",             
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_0.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_1.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_2.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_3.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_4.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_5.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_6.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_7.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_8.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/zwzx/zwzx_0.json',
#             '/nvfile-heatstorage/cjf/share/data/0617/zwzx/zwzx_1.json',
#         ] if debug == False else ["/nvfile-heatstorage/cjf/share/data/0617/xhzx/xhzx_10.json"],
#         filter_cfg=dict(
#             dst_size=dst_size,
#             dst_num_frames = dst_num_frames,
#             dst_fps = dst_fps,
#             multiple=16,
#             min_area=dst_size[0] * dst_size[1] * (2 if dst_size[1] == 480 else 1),
#             optical_flow_th=1.5,
#             aesthetic_th=5,
#             bucket_size_th=4,
#             motion_th=0,
#             clearity_th=0.9,
#             laplacian_th=30,
#             training_suitability_th=5.0,
#             area_th=dst_size[0] * dst_size[1] * (2 if dst_size[1] == 480 else 1),
#         ),
#         transforms=[
#             dict(
#                 type="SampleImages",
#                 num_frames=dst_num_frames,
#             ),
#             dict(
#                 type="PromptGenerator",
#                 clean_prompt=True,
#                 default_prompt_prob=0.1,
#             ),
#             dict(
#                 type="GenerateRawFirstRefImage",
#             ),
#             dict(
#                 type="PackInputs",
#                 deterministic=True,
#                 image_keys=[
#                     "images",
#                 ],
#                 embedding_keys=[
#                     "raw_first_image", 
#                 ],  
#             ),
#         ],
#     ),
#     eval=dict(
#         data_path_list=[
#             "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
#         ],
#     ),
#     models=dict(
#         text_encoder_path="/workspace/dense_models/models_t5_umt5-xxl-enc-bf16.pth",
#         vae_path="/workspace/dense_models/Wan2.1_VAE.pth",
#         image_encoder_path="/workspace/dense_models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
#         dit_path="/workspace/dense_models/wan8b_480p/raw.pth", 
#         tiled=True, 
#         tile_size=(34, 34),
#         tile_stride=(18, 16), 
#         train_on_low_noise=TRAIN_ON_LOW_NOISE,
#     ),
#     ### 优化器optimizer配置
#     optimizer=dict(
#         type="AdamW",
#         lr=2e-5,
#         weight_decay=1e-3,
#     ),
#     accelerator=dict(
#         mixed_precision='bf16',
#         log_with="tensorboard",
#     ),
#     ### 学习率scheduler配置
#     scheduler=dict(
#         type="CosineScheduler",
#     ),
#     sampler=dict(
#         type="BucketVariableBatchSampler",
#         shuffle=True,
#         bucket_config={
#             f"832x480":{
#                 "81": {
#                     "bsz": 3,
#                     "prob": 1.0
#                 }
#             }
#         }
#     ),
#     dataloaders=dict(
#         num_workers=NUM_WORKERS,
#     ),
#     ### 训练过程train配置
#     train=dict(
#         max_grad_norm=1,
#         grad_norm_type=2,
#         model_resume=True,
#         checkpoint_save_optimizer=True,
#         max_epochs=10,
#         gradient_accumulation_steps=1,
#         mixed_precision="bf16",  # fp16, bf16
#         checkpoint_interval=200,
#         log_with="tensorboard",
#         log_interval=1,
#         with_ema=False,
#         activation_checkpointing=True,
#         activation_class_names=[
#             "DiTBlock",
#         ],
#     ),
#     test=dict(),
# )
