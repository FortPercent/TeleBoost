import os

from vast.datasets.config.t2v_200w import get_data_list

# 训练配置
dst_size = (1280, 720)
dst_num_frames = 45
dst_fps = 15

config = dict(
    ## log&ckpts路径
    runners=["projects.hunyuanvideo.adaptors.HunYuanVideoTrainer"],
    ## 分布式配置for luancher
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type="DEEPSPEED",
        deepspeed_config=dict(
            deepspeed_config_file=os.path.join(
                os.getcwd(), "configs/accelerate_configs/zero2.json"
            ),
        ),
        num_machines=os.environ.get("WORLD_SIZE", 1),
        until_completion=True,
    ),
    ## 训练配置for runner
    ### dataloader配置
    dataloaders=dict(
        #### dataloader train配置
        train=dict(
            dataset=dict(
                type="ClipDataset",
                data_path_list=[
                    "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_1.json",
                    "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_2.json",
                    "/nvfile-heatstorage/Text2Video/annotations/200w/pack_zwzx_3.json",
                    "/nvfile-heatstorage/Text2Video/annotations/200w_nobody/pack_zwzx_1_slice_new_0.json",
                    "/nvfile-heatstorage/Text2Video/annotations/150w/pexels_v0.0.8.json",
                    "/nvfile-heatstorage/Text2Video/annotations/150w/mixkit_v0.0.7.json",
                    "/nvfile-heatstorage/Text2Video/annotations/150w/pixapay_v0.0.7.json",
                                        
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-20.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-21.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-22.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-23.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-24.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-25.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-1.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-2.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-3.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-4.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-26-5.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-1.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-2..json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-3.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-4.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-27-5.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-1.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-2.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-3.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-4.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-5.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-6.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-7.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-28-8.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-1.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-2.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-3.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-4.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-5.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-6.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-7.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-8.json",
                    "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-9.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-10.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-11.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-12.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-29-13.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-4.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-5.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-6.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-7.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-8.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-30-9.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-4.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-5.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-6.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-7.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-1-31-8.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-4.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-5.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-01-6.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-02-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-02-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-02-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-03-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-03-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-04-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-04-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-04-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-4.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-5.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-6.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-05-7.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-06-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-06-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-06-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-4.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-5.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-6.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-7.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-8.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-07-9.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-08-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-08-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-08-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-4.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-09-5.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-10-1.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-10-2.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-10-3.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-10-4.json",
                    # "/nvfile-heatstorage/Text2Video/annotations/koala/koala-2-11-1.json",
                ],
                filter_cfg=dict(
                    dst_size=dst_size,
                    dst_num_frames=dst_num_frames,
                    dst_fps=dst_fps,
                    multiple=16,
                    min_area=dst_size[0] * dst_size[1],
                    optical_flow_th=3,
                    aesthetic_th=4,
                    bucket_size_th=4,
                    motion_th=0,
                    clearity_th=0.9,
                    training_suitability_th=3.5,
                    area_th=720 * 480,
                ),
                transforms=[
                    dict(
                        type="SampleImages",
                        num_frames=dst_num_frames,
                    ),
                    dict(
                        type="PromptGenerator",
                        clean_prompt=True,
                        default_prompt_prob=0.1,
                        field_weights={
                            'dense_caption': 1.0,
                            'subject': 1.0,
                            'background': 1.0,
                            'style': 1.0,
                            'shot_type': 1.0,
                            'lighting': 1.0,
                            'atmosphere': 1.0,
                        },
                    ),
                    dict(
                        type="PromptToTransformerEmbedding",
                        model_name="llama",
                        model_path="/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/text_encoder",
                        max_length=256,
                        with_attention_mask=True,
                        padding="max_length",   # [False/"do_not_pad", "max_length", True/"longest"]
                    ),
                    dict(
                        type="PromptToClipEmbedding",
                        model_path="/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/text_encoder_2",
                    ),
                    dict(
                        type="GenerateFirstRefImage",
                    ),
                    dict(
                        type="PackInputs",
                        image_keys=["images", "first_ref_image"],
                        embedding_keys=["prompt_embeds", "prompt_masks", "clip_text_embed",],
                        dst_size=dst_size,
                    ),
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=1,
            sampler=dict(
                type="DefaultSampler",
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
        #### dataloader eval配置
        eval=None,
    ),
    ### 模型model配置
    models=dict(
        pretrained="/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b",
        # transformer_pretrained="/nvfile-heatstorage/ljq/repos/vast/work_dirs/hunyuanvideo_i2vhy_newdataset_720p_1e5_spring_newdata_0210/models/checkpoint_epoch_1_step_2700/transformer_safetensor",
        token_replace="first",
        transformer=dict(
            in_channels=16,  # with ref images 16->32, with ref and cn_images 16->48
        ),
        loss=dict(),
        vae=dict(
            vae_slicing=True,
            vae_tiling=True,
        ),
        # flow matching schdule
        scheduler=dict(
            flow_resolution_shifting=False,
            flow_base_image_seq_len=256,
            flow_max_image_seq_len=4096,
            flow_base_shift=0.5,
            flow_max_shift=1.15,
            flow_shift=1.0,
            flow_weighting_scheme="none",
            flow_logit_mean=0.0,
            flow_logit_std=1.0,
            flow_mode_scale=1.29,
        ),
    ),
    ### 优化器optimizer配置
    optimizers=dict(
        type="AdamW",
        lr=1e-5,
        weight_decay=1e-2,
    ),
    ### 学习率scheduler配置
    schedulers=dict(
        type="ConstantScheduler",
    ),
    ### 训练过程train配置
    train=dict(
        resume=True,
        checkpoint_save_optimizer=True,
        max_epochs=10,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",  # fp16, bf16
        checkpoint_interval=100,
        checkpoint_total_limit=-1,
        log_with="tensorboard",
        log_interval=1,
        with_ema=False,
        activation_checkpointing=True,
        activation_class_names=[
            "HunyuanVideoTransformerBlock",
            "HunyuanVideoSingleTransformerBlock",
        ],
    ),
    ### 测试过程test配置
    test=dict(),
)
