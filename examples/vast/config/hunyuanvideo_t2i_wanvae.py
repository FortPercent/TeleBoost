import os

dst_size = (256, 256)

config = dict(
    ## log&ckpts路径
    runners=["projects.hunyuanvideo.adaptors.hunyuanvideo_t2itrainer_wanvae.HunYuanVideoT2ITrainerWanVAE"],
    ## 分布式配置for luancher
    launch=dict(
        gpu_ids=[0,1,2,3,4,5,6,7],
        distributed_type="DEEPSPEED",
        deepspeed_config=dict(
            deepspeed_config_file=os.path.join(
                os.getcwd(), "configs/accelerate_configs/zero2.json"
            ),
        ),
        num_machines=os.environ.get("WORLD_SIZE", 1),
        until_completion=False,
    ),
    ## 训练配置for runner
    accelerator=dict(
        mixed_precision='bf16',
        log_with="tensorboard",
    ),

    dataset=dict(
        type="VariableImageDataset",
        data_path_list=[
            "/nvfile-heatstorage/Text2Video/annotations/image_data/mfw_1024_v1.json",
            "/nvfile-heatstorage/Text2Video/annotations/image_data/0830_430w_kolors_tars.json",
            "/nvfile-heatstorage/Text2Video/annotations/image_data/mid_186000_new.json",
            "/nvfile-heatstorage/Text2Video/annotations/image_data/untared_md.json"
        ],
        filter_cfg=dict(
            aesthetic_th=4,
            watermark_th=1.0,
            unsafe_ch=1.0,
            area_th=256*256,
        ),
        transforms=[
            dict(
                type="SampleImages",
            ),
            dict(
                type="PromptGenerator",
                clean_prompt=True,
                #short_prompt_prob=0.0, deprecated
                default_prompt_prob=0.1,
            ),
            dict(
                type="PackInputs",
                image_keys=["images",],
                #dst_size=dst_size, deprecated
            ),
        ],
    ),

    # sampler配置
    sampler=dict(
        type="BucketVariableBatchSampler",
        bucket_config={
            "256px": {
                "1": {
                        "bsz": 64,
                        "prob": 1.0
                },
            }
        }
    ),

    ### dataloader配置
    dataloader=dict(
        num_workers=1,
    ),

    ### 模型model配置
    models=dict(
        pretrained="/data01/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/",
        vae_pretrained="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/vae",
        transformer_pretrained="/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_2p6b/transformer",
        transformer=dict(
            in_channels=16,  # with ref images 16->32, with ref and cn_images 16->48
        ),
        loss=dict(),
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
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=1e-2,
    ),

    ### 学习率scheduler配置
    scheduler=dict(
        type="ConstantScheduler",
    ),

    ### 训练过程train配置
    train=dict(
        model_resume=True,
        checkpoint_save_optimizer=True,
        max_epochs=100,
        gradient_accumulation_steps=1,
        max_grad_norm=1,
        grad_norm_type=2,
        checkpoint_interval=1000,
        log_interval=1,
        activation_checkpointing=True,
        activation_class_names=[
            "HunyuanVideoTransformerBlock",
            "HunyuanVideoSingleTransformerBlock",
        ],
    ),
    ### 测试过程test配置
    test=dict(),
)
