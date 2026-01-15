import os
import math

dst_size = (832, 480)
dst_fps = 16
dst_num_frames = 81

config = dict(
    dataset=dict(
        type="WanDPODataset",

        # === 原来 args 里的 ===
        dataset_base_path="",
        # no use
        # dataset_metadata_path="/nvfile-heatstorage/AIGC_H100/liangyzh/dmz_trans/dpo_csv/prompt_video_pairs_matched_image_re.csv",
        dataset_metadata_path="/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/prompt_video_pairs_matched_image.csv",
        data_path_list=[
            # "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/prompt_video_pairs_matched_image.csv"
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part0.csv",
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part1.csv",
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part2.csv",
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part3.csv",
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part4.csv",
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part5.csv",
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part6.csv",
            "/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/data/out_parts/prompt_video_pairs_matched_image.part7.csv"
            
            # "/nvfile-heatstorage/AIGC_H100/liangyzh/dmz_trans/dpo_csv/prompt_video_pairs_matched_image_re_part0.csv",
            # "/nvfile-heatstorage/AIGC_H100/liangyzh/dmz_trans/dpo_csv/prompt_video_pairs_matched_image_re_part1.csv",
            # "/nvfile-heatstorage/AIGC_H100/liangyzh/dmz_trans/dpo_csv/prompt_video_pairs_matched_image_re_part2.csv",
        ],
        dataset_repeat=2,

        # === DPO 语义 ===
        chosen_video_key="chosen",
        # chosen_video_key = "positive_video_path", 
        rejected_video_key="rejected",
        # rejected_video_key = "negative_video_path",
        # === 视频尺寸 & 时序 ===
        # height=720,
        height=480,
        width=832,
        # width=1280,
        num_frames=49,

        # === 这些原脚本是写死的，现在 config 化 ===
        time_division_factor=4,
        time_division_remainder=1,
        height_division_factor=16,
        width_division_factor=16,

        # 可选
        max_pixels=400000,
        

        transforms=[
            dict(
                type="SampleImages",
                num_frames=dst_num_frames,
            ),
            dict(
                type="InjectPromptToTopLevel",
                prompt_key="prompt"
            ),
            dict(
                type="GenerateRawFirstRefImage",
                
            ),
            dict(
                type="PackInputs",
                deterministic=True,
                image_keys=[
                    "images",
                ],
                embedding_keys=[
                    "raw_first_image", 
                    "input_image"  # 保留data_dict --> input_dict新建的这个
                ],  
            ),
            dict(
                type="LoadInputImageAsFirstFrame",
                key="input_image",
                output_key="raw_first_image",
            )
        ],
    ),
    eval=dict(
        data_path_list=[
            "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
        ],
        eval_time_steps=[200,400,600,800,1000]
    ),
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
        seed=42,
        drop_last=True,
        infinite=True,
    ),


    model_config=dict(
        dit=dict(
            type="ParallelTeleaiModel", # ParallelTeleaiModel

            # 这里需要修改 TODO
            config=dict(
                has_image_input=False, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36, # t2v:16 i2v:36
                dim=5120, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=13824, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=40, # 1.3B:12 10B:40 14B:40
                num_layers=40, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=False, 
            ),

            # === DiT 训练 & 行为（从 WanTrainingModule 下沉）===
            train=dict(
                trainable_models="dit",                 # 等价于 trainable_models="dit"
                use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=True,
                enable_fp8_training=False,
                # LoRA
                lora=dict(
                    enable=False,
                    base_model=None,
                    target_modules="q,k,v,o,ffn.0,ffn.2",
                    rank=32,
                    checkpoint=None,
                ),

                # DPO
                dpo=dict(
                    enable=True,
                    beta=0.1,
                ),

                # forward contract
                extra_inputs=["input_image"],
            ),
        ),
        encoder=dict(
            type="teleai_encoder", # teleai_encoder
            encoder_schema=['context', 'img_emb_y', 'latents'],
            vae=dict(
                path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth",
                tiler_kwargs=dict(
                    tiled=False,
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                ),
                torch_compile=False
            ),
            text_encoder=dict(
                path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/google/umt5-xxl",
            ),
            image_encoder=dict(
                path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                torch_compile=True
            ),
            depth_model=dict(
                path="/nvfile-heatstorage/ai_infra/ckpts/lit117/qiuyang/video_depth_anything_vitl.pth",
            ),
        ),


        training=dict(
            diffusion=dict(
                max_timestep_boundary=0.358,
                min_timestep_boundary=0.0,
            ),

            dpo_io=dict(
                chosen_key="chosen",
                rejected_key="rejected",
            ),
             scheduler=dict(
                num_train_timesteps=1000,
            )
        ),
    ),
)
