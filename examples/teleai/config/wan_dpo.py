import os
import math
import torch

dst_size = (832, 480)
dst_fps = 16
dst_num_frames = 49

config = dict(
    dataset=dict(
        type="WanDPODataset",

        dataset_base_path="",
        dataset_metadata_path="/path/to/dpo_csv/prompt_video_pairs_enhanced.csv",
        data_path_list=[
            "/path/to/dpo_csv/prompt_video_pairs_enhanced_part0.csv",
            "/path/to/dpo_csv/prompt_video_pairs_enhanced_part1.csv",
            "/path/to/dpo_csv/prompt_video_pairs_enhanced_part2.csv",
            "/path/to/dpo_csv/prompt_video_pairs_enhanced_part3.csv",
            "/path/to/dpo_csv/prompt_video_pairs_enhanced_part4.csv",
            "/path/to/dpo_csv/prompt_video_pairs_enhanced_part5.csv",
            "/path/to/dpo_csv/prompt_video_pairs_enhanced_part6.csv",
            "/path/to/dpo_csv/prompt_video_pairs_enhanced_part7.csv",
        ],
        dataset_repeat=2,

        chosen_video_key="chosen",
        rejected_video_key="rejected",

        height=480,
        width=832,
        num_frames=49,

        time_division_factor=4,
        time_division_remainder=1,
        height_division_factor=16,
        width_division_factor=16,

        max_pixels=1920 * 1080,

        transforms=[
            dict(
                type="InjectRawFirstImageFromVideo",
                video_key="video",
                output_key="raw_first_image",
            ),
            dict(
                type="PreprocessVideoToTensor",
                input_key="video",
                output_key="video",
                torch_dtype="bfloat16",
                pattern="B C T H W",
                min_value=-1,
                max_value=1,
                skip_if_tensor=True,
            ),
            dict(
                type="InjectImagesFromVideoTensor",
                video_key="video",
                output_key="images",
            ),
            dict(
                type="InjectPromptToTopLevel",
                prompt_key="prompt",
            ),
            dict(
                type="PackInputsNoResize",
                normalize=False,
                image_keys=["images"],
                embedding_keys=["raw_first_image", "input_image"],
            ),
        ],
    ),
    eval=dict(
        data_path_list=[
            "/path/to/dpo_data/prompt_video_pairs_matched_image.csv",
        ],
        eval_time_steps=[200, 400, 600, 800, 1000],
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
            type="ParallelTeleaiModel",

            # Architecture sizes — uncomment the row matching your target model.
            #   1.3B: dim=1536, ffn_dim=8960,  num_heads=12, num_layers=30
            #   10B:  dim=5120, ffn_dim=13824, num_heads=40, num_layers=30
            #   14B:  dim=5120, ffn_dim=13824, num_heads=40, num_layers=40
            # in_dim: t2v=16, i2v=36 (Wan2.1) | i2v Wan2.2=36 also
            config=dict(
                has_image_input=False,
                patch_size=[1, 2, 2],
                in_dim=36,
                dim=5120,
                ffn_dim=13824,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=40,
                num_layers=40,
                eps=1e-6,
                has_image_pos_emb=False,
            ),

            train=dict(
                trainable_models="dit",
                use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=True,
                enable_fp8_training=False,
                lora=dict(
                    enable=False,
                    base_model=None,
                    target_modules="q,k,v,o,ffn.0,ffn.2",
                    rank=32,
                    checkpoint=None,
                ),
                dpo=dict(
                    enable=True,
                    beta=0.1,
                ),
                extra_inputs=["input_image"],
            ),
        ),
        encoder=dict(
            type="teleai_encoder",
            encoder_schema=['context', 'img_emb_y', 'latents'],
            vae=dict(
                type="DiffSynthWanVideoVAE",
                path="/path/to/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                tiler_kwargs=dict(
                    tiled=False,
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                ),
                torch_compile=False,
            ),
            text_encoder=dict(
                path="/path/to/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/path/to/Wan2.1-I2V-14B-480P/google/umt5-xxl",
            ),
            image_encoder=dict(
                path="/path/to/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                torch_compile=False,
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
            ),
        ),
    ),
)
