dst_size = (720, 480)
dst_fps = 15
dst_num_frames = 81
NUM_WORKERS = 1

config = dict(
    dataloaders=dict(
        train=dict(
            dataset=dict(
                type="TensorDataset",
                pth_paths=[
                    "/nvfile-heatstorage/dj/datasets/istock/part_0_50000",

                ],
                metadata_paths=[
                    "/gemini/platform/public/xiangxunzhi/filter_videos_16fps_csvs",
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

                transforms=[
                    dict(
                        type="SampleImages",
                        num_frames=dst_num_frames,
                    ),
                    dict(
                        type="PromptGenerator",
                        clean_prompt=True,
                        default_prompt_prob=0.1,
                    ),
                    dict(
                        type="PackInputs",
                        image_keys=[
                            "images",
                        ],
                        #dst_size=dst_size,
                    ),
                    dict(
                        type="GenerateRefImagesWithMask",
                        mask_cfg={
                            "t2v": 0.0,
                            "i2v": 0.4,
                            "clear": 0.0,
                            "continuation": 0.2,
                            "random": 0.0,
                            "transition": 0.4
                        },
                        min_clear_ratio=0.0,
                        max_clear_ratio=1.0,
                    ),
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=NUM_WORKERS,
            sampler=dict(
                type="DefaultSampler",
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
    ),
)
