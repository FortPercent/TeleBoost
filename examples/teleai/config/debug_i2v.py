import os
import math

dst_size = (832, 480)
dst_fps = 16
dst_num_frames = 81

config = dict(
    dataset=dict(
        type="ClipDataset",
        serialize_data=False,
        data_path_list=[
            "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
        ],
        filter_cfg=dict(
            dst_size=dst_size,
            dst_num_frames = dst_num_frames,
            dst_fps = dst_fps,
            multiple=16,
            min_area=dst_size[0] * dst_size[1] * (2 if dst_size[1] == 480 else 1),
            optical_flow_th=1.5,
            aesthetic_th=5,
            bucket_size_th=4,
            motion_th=0,
            clearity_th=0.9,
            laplacian_th=30,
            training_suitability_th=5.0,
            area_th=dst_size[0] * dst_size[1] * (2 if dst_size[1] == 480 else 1),
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
                ],  
            ),
        ],
    ),
    eval=dict(
        data_path_list=[
            "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
        ],
    ),
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
        seed=42,
        drop_last=True,
        infinite=True,
    )
)
