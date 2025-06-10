# Copyright (c) 2025, TeleAI-infra Team and NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import torch

from megatron.training import (
    get_args
)
from megatron.core import mpu
from typing import get_origin
from einops import rearrange
from torchvision.transforms.functional import to_pil_image
import numpy as np
import torch.distributed as dist
def get_batch_on_this_tp_rank_vast(data_iterator):
    args = get_args()
    data_dict = ['images', 'prompt_embeds', 'prompt_masks', 'clip_text_embed', 'ref_mask', 'ref_images']

    def _broadcast(item):
        if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
           
        batch = {}
        data_dict = [d for d in data.keys()]
        for param in data_dict:
            if isinstance(data[param], list): # prompt is list
                pass
            elif isinstance(data[param], torch.Tensor):
                batch.update({param: data[param].cuda(non_blocking = True)})
            else:
                raise NotImplementedError(f"Unsupported data type: {type(data[param])}")


        # Step 1: 保存每部分的大小信息（只在 Rank 0 执行）
        sizes_info = {key: tensor.size() if tensor is not None else None for key, tensor in batch.items()}
        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        for param in batch.keys():
            _broadcast(batch[param])

    else:
        sizes_info = None 
        sizes_info_list = [sizes_info]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        batch = {}
        data_dict = [d for d in sizes_info_list[0].keys()]
        for param in data_dict:
            if param == "prompt_masks":
                batch. update({param: torch.empty(sizes_info_list[0][param], dtype=torch.int64, device=torch.cuda.current_device())})
            else:
                batch. update({param: torch.empty(sizes_info_list[0][param], dtype=torch.float32, device=torch.cuda.current_device())})
        for param in batch.keys():
            _broadcast(batch[param])

    return batch


# def unpack_tensors(packed_tensor, intervals):
#     token_length = int(packed_tensor[0:1].item())
#     text_feature = packed_tensor[intervals[1]:intervals[2]]
#     rest_features = tuple([packed_tensor[intervals[i-1]:intervals[i]] for i in range(3, len(intervals))])
#     return (token_length, text_feature,) + rest_features

def unpack_tensors(packed_tensor, intervals):
    features = tuple([packed_tensor[intervals[i-1]:intervals[i]] for i in range(1, len(intervals))])
    return features


def get_batch_on_this_tp_cp_rank_vast(data_iterator,max_length):
    def _broadcast(item):
        if item is not None:
            import torch.distributed as dist
            rank = dist.get_rank()
            torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
    
    transformer_dim = 4096
    img_dim = 20
    video_dim = 16
    image_scale = 8
    frame_scale = 4
    clip_dim=768

    if mpu.get_tensor_context_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
        
        sizes_info = {}
        type_info = {}
        batch=dict(data)
        dtype_wan = torch.bfloat16

        from teletron.core.parallel_state import get_comm_pair
        comm_pair = get_comm_pair()

        tensors_info = torch.empty((16), device=torch.cuda.current_device(), dtype=torch.int32)
        req = dist.irecv(tensors_info, comm_pair.producer, tag=0)
        req.wait()
        # print(f"size info: {tensors_info}")

        args = get_args()
        if args.distributed_vae:
            transformer_embedding_size = tensors_info[0] * tensors_info[1] * tensors_info[2]
            clip_embedding_size = tensors_info[3] * tensors_info[4] * tensors_info[5]
            first_img_embedding_size = tensors_info[6] * tensors_info[7] * tensors_info[8] * tensors_info[9] * tensors_info[10]
            video_embedding_size = tensors_info[11] * tensors_info[12] * tensors_info[13] * tensors_info[14] * tensors_info[15]

            recv_tensor = torch.empty(( transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size ), device=torch.cuda.current_device(), dtype=torch.bfloat16)

            intervals = [0, 
                        transformer_embedding_size, 
                        transformer_embedding_size +clip_embedding_size,
                        transformer_embedding_size +clip_embedding_size +first_img_embedding_size,
                        transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size 
                        ]

            
            
            # print("tensor shape: ", prompt_ids[0].size(1))
            req = dist.irecv(recv_tensor, comm_pair.producer, tag = 0)
            req.wait()

            context, clip_feature, img_y, latents = unpack_tensors(recv_tensor, intervals)
            context = context.view(tensors_info[0] , tensors_info[1] , tensors_info[2])
            clip_feature = clip_feature.view(tensors_info[3] , tensors_info[4] , tensors_info[5])
            img_y = img_y.view(tensors_info[6] , tensors_info[7] , tensors_info[8] , tensors_info[9] , tensors_info[10])
            latents = latents.view(tensors_info[11] , tensors_info[12] , tensors_info[13] , tensors_info[14] , tensors_info[15])
        else:
            pass
        
        # image_size = data['images'].size()
        # # image_size[1] = 81


        

        # args = get_args()
        # if args.distributed_vae:
        #     token_length_size = 1
        #     transformer_embedding_size = max_length* transformer_dim
        #     clip_embedding_size = (max_length+2)*1280
        #     first_img_embedding_size = img_dim *( image_size[1]//frame_scale + 1)*(image_size[3]//image_scale)*(image_size[4]//image_scale)
        #     video_embedding_size = video_dim *( image_size[1]//frame_scale + 1)*(image_size[3]//image_scale)*(image_size[4]//image_scale)
            
        #     recv_tensor = torch.empty((token_length_size + transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size ), device=torch.cuda.current_device(), dtype=torch.bfloat16)

        #     intervals = [0, 
        #                 token_length_size, 
        #                 token_length_size + transformer_embedding_size, 
        #                 token_length_size + transformer_embedding_size +clip_embedding_size,
        #                 token_length_size + transformer_embedding_size +clip_embedding_size +first_img_embedding_size,
        #                 token_length_size + transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size 
        #                 ]

            
            
        #     # print("tensor shape: ", prompt_ids[0].size(1))
        #     req = dist.irecv(recv_tensor, comm_pair.producer, tag = 0)
        #     req.wait()

        #     token_length, context, clip_feature, img_y, latents = unpack_tensors(recv_tensor, intervals)
        #     context = context.view(1, token_length, transformer_dim)
        #     clip_feature = clip_feature.view(1, token_length+2, 1280)
        #     img_y = img_y.view(1, img_dim, ( image_size[1]//frame_scale + 1), image_size[3]//image_scale, image_size[4]//image_scale)
        #     latents = latents.view(1, video_dim, image_size[1]//frame_scale + 1, image_size[3]//image_scale, image_size[4]//image_scale)
        # else:
        #     # TODO: remove use fix data
        #     pass


        batch["context"] = context
        batch["clip_feature"] = clip_feature
        batch["image_emb_y"] = img_y
        batch["latents"] = latents
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = tensor.to(torch.cuda.current_device())
        for key, tensor in batch.items():
            sizes_info[key] = tensor.size() if tensor is not None and isinstance(tensor, torch.Tensor)  else len(tensor)
            type_info[key] = tensor.dtype if tensor is not None and isinstance(tensor, torch.Tensor) else type(tensor)

        # print("sizes_info: ", sizes_info)
        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        type_info = torch.distributed.broadcast_object_list([type_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        for key, tensor in batch.items():
            if isinstance(tensor, list):
                torch.distributed.broadcast_object_list(tensor, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
            elif isinstance(tensor, torch.Tensor):
                _broadcast(tensor)
            else:
                raise NotImplementedError(f"Unsupported data type: {type(tensor)}")

    else:
        sizes_info_list = [None]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        type_info_list =[None]
        torch.distributed.broadcast_object_list(type_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        
        batch = {}
        for key, value in sizes_info_list[0].items():
            dtype = type_info_list[0][key]
            
            if isinstance(dtype, torch.dtype):  # dtype 是 torch.float32 这种
                tensor = torch.empty(value, dtype=dtype, device=torch.cuda.current_device())
                _broadcast(tensor)
                batch[key] = tensor
            
            else:  # 表示这是个 list 类型对象
                tensor = [None]*value
                torch.distributed.broadcast_object_list(
                    tensor,
                    src=mpu.get_tensor_context_parallel_src_rank(),
                    group=mpu.get_tensor_context_parallel_group()
                )
                batch[key] = tensor
    
    return batch

def forward_vae(images):
        images = images.to(self.vae.dtype)

        # import torch.distributed as dist
        # rank = dist.get_rank()
        # torch.save(images, f"images_{rank}.pt")

        with torch.no_grad():

            images = rearrange(images, "b f c h w -> b c f h w")
            latents = self.vae.encode(images)
            latents = latents.latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std
        return latents

def encode_prompt(prompter,prompt, positive=True):
    prompt_emb = prompter.encode_prompt(
        prompt, positive=positive, device=torch.cuda.current_device()
    )
    return {"context": prompt_emb}

def encode_image(
    image_encoder,
    image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    msk = torch.ones(1, num_frames, height // 8, width // 8, device=torch.cuda.current_device())
    msk[:, 1:] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)],
        dim=1,
    )
    y = self.vae.encode(
        [vae_input.to(dtype=self.dtype, device=torch.cuda.current_device())],
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=self.dtype, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=self.dtype, device=torch.cuda.current_device())
    y = y.to(dtype=self.dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}

def encode_first_last_image(
    vae,
    image_encoder,
    pil_first_image,
    pil_last_image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
):
    first_image = preprocess_image(pil_first_image.resize((width, height))).to(
        torch.cuda.current_device()
    )
    last_image = preprocess_image(pil_last_image.resize((width, height))).to(
        torch.cuda.current_device()
    )
    # if self.dit.has_image_pos_emb:
    #     clip_context = torch.cat([self.image_encoder.encode_image([first_image]),
    #                             self.image_encoder.encode_image([last_image])], dim=1)
    # else:
    #     clip_context = self.image_encoder.encode_image([first_image])
    clip_context = torch.cat(
        [
            image_encoder.encode_image([first_image]),
            image_encoder.encode_image([last_image]),
        ],
        dim=1,
    )
    msk = torch.ones(1, num_frames, height // 8, width // 8, device=torch.cuda.current_device())
    msk[:, 1:-1] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [
            first_image.transpose(0, 1),
            torch.zeros(3, num_frames - 2, height, width).to(first_image.device),
            last_image.transpose(0, 1),
        ],
        dim=1,
    )
    y = vae.encode(
        [vae_input.to(dtype=torch.bfloat16, device=torch.cuda.current_device())],
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    y = y.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}

def tensor2video(self, frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def prepare_extra_input(self, latents=None):
        return {}
    
    
def encode_video(vae, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    latents = vae.encode(
        input_video,
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    return latents


def decode_video(vae, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    frames = vae.decode(
        latents,
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    return frames

def check_resize_height_width(self, height, width):
    if height % self.height_division_factor != 0:
        height = (
            (height + self.height_division_factor - 1)
            // self.height_division_factor
            * self.height_division_factor
        )
        print(
            f"The height cannot be evenly divided by {self.height_division_factor}. We round it up to {height}."
        )
    if width % self.width_division_factor != 0:
        width = (
            (width + self.width_division_factor - 1)
            // self.width_division_factor
            * self.width_division_factor
        )
        print(
            f"The width cannot be evenly divided by {self.width_division_factor}. We round it up to {width}."
        )
    return height, width

def preprocess_image(image):
    image = (
        torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    return image
def preprocess_images(images):
    return [preprocess_image(image) for image in images]

def vae_output_to_image(vae_output):
    image = vae_output[0].cpu().float().permute(1, 2, 0).numpy()
    image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
    return image

def vae_output_to_video(vae_output):
    video = vae_output.cpu().permute(1, 2, 0).numpy()
    video = [
        Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8")) for image in video
    ]
    return video

def merge_latents(value, latents, masks, scales, blur_kernel_size=33, blur_sigma=10.0):
    if len(latents) > 0:
        blur = GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
        height, width = value.shape[-2:]
        weight = torch.ones_like(value)
        for latent, mask, scale in zip(latents, masks, scales):
            mask = (
                preprocess_image(mask.resize((width, height))).mean(dim=1, keepdim=True)
                > 0
            )
            mask = mask.repeat(1, latent.shape[1], 1, 1).to(
                dtype=latent.dtype, device=latent.device
            )
            mask = blur(mask)
            value += latent * mask * scale
            weight += mask * scale
        value /= weight
    return value

def control_noise_via_local_prompts(
    prompt_emb_global,
    prompt_emb_locals,
    masks,
    mask_scales,
    inference_callback,
    special_kwargs=None,
    special_local_kwargs_list=None,
):
    if special_kwargs is None:
        noise_pred_global = inference_callback(prompt_emb_global)
    else:
        noise_pred_global = inference_callback(prompt_emb_global, special_kwargs)
    if special_local_kwargs_list is None:
        noise_pred_locals = [
            inference_callback(prompt_emb_local) for prompt_emb_local in prompt_emb_locals
        ]
    else:
        noise_pred_locals = [
            inference_callback(prompt_emb_local, special_kwargs)
            for prompt_emb_local, special_kwargs in zip(
                prompt_emb_locals, special_local_kwargs_list
            )
        ]
    noise_pred = merge_latents(noise_pred_global, noise_pred_locals, masks, mask_scales)
    return noise_pred

def extend_prompt(prompt, local_prompts, masks, mask_scales):
    local_prompts = local_prompts or []
    masks = masks or []
    mask_scales = mask_scales or []
    extended_prompt_dict = self.prompter.extend_prompt(prompt)
    prompt = extended_prompt_dict.get("prompt", prompt)
    local_prompts += extended_prompt_dict.get("prompts", [])
    masks += extended_prompt_dict.get("masks", [])
    mask_scales += [100.0] * len(extended_prompt_dict.get("masks", []))
    return prompt, local_prompts, masks, mask_scales