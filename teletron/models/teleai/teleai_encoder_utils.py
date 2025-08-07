import torch
from einops import rearrange
from torchvision.transforms.functional import to_pil_image
import numpy as np

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
    vae,
    image_encoder,
    image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
    dtype=torch.bfloat16
):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    msk = torch.ones(1, num_frames, height // 8, width // 8, device=torch.cuda.current_device())
    # print("msk create shape:", 1, num_frames, height // 8, width // 8 ) # 1, 81, 56, 98
    msk[:, 1:] = 0 # 1, 1:81, 56, 98
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
    ) # 1, 4, 56, 98; # 1, 80, 56, 98 => 1, 84, 56, 98
    # print("msk view shape:", 1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8) # 1, 21, 4, 56, 98
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)],
        dim=1,
    )
    y = vae.encode(
        [vae_input.to(dtype=dtype, device=torch.cuda.current_device())],
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}

def encode_image_with_mask(
        vae,
        image_encoder, 
        image, 
        num_frames,
        height, 
        width, 
        msk, 
        ref_images, 
        tiled=False, 
        tile_size=(34, 34), 
        tile_stride=(18, 16),
        dtype=torch.bfloat16
    ):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    ref_images = rearrange(ref_images, 'b t c h w -> b c t h w')
    y = encode_video(
        vae, 
        ref_images.to(dtype=dtype, device=torch.cuda.current_device()),
        tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
    y = y.unsqueeze(0)
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    msk = msk.transpose(1, 2).to(torch.cuda.current_device())
    y = torch.concat([msk, y], dim=1)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
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
    dtype=torch.bfloat16
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
        [vae_input.to(dtype=dtype, device=torch.cuda.current_device())],
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}

# def report_memory():


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


def get_encoder_features(batch, prompter, vae, tiler_kwargs, image_encoder, dtype=torch.bfloat16):
    with torch.no_grad():
        prompt_emb = encode_prompt(prompter,batch["dense_prompt"][0])
        latents = encode_video(vae,
            rearrange(batch["images"], "b t c h w -> b c t h w").to(
                dtype=dtype, device=torch.cuda.current_device()
            ),
            **tiler_kwargs,
        )[0]
        _, num_frames, _, height, width = batch["images"].shape
        # print("images: ",height, width )
        if 'raw_last_image' in batch:
            raw_first_image = batch["raw_first_image"]
            pil_first_image = to_pil_image(
                raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            raw_last_image = batch['raw_last_image']
            pil_last_image = to_pil_image(
                raw_last_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            image_emb = encode_first_last_image(
                vae,image_encoder, pil_first_image, pil_last_image, num_frames, height, width, dtype=dtype
            )
        elif 'raw_first_image' in batch:
            raw_first_image = batch["raw_first_image"]
            pil_image = to_pil_image(
                raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            image_emb = encode_image(vae, image_encoder, pil_image, num_frames, height, width, dtype=dtype)
        elif 'ref_images' in batch:
            first_image = (batch['ref_images'] + 1) / 2 * 255
            ref_mask = batch["ref_mask"]
            ref_images = batch["ref_images"]
            pil_image = to_pil_image(first_image[0][0].cpu().permute(1,2,0).numpy().astype(np.uint8))
            image_emb = encode_image_with_mask(vae, image_encoder, pil_image, num_frames,
                                                height, width, ref_mask, ref_images, dtype=dtype)
        
        
        latents = latents.unsqueeze(0).to(dtype=dtype, device=torch.cuda.current_device())

        # Data
        prompt_emb["context"] = prompt_emb["context"][0].to(
            dtype=dtype, device=torch.cuda.current_device()
        )
        prompt_emb["context"] = prompt_emb["context"].unsqueeze(0)

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = (
                image_emb["clip_feature"][0]
                .to(dtype=dtype, device=torch.cuda.current_device())
                .unsqueeze(0)
            )
        if "y" in image_emb:
            image_emb["y"] = (
                image_emb["y"][0]
                .to(dtype=dtype, device=torch.cuda.current_device())
                .unsqueeze(0)
            )
    return prompt_emb, image_emb, latents

@torch.no_grad
def get_context(batch, prompter, dtype=torch.bfloat16):
    prompt_emb = encode_prompt(prompter, batch["struct_prompt"][0])
    prompt_emb["context"] = prompt_emb["context"].to(
            dtype=dtype, device=torch.cuda.current_device()
    )
    return prompt_emb["context"]

@torch.no_grad
def get_img_clip_feature(batch, image_encoder, dtype=torch.bfloat16):
    _, num_frames, _, height, width = batch["images"].shape
    if 'raw_last_image' in batch:
        raise NotImplementedError("raw_last_image is not supported yet")
        raw_first_image = batch["raw_first_image"]
        pil_first_image = to_pil_image(
            raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        raw_last_image = batch['raw_last_image']
        pil_last_image = to_pil_image(
            raw_last_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        image_emb = encode_first_last_image(
            vae,image_encoder, pil_first_image, pil_last_image, num_frames, height, width, dtype=dtype
        )
    elif 'raw_first_image' in batch:
        raw_first_image = batch["raw_first_image"]
        pil_image = to_pil_image(
            raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        image = preprocess_image(pil_image.resize((width, height))).to(torch.cuda.current_device())
        clip_context = image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    elif 'ref_images' in batch:
        raise NotImplementedError("ref_images is not supported yet")
        first_image = (batch['ref_images'] + 1) / 2 * 255
        ref_mask = batch["ref_mask"]
        ref_images = batch["ref_images"]
        pil_image = to_pil_image(first_image[0][0].cpu().permute(1,2,0).numpy().astype(np.uint8))
        image_emb = encode_image_with_mask(vae, image_encoder, pil_image, num_frames,
                                            height, width, ref_mask, ref_images, dtype=dtype)
    return clip_context

@torch.no_grad
def get_img_emb_y(batch, vae, dtype=torch.bfloat16):
    _, num_frames, _, height, width = batch["images"].shape
    if 'ref_images' in batch:
        # assert False, "ref_images is not supported yet"
        ref_images = rearrange(batch["ref_images"], "b t c h w -> b c t h w")
        y = vae.encode(
            ref_images.to(dtype=dtype, device=torch.cuda.current_device()),
            device=torch.cuda.current_device(),
            tiled=False,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )
        msk = batch['ref_mask'].transpose(1, 2).to(dtype=dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y], dim=1)
    
    elif 'raw_first_image' in batch:
        raw_first_image = batch["raw_first_image"]
        pil_image = to_pil_image(
            raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        image = preprocess_image(pil_image.resize((width, height))).to(torch.cuda.current_device())
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=torch.cuda.current_device())

        msk[:, 1:] = 0 # 1, 1:81, 56, 98
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        ) # 1, 4, 56, 98; # 1, 80, 56, 98 => 1, 84, 56, 98

        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8) # 1, 21, 4, 56, 98
        msk = msk.transpose(1, 2)[0]
        vae_input = torch.concat(
            [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)],
            dim=1,
        )
        y = vae.encode(
            [vae_input.to(dtype=dtype, device=torch.cuda.current_device())],
            device=torch.cuda.current_device(),
            tiled=False,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )[0]
        y = y.to(dtype=dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=dtype, device=torch.cuda.current_device())
    
    
    return y

@torch.no_grad
def get_latents(batch, vae, dtype=torch.bfloat16):
    latents = encode_video(vae,
            rearrange(batch["images"], "b t c h w -> b c t h w").to(
                dtype=dtype, device=torch.cuda.current_device()
            ),
            tiled=False,
            tile_size=(34, 34), 
            tile_stride=(18, 16),
        )[0]
    latents = latents.unsqueeze(0).to(dtype=dtype, device=torch.cuda.current_device())
    return latents

@torch.no_grad
def get_noise(batch, dtype=torch.bfloat16):
    if 'latents' in batch:
        return torch.randn_like(batch['latents']).to(dtype=dtype, device=torch.cuda.current_device())
    else:
        bsz, num_frames, _, height, width = batch["images"].shape
        return torch.randn(bsz, 16, (num_frames + 3) // 4, height // 8, width // 8).to(dtype=dtype, device=torch.cuda.current_device())

@torch.no_grad
def get_fake_latents(batch, vae, dtype=torch.bfloat16):
    latents = batch["latents"]
    bsz, num_frames, video_channels, height, width = batch["images"].shape
    
    low_res_video = torch.nn.functional.interpolate(
        rearrange(batch["images"], "b t c h w -> (b t) c h w"),
        size=(height // 2, width // 2),
        mode='bilinear'
    ).reshape(bsz, num_frames, video_channels, height // 2, width // 2)
    
    low_res_latent = encode_video(vae,
        rearrange(low_res_video, "b t c h w -> b c t h w").to(
            dtype=dtype, device=torch.cuda.current_device()
        ),
        tiled=True,
        tile_size=(34, 34), 
        tile_stride=(18, 16),
    ) # b c t h w
    
    bsz, latent_channels, latent_frames, latent_height, latent_width = latents.shape
    fake_latents = torch.nn.functional.interpolate(
        rearrange(low_res_latent, "b c t h w -> (b t) c h w"),
        size=(latent_height, latent_width),
        mode='nearest'
    ).reshape(bsz, latent_frames, latent_channels, latent_height, latent_width)[0] # t, c, h, w
    fake_latents = fake_latents.permute(1, 0, 2, 3) # c, t, h, w
    
    fake_latents = fake_latents.unsqueeze(0).to(dtype=dtype, device=torch.cuda.current_device())

    return fake_latents

@torch.no_grad
def get_frame_interval(batch, dtype=torch.bfloat16):
    return batch['frame_interval'].to(
        dtype=dtype, device=torch.cuda.current_device()
    )