import argparse
import os
import random
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from einops import rearrange
from torchvision.io import write_video
import pandas as pd

from teletron.utils.misc import set_seed
from teletron.models.causwan.modules.wan_wrapper import WanVAEWrapper

# =====================  Argument Parsing  =====================
parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, default="/gemini/space/xxz/check_data_latent",
                    help="Where to save generated & original videos")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed for reproducibility")
parser.add_argument("--sample_num", type=int, default=100,
                    help="Number of random samples to process")
args = parser.parse_args()

# =====================  DDP / Device  =====================
if "LOCAL_RANK" in os.environ:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + local_rank)
else:
    local_rank = 0
    device = torch.device("cuda")
    set_seed(args.seed)

random.seed(args.seed)

torch.set_grad_enabled(False)
vae = WanVAEWrapper().to(device)
vae.requires_grad_(False)

# =====================  Dataset  =====================
class CheckTensorDataset(torch.utils.data.Dataset):
    """Return dict with latent tensor + original video path + text"""
    def __init__(self, latent_root: str, csv_path: str, video_root: str):
        self.records = []
        meta = pd.read_csv(csv_path)

        name_col = "file_name" if "file_name" in meta.columns else "file_path"
        text_col = next(
            (c for c in ["text", "caption", "prompt", "description"] if c in meta.columns),
            None
        )
        if text_col is None:
            raise ValueError("CSV must contain one of the following text columns: text, caption, prompt, description")

        for fname, text in zip(meta[name_col], meta[text_col]):
            latent_path = os.path.join(latent_root, fname) + ".tensors.pth"
            video_path = os.path.join(video_root, fname)
            self.records.append((latent_path, video_path, str(text).strip()))

        assert self.records, "No samples found"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        latent_path, video_path, text = self.records[idx]
        try:
            data = torch.load(latent_path, weights_only=True, map_location="cpu")
        except Exception:
            data = {"latents": torch.zeros(1)}
        data["__video_path"] = video_path
        data["__text"] = text
        return data

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output

# ---------- paths ----------
video_root  = "/gemini/space/xxz/datasets/1-HumanData/merged_videos"
latent_root = "/gemini/space/xxz/datasets/1-HumanData/merged_videos_latents"
csv_path    = "/gemini/space/xxz/datasets/1-HumanData/merged_videos.csv"
csv_path    = "/gemini/space/xxz/datasets/1-HumanData/filtered.csv"

# video_root  = "/gemini/space/xxz/datasets/2-EnviromentData/merged_videos"
# latent_root = "/gemini/space/xxz/datasets/2-EnviromentData/merged_videos_latents"
# csv_path    = "/gemini/space/xxz/datasets/2-EnviromentData/merged_videos.csv"
# csv_path    = "/gemini/space/xxz/datasets/2-EnviromentData/filtered.csv"

# ---------- Load dataset & sample ----------
dataset = CheckTensorDataset(latent_root, csv_path, video_root)

if len(dataset) < args.sample_num:
    raise ValueError(f"Dataset only contains {len(dataset)} samples, less than requested {args.sample_num}.")
sampled_indices = random.sample(range(len(dataset)), args.sample_num)
sampled_indices = range(len(dataset))
subset = Subset(dataset, sampled_indices)

dataloader = DataLoader(subset, batch_size=1, num_workers=2, drop_last=False)

if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
if dist.is_initialized():
    dist.barrier()

# =====================  Main Loop  =====================
for batch_idx, batch in tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            disable=(local_rank != 0)):

    # Skip empty batches
    if batch["latents"].sum() == 0:
        continue

    # ==================== Data Preparation ====================
    latents = rearrange(batch["latents"], "b c t h w -> b t c h w").to(device).float()
    src_path = batch["__video_path"][0]
    stem = Path(src_path).stem
    ext = Path(src_path).suffix
    text = batch["__text"][0]

    if local_rank == 0:
        print(f"\n[{stem + ext}] → \"{text}\"")
        print(f'Input latents shape: {latents.shape}')

    # ==================== Generate vid_gen (Original Video) ====================
    keep_indices = [0, 10, 10, 11, 12]
    mask_latents = torch.zeros_like(latents)
    mask_latents[:, :len(keep_indices)] = latents[:, keep_indices]

    vid_gen = vae.decode_to_pixel(latents)
    vid_gen = (vid_gen * 0.5 + 0.5).clamp(0, 1)

    vid_gen_mask = vae.decode_to_pixel(mask_latents)
    vid_gen_mask = (vid_gen_mask * 0.5 + 0.5).clamp(0, 1)
    
    if local_rank == 0:
        print(f'vid_gen shape: {vid_gen.shape}')

    # ==================== Generate vid_test (Test Video) ====================
    # Create test video with last 5 frames copied to beginning
    vid_test = torch.zeros_like(vid_gen)
    vid_test[:, 0:5] = vid_gen[:, -5:]
    
    # Encode-decode cycle for testing
    vid_test_normalized = vid_test * 2.0 - 1.0
    vid_test_rearranged = rearrange(vid_test_normalized, "b t c h w -> b c t h w")
    vid_test_latents = vae.encode_to_latent(vid_test_rearranged)
    
    # Decode back to pixels
    vid_test = vae.decode_to_pixel(vid_test_latents)
    vid_test = (vid_test * 0.5 + 0.5).clamp(0, 1)
    
    if local_rank == 0:
        print(f'vid_test shape: {vid_test.shape}')

    # ==================== Generate vid_img (Image Video) ====================
    # Extract start frame (second to last frame) and encode
    start_frame_pixels = vid_gen[:, -5:-4, :] * 2.0 - 1.0
    start_frame_latents = encode(vae, (
        start_frame_pixels
    ).transpose(2, 1)).transpose(2, 1)
    
    # Concatenate with repeated last latent frame
    repeated_latents = latents[:, -1:].repeat(1, 20, 1, 1, 1)
    combined_latents = torch.cat([start_frame_latents, repeated_latents], dim=1)
    
    # Decode to image video
    vid_img = vae.decode_to_pixel(combined_latents)
    vid_img = (vid_img * 0.5 + 0.5).clamp(0, 1)
    
    if local_rank == 0:
        print(f'vid_img shape: {vid_img.shape}')

    # ==================== Convert to Saving Format ====================
    # Convert all videos to uint8 format for saving
    vid_gen_save = 255.0 * rearrange(vid_gen, "b t c h w -> b t h w c").cpu()
    vid_gen_mask_save = 255.0 * rearrange(vid_gen_mask, "b t c h w -> b t h w c").cpu()
    vid_test_save = 255.0 * rearrange(vid_test[:, 0:5], "b t c h w -> b t h w c").cpu()
    vid_img_save = 255.0 * rearrange(vid_img[:, :5], "b t c h w -> b t h w c").cpu()

    # ==================== Save Videos ====================
    if local_rank == 0:
        # Define output paths
        vid_gen_path = os.path.join(args.output_folder, f"{stem}_vid_gen{ext}")
        vid_gen_mask_path = os.path.join(args.output_folder, f"{stem}_vid_gen_mask{ext}")
        vid_test_path = os.path.join(args.output_folder, f"{stem}_vid_test{ext}")
        vid_img_path = os.path.join(args.output_folder, f"{stem}_vid_img{ext}")
        
        # Save all three videos
        write_video(vid_gen_path, vid_gen_save[0], fps=16)
        write_video(vid_gen_mask_path, vid_gen_mask_save[0], fps=16)
        write_video(vid_test_path, vid_test_save[0], fps=16)
        write_video(vid_img_path, vid_img_save[0], fps=16)
        
        print(f"Saved: {vid_gen_path}")
        print(f"Saved: {vid_gen_mask_path}")
        print(f"Saved: {vid_test_path}")
        print(f"Saved: {vid_img_path}")