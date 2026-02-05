# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples from a SiT-REPA model using the BA-solver. Done.
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from models.side_net import BASideNet
from BA_utils_side import BA_bidirectional_anchor_sampler, BA_single_anchor_sampler
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from utils import load_legacy_checkpoints, download_model

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # ============================================================
    # 1. Load Base Model (SiT)
    # ============================================================
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    
    base_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        **block_kwargs,
    ).to(device)

    # Load Base Checkpoint
    if args.base_ckpt is None:
        # Default download logic for standard SiT
        args.base_ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(args.base_ckpt, map_location=f'cuda:{device}')

    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
        )
    
    # Handle potentially nested state dicts
    if 'model' in state_dict:
        state_dict = state_dict['model']
    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()  
    
    # ============================================================
    # 2. Load Side Network (BA)
    # ============================================================
    side_model = BASideNet(
        in_channels=args.SideNet_in_channels, 
        base_channels=args.SideNet_base_channels,
        h_emb_dim=args.SideNet_h_emb_dim, 
        num_layers=args.SideNet_depth      
    ).to(device)

    assert args.side_ckpt is not None, "Must provide --side-ckpt for BA generation"
    
    side_ckpt = torch.load(args.side_ckpt, map_location=f'cuda:{device}')
    
    # In training script, SideNet is saved under 'side_model' key
    if 'side_model' in side_ckpt:
        side_state_dict = side_ckpt['side_model']
    else:
        side_state_dict = side_ckpt

    # Remove 'module.' prefix if present (from DDP training)
    side_state_dict = {k.replace("module.", ""): v for k, v in side_state_dict.items()}
    
    side_model.load_state_dict(side_state_dict)
    side_model.eval()

    if rank == 0:
        print(f"Loaded Base SiT from {args.base_ckpt}")
        print(f"Loaded Side BA model from {args.side_ckpt}")

    # ============================================================
    # 3. Setup VAE
    # ============================================================
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    # Create folder to save samples
    model_string_name = args.model.replace("/", "-")
    # Naming convention based on Side Checkpoint
    ckpt_string_name = os.path.basename(args.side_ckpt).replace(".pt", "")
    folder_name = f"BA-{ckpt_string_name}-steps-{args.num_steps}-cfg-{args.cfg_scale}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Calculation for DDP sampling
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")

    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    # Latent Constants
    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(device)

    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, base_model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        with torch.no_grad():
            # ============================================================
            # BA Sampler Call
            # ============================================================
            samples = BA_bidirectional_anchor_sampler(
                base_model=base_model,
                side_model=side_model,
                latents=z,
                y=y,
                cfg_interval_end=args.cfg_interval_end,
                cfg_interval_start=args.cfg_interval_start,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale
            ).to(torch.float32)

            # Decode
            samples = vae.decode((samples - latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        
        total += global_batch_size

    dist.barrier()

    dist.destroy_process_group()
    if rank == 0:
        print("Rank 0: Starting NPZ creation...")
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--base-ckpt", type=str, default=None, help="Path to Base SiT checkpoint (Frozen)")
    parser.add_argument("--side-ckpt", type=str, required=True, help="Path to Side BA checkpoint (Trainable)")
    parser.add_argument("--sample-dir", type=str, default="samples-test")
    
    # Base Model Config
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    # Sampling Config
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--num-steps", type=int, default=7, help="Number of BA sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cfg-interval-end", type=float, default=0.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.7)
    
    # Misc
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    #SideNet
    parser.add_argument("--SideNet-depth", type=int, default=4)
    parser.add_argument("--SideNet-in-channels", type=int, default=4)
    parser.add_argument("--SideNet-base-channels", type=int, default=256)
    parser.add_argument("--SideNet-h-emb-dim", type=int, default=256)

    args = parser.parse_args()
    main(args)
