import argparse
import logging
import os
import json
from copy import deepcopy
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# Import Models
from models.sit import SiT_models
from models.side_net import BASideNet 

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
import wandb
import math

# Import BA Logic (Side Tuning Version)
from BA_utils_side import BALoss, BA_single_anchor_sampler

logger = get_logger(__name__)

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

# ============================================================
# EMA 
# ============================================================
class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def step(self, model):
        model_params = dict(model.named_parameters())
        ema_params = dict(self.model.named_parameters())
        for name, param in model_params.items():
            if name in ema_params:
                ema_params[name].mul_(self.decay).add_(
                    param.data.to(ema_params[name].dtype), alpha=1 - self.decay
                )
    
    def to(self, device):
        self.model.to(device)

def main(args):
    # 1. Setup Accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=4)

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    device = accelerator.device

    # ============================================================
    # 2. Load & Freeze Base Model (SiT)
    # ============================================================
    assert args.resolution % 8 == 0
    latent_size = args.resolution // 8
    
    # Block kwargs for SiT
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    # Initialize Base Model (SiT)
    base_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        learn_sigma=False,
        **block_kwargs
    ).to(device)

    # Load Pre-trained Checkpoint
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        m, u = base_model.load_state_dict(state_dict, strict=False) 
        if accelerator.is_main_process:
            logger.info(f"Loaded frozen base model from {args.pretrained_ckpt}")
            if len(m) > 0: 
                logger.warning(f"Missing keys in base model: {len(m)} (First few: {m[:5]})")

    # Freeze Base Model Completely
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    
    if args.mixed_precision == "fp16":
        base_model.to(dtype=torch.float16)
    elif args.mixed_precision == "bf16":
        base_model.to(dtype=torch.bfloat16)

    # ============================================================
    # 3. Create Side Network (Trainable) & EMA
    # ============================================================
    side_model = BASideNet(
        in_channels=args.SideNet_in_channels, 
        base_channels=args.SideNet_base_channels,
        h_emb_dim=args.SideNet_h_emb_dim, 
        num_layers=args.SideNet_depth 
    ).to(device)
    
    side_model.train() 

    ema_side_model = EMAModel(side_model, decay=args.ema_decay)
    ema_side_model.to(device)

    # ============================================================
    # 4. Optimizer (Only for SideNet)
    # ============================================================
    if accelerator.is_main_process:
        logger.info(f"Trainable Parameters (SideNet): {sum(p.numel() for p in side_model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        side_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 5. Data
    transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = ImageFolder(args.data_dir, transform=transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()
    
    latents_scale = torch.tensor([0.18215]*4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.]*4).view(1, 4, 1, 1).to(device)

    # 6. Prepare
    side_model, optimizer, train_dataloader = accelerator.prepare(
        side_model, optimizer, train_dataloader
    )

    # Sampling noise (Fixed for visualization)
    sample_batch_size = 16
    ys_sample = torch.randint(1000, size=(sample_batch_size,), device=device)
    xT_sample = torch.randn((sample_batch_size, 4, latent_size, latent_size), device=device)
    
    # Loss Function
    loss_fn = BALoss()

    # 7. Resume Logic
    global_step = 0
    start_epoch = 0
    if args.resume_step > 0:
        ckpt_path = f"{save_dir}/checkpoints/BA_side_{args.resume_step:06d}.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            # Load SideNet
            unwrapped_side = accelerator.unwrap_model(side_model)
            unwrapped_side.load_state_dict(ckpt['side_model'])

            # Load EMA Model
            if 'ema_side_model' in ckpt:
                ema_side_model.model.load_state_dict(ckpt['ema_side_model'])
            else:
                logger.warning("No EMA state found in checkpoint. Re-initializing EMA from current SideNet state.")
                ema_side_model = EMAModel(unwrapped_side, decay=args.ema_decay)
                ema_side_model.to(device)
            
            # Load Optimizer
            optimizer.load_state_dict(ckpt['optimizer'])
            
            global_step = args.resume_step
            start_epoch = global_step // len(train_dataloader)
            
            if accelerator.is_main_process:
                logger.info(f"Resumed Side Tuning from step {global_step}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Resume step {args.resume_step} requested but checkpoint not found at {ckpt_path}")

    # 8. Training Loop
    if accelerator.is_main_process:
        accelerator.init_trackers("BA_SideTuning", config=vars(args))

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(start_epoch, args.epochs):
        side_model.train()
        for x, y in train_dataloader:
            with torch.no_grad():
                x = x.to(device)
                posterior = vae.encode(x).latent_dist
                x_lat = posterior.sample().mul_(0.18215)
            
            with accelerator.accumulate(side_model):
                loss, loss_v, loss_u = loss_fn(
                    base_model,       # Frozen SiT
                    side_model,       # Trainable SideNet
                    x_lat,            # Latents
                    model_kwargs={'y': y}
                )
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(side_model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                # 步进更新 EMA
                unwrapped_side = accelerator.unwrap_model(side_model)
                ema_side_model.step(unwrapped_side)

                progress_bar.update(1)
                global_step += 1
                
                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrapped_side = accelerator.unwrap_model(side_model)
                        state = {
                            'side_model': unwrapped_side.state_dict(),
                            'ema_side_model': ema_side_model.model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'step': global_step
                        }

                        torch.save(state, f"{save_dir}/checkpoints/BA_side_{global_step:06d}.pt")
                        logger.info(f"Saved SideNet checkpoint at step {global_step}")

                # Sampling
                if global_step % args.sampling_steps == 0:
                    ema_side_model.model.eval()
                    with torch.no_grad():
                        with accelerator.autocast():
                            samples = BA_single_anchor_sampler(
                                base_model=base_model,
                                side_model=ema_side_model.model, 
                                latents=xT_sample,
                                y=ys_sample,
                                num_steps=args.num_steps, 
                                nodes=args.nodes,
                                cfg_scale=args.cfg_scale
                            )
                        
                        samples = samples.to(torch.float32)
                        samples = vae.decode((samples - latents_bias) / latents_scale).sample
                        samples = (samples + 1) / 2.
                    if accelerator.is_main_process:
                        grid = array2grid(samples)
                        # Save
                        save_path = os.path.join(save_dir, f"sample_step_{global_step:06d}.png")
                        Image.fromarray(grid).save(save_path)
                        print(f"Saved samples to {save_path}")
                        
            logs = {"loss": loss.item(), "loss_v": loss_v.item(), "loss_u": loss_u.item()}
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-ckpt", type=str, required=True, help="Path to base SiT checkpoint")
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--output-dir", type=str, default="exps_BA")
    parser.add_argument("--exp-name", type=str, default="BA_side_tuning")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    
    # Model
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    
    # SiT Configs (needed for loading)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # BA & Sampling
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=2) # only for Single-anchor solver
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=300000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="fp16")
    parser.add_argument("--resume-step", type=int, default=0)
    
    # Checkpointing
    parser.add_argument("--checkpointing-steps", type=int, default=2000)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    # SideNet
    parser.add_argument("--SideNet-depth", type=int, default=4)
    parser.add_argument("--SideNet-in-channels", type=int, default=4)
    parser.add_argument("--SideNet-base-channels", type=int, default=256)
    parser.add_argument("--SideNet-h-emb-dim", type=int, default=256)

    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.9996, help="Decay rate for EMA model")

    args = parser.parse_args()
    main(args)
