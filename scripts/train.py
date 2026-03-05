#!/usr/bin/env python3
"""
Train a conditional diffusion model for depth prediction from RGB images.

With a full model and training loop.

Example
-------
python scripts/train.py \
  --data-root /media/$USER/ExternalDrive/diffdepth_data/processed/train \
  --epochs 20 \
  --batch-size 16 \
  --out-dir checkpoints
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# creating a class for the dataset that loads the preprocessed .pt chunk files
# which contain lists of samples with 'rgb' and 'depth' tensors, so we can efficiently load them
# in a batch
class DepthDataset(Dataset):
    def __init__(self, root_dir: str | Path):
        root_dir = Path(root_dir)
        self.files = sorted([str(root_dir / f) for f in os.listdir(root_dir) if f.endswith(".pt")])
        if not self.files:
            raise FileNotFoundError(f"No .pt chunk files found in: {root_dir}")
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Always load on CPU; DataLoader workers + CUDA tensors is a common footgun.
        return torch.load(self.files[idx], map_location="cpu")


# chunk_collate for combining lists of samples from each chunk into a single list for mini-batch
# processing in loop
def chunk_collate(batch):
    samples = []
    for chunk in batch:
        samples.extend(chunk)
    return samples


# linear_beta_schedule for diffusion noise schedule
def linear_beta_schedule(timesteps: int):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)



# get_index_from_list to retrieve the appropriate alpha_cumprod value for each timestep in the batch
def get_index_from_list(vals: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
    batch_size = t.shape[0]
    out = vals.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))



# forward diffusion sample so that we can add noise to the clean depth maps according to the 
# diffusion process, which the model will learn to reverse during training
def forward_diffusion_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod = torch.sqrt(get_index_from_list(alphas_cumprod, t, x0.shape))
    sqrt_one_minus = torch.sqrt(1 - get_index_from_list(alphas_cumprod, t, x0.shape))
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus * noise, noise


# preprocess_batch to convert lists of samples into batched tensors for model input
def preprocess_batch(samples: list[dict[str, torch.Tensor]], device: torch.device):
    rgbs = torch.stack([s["rgb"] for s in samples]).to(device)
    depths = torch.stack([s["depth"] for s in samples]).to(device)
    return rgbs, depths


def _auto_depth_divisor(depths: torch.Tensor) -> float:
    # Heuristic: if depth is already in [0, 1], keep it.
    # If it looks like 8-bit PNG, scale by 255. If 16-bit PNG, scale by 65535.
    # This keeps diffusion inputs roughly unit scale.
    max_val = float(depths.detach().max().item())
    if max_val <= 1.0:
        return 1.0
    if max_val <= 255.0:
        return 255.0
    if max_val <= 65535.0:
        return 65535.0
    return max_val


def normalize_depth(depths: torch.Tensor, mode: str, divisor: float | None) -> torch.Tensor:
    if mode == "none":
        return depths

    if divisor is None:
        divisor = _auto_depth_divisor(depths)
    divisor = float(divisor)
    if divisor <= 0:
        raise ValueError("depth divisor must be > 0")

    depths01 = depths / divisor
    if mode == "0_1" or mode == "auto":
        return depths01
    if mode == "-1_1":
        return depths01 * 2.0 - 1.0
    raise ValueError(f"Unknown depth normalization mode: {mode}")



# sinusoidal_embedding for encoding the diffusion timestep into a vector that can be input to the model
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int):
    half = dim // 2
    emb_scale = math.log(10000) / max(half - 1, 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb_scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ResidualBlock for the U-Net architecture, which includes the time embedding projection and skip 
# conneciton for the residual conneciton.
# The block consists of two convolustional layers with group normalization and SiLU activation,
# and a linear layer to project the time embedding to the appropriate number of channels, which is 
# added to the activations after the first convolution. 
# The skip connection is either an identity or a 1x1 convolution if the intput and output channles differ
class ResidualBlock (nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # instantiate the residual connection, so we can add it back after the convs
        residual = self.skip(x)
        # this is the first hidden layer of the block to apply first conv, then norm, then activation
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + residual




# This is the main model architecture for the conditional UNet structure
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base: int = 64, time_dim: int = 256):
        super().__init__()
        # time_mlp for the time embedding projection, which basically does the sinusoidal embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # this is the encoder and decoder structure of the UNet, which we first have the down sampling
        # path with two residual blocks and conv layers for downsampling.
        self.enc1 = ResidualBlock(in_channels, base, time_dim)
        self.down1 = nn.Conv2d(base, base, kernel_size=4, stride = 2, padding=1)

        self.enc2 = ResidualBlock(base, base * 2, time_dim)
        self.down2 = nn.Conv2d(base * 2, base * 2, kernel_size=4, stride=2, padding=1)

        # mid section with the most dense and compressed information included is here
        self.mid = ResidualBlock(base * 2, base * 4, time_dim)

        # upsampling so we undo the downsampling with conv transpose layers, and then we have the 
        # corresponding residual blocks for the decoder, which also include skip connections
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base * 4, base * 2, time_dim)

        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base * 2, base, time_dim)

        self.out = nn.Conv2d(base, out_channels, kernel_size=1)
        self.time_dim = time_dim

    def forward(self, noisy_depth: torch.Tensor, rgb: torch.Tensor, t: torch.Tensor):
        x = torch.cat([noisy_depth, rgb], dim=1)
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(self.down1(x1), t_emb)
        xm = self.mid(self.down2(x2), t_emb)

        x = self.up1(xm)
        x = self.dec1(torch.cat([x, x2], dim=1), t_emb)

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x1], dim=1), t_emb)

        return self.out(x)


# iterate_minibatches is a helper function to take the list of samples from each chunk and 
# create mini-batches of the specified batch size for training, with an option to shuffle the samples
def iterate_minibatches(samples: list[dict[str, torch.Tensor]], batch_size: int, shuffle: bool = True):
    indices = list(range(len(samples)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        yield [samples[i] for i in batch_ids]



# training loop
# iterates over the dataset for the specified number of epochs, and for each chunk of samples,
# it creates mini-batches and processes them through the model, computes the loss, and updates the model
# weghts using AdamW. 
# the loop also includes mixed precision training with torch.amp which basically does the forward
# and backward passes in half precision to save memory and speed up training on compatible GPUs, while
# still maintaining the stability of the training process.
def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    dataset = DepthDataset(args.data_root)
    # wrap dataloader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=chunk_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ConditionalUNet(base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    # initialize the noise schedule for the diffusion process
    betas = linear_beta_schedule(args.timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    # GradScaler is only needed/valid for fp16. BF16 generally trains fine without scaling,
    # and is often more stable on ROCm.
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        updates = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for chunk_samples in progress:
            for mini_samples in iterate_minibatches(chunk_samples, args.batch_size, shuffle=True):
                # take the mini-batch of samples and preprocess them into tensors for model input
                rgb, depth = preprocess_batch(mini_samples, device=device)

                # Normalize depth to a roughly unit scale for diffusion stability.
                depth = normalize_depth(depth, mode=args.depth_norm, divisor=args.depth_divisor)
                # sample random timesteps
                t = torch.randint(0, args.timesteps, (depth.shape[0],), device=device).long()
                # sample the noisy depth maps according to the forward process
                noisy_depth, noise = forward_diffusion_sample(depth, t, alphas_cumprod)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    # predict the noise from the noisy depth and RGB input
                    # then compute the MSE loss between the predicted noise and the acutal noise
                    noise_pred = model(noisy_depth, rgb, t)
                    loss = F.mse_loss(noise_pred, noise)
                
                if use_scaler:
                    # scaler does the backward pass with scaling for mixed precision fp16
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                # updates for each parameter that we are interested in for logging
                loss_value = float(loss.detach().item())
                epoch_loss += loss_value
                updates += 1
                global_step += 1
                progress.set_postfix(loss=f"{loss_value:.5f}")

        avg_loss = epoch_loss / max(updates, 1)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}, updates={updates}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "avg_loss": avg_loss,
            }
            ckpt_path = out_dir / f"diffdepth_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train conditional diffusion model for depth prediction")
    parser.add_argument("--data-root", required=True, help="Path to processed train chunks (*.pt)")
    parser.add_argument("--out-dir", default="checkpoints", help="Directory to save model checkpoints")

    parser.add_argument("--cpu", action="store_true", help="Force CPU training")

    parser.add_argument(
        "--depth-norm",
        default="auto",
        choices=["auto", "none", "0_1", "-1_1"],
        help="Depth normalization applied during training",
    )
    parser.add_argument(
        "--depth-divisor",
        type=float,
        default=None,
        help="Optional divisor for depth normalization (e.g., 255 or 65535). If omitted, uses an auto heuristic.",
    )

    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size within each chunk")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps T")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value (<=0 disables)")

    parser.add_argument("--base-channels", type=int, default=64, help="Base channels for U-Net")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")

    parser.add_argument("--save-every", type=int, default=1, help="Checkpoint frequency in epochs")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument(
        "--amp-dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Autocast dtype when --amp is enabled (bf16 is usually more stable on ROCm)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
