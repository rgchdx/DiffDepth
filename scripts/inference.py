"""scripts/inference.py

Diffusion inference for depth prediction.

This script matches the training code in scripts/train.py:
- Loads a ConditionalUNet checkpoint.
- Runs DDPM reverse diffusion to sample a depth map conditioned on an RGB image.

You can run inference either:
1) On processed chunk files produced by scripts/preprocess.py (recommended), or
2) Directly from a 2-column CSV of (rgb_path, depth_path) for the NYU split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image

from train import ConditionalUNet, linear_beta_schedule


@dataclass(frozen=True)
class InferenceConfig:
    timesteps: int
    depth_norm: str
    depth_divisor: float | None


# locad the model checkpoint and return the model and inference config
def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[ConditionalUNet, InferenceConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {}) or {}
    base_channels = int(args.get("base_channels", 64))
    timesteps = int(args.get("timesteps", 1000))
    depth_norm = str(args.get("depth_norm", "auto"))
    depth_divisor = args.get("depth_divisor", None)
    if depth_divisor is not None:
        depth_divisor = float(depth_divisor)

    model = ConditionalUNet(base=base_channels).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return model, InferenceConfig(timesteps=timesteps, depth_norm=depth_norm, depth_divisor=depth_divisor)



# Iterate over samples from processed chunk files. Each chunk file is expected
# to contain a list of samples, where each sample is a dict with at least an
# "rgb" key containing a (3,H,W) tensor in [0,1]. The "depth" key is optional
# and not used for inference.
def _iter_samples_from_chunks(root: Path, limit: int | None) -> Iterator[dict[str, torch.Tensor]]:
    pt_files = sorted(root.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt chunk files found in: {root}")
    
    yielded = 0
    for pt in pt_files:
        chunk = torch.load(pt, map_location="cpu")
        if not isinstance(chunk, list):
            raise ValueError(f"Expected a list of samples in chunk file: {pt}")
        for sample in chunk:
            yield sample
            yielded += 1
            if limit is not None and yielded >= limit:
                return



def _load_rgb_from_path(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    rgb = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return rgb


# Single reverse diffusion step from DDPM, returning the predicted x t-1
# given x t and the predicted noise epsilon from model.
def _p_sample(
    model: ConditionalUNet,
    x: torch.Tensor,
    rgb: torch.Tensor,
    t: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
):
    b = x.shape[0]
    t_tensor = torch.full((b,), t, device=x.device, dtype=torch.long)
    eps = model(x, rgb, t_tensor)

    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    if t == 0:
        alpha_bar_prev = torch.tensor(1.0, device=x.device)
    else:
        alpha_bar_prev = alphas_cumprod[t - 1]

    posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

    mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps)

    if t == 0:
        return mean

    noise = torch.randn_like(x)
    return mean + torch.sqrt(posterior_var) * noise



@torch.no_grad()
def sample_depth(
    model: ConditionalUNet,
    rgb: torch.Tensor,
    timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    # rgb: (B,3,H,W) in [0,1]
    b, _, h, w = rgb.shape
    x = torch.randn((b, 1, h, w), device=device)

    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for t in reversed(range(timesteps)):
        x = _p_sample(model, x, rgb, t, betas, alphas, alphas_cumprod)
    return x


def _auto_depth_divisor_for_viz(depth: torch.Tensor) -> float:
    max_val = float(depth.detach().max().item())
    if max_val <= 1.0:
        return 1.0
    if max_val <= 255.0:
        return 255.0
    if max_val <= 65535.0:
        return 65535.0
    return max_val


def depth_to_uint8(depth: torch.Tensor, depth_norm: str, depth_divisor: float | None) -> torch.Tensor:
    # depth: (1, H, W) or (H, W)
    if depth.ndim == 3:
        depth = depth[0]

    if depth_norm == "-1_1":
        depth01 = (depth + 1.0) / 2.0
    elif depth_norm in ("0_1", "auto"):
        depth01 = depth
    elif depth_norm == "none":
        div = float(depth_divisor) if depth_divisor is not None else _auto_depth_divisor_for_viz(depth)
        depth01 = depth / div
    else:
        # Fallback for unexpected values
        depth01 = depth

    depth01 = depth01.clamp(0.0, 1.0)
    return (depth01 * 255.0).round().to(torch.uint8)


def depth_to_uint8_viz(
    depth: torch.Tensor,
    viz: str,
    depth_norm: str,
    depth_divisor: float | None,
) -> torch.Tensor:
    """Convert a predicted depth map to uint8 for visualization.

    `viz=minmax` is best for quickly seeing structure, even when the model's
    absolute scale is off early in training.
    """

    if depth.ndim == 3:
        depth = depth[0]
    d = depth.detach().float().cpu()

    if viz == "clamp":
        return depth_to_uint8(d, depth_norm=depth_norm, depth_divisor=depth_divisor)

    if viz == "minmax":
        dmin = float(d.min().item())
        dmax = float(d.max().item())
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
            return torch.zeros_like(d, dtype=torch.uint8)
        d01 = (d - dmin) / (dmax - dmin)
        return (d01.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    if viz == "p2p98":
        lo = float(torch.quantile(d.flatten(), 0.02).item())
        hi = float(torch.quantile(d.flatten(), 0.98).item())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return torch.zeros_like(d, dtype=torch.uint8)
        d01 = (d - lo) / (hi - lo)
        return (d01.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    raise ValueError(f"Unknown --viz mode: {viz}")


def save_depth_png(depth_uint8: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(depth_uint8.cpu().numpy(), mode="L")
    img.save(out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DiffDepth inference (conditional diffusion depth sampling)")
    p.add_argument("--ckpt", required=True, help="Path to a checkpoint file (*.pt) from scripts/train.py")
    p.add_argument("--out-dir", required=True, help="Directory to write predicted depth PNGs")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-root", help="Processed chunk root (folder containing *.pt chunks)")
    src.add_argument("--csv-path", help="2-column CSV (rgb_path, depth_path) for NYU split")
    p.add_argument("--project-root", default=None, help="Root to resolve relative paths in --csv-path")

    p.add_argument("--limit", type=int, default=20, help="Max samples to run")
    p.add_argument("--size", type=int, default=256, help="Resize size for CSV/raw inference")
    p.add_argument("--timesteps", type=int, default=None, help="Override diffusion timesteps (default: from checkpoint)")
    p.add_argument(
        "--viz",
        default="minmax",
        choices=["minmax", "clamp", "p2p98"],
        help="How to scale predictions when saving PNGs (minmax is best for early checkpoints)",
    )
    p.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save raw predicted depth tensors as .pt files (for debugging)",
    )
    p.add_argument(
        "--depth-norm",
        default=None,
        choices=["auto", "none", "0_1", "-1_1"],
        help="Depth normalization mode used during training (default: from checkpoint)",
    )
    p.add_argument(
        "--depth-divisor",
        type=float,
        default=None,
        help="Depth divisor used during training (default: from checkpoint/auto)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)

    model, cfg = load_checkpoint(ckpt_path, device=device)
    timesteps = cfg.timesteps if args.timesteps is None else int(args.timesteps)
    depth_norm = cfg.depth_norm if args.depth_norm is None else str(args.depth_norm)
    depth_divisor = cfg.depth_divisor if args.depth_divisor is None else args.depth_divisor

    if args.data_root:
        data_root = Path(args.data_root)
        sample_iter = _iter_samples_from_chunks(data_root, limit=args.limit)
        for i, sample in enumerate(sample_iter):
            rgb = sample["rgb"].unsqueeze(0).to(device)
            pred = sample_depth(model, rgb=rgb, timesteps=timesteps, device=device)
            if args.save_raw:
                (out_dir / "raw").mkdir(parents=True, exist_ok=True)
                torch.save(pred[0].detach().cpu(), out_dir / "raw" / f"pred_{i:05d}.pt")
            depth_uint8 = depth_to_uint8_viz(
                pred[0], viz=args.viz, depth_norm=depth_norm, depth_divisor=depth_divisor
            )
            save_depth_png(depth_uint8, out_dir / f"pred_{i:05d}.png")
    else:
        if args.project_root is None:
            raise SystemExit("--project-root is required when using --csv-path")
        import pandas as pd

        project_root = Path(args.project_root).resolve()
        df = pd.read_csv(Path(args.csv_path), header=None)
        if df.shape[1] < 1:
            raise ValueError("CSV must have at least one column for rgb_path")

        n = min(int(args.limit), len(df))
        for i in range(n):
            rgb_rel = str(df.iloc[i, 0])
            rgb_path = (project_root / rgb_rel).resolve() if not Path(rgb_rel).is_absolute() else Path(rgb_rel)
            rgb = _load_rgb_from_path(rgb_path, size=int(args.size)).unsqueeze(0).to(device)
            pred = sample_depth(model, rgb=rgb, timesteps=timesteps, device=device)
            if args.save_raw:
                (out_dir / "raw").mkdir(parents=True, exist_ok=True)
                torch.save(pred[0].detach().cpu(), out_dir / "raw" / f"pred_{i:05d}.pt")
            depth_uint8 = depth_to_uint8_viz(
                pred[0], viz=args.viz, depth_norm=depth_norm, depth_divisor=depth_divisor
            )
            save_depth_png(depth_uint8, out_dir / f"pred_{i:05d}.png")

    print(f"Wrote predictions to: {out_dir}")


if __name__ == "__main__":
    main()
