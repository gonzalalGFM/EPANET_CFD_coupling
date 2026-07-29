#!/usr/bin/env python3
"""Phase 2: ML inference -- given case_param, predict U and nut on regular grid.

Given:
  case_param: inlet velocity (scalar)

Returns:
  U_grid [80, 50, 50, 3]: velocity field
  nut_grid [80, 50, 50]: turbulent viscosity field
"""

import json
from pathlib import Path


def _repo_root() -> Path:
    """Repository root (parent of coupling/)."""
    return Path(__file__).resolve().parents[1]

from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet2D(nn.Module):
    """2D UNet for Y-slices."""

    def __init__(self, in_ch=3, out_ch=4, base=32, depth=3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth)]

        # Encoder
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch if i == 0 else chans[i - 1], chans[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(chans[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(chans[i], chans[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(chans[i]),
                    nn.ReLU(inplace=True),
                )
            )
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.dec_convs = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.dec_convs.append(
                nn.Sequential(
                    nn.Conv2d(chans[i] + chans[i + 1], chans[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(chans[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(chans[i], chans[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(chans[i]),
                    nn.ReLU(inplace=True),
                )
            )
        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x
        for i, down in enumerate(self.downs):
            out = down(out)
            if i != len(self.downs) - 1:
                skips.append(out)
                out = self.pool(out)

        # Decoder
        for skip, dec in zip(reversed(skips), self.dec_convs):
            out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = dec(out)

        return self.out(out)


def load_model_and_stats(
    model_path: Path, data_path: Path, device: str
) -> Tuple[UNet2D, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load trained UNet and normalization statistics."""
    model = UNet2D(in_ch=3, out_ch=4, base=48, depth=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with h5py.File(data_path, "r") as h5:
        # These are stored as float64 arrays in processed_data.h5
        input_mean = np.array(h5["processed"]["Cube03"]["mask"][:].flatten())[:0]  # placeholder
        input_std = np.array(h5["processed"]["Cube03"]["mask"][:].flatten())[:0]   # placeholder

    # Compute stats from a training case (Cube03)
    with h5py.File(data_path, "r") as h5:
        proc = h5["processed"]
        in_sum = np.zeros(3, dtype=np.float64)
        in_sq = np.zeros(3, dtype=np.float64)
        out_sum = np.zeros(4, dtype=np.float64)
        out_sq = np.zeros(4, dtype=np.float64)
        count = 0

        for case_name in proc:
            grp = proc[case_name]
            Ny = grp["mask"].shape[1]
            for j in range(Ny):
                mask = np.nan_to_num(grp["mask"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                dist = np.nan_to_num(grp["distance"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                case_param = np.nan_to_num(grp["case_param"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                U = np.nan_to_num(grp["U"][:, j, :, :], nan=0.0, posinf=0.0, neginf=0.0)
                nut = np.nan_to_num(grp["nut"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                inp = np.stack([mask, dist, case_param], axis=0)
                out = np.concatenate([np.moveaxis(U, -1, 0), nut[None, ...]], axis=0)
                in_sum += inp.reshape(3, -1).sum(axis=1)
                in_sq += (inp ** 2).reshape(3, -1).sum(axis=1)
                out_sum += out.reshape(4, -1).sum(axis=1)
                out_sq += (out ** 2).reshape(4, -1).sum(axis=1)
                count += inp.shape[1] * inp.shape[2]

    eps = 1e-6
    input_mean = in_sum / count
    input_std = np.sqrt(in_sq / count - input_mean ** 2) + eps
    output_mean = out_sum / count
    output_std = np.sqrt(out_sq / count - output_mean ** 2) + eps

    return model, input_mean, input_std, output_mean, output_std


def infer_hydraulics(
    case_param: float,
    model: UNet2D,
    mask: np.ndarray,
    distance: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run UNet inference for all Y-slices.

    Parameters:
      case_param: inlet velocity (scalar)
      mask, distance: [80, 50, 50] arrays

    Returns:
      U_grid [80, 50, 50, 3]
      nut_grid [80, 50, 50]
    """
    Nx, Ny, Nz = mask.shape
    U_grid = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    nut_grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    case_param_img = np.full((Nx, Nz), case_param, dtype=np.float32)

    with torch.no_grad():
        for j in range(Ny):
            # Build input for this Y-slice
            inp = np.stack([mask[:, j, :], distance[:, j, :], case_param_img], axis=0)
            inp = (inp - input_mean[:, None, None]) / input_std[:, None, None]
            inp_t = torch.from_numpy(inp[None].astype(np.float32)).to(device)

            # Inference
            out_t = model(inp_t).cpu().numpy()[0]  # [4, Nx, Nz]

            # Denormalize
            out = out_t * output_std[:, None, None] + output_mean[:, None, None]

            # Unpack into U and nut
            U_grid[:, j, :, :] = np.moveaxis(out[:3], 0, -1)  # [Ux, Uy, Uz] -> [Nx, Nz, 3]
            nut_grid[:, j, :] = out[3]

    return U_grid, nut_grid


def main():
    """Standalone test of inference."""
    config_file = _repo_root() / "configs" / "config.json"
    if not config_file.exists():
        config_file = _repo_root() / "config.json"
    with open(config_file) as f:
        config = json.load(f)

    base_dir = _repo_root()
    model_path = base_dir / config["ml"]["model_weights"]
    data_path = base_dir / config["ml"]["processed_data_h5"]
    masks_path = base_dir / "precomputed_masks.npz"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[infer_hydraulics] Using device: {device}")
    print(f"[infer_hydraulics] Loading model from {model_path}...")

    model, input_mean, input_std, output_mean, output_std = load_model_and_stats(model_path, data_path, device)

    # Load mask and distance
    print(f"[infer_hydraulics] Loading mask and distance from {masks_path}...")
    with np.load(masks_path) as data:
        mask = data["mask"]
        distance = data["distance"]

    # Test inference with a known case_param
    test_case_param = 0.072  # Cube03

    print(f"[infer_hydraulics] Inferring for case_param={test_case_param}...")
    U_grid, nut_grid = infer_hydraulics(
        test_case_param, model, mask, distance, input_mean, input_std, output_mean, output_std, device
    )

    print(f"  U_grid shape: {U_grid.shape}, range: [{U_grid.min():.6f}, {U_grid.max():.6f}]")
    print(f"  nut_grid shape: {nut_grid.shape}, range: [{nut_grid.min():.6f}, {nut_grid.max():.6f}]")

    # Save output for Phase 3
    output_file = base_dir / "inference_output.npz"
    np.savez(output_file, U_grid=U_grid, nut_grid=nut_grid, case_param=test_case_param)
    print(f"[infer_hydraulics] Saved inference output to {output_file}")


if __name__ == "__main__":
    main()
