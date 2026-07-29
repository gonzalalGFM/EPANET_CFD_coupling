#!/usr/bin/env python3
"""Phase 2b: Pressure inference -- predict p from U + geometry on regular grid.

Two-stage pipeline:
  Stage 1 (infer_hydraulics.py): mask + distance + case_param  --> U, nut
  Stage 2 (this):                mask + dist + cparam + U + y_pos  --> p

Inputs per Y-slice (7 channels):
  mask, distance, case_param, Ux, Uy, Uz, y_pos

Output per Y-slice (1 channel):
  p (pressure)
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
    """2D UNet for pressure prediction on Y-slices (7 -> 1 channels)."""

    def __init__(self, in_ch=7, out_ch=1, base=48, depth=4):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth)]

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
        for skip, dec in zip(reversed(skips), self.dec_convs):
            out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = dec(out)
        return self.out(out)


def load_pressure_model_and_stats(
    model_path: Path,
    data_path: Path,
    device: str,
) -> Tuple[UNet2D, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load trained pressure UNet and compute normalization statistics.

    Args:
        model_path: path to pressure_unet.pt
        data_path: path to processed_data_with_pressure.h5
        device: 'cuda' or 'cpu'

    Returns:
        (model, input_mean, input_std, output_mean, output_std)
        input stats shape: (7,), output stats shape: (1,)
    """
    model = UNet2D(in_ch=7, out_ch=1, base=48, depth=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Compute normalization stats from training data (same as train_pressure_unet.py)
    in_sum = np.zeros(7, dtype=np.float64)
    in_sq = np.zeros(7, dtype=np.float64)
    out_sum = np.zeros(1, dtype=np.float64)
    out_sq = np.zeros(1, dtype=np.float64)
    count = 0

    with h5py.File(data_path, "r") as h5:
        proc = h5["processed"]
        for case_name in proc:
            grp = proc[case_name]
            Ny = grp["mask"].shape[1]
            for j in range(Ny):
                m = np.nan_to_num(grp["mask"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                dist = np.nan_to_num(grp["distance"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                cparam = np.nan_to_num(grp["case_param"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                U = np.nan_to_num(grp["U"][:, j, :, :], nan=0.0, posinf=0.0, neginf=0.0)
                p = np.nan_to_num(grp["p"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)

                y_pos = np.full_like(m, j / max(Ny - 1, 1), dtype=np.float64)

                inp = np.stack(
                    [m, dist, cparam, U[..., 0], U[..., 1], U[..., 2], y_pos],
                    axis=0,
                )
                out = p[None, ...]

                n_pix = inp.shape[1] * inp.shape[2]
                in_sum += inp.reshape(7, -1).sum(axis=1)
                in_sq += (inp ** 2).reshape(7, -1).sum(axis=1)
                out_sum += out.reshape(1, -1).sum(axis=1)
                out_sq += (out ** 2).reshape(1, -1).sum(axis=1)
                count += n_pix

    eps = 1e-6
    input_mean = in_sum / count
    input_std = np.sqrt(in_sq / count - input_mean ** 2) + eps
    output_mean = out_sum / count
    output_std = np.sqrt(out_sq / count - output_mean ** 2) + eps

    return model, input_mean, input_std, output_mean, output_std


def infer_pressure(
    U_grid: np.ndarray,
    mask: np.ndarray,
    distance: np.ndarray,
    case_param: float,
    model: UNet2D,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
    device: str,
) -> np.ndarray:
    """Run pressure UNet inference for all Y-slices.

    Args:
        U_grid: (Nx, Ny, Nz, 3) predicted velocity field
        mask: (Nx, Ny, Nz) fluid mask
        distance: (Nx, Ny, Nz) distance to wall
        case_param: inlet velocity scalar
        model: trained pressure UNet
        input_mean, input_std: (7,) normalization for inputs
        output_mean, output_std: (1,) normalization for output
        device: 'cuda' or 'cpu'

    Returns:
        p_grid: (Nx, Ny, Nz) pressure field
    """
    Nx, Ny, Nz = mask.shape
    p_grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    case_param_img = np.full((Nx, Nz), case_param, dtype=np.float32)

    with torch.no_grad():
        for j in range(Ny):
            y_pos = np.full((Nx, Nz), j / max(Ny - 1, 1), dtype=np.float32)

            inp = np.stack([
                mask[:, j, :],
                distance[:, j, :],
                case_param_img,
                U_grid[:, j, :, 0],
                U_grid[:, j, :, 1],
                U_grid[:, j, :, 2],
                y_pos,
            ], axis=0)  # (7, Nx, Nz)

            # Normalize
            inp = (inp - input_mean[:, None, None]) / input_std[:, None, None]
            inp_t = torch.from_numpy(inp[None].astype(np.float32)).to(device)

            # Inference
            out_t = model(inp_t).cpu().numpy()[0]  # (1, Nx, Nz)

            # Denormalize
            out = out_t * output_std[:, None, None] + output_mean[:, None, None]
            p_grid[:, j, :] = out[0]

    return p_grid


def main():
    """Standalone test of pressure inference."""
    config_file = _repo_root() / "configs" / "config.json"
    if not config_file.exists():
        config_file = _repo_root() / "config.json"
    with open(config_file) as f:
        config = json.load(f)

    base_dir = _repo_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load velocity model and infer U
    from infer_hydraulics import load_model_and_stats as load_vel_model, infer_hydraulics

    vel_model_path = base_dir / config["ml"]["model_weights"]
    vel_data_path = base_dir / config["ml"]["processed_data_h5"]
    vel_model, vi_mean, vi_std, vo_mean, vo_std = load_vel_model(vel_model_path, vel_data_path, device)

    masks_path = base_dir / "precomputed_masks.npz"
    with np.load(masks_path) as data:
        mask = data["mask"]
        distance = data["distance"]

    test_case_param = 0.072
    print(f"[infer_pressure] Inferring velocity for case_param={test_case_param}...")
    U_grid, _ = infer_hydraulics(
        test_case_param, vel_model, mask, distance, vi_mean, vi_std, vo_mean, vo_std, device
    )

    # Load pressure model and infer p
    pressure_model_path = base_dir / config["ml"].get("pressure_model_weights", "pressure_unet.pt")
    pressure_data_path = base_dir / config["ml"].get("processed_data_pressure_h5", "processed_data_with_pressure.h5")

    print(f"[infer_pressure] Loading pressure model from {pressure_model_path}...")
    p_model, pi_mean, pi_std, po_mean, po_std = load_pressure_model_and_stats(
        pressure_model_path, pressure_data_path, device
    )

    print(f"[infer_pressure] Inferring pressure...")
    p_grid = infer_pressure(
        U_grid, mask, distance, test_case_param,
        p_model, pi_mean, pi_std, po_mean, po_std, device
    )

    print(f"  p_grid shape: {p_grid.shape}, range: [{p_grid.min():.6f}, {p_grid.max():.6f}]")

    output_file = base_dir / "pressure_inference_output.npz"
    np.savez(output_file, p_grid=p_grid, case_param=test_case_param)
    print(f"[infer_pressure] Saved pressure output to {output_file}")


if __name__ == "__main__":
    main()
