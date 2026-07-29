#!/usr/bin/env python3
"""Train a dedicated 2D UNet for pressure prediction on Y-slices.

Two-stage pipeline approach:
  Stage 1 (existing): mask + distance + case_param --> U, nut
  Stage 2 (this):     mask + distance + case_param + Ux + Uy + Uz + y_pos --> p

Inputs per slice (7 channels):
- mask (1)
- distance to wall (1)
- case_param constant image (1)
- U_x ground-truth velocity (1)
- U_y ground-truth velocity (1)
- U_z ground-truth velocity (1)
- y_position normalised slice index (1)

Output per slice (1 channel):
- p (pressure)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_data_with_pressure.h5"
EPOCHS = 1000
BATCH_SIZE = 8
LR = 1e-4
BASE_CHANNELS = 48
DEPTH = 4
VAL_FRACTION = 0.2
NUM_WORKERS = 4
PLOT_DIR = Path(__file__).parent / "training_plots_pressure_only"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_CASE = "Cube03"
BASE_CMAP = "coolwarm"

# Loss weights
MSE_WEIGHT = 1.0
GRAD_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PressureSliceDataset(Dataset):
    """Y-slice dataset with velocity as input for pressure prediction."""

    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self.entries: list[tuple[str, int, int]] = []
        self.input_mean: np.ndarray | None = None
        self.input_std: np.ndarray | None = None
        self.output_mean: np.ndarray | None = None
        self.output_std: np.ndarray | None = None

        with h5py.File(h5_path, "r") as h5:
            proc = h5["processed"]
            for case_name in proc:
                grp = proc[case_name]
                Ny = grp["mask"].shape[1]
                for j in range(Ny):
                    self.entries.append((case_name, j, Ny))
        self._compute_stats()

    def __len__(self):
        return len(self.entries)

    # ---- normalisation stats ------------------------------------------------
    def _compute_stats(self):
        in_sum = np.zeros(7, dtype=np.float64)
        in_sq = np.zeros(7, dtype=np.float64)
        out_sum = np.zeros(1, dtype=np.float64)
        out_sq = np.zeros(1, dtype=np.float64)
        count = 0

        with h5py.File(self.h5_path, "r") as h5:
            for case_name, j, Ny in self.entries:
                grp = h5["processed"][case_name]
                mask = np.nan_to_num(grp["mask"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                dist = np.nan_to_num(grp["distance"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                cparam = np.nan_to_num(grp["case_param"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
                U = np.nan_to_num(grp["U"][:, j, :, :], nan=0.0, posinf=0.0, neginf=0.0)
                p = np.nan_to_num(grp["p"][:, j, :], nan=0.0, posinf=0.0, neginf=0.0)

                y_pos = np.full_like(mask, j / max(Ny - 1, 1), dtype=np.float64)

                inp = np.stack(
                    [mask, dist, cparam, U[..., 0], U[..., 1], U[..., 2], y_pos],
                    axis=0,
                )  # (7, Nx, Nz)
                out = p[None, ...]  # (1, Nx, Nz)

                n_pix = inp.shape[1] * inp.shape[2]
                in_sum += inp.reshape(7, -1).sum(axis=1)
                in_sq += (inp ** 2).reshape(7, -1).sum(axis=1)
                out_sum += out.reshape(1, -1).sum(axis=1)
                out_sq += (out ** 2).reshape(1, -1).sum(axis=1)
                count += n_pix

        eps = 1e-6
        self.input_mean = in_sum / count
        self.input_std = np.sqrt(in_sq / count - self.input_mean ** 2) + eps
        self.output_mean = out_sum / count
        self.output_std = np.sqrt(out_sq / count - self.output_mean ** 2) + eps

    # ---- getitem ------------------------------------------------------------
    def __getitem__(self, idx):
        case_name, j, Ny = self.entries[idx]
        with h5py.File(self.h5_path, "r") as h5:
            grp = h5["processed"][case_name]
            mask = grp["mask"][:, j, :]
            dist = grp["distance"][:, j, :]
            cparam = grp["case_param"][:, j, :]
            U = grp["U"][:, j, :, :]  # (Nx, Nz, 3)
            p = grp["p"][:, j, :]  # (Nx, Nz)

        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
        cparam = np.nan_to_num(cparam, nan=0.0, posinf=0.0, neginf=0.0)
        U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        y_pos = np.full_like(mask, j / max(Ny - 1, 1), dtype=np.float64)

        inputs = np.stack(
            [mask, dist, cparam, U[..., 0], U[..., 1], U[..., 2], y_pos],
            axis=0,
        )
        outputs = p[None, ...]

        inputs = (inputs - self.input_mean[:, None, None]) / self.input_std[:, None, None]
        outputs = (outputs - self.output_mean[:, None, None]) / self.output_std[:, None, None]

        return (
            torch.from_numpy(inputs.astype(np.float32)),
            torch.from_numpy(outputs.astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Model (same UNet architecture, different channel counts)
# ---------------------------------------------------------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    def __init__(self, in_ch=7, out_ch=1, base=48, depth=4):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth)]

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(conv_block(in_ch if i == 0 else chans[i - 1], chans[i]))
        self.pool = nn.MaxPool2d(2)

        self.dec_convs = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.dec_convs.append(conv_block(chans[i] + chans[i + 1], chans[i]))
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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def spatial_gradient_loss(pred, target):
    """MSE on finite-difference spatial gradients dp/dx and dp/dz."""
    pred_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
    pred_dz = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_dz = target[:, :, :, 1:] - target[:, :, :, :-1]
    return F.mse_loss(pred_dx, tgt_dx) + F.mse_loss(pred_dz, tgt_dz)


def combined_loss(preds, targets, mse_w, grad_w):
    mse = F.mse_loss(preds, targets)
    grad = spatial_gradient_loss(preds, targets)
    return mse_w * mse + grad_w * grad, mse.item(), grad.item()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def make_loaders(h5_path: Path, batch_size: int, val_fraction: float, num_workers: int):
    ds = PressureSliceDataset(h5_path)
    val_size = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, len(ds), ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    h5_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    base_channels: int,
    depth: int,
    val_fraction: float,
    num_workers: int,
    mse_weight: float,
    grad_weight: float,
):
    train_loader, val_loader, total_len, ds = make_loaders(h5_path, batch_size, val_fraction, num_workers)
    model = UNet2D(in_ch=7, out_ch=1, base=base_channels, depth=depth).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)

    history = {
        "train": [], "val": [],
        "train_mse": [], "train_grad": [],
        "val_mse": [], "val_grad": [],
    }

    for epoch in range(1, epochs + 1):
        # ---- train ----------------------------------------------------------
        model.train()
        t_loss, t_mse, t_grad, t_n = 0.0, 0.0, 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            loss, mse_val, grad_val = combined_loss(preds, targets, mse_weight, grad_weight)
            optim.zero_grad()
            loss.backward()
            optim.step()
            bs = inputs.size(0)
            t_loss += loss.item() * bs
            t_mse += mse_val * bs
            t_grad += grad_val * bs
            t_n += bs
        scheduler.step()
        history["train"].append(t_loss / t_n)
        history["train_mse"].append(t_mse / t_n)
        history["train_grad"].append(t_grad / t_n)

        # ---- val ------------------------------------------------------------
        model.eval()
        v_loss, v_mse, v_grad, v_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = model(inputs)
                loss, mse_val, grad_val = combined_loss(preds, targets, mse_weight, grad_weight)
                bs = inputs.size(0)
                v_loss += loss.item() * bs
                v_mse += mse_val * bs
                v_grad += grad_val * bs
                v_n += bs
        history["val"].append(v_loss / v_n)
        history["val_mse"].append(v_mse / v_n)
        history["val_grad"].append(v_grad / v_n)

        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{epochs} | train {history['train'][-1]:.6f} "
            f"(mse {history['train_mse'][-1]:.6f} grad {history['train_grad'][-1]:.6f}) | "
            f"val {history['val'][-1]:.6f} | lr {cur_lr:.2e}"
        )

    # ---- save model ---------------------------------------------------------
    save_path = Path(__file__).resolve().parents[1] / "data" / "pressure_unet.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved model to", save_path)

    # ---- plots --------------------------------------------------------------
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    _plot_loss_history(history)
    _plot_scatter(model, train_loader, val_loader, ds, device)
    _plot_spatial(model, val_loader, ds, device)
    return model, ds


# ---------------------------------------------------------------------------
# Post-training plots
# ---------------------------------------------------------------------------
def _plot_loss_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        [("train", "val"), ("train_mse", "val_mse"), ("train_grad", "val_grad")],
        ["Total loss", "MSE component", "Gradient component"],
    ):
        ax.plot(history[key[0]], label="train")
        ax.plot(history[key[1]], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    path = PLOT_DIR / "loss_history.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


def _write_scatter(tgt, pred, fname, label, folder=PLOT_DIR):
    max_points = 50000
    tgt = tgt.reshape(-1)
    pred = pred.reshape(-1)
    if tgt.numel() > max_points:
        idx = torch.randperm(tgt.numel())[:max_points]
        tgt = tgt[idx]
        pred = pred[idx]
    plt.figure(figsize=(5, 5))
    plt.scatter(tgt.numpy(), pred.numpy(), s=2, alpha=0.25)
    lims = [min(tgt.min().item(), pred.min().item()), max(tgt.max().item(), pred.max().item())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.plot(lims, [0.9 * lv for lv in lims], linestyle=":", color="gray", linewidth=1)
    plt.plot(lims, [1.1 * lv for lv in lims], linestyle=":", color="gray", linewidth=1)
    plt.xlabel(f"target {label}")
    plt.ylabel(f"pred {label}")
    plt.tight_layout()
    out_path = folder / fname
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved scatter plot to {out_path}")


def _plot_scatter(model, train_loader, val_loader, ds, device):
    out_mean = torch.tensor(ds.output_mean)
    out_std = torch.tensor(ds.output_std)

    def denorm(t):
        return t * out_std[None, :, None, None] + out_mean[None, :, None, None]

    with torch.no_grad():
        for prefix, loader in [("train", train_loader), ("val", val_loader)]:
            inp, tgt = next(iter(loader))
            pred = model(inp.to(device)).cpu()
            pred_d = denorm(pred)
            tgt_d = denorm(tgt)
            _write_scatter(tgt_d[:, 0], pred_d[:, 0], f"{prefix}_scatter_p.png", "p")


def _plot_spatial(model, val_loader, ds, device):
    with torch.no_grad():
        sample_inp, sample_tgt = val_loader.dataset[0]
        sample_pred = model(sample_inp.unsqueeze(0).to(device)).cpu()[0]

    out_mean = torch.tensor(ds.output_mean).view(-1, 1, 1)
    out_std = torch.tensor(ds.output_std).view(-1, 1, 1)
    tgt_d = sample_tgt * out_std + out_mean
    pred_d = sample_pred * out_std + out_mean

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(tgt_d[0], cmap=BASE_CMAP)
    axes[0].set_title("target p")
    axes[1].imshow(pred_d[0], cmap=BASE_CMAP)
    axes[1].set_title("pred p")
    axes[2].imshow((pred_d[0] - tgt_d[0]), cmap="bwr")
    axes[2].set_title("diff p")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path = PLOT_DIR / "spatial_val.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


# ---------------------------------------------------------------------------
# Full-case evaluation
# ---------------------------------------------------------------------------
def evaluate_case(h5_path: Path, case_name: str, model: nn.Module, ds: PressureSliceDataset, device: str):
    out_dir = PLOT_DIR / f"val_{case_name.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        grp = h5["processed"][case_name]
        mask = grp["mask"][:]
        dist = grp["distance"][:]
        cparam = grp["case_param"][:]
        U = grp["U"][:]
        p = grp["p"][:]

    Nx, Ny, Nz = mask.shape
    pred_p = np.zeros_like(p)

    in_mean = ds.input_mean[:, None, None]
    in_std = ds.input_std[:, None, None]
    out_mean = ds.output_mean[:, None, None]
    out_std = ds.output_std[:, None, None]

    model.eval()
    with torch.no_grad():
        for j in range(Ny):
            m = np.nan_to_num(mask[:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
            d = np.nan_to_num(dist[:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
            cp = np.nan_to_num(cparam[:, j, :], nan=0.0, posinf=0.0, neginf=0.0)
            u = np.nan_to_num(U[:, j, :, :], nan=0.0, posinf=0.0, neginf=0.0)
            y_pos = np.full_like(m, j / max(Ny - 1, 1), dtype=np.float64)

            inp = np.stack([m, d, cp, u[..., 0], u[..., 1], u[..., 2], y_pos], axis=0)
            inp = (inp - in_mean) / in_std
            out = model(torch.from_numpy(inp[None].astype(np.float32)).to(device)).cpu().numpy()[0]
            out = out * out_std + out_mean
            pred_p[:, j, :] = out[0]

    diff_p = pred_p - p

    # ---- helper: write X-slice summary -------------------------------------
    def write_summary(data_3d, fname, cmap, label, overlay=None):
        Nx_ = data_3d.shape[0]
        ncols = 8
        nrows = math.ceil(Nx_ / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = axes.ravel()
        last_im = None
        for i in range(Nx_):
            ax = axes[i]
            im = ax.imshow(data_3d[i, :, :].T, origin="lower", cmap=cmap, interpolation="nearest")
            last_im = im
            if overlay is not None:
                ax.contour(overlay[i, :, :].T, levels=[0.5], colors="red", linewidths=0.4)
            ax.set_title(f"X={i}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        for k in range(Nx_, len(axes)):
            axes[k].axis("off")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.tolist(), shrink=0.6, label=label)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    def write_pair(data_pred, data_gt, fname, cmap, lbl_pred, lbl_gt, overlay=None):
        Nx_ = data_pred.shape[0]
        panels = Nx_ * 2
        ncols = 8
        nrows = math.ceil(panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = axes.ravel()
        last_im = None
        idx = 0
        for i in range(Nx_):
            ax = axes[idx]
            im = ax.imshow(data_pred[i, :, :].T, origin="lower", cmap=cmap, interpolation="nearest")
            last_im = im
            if overlay is not None:
                ax.contour(overlay[i, :, :].T, levels=[0.5], colors="red", linewidths=0.4)
            ax.set_title(f"{lbl_pred} X={i}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
            if idx < len(axes):
                ax = axes[idx]
                ax.imshow(data_gt[i, :, :].T, origin="lower", cmap=cmap, interpolation="nearest")
                if overlay is not None:
                    ax.contour(overlay[i, :, :].T, levels=[0.5], colors="red", linewidths=0.4)
                ax.set_title(f"{lbl_gt} X={i}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                idx += 1
        for k in range(idx, len(axes)):
            axes[k].axis("off")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.tolist(), shrink=0.6, label=f"{lbl_pred}/{lbl_gt}")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    write_summary(p, f"{case_name}_gt_p.png", BASE_CMAP, "p gt", overlay=mask)
    write_summary(pred_p, f"{case_name}_pred_p.png", BASE_CMAP, "p pred", overlay=mask)
    write_summary(diff_p, f"{case_name}_diff_p.png", "bwr", "p pred-gt", overlay=mask)
    write_pair(pred_p, p, f"{case_name}_pair_p.png", BASE_CMAP, "pred p", "gt p", overlay=mask)

    # ---- gradient comparison ------------------------------------------------
    gt_dx = np.diff(p, axis=0)
    pred_dx = np.diff(pred_p, axis=0)
    gt_dz = np.diff(p, axis=2)
    pred_dz = np.diff(pred_p, axis=2)
    write_pair(pred_dx, gt_dx, f"{case_name}_grad_dx_pair.png", "bwr", "pred dp/dx", "gt dp/dx", overlay=mask[:-1])
    write_pair(pred_dz, gt_dz, f"{case_name}_grad_dz_pair.png", "bwr", "pred dp/dz", "gt dp/dz", overlay=mask[:, :, :-1])

    # ---- metrics ------------------------------------------------------------
    valid = ~np.isnan(p) & ~np.isnan(pred_p)
    mse = np.mean((pred_p[valid] - p[valid]) ** 2)
    mae = np.mean(np.abs(pred_p[valid] - p[valid]))
    p_range = np.nanmax(p) - np.nanmin(p)
    nrmse = np.sqrt(mse) / p_range if p_range > 0 else float("inf")
    print(f"[{case_name}] MSE={mse:.6e}  MAE={mae:.6e}  NRMSE={nrmse:.4f}")
    print(f"Saved case eval plots to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train dedicated pressure UNet on Y-slices.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--base-ch", type=int, default=BASE_CHANNELS)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--val-frac", type=float, default=VAL_FRACTION)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--eval-case", type=str, default=EVAL_CASE)
    parser.add_argument("--mse-weight", type=float, default=MSE_WEIGHT)
    parser.add_argument("--grad-weight", type=float, default=GRAD_WEIGHT)
    args = parser.parse_args()

    model, ds = train(
        args.data,
        args.epochs,
        args.batch_size,
        args.lr,
        args.device,
        args.base_ch,
        args.depth,
        args.val_frac,
        args.num_workers,
        args.mse_weight,
        args.grad_weight,
    )
    if args.eval_case:
        evaluate_case(args.data, args.eval_case, model, ds, args.device)


if __name__ == "__main__":
    main()
