#!/usr/bin/env python3
"""Train a 2D UNet on Y-slices from processed_data.h5.

Inputs per slice (channels):
- mask (1)
- distance to wall (1)
- case_param constant image (1)

Outputs per slice (channels):
- U vector components (3)
- nut (1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import math


# Hyperparameters (modify here)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_data.h5"
EPOCHS = 500
BATCH_SIZE = 8
LR = 1e-5
USE_COSINE_LR = True
COSINE_LR_MAX = 1e-4
COSINE_LR_MIN = 1e-6
BASE_CHANNELS = 48  # paper deployed surrogate
DEPTH = 4  # paper deployed surrogate
VAL_FRACTION = 0.2
NUM_WORKERS = 4
PLOT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "training_plots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_CASE = "Cube03"
BASE_CMAP = "coolwarm"  # ParaView-style Cool to Warm

# Loss weights (modify here)
MSE_WEIGHT = 1
SSIM_WEIGHT = 0
U_LOSS_WEIGHT = 1#0.73
NUT_LOSS_WEIGHT = 1#0.93


class SliceDataset(Dataset):
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self.entries = []
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        with h5py.File(h5_path, "r") as h5:
            proc = h5["processed"]
            for case_name in proc:
                grp = proc[case_name]
                Ny = grp["mask"].shape[1]
                for j in range(Ny):
                    self.entries.append((case_name, j))
        self._compute_stats()

    def __len__(self):
        return len(self.entries)

    def _compute_stats(self):
        """Compute mean/std over all slices for inputs and outputs."""
        in_sum = np.zeros(3, dtype=np.float64)
        in_sq = np.zeros(3, dtype=np.float64)
        out_sum = np.zeros(4, dtype=np.float64)
        out_sq = np.zeros(4, dtype=np.float64)
        count = 0
        with h5py.File(self.h5_path, "r") as h5:
            for case_name, j in self.entries:
                grp = h5["processed"][case_name]
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
        self.input_mean = in_sum / count
        self.input_std = np.sqrt(in_sq / count - self.input_mean ** 2) + eps
        self.output_mean = out_sum / count
        self.output_std = np.sqrt(out_sq / count - self.output_mean ** 2) + eps

    def __getitem__(self, idx):
        case_name, j = self.entries[idx]
        with h5py.File(self.h5_path, "r") as h5:
            grp = h5["processed"][case_name]
            mask = grp["mask"][:, j, :]
            dist = grp["distance"][:, j, :]
            case_param = grp["case_param"][:, j, :]
            U = grp["U"][:, j, :, :]  # (Nx, Nz, 3)
            nut = grp["nut"][:, j, :]

        # Replace NaNs/Infs to avoid NaN loss
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
        case_param = np.nan_to_num(case_param, nan=0.0, posinf=0.0, neginf=0.0)
        U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        nut = np.nan_to_num(nut, nan=0.0, posinf=0.0, neginf=0.0)

        inputs = np.stack([mask, dist, case_param], axis=0)
        outputs = np.concatenate([np.moveaxis(U, -1, 0), nut[None, ...]], axis=0)

        # Normalize
        inputs = (inputs - self.input_mean[:, None, None]) / self.input_std[:, None, None]
        outputs = (outputs - self.output_mean[:, None, None]) / self.output_std[:, None, None]

        return torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(outputs.astype(np.float32))


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
    def __init__(self, in_ch=3, out_ch=4, base=32, depth=3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth)]

        # Encoder
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(conv_block(in_ch if i == 0 else chans[i - 1], chans[i]))
        self.pool = nn.MaxPool2d(2)

        # Decoder
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

        # Decoder
        for skip, dec in zip(reversed(skips), self.dec_convs):
            out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = dec(out)

        return self.out(out)


def make_loaders(h5_path: Path, batch_size: int, val_fraction: float, num_workers: int):
    ds = SliceDataset(h5_path)
    val_size = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, len(ds), ds


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
    ssim_weight: float,
    u_loss_weight: float,
    nut_loss_weight: float,
    use_cosine_lr: bool,
):
    train_loader, val_loader, total_len, ds = make_loaders(h5_path, batch_size, val_fraction, num_workers)
    model = UNet2D(in_ch=3, out_ch=4, base=base_channels, depth=depth).to(device)
    optim_lr = COSINE_LR_MAX if use_cosine_lr else lr
    optim = torch.optim.Adam(model.parameters(), lr=optim_lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=COSINE_LR_MIN)
        if use_cosine_lr
        else None
    )

    def ssim_loss(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
        # compute per-channel SSIM over whole slice (no window)
        mu_x = x.mean(dim=[2, 3])
        mu_y = y.mean(dim=[2, 3])
        sigma_x = ((x - mu_x[:, :, None, None]) ** 2).mean(dim=[2, 3])
        sigma_y = ((y - mu_y[:, :, None, None]) ** 2).mean(dim=[2, 3])
        sigma_xy = ((x - mu_x[:, :, None, None]) * (y - mu_y[:, :, None, None])).mean(dim=[2, 3])
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-8)
        return 1.0 - ssim.mean()

    def combined_loss(preds, targets):
        mse_u = F.mse_loss(preds[:, :3], targets[:, :3])
        mse_nut = F.mse_loss(preds[:, 3:], targets[:, 3:])
        ssim_u = ssim_loss(preds[:, :3], targets[:, :3])
        ssim_nut = ssim_loss(preds[:, 3:], targets[:, 3:])
        mse_term = u_loss_weight * mse_u + nut_loss_weight * mse_nut
        ssim_term = u_loss_weight * ssim_u + nut_loss_weight * ssim_nut
        return mse_weight * mse_term + ssim_weight * ssim_term

    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            loss = combined_loss(preds, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * inputs.size(0)
        train_avg = total_loss / len(train_loader.dataset)
        history["train"].append(train_avg)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = model(inputs)
                loss = combined_loss(preds, targets)
                val_loss += loss.item() * inputs.size(0)
        val_avg = val_loss / len(val_loader.dataset)
        history["val"].append(val_avg)

        if scheduler is not None:
            scheduler.step()
        current_lr = optim.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs} | train {train_avg:.5f} | val {val_avg:.5f} | lr {current_lr:.2e}")

    out_pt = Path(__file__).resolve().parents[1] / "data" / "unet2d_slices.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_pt)
    print("Saved model to", out_pt)

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    plot_path = PLOT_DIR / "loss_history.png"
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("L1 loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print("Saved loss history to", plot_path)

    # Scatter plots on a validation and training batch (first batch), denormalized
    with torch.no_grad():
        val_inputs, val_targets = next(iter(val_loader))
        val_preds = model(val_inputs.to(device)).cpu()
        train_inputs, train_targets = next(iter(train_loader))
        train_preds = model(train_inputs.to(device)).cpu()

    def write_scatter(tgt, pred, fname, label, folder=PLOT_DIR):
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
        plt.plot(lims, [0.9 * l for l in lims], linestyle=":", color="gray", linewidth=1)
        plt.plot(lims, [1.1 * l for l in lims], linestyle=":", color="gray", linewidth=1)
        plt.xlabel(f"target {label}")
        plt.ylabel(f"pred {label}")
        plt.tight_layout()
        out_path = folder / fname
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved scatter plot to {out_path}")

    out_mean_full = torch.tensor(ds.output_mean)
    out_std_full = torch.tensor(ds.output_std)

    def denorm_batch(preds, targets):
        preds_d = preds * out_std_full[None, :, None, None] + out_mean_full[None, :, None, None]
        tgt_d = targets * out_std_full[None, :, None, None] + out_mean_full[None, :, None, None]
        return preds_d, tgt_d

    val_denorm, val_tgt_denorm = denorm_batch(val_preds, val_targets)
    train_denorm, train_tgt_denorm = denorm_batch(train_preds, train_targets)

    def scatter_set(prefix, tgt_d, pred_d):
        write_scatter(tgt_d[:, 3], pred_d[:, 3], f"{prefix}_scatter_nut.png", "nut")
        write_scatter(tgt_d[:, 0], pred_d[:, 0], f"{prefix}_scatter_Ux.png", "Ux")
        write_scatter(tgt_d[:, 1], pred_d[:, 1], f"{prefix}_scatter_Uy.png", "Uy")
        write_scatter(tgt_d[:, 2], pred_d[:, 2], f"{prefix}_scatter_Uz.png", "Uz")
        tgt_mag = torch.linalg.norm(tgt_d[:, :3], dim=1)
        pred_mag = torch.linalg.norm(pred_d[:, :3], dim=1)
        write_scatter(tgt_mag, pred_mag, f"{prefix}_scatter_Umag.png", "|U|")

    scatter_set("val", val_tgt_denorm, val_denorm)
    scatter_set("train", train_tgt_denorm, train_denorm)

    # Spatial plots for first item of val set
    spatial_path = PLOT_DIR / "spatial_val.png"
    with torch.no_grad():
        sample_inputs, sample_targets = val_loader.dataset[0]
        sample_inputs = sample_inputs.unsqueeze(0).to(device)
        sample_preds = model(sample_inputs).cpu()[0]
        sample_targets = sample_targets

    # denormalize full outputs
    out_mean = torch.tensor(ds.output_mean).view(-1, 1, 1)
    out_std = torch.tensor(ds.output_std).view(-1, 1, 1)
    tgt_denorm = sample_targets * out_std + out_mean
    pred_denorm = sample_preds * out_std + out_mean

    channel_names = ["U_x", "U_y", "U_z", "nut"]
    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for i, name in enumerate(channel_names):
        axes[i, 0].imshow(tgt_denorm[i], cmap=BASE_CMAP)
        axes[i, 0].set_title(f"target {name}")
        axes[i, 1].imshow(pred_denorm[i], cmap=BASE_CMAP)
        axes[i, 1].set_title(f"pred {name}")
        axes[i, 2].imshow((pred_denorm[i] - tgt_denorm[i]), cmap="bwr")
        axes[i, 2].set_title(f"diff {name}")
        for k in range(3):
            axes[i, k].axis("off")
    fig.tight_layout()
    plt.savefig(spatial_path, dpi=150)
    plt.close()
    print("Saved spatial plot to", spatial_path)

    return model, ds


def evaluate_case(h5_path: Path, case_name: str, model: nn.Module, ds: SliceDataset, device: str):
    """Generate GT vs pred spatial plots for a full case using X-slices (height)."""
    case_name_norm = case_name
    out_dir = PLOT_DIR / f"val_{case_name_norm.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        grp = h5["processed"][case_name_norm]
        mask = grp["mask"][:]  # (Nx, Ny, Nz)
        dist = grp["distance"][:]
        case_param = grp["case_param"][:]
        U = grp["U"][:]  # (Nx, Ny, Nz, 3)
        nut = grp["nut"][:]

    Nx, Ny, Nz, _ = U.shape
    pred_U = np.zeros_like(U)
    pred_nut = np.zeros_like(nut)

    in_mean = ds.input_mean[:, None, None]
    in_std = ds.input_std[:, None, None]
    out_mean = ds.output_mean[:, None, None]
    out_std = ds.output_std[:, None, None]

    model.eval()
    with torch.no_grad():
        for j in range(Ny):
            inp = np.stack([mask[:, j, :], dist[:, j, :], case_param[:, j, :]], axis=0)
            inp = (inp - in_mean) / in_std
            out = model(torch.from_numpy(inp[None].astype(np.float32)).to(device)).cpu().numpy()[0]
            out = out * out_std + out_mean
            pred_U[:, j, :, :] = np.moveaxis(out[:3], 0, -1)
            pred_nut[:, j, :] = out[3]

    def write_summary_png_x(data_3d: np.ndarray, fname: str, cmap: str, label: str, overlay_mask: np.ndarray | None = None):
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
            if overlay_mask is not None:
                ax.contour(overlay_mask[i, :, :].T, levels=[0.5], colors="red", linewidths=0.4)
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

    def write_pair_png_x(
        data_pred: np.ndarray,
        data_gt: np.ndarray,
        fname: str,
        cmap: str,
        label_pred: str,
        label_gt: str,
        overlay_mask: np.ndarray | None = None,
    ):
        Nx_ = data_pred.shape[0]
        panels = Nx_ * 2
        ncols = 8
        nrows = math.ceil(panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = axes.ravel()
        last_im = None
        idx = 0
        for i in range(Nx_):
            # pred
            ax = axes[idx]
            im = ax.imshow(data_pred[i, :, :].T, origin="lower", cmap=cmap, interpolation="nearest")
            last_im = im
            if overlay_mask is not None:
                ax.contour(overlay_mask[i, :, :].T, levels=[0.5], colors="red", linewidths=0.4)
            ax.set_title(f"{label_pred} X={i}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
            # gt
            if idx < len(axes):
                ax = axes[idx]
                ax.imshow(data_gt[i, :, :].T, origin="lower", cmap=cmap, interpolation="nearest")
                if overlay_mask is not None:
                    ax.contour(overlay_mask[i, :, :].T, levels=[0.5], colors="red", linewidths=0.4)
                ax.set_title(f"{label_gt} X={i}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                idx += 1
        for k in range(idx, len(axes)):
            axes[k].axis("off")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.tolist(), shrink=0.6, label=f"{label_pred}/{label_gt}")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # Magnitude and nut comparisons
    gt_mag = np.linalg.norm(U, axis=-1)
    pred_mag = np.linalg.norm(pred_U, axis=-1)
    diff_mag = pred_mag - gt_mag
    diff_nut = pred_nut - nut

    write_summary_png_x(gt_mag, f"{case_name_norm}_gt_Umag.png", BASE_CMAP, "|U| gt", overlay_mask=mask)
    write_summary_png_x(pred_mag, f"{case_name_norm}_pred_Umag.png", BASE_CMAP, "|U| pred", overlay_mask=mask)
    write_summary_png_x(diff_mag, f"{case_name_norm}_diff_Umag.png", "bwr", "|U| pred-gt", overlay_mask=mask)

    write_summary_png_x(nut, f"{case_name_norm}_gt_nut.png", BASE_CMAP, "nut gt", overlay_mask=mask)
    write_summary_png_x(pred_nut, f"{case_name_norm}_pred_nut.png", BASE_CMAP, "nut pred", overlay_mask=mask)
    write_summary_png_x(diff_nut, f"{case_name_norm}_diff_nut.png", "bwr", "nut pred-gt", overlay_mask=mask)
    write_pair_png_x(pred_mag, gt_mag, f"{case_name_norm}_pair_Umag.png", "viridis", "pred |U|", "gt |U|", overlay_mask=mask)
    write_pair_png_x(pred_nut, nut, f"{case_name_norm}_pair_nut.png", BASE_CMAP, "pred nut", "gt nut", overlay_mask=mask)

    # Component-wise comparisons (U_x, U_y, U_z)
    comp_names = ["Ux", "Uy", "Uz"]
    for ci, name in enumerate(comp_names):
        write_summary_png_x(U[..., ci], f"{case_name_norm}_gt_{name}.png", BASE_CMAP, f"{name} gt", overlay_mask=mask)
        write_summary_png_x(pred_U[..., ci], f"{case_name_norm}_pred_{name}.png", BASE_CMAP, f"{name} pred", overlay_mask=mask)
        write_summary_png_x(pred_U[..., ci] - U[..., ci], f"{case_name_norm}_diff_{name}.png", "bwr", f"{name} pred-gt", overlay_mask=mask)
        write_pair_png_x(pred_U[..., ci], U[..., ci], f"{case_name_norm}_pair_{name}.png", BASE_CMAP, f"pred {name}", f"gt {name}", overlay_mask=mask)

    print(f"Saved case eval plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train UNet on processed Y-slices.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--base-ch", type=int, default=BASE_CHANNELS, help="Base number of filters")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Number of UNet levels")
    parser.add_argument("--val-frac", type=float, default=VAL_FRACTION, help="Validation fraction")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--eval-case", type=str, default=EVAL_CASE, help="Case name to render full-volume eval plots")
    parser.add_argument("--mse-weight", type=float, default=MSE_WEIGHT, help="Weight for MSE term in loss")
    parser.add_argument("--ssim-weight", type=float, default=SSIM_WEIGHT, help="Weight for SSIM term in loss")
    parser.add_argument("--u-loss-weight", type=float, default=U_LOSS_WEIGHT, help="Weight for U channels in loss")
    parser.add_argument("--nut-loss-weight", type=float, default=NUT_LOSS_WEIGHT, help="Weight for nut channel in loss")
    parser.add_argument("--use-cosine-lr", action="store_true", default=USE_COSINE_LR, help="Enable cosine LR schedule (1e-4 to 1e-6)")
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
        args.ssim_weight,
        args.u_loss_weight,
        args.nut_loss_weight,
        args.use_cosine_lr,
    )
    if args.eval_case:
        evaluate_case(args.data, args.eval_case, model, ds, args.device)


if __name__ == "__main__":
    main()

