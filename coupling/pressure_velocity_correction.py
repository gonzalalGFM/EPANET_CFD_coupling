#!/usr/bin/env python3
"""GPU-accelerated pressure-velocity correction.

Uses an FFT-based spectral Poisson solver on GPU with mask-aware
divergence computation and correction.  This replaces the DCT-based
approach (CPU, alpha=0.1) that was limited because it operates on the
full rectangular domain including solid cells.

Key improvements over the DCT approach:
  - GPU: FFT on CUDA tensors
  - Mask-aware: divergence and correction restricted to fluid cells
  - Iterative outer loop with auto-stop: progressively reduces
    divergence, stops when diminishing returns
  - Exact spectral solve per iteration

Pipeline per outer iteration:
    1. div = divergence(U) * mask        (only fluid cells)
    2. p'  = FFT_solve(Lap(p') = div)    (exact, periodic BCs)
    3. U   = U - alpha * grad(p') * mask (correct fluid cells only)
    4. Check convergence; stop if mean|div| increases
"""

import math
from typing import Tuple

import torch


def compute_divergence_3d(
    U: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """Compute divergence of velocity field using central differences.

    Args:
        U: (Nx, Ny, Nz, 3) velocity tensor on GPU
        dx, dy, dz: grid spacing in each direction

    Returns:
        div: (Nx, Ny, Nz) divergence field
    """
    Nx, Ny, Nz, _ = U.shape
    div = torch.zeros(Nx, Ny, Nz, device=U.device, dtype=U.dtype)

    # Central differences interior, one-sided at boundaries
    div[1:-1, :, :] += (U[2:, :, :, 0] - U[:-2, :, :, 0]) / (2.0 * dx)
    div[0, :, :] += (U[1, :, :, 0] - U[0, :, :, 0]) / dx
    div[-1, :, :] += (U[-1, :, :, 0] - U[-2, :, :, 0]) / dx

    div[:, 1:-1, :] += (U[:, 2:, :, 1] - U[:, :-2, :, 1]) / (2.0 * dy)
    div[:, 0, :] += (U[:, 1, :, 1] - U[:, 0, :, 1]) / dy
    div[:, -1, :] += (U[:, -1, :, 1] - U[:, -2, :, 1]) / dy

    div[:, :, 1:-1] += (U[:, :, 2:, 2] - U[:, :, :-2, 2]) / (2.0 * dz)
    div[:, :, 0] += (U[:, :, 1, 2] - U[:, :, 0, 2]) / dz
    div[:, :, -1] += (U[:, :, -1, 2] - U[:, :, -2, 2]) / dz

    return div


def compute_gradient_3d(
    p: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """Compute gradient of scalar field using central differences.

    Args:
        p: (Nx, Ny, Nz) pressure tensor on GPU
        dx, dy, dz: grid spacing in each direction

    Returns:
        grad_p: (Nx, Ny, Nz, 3) gradient field
    """
    Nx, Ny, Nz = p.shape
    grad_p = torch.zeros(Nx, Ny, Nz, 3, device=p.device, dtype=p.dtype)

    grad_p[1:-1, :, :, 0] = (p[2:, :, :] - p[:-2, :, :]) / (2.0 * dx)
    grad_p[0, :, :, 0] = (p[1, :, :] - p[0, :, :]) / dx
    grad_p[-1, :, :, 0] = (p[-1, :, :] - p[-2, :, :]) / dx

    grad_p[:, 1:-1, :, 1] = (p[:, 2:, :] - p[:, :-2, :]) / (2.0 * dy)
    grad_p[:, 0, :, 1] = (p[:, 1, :] - p[:, 0, :]) / dy
    grad_p[:, -1, :, 1] = (p[:, -1, :] - p[:, -2, :]) / dy

    grad_p[:, :, 1:-1, 2] = (p[:, :, 2:] - p[:, :, :-2]) / (2.0 * dz)
    grad_p[:, :, 0, 2] = (p[:, :, 1] - p[:, :, 0]) / dz
    grad_p[:, :, -1, 2] = (p[:, :, -1] - p[:, :, -2]) / dz

    return grad_p


def fft_poisson_3d(
    rhs: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """Solve Lap(p) = rhs using FFT on GPU (periodic BCs).

    Exact spectral solver using eigenvalues that match the 3-point
    central-difference Laplacian stencil.  Zero-mean solution.

    Args:
        rhs: (Nx, Ny, Nz) right-hand side
        dx, dy, dz: grid spacing

    Returns:
        p: (Nx, Ny, Nz) solution
    """
    Nx, Ny, Nz = rhs.shape
    device = rhs.device
    dtype = rhs.dtype

    nx = torch.arange(Nx, device=device, dtype=dtype)
    ny = torch.arange(Ny, device=device, dtype=dtype)
    nz = torch.arange(Nz, device=device, dtype=dtype)

    eig_x = (2.0 * torch.cos(2.0 * math.pi * nx / Nx) - 2.0) / (dx * dx)
    eig_y = (2.0 * torch.cos(2.0 * math.pi * ny / Ny) - 2.0) / (dy * dy)
    eig_z = (2.0 * torch.cos(2.0 * math.pi * nz / Nz) - 2.0) / (dz * dz)

    eigenvalues = eig_x[:, None, None] + eig_y[None, :, None] + eig_z[None, None, :]
    eigenvalues[0, 0, 0] = 1.0  # avoid division by zero for k=0 mode

    rhs_hat = torch.fft.fftn(rhs)
    p_hat = rhs_hat / eigenvalues
    p_hat[0, 0, 0] = 0.0  # zero-mean solution

    return torch.fft.ifftn(p_hat).real


def pressure_velocity_correction(
    U_grid: "np.ndarray",
    p_ml: "np.ndarray",
    mask: "np.ndarray",
    dx: float,
    dy: float,
    dz: float,
    n_outer: int = 10,
    n_inner: int = 100,
    alpha: float = 0.3,
    device: str = "cuda",
) -> "Tuple[np.ndarray, np.ndarray, list]":
    """Iterative pressure-velocity correction on GPU.

    Uses FFT-based spectral Poisson solver with mask-aware correction.
    Divergence is computed only in fluid cells and velocity correction
    is applied only to fluid cells.  Auto-stops when the mean divergence
    starts increasing (diminishing returns from the masked FFT solve).

    Args:
        U_grid: (Nx, Ny, Nz, 3) velocity field (numpy)
        p_ml: (Nx, Ny, Nz) ML-predicted pressure (numpy, kept for API)
        mask: (Nx, Ny, Nz) fluid mask, 1=fluid 0=solid (numpy)
        dx, dy, dz: grid spacing
        n_outer: max number of outer correction iterations
        n_inner: unused (FFT solve is exact); kept for API compat
        alpha: fraction of pressure gradient correction per step
        device: 'cuda' or 'cpu'

    Returns:
        U_corrected: (Nx, Ny, Nz, 3) corrected velocity (numpy)
        p_final: (Nx, Ny, Nz) final pressure correction (numpy)
        div_history: list of (max_div, mean_div) per outer iteration
    """
    import numpy as np

    U = torch.from_numpy(U_grid.astype(np.float64)).to(device)
    mask_t = torch.from_numpy(mask.astype(np.float64)).to(device)
    mask_3d = mask_t.unsqueeze(-1)  # (Nx, Ny, Nz, 1)

    # Note: solid cells are NOT zeroed — the ML velocity is preserved
    # everywhere, and only fluid cells are corrected.  This matches the
    # DCT approach and avoids introducing artificial boundary divergence.

    div_history = []
    p_corr = None

    # Initial divergence (full domain for comparable metrics)
    div0_full = compute_divergence_3d(U, dx, dy, dz)
    max_div0 = float(torch.abs(div0_full).max())
    mean_div0 = float(torch.abs(div0_full).mean())
    div_history.append((max_div0, mean_div0))
    print(
        f"  [pressure_correction] Initial: "
        f"max|div|={max_div0:.6e}, mean|div|={mean_div0:.6e}"
    )

    # Keep best state for rollback if divergence increases
    U_best = U.clone()
    best_mean_div = mean_div0

    for outer in range(n_outer):
        # 1. Divergence in fluid cells only (masked RHS avoids solid artifacts)
        div = compute_divergence_3d(U, dx, dy, dz) * mask_t

        # 2. Exact spectral Poisson solve
        p_corr = fft_poisson_3d(div, dx, dy, dz)

        # 3. Correct velocity in fluid cells only
        grad_p = compute_gradient_3d(p_corr, dx, dy, dz)
        U = U - alpha * grad_p * mask_3d

        # 4. Full-domain divergence for monitoring
        div_new = compute_divergence_3d(U, dx, dy, dz)
        max_div = float(torch.abs(div_new).max())
        mean_div = float(torch.abs(div_new).mean())
        div_history.append((max_div, mean_div))

        reduction = (1.0 - mean_div / (mean_div0 + 1e-12)) * 100
        print(
            f"  [pressure_correction] Outer {outer+1}/{n_outer}: "
            f"max|div|={max_div:.6e}, mean|div|={mean_div:.6e} "
            f"({reduction:.1f}% mean reduction from initial)"
        )

        # Track best state
        if mean_div < best_mean_div:
            U_best = U.clone()
            best_mean_div = mean_div
        else:
            # Mean divergence increased — roll back and stop
            print(
                f"  [pressure_correction] Auto-stop: mean|div| increased, "
                f"rolling back to best state"
            )
            U = U_best
            div_history.append((
                float(torch.abs(compute_divergence_3d(U, dx, dy, dz)).max()),
                best_mean_div,
            ))
            break

        if max_div < 1e-8:
            print(f"  [pressure_correction] Converged at outer iteration {outer+1}")
            break

    final_reduction = (1.0 - best_mean_div / (mean_div0 + 1e-12)) * 100
    print(
        f"  [pressure_correction] Final: mean|div| reduced by "
        f"{final_reduction:.1f}%"
    )

    U_corrected = U.cpu().numpy().astype(np.float32)
    p_final = p_corr.cpu().numpy().astype(np.float32) if p_corr is not None else np.zeros_like(U_grid[..., 0])

    return U_corrected, p_final, div_history
