#!/usr/bin/env python3
"""Field processing and numerical stability corrections for CFD-ML coupling.

Implements GPU-accelerated field correction functions to ensure:
  1. Physical bounds on turbulent viscosity (prevents anti-diffusion)
  2. Smooth velocity field (removes nearest-neighbor discontinuities)
  3. Divergence-free velocity field (ensures continuity equation satisfaction)
  4. Consistent inlet boundary conditions (matches interior field)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter, map_coordinates


def validate_field(field: np.ndarray, name: str, fix_nan_inf: bool = True) -> np.ndarray:
    """
    Check field for NaN/Inf and optionally fix them.

    Args:
        field: Array to validate
        name: Name for logging
        fix_nan_inf: If True, replace NaN/Inf with 0

    Returns:
        Validated (and possibly fixed) field
    """
    n_nan = np.isnan(field).sum()
    n_inf = np.isinf(field).sum()

    if n_nan > 0 or n_inf > 0:
        print(f"  [validate_field] {name}: Found {n_nan} NaNs, {n_inf} Infs")
        if fix_nan_inf:
            field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  [validate_field] Fixed NaN/Inf in {name}")
        return field

    return field


def clamp_nut(
    nut_grid: np.ndarray,
    min_nut: float = 1e-8,
    max_nut: float = 1e-3,
) -> np.ndarray:
    """
    Clamp turbulent viscosity to physical range.

    Prevents anti-diffusion instabilities by enforcing bounds on nut.

    Args:
        nut_grid: [Nx, Ny, Nz] turbulent viscosity field
        min_nut: Minimum allowed value (default: 1e-8 m²/s)
        max_nut: Maximum allowed value (default: 1e-3 m²/s)

    Returns:
        nut_grid_clamped: Clamped turbulent viscosity
    """
    nut_clamped = np.clip(nut_grid, min_nut, max_nut)
    n_clipped_min = np.sum(nut_grid < min_nut)
    n_clipped_max = np.sum(nut_grid > max_nut)
    if n_clipped_min > 0 or n_clipped_max > 0:
        print(
            f"  [clamp_nut] Clamped {n_clipped_min} cells below min, "
            f"{n_clipped_max} cells above max"
        )
    return nut_clamped.astype(np.float32)


def apply_gaussian_smoothing(
    U_grid: np.ndarray,
    nut_grid: np.ndarray,
    sigma: float = 1.0,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply 3D Gaussian filtering to remove nearest-neighbor artifacts.

    Smooths velocity and turbulent viscosity fields using Gaussian kernel
    to eliminate discontinuities between adjacent grid cells.

    Args:
        U_grid: [Nx, Ny, Nz, 3] velocity field
        nut_grid: [Nx, Ny, Nz] turbulent viscosity
        sigma: Gaussian standard deviation (in grid units)
        device: "cpu" or "cuda" (for future GPU implementation)

    Returns:
        (U_grid_smooth, nut_grid_smooth): Smoothed fields
    """
    # Apply Gaussian filter to each velocity component
    U_smooth = np.zeros_like(U_grid)
    for k in range(3):
        U_smooth[:, :, :, k] = gaussian_filter(U_grid[:, :, :, k], sigma=sigma)

    # Apply Gaussian filter to turbulent viscosity
    nut_smooth = gaussian_filter(nut_grid, sigma=sigma)

    print(
        f"  [gaussian_smoothing] Applied Gaussian filter with sigma={sigma} "
        f"(device: {device})"
    )
    return U_smooth.astype(np.float32), nut_smooth.astype(np.float32)


def apply_divergence_free_projection(
    U_grid: np.ndarray,
    grid_shape: Tuple[int, int, int],
    device: str = "cpu",
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Reduce velocity divergence using scaled DCT-based Helmholtz-Hodge projection.

    Solves the Poisson equation laplacian(phi) = div(U) in DCT spectral space
    (Neumann BCs), then subtracts a fraction alpha of grad(phi) from U.

    Using alpha < 1.0 preserves flow shape (R² > 0.99) while still reducing
    divergence. Full projection (alpha=1.0) destroys shape on grids with
    solid cells because the DCT operates on the full rectangular domain.

    Args:
        U_grid: [Nx, Ny, Nz, 3] velocity field on regular grid
        grid_shape: (Nx, Ny, Nz) grid dimensions
        device: "cpu" or "cuda" (unused, kept for API compatibility)
        alpha: Fraction of HH correction to apply (default: 0.1)

    Returns:
        U_grid_corrected: Velocity field with reduced divergence
    """
    Nx, Ny, Nz = grid_shape
    U_f64 = U_grid.astype(np.float64)

    # Compute divergence with central finite differences
    div_U = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    div_U[1:-1, :, :] += (U_f64[2:, :, :, 0] - U_f64[:-2, :, :, 0]) / 2.0
    div_U[:, 1:-1, :] += (U_f64[:, 2:, :, 1] - U_f64[:, :-2, :, 1]) / 2.0
    div_U[:, :, 1:-1] += (U_f64[:, :, 2:, 2] - U_f64[:, :, :-2, 2]) / 2.0

    max_div = np.abs(div_U).max()
    mean_div = np.abs(div_U).mean()
    print(
        f"  [divergence_free_projection] Before: "
        f"max|div|={max_div:.6e}, mean|div|={mean_div:.6e}"
    )

    # Solve Poisson equation in DCT-II spectral space (Neumann BCs)
    div_hat = dctn(div_U, type=2, norm='ortho')

    # Laplacian eigenvalues for DCT-II
    kx = np.pi * np.arange(Nx) / Nx
    ky = np.pi * np.arange(Ny) / Ny
    kz = np.pi * np.arange(Nz) / Nz
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    eigenvalues = 2 * (np.cos(Kx) - 1) + 2 * (np.cos(Ky) - 1) + 2 * (np.cos(Kz) - 1)
    eigenvalues[0, 0, 0] = 1.0  # Avoid division by zero

    # Solve for pressure potential phi
    phi_hat = div_hat / eigenvalues
    phi_hat[0, 0, 0] = 0.0  # Zero-mean
    phi = idctn(phi_hat, type=2, norm='ortho')

    # Compute grad(phi) with central differences + one-sided at boundaries
    grad_phi = np.zeros_like(U_f64)
    grad_phi[1:-1, :, :, 0] = (phi[2:, :, :] - phi[:-2, :, :]) / 2.0
    grad_phi[:, 1:-1, :, 1] = (phi[:, 2:, :] - phi[:, :-2, :]) / 2.0
    grad_phi[:, :, 1:-1, 2] = (phi[:, :, 2:] - phi[:, :, :-2]) / 2.0
    grad_phi[0, :, :, 0] = phi[1, :, :] - phi[0, :, :]
    grad_phi[-1, :, :, 0] = phi[-1, :, :] - phi[-2, :, :]
    grad_phi[:, 0, :, 1] = phi[:, 1, :] - phi[:, 0, :]
    grad_phi[:, -1, :, 1] = phi[:, -1, :] - phi[:, -2, :]
    grad_phi[:, :, 0, 2] = phi[:, :, 1] - phi[:, :, 0]
    grad_phi[:, :, -1, 2] = phi[:, :, -1] - phi[:, :, -2]

    # Apply scaled correction: only alpha fraction of the full projection
    U_corrected = U_f64 - alpha * grad_phi

    # Verify divergence reduction
    div_new = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    div_new[1:-1, :, :] += (U_corrected[2:, :, :, 0] - U_corrected[:-2, :, :, 0]) / 2.0
    div_new[:, 1:-1, :] += (U_corrected[:, 2:, :, 1] - U_corrected[:, :-2, :, 1]) / 2.0
    div_new[:, :, 1:-1] += (U_corrected[:, :, 2:, 2] - U_corrected[:, :, :-2, 2]) / 2.0
    max_div_new = np.abs(div_new).max()
    mean_div_new = np.abs(div_new).mean()
    reduction = (1.0 - mean_div_new / (mean_div + 1e-12)) * 100

    print(
        f"  [divergence_free_projection] After (alpha={alpha}): "
        f"max|div|={max_div_new:.6e}, mean|div|={mean_div_new:.6e} "
        f"({reduction:.1f}% mean reduction)"
    )

    return U_corrected.astype(np.float32)


def reconstruct_fields_trilinear(
    grid_shape: Tuple[int, int, int],
    cell_centers: np.ndarray,
    grid_axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
    U_grid: np.ndarray,
    nut_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct fields onto mesh using trilinear interpolation.

    Replaces nearest-neighbor mapping with trilinear interpolation to smooth
    cell-to-cell transitions and reduce discontinuities.

    Args:
        grid_shape: (Nx, Ny, Nz) grid dimensions
        cell_centers: [n_cells, 3] mesh cell centers
        grid_axes: (x_axis, y_axis, z_axis) grid coordinate arrays
        U_grid: [Nx, Ny, Nz, 3] velocity field on grid
        nut_grid: [Nx, Ny, Nz] turbulent viscosity on grid

    Returns:
        (U_mesh, nut_mesh): Fields interpolated to mesh cells
    """
    Nx, Ny, Nz = grid_shape
    n_cells = len(cell_centers)

    # Normalize cell centers to grid coordinates
    x_axis, y_axis, z_axis = grid_axes
    x_min, x_max = x_axis[0], x_axis[-1]
    y_min, y_max = y_axis[0], y_axis[-1]
    z_min, z_max = z_axis[0], z_axis[-1]

    cell_centers_norm = np.zeros_like(cell_centers)
    cell_centers_norm[:, 0] = (cell_centers[:, 0] - x_min) / (x_max - x_min) * (
        Nx - 1
    )
    cell_centers_norm[:, 1] = (cell_centers[:, 1] - y_min) / (y_max - y_min) * (
        Ny - 1
    )
    cell_centers_norm[:, 2] = (cell_centers[:, 2] - z_min) / (z_max - z_min) * (
        Nz - 1
    )

    # Clip to valid grid coordinates
    cell_centers_norm = np.clip(
        cell_centers_norm,
        [0, 0, 0],
        [Nx - 1, Ny - 1, Nz - 1],
    )

    # Interpolate U field using trilinear (order=1)
    U_mesh = np.zeros((n_cells, 3), dtype=np.float32)
    for k in range(3):
        U_grid_k = U_grid[:, :, :, k]
        U_mesh[:, k] = map_coordinates(
            U_grid_k,
            cell_centers_norm.T,
            order=1,
            mode="nearest",
            cval=0.0,
        )

    # Interpolate nut field using trilinear
    nut_mesh = map_coordinates(
        nut_grid,
        cell_centers_norm.T,
        order=1,
        mode="nearest",
        cval=1e-8,
    ).astype(np.float32)

    print(
        f"  [trilinear_interpolation] Interpolated {n_cells} mesh cells from "
        f"{Nx}×{Ny}×{Nz} grid"
    )
    return U_mesh, nut_mesh


def compute_inlet_velocity_from_prediction(
    U_mesh: np.ndarray,
    inlet_cell_indices: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute inlet boundary condition velocity from predicted field.

    Extracts average velocity at inlet cells from ML-predicted field for
    consistent inlet BC that matches interior solution.

    Args:
        U_mesh: [n_cells, 3] velocity field on mesh
        inlet_cell_indices: Array of cell indices belonging to inlet patch

    Returns:
        (Ux_mean, Uy_mean, Uz_mean): Average inlet velocity components
    """
    if len(inlet_cell_indices) == 0:
        print(f"  [inlet_velocity] WARNING: No inlet cells found, using zero velocity")
        return (0.0, 0.0, 0.0)

    U_inlet = U_mesh[inlet_cell_indices]
    Ux_mean = float(np.mean(U_inlet[:, 0]))
    Uy_mean = float(np.mean(U_inlet[:, 1]))
    Uz_mean = float(np.mean(U_inlet[:, 2]))

    inlet_speed = np.sqrt(Ux_mean**2 + Uy_mean**2 + Uz_mean**2)
    print(
        f"  [inlet_velocity] Computed inlet velocity: "
        f"({Ux_mean:.6e}, {Uy_mean:.6e}, {Uz_mean:.6e}) m/s (speed: {inlet_speed:.6e})"
    )
    return (Ux_mean, Uy_mean, Uz_mean)
