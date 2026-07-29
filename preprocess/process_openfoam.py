#!/usr/bin/env python3
"""End-to-end processing of parametric tank OpenFOAM cases.

Pipeline steps:
- Read U (vector) and nut (scalar) fields at time=1800 for each CubeXX case.
- Compute cell centres from polyMesh and store raw fields + centres to HDF5.
- Interpolate onto a regular grid (80 x 50 x 50) and store to HDF5 (processed).
- Build binary mask and distance-to-wall from interpolated data.
- Persist the contents of cases.txt.
- Generate Y-slice plots for Cube03 to inspect interpolation quality.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

# Target regular grid resolution (Nx, Ny, Nz)
REG_GRID = (80, 50, 50)
TIME_STEP = "1800"
# Distance buffer (relative to max grid spacing) to tag a grid point as wall-proximate
# Lowered to make masks less conservative.
WALL_BUFFER_FACTOR = 0.75


@dataclass
class CaseData:
    name: str
    param_value: float
    centers: np.ndarray  # (N, 3)
    U: np.ndarray  # (N, 3)
    nut: np.ndarray  # (N,)
    U_grid: np.ndarray  # (Nx, Ny, Nz, 3)
    nut_grid: np.ndarray  # (Nx, Ny, Nz)
    mask: np.ndarray  # (Nx, Ny, Nz), bool
    distance: np.ndarray  # (Nx, Ny, Nz)


def parse_cases_txt(path: Path) -> Dict[str, float]:
    """Return mapping Case_XX -> param_value (comma decimal -> dot)."""
    mapping: Dict[str, float] = {}
    current_name: str | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Case_"):
            current_name = line
        else:
            value = float(line.replace(",", "."))
            if current_name is None:
                raise ValueError(f"Found value {line} before case name")
            mapping[current_name] = value
            current_name = None
    return mapping


def load_boundaries_csv(
    path: Path, tube_layers: int = REG_GRID[2], tube_axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Load boundary point cloud; return (wall_like_points, internal_points).

    The CSV currently contains sparse samples for the tube columns (mostly at
    the top/bottom caps).  To keep those tubes present along the full height,
    we densify them by extruding each unique footprint (the two non-height
    coordinates) across the full range of the height axis using `tube_layers`
    uniformly spaced levels.  Default height axis is 0 (x).
    """
    if tube_axis not in (0, 1, 2):
        raise ValueError("tube_axis must be 0, 1 or 2")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    labels = data["Block_Name"]
    pts = np.vstack([data["Points_0"], data["Points_1"], data["Points_2"]]).T.astype(float)
    wall_mask = labels != "Internal"
    wall_pts = pts[wall_mask]
    internal_pts = pts[~wall_mask]

    # Extrude sparse tube points along the height axis so columns exist mid-height.
    if tube_layers and np.any(labels == "Tubes"):
        tube_pts = pts[labels == "Tubes"]
        footprint_axes = [ax for ax in (0, 1, 2) if ax != tube_axis]
        footprints = np.unique(tube_pts[:, footprint_axes], axis=0)
        h_min, h_max = pts[:, tube_axis].min(), pts[:, tube_axis].max()
        h_levels = np.linspace(h_min, h_max, tube_layers)
        tube_stack = np.zeros((len(footprints) * len(h_levels), 3), dtype=float)
        tube_stack[:, tube_axis] = np.tile(h_levels, len(footprints))
        for i, ax in enumerate(footprint_axes):
            tube_stack[:, ax] = np.repeat(footprints[:, i], len(h_levels))
        wall_pts = np.vstack([wall_pts, tube_stack])
        print(
            f"[info] Extruded {len(footprints)} tube footprints across {tube_layers} levels "
            f"along axis {tube_axis} ({h_min:.3f}..{h_max:.3f})"
        )
    return wall_pts, internal_pts


def characteristic_spacing(points: np.ndarray, k: int = 6) -> float:
    """Median distance to k-th neighbour as characteristic spacing."""
    if len(points) <= k:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)
    # dists[:, 0] is zero (self); take kth neighbor distance
    kth = dists[:, k - 1]
    return float(np.median(kth))


def _read_count_and_iter(path: Path) -> Tuple[int, Iterable[str]]:
    """Skip OpenFOAM header, return count and iterator over payload lines."""
    lines = path.read_text().splitlines()
    count_idx = None
    for i, line in enumerate(lines):
        if line.strip().isdigit():
            count_idx = i
            break
    if count_idx is None:
        raise ValueError(f"Could not find item count in {path}")
    count = int(lines[count_idx].strip())
    # Skip the '(' line that follows the count
    payload = lines[count_idx + 2 :]
    return count, payload


def read_points(path: Path) -> np.ndarray:
    count, payload = _read_count_and_iter(path)
    pts = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        vals = payload[i].strip().strip("()").split()
        pts[i] = [float(v) for v in vals]
    return pts


def read_owner(path: Path) -> np.ndarray:
    count, payload = _read_count_and_iter(path)
    owners = np.empty(count, dtype=np.int64)
    for i in range(count):
        owners[i] = int(payload[i].strip())
    return owners


def read_neighbour(path: Path) -> np.ndarray:
    count, payload = _read_count_and_iter(path)
    neigh = np.empty(count, dtype=np.int64)
    for i in range(count):
        neigh[i] = int(payload[i].strip())
    return neigh


def read_faces(path: Path) -> List[List[int]]:
    count, payload = _read_count_and_iter(path)
    faces: List[List[int]] = []
    for i in range(count):
        line = payload[i].strip()
        m = re.match(r"(\d+)\((.*)\)", line)
        if not m:
            raise ValueError(f"Malformed face line {i} in {path}")
        face_pts = [int(x) for x in m.group(2).split()]
        faces.append(face_pts)
    return faces


def compute_cell_centers(mesh_dir: Path, expected_cells: int | None = None) -> np.ndarray:
    points = read_points(mesh_dir / "points")
    faces = read_faces(mesh_dir / "faces")
    owners = read_owner(mesh_dir / "owner")
    neigh = read_neighbour(mesh_dir / "neighbour")

    # Derive number of cells from both owner and neighbour (some meshes may
    # have neighbour labels exceeding owner.max()).
    n_cells = int(max(owners.max(), neigh.max())) + 1
    if expected_cells:
        n_cells = max(n_cells, expected_cells)
    # Faster accumulation: sum face centroids into each cell and divide by count
    accum = np.zeros((n_cells, 3), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)

    neigh_len = len(neigh)
    for idx, pts in enumerate(faces):
        face_pts = np.asarray(pts, dtype=np.int64)
        centroid = points[face_pts].mean(axis=0)

        own = int(owners[idx])
        if own < n_cells:
            accum[own] += centroid
            counts[own] += 1
        if idx < neigh_len:
            nb = int(neigh[idx])
            if nb < n_cells:
                accum[nb] += centroid
                counts[nb] += 1

    centers = np.empty((n_cells, 3), dtype=np.float64)
    centers[:] = np.nan
    nonzero = counts > 0
    centers[nonzero] = accum[nonzero] / counts[nonzero][:, None]
    return centers


def read_scalar_field(path: Path, expected: int | None = None) -> np.ndarray:
    count, payload = _read_count_and_iter(path)
    if expected is not None and count != expected:
        print(f"[warn] {path} count {count} != expected {expected}; using file count")
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        out[i] = float(payload[i].strip().strip("()"))
    return out


def read_vector_field(path: Path, expected: int | None = None) -> np.ndarray:
    count, payload = _read_count_and_iter(path)
    if expected is not None and count != expected:
        print(f"[warn] {path} count {count} != expected {expected}; using file count")
    out = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        vals = payload[i].strip().strip("()").split()
        out[i] = [float(v) for v in vals]
    return out


def build_regular_grid(centers: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    grid_axes = tuple(np.linspace(lo, hi, n) for lo, hi, n in zip(mins, maxs, REG_GRID))
    gx, gy, gz = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([gx, gy, gz], axis=-1)
    return grid_points, grid_axes


def query_nn(points: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest-neighbor indices and distances for queries."""
    tree = cKDTree(points)
    dist, idx = tree.query(queries, k=1, workers=-1)
    return idx, dist


def build_mask_from_boundaries(
    grid_points_flat: np.ndarray,
    spacing: Tuple[float, float, float],
    wall_points: np.ndarray,
    buffer_factor: float = WALL_BUFFER_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify grid points as fluid if far enough from wall-like points."""
    if len(wall_points) == 0:
        raise ValueError("No wall/inlet/outlet points found in boundary CSV.")
    wall_tree = cKDTree(wall_points)
    dist_to_wall, _ = wall_tree.query(grid_points_flat, k=1, workers=-1)
    dist_grid = dist_to_wall.reshape(REG_GRID)
    char_space = characteristic_spacing(wall_points)
    # Use a conservative buffer: scaled grid spacing, but not below 2x local wall spacing
    buffer = max(buffer_factor * max(spacing), 2.0 * char_space)
    mask = dist_grid > buffer
    return mask.astype(np.uint8), dist_grid


def process_case(
    case_dir: Path,
    case_name: str,
    param_value: float,
    out_plot_dir: Path,
    wall_points: np.ndarray,
    solid_dilate: int = 0,
) -> CaseData:
    time_dir = case_dir / TIME_STEP
    mesh_dir = case_dir / "constant" / "polyMesh"

    owners = read_owner(mesh_dir / "owner")
    neigh = read_neighbour(mesh_dir / "neighbour")
    base_cells = int(max(owners.max(), neigh.max())) + 1

    U = read_vector_field(time_dir / "U", expected=None)
    nut = read_scalar_field(time_dir / "nut", expected=None)

    n_cells = max(base_cells, len(U), len(nut))
    centers = compute_cell_centers(mesh_dir, expected_cells=n_cells)

    # Drop any cells without geometry (NaN centers) and align fields
    valid_mask = ~np.isnan(centers).any(axis=1)
    centers = centers[valid_mask]
    U = U[valid_mask[: len(U)]]
    nut = nut[valid_mask[: len(nut)]]

    grid_points, grid_axes = build_regular_grid(centers)
    grid_points_flat = grid_points.reshape(-1, 3)
    idx, dist = query_nn(centers, grid_points_flat)
    dist_grid = dist.reshape(REG_GRID)

    U_grid = np.empty((*REG_GRID, 3), dtype=np.float64)
    for c in range(3):
        U_grid[..., c] = U[idx, c].reshape(REG_GRID)
    nut_grid = nut[idx].reshape(REG_GRID)

    spacing = tuple(float(axis[1] - axis[0]) for axis in grid_axes)
    # Geometric validity: ignore grid points too far from any cell center
    mask_geom = dist_grid <= (1.5 * max(spacing))
    nut_grid = np.where(mask_geom, nut_grid, np.nan)
    U_grid = np.where(mask_geom[..., None], U_grid, np.nan)

    # Boundary-based mask/distance
    mask, distance = build_mask_from_boundaries(grid_points_flat, spacing, wall_points)
    if solid_dilate > 0:
        solid = ~mask.astype(bool)
        solid = binary_dilation(solid, iterations=solid_dilate)
        mask = (~solid).astype(np.uint8)

    if case_name.lower() == "cube03":
        out_plot_dir.mkdir(parents=True, exist_ok=True)
        mag = np.linalg.norm(U_grid, axis=-1)

        def write_summary_png(
            data_3d: np.ndarray, fname: str, cmap: str, label: str, overlay_mask: np.ndarray | None = None
        ):
            Nx = data_3d.shape[0]  # slice along x (height)
            ncols = 8
            nrows = math.ceil(Nx / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
            axes = axes.ravel()
            last_im = None
            for j in range(Nx):
                ax = axes[j]
                im = ax.imshow(
                    data_3d[j, :, :].T,
                    origin="lower",
                    cmap=cmap,
                    interpolation="nearest",
                )
                last_im = im
                if overlay_mask is not None:
                    ax.contour(overlay_mask[j, :, :].T, levels=[0.5], colors="red", linewidths=0.4)
                ax.set_title(f"X={j}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            for k in range(Nx, len(axes)):
                axes[k].axis("off")
            if last_im is not None:
                fig.colorbar(last_im, ax=axes.tolist(), shrink=0.6, label=label)
            fig.tight_layout()
            fig.savefig(out_plot_dir / fname, dpi=200)
            plt.close(fig)

        # Apply masking for visualization (show walls as white via NaN)
        mag_vis = np.where(mask, mag, np.nan)
        nut_vis = np.where(mask, nut_grid, np.nan)

        # Main summaries (base threshold)
        write_summary_png(mag_vis, "summary_U.png", "viridis", "|U|", overlay_mask=mask)
        write_summary_png(nut_vis, "summary_nut.png", "plasma", "nut", overlay_mask=mask)
        write_summary_png(mask.astype(float), "summary_mask_base.png", "gray", "mask", overlay_mask=None)
        write_summary_png(distance, "summary_distance_base.png", "magma", "distance", overlay_mask=None)

        # Sensitivity to wall buffer factor for mask/distance
        test_buffers = [0.025, 0.001, 0.075, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25]
        for bf in test_buffers:
            mask_t, dist_t = build_mask_from_boundaries(grid_points_flat, spacing, wall_points, buffer_factor=bf)
            suffix = f"{bf:.2f}".replace(".", "p")
            write_summary_png(
                mask_t.astype(float),
                f"summary_mask_thr{suffix}.png",
                "gray",
                f"mask buffer={bf}",
                overlay_mask=None,
            )
            write_summary_png(
                dist_t,
                f"summary_distance_thr{suffix}.png",
                "magma",
                f"distance buffer={bf}",
                overlay_mask=None,
            )

    return CaseData(
        name=case_name,
        param_value=param_value,
        centers=centers,
        U=U,
        nut=nut,
        U_grid=U_grid,
        nut_grid=nut_grid,
        mask=mask,
        distance=distance,
    )


def write_h5(out_path: Path, cases: List[CaseData], cases_txt: str):
    with h5py.File(out_path, "w") as h5:
        # Store cases.txt as bytes to stay NumPy 2.0 compatible
        h5.create_dataset("cases_txt", data=np.bytes_(cases_txt))
        raw_grp = h5.create_group("raw")
        proc_grp = h5.create_group("processed")
        for c in cases:
            g_raw = raw_grp.create_group(c.name)
            g_raw.create_dataset("centers", data=c.centers)
            g_raw.create_dataset("U", data=c.U)
            g_raw.create_dataset("nut", data=c.nut)

            g_proc = proc_grp.create_group(c.name)
            g_proc.create_dataset("U", data=c.U_grid)
            g_proc.create_dataset("nut", data=c.nut_grid)
            g_proc.create_dataset("mask", data=c.mask)
            g_proc.create_dataset("distance", data=c.distance)
            g_proc.create_dataset("case_param", data=np.full(REG_GRID, c.param_value, dtype=np.float64))


def find_case_dirs(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.iterdir() if p.is_dir() and re.match(r"Cube\d+", p.name)])


def main():
    parser = argparse.ArgumentParser(description="Process OpenFOAM tank cubes to HDF5.")
    parser.add_argument("--base", type=Path, default=Path(__file__).parent, help="Base directory containing CubeXX")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed_data.h5",
        help="Output HDF5 file",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "preprocess_plots",
        help="Output folder for Cube03 Y-slices",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=Path(__file__).parent / "cell_data_walls_inlet_outlet_cols.csv",
        help="CSV with boundary point cloud",
    )
    parser.add_argument(
        "--tube-layers",
        type=int,
        default=REG_GRID[2],
        help="Number of z-levels to extrude tube points across the height",
    )
    parser.add_argument(
        "--tube-axis",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Axis index representing height for tube extrusion (0=x,1=y,2=z)",
    )
    parser.add_argument(
        "--solid-dilate",
        type=int,
        default=0,
        help="Binary dilation iterations on solid regions (thickens walls/tubes)",
    )
    parser.add_argument(
        "--wall-buffer-factor",
        type=float,
        default=WALL_BUFFER_FACTOR,
        help="Multiplier on max grid spacing for wall distance buffer",
    )
    args = parser.parse_args()

    cases_map = parse_cases_txt(args.base / "cases.txt")
    wall_pts, _ = load_boundaries_csv(
        args.boundaries, tube_layers=args.tube_layers, tube_axis=args.tube_axis
    )
    case_dirs = find_case_dirs(args.base)

    processed: List[CaseData] = []
    for case_dir in case_dirs:
        case_name = case_dir.name
        # Map Case_XX name in txt to CubeXX
        lookup_name = f"Case_{case_name[-2:]}"
        param_val = cases_map.get(lookup_name)
        if param_val is None:
            raise ValueError(f"No parameter found for {case_name}")
        print(f"Processing {case_name} (param {param_val})")
        processed.append(
            process_case(
                case_dir,
                case_name,
                param_val,
                args.plots,
                wall_pts,
                solid_dilate=args.solid_dilate,
            )
        )

    write_h5(args.out, processed, (args.base / "cases.txt").read_text())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

