#!/usr/bin/env python3
"""Phase 0: Pre-compute cell-to-grid index map and inlet area.

Reads the BaseCaseChlorine mesh, builds the regular grid (80 x 50 x 50),
and saves:
  - cell_to_grid_map.npy: for each of 334,736 cells, the nearest grid point (i,j,k)
  - inlet_area.txt: total area of the Inlet patch
  - precomputed_masks.npz: mask and distance arrays
"""

import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


def _read_count_and_iter(path: Path) -> Tuple[int, List[str]]:
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
    payload = lines[count_idx + 2 :]
    return count, payload


def read_points(path: Path) -> np.ndarray:
    """Read OpenFOAM points file."""
    count, payload = _read_count_and_iter(path)
    pts = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        vals = payload[i].strip().strip("()").split()
        pts[i] = [float(v) for v in vals]
    return pts


def read_owner(path: Path) -> np.ndarray:
    """Read OpenFOAM owner file."""
    count, payload = _read_count_and_iter(path)
    owners = np.empty(count, dtype=np.int64)
    for i in range(count):
        owners[i] = int(payload[i].strip())
    return owners


def read_neighbour(path: Path) -> np.ndarray:
    """Read OpenFOAM neighbour file."""
    count, payload = _read_count_and_iter(path)
    neigh = np.empty(count, dtype=np.int64)
    for i in range(count):
        neigh[i] = int(payload[i].strip())
    return neigh


def read_faces(path: Path) -> List[List[int]]:
    """Read OpenFOAM faces file."""
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


def compute_cell_centers(mesh_dir: Path) -> np.ndarray:
    """Compute cell centers from polyMesh."""
    points = read_points(mesh_dir / "points")
    faces = read_faces(mesh_dir / "faces")
    owners = read_owner(mesh_dir / "owner")
    neigh = read_neighbour(mesh_dir / "neighbour")

    n_cells = int(max(owners.max(), neigh.max())) + 1
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


def build_regular_grid(centers: np.ndarray, grid_shape: Tuple[int, int, int]):
    """Build regular grid from cell centers."""
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    grid_axes = tuple(np.linspace(lo, hi, n) for lo, hi, n in zip(mins, maxs, grid_shape))
    gx, gy, gz = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([gx, gy, gz], axis=-1)
    return grid_points, grid_axes


def query_nn(points: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest-neighbor indices and distances for queries."""
    tree = cKDTree(points)
    dist, idx = tree.query(queries, k=1, workers=-1)
    return idx, dist


def read_boundary_faces(boundary_file: Path) -> dict:
    """Parse OpenFOAM boundary file; return dict of {patch_name -> (nFaces, startFace)}."""
    lines = boundary_file.read_text().splitlines()
    patches = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and not line.startswith("//") and not line.startswith("/*"):
            # Try to parse as patch name
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line == "{":
                    patch_name = line
                    # Find nFaces and startFace
                    j = i + 2
                    nfaces = None
                    startface = None
                    while j < len(lines) and lines[j].strip() != "}":
                        bline = lines[j].strip()
                        if bline.startswith("nFaces"):
                            nfaces = int(bline.split()[-1].rstrip(";"))
                        elif bline.startswith("startFace"):
                            startface = int(bline.split()[-1].rstrip(";"))
                        j += 1
                    if nfaces is not None and startface is not None:
                        patches[patch_name] = (nfaces, startface)
                    i = j
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    return patches


def compute_inlet_area(mesh_dir: Path) -> float:
    """Compute total area of Inlet patch."""
    boundary_file = mesh_dir / "boundary"
    patches = read_boundary_faces(boundary_file)

    if "Inlet" not in patches:
        raise ValueError("Inlet patch not found in boundary file")

    nfaces, startface = patches["Inlet"]

    # Read faces and points
    points = read_points(mesh_dir / "points")
    faces = read_faces(mesh_dir / "faces")

    # Compute area for each inlet face
    total_area = 0.0
    for i in range(startface, startface + nfaces):
        face_pts = np.asarray(faces[i], dtype=np.int64)
        face_coords = points[face_pts]

        # Area of polygon: sum of triangle areas from first vertex
        area = 0.0
        for j in range(1, len(face_pts) - 1):
            v1 = face_coords[j] - face_coords[0]
            v2 = face_coords[j + 1] - face_coords[0]
            cross = np.cross(v1, v2)
            area += 0.5 * np.linalg.norm(cross)
        total_area += area

    return total_area


def find_inlet_cell_indices(mesh_dir: Path) -> np.ndarray:
    """
    Find mesh cell indices that belong to Inlet patch.

    Returns:
        inlet_cell_indices: Array of cell indices in inlet patch
    """
    boundary_file = mesh_dir / "boundary"
    patches = read_boundary_faces(boundary_file)

    if "Inlet" not in patches:
        print(f"[find_inlet_cell_indices] WARNING: Inlet patch not found in boundary file")
        return np.array([], dtype=np.int32)

    nfaces, startface = patches["Inlet"]

    # Read face ownership data
    owners = read_owner(mesh_dir / "owner")

    # Collect all cell indices that own inlet faces
    inlet_cells = set()
    for i in range(startface, startface + nfaces):
        inlet_cells.add(int(owners[i]))

    inlet_cell_indices = np.array(sorted(inlet_cells), dtype=np.int32)
    return inlet_cell_indices


def characteristic_spacing(points: np.ndarray, k: int = 6) -> float:
    """Median distance to k-th neighbour as characteristic spacing."""
    if len(points) <= k:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)
    kth = dists[:, k - 1]
    return float(np.median(kth))


def load_boundaries_csv(
    path: Path, tube_layers: int = 50, tube_axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Load boundary point cloud from CSV."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    labels = data["Block_Name"]
    pts = np.vstack([data["Points_0"], data["Points_1"], data["Points_2"]]).T.astype(float)
    wall_mask = labels != "Internal"
    wall_pts = pts[wall_mask]
    internal_pts = pts[~wall_mask]

    # Extrude sparse tube points
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


def build_mask_from_boundaries(
    grid_points_flat: np.ndarray,
    spacing: Tuple[float, float, float],
    wall_points: np.ndarray,
    buffer_factor: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify grid points as fluid if far enough from wall-like points."""
    if len(wall_points) == 0:
        raise ValueError("No wall/inlet/outlet points found in boundary CSV.")
    wall_tree = cKDTree(wall_points)
    dist_to_wall, _ = wall_tree.query(grid_points_flat, k=1, workers=-1)
    dist_grid = dist_to_wall.reshape((80, 50, 50))
    char_space = characteristic_spacing(wall_points)
    buffer = max(buffer_factor * max(spacing), 2.0 * char_space)
    mask = dist_grid > buffer
    return mask.astype(np.uint8), dist_grid


def main():
    # Load config (repository root)
    base_dir = Path(__file__).resolve().parents[1]
    config_file = base_dir / "configs" / "config.json"
    if not config_file.exists():
        config_file = base_dir / "config.json"
    with open(config_file) as f:
        config = json.load(f)
    mesh_dir = base_dir / config["openfoam"]["base_case_dir"] / "constant" / "polyMesh"
    grid_shape = tuple(config["ml"]["grid_shape"])

    print("[precompute_mapping] Starting...")

    # Read mesh and compute cell centers
    print("[precompute_mapping] Reading mesh and computing cell centers...")
    centers = compute_cell_centers(mesh_dir)
    valid_mask = ~np.isnan(centers).any(axis=1)
    centers = centers[valid_mask]
    print(f"  Found {len(centers)} valid cells")

    # Build regular grid
    print(f"[precompute_mapping] Building regular grid {grid_shape}...")
    grid_points, grid_axes = build_regular_grid(centers, grid_shape)
    grid_points_flat = grid_points.reshape(-1, 3)
    print(f"  Grid bounds: {[ax[[0, -1]] for ax in grid_axes]}")

    # Create cell-to-grid map
    print("[precompute_mapping] Creating cell-to-grid index map...")
    idx, dist = query_nn(grid_points_flat, centers)
    cell_to_grid_map = np.unravel_index(idx, grid_shape)
    cell_to_grid_map = np.column_stack(cell_to_grid_map).astype(np.int16)
    map_path = base_dir / config["mapping"]["index_map_file"]
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(map_path, cell_to_grid_map)
    print(f"  Saved {map_path} shape {cell_to_grid_map.shape}")

    # Compute inlet area
    print("[precompute_mapping] Computing inlet area...")
    inlet_area = compute_inlet_area(mesh_dir)
    with open(base_dir / "inlet_area.txt", "w") as f:
        f.write(f"{inlet_area}\n")
    print(f"  Inlet area: {inlet_area:.6f} m2")

    # Compute and save mask + distance
    print("[precompute_mapping] Computing mask and distance arrays...")
    boundary_csv = base_dir / "cell_data_walls_inlet_outlet_cols.csv"
    if boundary_csv.exists():
        wall_pts, _ = load_boundaries_csv(boundary_csv, tube_layers=50, tube_axis=0)
        spacing = tuple(float(axis[1] - axis[0]) for axis in grid_axes)
        mask, distance = build_mask_from_boundaries(grid_points_flat, spacing, wall_pts, buffer_factor=0.75)
        mask = mask.reshape(grid_shape)
        distance = distance.reshape(grid_shape)
        np.savez(base_dir / "precomputed_masks.npz", mask=mask, distance=distance)
        print(f"  Saved precomputed_masks.npz")
    else:
        print(f"  Warning: {boundary_csv} not found, skipping mask/distance")

    # Find and save inlet cell indices for field processing
    print("[precompute_mapping] Finding inlet cell indices...")
    inlet_cell_indices = find_inlet_cell_indices(mesh_dir)
    np.save(base_dir / "inlet_cell_indices.npy", inlet_cell_indices)
    print(f"  Saved inlet_cell_indices.npy ({len(inlet_cell_indices)} cells)")

    print("[precompute_mapping] Done!")


if __name__ == "__main__":
    main()
