#!/usr/bin/env python3
"""Phase 5: Full coupling loop -- EPANET + ML inference + OpenFOAM.

Main orchestrator:
  1. Loop through EPANET quality timesteps
  2. For each step:
     - Get demand at node 13 from EPANET
     - Convert demand to case_param (inlet velocity)
     - Infer U, nut via UNet
     - Reconstruct fields onto mesh
     - Run OpenFOAM chlorine solver
     - Read outlet concentration
     - Update node 13 quality in EPANET
     - Advance EPANET
"""

import csv
import json
import shutil
import time
from pathlib import Path


def _repo_root() -> Path:
    """Repository root (parent of coupling/)."""
    return Path(__file__).resolve().parents[1]


import h5py
import numpy as np
import torch
from epyt import epanet

from field_processing import (
    apply_gaussian_smoothing,
    clamp_nut,
    compute_inlet_velocity_from_prediction,
    reconstruct_fields_trilinear,
    validate_field,
)
from infer_hydraulics import UNet2D, infer_hydraulics, load_model_and_stats
from infer_pressure import load_pressure_model_and_stats, infer_pressure
from pressure_velocity_correction import pressure_velocity_correction
from reconstruct_fields import reconstruct_fields
from run_openfoam import run_openfoam_case


def demand_to_velocity(demand_gpm: float, inlet_area: float, conversion_factor: float) -> float:
    """Convert demand (GPM) to inlet velocity (m/s)."""
    Q_m3s = demand_gpm * conversion_factor
    velocity = Q_m3s / inlet_area if inlet_area > 0 else 0.0
    return velocity


def load_lookup_cases(h5_path):
    """
    Load all cases from processed_data.h5, returning a list of
    (case_name, velocity_scalar, U_grid, nut_grid) tuples sorted by velocity.
    """
    cases = []
    with h5py.File(h5_path, "r") as h5:
        proc = h5["processed"]
        for case_name in sorted(proc.keys()):
            grp = proc[case_name]
            velocity = float(grp["case_param"][0, 0, 0])
            U_grid = np.array(grp["U"], dtype=np.float32)       # [80,50,50,3]
            nut_grid = np.array(grp["nut"], dtype=np.float32)    # [80,50,50]
            cases.append((case_name, velocity, U_grid, nut_grid))
    cases.sort(key=lambda x: x[1])
    return cases


def find_nearest_case(lookup_cases, target_velocity):
    """
    Find the case whose velocity is closest to target_velocity.
    Returns (case_name, U_grid, nut_grid, case_velocity).
    """
    best_idx = 0
    best_dist = abs(lookup_cases[0][1] - target_velocity)
    for idx, (_, vel, _, _) in enumerate(lookup_cases):
        dist = abs(vel - target_velocity)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    name, vel, U_grid, nut_grid = lookup_cases[best_idx]
    return name, U_grid, nut_grid, vel


def main():
    # Load config
    config_file = _repo_root() / "configs" / "config.json"
    if not config_file.exists():
        config_file = _repo_root() / "config.json"
    with open(config_file) as f:
        config = json.load(f)

    base_dir = _repo_root()
    logs_dir = base_dir / "logs"
    runs_folder_name = config["openfoam"].get("run_folder_name", "runs")
    runs_dir = base_dir / runs_folder_name
    logs_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)

    # Clean up any leftover run directories from previous sessions
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            shutil.rmtree(d)

    # Load inlet area
    inlet_area_file = base_dir / "inlet_area.txt"
    if not inlet_area_file.exists():
        print("[run_workflow] ERROR: inlet_area.txt not found; run precompute_mapping.py first")
        return

    inlet_area = float(inlet_area_file.read_text().strip())
    print(f"[run_workflow] Inlet area: {inlet_area:.6f} m2")

    # Check lookup mode
    use_lookup = config["ml"].get("use_lookup", False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = base_dir / config["ml"]["processed_data_h5"]

    if use_lookup:
        print(f"[run_workflow] LOOKUP MODE enabled -- loading pre-existing CFD cases from {data_path}...")
        lookup_cases = load_lookup_cases(data_path)
        print(f"[run_workflow] Loaded {len(lookup_cases)} lookup cases:")
        for name, vel, _, _ in lookup_cases:
            print(f"    {name}: velocity = {vel:.6f} m/s")
        # Placeholders not used in lookup mode
        model = None
        input_mean = input_std = output_mean = output_std = None
        p_model = pi_mean = pi_std = po_mean = po_std = None
        mask = distance = None
    else:
        lookup_cases = None
        # Load ML model
        print(f"[run_workflow] Loading ML model (device: {device})...")
        model_path = base_dir / config["ml"]["model_weights"]
        model, input_mean, input_std, output_mean, output_std = load_model_and_stats(model_path, data_path, device)

        # Load pressure model
        print(f"[run_workflow] Loading pressure model...")
        pressure_model_path = base_dir / config["ml"]["pressure_model_weights"]
        pressure_data_path = base_dir / config["ml"]["processed_data_pressure_h5"]
        p_model, pi_mean, pi_std, po_mean, po_std = load_pressure_model_and_stats(
            pressure_model_path, pressure_data_path, device
        )

        # Load mask and distance
        masks_path = base_dir / "precomputed_masks.npz"
        if not masks_path.exists():
            print(f"[run_workflow] WARNING: {masks_path} not found; mask/distance will be unavailable")
            mask = np.ones((80, 50, 50), dtype=np.uint8)
            distance = np.zeros((80, 50, 50), dtype=np.float32)
        else:
            with np.load(masks_path) as data:
                mask = data["mask"]
                distance = data["distance"]

    # Load mapping
    map_file = base_dir / config["mapping"]["index_map_file"]
    cell_to_grid_map = np.load(map_file)
    grid_shape = tuple(config["ml"]["grid_shape"])

    # Compute cell centers and grid axes for field processing
    # (needed for trilinear interpolation)
    print(f"[run_workflow] Computing mesh cell centers and grid axes...")
    import sys
    _pp = str(_repo_root() / "preprocess")
    if _pp not in sys.path:
        sys.path.insert(0, _pp)
    from precompute_mapping import compute_cell_centers, build_regular_grid
    mesh_dir = base_dir / config["openfoam"]["base_case_dir"] / "constant" / "polyMesh"
    centers_full = compute_cell_centers(mesh_dir)
    valid_mask = ~np.isnan(centers_full).any(axis=1)
    cell_centers = centers_full[valid_mask]
    grid_points, grid_axes = build_regular_grid(cell_centers, grid_shape)

    if not use_lookup:
        # Compute grid spacing for pressure-velocity correction
        x_axis, y_axis, z_axis = grid_axes
        dx = float(x_axis[1] - x_axis[0])
        dy = float(y_axis[1] - y_axis[0])
        dz = float(z_axis[1] - z_axis[0])
        print(f"[run_workflow] Grid spacing: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")

    # Load EPANET network
    inp_file = base_dir / config["epanet"]["inp_file"]
    target_node_id = config["epanet"]["target_node_id"]
    upstream_node_id = config["epanet"]["upstream_node_id"]

    print(f"[run_workflow] Loading EPANET network from {inp_file}...")
    d = epanet(str(inp_file))
    d.plot_close()

    # Set up manual controls
    d.deleteControls()
    tank_id = "2"
    pump_id = "9"
    tank_index = d.getNodeIndex(tank_id)
    pump_index = d.getLinkIndex(pump_id)
    tank_elevation = d.getNodeElevations(tank_index)
    target_node_index = d.getNodeIndex(target_node_id)
    upstream_node_index = d.getNodeIndex(upstream_node_id)

    below_level = 110
    above_level = 140

    # OpenFOAM parameters
    base_case_dir = base_dir / config["openfoam"]["base_case_dir"]
    local_end_time = config["openfoam"]["local_end_time"]
    solver_command = config["openfoam"]["solver_command"]
    n_procs = config["openfoam"].get("n_procs", 1)
    save_postproc_only = config["openfoam"].get("save_postprocessing_only", True)
    postproc_db_dir = base_dir / config["openfoam"].get("postprocessing_database_dir", "database_postProcessing") if save_postproc_only else None
    conversion_factor = config["units"]["epanet_demand_to_m3s"]
    delta_t = config["openfoam"].get("delta_t", 0.1)
    write_interval = config["openfoam"].get("write_interval", 500)
    duration_hours = config["epanet"].get("duration_hours", 96)

    # Open hydraulic + quality analyses
    d.openHydraulicAnalysis()
    d.initializeHydraulicAnalysis(0)
    d.openQualityAnalysis()
    d.initializeQualityAnalysis(0)

    # SETPOINT source at node 13: lets us override its quality with the OpenFOAM result
    d.setNodeSourceType(target_node_index, 'SETPOINT')
    d.setNodeSourceQuality(target_node_index, 0.5)  # initial value matches .inp

    # CSV logging
    csv_file = logs_dir / "coupling_history.csv"
    csv_f = open(csv_file, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_f,
        fieldnames=[
            "global_time_s",
            "demand_node13_gpm",
            "velocity_m_s",
            "tank_level_ft",
            "pump_status",
            "quality_upstream_epanet",
            "quality_node13_before",
            "quality_node13_after_openfoam",
            "outlet_ccl",
            "outlet_cf",
            "outlet_cs",
            "openfoam_success",
            "openfoam_cumul_time_s",
        ],
    )
    csv_writer.writeheader()

    tstep = 1
    i = 0
    global_time = 0
    prev_run_dir = None
    conversion_factor_gpm_to_m3s = config["units"]["epanet_demand_to_m3s"]

    print("[run_workflow] Starting coupling loop...")
    print("=" * 80)

    workflow_start_time = time.time()
    iteration_times = []

    while tstep > 0:
        iteration_start = time.time()
        # Hydraulic step
        H = d.getNodeHydraulicHead()
        current_tank_head = H[tank_index - 1] - tank_elevation

        # Pump control
        if current_tank_head < below_level:
            d.setLinkStatus(pump_index, 1)
        if current_tank_head > above_level:
            d.setLinkStatus(pump_index, 0)

        t_ret = d.runHydraulicAnalysis()
        global_time = int(t_ret)
        d.runQualityAnalysis()

        # Get actual demand at node 13 (pattern-adjusted, same as quality reads below)
        demand_node13_gpm = float(d.getNodeActualDemand(target_node_index))

        # Convert demand to velocity
        velocity = demand_to_velocity(demand_node13_gpm, inlet_area, conversion_factor_gpm_to_m3s)
        velocity = np.clip(velocity, 0.01, 0.15)  # Clamp to training range

        # Read quality dynamically from EPANET
        quality_upstream = float(d.getNodeActualQuality(upstream_node_index))
        quality_node13_before = float(d.getNodeActualQuality(target_node_index))

        print(
            f"\n[run_workflow] ITERATION {i} / TIME {global_time}s / DEMAND {demand_node13_gpm:.2f} GPM / VELOCITY {velocity:.4f} m/s"
        )
        print("-" * 80)

        infer_time = 0.0

        if use_lookup:
            # --- LOOKUP MODE: use nearest pre-existing CFD case ---
            lookup_start = time.time()
            case_name, U_grid, nut_grid, case_vel = find_nearest_case(lookup_cases, velocity)
            vel_diff = abs(velocity - case_vel)
            print(f"  [lookup] Selected {case_name} (velocity={case_vel:.6f} m/s, target={velocity:.6f} m/s, diff={vel_diff:.6f} m/s)")
            infer_time = time.time() - lookup_start

            # Trilinear interpolation to mesh (still needed for grid→mesh mapping)
            recon_start = time.time()
            print(f"  [reconstruct] Mapping fields to mesh with trilinear interpolation...")
            U_mesh, nut_mesh = reconstruct_fields_trilinear(
                grid_shape, cell_centers, grid_axes, U_grid, nut_grid
            )
            recon_time = time.time() - recon_start
            print(f"  [reconstruct] Completed in {recon_time:.2f}s")

            # Enforce minimum nut on mesh to prevent zero diffusivity in solver
            nut_floor = 1e-7
            n_below_floor = np.sum(nut_mesh < nut_floor)
            if n_below_floor > 0:
                nut_mesh = np.clip(nut_mesh, nut_floor, None)
                print(f"  [field_process] Raised {n_below_floor} nut_mesh cells to floor {nut_floor}")
        else:
            # --- INFERENCE MODE: full ML pipeline ---
            infer_start = time.time()
            print(f"  [infer] Running velocity UNet inference...")
            U_grid, nut_grid = infer_hydraulics(
                velocity, model, mask, distance, input_mean, input_std, output_mean, output_std, device
            )
            infer_time = time.time() - infer_start
            print(f"  [infer] Velocity inference completed in {infer_time:.2f}s")

            # P0: Validate and fix any NaN/Inf from inference
            print(f"  [field_process] Validating inferred fields...")
            U_grid = validate_field(U_grid, "U_grid")
            nut_grid = validate_field(nut_grid, "nut_grid")

            # P0: Clamp turbulent viscosity to physical range
            print(f"  [field_process] Clamping turbulent viscosity...")
            nut_grid = clamp_nut(nut_grid, min_nut=1e-8, max_nut=1e-3)

            # P1: Gaussian smoothing to remove nearest-neighbor artifacts (reduced intensity)
            print(f"  [field_process] Applying Gaussian smoothing...")
            U_grid, nut_grid = apply_gaussian_smoothing(U_grid, nut_grid, sigma=0.5, device=device)

            # P1: Pressure inference + GPU iterative correction
            print(f"  [infer] Running pressure UNet inference...")
            p_infer_start = time.time()
            p_grid = infer_pressure(
                U_grid, mask, distance, velocity,
                p_model, pi_mean, pi_std, po_mean, po_std, device
            )
            print(f"  [infer] Pressure inference completed in {time.time() - p_infer_start:.2f}s")

            print(f"  [field_process] Running GPU pressure-velocity correction...")
            corr_start = time.time()
            U_grid, _, _ = pressure_velocity_correction(
                U_grid, p_grid, mask, dx, dy, dz,
                n_outer=15, alpha=0.3, device=device,
            )
            print(f"  [field_process] GPU correction completed in {time.time() - corr_start:.2f}s")

            # P1: Validate fields after processing
            print(f"  [field_process] Validating processed fields...")
            U_grid = validate_field(U_grid, "U_grid_processed")
            nut_grid = validate_field(nut_grid, "nut_grid_processed")

            # P2: Trilinear interpolation to mesh
            recon_start = time.time()
            print(f"  [reconstruct] Mapping fields to mesh with trilinear interpolation...")
            U_mesh, nut_mesh = reconstruct_fields_trilinear(
                grid_shape, cell_centers, grid_axes, U_grid, nut_grid
            )
            recon_time = time.time() - recon_start
            print(f"  [reconstruct] Completed in {recon_time:.2f}s")

            # Validate mesh fields before passing to OpenFOAM
            print(f"  [field_process] Validating mesh fields...")
            U_mesh = validate_field(U_mesh, "U_mesh")
            nut_mesh = validate_field(nut_mesh, "nut_mesh")

            # Enforce minimum nut on mesh to prevent zero diffusivity in solver
            nut_floor = 1e-7
            n_below_floor = np.sum(nut_mesh < nut_floor)
            if n_below_floor > 0:
                nut_mesh = np.clip(nut_mesh, nut_floor, None)
                print(f"  [field_process] Raised {n_below_floor} nut_mesh cells to floor {nut_floor}")

        # Extract inlet velocity from field (common to both modes)
        print(f"  [field_process] Computing inlet boundary condition...")
        inlet_cell_indices_file = base_dir / "inlet_cell_indices.npy"
        if inlet_cell_indices_file.exists():
            inlet_cell_indices = np.load(inlet_cell_indices_file)
            inlet_velocity = compute_inlet_velocity_from_prediction(U_mesh, inlet_cell_indices)
        else:
            print(f"  [field_process] WARNING: inlet_cell_indices.npy not found, using zero velocity")
            inlet_velocity = (0.0, 0.0, 0.0)

        # Run OpenFOAM
        run_num = i
        run_dir = runs_dir / f"run_{run_num:04d}"
        openfoam_start = time.time()
        print(f"  [openfoam] Running solver in {run_dir}...")

        success, outlet_vals, outlet_area, prev_species_dir = run_openfoam_case(
            base_case_dir,
            run_dir,
            U_mesh,
            nut_mesh,
            ccl_inlet=(quality_upstream * 1e-3) if quality_upstream is not None else 0.5e-3,
            cf_inlet=0.000336,
            cs_inlet=0.001344,
            start_time=0.0,
            end_time=float(local_end_time),
            prev_run_dir=prev_run_dir,
            solver_command=solver_command,
            n_procs=n_procs,
            save_postprocessing_only=save_postproc_only,
            postprocessing_db_dir=postproc_db_dir,
            delta_t=delta_t,
            write_interval=write_interval,
            inlet_velocity=inlet_velocity,
        )
        openfoam_time = time.time() - openfoam_start
        print(f"  [openfoam] Completed in {openfoam_time:.2f}s" if success else f"  [openfoam] FAILED after {openfoam_time:.2f}s")

        outlet_ccl = None
        outlet_cf = None
        outlet_cs = None
        if success and outlet_vals:
            outlet_ccl, outlet_cf, outlet_cs = outlet_vals
            # Convert OpenFOAM kg/m³ → EPANET mg/L
            quality_node13_after = outlet_ccl * 1e3
            d.setNodeSourceQuality(target_node_index, outlet_ccl * 1e3)
        else:
            quality_node13_after = quality_node13_before

        # Get tank level and pump status for dashboard
        tank_level_ft = float(H[tank_index - 1] - tank_elevation)
        pump_status = int(d.getLinkStatus(pump_index))

        # Log
        csv_writer.writerow(
            {
                "global_time_s": global_time,
                "demand_node13_gpm": demand_node13_gpm,
                "velocity_m_s": velocity,
                "tank_level_ft": tank_level_ft,
                "pump_status": pump_status,
                "quality_upstream_epanet": quality_upstream,
                "quality_node13_before": quality_node13_before,
                "quality_node13_after_openfoam": quality_node13_after,
                "outlet_ccl": outlet_ccl,
                "outlet_cf": outlet_cf,
                "outlet_cs": outlet_cs,
                "openfoam_success": success,
                "openfoam_cumul_time_s": (i + 1) * local_end_time,
            }
        )
        csv_f.flush()

        # Summary for this iteration
        iteration_total = time.time() - iteration_start
        iteration_times.append(iteration_total)
        mode_label = "lookup" if use_lookup else "infer"
        print(f"  [summary] Iteration {i} total: {iteration_total:.2f}s ({mode_label}: {infer_time:.2f}s, recon: {recon_time:.2f}s, openfoam: {openfoam_time:.2f}s)")
        cumul_of_time = (i + 1) * local_end_time
        progress_pct = global_time / (duration_hours * 3600) * 100
        print(f"  [time] EPANET hydraulic: {global_time}s ({global_time/3600:.1f} h)  |  OpenFOAM transport: {cumul_of_time}s ({cumul_of_time/3600:.1f} h)  |  Progress: {progress_pct:.1f} %")
        print("=" * 80)

        prev_run_dir = prev_species_dir if save_postproc_only else run_dir
        i += 1
        tstep = d.nextHydraulicAnalysisStep()
        d.nextQualityAnalysisStep()

        # Limit iterations for testing
        #if i >= 5:
        #    print("[run_workflow] Stopping after 5 iterations (test mode)")
        #    break

    d.closeQualityAnalysis()
    d.closeHydraulicAnalysis()
    csv_f.close()
    d.unload()

    # Final statistics
    total_time = time.time() - workflow_start_time
    avg_time = np.mean(iteration_times) if iteration_times else 0
    print("\n" + "=" * 80)
    print(f"[run_workflow] WORKFLOW COMPLETE")
    print(f"  Total iterations: {i}")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Average per iteration: {avg_time:.2f}s")
    print(f"  PostProcessing database: {postproc_db_dir if postproc_db_dir else 'N/A'}")
    print(f"  Logged to: {csv_file}")
    cumul_of_time = i * local_end_time
    pct = global_time / (duration_hours * 3600) * 100 if duration_hours > 0 else 0.0
    ratio = cumul_of_time / global_time if global_time > 0 else 0.0
    print(f"  EPANET hydraulic time:  {global_time} s  ({global_time/3600:.1f} h)  /  {duration_hours} h  ({pct:.1f} %)")
    print(f"  OpenFOAM transport time (cumul): {cumul_of_time} s  ({cumul_of_time/3600:.1f} h)")
    print(f"  Transport / hydraulic ratio: {ratio:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
