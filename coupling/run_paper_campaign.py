#!/usr/bin/env python3
"""EPANET warmup then ML + OpenFOAM coupling campaign.

Run from the repository root:
  python3 coupling/run_paper_campaign.py
  python3 coupling/run_paper_campaign.py --warmup-only
  python3 coupling/run_paper_campaign.py --coupled-hours 96
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from epyt import epanet

# Package paths: coupling/ on sys.path, repo root for data/configs
COUPLING_DIR = Path(__file__).resolve().parent
ROOT = COUPLING_DIR.parent
if str(COUPLING_DIR) not in sys.path:
    sys.path.insert(0, str(COUPLING_DIR))
PAPER_RUNS = ROOT / "outputs" / "campaign"
PAPER_RUNS.mkdir(parents=True, exist_ok=True)

from field_processing import (  # noqa: E402
    apply_gaussian_smoothing,
    clamp_nut,
    compute_inlet_velocity_from_prediction,
    reconstruct_fields_trilinear,
    validate_field,
)
from infer_hydraulics import infer_hydraulics, load_model_and_stats  # noqa: E402
from infer_pressure import infer_pressure, load_pressure_model_and_stats  # noqa: E402
from pressure_velocity_correction import pressure_velocity_correction  # noqa: E402
from run_openfoam import run_openfoam_case  # noqa: E402
from run_workflow import demand_to_velocity  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ml_openfoam_step(
    *,
    velocity: float,
    quality_upstream: float,
    model,
    p_model,
    mask,
    distance,
    input_mean,
    input_std,
    output_mean,
    output_std,
    pi_mean,
    pi_std,
    po_mean,
    po_std,
    device: str,
    grid_shape,
    cell_centers,
    grid_axes,
    dx: float,
    dy: float,
    dz: float,
    base_case_dir: Path,
    run_dir: Path,
    prev_run_dir,
    solver_command: str,
    n_procs: int,
    local_end_time: float,
    delta_t: float,
    write_interval: int,
    save_postproc_only: bool,
    postproc_db_dir,
    inlet_cell_indices,
    cf_inlet: float = 0.000336,
    cs_inlet: float = 0.001344,
    reset_species_ic: bool = True,
) -> tuple[bool, float | None, float | None, float | None, Path | None, dict]:
    """One ML + OpenFOAM iteration. Returns success, CCl/Cf/Cs (kg/m3), prev_species, timing dict."""
    timings: dict[str, float] = {}

    t0 = time.time()
    U_grid, nut_grid = infer_hydraulics(
        velocity, model, mask, distance, input_mean, input_std, output_mean, output_std, device
    )
    U_grid = validate_field(U_grid, "U_grid")
    nut_grid = validate_field(nut_grid, "nut_grid")
    nut_grid = clamp_nut(nut_grid, min_nut=1e-8, max_nut=1e-3)
    U_grid, nut_grid = apply_gaussian_smoothing(U_grid, nut_grid, sigma=0.5, device=device)
    timings["ml_vel_s"] = time.time() - t0

    t0 = time.time()
    p_grid = infer_pressure(
        U_grid, mask, distance, velocity, p_model, pi_mean, pi_std, po_mean, po_std, device
    )
    U_grid, _, _ = pressure_velocity_correction(
        U_grid, p_grid, mask, dx, dy, dz, n_outer=15, alpha=0.3, device=device
    )
    U_grid = validate_field(U_grid, "U_grid_processed")
    nut_grid = validate_field(nut_grid, "nut_grid_processed")
    timings["ml_p_corr_s"] = time.time() - t0

    t0 = time.time()
    U_mesh, nut_mesh = reconstruct_fields_trilinear(
        grid_shape, cell_centers, grid_axes, U_grid, nut_grid
    )
    U_mesh = validate_field(U_mesh, "U_mesh")
    nut_mesh = validate_field(nut_mesh, "nut_mesh")
    nut_mesh = np.clip(nut_mesh, 1e-7, None)
    if inlet_cell_indices is not None:
        inlet_velocity = compute_inlet_velocity_from_prediction(U_mesh, inlet_cell_indices)
    else:
        inlet_velocity = (0.0, 0.0, 0.0)
    timings["recon_s"] = time.time() - t0

    t0 = time.time()
    success, outlet_vals, _, prev_species_dir = run_openfoam_case(
        base_case_dir,
        run_dir,
        U_mesh,
        nut_mesh,
        ccl_inlet=(quality_upstream * 1e-3) if quality_upstream is not None else 0.5e-3,
        cf_inlet=cf_inlet,
        cs_inlet=cs_inlet,
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
        reset_species_ic=reset_species_ic,
    )
    timings["openfoam_s"] = time.time() - t0
    timings["ml_total_s"] = timings["ml_vel_s"] + timings["ml_p_corr_s"] + timings["recon_s"]

    outlet_ccl = outlet_cf = outlet_cs = None
    if success and outlet_vals:
        outlet_ccl, outlet_cf, outlet_cs = outlet_vals
    return success, outlet_ccl, outlet_cf, outlet_cs, prev_species_dir, timings


def run_campaign(
    cfg: dict,
    *,
    warmup_only: bool = False,
    coupled_only: bool = False,
    coupled_hours_override: int | None = None,
) -> Path:
    out_dir = PAPER_RUNS
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup_h = int(cfg["warmup_hours"])
    coupled_h = int(coupled_hours_override if coupled_hours_override is not None else cfg["coupled_hours"])
    if warmup_only:
        coupled_h = 0
    if coupled_only:
        warmup_h = 0

    total_h = warmup_h + coupled_h
    warmup_s = warmup_h * 3600
    total_s = total_h * 3600

    nodes_log = list(cfg.get("nodes_to_log", ["12", "13", "2"]))
    links_log = list(cfg.get("links_to_log", ["9", "12"]))

    # --- ML / mesh (only if coupled phase exists) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = p_model = None
    mask = distance = None
    input_mean = input_std = output_mean = output_std = None
    pi_mean = pi_std = po_mean = po_std = None
    cell_centers = grid_axes = None
    dx = dy = dz = 0.0
    inlet_cell_indices = None
    grid_shape = tuple(cfg["ml"]["grid_shape"])

    runs_dir = ROOT / cfg["openfoam"]["run_folder_name"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    postproc_db = ROOT / cfg["openfoam"]["postprocessing_database_dir"]
    postproc_db.mkdir(parents=True, exist_ok=True)

    if coupled_h > 0:
        print(f"[paper] Loading ML models on {device}...")
        data_path = ROOT / cfg["ml"]["processed_data_h5"]
        model, input_mean, input_std, output_mean, output_std = load_model_and_stats(
            ROOT / cfg["ml"]["model_weights"], data_path, device
        )
        p_model, pi_mean, pi_std, po_mean, po_std = load_pressure_model_and_stats(
            ROOT / cfg["ml"]["pressure_model_weights"],
            ROOT / cfg["ml"]["processed_data_pressure_h5"],
            device,
        )
        masks_path = ROOT / "precomputed_masks.npz"
        with np.load(masks_path) as data:
            mask = data["mask"]
            distance = data["distance"]

        import sys
        _pp = str(ROOT / "preprocess")
        if _pp not in sys.path:
            sys.path.insert(0, _pp)
        from precompute_mapping import build_regular_grid, compute_cell_centers

        mesh_dir = ROOT / cfg["openfoam"]["base_case_dir"] / "constant" / "polyMesh"
        centers_full = compute_cell_centers(mesh_dir)
        valid = ~np.isnan(centers_full).any(axis=1)
        cell_centers = centers_full[valid]
        _, grid_axes = build_regular_grid(cell_centers, grid_shape)
        x_axis, y_axis, z_axis = grid_axes
        dx = float(x_axis[1] - x_axis[0])
        dy = float(y_axis[1] - y_axis[0])
        dz = float(z_axis[1] - z_axis[0])

        ici = ROOT / "inlet_cell_indices.npy"
        inlet_cell_indices = np.load(ici) if ici.exists() else None

        inlet_area = float((ROOT / "inlet_area.txt").read_text().strip())
    else:
        inlet_area = float((ROOT / "inlet_area.txt").read_text().strip()) if (ROOT / "inlet_area.txt").exists() else 1.0

    conversion = cfg["units"]["epanet_demand_to_m3s"]
    base_case_dir = ROOT / cfg["openfoam"]["base_case_dir"]
    solver = cfg["openfoam"]["solver_command"]
    if not Path(solver).exists():
        solver = cfg["openfoam"]["solver_command_fallback"]
    n_procs = int(cfg["openfoam"]["n_procs"])
    local_end_time = float(cfg["openfoam"]["local_end_time"])
    delta_t = float(cfg["openfoam"]["delta_t"])
    write_interval = int(cfg["openfoam"]["write_interval"])
    save_postproc_only = bool(cfg["openfoam"].get("save_postprocessing_only", True))
    cf_inlet = float(cfg["openfoam"].get("cf_inlet", 0.000336))
    cs_inlet = float(cfg["openfoam"].get("cs_inlet", 0.001344))
    reset_species_ic = bool(cfg["openfoam"].get("reset_species_ic", True))
    if coupled_h > 0:
        print(
            f"[paper] OpenFOAM organics cf={cf_inlet:.6e} cs={cs_inlet:.6e} "
            f"reset_species_ic={reset_species_ic}"
        )

    # Clean previous OF runs for this campaign
    if coupled_h > 0:
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                shutil.rmtree(d)

    # --- EPANET ---
    inp = ROOT / cfg["epanet"]["inp_file"]
    target_id = cfg["epanet"]["target_node_id"]
    upstream_id = cfg["epanet"]["upstream_node_id"]

    print(f"[paper] EPANET {inp.name}: warmup={warmup_h} h, coupled={coupled_h} h, total={total_h} h")
    d = epanet(str(inp))
    d.plot_close()
    d.deleteControls()
    d.setTimeSimulationDuration(int(total_s))

    tank_id, pump_id = "2", "9"
    tank_index = d.getNodeIndex(tank_id)
    pump_index = d.getLinkIndex(pump_id)
    tank_elevation = d.getNodeElevations(tank_index)
    target_index = d.getNodeIndex(target_id)
    upstream_index = d.getNodeIndex(upstream_id)

    node_indices = {nid: d.getNodeIndex(nid) for nid in nodes_log}
    link_indices = {lid: d.getLinkIndex(lid) for lid in links_log}

    below_level, above_level = 110.0, 140.0

    d.openHydraulicAnalysis()
    d.initializeHydraulicAnalysis(0)
    d.openQualityAnalysis()
    d.initializeQualityAnalysis(0)

    # Keep native EPANET quality during warmup; switch node 13 to SETPOINT on first coupled step
    setpoint_armed = False

    fieldnames = [
        "i",
        "phase",
        "global_time_s",
        "global_time_h",
        "demand_node13_gpm",
        "velocity_m_s",
        "tank_level_ft",
        "pump_status",
        "quality_upstream",
        "quality_node13_epanet",
        "quality_node13_after",
        "outlet_ccl_kg_m3",
        "outlet_cf_kg_m3",
        "outlet_cs_kg_m3",
        "openfoam_success",
        "wall_s",
        "epanet_s",
        "ml_total_s",
        "ml_vel_s",
        "ml_p_corr_s",
        "recon_s",
        "openfoam_s",
    ]
    for nid in nodes_log:
        fieldnames += [f"q_{nid}", f"demand_{nid}", f"head_{nid}"]
    for lid in links_log:
        fieldnames += [f"flow_{lid}", f"status_{lid}"]

    csv_path = out_dir / "campaign_history.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()

    tstep = 1
    i = 0
    global_time = 0
    prev_run_dir = None
    coupled_run_idx = 0
    campaign_t0 = time.time()

    print("[paper] Starting campaign loop...")
    while tstep > 0:
        iter_t0 = time.time()
        epanet_t0 = time.time()

        H = d.getNodeHydraulicHead()
        tank_level = float(H[tank_index - 1] - tank_elevation)
        if tank_level < below_level:
            d.setLinkStatus(pump_index, 1)
        if tank_level > above_level:
            d.setLinkStatus(pump_index, 0)

        t_ret = d.runHydraulicAnalysis()
        global_time = int(t_ret)
        d.runQualityAnalysis()

        demand13 = float(d.getNodeActualDemand(target_index))
        q_up = float(d.getNodeActualQuality(upstream_index))
        q13 = float(d.getNodeActualQuality(target_index))
        pump_status = int(d.getLinkStatus(pump_index))

        velocity = float(np.clip(demand_to_velocity(demand13, inlet_area, conversion), 0.01, 0.15))

        phase = "warmup" if global_time < warmup_s else "coupled"
        if coupled_only:
            phase = "coupled"
        if warmup_only:
            phase = "warmup"

        epanet_s = time.time() - epanet_t0

        row = {
            "i": i,
            "phase": phase,
            "global_time_s": global_time,
            "global_time_h": global_time / 3600.0,
            "demand_node13_gpm": demand13,
            "velocity_m_s": velocity,
            "tank_level_ft": tank_level,
            "pump_status": pump_status,
            "quality_upstream": q_up,
            "quality_node13_epanet": q13,
            "quality_node13_after": q13,
            "outlet_ccl_kg_m3": "",
            "outlet_cf_kg_m3": "",
            "outlet_cs_kg_m3": "",
            "openfoam_success": "",
            "wall_s": 0.0,
            "epanet_s": epanet_s,
            "ml_total_s": "",
            "ml_vel_s": "",
            "ml_p_corr_s": "",
            "recon_s": "",
            "openfoam_s": "",
        }

        # Extra node / link sensors
        demands = d.getNodeActualDemand()
        quals = d.getNodeActualQuality()
        heads = d.getNodeHydraulicHead()
        for nid, idx in node_indices.items():
            row[f"q_{nid}"] = float(quals[idx - 1])
            row[f"demand_{nid}"] = float(demands[idx - 1])
            row[f"head_{nid}"] = float(heads[idx - 1])
        flows = d.getLinkFlows()
        statuses = d.getLinkStatus()
        for lid, idx in link_indices.items():
            row[f"flow_{lid}"] = float(flows[idx - 1])
            row[f"status_{lid}"] = int(statuses[idx - 1])

        if phase == "coupled" and coupled_h > 0:
            if not setpoint_armed:
                d.setNodeSourceType(target_index, "SETPOINT")
                setpoint_armed = True
                print("[paper] Armed SETPOINT source at node 13 for OpenFOAM feedback")

            run_dir = runs_dir / f"run_{coupled_run_idx:04d}"
            print(
                f"\n[paper] COUPLED i={i} t={global_time/3600:.1f}h "
                f"demand={demand13:.2f} GPM v={velocity:.4f} m/s"
            )
            success, ccl, cf, cs, prev_species, tm = ml_openfoam_step(
                velocity=velocity,
                quality_upstream=q_up,
                model=model,
                p_model=p_model,
                mask=mask,
                distance=distance,
                input_mean=input_mean,
                input_std=input_std,
                output_mean=output_mean,
                output_std=output_std,
                pi_mean=pi_mean,
                pi_std=pi_std,
                po_mean=po_mean,
                po_std=po_std,
                device=device,
                grid_shape=grid_shape,
                cell_centers=cell_centers,
                grid_axes=grid_axes,
                dx=dx,
                dy=dy,
                dz=dz,
                base_case_dir=base_case_dir,
                run_dir=run_dir,
                prev_run_dir=prev_run_dir,
                solver_command=solver,
                n_procs=n_procs,
                local_end_time=local_end_time,
                delta_t=delta_t,
                write_interval=write_interval,
                save_postproc_only=save_postproc_only,
                postproc_db_dir=postproc_db,
                inlet_cell_indices=inlet_cell_indices,
                cf_inlet=cf_inlet,
                cs_inlet=cs_inlet,
                reset_species_ic=reset_species_ic,
            )
            row["openfoam_success"] = int(bool(success))
            row["ml_total_s"] = tm.get("ml_total_s", "")
            row["ml_vel_s"] = tm.get("ml_vel_s", "")
            row["ml_p_corr_s"] = tm.get("ml_p_corr_s", "")
            row["recon_s"] = tm.get("recon_s", "")
            row["openfoam_s"] = tm.get("openfoam_s", "")
            if success and ccl is not None:
                row["outlet_ccl_kg_m3"] = ccl
                row["outlet_cf_kg_m3"] = cf
                row["outlet_cs_kg_m3"] = cs
                q_after = float(ccl) * 1e3  # kg/m3 → mg/L
                d.setNodeSourceQuality(target_index, q_after)
                row["quality_node13_after"] = q_after
                row[f"q_{target_id}"] = q_after
            prev_run_dir = prev_species if save_postproc_only else run_dir
            coupled_run_idx += 1
            print(
                f"  [paper] OF={'OK' if success else 'FAIL'} "
                f"ml={tm.get('ml_total_s', 0):.2f}s of={tm.get('openfoam_s', 0):.2f}s"
            )
        else:
            if i % 24 == 0 or global_time == 0:
                print(
                    f"[paper] WARMUP i={i} t={global_time/3600:.1f}h "
                    f"tank={tank_level:.1f} ft q13={q13:.3f} mg/L"
                )

        row["wall_s"] = time.time() - iter_t0
        writer.writerow(row)
        csv_f.flush()

        i += 1
        tstep = d.nextHydraulicAnalysisStep()
        d.nextQualityAnalysisStep()

        if global_time >= total_s and tstep > 0:
            # Safety stop if duration slightly overshoots
            break

    d.closeQualityAnalysis()
    d.closeHydraulicAnalysis()
    csv_f.close()
    d.unload()

    elapsed = time.time() - campaign_t0
    print(f"\n[paper] Campaign finished in {elapsed/60:.1f} min → {csv_path}")

    # Timing summary JSON
    import pandas as pd

    df = pd.read_csv(csv_path)
    summary = {
        "warmup_hours": warmup_h,
        "coupled_hours": coupled_h,
        "n_steps": int(len(df)),
        "wall_total_s": float(elapsed),
        "n_procs_openfoam": n_procs if coupled_h > 0 else None,
        "openfoam_cf_inlet": cf_inlet if coupled_h > 0 else None,
        "openfoam_cs_inlet": cs_inlet if coupled_h > 0 else None,
        "reset_species_ic": reset_species_ic if coupled_h > 0 else None,
        "phases": {},
    }
    for phase, g in df.groupby("phase"):
        summary["phases"][phase] = {
            "n_steps": int(len(g)),
            "wall_s_mean": float(g["wall_s"].mean()),
            "wall_s_std": float(g["wall_s"].std(ddof=1)) if len(g) > 1 else 0.0,
            "epanet_s_mean": float(g["epanet_s"].mean()),
        }
        if phase == "coupled":
            for col in ["ml_total_s", "openfoam_s"]:
                s = pd.to_numeric(g[col], errors="coerce").dropna()
                if len(s):
                    summary["phases"][phase][f"{col}_mean"] = float(s.mean())
                    summary["phases"][phase][f"{col}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0

    summary_path = out_dir / "timings_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[paper] Wrote {summary_path}")
    return csv_path


def main():
    p = argparse.ArgumentParser(description="Paper campaign: warmup then ML+OpenFOAM")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "config_paper.json")
    p.add_argument("--warmup-only", action="store_true")
    p.add_argument("--coupled-only", action="store_true")
    p.add_argument("--coupled-hours", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    csv_path = run_campaign(
        cfg,
        warmup_only=args.warmup_only,
        coupled_only=args.coupled_only,
        coupled_hours_override=args.coupled_hours,
    )
    # Auto-plot if campaign produced data
    try:
        from plot_paper_results import make_all_plots

        make_all_plots(csv_path.parent)
    except Exception as exc:
        print(f"[paper] Plotting deferred ({exc}); run plot_paper_results.py later")


if __name__ == "__main__":
    main()
