#!/usr/bin/env python3
"""Phase 4: OpenFOAM case management -- copy, patch, run, extract output.

Handles:
  - Copy base case to run directory
  - Update U, nut, CCl, Cf, Cs fields
  - Update controlDict (startTime, endTime)
  - Run solver
  - Parse postprocessing output
"""

import json
import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    """Repository root (parent of coupling/)."""
    return Path(__file__).resolve().parents[1]

from typing import Optional, Tuple

import numpy as np


def update_openfoam_controldict(controldict_path: Path, start_time: float, end_time: float, delta_t: float = 0.1, write_interval: int = 500):
    """Update startTime, endTime, deltaT, writeInterval, and startFrom in controlDict."""
    lines = controldict_path.read_text().splitlines()
    updated = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("startFrom") and not stripped.startswith("//"):
            updated.append("startFrom\tstartTime;")
        elif stripped.startswith("startTime") and not stripped.startswith("//"):
            updated.append(f"startTime\t{start_time};")
        elif stripped.startswith("stopAt") and not stripped.startswith("//"):
            updated.append("stopAt\tendTime;")
        elif stripped.startswith("endTime") and not stripped.startswith("//"):
            updated.append(f"endTime\t{end_time};")
        elif stripped.startswith("deltaT") and not stripped.startswith("//"):
            updated.append(f"deltaT\t{delta_t};")
        elif stripped.startswith("writeInterval") and not stripped.startswith("//"):
            updated.append(f"writeInterval\t{write_interval};")
        else:
            updated.append(line)
    controldict_path.write_text("\n".join(updated) + "\n")


def update_decompose_par_dict(decompose_path: Path, n_procs: int):
    """Overwrite numberOfSubdomains so decomposePar matches the mpirun -np value."""
    lines = decompose_path.read_text().splitlines()
    updated = []
    for line in lines:
        if line.strip().startswith("numberOfSubdomains"):
            updated.append(f"numberOfSubdomains\t{n_procs};")
        else:
            updated.append(line)
    decompose_path.write_text("\n".join(updated) + "\n")


def stabilize_solver_settings(run_dir: Path):
    """Patch fvSchemes and ASM1fvSolution for numerical stability.

    Changes species transport from linearUpwind (unbounded, can produce negative
    concentrations) to upwind (first-order but unconditionally stable). Also
    tightens linear solver tolerances and adds maxIter.

    Keeps Euler time scheme (backward/BDF2 requires two time levels and crashes
    at t=0 when starting from a single initial condition).

    This prevents SIGFPE crashes caused by negative concentrations interacting
    with high reaction rates (kf=2890).
    """
    # Patch fvSchemes: switch species from linearUpwind to upwind
    fv_schemes = run_dir / "system" / "fvSchemes"
    if fv_schemes.exists():
        content = fv_schemes.read_text()
        # Replace the species convection scheme
        content = content.replace(
            "CCl       linearUpwind grad(CCl);",
            "CCl       upwind;"
        )
        content = content.replace(
            "Cf       linearUpwind grad(Cf);",
            "Cf       upwind;"
        )
        content = content.replace(
            "Cf        linearUpwind grad(Cf);",
            "Cf        upwind;"
        )
        content = content.replace(
            "Cs       linearUpwind grad(Cs);",
            "Cs       upwind;"
        )
        content = content.replace(
            "Cs        linearUpwind grad(Cs);",
            "Cs        upwind;"
        )
        # Keep Euler time scheme — do NOT switch to backward.
        # Backward (BDF2) requires two time levels; starting from a single
        # initial condition (t=0 only) causes immediate SIGFPE at the first
        # timestep because the solver cannot compute the BDF2 coefficients.
        fv_schemes.write_text(content)
        print(f"[run_openfoam] Patched fvSchemes: species div -> upwind, ddt kept as Euler")

    # Patch ASM1fvSolution: tighter tolerances, add maxIter
    fv_solution = run_dir / "system" / "ASM1fvSolution"
    if fv_solution.exists():
        new_content = """{
CCl
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-06;
        relTol          0;
        maxIter         1000;
     }
Cf
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-06;
        relTol          0;
        maxIter         1000;
     }
Cs
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-06;
        relTol          0;
        maxIter         1000;
     }

}
"""
        fv_solution.write_text(new_content)
        print(f"[run_openfoam] Patched ASM1fvSolution: tighter tolerances, maxIter=1000")


def update_inlet_bc_scalar(
    field_path: Path, inlet_value: float, label: str
):
    """Update inlet boundary condition value for a scalar field."""
    lines = field_path.read_text().splitlines()
    updated = []
    in_inlet = False
    for i, line in enumerate(lines):
        if "Inlet" in line:
            in_inlet = True
        if in_inlet and "value" in line and "uniform" in line:
            updated.append(f"        value           uniform {inlet_value:.16e};")
            in_inlet = False
        else:
            updated.append(line)
    field_path.write_text("\n".join(updated) + "\n")


def set_uniform_internal_field(field_path: Path, value: float) -> bool:
    """Rewrite uniform internalField (used to clear Hydrodeca leftover ICs)."""
    import re

    text = field_path.read_text()
    text2, n = re.subn(
        r"(internalField\s+uniform\s+)([0-9.eE+-]+)",
        rf"\g<1>{value:.16e}",
        text,
        count=1,
    )
    if n:
        field_path.write_text(text2)
    return n > 0


def parse_postprocessing_output(run_dir: Path) -> Optional[Tuple[float, float, float]]:
    """
    Parse the postprocessing output to extract CCl, Cf, Cs values at Outlet.

    Returns:
      (ccl_integrated, cf_integrated, cs_integrated) or None if output not found
    """
    # Look for postProcessing/Outlet01Integ_ASM1/*/functionObjectProperties
    postproc_dir = run_dir / "postProcessing" / "Outlet01Integ_ASM1"
    if not postproc_dir.exists():
        return None

    # Find the latest timestep directory
    timestep_dirs = sorted(
        [d for d in postproc_dir.iterdir() if d.is_dir() and d.name.replace(".", "").replace("-", "").isdigit()]
    )
    if not timestep_dirs:
        return None

    latest_dir = timestep_dirs[-1]
    output_file = latest_dir / "surfaceFieldValue.dat"

    if not output_file.exists():
        return None

    # Parse the output file
    # Detect column order from header: "areaIntegrate(CCl) areaIntegrate(Cs) areaIntegrate(Cf)"
    lines = output_file.read_text().splitlines()
    if len(lines) < 2:
        return None

    # Parse header to determine column mapping
    header_line = ""
    for line in lines:
        if line.strip().startswith("# Time"):
            header_line = line
            break

    col_map = {"CCl": 2, "Cf": 3, "Cs": 4}  # default fallback
    if header_line:
        headers = header_line.split("\t")
        for idx, h in enumerate(headers):
            h = h.strip()
            if "areaIntegrate(CCl)" in h:
                col_map["CCl"] = idx
            elif "areaIntegrate(Cf)" in h:
                col_map["Cf"] = idx
            elif "areaIntegrate(Cs)" in h:
                col_map["Cs"] = idx

    # Last line should have the latest values
    last_line = lines[-1].strip()
    parts = last_line.split()

    max_col = max(col_map.values())
    if len(parts) > max_col:
        ccl = float(parts[col_map["CCl"]])
        cf = float(parts[col_map["Cf"]])
        cs = float(parts[col_map["Cs"]])
        return (ccl, cf, cs)

    return None


def get_outlet_area(run_dir: Path) -> Optional[float]:
    """Extract outlet area from postprocessing output."""
    postproc_dir = run_dir / "postProcessing" / "Outlet01Integ_ASM1"
    if not postproc_dir.exists():
        return None

    timestep_dirs = sorted(
        [d for d in postproc_dir.iterdir() if d.is_dir() and d.name.replace(".", "").replace("-", "").isdigit()]
    )
    if not timestep_dirs:
        return None

    latest_dir = timestep_dirs[-1]
    output_file = latest_dir / "surfaceFieldValue.dat"

    if not output_file.exists():
        return None

    lines = output_file.read_text().splitlines()
    if len(lines) < 2:
        return None

    last_line = lines[-1].strip()
    parts = last_line.split()

    if len(parts) >= 2:
        area = float(parts[1])
        return area

    return None


def run_parallel_decomposition(run_dir: Path, n_procs: int) -> bool:
    """Run decomposePar to prepare mesh for parallel execution."""
    if n_procs <= 1:
        return True

    print(f"[run_openfoam] Decomposing mesh for {n_procs} processors...")
    try:
        result = subprocess.run(
            ["decomposePar", "-force"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[run_openfoam] decomposePar failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return False
        return True
    except FileNotFoundError:
        print(f"[run_openfoam] decomposePar not found on PATH (required for parallel execution)")
        return False


def run_parallel_reconstruction(run_dir: Path) -> bool:
    """Run reconstructPar to merge parallel solutions."""
    print(f"[run_openfoam] Reconstructing solution from parallel processors...")
    try:
        result = subprocess.run(
            ["reconstructPar"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[run_openfoam] reconstructPar failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return False
        return True
    except FileNotFoundError:
        print(f"[run_openfoam] reconstructPar not found on PATH")
        return False


def run_openfoam_case(
    base_case_dir: Path,
    run_dir: Path,
    U_mesh: np.ndarray,
    nut_mesh: np.ndarray,
    ccl_inlet: float,
    cf_inlet: float,
    cs_inlet: float,
    start_time: float,
    end_time: float,
    prev_run_dir: Optional[Path] = None,
    solver_command: str = "Chlorine_OrganicReaction",
    n_procs: int = 1,
    save_postprocessing_only: bool = True,
    postprocessing_db_dir: Optional[Path] = None,
    delta_t: float = 0.1,
    write_interval: int = 500,
    inlet_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    reset_species_ic: bool = True,
) -> Tuple[bool, Optional[Tuple[float, float, float]], float, Optional[Path]]:
    """
    Copy base case, update fields, run solver, and extract output.

    reset_species_ic: if True and there is no prev_run_dir warm-start, rewrite
      CCl/Cf/Cs internalField to the inlet values (avoids Hydrodeca leftover ICs).

    Returns:
      (success: bool, (ccl_mean, cf_mean, cs_mean), outlet_area)
    """
    print(f"[run_openfoam] Setting up case in {run_dir}...")

    # Clean and copy base case
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(base_case_dir, run_dir)

    # Patch solver settings for numerical stability
    stabilize_solver_settings(run_dir)

    # Create 0/ if it doesn't exist
    field_dir = run_dir / "0"
    field_dir.mkdir(exist_ok=True)

    # Write U and nut
    _write_u_field(field_dir / "U", U_mesh, inlet_velocity=inlet_velocity)
    _write_nut_field(field_dir / "nut", nut_mesh)

    # Update CCl, Cf, Cs inlet BCs and possibly set ICs
    warmed = False
    if prev_run_dir:
        # Copy species fields from previous run's latestTime
        prev_latest = _find_latest_timestep(prev_run_dir)
        if prev_latest:
            for species in ["CCl", "Cf", "Cs"]:
                src = prev_latest / species
                dst = field_dir / species
                if src.exists():
                    shutil.copy(src, dst)
                    warmed = True

    # If ICs don't exist (fresh start), create them
    for species, inlet_val in [("CCl", ccl_inlet), ("Cf", cf_inlet), ("Cs", cs_inlet)]:
        species_file = field_dir / species
        if not species_file.exists():
            _write_species_field(species_file, len(U_mesh), inlet_val, species)
        else:
            # Always update inlet BC to current EPANET/upstream value
            update_inlet_bc_scalar(species_file, inlet_val, species)
            # On fresh steps, also clear Hydrodeca base-case internalField
            if reset_species_ic and not warmed:
                if set_uniform_internal_field(species_file, inlet_val):
                    print(
                        f"[run_openfoam] Reset {species} internalField to "
                        f"{inlet_val:.6e} (consistent IC)"
                    )

    # Update controlDict
    update_openfoam_controldict(run_dir / "system" / "controlDict", start_time, end_time, delta_t, write_interval)

    # Parallel decomposition if needed
    if n_procs > 1:
        decompose_dict = run_dir / "system" / "decomposeParDict"
        if decompose_dict.exists():
            update_decompose_par_dict(decompose_dict, n_procs)
        if not run_parallel_decomposition(run_dir, n_procs):
            print(f"[run_openfoam] Decomposition failed; falling back to serial execution")
            n_procs = 1

    # Run solver (serial or parallel)
    if n_procs > 1:
        print(f"[run_openfoam] Running solver in parallel with {n_procs} processes...")
        solver_cmd_list = ["mpirun", "--oversubscribe", "-np", str(n_procs), solver_command, "-parallel"]
    else:
        print(f"[run_openfoam] Running solver: {solver_command}...")
        solver_cmd_list = [solver_command]

    try:
        log_path = run_dir / "log.solver"
        result = subprocess.run(
            solver_cmd_list,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        # Save solver output for diagnostics
        log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
        if result.returncode != 0:
            print(f"[run_openfoam] Solver failed with return code {result.returncode}")
            print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
            # Show last 30 lines of stdout for crash context
            stdout_lines = result.stdout.strip().split("\n")
            print(f"  stdout (last 30 lines):")
            for line in stdout_lines[-30:]:
                print(f"    {line}")
            print(f"  Full log saved to: {log_path}")
            return (False, None, 0.0, None)
    except FileNotFoundError as e:
        print(f"[run_openfoam] Solver execution failed: {str(e)}")
        print(f"  Command: {' '.join(solver_cmd_list)}")
        print(f"  Please verify the solver path in config.json and OpenFOAM installation")
        return (False, None, 0.0, None)

    # Parallel reconstruction if needed
    if n_procs > 1:
        if not run_parallel_reconstruction(run_dir):
            print(f"[run_openfoam] WARNING: Reconstruction failed; results may be in processor directories")

    # Parse output
    print(f"[run_openfoam] Parsing postprocessing output...")
    outlet_vals = parse_postprocessing_output(run_dir)
    outlet_area = get_outlet_area(run_dir)

    # Save postProcessing and latest species fields if requested
    saved_fields_dir = None
    if save_postprocessing_only and postprocessing_db_dir:
        postprocessing_db_dir.mkdir(parents=True, exist_ok=True)
        run_postproc = run_dir / "postProcessing"
        if run_postproc.exists():
            db_run_postproc = postprocessing_db_dir / run_dir.name
            if db_run_postproc.exists():
                shutil.rmtree(db_run_postproc)
            shutil.copytree(run_postproc, db_run_postproc)
            print(f"[run_openfoam] Saved postProcessing to {db_run_postproc}")

        # Save latest species fields for continuity in next iteration
        latest_ts = _find_latest_timestep(run_dir)
        if latest_ts:
            saved_fields_dir = run_dir.parent / ".latest_species"
            if saved_fields_dir.exists():
                shutil.rmtree(saved_fields_dir)
            saved_fields_dir.mkdir(parents=True)
            ts_dir = saved_fields_dir / latest_ts.name
            ts_dir.mkdir()
            for species in ["CCl", "Cf", "Cs"]:
                src = latest_ts / species
                if src.exists():
                    shutil.copy(src, ts_dir / species)

        # Remove full run directory to save space
        if run_dir.exists():
            shutil.rmtree(run_dir)
            print(f"[run_openfoam] Cleaned up full run directory to save space")

    if outlet_vals and outlet_area:
        ccl_int, cf_int, cs_int = outlet_vals
        ccl_mean = ccl_int / outlet_area
        cf_mean = cf_int / outlet_area
        cs_mean = cs_int / outlet_area
        print(f"[run_openfoam] Outlet: CCl={ccl_mean:.6e}, Cf={cf_mean:.6e}, Cs={cs_mean:.6e}")
        return (True, (ccl_mean, cf_mean, cs_mean), outlet_area, saved_fields_dir)
    else:
        print(f"[run_openfoam] Could not parse postprocessing output")
        return (True, None, 0.0, saved_fields_dir)


def _find_latest_timestep(case_dir: Path) -> Optional[Path]:
    """Find the latest timestep directory (not 0/)."""
    dirs = []
    for d in case_dir.iterdir():
        if d.is_dir() and d.name != "0" and d.name != "constant" and d.name != "system":
            try:
                dirs.append((float(d.name), d))
            except ValueError:
                pass
    if dirs:
        dirs.sort()
        return dirs[-1][1]
    return None


def _write_u_field(path: Path, U_mesh: np.ndarray, inlet_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Write U field file with computed inlet boundary condition."""
    n_cells = len(U_mesh)
    Ux, Uy, Uz = inlet_velocity
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2506                                  |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
%d
(
""" % n_cells

    for vals in U_mesh:
        content += f"({vals[0]:.16e} {vals[1]:.16e} {vals[2]:.16e})\n"

    content += f""")
;

boundaryField
{{
    Inlet
    {{
        type            fixedValue;
        value           uniform ({Ux:.16e} {Uy:.16e} {Uz:.16e});
    }}
    Outlet
    {{
        type            zeroGradient;
    }}
    Top
    {{
        type            zeroGradient;
    }}
    Wall
    {{
        type            noSlip;
    }}
    Internal
    {{
        type            zeroGradient;
    }}
    Tubes
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
    path.write_text(content)


def _write_nut_field(path: Path, nut_mesh: np.ndarray):
    """Write nut field file.

    Uses 'zeroGradient' BC on all boundaries.  The nut field is frozen (read once,
    never re-solved), so wall functions are unnecessary and dangerous: nutkWallFunction
    tries to look up k from the turbulence model, and if k is unavailable (as in
    this species-only solver) the evaluation produces NaN → SIGFPE.
    """
    n_cells = len(nut_mesh)

    content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2506                                  |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";
    class       volScalarField;
    location    "0";
    object      nut;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   nonuniform List<scalar>
%d
(
""" % n_cells

    for val in nut_mesh:
        content += f"{val:.16e}\n"

    content += """)
;

boundaryField
{
    Inlet
    {
        type            zeroGradient;
    }
    Outlet
    {
        type            zeroGradient;
    }
    Top
    {
        type            zeroGradient;
    }
    Wall
    {
        type            zeroGradient;
    }
    Internal
    {
        type            zeroGradient;
    }
    Tubes
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //
"""
    path.write_text(content)


def _write_species_field(path: Path, n_cells: int, inlet_value: float, species: str):
    """Write species field file (CCl, Cf, or Cs)."""
    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2506                                  |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";
    class       volScalarField;
    location    "0";
    object      {species};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [1 -3 0 0 0 0 0];

internalField   uniform {inlet_value:.16e};

boundaryField
{{
    Inlet
    {{
        type            fixedValue;
        value           uniform {inlet_value:.16e};
    }}
    Outlet
    {{
        type            zeroGradient;
    }}
    Top
    {{
        type            zeroGradient;
    }}
    Wall
    {{
        type            zeroGradient;
    }}
    Internal
    {{
        type            zeroGradient;
    }}
    Tubes
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
    path.write_text(content)


def main():
    """Standalone test of OpenFOAM case management."""
    config_file = _repo_root() / "configs" / "config.json"
    if not config_file.exists():
        config_file = _repo_root() / "config.json"
    with open(config_file) as f:
        config = json.load(f)

    base_dir = _repo_root()
    base_case_dir = base_dir / config["openfoam"]["base_case_dir"]
    inference_file = base_dir / "inference_output.npz"
    map_file = base_dir / config["mapping"]["index_map_file"]

    if not inference_file.exists():
        print("[run_openfoam] No inference output found; run infer_hydraulics.py first")
        return

    # Load fields
    with np.load(inference_file) as data:
        U_grid = data["U_grid"]
        nut_grid = data["nut_grid"]

    # Load mapping and reconstruct
    cell_to_grid_map = np.load(map_file)
    grid_shape = tuple(config["ml"]["grid_shape"])

    print("[run_openfoam] Reconstructing mesh fields...")
    U_mesh = np.zeros((len(cell_to_grid_map), 3), dtype=np.float32)
    nut_mesh = np.zeros(len(cell_to_grid_map), dtype=np.float32)
    for c in range(len(cell_to_grid_map)):
        i, j, k = cell_to_grid_map[c]
        U_mesh[c] = U_grid[i, j, k, :]
        nut_mesh[c] = nut_grid[i, j, k]

    # Run test case
    run_dir = base_dir / "test_run"
    solver_cmd = config["openfoam"]["solver_command"]
    n_procs = config["openfoam"].get("n_procs", 1)
    save_postproc_only = config["openfoam"].get("save_postprocessing_only", True)
    postproc_db_dir = base_dir / config["openfoam"].get("postprocessing_database_dir", "database_postProcessing") if save_postproc_only else None

    delta_t = config["openfoam"].get("delta_t", 0.1)
    write_interval = config["openfoam"].get("write_interval", 500)

    success, outlet_vals, area, _ = run_openfoam_case(
        base_case_dir,
        run_dir,
        U_mesh,
        nut_mesh,
        ccl_inlet=0.001,
        cf_inlet=0.0004,
        cs_inlet=0.0015,
        start_time=0.0,
        end_time=100.0,
        solver_command=solver_cmd,
        n_procs=n_procs,
        save_postprocessing_only=save_postproc_only,
        postprocessing_db_dir=postproc_db_dir,
        delta_t=delta_t,
        write_interval=write_interval,
    )

    if success and outlet_vals:
        ccl_mean, cf_mean, cs_mean = outlet_vals
        print(f"[run_openfoam] Success! Mean outlet concentrations:")
        print(f"  CCl: {ccl_mean:.6e}")
        print(f"  Cf:  {cf_mean:.6e}")
        print(f"  Cs:  {cs_mean:.6e}")
    else:
        print(f"[run_openfoam] Case run completed but output parsing failed")


if __name__ == "__main__":
    main()
