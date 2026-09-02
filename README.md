# EPANET + ML + OpenFOAM coupling

Repository for paper entitled **Coupling 1D Network Hydraulics with ML-Accelerated 3D CFD for High-Fidelity Water Quality Modeling in Water Distribution Networks**.

Three parts:

1. **preprocess/** — take OpenFOAM cube runs, put them on a regular grid, write HDF5  
2. **training/** — train the slice-wise U-Nets (velocity/nut, then pressure)  
3. **coupling/** — online loop: EPANET → ML fields → OpenFOAM species → SETPOINT back to EPANET  

```bash
pip install -r requirements.txt
```

You need OpenFOAM (`Chlorine_OrganicReaction` on PATH), a case folder (`openfoam_case/`), an EPANET inp (`network.inp`), and a GPU for training/inference.

**Zenodo (fill in after deposit):**

- Cube CFD cases: `https://doi.org/10.5281/zenodo.XXXXXXX`  
  unpack somewhere and point `--base` at that folder  
- Chlorine / organic-reaction solver: `https://doi.org/10.5281/zenodo.YYYYYYY`  
  build/install, then put `Chlorine_OrganicReaction` on `PATH` (or set `solver_command` in `configs/`)

Typical order:

```bash
# 1. build dataset from your Cube* cases
python3 preprocess/process_openfoam.py --base /path/to/zenodo_cubes   # ← replace
python3 preprocess/process_openfoam_with_pressure.py --base /path/to/zenodo_cubes
python3 preprocess/precompute_mapping.py

# 2. train (defaults match the paper: base=48, depth=4)
python3 training/train_unet.py
python3 training/train_pressure_unet.py

# 3. couple
python3 coupling/run_workflow.py
# or longer campaign:
python3 coupling/run_paper_campaign.py
```

Edit paths / MPI ranks / organics in `configs/`. Keep absolute machine paths out of git.

Weights and HDF5 go under `data/` (gitignored). Put your mesh + network at the repo root.
