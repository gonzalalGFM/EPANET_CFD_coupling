#!/usr/bin/env python3
"""Plots + conclusions for campaign outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COUPLING_DIR = Path(__file__).resolve().parent
ROOT = COUPLING_DIR.parent
PAPER_RUNS = ROOT / "outputs" / "campaign"
OUT = PAPER_RUNS

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def _load(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    csv_path = out_dir / "campaign_history.csv"
    df = pd.read_csv(csv_path)
    summary_path = out_dir / "timings_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return df, summary


def fig_flow(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    t = df["global_time_h"]

    axes[0].plot(t, df["demand_node13_gpm"], color="#1d3557", lw=1.4)
    axes[0].set_ylabel("Demand node 13 (GPM)")
    axes[0].set_title("Network flow / operations")
    _shade_phases(axes[0], df)

    axes[1].plot(t, df["tank_level_ft"], color="#2a9d8f", lw=1.4)
    axes[1].axhline(110, color="#e76f51", ls="--", lw=0.9, label="pump on < 110 ft")
    axes[1].axhline(140, color="#264653", ls="--", lw=0.9, label="pump off > 140 ft")
    axes[1].set_ylabel("Tank level (ft)")
    axes[1].legend(fontsize=8, loc="upper right")
    _shade_phases(axes[1], df)

    axes[2].step(t, df["pump_status"], where="post", color="#e9c46a", lw=1.2)
    if "flow_9" in df.columns:
        ax2b = axes[2].twinx()
        ax2b.plot(t, df["flow_9"], color="#457b9d", lw=1.0, alpha=0.85)
        ax2b.set_ylabel("Pump flow (GPM)", color="#457b9d")
    axes[2].set_ylabel("Pump status")
    axes[2].set_xlabel("Network time (h)")
    axes[2].set_ylim(-0.1, 1.3)
    _shade_phases(axes[2], df)

    fig.savefig(out_dir / "fig_flow.png")
    fig.savefig(out_dir / "fig_flow.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'fig_flow.png'}")


def fig_quality(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, constrained_layout=True)
    t = df["global_time_h"]

    axes[0].plot(t, df["quality_upstream"], label="Node 12 (upstream)", color="#264653", lw=1.3)
    axes[0].plot(t, df["quality_node13_epanet"], label="Node 13 (EPANET before OF)", color="#a8dadc", lw=1.1)
    axes[0].plot(t, df["quality_node13_after"], label="Node 13 (after / used)", color="#e76f51", lw=1.4)
    axes[0].set_ylabel(r"$C_{\mathrm{Cl}}$ (mg/L)")
    axes[0].set_title("Chlorine quality at coupling nodes")
    axes[0].legend(fontsize=8, loc="best")
    _shade_phases(axes[0], df)

    # Extra junctions if present
    extra = [c for c in df.columns if c.startswith("q_") and c not in ("q_12", "q_13")]
    for c, col in zip(extra[:6], ["#1d3557", "#2a9d8f", "#e9c46a", "#f4a261", "#9b2226", "#457b9d"]):
        axes[1].plot(t, df[c], label=c.replace("q_", "Node "), lw=1.1, color=col)
    axes[1].set_ylabel(r"$C_{\mathrm{Cl}}$ (mg/L)")
    axes[1].set_xlabel("Network time (h)")
    axes[1].set_title("Chlorine at other logged nodes")
    if extra:
        axes[1].legend(fontsize=8, ncol=3, loc="best")
    _shade_phases(axes[1], df)

    fig.savefig(out_dir / "fig_quality.png")
    fig.savefig(out_dir / "fig_quality.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'fig_quality.png'}")


def fig_timings(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)

    for phase, color in [("warmup", "#2a9d8f"), ("coupled", "#e76f51")]:
        g = df[df["phase"] == phase]
        if g.empty:
            continue
        axes[0].plot(g["global_time_h"], g["wall_s"], ".", ms=3, alpha=0.7, color=color, label=phase)
    axes[0].set_xlabel("Network time (h)")
    axes[0].set_ylabel("Wall time per step (s)")
    axes[0].set_title("Per-step wall time")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    coupled = df[df["phase"] == "coupled"].copy()
    if not coupled.empty:
        for col, lab, color in [
            ("epanet_s", "EPANET", "#264653"),
            ("ml_total_s", "ML", "#2a9d8f"),
            ("openfoam_s", "OpenFOAM", "#e76f51"),
        ]:
            s = pd.to_numeric(coupled[col], errors="coerce")
            axes[1].plot(coupled["global_time_h"], s, "-o", ms=3, lw=1.1, color=color, label=lab)
        axes[1].set_xlabel("Network time (h)")
        axes[1].set_ylabel("Seconds")
        axes[1].set_title("Coupled step breakdown")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No coupled steps yet", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.savefig(out_dir / "fig_timings.png")
    fig.savefig(out_dir / "fig_timings.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'fig_timings.png'}")


def fig_paper_quality_coupling(df: pd.DataFrame, out_dir: Path) -> None:
    """Coupled-window chlorine: EPANET lumped vs OpenFOAM SETPOINT."""
    coupled = df[df["phase"] == "coupled"].copy()
    if coupled.empty:
        return
    t = coupled["global_time_h"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    ax.plot(t, coupled["quality_upstream"], color="#264653", lw=1.5, label="Node 12 (inlet)")
    ax.plot(
        t,
        coupled["quality_node13_epanet"],
        color="#457b9d",
        lw=1.2,
        ls="--",
        label="Node 13 (EPANET lumped)",
    )
    ax.plot(
        t,
        coupled["quality_node13_after"],
        color="#e76f51",
        lw=1.6,
        marker="o",
        ms=3,
        label="Node 13 (OpenFOAM SETPOINT)",
    )
    ax.set_xlabel("Network time (h)")
    ax.set_ylabel(r"Residual chlorine $C_{\mathrm{Cl}}$ (mg/L)")
    ax.set_title("Coupled quality at node 13 (calibrated organics, IC reset)")
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out_dir / "paper_quality_coupling.png")
    fig.savefig(out_dir / "paper_quality_coupling.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'paper_quality_coupling.png'}")


def fig_paper_quality_coupled_detail(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-step SETPOINT correction and outlet/inlet retention."""
    coupled = df[df["phase"] == "coupled"].copy()
    if coupled.empty:
        return
    q_ep = coupled["quality_node13_epanet"].astype(float)
    q_af = coupled["quality_node13_after"].astype(float)
    delta = q_af - q_ep
    inlet = coupled["quality_upstream"].astype(float)
    retention = q_af / inlet.replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    t = coupled["global_time_h"]

    axes[0].bar(t, delta, width=0.7, color="#e76f51", alpha=0.75, edgecolor="none")
    axes[0].axhline(0, color="#264653", lw=0.8)
    axes[0].set_xlabel("Network time (h)")
    axes[0].set_ylabel(r"$\Delta C_{\mathrm{Cl}}$ (OF $-$ EPANET) [mg/L]")
    axes[0].set_title("SETPOINT correction at node 13")

    axes[1].plot(t, retention, "o-", color="#2a9d8f", ms=4, lw=1.2)
    axes[1].set_xlabel("Network time (h)")
    axes[1].set_ylabel("Outlet / inlet retention")
    axes[1].set_title("Tank outlet vs upstream inlet")
    axes[1].set_ylim(0, max(1.05, float(retention.max()) * 1.05))

    fig.savefig(out_dir / "paper_quality_coupled_detail.png")
    fig.savefig(out_dir / "paper_quality_coupled_detail.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'paper_quality_coupled_detail.png'}")


def fig_paper_campaign_summary(df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    """Four-panel overview for the paper."""
    coupled = df[df["phase"] == "coupled"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)

    # A: demand + tank (coupled window)
    if not coupled.empty:
        t = coupled["global_time_h"]
        axes[0, 0].plot(t, coupled["demand_node13_gpm"], color="#1d3557", lw=1.4)
        axes[0, 0].set_ylabel("Demand (GPM)")
        axb = axes[0, 0].twinx()
        axb.plot(t, coupled["tank_level_ft"], color="#2a9d8f", lw=1.2, alpha=0.85)
        axb.set_ylabel("Tank level (ft)", color="#2a9d8f")
    axes[0, 0].set_title("Hydraulics (coupled window)")
    axes[0, 0].set_xlabel("Time (h)")

    # B: quality
    if not coupled.empty:
        axes[0, 1].plot(t, coupled["quality_node13_epanet"], "--", color="#457b9d", lw=1.2, label="EPANET")
        axes[0, 1].plot(t, coupled["quality_node13_after"], "-o", color="#e76f51", ms=3, lw=1.3, label="OpenFOAM")
        axes[0, 1].set_ylabel(r"$C_{\mathrm{Cl}}$ (mg/L)")
        axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Node 13 chlorine")
    axes[0, 1].set_xlabel("Time (h)")

    # C: timings pie (mean coupled step)
    phases = summary.get("phases", {}).get("coupled", {})
    of_s = phases.get("openfoam_s_mean", 0)
    ml_s = phases.get("ml_total_s_mean", 0)
    ep_s = phases.get("epanet_s_mean", 0) or 0
    other = max(0, phases.get("wall_s_mean", 0) - of_s - ml_s - ep_s)
    if of_s > 0:
        sizes = [of_s, ml_s, max(ep_s, 1e-9), other]
        labels = ["OpenFOAM", "ML", "EPANET", "Other"]
        colors = ["#e76f51", "#2a9d8f", "#264653", "#e9c46a"]
        axes[1, 0].pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 8},
        )
        axes[1, 0].set_title("Mean coupled-step cost share")
    else:
        axes[1, 0].set_axis_off()

    # D: text stats
    axes[1, 1].axis("off")
    if not coupled.empty:
        q_ep = coupled["quality_node13_epanet"].astype(float)
        q_af = coupled["quality_node13_after"].astype(float)
        delta = q_af - q_ep
        txt = (
            f"Campaign: {summary.get('warmup_hours', '?')} h warmup + "
            f"{summary.get('coupled_hours', '?')} h coupled\n"
            f"Wall clock: {summary.get('wall_total_s', 0)/60:.1f} min\n"
            f"Organics: cf={summary.get('openfoam_cf_inlet', '?')} "
            f"cs={summary.get('openfoam_cs_inlet', '?')}\n"
            f"IC reset: {summary.get('reset_species_ic', True)}\n\n"
            f"Node 13 after OF: {q_af.mean():.3f} ± {q_af.std(ddof=1):.3f} mg/L\n"
            f"EPANET before OF: {q_ep.mean():.3f} ± {q_ep.std(ddof=1):.3f} mg/L\n"
            f"Mean Δ (OF−EPANET): {delta.mean():+.3f} mg/L\n"
            f"OF success: {100*pd.to_numeric(coupled['openfoam_success'], errors='coerce').mean():.0f}%"
        )
        axes[1, 1].text(0.05, 0.95, txt, va="top", fontsize=9, family="monospace")

    fig.savefig(out_dir / "paper_campaign_summary.png")
    fig.savefig(out_dir / "paper_campaign_summary.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'paper_campaign_summary.png'}")


def fig_paper_hydraulics_overview(df: pd.DataFrame, out_dir: Path) -> None:
    """Full-horizon hydraulics with coupled shading."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, constrained_layout=True)
    t = df["global_time_h"]
    axes[0].plot(t, df["demand_node13_gpm"], color="#1d3557", lw=1.2)
    axes[0].set_ylabel("Demand node 13 (GPM)")
    axes[1].plot(t, df["tank_level_ft"], color="#2a9d8f", lw=1.2)
    axes[1].set_ylabel("Tank level (ft)")
    axes[1].set_xlabel("Network time (h)")
    for ax in axes:
        _shade_phases(ax, df)
    fig.savefig(out_dir / "paper_hydraulics_overview.png")
    fig.savefig(out_dir / "paper_hydraulics_overview.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'paper_hydraulics_overview.png'}")


def fig_paper_timings(df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    """Publication timing figure (alias of fig_timings with paper name)."""
    fig_timings(df, out_dir)
    for ext in ("png", "pdf"):
        src = out_dir / f"fig_timings.{ext}"
        dst = out_dir / f"paper_timings.{ext}"
        if src.exists():
            shutil.copy2(src, dst)
    print(f"wrote {out_dir / 'paper_timings.png'}")


def sync_paper_figures(out_dir: Path) -> None:
    """Copy pub figures into paper/latex_figures and paper/figures."""
    names = [
        "paper_campaign_summary",
        "paper_quality_coupling",
        "paper_quality_coupled_detail",
        "paper_hydraulics_overview",
        "paper_timings",
        "fig_quality",
    ]
    for sub in ("figures",):
        dest = ROOT / "docs" / sub
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            for ext in ("png", "pdf"):
                src = out_dir / f"{name}.{ext}"
                if src.exists():
                    shutil.copy2(src, dest / src.name)


def _shade_phases(ax, df: pd.DataFrame) -> None:
    if "phase" not in df.columns or df.empty:
        return
    t = df["global_time_h"].to_numpy()
    phase = df["phase"].to_numpy()
    # shade coupled region
    coupled = phase == "coupled"
    if coupled.any():
        t0 = t[coupled][0]
        t1 = t[coupled][-1]
        ax.axvspan(t0, t1, color="#e76f51", alpha=0.08, label=None)


def write_conclusions(df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    warmup = df[df["phase"] == "warmup"]
    coupled = df[df["phase"] == "coupled"]

    lines = [
        "# Paper campaign conclusions",
        "",
        f"- Warmup: **{summary.get('warmup_hours', '?')} h** EPANET-only "
        f"({len(warmup)} hydraulic steps).",
        f"- Coupled: **{summary.get('coupled_hours', '?')} h** ML + OpenFOAM "
        f"({len(coupled)} steps, OpenFOAM MPI ranks = "
        f"**{summary.get('n_procs_openfoam', '?')}**).",
        f"- Total campaign wall clock: **{summary.get('wall_total_s', 0)/60:.1f} min**.",
        "",
    ]
    if summary.get("openfoam_cf_inlet") is not None:
        lines += [
            f"- OpenFOAM organics (calibrated): "
            f"`cf_inlet={summary.get('openfoam_cf_inlet')}`, "
            f"`cs_inlet={summary.get('openfoam_cs_inlet')}`, "
            f"IC reset = **{summary.get('reset_species_ic', True)}**.",
            "",
        ]

    lines += ["## Timings", ""]

    phases = summary.get("phases", {})
    if "warmup" in phases:
        w = phases["warmup"]
        lines += [
            f"- Warmup step wall time: "
            f"**{w['wall_s_mean']*1e3:.3f} ± {w['wall_s_std']*1e3:.3f} ms** "
            f"(mean ± sample std).",
        ]
    if "coupled" in phases:
        c = phases["coupled"]
        lines += [
            f"- Coupled step wall time: "
            f"**{c['wall_s_mean']:.2f} ± {c.get('wall_s_std', 0):.2f} s**.",
        ]
        if "ml_total_s_mean" in c:
            lines.append(
                f"- ML pipeline (infer+pressure+recon): "
                f"**{c['ml_total_s_mean']:.3f} ± {c.get('ml_total_s_std', 0):.3f} s**."
            )
        if "openfoam_s_mean" in c:
            lines.append(
                f"- OpenFOAM species window: "
                f"**{c['openfoam_s_mean']:.2f} ± {c.get('openfoam_s_std', 0):.2f} s** "
                f"(dominant cost)."
            )
        if c.get("wall_s_mean", 0) > 0 and c.get("openfoam_s_mean"):
            share = 100.0 * c["openfoam_s_mean"] / c["wall_s_mean"]
            lines.append(f"- OpenFOAM share of coupled step: **~{share:.1f}%**.")

    lines += ["", "## Hydraulics / quality", ""]

    if not warmup.empty:
        lines.append(
            f"- Warmup end (t = {warmup['global_time_h'].iloc[-1]:.1f} h): "
            f"tank **{warmup['tank_level_ft'].iloc[-1]:.1f} ft**, "
            f"node-13 Cl **{warmup['quality_node13_after'].iloc[-1]:.3f} mg/L**, "
            f"upstream (12) **{warmup['quality_upstream'].iloc[-1]:.3f} mg/L**."
        )
    if not coupled.empty:
        q_ep = coupled["quality_node13_epanet"].astype(float)
        q_af = coupled["quality_node13_after"].astype(float)
        delta = (q_af - q_ep).to_numpy()
        lines += [
            f"- Coupled window: node-13 Cl after OpenFOAM "
            f"**{q_af.mean():.3f} ± {q_af.std(ddof=1) if len(q_af)>1 else 0:.3f} mg/L** "
            f"(mean ± std over steps).",
            f"- Mean SETPOINT correction vs EPANET-only at node 13: "
            f"**{delta.mean():+.3f} mg/L** "
            f"(OF − EPANET-before; positive ⇒ 3D tank raises residual).",
            f"- OpenFOAM success rate: "
            f"**{100.0 * pd.to_numeric(coupled['openfoam_success'], errors='coerce').mean():.0f}%**.",
        ]

    lines += [
        "",
        "## Takeaways",
        "",
        "1. A multi-week EPANET warmup is essentially free (sub-millisecond to "
        "millisecond hydraulic steps) and establishes a realistic tank/quality state "
        "before 3D enrichment.",
        "2. Once ML + OpenFOAM are enabled, wall time jumps by ~$10^4$–$10^5\\times$ "
        "per hydraulic hour; OpenFOAM species transport dominates.",
        "3. The ML hydraulic surrogate remains a small fraction of the coupled step, "
        "confirming that further speedups should target the species integrator "
        "(shorter local window, coarser mesh, or fewer MPI-idle inefficiencies), "
        "not GPU inference.",
        "4. Quality plots show whether the 3D tank SETPOINT systematically shifts "
        "node-13 chlorine relative to the native 0D EPANET residual — the physical "
        "motivation for the hybrid scheme.",
        "",
        "## Figures",
        "",
        "- `fig_flow.png` — demand, tank level, pump",
        "- `fig_quality.png` — chlorine at coupling and other nodes",
        "- `fig_timings.png` — wall-time profile and coupled breakdown",
        "- `paper_quality_coupling.png` — coupled-window quality (pub)",
        "- `paper_quality_coupled_detail.png` — SETPOINT correction detail",
        "- `paper_campaign_summary.png` — four-panel overview",
        "",
    ]

    path = out_dir / "conclusions.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def make_all_plots(out_dir: Path | None = None, *, sync_paper: bool = True) -> None:
    out_dir = Path(out_dir) if out_dir else OUT
    df, summary = _load(out_dir)
    fig_flow(df, out_dir)
    fig_quality(df, out_dir)
    fig_timings(df, out_dir)
    fig_paper_quality_coupling(df, out_dir)
    fig_paper_quality_coupled_detail(df, out_dir)
    fig_paper_campaign_summary(df, summary, out_dir)
    fig_paper_hydraulics_overview(df, out_dir)
    fig_paper_timings(df, summary, out_dir)
    write_conclusions(df, summary, out_dir)
    if sync_paper:
        sync_paper_figures(out_dir)


if __name__ == "__main__":
    make_all_plots()
