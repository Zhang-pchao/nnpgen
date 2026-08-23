#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rmse_mae(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    d = y - x
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    return rmse, mae


def _parity_hexbin(ax, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, note: str = "", max_points: int = 300000) -> Dict[str, float]:
    if x.size == 0 or y.size == 0:
        raise ValueError("empty parity data for {0}".format(title))

    if x.size != y.size:
        raise ValueError("size mismatch for {0}: {1} vs {2}".format(title, x.size, y.size))

    x_plot = x
    y_plot = y
    if x.size > max_points:
        rng = np.random.default_rng(2026)
        idx = rng.choice(x.size, size=max_points, replace=False)
        x_plot = x[idx]
        y_plot = y[idx]

    xy_min = float(min(np.min(x_plot), np.min(y_plot)))
    xy_max = float(max(np.max(x_plot), np.max(y_plot)))
    span = max(xy_max - xy_min, 1e-12)
    pad = span * 0.05

    hb = ax.hexbin(x_plot, y_plot, gridsize=160, bins="log", mincnt=1, cmap="viridis")
    ax.plot([xy_min - pad, xy_max + pad], [xy_min - pad, xy_max + pad], "k--", linewidth=1.2)
    ax.set_xlim(xy_min - pad, xy_max + pad)
    ax.set_ylim(xy_min - pad, xy_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    cb = plt.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("log10(count)", fontsize=9)

    rmse, mae = _rmse_mae(x, y)
    text = "RMSE={0:.6g}\nMAE={1:.6g}".format(rmse, mae)
    if note:
        text += "\n" + note
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    return {"rmse": rmse, "mae": mae, "points": int(x.size), "points_plotted": int(x_plot.size)}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arrays", required=True, help="Path to benchmark_arrays.npz")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--model-label", default="Model", help="Model name used in titles and prediction-axis labels")
    parser.add_argument("--reference-label", default="Reference", help="Reference-data name used on the x axes")
    parser.add_argument("--output-prefix", default="parity_density", help="Filename prefix for PNG, PDF, and summary outputs")


def run(args: argparse.Namespace) -> Dict[str, object]:
    arrays_path = Path(args.arrays)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_label = str(args.model_label).strip() or "Model"
    reference_label = str(args.reference_label).strip() or "Reference"
    output_prefix = str(args.output_prefix).strip()
    if not output_prefix or Path(output_prefix).name != output_prefix:
        raise ValueError("output-prefix must be a filename without directory separators")

    if not arrays_path.is_file():
        raise ValueError("arrays file not found: {0}".format(arrays_path))

    z = np.load(str(arrays_path))
    energy_true = np.array(z["energy_true"], dtype=float).reshape(-1)
    energy_pred = np.array(z["energy_pred"], dtype=float).reshape(-1)
    natoms = np.array(z["natoms"], dtype=float).reshape(-1)
    force_true = np.array(z["force_true"], dtype=float).reshape(-1)
    force_pred = np.array(z["force_pred"], dtype=float).reshape(-1)
    virial_true = np.array(z["virial_true"], dtype=float).reshape(-1) if "virial_true" in z.files else np.array([], dtype=float)
    virial_pred = np.array(z["virial_pred"], dtype=float).reshape(-1) if "virial_pred" in z.files else np.array([], dtype=float)

    if energy_true.size == 0:
        raise ValueError("energy arrays are empty")
    if natoms.size != energy_true.size:
        raise ValueError("natoms size mismatch with energy arrays")

    e_true_pa = energy_true / natoms
    e_pred_pa = energy_pred / natoms

    panels = ["energy", "force"]
    if virial_true.size > 0 and virial_pred.size > 0 and virial_true.size == virial_pred.size:
        panels.append("virial")

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), dpi=220)
    if len(panels) == 1:
        axes = [axes]

    metrics: Dict[str, Dict[str, float]] = {}

    k = 0
    metrics["energy_per_atom"] = _parity_hexbin(
        axes[k],
        e_true_pa,
        e_pred_pa,
        "Energy Parity",
        f"{reference_label} Energy (eV/atom)",
        f"{model_label} Energy (eV/atom)",
    )
    k += 1

    metrics["force_components"] = _parity_hexbin(
        axes[k],
        force_true,
        force_pred,
        "Force Parity",
        f"{reference_label} Force (eV/Ang)",
        f"{model_label} Force (eV/Ang)",
    )
    k += 1

    if "virial" in panels:
        metrics["virial_components"] = _parity_hexbin(
            axes[k],
            virial_true,
            virial_pred,
            "Virial Parity",
            f"{reference_label} Virial",
            f"{model_label} Virial",
        )

    fig.suptitle(f"{model_label} vs {reference_label} Parity", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / f"{output_prefix}.png"
    pdf_path = output_dir / f"{output_prefix}.pdf"
    fig.savefig(str(png_path), bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "created_at": _now(),
        "arrays": str(arrays_path),
        "output_dir": str(output_dir),
        "plot_png": str(png_path),
        "plot_pdf": str(pdf_path),
        "metrics": metrics,
        "counts": {
            "energy_frames": int(energy_true.size),
            "force_components": int(force_true.size),
            "virial_components": int(virial_true.size),
        },
    }

    summary_path = output_dir / f"{output_prefix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model-versus-reference parity density from benchmark arrays")
    add_arguments(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
