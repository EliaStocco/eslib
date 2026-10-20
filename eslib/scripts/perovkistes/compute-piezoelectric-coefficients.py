#!/usr/bin/env python
"""Compute and plot aligned strain--dipole piezoelectric coefficients.

Input must be an aligned ``piezo_raw_frames.csv`` produced by
``align-piezo-raw-data.py``.  The covariance estimator uses the engineering
Voigt order ``xx, yy, zz, yz, xz, xy`` and writes coefficients in m/V (C/N).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


KB_EV_K = 8.617333262145e-5
EA_PER_EV_TO_M_PER_V = 1.0e-10
VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
AXES = ("x", "y", "z")
DIPOLE = tuple(f"dipole_{axis}_eA" for axis in AXES)
STRAIN = ("strain_xx", "strain_yy", "strain_zz", "strain_yz_engineering", "strain_xz_engineering", "strain_xy_engineering")
REQUIRED = ("temperature_K", "run", "step", *DIPOLE, *STRAIN)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
MARKERS = ("o", "s", "^", "D", "v", "P")


def prepare_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="aligned piezo_raw_frames.csv")
    parser.add_argument("-o", "--output", type=Path, default=None, help="coefficient CSV output")
    parser.add_argument("--plot-output", type=Path, default=None, help="coefficient plot output")
    parser.add_argument("--supercell", default="unknown", help="supercell label recorded in output")
    parser.add_argument("--equilibration", type=float, default=0.10, help="fraction removed from each run (default: 0.10)")
    parser.add_argument("--min-frames", type=int, default=100, help="minimum post-equilibration frames per run")
    parser.add_argument("--no-plot", action="store_true", help="write the CSV only")
    parser.add_argument("--dpi", type=int, default=300, help="plot DPI (default: 300)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    return parser.parse_args()


def covariance_tensor(dipoles: np.ndarray, strains: np.ndarray, temperature: float) -> np.ndarray:
    dipole_delta = dipoles - dipoles.mean(axis=0)
    strain_delta = strains - strains.mean(axis=0)
    covariance = dipole_delta.T @ strain_delta / (len(dipoles) - 1)
    return covariance * EA_PER_EV_TO_M_PER_V / (KB_EV_K * temperature)


def compute(frame: pd.DataFrame, equilibration: float, min_frames: int, supercell: str):
    grouped, counts, skipped = {}, {}, []
    for (temperature, run), group in frame.groupby(["temperature_K", "run"], sort=True):
        group = group.sort_values("step")
        first = int(len(group) * equilibration)
        group = group.iloc[first:]
        if len(group) < min_frames:
            skipped.append((temperature, run, len(group), "too few post-equilibration frames"))
            continue
        dipoles = group.loc[:, DIPOLE].to_numpy(dtype=float)
        strains = group.loc[:, STRAIN].to_numpy(dtype=float)
        grouped.setdefault(float(temperature), []).append(covariance_tensor(dipoles, strains, float(temperature)))
        counts.setdefault(float(temperature), []).append(len(group))
    if not grouped:
        raise ValueError("no runs contain enough post-equilibration frames")

    records = []
    for temperature, tensors in sorted(grouped.items()):
        values = np.asarray(tensors)
        mean = values.mean(axis=0)
        stderr = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.full((3, 6), np.nan)
        record = {"supercell": supercell, "temperature_K": temperature, "n_runs": len(values), "mean_frames": float(np.mean(counts[temperature]))}
        record.update({f"d_{i}_{name}": mean[i, j] for i in range(3) for j, name in enumerate(VOIGT)})
        record.update({f"stderr_d_{i}_{name}": stderr[i, j] for i in range(3) for j, name in enumerate(VOIGT)})
        records.append(record)
    return pd.DataFrame.from_records(records), skipped


def plot(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temperature = summary["temperature_K"].to_numpy(dtype=float)
    component_groups = (("Normal strain", (0, 1, 2)), ("Shear strain", (3, 4, 5)))
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 8.4), sharex=True, constrained_layout=True)
    for i, cartesian in enumerate(AXES):
        for column, (group_name, components) in enumerate(component_groups):
            axis = axes[i, column]
            for j in components:
                voigt = VOIGT[j]
                signed_value = summary[f"d_{i}_{voigt}"].to_numpy(dtype=float) * 1.0e12
                value = np.abs(signed_value)
                error = summary[f"stderr_d_{i}_{voigt}"].to_numpy(dtype=float) * 1.0e12
                axis.plot(temperature, value, color=COLORS[j], marker=MARKERS[j], linewidth=1.6, markersize=4, label=voigt)
                if np.isfinite(error).any():
                    axis.fill_between(temperature, np.maximum(0.0, value - error), value + error,
                                      color=COLORS[j], alpha=0.13, linewidth=0)
            axis.set_ylabel(rf"$|d_{{{cartesian}J}}|$ (pC/N)")
            axis.grid(True, alpha=0.25)
            axis.set_axisbelow(True)
            axis.legend(title=rf"{group_name} $J$ for $M_{cartesian}$", ncol=3, frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Temperature (K)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> int:
    args = prepare_args()
    input_path = args.input.expanduser().resolve()
    output = (input_path.with_name("piezoelectric_coefficients_aligned.csv") if args.output is None else args.output.expanduser().resolve())
    plot_output = (output.with_name(f"{output.stem}_vs_temperature.png") if args.plot_output is None else args.plot_output.expanduser().resolve())
    try:
        if not 0.0 <= args.equilibration < 1.0 or args.min_frames < 2 or args.dpi < 1:
            raise ValueError("equilibration must be in [0, 1), min-frames >= 2, and dpi positive")
        targets = (output,) if args.no_plot else (output, plot_output)
        if len(set(targets)) != len(targets):
            raise ValueError("CSV and plot outputs must differ")
        for path in targets:
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
        frame = pd.read_csv(input_path)
        missing = [name for name in REQUIRED if name not in frame.columns]
        if missing:
            raise ValueError("input is missing column(s): " + ", ".join(missing))
        frame.loc[:, REQUIRED] = frame.loc[:, REQUIRED].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(frame.loc[:, REQUIRED].to_numpy(dtype=float)).all():
            raise ValueError("input contains non-finite required values")
        summary, skipped = compute(frame, args.equilibration, args.min_frames, args.supercell)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False, float_format="%.12g")
        if not args.no_plot:
            plot(summary, plot_output, args.dpi)
        print(f"Wrote {len(summary)} temperature summaries to {output}")
        if not args.no_plot:
            print(f"Wrote {plot_output}")
        if skipped:
            print(f"Skipped {len(skipped)} runs", file=sys.stderr)
    except (OSError, ValueError, FileNotFoundError, FileExistsError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
