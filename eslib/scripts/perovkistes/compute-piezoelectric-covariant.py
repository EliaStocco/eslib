#!/usr/bin/env python
"""Compute fixed-field and covariant piezoelectric fluctuation estimators.

The input is an aligned ``piezo_raw_frames.csv`` produced by
``align-piezo-raw-data.py``.  For every temperature/run, a post-equilibration
mean cell is used as the reference cell H_ref.  With lattice vectors stored as
rows, the Cartesian deformation gradient is

    F(t) = H(t).T @ inv(H_ref.T).

The script reports two explicitly labelled quantities (in m/V = C/N):

* ``fixed_cartesian``: beta cov(M_i, eta_J), the response at fixed Cartesian
  electric field, for an electric-enthalpy coupling -M dot E;
* ``reference_covariant``: beta cov((F^-1 M)_i, eta_J), the response to an
  electric field represented in the reference/material frame.

The latter removes affine dipole transport and is the finite-frame counterpart
of the covariant dipole derivative.  It is not interchangeable with the
fixed-Cartesian-field response; both are written explicitly.
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
AXES = ("x", "y", "z")
VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
DIPOLE = tuple(f"dipole_{axis}_eA" for axis in AXES)
CELL = tuple(f"cell_{i}{j}_angstrom" for i in range(3) for j in range(3))
REQUIRED = ("temperature_K", "run", "step", *DIPOLE, *CELL)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
MARKERS = ("o", "s", "^", "D", "v", "P")


def prepare_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="aligned piezo_raw_frames.csv")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("piezoelectric_coefficients_covariant.csv"),
                        help="coefficient CSV output")
    parser.add_argument("--plot-output", type=Path, default=None,
                        help="comparison plot output (defaults beside the CSV)")
    parser.add_argument("--plot-estimator", choices=("fixed_cartesian", "reference_covariant"),
                        default="fixed_cartesian",
                        help="coefficient convention displayed in the plot (default: fixed_cartesian)")
    parser.add_argument("--no-plot", action="store_true", help="write the CSV only")
    parser.add_argument("--dpi", type=int, default=300, help="plot DPI (default: 300)")
    parser.add_argument("--supercell", default="unknown", help="supercell label recorded in output")
    parser.add_argument("--equilibration", type=float, default=0.10,
                        help="fraction removed before computing each run reference (default: 0.10)")
    parser.add_argument("--min-frames", type=int, default=100,
                        help="minimum post-equilibration frames per run (default: 100)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    return parser.parse_args()


def deformation_gradient(cells: np.ndarray, reference_cell: np.ndarray) -> np.ndarray:
    """Return F such that H(t).T = F(t) @ H_ref.T for row-vector cells."""
    return np.einsum("nij,jk->nik", cells.transpose(0, 2, 1), np.linalg.inv(reference_cell.T))


def green_lagrange_voigt(deformation: np.ndarray) -> np.ndarray:
    """Green--Lagrange strain in xx, yy, zz, yz, xz, xy engineering Voigt order."""
    right_cauchy_green = np.einsum("nji,njk->nik", deformation, deformation)
    strain = 0.5 * (right_cauchy_green - np.eye(3))
    return np.column_stack((strain[:, 0, 0], strain[:, 1, 1], strain[:, 2, 2],
                            2.0 * strain[:, 1, 2], 2.0 * strain[:, 0, 2],
                            2.0 * strain[:, 0, 1]))


def pull_back_dipole(dipoles: np.ndarray, deformation: np.ndarray) -> np.ndarray:
    """Return F^-1 M, with each dipole treated as a Cartesian column vector."""
    return np.einsum("nij,nj->ni", np.linalg.inv(deformation), dipoles)


def covariance_tensor(dipoles: np.ndarray, strains: np.ndarray, temperature: float) -> np.ndarray:
    dipole_delta = dipoles - dipoles.mean(axis=0)
    strain_delta = strains - strains.mean(axis=0)
    covariance = dipole_delta.T @ strain_delta / (len(dipoles) - 1)
    return covariance * EA_PER_EV_TO_M_PER_V / (KB_EV_K * temperature)


def add_tensor(record: dict, prefix: str, tensor: np.ndarray) -> None:
    record.update({f"d_{prefix}_{i}_{voigt}": tensor[i, j]
                   for i in range(3) for j, voigt in enumerate(VOIGT)})


def add_error(record: dict, prefix: str, tensors: np.ndarray) -> None:
    stderr = (tensors.std(axis=0, ddof=1) / np.sqrt(len(tensors)) if len(tensors) > 1
              else np.full((3, 6), np.nan))
    record.update({f"stderr_d_{prefix}_{i}_{voigt}": stderr[i, j]
                   for i in range(3) for j, voigt in enumerate(VOIGT)})


def compute(frame: pd.DataFrame, equilibration: float, min_frames: int, supercell: str) -> tuple[pd.DataFrame, list]:
    by_temperature: dict[float, dict[str, list]] = {}
    skipped = []
    for (temperature, run), group in frame.groupby(["temperature_K", "run"], sort=True):
        group = group.sort_values("step").iloc[int(len(group) * equilibration):]
        if len(group) < min_frames:
            skipped.append((temperature, run, len(group), "too few post-equilibration frames"))
            continue
        cells = group.loc[:, CELL].to_numpy(dtype=float).reshape(-1, 3, 3)
        reference_cell = cells.mean(axis=0)
        try:
            deformation = deformation_gradient(cells, reference_cell)
            strains = green_lagrange_voigt(deformation)
            dipoles = group.loc[:, DIPOLE].to_numpy(dtype=float)
            covariant_dipoles = pull_back_dipole(dipoles, deformation)
        except np.linalg.LinAlgError as exc:
            skipped.append((temperature, run, len(group), str(exc)))
            continue
        item = by_temperature.setdefault(float(temperature), {
            "fixed_cartesian": [], "reference_covariant": [], "frames": [], "volumes": [],
        })
        item["fixed_cartesian"].append(covariance_tensor(dipoles, strains, float(temperature)))
        item["reference_covariant"].append(covariance_tensor(covariant_dipoles, strains, float(temperature)))
        item["frames"].append(len(group))
        item["volumes"].append(abs(float(np.linalg.det(reference_cell))))

    if not by_temperature:
        raise ValueError("no runs contain enough post-equilibration frames")
    records = []
    for temperature, values in sorted(by_temperature.items()):
        fixed = np.asarray(values["fixed_cartesian"])
        covariant = np.asarray(values["reference_covariant"])
        record = {
            "supercell": supercell, "temperature_K": temperature, "n_runs": len(fixed),
            "mean_frames": float(np.mean(values["frames"])),
            "mean_reference_volume_angstrom3": float(np.mean(values["volumes"])),
        }
        add_tensor(record, "fixed_cartesian", fixed.mean(axis=0))
        add_tensor(record, "reference_covariant", covariant.mean(axis=0))
        add_error(record, "fixed_cartesian", fixed)
        add_error(record, "reference_covariant", covariant)
        records.append(record)
    return pd.DataFrame.from_records(records), skipped


def plot(summary: pd.DataFrame, output: Path, dpi: int, estimator: str) -> None:
    """Plot one estimator's absolute values with run-to-run error bands."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    temperature = summary["temperature_K"].to_numpy(dtype=float)
    component_groups = (("Normal strain", (0, 1, 2)), ("Shear strain", (3, 4, 5)))
    estimator_label = {"fixed_cartesian": "fixed", "reference_covariant": "covariant"}[estimator]
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 8.4), sharex=True, constrained_layout=True)
    for i, cartesian in enumerate(AXES):
        for column, (group_name, components) in enumerate(component_groups):
            axis = axes[i, column]
            for j in components:
                voigt = VOIGT[j]
                value = np.abs(summary[f"d_{estimator}_{i}_{voigt}"].to_numpy(dtype=float) * 1.0e12)
                error = summary[f"stderr_d_{estimator}_{i}_{voigt}"].to_numpy(dtype=float) * 1.0e12
                axis.plot(temperature, value, color=COLORS[j], marker=MARKERS[j],
                          linewidth=1.6, markersize=4, label=voigt)
                if np.isfinite(error).any():
                    axis.fill_between(temperature, np.maximum(0.0, value - error), value + error,
                                      color=COLORS[j], alpha=0.13, linewidth=0)
            axis.set_ylabel(rf"$|d_{{{cartesian}J}}|$ (pC/N)")
            axis.grid(True, alpha=0.25)
            axis.set_axisbelow(True)
            component_handles = [Line2D([], [], color=COLORS[j], marker=MARKERS[j],
                                        linewidth=1.6, label=VOIGT[j]) for j in components]
            axis.legend(handles=component_handles,
                        title=rf"{group_name} $J$ for {estimator_label} $M_{cartesian}$",
                        ncol=3, frameon=False, loc="upper right")
    for axis in axes[-1, :]:
        axis.set_xlabel("Temperature (K)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> int:
    args = prepare_args()
    input_path, output_path = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    plot_output = (output_path.with_name(f"{output_path.stem}_vs_temperature.png") if args.plot_output is None
                   else args.plot_output.expanduser().resolve())
    try:
        if not 0.0 <= args.equilibration < 1.0 or args.min_frames < 2 or args.dpi < 1:
            raise ValueError("equilibration must be in [0, 1), min-frames must be at least 2, and dpi must be positive")
        targets = (output_path,) if args.no_plot else (output_path, plot_output)
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False, float_format="%.12g")
        print(f"Wrote {len(summary)} temperature summaries to {output_path}")
        if not args.no_plot:
            plot(summary, plot_output, args.dpi, args.plot_estimator)
            print(f"Wrote {plot_output}")
        if skipped:
            print(f"Skipped {len(skipped)} runs", file=sys.stderr)
    except (OSError, ValueError, FileNotFoundError, FileExistsError, pd.errors.ParserError, np.linalg.LinAlgError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
