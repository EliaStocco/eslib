#!/usr/bin/env python
"""Align run-mean lattice parameters and summarize them by temperature.

Input is the frame-resolved CSV written by ``analyze-lattice-vectors.py``.
Frames retain their native labels during the time average.  Only each run's
mean cell is canonicalized before independent runs are compared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lattice_vector_tools import alignment_transform, cellpar_from_cell, permute_components


PARAMETERS = ("a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg")
VECTORS = ("a_x", "a_y", "a_z", "b_x", "b_y", "b_z", "c_x", "c_y", "c_z")
REQUIRED = ("temperature_K", "run", "row", *PARAMETERS, *VECTORS)


def prepare_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV from analyze-lattice-vectors.py")
    parser.add_argument("--run-output", type=Path, default=Path("lattice_run_means.csv"), help="aligned per-run CSV")
    parser.add_argument("--summary-output", type=Path, default=Path("lattice_temperature_summary.csv"), help="temperature summary CSV")
    parser.add_argument(
        "--transform-output", type=Path, default=Path("lattice_alignment_transforms.npz"),
        help="per-run signed-basis and Cartesian rotation transforms (.npz)",
    )
    parser.add_argument("--discard", type=int, default=0, metavar="N", help="discard rows below N in every run")
    parser.add_argument("--stride", type=int, default=1, metavar="N", help="keep every Nth retained row")
    parser.add_argument("--right-angle-tolerance", type=float, default=1.0e-2, metavar="DEG", help="orthorhombic right-angle tolerance (default: 0.01)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    return parser.parse_args()


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError("input is missing column(s): " + ", ".join(missing))
    values = frame.loc[:, REQUIRED].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("input contains non-finite or non-numeric required values")
    if not np.allclose(values["run"], np.rint(values["run"])) or not np.allclose(values["row"], np.rint(values["row"])):
        raise ValueError("run and row columns must contain integers")
    values["run"] = values["run"].astype(int)
    values["row"] = values["row"].astype(int)
    return values


def make_run_table(frame: pd.DataFrame, right_angle_tolerance: float) -> tuple[pd.DataFrame, list[dict[str, np.ndarray | float | int]]]:
    records, transforms = [], []
    for (temperature, run), group in frame.groupby(["temperature_K", "run"], sort=True):
        if len(group) < 2:
            print(f"Warning: skipped ({temperature:g} K, run {run}) with fewer than two frames", file=sys.stderr)
            continue
        cell = group.loc[:, VECTORS].mean().to_numpy(dtype=float).reshape(3, 3)
        transform = alignment_transform(cell, right_angle_tolerance=right_angle_tolerance)
        order = tuple(transform["order"])
        canonical = cellpar_from_cell(transform["reference_cell"])
        fluctuations = group.loc[:, PARAMETERS].std(ddof=1).to_numpy(dtype=float)
        fluctuations = permute_components(fluctuations, order)
        record = {"temperature_K": float(temperature), "run": int(run), "n_rows": len(group)}
        record.update(dict(zip(PARAMETERS, canonical)))
        record.update({f"{name}_fluctuation": value for name, value in zip(PARAMETERS, fluctuations)})
        records.append(record)
        transforms.append({
            "temperature_K": float(temperature), "run": int(run), "n_rows": len(group),
            **transform,
        })
    if not records:
        raise ValueError("no runs contain at least two retained frames")
    return pd.DataFrame.from_records(records), transforms


def write_transforms(path: Path, transforms: list[dict[str, np.ndarray | float | int]]) -> Path:
    """Write machine-readable transforms and a CSV table for inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        temperature_K=np.asarray([item["temperature_K"] for item in transforms], dtype=float),
        run=np.asarray([item["run"] for item in transforms], dtype=int),
        n_rows=np.asarray([item["n_rows"] for item in transforms], dtype=int),
        order=np.stack([item["order"] for item in transforms]),
        signs=np.stack([item["signs"] for item in transforms]),
        A=np.stack([item["A"] for item in transforms]),
        R=np.stack([item["R"] for item in transforms]),
        reference_cell=np.stack([item["reference_cell"] for item in transforms]),
    )
    table_records = []
    for item in transforms:
        record = {key: item[key] for key in ("temperature_K", "run", "n_rows")}
        record.update({f"order_{i}": value for i, value in enumerate(item["order"])})
        record.update({f"sign_{i}": value for i, value in enumerate(item["signs"])})
        for name in ("A", "R", "reference_cell"):
            record.update({f"{name}_{i}{j}": item[name][i, j] for i in range(3) for j in range(3)})
        table_records.append(record)
    table_path = path.with_suffix(".csv")
    pd.DataFrame.from_records(table_records).to_csv(table_path, index=False, float_format="%.12g")
    return table_path


def make_summary(run_table: pd.DataFrame) -> pd.DataFrame:
    records = []
    metrics = (*PARAMETERS, *(f"{name}_fluctuation" for name in PARAMETERS))
    for temperature, group in run_table.groupby("temperature_K", sort=True):
        record = {"temperature_K": float(temperature), "n_runs": len(group), "total_rows": int(group["n_rows"].sum())}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std_between_runs"] = std
            record[f"{metric}_sem"] = std / np.sqrt(len(values))
        records.append(record)
    return pd.DataFrame.from_records(records)


def main() -> int:
    args = prepare_args()
    input_path = args.input.expanduser().resolve()
    run_output = args.run_output.expanduser().resolve()
    summary_output = args.summary_output.expanduser().resolve()
    transform_output = args.transform_output.expanduser().resolve()
    transform_table_output = transform_output.with_suffix(".csv")
    try:
        if args.discard < 0 or args.stride < 1 or args.right_angle_tolerance <= 0.0:
            raise ValueError("--discard >= 0, --stride >= 1, and a positive right-angle tolerance are required")
        if len({run_output, summary_output, transform_output, transform_table_output}) != 4:
            raise ValueError("all output paths must differ")
        for path in (run_output, summary_output, transform_output, transform_table_output):
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
        frame = validate(pd.read_csv(input_path))
        frame = frame[frame["row"] >= args.discard]
        if args.stride > 1:
            frame = frame[(frame["row"] - args.discard) % args.stride == 0]
        if frame.empty:
            raise ValueError("no frames remain after --discard and --stride")
        run_table, transforms = make_run_table(frame, args.right_angle_tolerance)
        summary = make_summary(run_table)
        run_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        run_table.to_csv(run_output, index=False, float_format="%.12g")
        summary.to_csv(summary_output, index=False, float_format="%.12g")
        transform_table_output = write_transforms(transform_output, transforms)
        print(f"Wrote {len(run_table)} run means to {run_output}")
        print(f"Wrote {len(summary)} temperature summaries to {summary_output}")
        print(f"Wrote {len(transforms)} alignment transforms to {transform_output} and {transform_table_output}")
    except (OSError, ValueError, FileNotFoundError, FileExistsError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
