#!/usr/bin/env python
"""Align raw dipole--strain piezoelectric data into a common cell frame.

The input schema is the ``piezo_raw_frames.csv`` format produced from i-PI
trajectories.  One signed-basis and proper-rotation transform is determined
from the post-equilibration mean cell of each temperature/run, then applied to
every frame of that run.  The output keeps the same CSV schema, so it can be
passed directly to ``compute-piezoelectric-coefficients.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lattice_vector_tools import alignment_transform


DIPOLE = ("dipole_x_eA", "dipole_y_eA", "dipole_z_eA")
STRAIN = ("strain_xx", "strain_yy", "strain_zz", "strain_yz_engineering", "strain_xz_engineering", "strain_xy_engineering")
CELL = tuple(f"cell_{i}{j}_angstrom" for i in range(3) for j in range(3))
REQUIRED = ("temperature_K", "run", "step", *DIPOLE, *STRAIN, *CELL)


def prepare_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="raw piezo_raw_frames.csv")
    parser.add_argument("-o", "--output", type=Path, default=Path("piezo_raw_frames_aligned.csv"), help="aligned CSV output")
    parser.add_argument("--transform-output", type=Path, default=Path("piezo_alignment_transforms.npz"), help="per-run transform output")
    parser.add_argument("--equilibration", type=float, default=0.10, help="fraction removed before determining each run transform (default: 0.10)")
    parser.add_argument("--right-angle-tolerance", type=float, default=1.0e-2, metavar="DEG", help="orthorhombic right-angle tolerance (default: 0.01)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    return parser.parse_args()


def strain_tensor(values: np.ndarray) -> np.ndarray:
    """Convert engineering-Voigt strains to symmetric 3x3 matrices."""
    tensor = np.zeros((len(values), 3, 3), dtype=float)
    tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = values[:, 0], values[:, 1], values[:, 2]
    tensor[:, 1, 2] = tensor[:, 2, 1] = 0.5 * values[:, 3]
    tensor[:, 0, 2] = tensor[:, 2, 0] = 0.5 * values[:, 4]
    tensor[:, 0, 1] = tensor[:, 1, 0] = 0.5 * values[:, 5]
    return tensor


def engineering_voigt(tensor: np.ndarray) -> np.ndarray:
    return np.column_stack((tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2],
                            2.0 * tensor[:, 1, 2], 2.0 * tensor[:, 0, 2], 2.0 * tensor[:, 0, 1]))


def write_transforms(path: Path, transforms: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        temperature_K=np.asarray([item["temperature_K"] for item in transforms], dtype=float),
        run=np.asarray([item["run"] for item in transforms], dtype=int),
        n_frames=np.asarray([item["n_frames"] for item in transforms], dtype=int),
        order=np.stack([item["order"] for item in transforms]),
        signs=np.stack([item["signs"] for item in transforms]),
        A=np.stack([item["A"] for item in transforms]),
        R=np.stack([item["R"] for item in transforms]),
        reference_cell=np.stack([item["reference_cell"] for item in transforms]),
    )
    records = []
    for item in transforms:
        record = {key: item[key] for key in ("temperature_K", "run", "n_frames")}
        record.update({f"order_{i}": value for i, value in enumerate(item["order"])})
        record.update({f"sign_{i}": value for i, value in enumerate(item["signs"])})
        for name in ("A", "R", "reference_cell"):
            record.update({f"{name}_{i}{j}": item[name][i, j] for i in range(3) for j in range(3)})
        records.append(record)
    table_path = path.with_suffix(".csv")
    pd.DataFrame.from_records(records).to_csv(table_path, index=False, float_format="%.12g")
    return table_path


def main() -> int:
    args = prepare_args()
    input_path, output_path = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    transform_path = args.transform_output.expanduser().resolve()
    transform_table = transform_path.with_suffix(".csv")
    try:
        if not 0.0 <= args.equilibration < 1.0 or args.right_angle_tolerance <= 0.0:
            raise ValueError("equilibration must be in [0, 1) and right-angle tolerance must be positive")
        if len({output_path, transform_path, transform_table}) != 3:
            raise ValueError("all output paths must differ")
        for path in (output_path, transform_path, transform_table):
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
        frame = pd.read_csv(input_path)
        missing = [name for name in REQUIRED if name not in frame.columns]
        if missing:
            raise ValueError("input is missing column(s): " + ", ".join(missing))
        values = frame.loc[:, REQUIRED].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("input contains non-finite required values")

        transforms = []
        for (temperature, run), indices in values.groupby(["temperature_K", "run"], sort=True).groups.items():
            rows = np.asarray(list(indices), dtype=int)
            rows = rows[np.argsort(values.loc[rows, "step"].to_numpy())]
            first = int(len(rows) * args.equilibration)
            selected = rows[first:]
            if len(selected) < 2:
                raise ValueError(f"({temperature:g} K, run {run}) has fewer than two post-equilibration frames")
            mean_cell = values.loc[selected, CELL].to_numpy(dtype=float).mean(axis=0).reshape(3, 3)
            transform = alignment_transform(mean_cell, right_angle_tolerance=args.right_angle_tolerance)
            transforms.append({"temperature_K": float(temperature), "run": int(run), "n_frames": len(selected), **transform})

            rotation, basis = transform["R"], transform["A"]
            dipoles = values.loc[rows, DIPOLE].to_numpy(dtype=float) @ rotation
            strains = strain_tensor(values.loc[rows, STRAIN].to_numpy(dtype=float))
            aligned_strains = np.einsum("ij,njk,kl->nil", rotation.T, strains, rotation)
            cells = values.loc[rows, CELL].to_numpy(dtype=float).reshape(-1, 3, 3)
            aligned_cells = np.einsum("ij,njk,kl->nil", basis, cells, rotation)
            frame.loc[rows, DIPOLE] = dipoles
            frame.loc[rows, STRAIN] = engineering_voigt(aligned_strains)
            frame.loc[rows, CELL] = aligned_cells.reshape(-1, 9)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False, float_format="%.12g")
        transform_table = write_transforms(transform_path, transforms)
        print(f"Wrote {len(frame)} aligned frames to {output_path}")
        print(f"Wrote {len(transforms)} alignment transforms to {transform_path} and {transform_table}")
    except (OSError, ValueError, FileNotFoundError, FileExistsError, pd.errors.ParserError, np.linalg.LinAlgError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
