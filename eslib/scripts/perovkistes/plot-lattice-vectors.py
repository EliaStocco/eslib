#!/usr/bin/env python
"""Plot temperature-dependent lattice lengths and angles from an aligned summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


LENGTHS = ("a_angstrom", "b_angstrom", "c_angstrom")
ANGLES = ("alpha_deg", "beta_deg", "gamma_deg")
STYLES = {
    "a_angstrom": ("a", "#0072B2", "o"), "b_angstrom": ("b", "#D55E00", "s"),
    "c_angstrom": ("c", "#009E73", "^"), "alpha_deg": (r"$\alpha$", "#0072B2", "o"),
    "beta_deg": (r"$\beta$", "#D55E00", "s"), "gamma_deg": (r"$\gamma$", "#009E73", "^"),
}


def prepare_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="temperature summary CSV from align-lattice-vectors.py")
    parser.add_argument("-o", "--output", type=Path, default=Path("lattice_vectors.png"), help="plot output")
    parser.add_argument("--uncertainty", choices=("sem", "std", "none"), default="sem", help="shaded band (default: sem)")
    parser.add_argument("--length-scale", type=float, default=1.0, help="divide displayed lengths and their uncertainty by this value")
    parser.add_argument("--style", type=Path, default=None, help="optional Matplotlib .mplstyle file")
    parser.add_argument("--dpi", type=int, default=300, help="output DPI (default: 300)")
    parser.add_argument("--show", action="store_true", help="display the figure")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    return parser.parse_args()


def main() -> int:
    args = prepare_args()
    input_path, output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    try:
        if args.length_scale <= 0.0 or args.dpi < 1:
            raise ValueError("--length-scale and --dpi must be positive")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {output}; pass --overwrite to replace it")
        summary = pd.read_csv(input_path)
        required = ["temperature_K"]
        suffix = "sem" if args.uncertainty == "sem" else "std_between_runs"
        for name in (*LENGTHS, *ANGLES):
            required.append(f"{name}_mean")
            if args.uncertainty != "none":
                required.append(f"{name}_{suffix}")
        missing = [name for name in required if name not in summary.columns]
        if missing:
            raise ValueError("summary is missing column(s): " + ", ".join(missing))
        summary = summary.sort_values("temperature_K")
        if not np.isfinite(summary[required].to_numpy(dtype=float)).all():
            raise ValueError("summary contains non-finite plot values")
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if args.style is not None:
            plt.style.use(args.style.expanduser().resolve())

        temperature = summary["temperature_K"].to_numpy(dtype=float)
        figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)
        for axis, names, ylabel, legend_title in (
            (axes[0], LENGTHS, "Cell length (Å)", "Lengths"),
            (axes[1], ANGLES, "Cell angle (degrees)", "Angles"),
        ):
            for name in names:
                label, color, marker = STYLES[name]
                scale = args.length_scale if name in LENGTHS else 1.0
                center = summary[f"{name}_mean"].to_numpy(dtype=float) / scale
                axis.plot(temperature, center, label=label, color=color, marker=marker, linewidth=1.8, markersize=4.2)
                if args.uncertainty != "none":
                    spread = summary[f"{name}_{suffix}"].to_numpy(dtype=float) / scale
                    axis.fill_between(temperature, center - spread, center + spread, color=color, alpha=0.13, linewidth=0)
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            axis.set_axisbelow(True)
            axis.legend(title=legend_title, frameon=False, ncol=3)
            axis.margins(x=0.02)
        axes[1].set_xlabel("Temperature (K)")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=args.dpi)
        if args.show:
            plt.show()
        plt.close(figure)
        print(f"Wrote {output}")
    except (OSError, ValueError, FileNotFoundError, FileExistsError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
