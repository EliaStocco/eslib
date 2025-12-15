#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------#
description = "Compare two matrices."

legend_options = {
    "facecolor":'white', 
    "framealpha":1,
    "edgecolor":"black",
    "loc": "best",
}

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-a","--matrix_A", **argv, required=True, type=str, help="txt file with matrix A")
    parser.add_argument("-b","--matrix_B", **argv, required=True, type=str, help="txt file with matrix B")
    parser.add_argument("-o","--output"  , **argv, required=False, type=str,
                        default="diff.png", help="output filename for comparison plot")
    return parser

#---------------------------------------#
def main(args):

    print(f"Reading A: {args.matrix_A}")
    A = np.loadtxt(args.matrix_A)
    print(f"Reading B: {args.matrix_B}")
    B = np.loadtxt(args.matrix_B)

    if A.shape != B.shape:
        raise ValueError("Matrices must have identical shape.")

    #================= Data Extraction =================#
    diag_A = np.diag(A)
    diag_B = np.diag(B)
    off_A  = A[~np.eye(A.shape[0], dtype=bool)]
    off_B  = B[~np.eye(B.shape[0], dtype=bool)]

    diff_diag = np.abs(diag_A - diag_B)
    diff_off  = np.abs(off_A - off_B)
    rel_diag  = diff_diag / np.maximum(np.abs(diag_B), 1e-20)
    rel_off   = diff_off  / np.maximum(np.abs(off_B), 1e-20)

    #================= Plot Settings =================#
    bins=60
    fig,axs = plt.subplots(2,2,figsize=(10,6))
    # fig.suptitle("Matrix comparison: diagonal vs off-diagonal")

    for ax in axs.flat:
        ax.grid(alpha=0.25, linewidth=0.7, linestyle=":")

    #================= (1) Diagonal =================#
    axs[0,0].hist(np.abs(diag_A), bins=bins, density=True, histtype="step", color='blue',
                  linewidth=2, label="A diag")
    axs[0,0].hist(np.abs(diag_B), bins=bins, density=True, histtype="step", color='red',
                  linewidth=2, linestyle="--", label="B diag")
    axs[0,0].set_xscale("log"); axs[0,0].set_yscale("log")
    axs[0,0].set_ylabel("Probability density")
    axs[0,0].set_title("|A| vs |B| (diagonal)")
    axs[0,0].legend(**legend_options)

    #================= (2) Off-Diagonal =================#
    axs[0,1].hist(np.abs(off_A), bins=bins, density=True, histtype="step", color='blue',
                  linewidth=2, label="A off")
    axs[0,1].hist(np.abs(off_B), bins=bins, density=True, histtype="step", color='red',
                  linewidth=2, linestyle="--", label="B off")
    axs[0,1].set_xscale("log"); axs[0,1].set_yscale("log")
    axs[0,1].set_title("|A| vs |B| (off-diagonal)")
    axs[0,1].legend(**legend_options)

    #================= (3) Absolute difference ==========#
    axs[1,0].hist(diff_diag, bins=bins, density=True, histtype="step", color='blue',
                  linewidth=2, label="diag")
    axs[1,0].hist(diff_off, bins=bins, density=True, histtype="step", color='red',
                  linewidth=2, linestyle="--", label="off")
    axs[1,0].set_xscale("log"); axs[1,0].set_yscale("log")
    axs[1,0].set_ylabel("Probability density")
    axs[1,0].set_title("|A - B| distribution")
    axs[1,0].legend(**legend_options)

    #================= (4) Relative difference ==========#
    # Relative difference scatter plot
    axs[1,1].scatter(off_B, rel_off, color='red', s=20, alpha=0.7, label='off')
    axs[1,1].scatter(diag_B, rel_diag, color='blue', s=20, alpha=0.7, label='diag')

    axs[1,1].set_xscale('symlog')
    axs[1,1].set_yscale('symlog')
    axs[1,1].set_xlabel("|B| values")
    axs[1,1].set_ylabel("Relative difference |A-B|/|B|")
    axs[1,1].set_title("Relative difference vs |B|")
    axs[1,1].legend(**legend_options)
    axs[1,1].grid(alpha=0.25, linestyle=':')

    fig.tight_layout()
    fig.savefig(args.output, dpi=350)
    print(f"\nSaved comparison → {args.output}\n")

#---------------------------------------#
if __name__ == "__main__":
    main(prepare_args(description).parse_args())
