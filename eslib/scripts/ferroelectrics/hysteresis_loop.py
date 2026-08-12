#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.formatting import esfmt
from eslib.input import str2bool
from eslib.plot import use_default_style

#---------------------------------------#
description = "Produce the hysteresis loop plot and analysis."
#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-d" , "--dipole"               , **argv, type=str, required=True , help="file with the dipoles")
    parser.add_argument("-dc" , "--dipole_column"       , **argv, type=int, required=False, help="first column with the dipoles (default: %(default)s)", default=0)
    parser.add_argument("-e", "--electric_field"        , **argv, type=str, required=True , help="file with the electric field ")
    parser.add_argument("-ec", "--electric_field_column", **argv, type=int, required=False, help="first column with the electric field (default: %(default)s)", default=0)
    parser.add_argument("-v", "--volume"                , **argv, type=str, required=True , help="file with the volumes")
    parser.add_argument("-vc", "--volume_column"        , **argv, type=int, required=False, help="column with the volumes (default: %(default)s)", default=0)
    parser.add_argument("-p", "--plot"                  , **argv, type=str, required=False, help="output plot file (default: %(default)s)", default="hysteresis-loop.pdf")
    parser.add_argument("-o", "--output"                , **argv, type=str, required=False, help="output CSV file (default: %(default)s)", default="hysteresis-loop.csv")
    parser.add_argument("-rdm", "--remove_dipole_mean"  , **argv, type=str2bool, required=False, help="remove the mean from each dipole component (default: %(default)s)", default=True)
    parser.add_argument("-s", "--size"                  , **argv, type=float, required=False, help="marker size (default: %(default)s)", default=4)
    return  parser


#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #------------------#
    print(f"\tReading the dipoles from file '{args.dipole}' ... ", end="")
    usecols = [ args.dipole_column+n for n in range(3)]
    dipoles = np.loadtxt(args.dipole,usecols=usecols)
    print("done")
    
    #------------------#
    print(f"\tReading the electric field from file '{args.electric_field}' ... ", end="")
    usecols = [ args.electric_field_column+n for n in range(3)]
    efield = np.loadtxt(args.electric_field,usecols=usecols)
    print("done")

    #------------------#
    print(f"\tReading the volumes from file '{args.volume}' ... ", end="")
    volumes = np.loadtxt(args.volume, usecols=args.volume_column, ndmin=1)
    print("done")
    
    assert efield.shape == dipoles.shape, "different shapes"
    assert volumes.shape[0] == dipoles.shape[0], "different number of volumes and dipoles"
    if np.any(volumes == 0):
        raise ValueError("Volumes must be nonzero.")
    if args.remove_dipole_mean:
        dipoles -= np.mean(dipoles, axis=0, keepdims=True)
        print("\tRemoved the mean from each dipole component")
    polarizations = dipoles / volumes[:, np.newaxis]
    dipole_prefix = r"\Delta " if args.remove_dipole_mean else ""

    use_default_style()
    fig = plt.figure(figsize=(9, 11))
    grid = fig.add_gridspec(4, 3, height_ratios=(1, 1, 0.9, 0.9))
    correlation_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    direction_axis = fig.add_subplot(grid[1, :2])
    volume_axis = fig.add_subplot(grid[1, 2])
    efield_axis = fig.add_subplot(grid[2, :])
    polarization_axis = fig.add_subplot(grid[3, :], sharex=efield_axis)
    components = ("x", "y", "z")

    for index, component in enumerate(components):
        ax = correlation_axes[index]
        ax.scatter(efield[:, index], polarizations[:, index], s=args.size)
        ax.set_xlabel(rf"$E_{component}$")
        ax.set_ylabel(rf"${dipole_prefix}P_{component}$")
        ax.grid()

    field_norms = np.linalg.norm(efield, axis=1)
    nonzero = field_norms > np.finfo(float).eps
    if not np.any(nonzero):
        raise ValueError("The electric field is zero for every sample.")

    # The leading right singular vector is the direction shared by all field vectors.
    _, _, right_singular_vectors = np.linalg.svd(efield[nonzero], full_matrices=False)
    direction = right_singular_vectors[0]
    direction *= np.sign(direction[np.argmax(np.abs(direction))])
    field_amplitude = efield @ direction
    dipole_along_field = dipoles @ direction
    polarization_along_field = polarizations @ direction
    print("\tElectric-field direction:", direction)

    data = np.column_stack((
        efield, dipoles, volumes, polarizations, field_amplitude,
        dipole_along_field, polarization_along_field,
    ))
    header = (
        "Electric-field direction: " + " ".join(f"{value:.16g}" for value in direction)
        + "\nEx,Ey,Ez,dx,dy,dz,volume,Px,Py,Pz,E_along_direction,d_along_direction,P_along_direction"
    )
    print(f"\tSaving data to file '{args.output}' ... ", end="")
    np.savetxt(args.output, data, delimiter=",", header=header)
    print("done")

    direction_axis.scatter(field_amplitude, polarization_along_field, s=args.size)
    direction_axis.set_xlabel(r"$\mathbf{E} \cdot \hat{\mathbf{e}}$")
    direction_axis.set_ylabel(rf"${dipole_prefix}\mathbf{{P}} \cdot \hat{{\mathbf{{e}}}}$")
    direction_axis.grid()

    volume_axis.scatter(field_amplitude, volumes, s=args.size)
    volume_axis.set_xlabel(r"$\mathbf{E} \cdot \hat{\mathbf{e}}$")
    volume_axis.set_ylabel("volume")
    volume_axis.grid()

    steps = np.arange(efield.shape[0])
    for index, component in enumerate(components):
        efield_line, = efield_axis.plot(steps, efield[:, index], label=rf"$E_{component}$")
        efield_axis.scatter(steps, efield[:, index], s=args.size, color=efield_line.get_color())
        polarization_line, = polarization_axis.plot(steps, polarizations[:, index], label=rf"${dipole_prefix}P_{component}$")
        polarization_axis.scatter(steps, polarizations[:, index], s=args.size, color=polarization_line.get_color())

    efield_axis.set_ylabel(r"$\mathbf{E}$")
    polarization_axis.set_ylabel(rf"${dipole_prefix}\mathbf{{P}}$")
    polarization_axis.set_xlabel("sample index")
    for ax in (efield_axis, polarization_axis):
        ax.grid()
        ax.legend()

    zero_crossings = np.flatnonzero(field_amplitude[:-1] * field_amplitude[1:] < 0)
    zero_positions = zero_crossings + field_amplitude[zero_crossings] / (
        field_amplitude[zero_crossings] - field_amplitude[zero_crossings + 1]
    )
    zero_positions = np.unique(
        np.concatenate((np.flatnonzero(field_amplitude == 0), zero_positions))
    )
    for position in zero_positions:
        for ax in (efield_axis, polarization_axis):
            ax.axvline(position, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    print(f"\tSaving plot to file '{args.plot}' ... ", end="")
    fig.savefig(args.plot)
    print("done")
    plt.close(fig)

#---------------------------------------#
if __name__ == "__main__":
    main()
