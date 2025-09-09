
# To Do
- optimize `eslib/fortran/msd/_msd_fort.f90` and use `MPI`
- fix all `time-series` scripts as done for `tacf+IR.py`
- `dot` and `rbc` in `physical_tensor.py` should be made the same function (debugging necessary)
- the dynamical matrix eigenvetors are not correctly normalized when computed from `phonopy`

# General
- improve documentation for all the functions
- remove old scripts
- improve code quality with type hinting
- provide a clean way to install this repository (use `environment.yml`)
- add automatic tests with `pytest`
- use `logging` (or `eslog`) instead of `print` to better indentation and profiling/timing
- remove `hdf5` dependencies
- find an expression for the effective mass of a normal mode
- add a `README.md` for each folder
- consider using `yield` in `aseio.py` for long trajectories (not sure)
- add documentations as `pdf` (e.g. `phonon projection`)
- clean/remove `DipoleModel` and/or `MACEModel` 
- splits this repo into many smaller repos, each one taking care of something different:
    - `IO` and manipulation of `extxyz` (a `ase` wrapper)
    - general tools for profiling, reading data from file, `pandas` dataframe, etc
    - time-series manipulation (IR, susceptibility, VDOS, etc)
    - the rest can remain here


# Working on 
- implement finite difference methods for vibrational modes exploiting symmetries (as done in `phononpy`) using `spglib`

# Improvements:
- find a way to "standardize" `DipoleModel`, `MACEModel` and `Calculator`: I need to build a class on top of `ase.calculator`
