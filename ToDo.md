# General Improvements
- improve documentation for all the functions
- remove old scripts
- improve code quality with type hinting
- provide a clean way to install this repository
- add automatic tests with `pytest`
- use `logging` instead of `print` to better indentation and profiling/timing

# General Ideas
- create "backup file" for `soap.py`
- find a way to restart `fps.py` from an interrupted script
- find an expression for the effective mass of a normal mode
- implement a `CLI` based on `argparse` to manipulate, inspect and perform calculations on trajectories
- move changes made to `MACE` into `eslib`
- add a `README.md` for each folder
- use other personal repositories
- consider using `yield` in `aseio.py` for long trajectories
- add documentations ad `pdf` (e.g. `phonon projection`)

# To Do
- save `Properties` to `hdf5` files in stead of `pickle`
- optimize `hdf5` `IO` by defining a way to store trajectories optimized for MD
- `dot` and `rbc` in `physical_tensor.py` should be made the same function (debugging necessary)

# Working on 
- implement finite difference methods for vibrational modes exploiting symmetries (as done in `phononpy`) using `spglib`

# Improvements:
- find a way to "standardize" `DipoleModel`, `MACEModel` and `Calculator`: I need to build a class on top of `ase.calculator`
- implement a class to deal with `i-PI` properties

# Bugs
- the dynamical matrix eigenvetors are not correctly normalized when computed from `phonopy`