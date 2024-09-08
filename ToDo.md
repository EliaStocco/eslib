# General Improvements
- improve documentation for all the functions
- remove old scripts
- improve code quality with type hinting
- provide a clean way to install this repository
- add automatic tests with `pytest`

# General Ideas
- create "backup file" for `soap.py`
- find a way to restart `fps.py` from an interrupted script
- find an expression for the effective mass of a normal mode
- implement a `CLI` based on `argparse` to manipulate, inspect and perform calculations on trajectories
- move changes made to `MACE` into `eslib`
- add a `README.md` for each folder
- use other personal repositories
- consider using `yield` in `aseio.py` for long trajectories

# To Do
- save `Properties` to `netcdf` files in stead of `pickle`
- `dot` and `rbc` in `physical_tensor.py` should be made the same function (debugging necessary)

# Working on 
- implement finite difference methods for vibrational modes exploiting symmetries (as done in `phononpy`) using `spglib`

# Improvements:
- find a way to "standardize" `DipoleModel`, `MACEModel` and `Calculator`: I need to build a class on top of `ase.calculator`
- implement a class to deal with `i-PI` properties