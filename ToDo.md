# General Improvements
- improve documentation for all the functions
- remove old scripts
- improve code quality with type hinting
- provide a clean way to install this repository

# To Do
- create "backup file" for `soap.py`
- find a way to restart `fps.py` from an interrupted script
- find an expression for the effective mass of a normal mode
- implement a `CLI` based on `argparse` to manipulate, inspect and perform calculations on trajectories
- move changes made to `MACE` into `eslib`
- add automatic printing of user messages in scripts
- add a `README.md` for each folder
- use other personal repositories
- consider using `yield` in `aseio.py` for long trajectories

# Working on 
- add automatic tests with `pytest`
- implement finite difference methods for vibrational modes exploiting symmetries (as done in `phononpy`) using `spglib`
- find a way to "standardize" `DipoleModel`, `MACEModel` and `Calculator`: I need to build a class on top of `ase.calculator`
- implement a class to deal with `i-PI` properties
- add a script to transfrom an atomic structure into its Niggli, Minkowski, and standard form.