from ase.io import read
from ase import Atoms
from .vectorize import easyvectorize
from eslib.functions import read_comments_xyz
import re
import ipi.utils.mathtools as mt
import numpy as np
from eslib.formatting import warning

deg2rad = np.pi / 180.0
abcABC      = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")
abcABCunits = re.compile(r'\{([^}]+)\}')
step        = re.compile(r"Step:\s+(\d+)")

# Example
# trajectory = Trajectory(file)
# structure  = trajectory[0]                               # --> ase.Atoms
# positions  = trajectory.positions                        # --> (N,3,3)
# dipole     = trajectory.call(lambda e: e.info["dipole"]) # --> (N,3)

def info(t,name):
    return t.call(lambda e:e.info[name])
def array(t,name):
    return t.call(lambda e:e.arrays[name])
    
def trajectory(file,format:str=None,check:bool=True,index=":",pbc=True,remove_replicas:bool=False):

    format = format.lower() if format is not None else None
    f = "xyz" if format in ["i-pi","ipi"] else format
    remove_replicas = False if format not in ["i-pi","ipi"] else remove_replicas

    atoms = read(file,index=index,format=f)
    if type(atoms) != list:
        atoms = [atoms]
    for n in range(len(atoms)):
        # atoms[n].info = dict()
        atoms[n].calc = None # atoms[n].set_calculator(None)

    pbc = pbc if pbc is not None else True
    if format in ["i-pi","ipi"]:
        for n in range(len(atoms)):
            atoms[n].info = dict()

        # try : 
        
        comments = read_comments_xyz(file,Nread=len(atoms))

        if pbc:
            strings = [ abcABC.search(comment) for comment in comments ]
            cells = np.zeros((len(strings),3,3))
            for n,cell in enumerate(strings):
                a, b, c = [float(x) for x in cell.group(1).split()[:3]]
                alpha, beta, gamma = [float(x) * deg2rad for x in cell.group(1).split()[3:6]]
                cells[n] = mt.abc2h(a, b, c, alpha, beta, gamma)

        if remove_replicas:
            strings = [ step.search(comment).group(1) for comment in comments ]
            steps = np.asarray([int(i) for i in strings],dtype=int)
            test, indices = np.unique(steps, return_index=True)
            if steps.shape != test.shape:
                print("\t{:s}: there could be replicas. Specify '-rr/--remove_replicas true' to remove them.".format(warning))
            # if len(indices) != len(steps):
            #     pass
            atoms = [atoms[index] for index in indices]
            # for n in range(len(atoms)):
            #     atoms[n].info["step"] = indices[n]

        # matches = re.findall(abcABCunits,comments[0])
        # if len(matches) != 2 :
        #     raise ValueError("wrong number of matches")
        # else :
        #     units = {
        #         "positions" : matches[0],
        #         "cell" : matches[1]
        #     }
        if pbc:
            for n in range(len(atoms)):
                atoms[n].set_cell(cells[n].T)
                atoms[n].set_pbc(True)

        # except:
        #     pass
    
    return easyvectorize(Atoms)(atoms)





