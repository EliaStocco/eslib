from ase.io import read
import numpy as np

def generate_all_torun(torun:dict)->dict:
    out = dict()
    for name, test in torun.items():
        out[name] = test
        if "fold" in test:
            pass
        else:
            # check whether the system is periodic
            atoms = read(test['structure'])
            if np.all(atoms.get_pbc()) :
                # in that case let's do the test twice: with and without folding
                out[name]["fold"] = False

                out[name+"+fold"] = test
                out[name+"+fold"]["fold"] = True
            else :
                # if the system is not periodic folding makes no sense
                out[name]["fold"] = False
    return out