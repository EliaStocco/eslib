import pytest
import numpy as np
from eslib.functions import suppress_output
from eslib.nn.user import get_model
from ase.io import read

torun = {
    "aile3nn-water" :
    {
        "instructions" : "tests/models/aile3nn-water/instructions.json",
        "parameters"   : "tests/models/aile3nn-water/parameters.pth",
        "structure"    : "tests/structures/water/water.au.xyz",
    },
    "aile3nnOxN-water" :
    {
        "instructions" : "tests/models/aile3nnOxN-water/instructions.json",
        "parameters"   : "tests/models/aile3nnOxN-water/parameters.pth",
        "structure"    : "tests/structures/water/water.au.xyz",
    },
    "aile3nn-LiNbO3" :
    {
        "instructions" : "tests/models/aile3nn-LiNbO3/instructions.json",
        "parameters"   : "tests/models/aile3nn-LiNbO3/parameters.pth",
        "structure"    : "tests/structures/LiNbO3/LiNbO3.au.xsf",
    },
}

@pytest.mark.parametrize("name, tests", torun.items())
def test_check_e3nn_equivariance(name, tests):
    print("Running test '{:s}'.".format(name))  
    #
    atoms = read(tests['structure'],index=0)
    #
    model = get_model(tests['instructions'],tests['parameters'])
    from eslib.nn.network import iPIinterface
    if not isinstance(model,iPIinterface):
        raise ValueError("'model' should be 'iPIinterface'.")
    #
    model.get_from_structure(atoms)
    pass