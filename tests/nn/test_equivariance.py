import pytest
import numpy as np
from eslib.scripts.nn.test_e3nn import translation, rotation

torun = {
    "aile3nn-water" :
    {
        "instructions" : "tests/models/aile3nn-water/instructions.json",
        "parameters"   : "tests/models/aile3nn-water/parameters.pth",
        "structure"    : "tests/structures/water/water.au.xyz",
        "number"       : 100
    },
    "aile3nn-LiNbO3" :
    {
        "instructions" : "tests/models/aile3nn-LiNbO3/instructions.json",
        "parameters"   : "tests/models/aile3nn-LiNbO3/parameters.pth",
        "structure"    : "tests/structures/LiNbO3/LiNbO3.au.extxyz",
        "number"       : 100
    },
}

@pytest.mark.parametrize("name, tests", torun.items())
def test_check_e3nn_equivariance(name, tests):
    print("\tRunning test for {:s}".format(name))
    functions = [translation,rotation]
    for func in functions:
        result:np.ndarray = func(tests)
        assert np.allclose(result, np.zeros(result.shape))
