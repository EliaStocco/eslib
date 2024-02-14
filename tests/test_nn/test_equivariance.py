import pytest
import numpy as np
from eslib.scripts.nn.test_e3nn import translation, rotation, pbc
from eslib.functions import suppress_output

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

@pytest.fixture(params=[(translation, 'translation'), (rotation, 'rotation'), (pbc, 'pbc')])
#@pytest.fixture(params=[(rotation, 'rotation'), (pbc, 'pbc')])
def functions(request):
    return request.param

@pytest.mark.parametrize("name, tests", torun.items())
def test_check_e3nn_equivariance(name, tests, functions):
    function, function_name = functions
    print("\tRunning '{:s}' test for '{:s}' ... ".format(function_name,name),end="")    
    with suppress_output():
        result = function(tests)
    if isinstance(result, np.ndarray):
        comparison = np.zeros(result.shape)
    elif isinstance(result, (float, int)):
        comparison = 0.0
    else:
        comparison = None
    print("done")
    output = np.allclose(result, comparison)
    if not output:
        pass # just for debugging
    assert output