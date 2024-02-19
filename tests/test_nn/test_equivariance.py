import pytest
import numpy as np
from eslib.scripts.nn.test_e3nn import translation, rotation, pbc, permute
from eslib.functions import suppress_output
from .functions import generate_all_torun

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

@pytest.fixture(params=[(translation, 'translation'),\
                        (rotation, 'rotation'),\
                        (pbc, 'pbc'),\
                        (permute,'permute')])
def functions(request):
    return request.param

# @pytest.mark.parametrize("name, tests", generate_all_torun(torun).items())
# def test_check_e3nn_equivariance(name, tests, functions):
#     function, function_name = functions
#     print("Running '{:s}' test for '{:s}'.".format(function_name,name))    
#     with suppress_output():
#         result = function(tests)
#     if isinstance(result, np.ndarray):
#         comparison = np.zeros(result.shape)
#     elif isinstance(result, (float, int)):
#         comparison = 0.0
#     else:
#         comparison = None
#     output = np.allclose(result, comparison)
#     if not output:
#         pass # just for debugging
#     assert output