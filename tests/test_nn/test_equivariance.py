import pytest
import numpy as np
from eslib.scripts.nn.test_e3nn import translation, rotation, pbc, permute
from eslib.functions import suppress_output
from ase.io import read

torun = {
    "aile3nn-water" :
    {
        "instructions" : "tests/models/aile3nn-water/instructions.json",
        "parameters"   : "tests/models/aile3nn-water/parameters.pth",
        "structure"    : "tests/structures/water/water.au.xyz",
    },
    "aile3nn-LiNbO3" :
    {
        "instructions" : "tests/models/aile3nn-LiNbO3/instructions.json",
        "parameters"   : "tests/models/aile3nn-LiNbO3/parameters.pth",
        "structure"    : "tests/structures/LiNbO3/LiNbO3.au.extxyz",
    },
}

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

@pytest.fixture(params=[#(translation, 'translation'),\
                        #(rotation, 'rotation'),\
                        #(pbc, 'pbc'),\
                        (permute,'permute')])
def functions(request):
    return request.param

@pytest.mark.parametrize("name, tests", generate_all_torun(torun).items())
def test_check_e3nn_equivariance(name, tests, functions):
    function, function_name = functions
    print("Running '{:s}' test for '{:s}' ... ".format(function_name,name))    
    with suppress_output():
        result = function(tests)
    if isinstance(result, np.ndarray):
        comparison = np.zeros(result.shape)
    elif isinstance(result, (float, int)):
        comparison = 0.0
    else:
        comparison = None
    output = np.allclose(result, comparison)
    if not output:
        pass # just for debugging
    assert output