import os
import shutil
from contextlib import contextmanager

import pytest

def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)

def get_main_function(folder:str, file:str)->callable:
    filepath = "eslib.scripts.{:s}.{:s}".format(folder,file).split(".py")[0]
    try:
        return import_from(filepath,"main")
    except Exception as e:
        raise AttributeError(f"Problem importing 'main' from file '{filepath}.py': {e}")
    
@contextmanager
def change_directory():
    # Save the current working directory
    old_directory = os.getcwd()

    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the desired directory relative to the current script
    target_directory = os.path.abspath(os.path.join(current_directory, "../../"))
    os.chdir(target_directory)
    
    try:
        yield
    finally:
        # Restore the original working directory
        os.chdir(old_directory)
    
tmp_folder = "tmp"

torun = {
    "rdf" :
    {
        "folder"  : "analysis",
        "file"   : "rdf.py",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/LiNbO3.au.extxyz",
            "elements"           : ["Li","Nb"],
            "rmax"               : 4,
            "output"             : f"{tmp_folder}/rdf.csv", 
        },
    },
    "gtrdf" :
    {
        "folder"  : "analysis",
        "file"   : "gtrdf.py",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/LiNbO3.au.extxyz",
            "elements"           : ["Li","Nb"],
            "rmax"               : 4,
            "output"             : f"{tmp_folder}/rdf.csv", 
        },
    },
    "check-prop" : # info to txt
    {
        "folder"  : "properties",
        "file"   : "check-properties.py",
        "kwargs"  : {
            "input"              : "tests/structures/333-water/i-pi.properties.out",   
            "output"             : f"{tmp_folder}/prop.pickle", 
        },
        "clean" : False
    },
    "prop2array" : # info to txt
    {
        "folder"  : "properties",
        "file"   : "prop2array.py",
        "kwargs"  : {
            "input"              : f"{tmp_folder}/prop.pickle", 
            "keyword"            : "dipole",   
            "output"             : f"{tmp_folder}/dipole.txt", 
        },
    },
    "extxyz2array" : # info to txt
    {
        "folder"  : "build",
        "file"   : "extxyz2array.py",
        "kwargs"  : {
            "input"              : "tests/structures/water-wire/water-wire.extxyz",   
            "keyword"            : "MACE_dipole",   
            "output"             : f"{tmp_folder}/MACE_dipole.txt", 
        },
    },
    "extxyz2array" : # info to npy
    {
        "folder"  : "build",
        "file"   : "extxyz2array.py",
        "kwargs"  : {
            "input"              : "tests/structures/water-wire/water-wire.extxyz",   
            "keyword"            : "MACE_dipole",   
            "output"             : f"{tmp_folder}/MACE_dipole.npy", 
        },
    },
    "extxyz2array" : # array to txt
    {
        "folder"  : "build",
        "file"   : "extxyz2array.py",
        "kwargs"  : {
            "input"              : "tests/structures/water-wire/water-wire.extxyz",   
            "keyword"            : "MACE_atomic_dipoles",   
            "output"             : f"{tmp_folder}/MACE_atomic_dipoles.txt", 
        },
    },
    "extxyz2array" : # array to txt
    {
        "folder"  : "build",
        "file"   : "extxyz2array.py",
        "kwargs"  : {
            "input"              : "tests/structures/water-wire/water-wire.extxyz",   
            "keyword"            : "MACE_atomic_dipoles",   
            "output"             : f"{tmp_folder}/MACE_atomic_dipoles.npy", 
        },
    },
    # "eval-model-PES" :
    # {
    #     "folder"  : "nn",
    #     "file"   : "eval-model.py",
    #     "kwargs"  : {
    #         "input" : "tests/structures/LiNbO3/geometries.extxyz" ,   
    #         "model" : "tests/models/MACE-LiNbO3/MACE.LiNbO3.pickle",
    #         "output": f"{tmp_folder}/mace.extxyz"
    #     },
    # },
    # "eval-model-PES+save-info-arrays" :
    # {
    #     "folder"  : "nn",
    #     "file"   : "eval-model.py",
    #     "kwargs"  : {
    #         "input" : "tests/structures/LiNbO3/geometries.extxyz" ,   
    #         "model" : "tests/models/MACE-LiNbO3/MACE.LiNbO3.pickle",
    #         "output": f"{tmp_folder}/mace.extxyz",
    #         "names" : ['MACE_energy','MACE_forces'],
    #         "shapes": [[-1,1],[-1,3]],
    #         "data_output" : [f'{tmp_folder}/MACE_energy.txt',f'{tmp_folder}/MACE_forces.txt']
    #     },
    # },
    "backward compatibility" :
    {
        "folder"  : "inspect",
        "file"   : "trajectory-summary.py",
        "kwargs"  : {
            "input" : "tests/structures/back-comp/pos.300K.E=0.00.pickle" ,   
        },
    },
    "extras2dipole" :
    {
        "folder"  : "dipole",
        "file"   : "extras2dipole",
        "kwargs"  : {
            "input"  : "tests/data/i-pi.extras_0",  
            "output" :  "tests/dipoles.txt",  
        },
    },
    "get-iPI-cell" :
    {
        "folder"  : "inspect",
        "file"   : "get-iPI-cell",
        "kwargs"  : {
            "input" : "tests/structures/bulk-water/bulk-water.au.extxyz",            
        },
    },
    "convert-file" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/bulk-water/bulk-water.au.extxyz",   
            "remove_properties"  : "true",   
            "output"             : f"{tmp_folder}/output.extxyz",     
        },
    },
    # "netcdf-write" :
    # {
    #     "folder"  : "convert",
    #     "file"   : "convert-file",
    #     "kwargs"  : {
    #         "input"              : "tests/structures/LiNbO3/long.n=10000.extxyz",   
    #         "output"             : f"{tmp_folder}/output.nc",     
    #     },
    # },
    "netcdf-read" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/long.n=10000.nc",   
            "output"             : f"{tmp_folder}/output.extxyz",     
        },
    },
    "pickle-write" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/long.n=10000.extxyz",   
            "output"             : f"{tmp_folder}/output.pickle",     
        },
        "clean" : False
    },
    "pickle-inspect" : {
        "folder"  : "inspect",
        "file"   : "trajectory-summary.py",
        "kwargs"  : {
            "input" : f"{tmp_folder}/output.pickle",  
        },
        "clean" : False
    },
    "pickle-read" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : f"{tmp_folder}/output.pickle",   
            "input_format"       : "pickle",
            "output"             : f"{tmp_folder}/output.nc",  
        },
    },
    "hdf5-write" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/long.n=10000.extxyz",   
            "output"             : f"{tmp_folder}/output.h5",  
        },
        "clean" : False
    },
    "hdf5-read" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : f"{tmp_folder}/output.h5",   
            "index"              : "120::10",
            "output"             : f"{tmp_folder}/output.extxyz",    
        },
        "clean" : False
    },
    "hdf5-test" :
    {
        "folder"  : "inspect",
        "file"   : "trajectory-summary.py",
        "kwargs"  : {
            "input"              : f"{tmp_folder}/output.h5", 
        },
    },
    "information-and-primitive.py" :
    {
        "folder"  : "inspect",
        "file"   : "information-and-primitive.py",
        "kwargs"  : {
            "input"  : "tests/structures/bulk-water/bulk-water.au.extxyz",   
        },
    },
    "extxyz2pickle" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/bulk-water/bulk-water.au.extxyz",   
            "remove_properties"  : "true",   
            "output"             : f"{tmp_folder}/output.pickle", 
        },
    },
    "pickle2extxyz" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/bulk-water/bulk-water.au.pickle" ,   
            "remove_properties"  : "true",   
            "output"             : f"{tmp_folder}/output.extxyz",    
        },
    },
    "trajectory-summary" :
    {
        "folder"  : "inspect",
        "file"   : "trajectory-summary.py",
        "kwargs"  : {
            "input" : "tests/structures/bulk-water/bulk-water.au.extxyz" ,   
        },
    },
    "trajectory-summary-empty" :
    {
        "folder"  : "inspect",
        "file"   : "trajectory-summary.py",
        "kwargs"  : {
            "input" : "tests/structures/bulk-water/bulk-water.au.empty.extxyz" ,   
        },
    },
    "ipi" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/i-pi.positions_0.xyz",   
            "input_format"       : "ipi",   
            "output"             : f"{tmp_folder}/output.extxyz",   
        },
    },
    "long-raw" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/raw-ipi.n=10000.xyz",   
            "input_format"       : "ipi",   
            "output"             : f"{tmp_folder}/output.extxyz",  
        },
    },
    "long" :
    {
        "folder"  : "convert",
        "file"   : "convert-file",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/long.n=10000.extxyz",   
            "input_format"       : "extxyz",   
            "output"             : f"{tmp_folder}/output.extxyz",     
        },
    },
    "rmse-dipole" :
    {
        "folder"  : "metrics",
        "file"   : "compute-metrics",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/geometries.extxyz",   
            "expected"           : "dipole",
            "predicted"          : "MACE_dipole",
            "metrics"            : "rmse",
            "statistics"         : "True",
            "output"             : f"{tmp_folder}/rmse.json",     
        },
    },
    "rmse-bec" :
    {
        "folder"  : "metrics",
        "file"   : "compute-metrics",
        "kwargs"  : {
            "input"              : "tests/structures/LiNbO3/geometries.extxyz",   
            "expected"           : "BEC",
            "predicted"          : "MACE_BEC",
            "metrics"            : "rmse",
            "statistics"         : "True",
            "output"             : f"{tmp_folder}/bec.json",     
        },
    },
}


@pytest.mark.parametrize("name, test", torun.items())
def test_scripts(name, test):

    print("Running test '{:s}'.".format(name))   
    main = get_main_function(test["folder"],test["file"])
    with change_directory():
        if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)

        # from eslib.classes.aseio import set_parallel
        # from copy import copy 

        # kwargs = test["kwargs"]
        # with timing(True):
        #     set_parallel(True)
        #     main(copy(kwargs))
        
        # with timing(True):
        #     set_parallel(False)
        #     main(copy(kwargs))

        main(test["kwargs"])

        if 'clean' in test and not test['clean']:
            pass
        else:
            shutil.rmtree(tmp_folder)

    pass    

if __name__ == "__main__":
    for name, test in torun.items():
        test_scripts(name, test)
      
# { 
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "/home/stoccoel/google-personal/codes/eslib/tests/test_scripts/test_scripts.py",
#             "cwd" : "/home/stoccoel/google-personal/codes/eslib/",
#             "console": "integratedTerminal",
#             "justMyCode": true,
#         }
#     ]
# }