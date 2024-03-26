import pytest
import os
from contextlib import contextmanager

def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)

def get_main_function(folder:str, file:str)->callable:
    filepath = "eslib.scripts.{:s}.{:s}".format(folder,file).split(".py")[0]
    try:
        return import_from(filepath,"main")
    except:
        raise AttributeError("Problem importing 'main' from file '{:s}.py'".format(filepath))
    
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
    

torun = {
    "get-iPI-cell" :
    {
        "folder"  : "inspect",
        "file"   : "get-iPI-cell",
        "kwargs"  : {
            "input" : "tests/structures/bulk-water/bulk-water.au.extxyz",            
        },
    },
}


@pytest.mark.parametrize("name, test", torun.items())
def test_scripts(name, test):

    print("Running '{:s}' test.".format(name))   
    main = get_main_function(test["folder"],test["file"])
    with change_directory():
        main(test["kwargs"])

    

if __name__ == "__main__":
    for name, test in torun.items():
        test_scripts(name, test)
    
# { 
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "/home/stoccoel/google-personal/codes/eslib/tests/test_scripts/test_scritps.py",
#             "cwd" : "/home/stoccoel/google-personal/codes/eslib/tests/test_scripts/",
#             "console": "integratedTerminal",
#             "justMyCode": true,
#         }
#     ]
# }