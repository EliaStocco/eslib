import pytest
import json
import os
import shutil
from eslib.scripts.nn.train_e3nn_model import main

DELETE = True

torun = {
    "aile3nnOxN-water-light" :
    {
        "training"  : "tests/test_train/aile3nnOxN-water-light/training.json",
        "network"   : "tests/test_train/aile3nnOxN-water-light/network.json",
    }
}


@pytest.mark.parametrize("name, test", torun.items())
def test_train(name, test):
    
    print("Running '{:s}' test.".format(name))   
    main(test) 

    if DELETE:
        with open(test["training"]) as f:
            data = json.load(f)
        for key in ["output_folder","checkpoint_folder","info-file"]:
            if key in data:
                file = data[key]
                # Remove a file
                try: os.remove(file)
                except: pass
                try: os.rmdir(file)
                except: pass
                try: shutil.rmtree(file)
                except: pass
            
    return 

if __name__ == "__main__":
    for name, test in torun.items():
        test_train(name, test)
    