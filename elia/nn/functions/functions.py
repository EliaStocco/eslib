import torch
import json5 as json
import importlib
from ase.io import read 
import numpy as np
from prettytable import PrettyTable
import subprocess
from elia.functions import add_default
import warnings

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def get_data_from_dataset(dataset,variable):
    # Extract data for the specified variable from the dataset
    v = getattr(dataset[0],variable)
    data = torch.full((len(dataset),*v.shape),np.nan)
    for n,x in enumerate(dataset):
        data[n,:] = getattr(x,variable)
    return data

def get_class(module_name, class_name):
    try:
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Get the class from the module
        class_obj = getattr(module, class_name)
        
        # Create an instance of the class
        #instance = class_obj()
        
        return class_obj
    
    except ImportError:
        raise ValueError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'.")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
    
def get_model(instructions,parameters:str):

    if type(instructions) == str :

        with open(instructions, "r") as json_file:
            _instructions = json.load(json_file)
        instructions = _instructions

    # instructions['kwargs']["normalization"] = None

    # wxtract values for the instructions
    kwargs = instructions['kwargs']
    cls    = instructions['class']
    mod    = instructions['module']

    # get the class to be instantiated
    # Call the function and suppress the warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore") #, category=UserWarning)
        class_obj = get_class(mod,cls)
    
    # instantiate class
    #try :
    model = class_obj(**kwargs)
    if not model :
        raise ValueError("Error instantiating class '{:s}' from module '{:s}'".format(cls,mod))
    
    try : 
        N = model.n_parameters()
        print("\tLoaded model has {:d} parameters".format(N))
    except :
        print("\tCannot count parameters")
    

    # total_parameters = sum(p.numel() for p in model.parameters())

    # Load the parameters from the saved file
    if parameters is not None:
        checkpoint = torch.load(parameters)
        # https://stackoverflow.com/questions/63057468/how-to-ignore-and-initialize-missing-keys-in-state-dict
        model.load_state_dict(checkpoint,strict=False)

    model.eval()

    # Store the chemical species that will be used during the simulation.
    if "chemical-symbols" in instructions:
        model._symbols = instructions["chemical-symbols"]
    else:
        print("!Warning: no chemical symbols provided in the input json file. You need to provide them before using 'get', 'get_jac', or 'get_value_and_jac'.")

    return model

def get_function(module,function):
    # Step 4: Use importlib to import the module dynamically
    module = importlib.import_module(module)

    # Step 5: Call the function from the loaded module
    function = getattr(module, function)
    
    return function

def bash_as_function(script_path,opts=None):
    """run a bash script by calling this function"""
    default = {
        "print" : False
    }
    opts = add_default(opts,default)
    def wrapper():
        try:
            # Run the Bash script using subprocess
            completed_process = subprocess.run(
                ['bash', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,  # Set this to True if you want to use shell features like piping
                env=None,  # Use the current environment
            )

            # Check the return code to see if the script executed successfully
            if completed_process.returncode == 0:
                if opts["print"] : print("Script executed successfully.")
                if opts["print"] : print("Script output:")
                if opts["print"] : print(completed_process.stdout)
            else:
                if opts["print"] : print("Script encountered an error.")
                if opts["print"] : print("Error output:")
                if opts["print"] : print(completed_process.stderr)
        except:
            print("Error running script '{:s}'".format(script_path))            
        return
    return wrapper