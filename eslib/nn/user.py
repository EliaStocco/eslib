import warnings
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
# from eslib.nn.network import iPIinterface
from typing import Union
import importlib
import json
import torch

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
    
def get_model(instructions:str,parameters:str=None)->Union[torch.nn.Module]:
    """Return a Neural Network given a JSON file with the `instructions` to instantiate it and the `parameters` to be used."""
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
    
    if not isinstance(model,torch.nn.Module):
        raise ValueError("'model' should be a 'torch.nn.Module' object.")
    
    from eslib.nn.network import iPIinterface
    if isinstance(model,iPIinterface):
        try : 
            N = model.n_parameters()
            print("\tLoaded model has {:d} parameters".format(N))
        except :
            print("\tCannot count parameters")

        # Store the chemical species that will be used during the simulation.
        if "chemical-symbols" in instructions:
            # model._symbols = instructions["chemical-symbols"]
            try: model.store_chemical_species(instructions["chemical-symbols"])
            except: pass
        else:
            print("!Warning: no chemical symbols provided in the input json file. \
                  You need to provide them before using 'get', 'get_jac', or 'get_value_and_jac'.")

    # Load the parameters from the saved file
    if parameters is not None:
        checkpoint = torch.load(parameters)
        # https://stackoverflow.com/questions/63057468/how-to-ignore-and-initialize-missing-keys-in-state-dict
        model.load_state_dict(checkpoint,strict=False)

    model.eval()

    return model

def get_function(module,function):
    # Step 4: Use importlib to import the module dynamically
    module = importlib.import_module(module)

    # Step 5: Call the function from the loaded module
    function = getattr(module, function)
    
    return function
