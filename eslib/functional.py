import json5 as json5 as json
import os
from .functions import add_default

def load_json_config(func):
    def wrapper(**kwargs):
        # Get the name of the function
        func_name = func.__name__

        if "opts" in kwargs :
            opts = kwargs["opts"]
        else :
            opts = dict()

        # Check if a JSON file exists for this function
        json_file_path = os.path.join('{:s}.json'.format(func_name))
        if not os.path.exists(json_file_path):
            json_file_path = os.path.join('{:s}/{:s}.json'.format("opts",func_name))
            if not os.path.exists(json_file_path):
                json_file_path = None

        if json_file_path is not None:
            # Load and merge JSON values into the opts dictionary
            with open(json_file_path, 'r') as json_file:
                json_data = json.load(json_file)
                opts = add_default(opts,json_data)

        # Call the original function with the updated opts dictionary
        kwargs["opts"] = opts
        return func(**kwargs)

    return wrapper

