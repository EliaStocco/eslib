import ast
import os
import subprocess

import pytest

folders = ["."]
exclude = ["./eslib/functional.py", "./eslib/old/classes.py", "./eslib/tests"]

def check_main_function(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            return True
    return False

def run_script_with_args(script, args=['-h']):
    try:
        result = subprocess.run(
            [script] + args,
            check=True,
            capture_output=True,
            text=False
        )
        # print("Script output:\n", result.stdout)
        # print("Script executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print("Error while executing the script '{:s}'".format(script))
        # print("Return code:", e.returncode)
        # print("Error output:\n", e.stderr)
        return False
    except Exception as e:
        print("An unexpected error occurred:", str(e))
        return False
    except:
        return False
    
# @pytest.mark.parametrize("folder", folders)
# def test_help(folder):
#     # Directory containing your project files

#     # Traverse through all Python files in the project directory
#     successfull = list()
#     for root, _, files in os.walk(folder):
#         for file in files:
#             if file.endswith(".py"):
#                 file_path = os.path.join(root, file)
#                 if file_path in exclude: 
#                     continue
#                 if check_main_function(file_path):
#                     successfull.append(run_script_with_args(file))

#     # Use an assertion to ensure that all scripts executed successfully
#     assert all(successfull), "Not all scripts executed successfully"

# if __name__ == "__main__":
#     for folder in folders:
#         test_help(folder)

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
#             "program": "/home/stoccoel/google-personal/codes/eslib/tests/test_help/test_help.py",
#             "cwd" : "/home/stoccoel/google-personal/codes/eslib/tests/test_help/",
#             "console": "integratedTerminal",
#             "justMyCode": true,
#         }
#     ]
# }