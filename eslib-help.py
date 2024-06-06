#!/usr/bin/env python
import os
import fnmatch
import ast
import argparse
from eslib.input import str2bool, slist
from eslib.formatting import esfmt

description = "Search for scripts and their description in 'eslib'."

IGNORED_FILES = [
    'eslib/scripts/eslib.py',
    # Add more files as needed
]

def get_relative_path(master_filepath, filepath):
    # Get the relative path of 'filepath' with respect to 'master_filepath'
    relative_path = os.path.relpath(filepath, os.path.dirname(master_filepath))
    
    # Split the relative path into folder and filename
    folder, filename = os.path.split(relative_path)
    
    return folder, filename

def find_python_files_in_folders(folders,current_directory):
    python_files = []
    for folder in folders:
        folder_path = os.path.join(current_directory, folder)
        for root, _, files in os.walk(folder_path):
            for filename in fnmatch.filter(files, '*.py'):
                filepath = os.path.join(root, filename)
                python_files.append(filepath)
    return python_files

def find_main_functions_in_files(filepaths):
    files_with_main = []
    for filepath in filepaths:
        if filepath in IGNORED_FILES:
            continue  # Skip ignored files
        with open(filepath, 'r') as file:
            try:
                tree = ast.parse(file.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == 'main':
                        files_with_main.append(filepath)
                        break
            except SyntaxError:
                print(f"Syntax error in {filepath}. Skipping.")
    return files_with_main

def read_global_variable(filepath, variable_name):
    with open(filepath, 'r') as file:
        code = file.read()
        tree = ast.parse(code)

        # Find all global assignments
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        # Found the variable assignment
                        value_node = node.value
                        # Depending on the complexity, you may need to traverse the value_node
                        # For simple cases, we can evaluate the value_node using `ast.literal_eval`
                        try:
                            value = ast.literal_eval(value_node)
                            return value
                        except (ValueError, TypeError, SyntaxError):
                            # If ast.literal_eval fails, the expression is too complex
                            # You may want to handle this case differently based on your requirements
                            pass
    return None  # Variable not found

def extract_description_from_files(filepaths):
    descriptions = [None]*len(filepaths)
    for n,filepath in enumerate(filepaths):
        if filepath in IGNORED_FILES:
            continue  # Skip ignored files
        description = "Not found"  # Default value
        try:
            description = read_global_variable(filepath,"description")
        except:
            description = "Not found"
        if description is None:
            description = "Not found"
        descriptions[n] =(filepath, description)
    return descriptions

#---------------------------------------#
def prepare_args(description):
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    folders = [None,"bec","build","charges","convert","descriptors","dipole","drive","inspect","metrics","modes","nn","oxn","plot","properties","text","time-series"]
    parser.add_argument("-f", "--folders"     , **argv,type=slist   , help="folders to search for Python files in (default: %(default)s)", default=None, choices=folders)
    parser.add_argument("-s", "--show_folders"     , action='store_true', help="show folders only")
    parser.add_argument("-d", "--descriptions", **argv, type=str2bool, help="whether to print descriptions (default: %(default)s)", default=True)
    return parser# .parse_args()

@esfmt(prepare_args,description)
def main(args):
    
    this_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(this_file) + "/eslib/scripts"
    if args.folders is None:
        folders = [os.path.dirname(this_file)]
    else:
        folders = [ current_directory + "/" + f for f in args.folders ] 

    print("\n\tLooking for scripts in '{:s}'\n".format(current_directory))
    python_files = find_python_files_in_folders(folders,current_directory)
    main_functions = find_main_functions_in_files(python_files)
    if args.descriptions:
        descriptions = extract_description_from_files(main_functions)
    else:
        descriptions = [ (a,None) for a in main_functions ]

    folder_dict = {}
    for (filepath, description) in descriptions or []:
        folder, filename = get_relative_path(this_file, filepath)
        if folder not in folder_dict:
            folder_dict[folder] = []
        folder_dict[folder].append((filename, description))

    # Print the grouped relative folders, file names, and descriptions
    # ANSI escape code for bold blue text
    BOLD_BLUE = "\033[1;34m"
    # ANSI escape code to reset text attributes
    RESET = "\033[0m"
    GREEN = "\033[0;32m"

    for folder, file_descriptions in folder_dict.items():
        # Print the folder name in bold blue
        print("\t{:s}{:s}:{:s}".format(BOLD_BLUE, folder, RESET))
        if not args.show_folders:
            max_filename_length = max(len(filename) for filename, _ in file_descriptions)
            for filename, description in file_descriptions:
                if args.descriptions:
                    # Only print the filename
                    print("\t - {:s}{:{}s}{:s}: {:s}".format(GREEN, filename, max_filename_length + 1, RESET, description if description else "No description found"))
                else:
                    # Print filename and description
                    print("\t - {:s}{:s}{:s}".format(GREEN, filename, RESET))
            print()


if __name__ == "__main__":
    main()