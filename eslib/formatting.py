import os
from contextlib import contextmanager
from eslib.functions import Dict2Obj, args_to_dict
from eslib.functions import add_default
from datetime import datetime
import inspect
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

float_format = '%24.12e' # Elia Stocco float format

#---------------------------------------#
error           = "***Error***"
warning         = "***Warning***"
closure         = "Job done :)"
input_arguments = "Input arguments"

#---------------------------------------#
# colors
error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
warning         = Fore.MAGENTA    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL

def line():
    print("-"*30)

def esfmt(prepare_parser:callable, description:str=None):
    """Decorator for the 'main' function of many scripts."""

    #---------------------------------------#
    # Description of the script's purpose
    description = description if description is not None else "Script without description."
    try: description = Fore.GREEN  + Style.BRIGHT + description + Style.RESET_ALL
    except: pass
    # print(description)

    start      = datetime.now()
    start_date = start.date().strftime("%Y-%m-%d")
    start_time = start.time().strftime("%H:%M:%S")

    @contextmanager
    def print_header(args:dict,main:callable):
        
        line()
        try: print("script file: {:s}".format(inspect.getfile(main))) #main.__file__))
        except: pass        
        print("working directory: {:s}".format(os.getcwd()))
        print("start date: {:s}".format(start_date))
        print("start time: {:s}".format(start_time))
        line()

        print("\n\t{:s}".format(description))
        print("\n\t{:s}:".format(input_arguments))
        for k in args.__dict__.keys():
            print("\t{:>20s}:".format(k), getattr(args, k))
        print()
    
    def print_end():
        print("\n\t{:s}\n".format(closure))

        end      = datetime.now()  
        end_date = end.date().strftime("%Y-%m-%d")
        end_time = end.time().strftime("%H:%M:%S")


        elapsed_seconds = int((end - start).total_seconds())

        def format_seconds_to_hhmmss(seconds):
            hours = seconds // (60*60)
            seconds %= (60*60)
            minutes = seconds // 60
            seconds %= 60
            return "%02i:%02i:%02i" % (hours, minutes, seconds)

        # # Convert elapsed time into hours, minutes, and seconds
        # hours   = int(elapsed_seconds // 3600)
        # minutes = int(elapsed_seconds % 3600 // 60)
        # seconds = int(elapsed_seconds % 60)
        line()
        print("end date: {:s}".format(end_date))
        print("end time: {:s}".format(end_time))
        print("elapsed time: {:s}".format(format_seconds_to_hhmmss(elapsed_seconds)))
        print("elapsed seconds: {:d}".format(elapsed_seconds))
        line()

    def wrapper(main: callable):
        def wrapped_main(args=dict()):
            # Call the specified prepare_parser function
            args_script = dict()
            if prepare_parser is not None:
                args_script = prepare_parser(description)
            if type(args) == dict:
                args = add_default(args,args_to_dict(args_script))
                args = Dict2Obj(args)

            if args is None:
                raise ValueError("code bug")

            # print the script's description and input arguments
            print_header(args,main)

            # run the script
            out = main(args)

            # print completion message
            print_end()

            return out

        return wrapped_main

    return wrapper

def matrix2str(matrix,
                 row_names=["x","y","z"], 
                 col_names=["1","2","3"], 
                 exp=False,
                 width=8, 
                 digits=2,
                 prefix="\t",
                 cols_align="^",
                 num_align=">"):
    """
    Print a formatted 3x3 matrix with customizable alignment.

    Parameters:
    - matrix: The 2D matrix to be printed.
    - row_names: List of row names. Default is ["x", "y", "z"].
    - col_names: List of column names. Default is ["1", "2", "3"].
    - exp: If True, format matrix elements in exponential notation; otherwise, use fixed-point notation.
    - width: Width of each matrix element.
    - digits: Number of digits after the decimal point.
    - prefix: Prefix string for each line.
    - cols_align: Alignment for column names. Use '<' for left, '^' for center, and '>' for right alignment.
    - num_align: Alignment for numeric values. Use '<' for left, '^' for center, and '>' for right alignment.

    Example:
    print_matrix(matrix, row_names=["A", "B", "C"], col_names=["X", "Y", "Z"], exp=True, width=10, digits=3, prefix="\t", cols_align="^", num_align=">")
    """
    # Determine the format string for each element in the matrix
    exp = "e" if exp else "f" 
    format_str = f'{{:{num_align}{width}.{digits}{exp}}}'
    format_str_all = [format_str]*matrix.shape[1]
    hello = f'{{:s}}| {{:s}} |' + f'{{:s}}'*matrix.shape[1] + f' |\n'
    # Find the maximum length of row names for formatting
    L = max([ len(i) for i in row_names ])
    row_str = f'{{:>{L}s}}'
    # Construct the header with column names
    text = '{:s}| ' + row_str + ' |' + (f'{{:{cols_align}{width}s}}')*matrix.shape[1] + ' |\n'
    text = text.format(prefix,"",*list(col_names))
    division = prefix + "|" + "-"*(len(text) - len(prefix) - 3) + "|\n"
    text = division + text + division 
    # Add row entries to the text
    for i, row in enumerate(matrix):
        name_str = row_str.format(row_names[i]) if row_names is not None else ""
        formatted_row = hello.format(prefix,name_str,*format_str_all)
        line = formatted_row.format(*list(row))
        text += line
    # Add a final divider and print the formatted matrix
    text += division
    return text