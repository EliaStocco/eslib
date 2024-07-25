import os, sys
from contextlib import contextmanager
from eslib.functions import Dict2Obj, args_to_dict
from eslib.functions import add_default
from datetime import datetime
import inspect
import json
from eslib.show import dict_to_list
import colorama
from colorama import Fore, Style
# from icecream import ic
colorama.init(autoreset=True)

os.environ["XDG_SESSION_TYPE"] = "xcb"

float_format = '%24.12e' # Elia Stocco float format

#---------------------------------------#
error           = "***Error***"
warning         = "***Warning***"
message         = "***Message***"
closure         = "Job done :)"
closure_error   = "***An error occurred :(***"
input_arguments = "Input arguments:"
everythingok    = "Everything ok!"

#---------------------------------------#
# colors
error           = Fore.RED     + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
closure_error   = Fore.RED     + Style.BRIGHT + closure_error.replace("*","")   + Style.RESET_ALL
closure         = Fore.BLUE    + Style.BRIGHT + closure                 + Style.RESET_ALL
input_arguments = Fore.GREEN   + Style.NORMAL + input_arguments         + Style.RESET_ALL
warning         = Fore.MAGENTA + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
message         = Fore.MAGENTA + Style.BRIGHT + message.replace("*","") + Style.RESET_ALL
everythingok    = Fore.BLUE    + Style.BRIGHT + everythingok            + Style.RESET_ALL


#---------------------------------------#
# def print_python_info():
#     # Print Python version
#     print("Python Version:", sys.version)

#     # Print Python executable path
#     print("Python Executable Path:", sys.executable)

#     # Print Conda environment
#     conda_env = os.environ.get('CONDA_DEFAULT_ENV')
#     if conda_env:
#         print("Conda Environment:", conda_env)
#     else:
#         print("Not using Conda environment.")

#---------------------------------------#
def line(start="",end="",N=30,mult=1):
    print(start+"-"*mult*N+end)

#---------------------------------------#
def get_path(main):
    global_path = inspect.getfile(main)
    local_path = os.path.basename(global_path)
    return local_path, global_path

#---------------------------------------#
def esfmt(prepare_parser:callable=None, description:str=None,documentation:str=None):
    """Decorator for the 'main' function of many scripts."""

    #---------------------------------------#
    # Description of the script's purpose
    description = description if description is not None else "Script without description."
    if documentation is not None:
        documentation = documentation.replace("\n","\n\t")
        documentation = Fore.GREEN  + "\n\tDocumentation:\n\t" + Style.RESET_ALL + documentation

    try: description = Fore.GREEN  + Style.BRIGHT + description + Style.RESET_ALL
    except: pass
    # print(description)

    start      = datetime.now()
    start_date = start.date().strftime("%Y-%m-%d")
    start_time = start.time().strftime("%H:%M:%S")

    @contextmanager
    def print_header(args:dict,main:callable):
        
        line(start="###")
        try: 
            local_path, global_path = get_path(main)
            print("{:20s}: {:s}".format("script file",local_path))
            print("{:20s}: {:s}".format("script global path",global_path))
            print("{:20s}: {:s}".format("working directory",os.getcwd()))
            vscode_args = json.dumps(sys.argv[1:])
            print("{:20}: \"args\" : {:s} ".format("VScode debugging",vscode_args))
            tmp = [f'"{a}"' if '*' in a else a for a in sys.argv[1:]]
            command_line = ' '.join(tmp)   
            local_path, global_path = get_path(main)
            print("{:20}: {:s} ".format("running script as",local_path),command_line)
        except: 
            pass    
        print("{:20s}:".format("python --version"), sys.version.replace("\n"," "))
        print("{:20s}:".format("which python"), sys.executable.replace("\n"," "))
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            print("{:20s}:".format("conda env"), conda_env)
            index_bin = sys.executable.find('/bin')
            _conda_env = sys.executable[:index_bin].split('/')[-1]
            if _conda_env != conda_env:
                print("{:s}: possible discrepancy between conda environment and python executable.".format(warning))
        else:
            print("not using conda env")
        print("{:20s}: {:s}".format("start date",start_date))
        print("{:20s}: {:s}".format("start time",start_time))
        line(start="###")

        print("\n\t{:s}".format(description))
        if documentation is not None: print(documentation)
        print("\n\t{:s}".format(input_arguments))
        for k in args.__dict__.keys():
            print("\t{:>20s}:".format(k), getattr(args, k))
        print()
    
    def print_end(ok:bool):
        if ok:
            print("\n\t{:s}\n".format(closure))
        else:
            print("\n\t{:s}\n".format(closure_error))

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
        line(end="###")
        print("end date: {:s}".format(end_date))
        print("end time: {:s}s ".format(end_time))
        print("elapsed time: {:s}".format(format_seconds_to_hhmmss(elapsed_seconds)))
        print("elapsed seconds: {:d}s".format(elapsed_seconds))
        line(end="###\n")

    def wrapper(main: callable):
        def wrapped_main(args=dict()):
            # Call the specified prepare_parser function
            args_script = dict()
            # if len(sys.argv) == 1:
            #     argv1 = dict_to_list(args)
            #     sys.argv.extend(argv1)
            
            if prepare_parser is not None:
                parser = prepare_parser(description)
                args_script = parser.parse_args(args=dict_to_list(args))
            if type(args) == dict:
                args = add_default(args,args_to_dict(args_script))
                args = Dict2Obj(args)

            if args is None:
                raise ValueError("code bug")

            # print the script's description and input arguments
            print_header(args,main)

            # run the script
            out = main(args)

            if out is None or out == 0 :
                # print completion message
                print_end(True)
            else:
                print_end(False)

            return out

        return wrapped_main

    return wrapper