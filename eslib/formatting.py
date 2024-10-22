import os, sys
from contextlib import contextmanager
from eslib.functions import Dict2Obj, args_to_dict
from eslib.functions import add_default
from datetime import datetime
from argparse import ArgumentParser
import inspect
import json
import psutil
from eslib.show import dict_to_list
import colorama
from colorama import Fore, Style
# from icecream import ic
colorama.init(autoreset=True)

os.environ["XDG_SESSION_TYPE"] = "xcb"

float_format = '%24.12e' # Elia Stocco float format
exp_format = '%24.12e' # Elia Stocco float format
dec_format = '%16.10f' # Elia Stocco float format
complex_format = '%.10f%+.10fj'


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
def line(start="",end="",N=30,mult=1):
    print(start+"-"*mult*N+end)

#---------------------------------------#
def get_path(main):
    global_path = inspect.getfile(main)
    local_path = os.path.basename(global_path)
    return local_path, global_path

#---------------------------------------#
# def format_seconds_to_hhmmss(seconds:int)->str:
#     hours = seconds // (60*60)
#     seconds %= (60*60)
#     minutes = seconds // 60
#     seconds %= 60
#     return "%02i:%02i:%02i" % (hours, minutes, seconds)

def format_seconds_to_hhmmss(seconds: float) -> str:
    """
    Utility function to format seconds into HH:MM:SS format.

    Args:
        seconds (float): Time duration in seconds.

    Returns:
        str: Time formatted as a string in HH:MM:SS format.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

#---------------------------------------#
_description = None
_documentation = None
def esfmt(prepare_parser:callable=None, description:str=None,documentation:str=None):
    """Decorator for the 'main' function of many scripts."""

    #---------------------------------------#
    # Description of the script's purpose
    description = description if description is not None else "Script without description."
    if documentation is not None:
        documentation = documentation.replace("\n","\n\t ")
        documentation = Fore.GREEN  + "\n\t Documentation:\n\t " + Style.RESET_ALL + documentation
    
    global _documentation
    global _description
    _documentation = documentation
    _description = description

    try: description = Fore.GREEN  + Style.BRIGHT + description + Style.RESET_ALL
    except: pass
    # print(description)

    start      = datetime.now()
    start_date = start.date().strftime("%Y-%m-%d")
    start_time = start.time().strftime("%H:%M:%S")

    # @contextmanager
    def print_header(args:dict,main:callable,help=False):
        
        line(start="###")
        try: 
            local_path, global_path = get_path(main)
            print("{:20s}: {:s}".format("script file",local_path))
            print("{:20s}: {:s}".format("script global path",global_path))
            if not help:
                print("{:20s}: {:s}".format("working directory",os.getcwd()))
                vscode_args = json.dumps(sys.argv[1:])
                print("{:20}: \"args\" : {:s} ".format("VScode debugging",vscode_args))
                tmp = [f'"{a}"' if '*' in a else a for a in sys.argv[1:]]
                command_line = ' '.join(tmp)   
                local_path, global_path = get_path(main)
                print("{:20}: {:s} ".format("running script as",local_path),command_line)
        except: 
            pass    
        if not help:
            print("{:20s}:".format("python --version"), sys.version.replace("\n"," "))
            print("{:20s}:".format("which python"), sys.executable.replace("\n"," "))
            conda_env = os.environ.get('CONDA_DEFAULT_ENV')
            print("{:20s}:".format("conda env"), conda_env if conda_env is not None else "none")
            # if conda_env:
            #     print("{:20s}:".format("conda env"), conda_env)
            #     # index_bin = sys.executable.find('/bin')
            #     # _conda_env = sys.executable[:index_bin].split('/')[-1]
            #     # if _conda_env != conda_env:
            #     #     print("{:s}: possible discrepancy between conda environment and python executable.".format(warning))
            #     # else:
            #     #     print()
            # else:
            #     print("no conda env")
            print("{:20s}: {:s}".format("start date",start_date))
            print("{:20s}: {:s}".format("start time",start_time))
        line(start="###")
        
        global _documentation
        global _description
        # global description
    
        if help:
            print("\n{:s}".format(description.replace("\n\t ","\n").replace("\t \t ","\t ")))
        else:
            print("\n\t {:s}".format(description))
        # if  help :
        #     description = None
        if _documentation is not None: 
            if help:
                _documentation = str(_documentation).replace("\n\t ","\n").replace("\t \t ","\t ")
            print(_documentation)
        
        if args is not None:
            print("\n\t {:s}".format(input_arguments))
            for k in args.__dict__.keys():
                print("\t {:>20s}:".format(k), getattr(args, k))
            print()
            
        # if help:
        #     description = None
    
    def print_end(ok:bool):
        if ok:
            print("\n\t {:s}\n".format(closure))
        else:
            print("\n\t {:s}\n".format(closure_error))

        end      = datetime.now()  
        end_date = end.date().strftime("%Y-%m-%d")
        end_time = end.time().strftime("%H:%M:%S")


        elapsed_seconds = int((end - start).total_seconds())


        # # Convert elapsed time into hours, minutes, and seconds
        # hours   = int(elapsed_seconds // 3600)
        # minutes = int(elapsed_seconds % 3600 // 60)
        # seconds = int(elapsed_seconds % 60)
        line(end="###")
        print("end date: {:s}".format(end_date))
        print("end time: {:s}s ".format(end_time))
        print("elapsed seconds: {:d}s".format(elapsed_seconds))
        print("elapsed time: {:s}".format(format_seconds_to_hhmmss(elapsed_seconds)))
        line(end="###\n")
        
    def wrapper(main: callable):
        def wrapped_main(args=dict()):
            # Call the specified prepare_parser function
            args_script = dict()
            # if len(sys.argv) == 1:
            #     argv1 = dict_to_list(args)
            #     sys.argv.extend(argv1)
            parser = None
            if len(sys.argv) > 1 and sys.argv[1] in ["-h","--help"]:
                print_header(None,main,help=True)
                parser:ArgumentParser = prepare_parser(None)

            if prepare_parser is not None:
                if parser is None:
                    parser:ArgumentParser = prepare_parser(description)
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

#---------------------------------------#
from typing import TypeVar, Callable
from eslib.classes.timing import timing as Timing
from colorama import Fore, Style

T = TypeVar('T', bound='eslog')

class eslog(Timing):
    """
    A logging class that inherits from the `Timing` class and provides logging functionality 
    with a timed context manager for logging messages. The class prints a message when 
    entering and leaving a context, showing the elapsed time upon exit.

    Attributes:
        message (str): The message to be displayed during logging.
    """
    
    message: str
    newline: bool

    def __init__(self: T, message: str = "", timing: bool = True) -> None:
        """
        Initialize the eslog class.

        Args:
            message (str): The message to log (default is an empty string).
            timing (bool): Flag to enable or disable timing (default is True).
        """
        # Pass the closure function to the parent class (Timing) which gets called at the end of the context.
        super().__init__(enabled=timing, func=self.closure_message)
        if message.startswith("\n"):
            self.message = message[1:]
            self.newline = True
        else:
            self.message = message
            self.newline = False
        

    def __enter__(self: T) -> T:
        """
        Enter the runtime context related to this object and print the initial message.

        Returns:
            T: The instance of the `eslog` class.
        """
        # Call the parent context manager's __enter__ method.
        super().__enter__()
        # Print the message when the context is entered.
        prefix = datetime.now().time().strftime("%H:%M:%S")
        prefix = Fore.YELLOW + prefix + Style.RESET_ALL
        if self.newline:
            print()
        print_message(message=self.message, prefix=prefix,end="\r", flush=True)
        return self

    def closure_message(self, elapsed_seconds: float) -> None:
        """
        Closure function to format and print the message when the context is exited,
        including the elapsed time.

        Args:
            elapsed_seconds (float): The elapsed time in seconds.
        """
        # Format the elapsed time to HH:MM:SS format and color it.
        elapsed_time_str = format_seconds_to_hhmmss(elapsed_seconds)
        elapsed_time_str = Fore.YELLOW + elapsed_time_str + Style.RESET_ALL
        # Print the message with the elapsed time and suffix indicating completion.
        print_message(message=self.message,
                      prefix=elapsed_time_str,
                      suffix=" ... done",
                      end="\n",
                      flush=True)


import string
from colorama import Fore, Style

def get_not_printable_length(text: str) -> int:
    """
    Calculate the length of printable characters in the text, ignoring ANSI color codes.
    
    Args:
        text (str): The string to measure, which may contain ANSI escape sequences.
        
    Returns:
        int: The length of the string considering only printable characters.
    """
    # Extracting printable characters only
    return len([char for char in text if char not in string.printable])


def print_message(message: str, prefix: str = "", suffix: str = " ... ", end: str = "\n", flush: bool = False) -> None:
    """
    Utility function to print a formatted message with optional prefix, suffix, and padding 
    to ensure the total output respects a given width, ignoring ANSI color sequences.

    Args:
        message (str): The main message to display.
        prefix (str): Optional prefix to display before the message (default is an empty string).
        suffix (str): Optional suffix to display after the message (default is " ... ").
        end (str): The end character to print (default is newline "\n").
        flush (bool): Whether to forcefully flush the output buffer (default is False).
        width (int): Total width of the output to ensure proper alignment (default is 120).
    """
    # TOT = 5 
    # N = get_not_printable_length(prefix)
    # if N > 0 :
    #     N = 8 # somehow it works
    # prefix = f"{prefix:<s} "
    full_message = f"{prefix:<s} {message}{suffix}"+" "*100
    # Print the padded message
    print(full_message, end=end, flush=flush)



# Example Usage
if __name__ == "__main__":
    # Simulate a task with timing and message logging
    with eslog(message="Processing task"):
        # Some dummy task simulation (sleep for 2 seconds)
        import time
        time.sleep(10)
