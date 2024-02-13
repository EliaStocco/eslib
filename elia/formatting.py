float_format = '%24.12e' # Elia Stocco float format

#---------------------------------------#
error           = "***Error***"
warning         = "***Warning***"
closure         = "Job done :)"
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
    warning         = Fore.MAGENTA    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
except:
    pass

def esfmt(prepare_parser:callable, description:str=None):
    """Decorator for the 'main' function of many scripts."""

    from contextlib import contextmanager

    #---------------------------------------#
    # Description of the script's purpose
    description = description if description is not None else "Script without description"
    try: description = Fore.GREEN  + Style.BRIGHT + description + Style.RESET_ALL
    except: pass
    # print(description)

    @contextmanager
    def print_header(args):
        print("\n\t{:s}".format(description))
        print("\n\t{:s}:".format(input_arguments))
        for k in args.__dict__.keys():
            print("\t{:>20s}:".format(k), getattr(args, k))
        print()
        yield

    def wrapper(main: callable):
        def wrapped_main():
            # Call the specified prepare_parser function

            args = prepare_parser(description)

            # Print the script's description and input arguments
            with print_header(args):
                # main
                main(args)

            #------------------#
            # Script completion message
            print("\n\t{:s}\n".format(closure))

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