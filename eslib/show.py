import pandas as pd


#---------------------------------------#
def show_dict(obj:dict,string:str="",width=30):
    fmt = "{:<s}{:" + str(width) + "s} : "
    for k in obj.keys():
        print(fmt.format(string,k),obj[k])

#---------------------------------------#
import pandas as pd


def print_df(df: pd.DataFrame, format_str: str = '{:>12.4f}') -> str:
    # Create a formatted string for the column names
    column_names = ''.join(['{:>12}'.format(col) for col in df.columns])
    
    # Print the headers and the separator line
    print('\n\t' + "-" * len(column_names))
    print('\t' + column_names)
    print('\t' + "-" * len(column_names))
    
    # Iterate over rows and print with the specified format
    for index, row in df.iterrows():
        formatted_row = '{:>12d}'.format(int(row[0]))  # First column as integer
        formatted_row += ''.join([format_str.format(value) for value in row[1:]])  # Remaining columns using user-provided format
        print('\t' + formatted_row)
    
    # Print the closing separator line
    print('\t' + "-" * len(column_names))



#---------------------------------------#
def print_cell(cell, tab="\t\t"):
    cell = cell.T
    string = tab + "{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format("", "", "x", "y", "z")
    for i in range(3):
        string += (
            "\n"
            + tab
            + "{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format(
                "lattice vector", i + 1, cell[i, 0], cell[i, 1], cell[i, 2]
            )
        )
    return string

#---------------------------------------#
def dict_to_string(d:dict):
    return ' '.join([f'--{key} {value}' for key, value in d.items()])

#---------------------------------------#
def dict_to_list(d:dict):
    if d == {} or d is None:
        return None
    result = []
    for key, value in d.items():
        result.extend([f'--{key}', str(value)])
    return result

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