import pandas as pd

#---------------------------------------#
def show_dict(obj:dict,string:str="",width=30):
    fmt = "{:<s}{:" + str(width) + "s} : "
    for k in obj.keys():
        print(fmt.format(string,k),obj[k])

#---------------------------------------#
def print_df(df:pd.DataFrame)->str:
    column_names = ''.join(['{:>12}'.format(col) for col in df.columns])
    print('\n\t' + "-"*len(column_names))
    print('\t' + column_names)
    print('\t' + "-"*len(column_names))
    # Iterate over rows and print with the specified format
    for index, row in df.iterrows():
        formatted_row = ''.join(['{:>12.2e}'.format(value) for value in row])
        print('\t' + formatted_row)
    print('\t' + "-"*len(column_names))

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
        result.extend([f'--{key}', value])
    return result
