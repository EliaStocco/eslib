import pandas as pd

def show_dict(obj:dict,string:str="",width=30):
    fmt = "{:<s}{:" + str(width) + "s} : "
    for k in obj.keys():
        print(fmt.format(string,k),obj[k])

def print_df(df:pd.DataFrame)->str:
    column_names = ''.join(['{:>12}'.format(col) for col in df.columns])
    print('\n\t' + "-"*len(column_names))
    print('\t' + column_names)
    print('\t' + "-"*len(column_names))
    # Iterate over rows and print with the specified format
    for index, row in df.iterrows():
        formatted_row = ''.join(['{:>12.1e}'.format(value) for value in row])
        print('\t' + formatted_row)
    print('\t' + "-"*len(column_names))