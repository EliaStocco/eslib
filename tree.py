import os

from eslib.formatting import esfmt

exclude = ['__pycache__','workflows/']
exclude = exclude + [ a+os.sep for a in exclude]

def generate_folder_tree(startpath,max_level):
    out = ""
    for root, dirs, files in os.walk(startpath):
        root = str(root)
        # print(root)
        if root in exclude or any( [a in root for a in exclude] ):
            continue
        level = root.replace(startpath, '').count(os.sep)
        if level > max_level+1:
            continue
        indent = ' ' * 4 * (level)
        out += f'{indent}{os.path.basename(root)}/\n'
        if level > max_level:
            continue
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            f = str(f)
            if not f.endswith('.py'):
                continue
            out += f'{subindent}{f}\n'
    return out

#---------------------------------------#
description='Generate a folder tree of the specified directory.'

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-f" , "--folder"   , **argv, type=str, required=False , help='The folder to generate the tree from (default: %(default)s)', default=".")
    parser.add_argument("-l" , "--level"   , **argv, type=int, required=False , help='The maximum depth level to display (default: %(default)s)', default=0)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    tree = generate_folder_tree(args.folder,args.level)
    print(tree)

    return


#---------------------------------------#
if __name__ == "__main__":
    main()
