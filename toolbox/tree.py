#!/usr/bin/env python

import os
import numpy as np
import argparse
from colorama import Fore, Style

def colorize(item, directory):
    if os.path.isdir(os.path.join(directory, item)):
        return Style.BRIGHT + Fore.YELLOW + item + "/" + Style.RESET_ALL
    elif item.endswith(".py"):
        return Fore.LIGHTCYAN_EX + item + Style.RESET_ALL
    elif item.endswith(".sh"):
        return Fore.GREEN + item + Style.RESET_ALL
    elif np.any([item.endswith(k) for k in [".md", ".gitignore", ".yml"]]) or item == "requirements.txt":
        return Fore.RED + item + Style.RESET_ALL
    else:
        return Fore.BLACK + item + Style.RESET_ALL

def tree(directory, level=0, max_level=2, output=None):
    if level > max_level:
        return

    with open(output, "a") as f:
        for item in os.listdir(directory):
            if item in ["__pycache__", ".git", ".vscode", ".pytest_cache"]:
                continue

            path = os.path.join(directory, item)
            f.write("|   " * level + "|-- " + colorize(item, directory) + "\n")

            if os.path.isdir(path):
                tree(os.path.join(directory, item), level + 1, max_level, output)

def main():
    parser = argparse.ArgumentParser(description="Print directory tree up to specified depth")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to print tree from (default: current directory)")
    parser.add_argument("-L", "--max-level", type=int, default=2, help="Maximum depth of tree (default: 2)")
    parser.add_argument("-o", "--output", default="tree_output.txt", help="Output file name (default: tree_output.txt)")
    args = parser.parse_args()

    tree(args.directory, max_level=args.max_level, output=args.output)

if __name__ == "__main__":
    main()
