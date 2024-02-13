
#!/usr/bin/env python
import os
import fnmatch
import argparse

def find_files(folder, pattern, file_extension):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, f'*.{file_extension}'):
            if pattern in filename:
                matches.append(os.path.join(root, filename))
    return matches

def main():
    parser = argparse.ArgumentParser(description="Search for files in a folder and its subfolders based on a pattern in the filename.")
    argv = {"metavar" : "\b",}
    parser.add_argument("-p","--pattern"  , **argv, help="pattern to search for in the filenames")
    parser.add_argument("-f","--folder"   , **argv, help="folder to search in (default: '.')", default=".")
    parser.add_argument("-e","--extension", **argv, help="file extension to filter (default: 'pdf')", default="pdf")

    args = parser.parse_args()

    result = find_files(args.folder, args.pattern, args.extension)

    if result:
        print("Files found:")
        for file_path in result:
            print("\"{:s}\" --> \"{:s}\"".format(file_path,file_path.replace(" ","\\ ")))
    else:
        print("No files found.")

if __name__ == "__main__":
    main()
