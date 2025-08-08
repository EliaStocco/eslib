#!/usr/bin/env python
import xml.etree.ElementTree as ET
from eslib.formatting import esfmt, warning
from eslib.input import itype

#---------------------------------------#
# Description of the script's purpose
description = "Replace the value of a specified XML tag in a file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", type=str, **argv, required=True,
                        help="XML input file")
    parser.add_argument("-t", "--tag", type=str, **argv, required=True,
                        help="XML tag name to replace")
    parser.add_argument("-v", "--value", type=str, **argv, required=True,
                        help="New value for the XML tag")
    parser.add_argument("-o", "--output", type=str, **argv, required=False,
                        help="Output file (default: overwrite input)")
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    xml_file = args.input
    tag_name = args.tag
    new_value = args.value
    output_file = args.output or xml_file

    print(f"\tReading XML file '{xml_file}' ... ", end="")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    print("done")

    # Find the tag (first occurrence)
    elem = root.find(f".//{tag_name}")
    if elem is None:
        raise ValueError(f"<{tag_name}> not found in '{xml_file}'")

    print(f"\tOld value of <{tag_name}>: {elem.text}")
    elem.text = new_value
    print(f"\tNew value of <{tag_name}>: {elem.text}")

    print(f"\tSaving updated XML to '{output_file}' ... ", end="")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
