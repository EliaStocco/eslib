#!/usr/bin/env python
def generate_readme(tree_output_file, readme_file):
    with open(tree_output_file, "r") as f:
        tree_output_content = f.read()

    with open(readme_file, "w") as f:
        f.write("# Directory Tree\n\n")
        f.write("Below is the directory tree of the project:\n\n")
        f.write("<pre><code>")
        f.write(tree_output_content)
        f.write("</code></pre>\n")

if __name__ == "__main__":
    tree_output_file = "tree_output.txt"
    readme_file = "README.md"
    generate_readme(tree_output_file, readme_file)
