import argparse

# Define command-line arguments
parser = argparse.ArgumentParser(description="Convert input numbers to LaTeX table.")
parser.add_argument("--input", help="input txt file")
parser.add_argument("--output", help="output LaTeX file")
parser.add_argument("--decimal", type=int, default=2, help="Number of decimal places to use in the table (default: 2)")
args = parser.parse_args()

# Read numbers from the input file
with open(args.input, 'r') as file:
    lines = file.readlines()

# Extract the numbers from the file
numbers = []
for line in lines:
    numbers.extend(map(float, line.split()))

# Define the LaTeX table format
latex_table = "\\begin{table}\n"
latex_table += "    \\centering\n"
latex_table += "    \\begin{tabular}{|c|c|c|c|c|}\n"
latex_table += "        \\hline\n"
latex_table += "         & & $\\partial R_x$ & $\\partial R_y$ & $\\partial R_z$ \\\\\n"
latex_table += "        \\hline\n"

# Define the row and column headers
row_headers = ["$P_x$", "$P_y$", "$P_z$"]
col_headers = ["", "", "H", "H", "H"]

# Format numbers with the specified decimal places
format_str = "{:." + str(args.decimal) + "f}"

# Fill in the table
for i, row_header in enumerate(row_headers):
    latex_table += "        "
    if i > 0:
        latex_table += "\\multirow{3}{*}{\\ce{H}}"
    latex_table += " & {:s}".format(row_header)
    for j in range(3):
        value = numbers[i * 3 + j]
        if j > 0:
            latex_table += " & "
        if i == j:
            latex_table += f"\\cellcolor{{red!50}} {format_str.format(value)}"
        else:
            latex_table += f"\\cellcolor{{white}} {format_str.format(value)}"
    latex_table += " \\\\\n"

# Complete the LaTeX table
latex_table += "        \\hline\n"
latex_table += "    \\end{tabular}\n"
latex_table += "    \\caption{DFT, finite difference, 4x4x4}\n"
latex_table += "\\end{table}\n"

# Write the LaTeX table to the output file
with open(args.output, 'w') as output:
    output.write(latex_table)

print("LaTeX table saved to '{:s}'".format(args.output))
