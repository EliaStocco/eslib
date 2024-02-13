import re
import numpy as np

def get_property_header(inputfile, N=1000, search=True):
    names = [None] * N
    restart = False

    with open(inputfile, "r") as ifile:
        icol = 0
        while True:
            line = ifile.readline()
            nline = line
            if not line:
                break
            elif "#" in line:
                line = line.split("-->")[1]
                line = line.split(":")[0]
                line = line.split(" ")[1]

                nline = nline.split("-->")[0]
                if "column" in nline:
                    lenght = 1
                else:
                    nline = nline.split("cols.")[1]
                    nline = nline.split("-")
                    a, b = int(nline[0]), int(nline[1])
                    lenght = b - a + 1

                if icol < N:
                    if not search:
                        if lenght == 1:
                            names[icol] = line
                            icol += 1
                        else:
                            for i in range(lenght):
                                names[icol] = line + "-" + str(i)
                                icol += 1
                    else:
                        names[icol] = line
                        icol += 1
                else:
                    restart = True
                    icol += 1

    if restart:
        return get_property_header(inputfile, N=icol)
    else:
        return names[:icol]


def getproperty(inputfile, propertyname, data=None, skip="0", show=False):
    def check(p, l):
        if not l.find(p):
            return False  # not found
        elif l[l.find(p) - 1] != " ":
            return False  # composite word
        elif l[l.find(p) + len(p)] == "{":
            return True
        elif l[l.find(p) + len(p)] != " ":
            return False  # composite word
        else:
            return True

    if type(propertyname) in [list, np.ndarray]:
        out = dict()
        units = dict()
        data = np.loadtxt(inputfile)
        for p in propertyname:
            out[p], units[p] = getproperty(inputfile, p, data, skip=skip)
        return out, units

    if show:
        print("\tsearching for '{:s}'".format(propertyname))

    skip = int(skip)

    # propertyname = " " + propertyname + " "

    # opens & parses the input file
    with open(inputfile, "r") as ifile:
        # ifile = open(inputfile, "r")

        # now reads the file one frame at a time, and outputs only the required column(s)
        icol = 0
        while True:
            try:
                line = ifile.readline()
                if len(line) == 0:
                    raise EOFError
                while "#" in line:  # fast forward if line is a comment
                    line = line.split(":")[0]
                    if check(propertyname, line):
                        cols = [int(i) - 1 for i in re.findall(r"\d+", line)]
                        if len(cols) == 1:
                            icol += 1
                            output = data[:, cols[0]]
                        elif len(cols) == 2:
                            icol += 1
                            output = data[:, cols[0] : cols[1] + 1]
                        elif len(cols) != 0:
                            raise ValueError("wrong string")
                        if icol > 1:
                            raise ValueError(
                                "Multiple instances for '{:s}' have been found".format(
                                    propertyname
                                )
                            )

                        l = line
                        p = propertyname
                        if l[l.find(p) + len(p)] == "{":
                            unit = l.split("{")[1].split("}")[0]
                        else:
                            unit = "atomic_unit"

                    # get new line
                    line = ifile.readline()
                    if len(line) == 0:
                        raise EOFError
                if icol <= 0:
                    print("Could not find " + propertyname + " in file " + inputfile)
                    raise EOFError
                else:
                    if show:
                        print("\tfound '{:s}'".format(propertyname))
                    return np.asarray(output), unit

            except EOFError:
                break