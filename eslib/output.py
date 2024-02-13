import os
import numpy as np

def output_folder(folder,show=True):
    if folder in ["", ".", "./"]:
        folder = "."
    elif not os.path.exists(folder):
        if show: print("\n\tCreating directory '{:s}'".format(folder))
        os.mkdir(folder)
    return folder

def output_file(folder, what):
    folder = output_folder(folder)
    return "{:s}/{:s}".format(folder, what)

def save2xyz(what, file, atoms, comment=""):
    if len(what.shape) == 1:  # just one configuration, NOT correctly formatted
        what = what.reshape((-1, 3))
        return save2xyz(what, file, atoms)

    elif len(what.shape) == 2:
        if what.shape[1] != 3:  # many configurations
            what = what.reshape((len(what), -1, 3))
            return save2xyz(what, file, atoms)

        else:  # just one configurations, correctly formatted
            return save2xyz(np.asarray([what]), file, atoms)

    elif len(what.shape) == 3:
        Na = what.shape[1]
        if what.shape[2] != 3:
            raise ValueError("wrong shape")

        with open(file, "w") as f:
            for i in range(what.shape[0]):
                pos = what[i, :, :]
                f.write(str(Na) + "\n")
                f.write("# {:s}\n".format(comment))
                for ii in range(Na):
                    f.write(
                        "{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(
                            atoms[ii], *pos[ii, :]
                        )
                    )
        return