import re
from copy import copy
from typing import Callable, TypeVar, Union

import numpy as np
import pandas as pd

from eslib.classes import Trajectory
from eslib.classes.io import pickleIO
from eslib.tools import convert
from eslib.classes.append import AppendableList

T = TypeVar('T', bound='Properties')

def find_key_by_value(family_map, value):
    for key, values in family_map.items():
        if value in values:
            return key
    return None  # Return None if the value is not found in any of the lists

FamilyMap = {
    "energy" : ["conserved","energy","potential","kinetic_md","Econserved"],
    "length" : ["positions"],
    "time"   : ["time"]
}

def is_1D(x:np.ndarray):
    if len(x.shape) == 1:
        return True
    shape = np.asarray(x.shape)[1:].astype(int)
    return np.allclose(shape,1)

class Properties(Trajectory):

    def __init__(self,info:dict=None):

        self.header     = list()
        self.properties = dict()
        self.units      = dict()
        self.raw_header = ""
        self.length     = 0 
        
        if info is not None:
            self.header     = info["header"]
            self.properties = info["properties"]
            self.units      = info["units"]
            self.raw_header = info["raw_header"]

        self.set_length()
        
        pass

    @classmethod
    @pickleIO.correct_extension_in
    def from_file(cls, **argv):
        """
        Load atomic structures from file.

        Attention: it's recommended to use keyword-only arguments.
        """
        return Properties.load(**argv)
        # return cls(traj)
        
    @pickleIO.correct_extension_out
    def to_file(self: T, file: str, format: Union[str, None] = None):
        """
        Write atomic structures to file.
        
        Attention: it's recommended to use keyword-only arguments.
        """
        raise ValueError("only pickle extension is supported")

    def set_length(self):
        length = None
        for k in self.header:
            if k not in self.properties:
                raise IndexError("'{:s}' is not a valid property.".format(k))
            self.properties[k] = np.asarray(self.properties[k])
            N = len(self.properties[k])
            if length is None:
                length = N
            elif length != N:
                raise ValueError("All the properties should have the same length.")
        self.length = length
    
    @classmethod
    def load(cls,file,index=None):
        header = get_property_header(file,search=True)
        properties,units = getproperty(file,header,index=index)
        raw_header = get_raw_header(file)
        # header  = get_property_header(file,search=False)
        info = {
            "header"     : header,
            "properties" : properties,
            "units"      : units,
            "raw_header" : raw_header,
        }
        return cls(info=info)
    
    def __getitem__(self,index):
        if type(index) == str:
            return self.properties[index]
        elif type(index) == int:
            out = dict()
            for k in self.header:
                out[k] = self.properties[k][index]
            return out
        elif type(index) == slice:
            out = copy(self)
            for k in self.header:
                out.properties[k] = self.properties[k][index]
            return out
        else:
            raise TypeError("index type not allowed/implemented")
        
    def __len__(self):
        return self.length
    
    # def to_pandas(self):
    #     df = pd.DataFrame(columns=self.header,dtype=object)
    #     for k in self.heder:
    #         df[k] = self.properties[k]
    #     return df
        
    def summary(self):
        # print("Properties of the object:")
        keys = list(self.properties.keys())
        size = [None]*len(keys)
        for n,k in enumerate(keys):
            size[n] = self.properties[k].shape
            # tmp = list(self.properties[k].shape[1:])
            # if len(tmp) == 0 :
            #     size[n] = 1
            # elif len(tmp) == 1:
            #     size[n] = tmp[0]
            # else :
            #     size[n] = tmp
        df = pd.DataFrame(columns=["name","unit","shape"])
        df["name"] = keys
        df["unit"] = [ self.units[k] for k in keys ]
        df["shape"] = size
        return df
    
    def _get_columns_lenght(self):
        """
        Returns a list of column counts (i.e., second dimension sizes) 
        for each property in the object.

        For each 2D array stored in `self.properties`, this method extracts 
        the size of the second dimension. If an array is 1D, it is assumed 
        to have a single column.

        Returns:
            List[int]: A list of integers representing the number of columns 
                    for each property array.
        """
        df = self.summary()
        shapes = df["shape"].values
        lengths = [sh[1] if len(sh) > 1 else 1 for sh in shapes]
        return lengths
    
    def to_numpy(self):
        lengths = self._get_columns_lenght()
        data = np.full((self.length, sum(lengths)),np.nan,dtype=float)
        i = 0 
        for n,k in enumerate(self.header):
            j = lengths[n] + i
            values = self.properties[k]
            if is_1D(values):
                values = values[:,np.newaxis]
            # if len(self.properties[k].shape) == 1:
            #     data[:,i:j] = self.properties[k]
            # else:
            data[:,i:j] = values
            i = j
        assert not np.any(np.isnan(data)), "Data contains NaN values."
        return data, lengths
    
    def from_numpy(self,data:np.ndarray):
        """
        Converts a 2D numpy array into the properties of the object.

        The first column is assumed to be the 'step' column, and the rest 
        are assigned to the properties in the header.

        Parameters:
            data (np.ndarray): 2D numpy array containing the data.
        """
        lengths = self._get_columns_lenght()
        data = np.atleast_2d(data)
        assert sum(lengths) == data.shape[1], f"Data has {data.shape[1]} columns, but the properties have {sum(lengths)} columns."
            
        i = 0 
        for n,k in enumerate(self.header):
            j = lengths[n] + i
            # shape = self.properties[k].shape
            to_assign = data[:,i:j]
            if is_1D(to_assign):
                to_assign = to_assign.flatten()
            # assert (len(shape) == 1 and to_assign.shape[1] ==1) or shape[1] == to_assign.shape[1], f"'{k}' has a different number of columns than the data."
            self.properties[k] = to_assign
            i = j
        self.set_length()
    
    def fix(self: T,func:Callable=None):
        """
        Ensures the 'step' axis is continuous by filling in missing steps.
        Missing rows are added with NaNs and then filled using linear interpolation.

        This affects all properties, including the 'step' column itself.

        Returns:
            T: The modified object with interpolated data.
        """
        steps = np.asarray(self.properties["step"]).astype(int)
        all_steps = np.arange(np.max(steps) + 1, dtype=int)

        # Early exit if steps are already complete and ordered
        if steps.shape == all_steps.shape and np.allclose(steps, all_steps):
            return self, "no fix performed"

        # Extract full data and shape info
        data, lengths = self.to_numpy()
        new_data = np.full((len(all_steps), sum(lengths)), np.nan)

        # Use only first occurrence of each step (in case of duplicates)
        usteps, indices = np.unique(steps, return_index=True)
        data = data[indices, :]
        new_data[usteps, :] = data
        
        if not np.any(np.isnan(new_data)):
            self.from_numpy(new_data)
            return self, "removed replicas"
        
        if func is None:
            func = lambda x,xp,fp: np.interp(x, xp, fp)

        # Interpolate all columns linearly over the missing rows
        for col in range(new_data.shape[1]):
            col_data = new_data[:, col]
            mask = np.isnan(col_data)
            xp = np.where(~mask)[0]
            fp = col_data[~mask]
            x = np.where(mask)[0]
            y_intep = func(x,xp,fp)
            new_data[x, col] = y_intep
        
        # Check for NaN values after interpolation
        if np.any(np.isnan(new_data)):
            raise ValueError("Interpolation failed, NaN values remain.")

        # Overwrite properties with interpolated data
        self.from_numpy(new_data)
        
        # Check if the 'step' column is continuous after interpolation
        new_steps = np.asarray(self.properties["step"]).astype(int)
        if not np.allclose(new_steps, all_steps):
            raise ValueError("Steps are not continuous after interpolation.")

        return self, "removed replicas and filled NaN"
    
    def to_ipi(self:T, file:str):
        header = copy(self.raw_header)
        header = [ h[1:] for h in header]
        header = "".join(header)[:-1]
        data, _ = self.to_numpy()
        np.savetxt(file, data, fmt="%16.8e", delimiter=" ", header=header, comments="#")
    
    # def remove_replicas(self,keyword="step",ofile:str=None):
    #     out = copy(self)
    #     steps = self.properties[keyword]
    #     usteps, indices = np.unique(steps, return_index=True)
    #     if ofile is not None: np.savetxt(ofile,usteps,fmt="%d")
    #     for k in self.properties.keys():
    #         out.properties[k] = self.properties[k][indices]
    #     out.set_length()
    #     assert np.allclose(np.arange(len(out)),out.properties[keyword])
    #     return out
    
    def set(self:T, name:str, data:np.ndarray,unit:str="atomic_unit", **argv) -> None:
        self.properties[name] = data
        self.units[name] = unit
        self.header = self.header + [name]


    def get(self:T, name:str, unit=None,**argv) -> np.ndarray:
        out = self.properties[name]
        if unit is None:
            return out
        else:
            family = find_key_by_value(FamilyMap,name)
            return convert(out,family,self.units[name],unit)

def get_raw_header(inputfile):
    header = AppendableList()
    with open(inputfile, "r") as ifile:
        icol = 0
        while True:
            line = ifile.readline()
            if not line:
                break
            elif "#" in line:
                if "-->" in line:
                    header.append(line)

    header = header.finalize()
    return header
    
def get_property_header(inputfile, N=1000, search=True, strip_units=True):
    """
    Extracts the property names from the header of a data file.
    
    - Keeps extra info in parentheses (e.g. '(atom=O)').
    - If strip_units=True, removes the '{unit}' part but keeps parentheses.
    - Returns a list of property names.
    """
    names = [None] * N
    restart = False

    with open(inputfile, "r") as ifile:
        icol = 0
        while True:
            line = ifile.readline()
            nline = line
            if not line:
                break
            elif "#" in line and "-->" in line:
                # Extract the property name portion after -->
                prop_part = line.split("-->")[1].split(":")[0].strip()

                # If requested, strip the {unit} part but keep parentheses
                if strip_units:
                    prop_part = re.sub(r"\{[^}]*\}", "", prop_part)

                # Figure out the number of columns
                nline = nline.split("-->")[0]
                if "column" in nline:
                    length = 1
                else:
                    nline = nline.split("cols.")[1]
                    a, b = map(int, nline.split("-"))
                    length = b - a + 1

                # Store property names
                if icol < N:
                    if not search:
                        if length == 1:
                            names[icol] = prop_part
                            icol += 1
                        else:
                            for i in range(length):
                                names[icol] = f"{prop_part}-{i}"
                                icol += 1
                    else:
                        names[icol] = prop_part
                        icol += 1
                else:
                    restart = True
                    icol += 1

    if restart:
        return get_property_header(inputfile, N=icol, search=search, strip_units=strip_units)
    else:
        return names[:icol]

def getproperty(inputfile, propertyname, data=None, skip="0", show=False, index=None, return_all=False):
    """
    Extract a property from file header-driven columns.

    - Matches only the property name portion after '-->' and before ':' to avoid hitting description text.
    - propertyname may be e.g. "temperature", "temperature(atom=O)", "temperature{kelvin}", etc.
    - By default returns (ndarray, unit). If return_all=True returns (dict_of_arrays, dict_of_units).
    """
    def parse_prop_token(s):
        """Parse a token like "temperature{kelvin}(atom=O)" into (base, unit, extra)."""
        if s is None:
            return None, None, None
        s = s.strip()
        # strip external quotes if accidentally supplied: "'temperature(...)'"
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1].strip()
        # match: base [ {unit} ] [ (extra) ]
        m = re.match(r'^\s*([^\{\(]+?)\s*(?:\{([^}]*)\})?\s*(?:\(([^)]*)\))?\s*$', s)
        if not m:
            # fallback: treat whole string as base
            return s.strip(), None, None
        base = m.group(1).strip()
        unit = m.group(2).strip() if m.group(2) else None
        extra = m.group(3).strip() if m.group(3) else None
        return base, unit, extra

    def norm(s):
        """Normalize for comparison: lowercase, remove whitespace."""
        if s is None:
            return None
        return re.sub(r'\s+', '', s).lower()

    skip = int(skip)

    # handle list of properties:
    if isinstance(propertyname, (list, np.ndarray)):
        out = {}
        units = {}
        # load numeric data once
        data = np.loadtxt(inputfile, skiprows=skip) if data is None else data
        if index is not None:
            data = data[index]
        for p in propertyname:
            arr, u = getproperty(inputfile, p, data, skip=skip, show=show, index=index, return_all=False)
            out[p] = arr
            units[p] = u
        return out, units

    if show:
        print(f"\tsearching for '{propertyname}'")

    # ensure numeric data loaded
    if data is None:
        data = np.loadtxt(inputfile, skiprows=skip)
        if index is not None:
            data = data[index]

    user_base, user_unit, user_extra = parse_prop_token(propertyname)
    user_base_n = norm(user_base)
    user_unit_n = norm(user_unit)
    user_extra_n = norm(user_extra)

    matches = []

    # iterate header lines, but only inspect the property token (after --> and before :)
    with open(inputfile, "r") as ifile:
        for line in ifile:
            if "#" not in line or "-->" not in line:
                continue
            # Extract property portion only: between --> and :
            try:
                prop_part = line.split("-->")[1].split(":")[0].strip()
            except Exception:
                # malformed header, skip
                continue

            header_base, header_unit, header_extra = parse_prop_token(prop_part)
            header_base_n = norm(header_base)
            header_unit_n = norm(header_unit)
            header_extra_n = norm(header_extra)

            # Matching logic:
            # - base names must be equal (case/space-insensitive)
            # - if user specified extra (parentheses) it must match header extra
            # - if user specified unit it must match header unit
            if header_base_n == user_base_n:
                if user_extra_n is not None and header_extra_n != user_extra_n:
                    continue
                if user_unit_n is not None and header_unit_n != user_unit_n:
                    continue

                # found candidate — extract columns from the original full line
                cols = [int(i) - 1 for i in re.findall(r"\d+", line)]
                if not cols:
                    continue
                if len(cols) == 1:
                    output = data[:, cols[0]]
                elif len(cols) == 2:
                    output = data[:, cols[0]:cols[1] + 1]
                else:
                    raise ValueError(f"Unexpected column format in line: {line.strip()}")

                unit = header_unit if header_unit else "atomic_unit"
                matches.append((output, unit, prop_part, line.strip()))

    if not matches:
        # As a tolerant fallback, try matching ignoring parentheses content in user string
        base_only = user_base
        if user_extra is not None:
            # attempt search using only base
            matches = []
            with open(inputfile, "r") as ifile:
                for line in ifile:
                    if "#" not in line or "-->" not in line:
                        continue
                    prop_part = line.split("-->")[1].split(":")[0].strip()
                    header_base, header_unit, header_extra = parse_prop_token(prop_part)
                    if norm(header_base) == norm(base_only):
                        cols = [int(i) - 1 for i in re.findall(r"\d+", line)]
                        if not cols:
                            continue
                        if len(cols) == 1:
                            output = data[:, cols[0]]
                        elif len(cols) == 2:
                            output = data[:, cols[0]:cols[1] + 1]
                        else:
                            raise ValueError(f"Unexpected column format in line: {line.strip()}")
                        unit = header_unit if header_unit else "atomic_unit"
                        matches.append((output, unit, prop_part, line.strip()))

    if not matches:
        raise ValueError(f"Could not find property '{propertyname}' in file '{inputfile}'")

    if return_all:
        # return all found variants keyed by the exact prop_part string
        return {m[2]: np.asarray(m[0]) for m in matches}, {m[2]: m[1] for m in matches}

    # default: pick the best single match
    if len(matches) == 1:
        if show:
            print(f"\tfound '{propertyname}': {matches[0][3]}")
        return np.asarray(matches[0][0]), matches[0][1]

    # prefer the one without extra (e.g. the generic 'temperature' vs 'temperature(atom=O)')
    for m in matches:
        hb, hu, he = parse_prop_token(m[2])
        if he is None:
            if show:
                print(f"\tselected generic match for '{propertyname}': {m[3]}")
            return np.asarray(m[0]), m[1]

    # otherwise return first match
    if show:
        print(f"\tmultiple matches, defaulting to first for '{propertyname}': {matches[0][3]}")
    return np.asarray(matches[0][0]), matches[0][1]
