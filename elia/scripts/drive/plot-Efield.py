#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as xmlet
import os
#import ast 
from elia.tools import convert

class ElectricField:

    def __init__(self, amp=None, freq=None, phase=None, peak=None, sigma=None):
        self.amp = amp if amp is not None else np.zeros(3)
        self.freq = freq if freq is not None else 0.0
        self.phase = phase if phase is not None else 0.0
        self.peak = peak if peak is not None else 0.0
        self.sigma = sigma if sigma is not None else np.inf

    def Efield(self,time):
        """Get the value of the external electric field (cartesian axes)"""
        if hasattr(time, "__len__"):
            return np.outer(self._get_Ecos(time) * self.Eenvelope(time), self.amp)
        else:
            return self._get_Ecos(time) * self.Eenvelope(time) * self.amp

    def _Eenvelope_is_on(self):
        return self.peak > 0.0 and self.sigma != np.inf

    def Eenvelope(self,time):
        """Get the gaussian envelope function of the external electric field"""
        # https://en.wikipedia.org/wiki/Normal_distribution
        if self._Eenvelope_is_on():
            x = time  # indipendent variable
            u = self.peak  # mean value
            s = self.sigma  # standard deviation
            return np.exp(
                -0.5 * ((x - u) / s) ** 2
            )  # the returned maximum value is 1, when x = u
        else:
            return 1.0

    def _get_Ecos(self, time):
        """Get the sinusoidal part of the external electric field"""
        # it's easier to define a function and compute this 'cos'
        # again everytime instead of define a 'depend_value'
        return np.cos(self.freq * time + self.phase)

def plt_clean():
    ###
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def compute(Ef,data,options):
    factor = convert ( 1 , "time" , options.unit , "picosecond" )
    t = np.arange(0,options.t_max,options.time_spacing) * factor
    tt = t * convert ( 1 , "time" , "picosecond" , "atomic_unit" )
    E = np.zeros( (len(t),3))    
    E = Ef.Efield(tt)
    f = Ef.Eenvelope(tt) * np.linalg.norm(data["amp"])
    En = np.linalg.norm(E,axis=1)
    return t,E,En,f

def FFT_plot(Ef,data,options):

    from elia.fourier import FourierAnalyzer

    t,E,En,f= compute(Ef,data,options)
    result = np.column_stack((E, f)).shape
    fft = FourierAnalyzer(t,result)

    # fft.plot_fourier_transform()
    # fft.plot_power_spectrum()
    # fft.plot_time_series()

    fig, ax = plt.subplots(figsize=(12,6))

    fft.freq, fft.spectrum

    for i in range(3):
        ax.plot(fft.freq,fft.fft[:,i])
    plt.show()

    ax.plot(t,En,label="$|E|$",color="gray",alpha=0.5)
    ax.plot(t,E[:,0],label="$E_x$",color="red",alpha=0.5)
    ax.plot(t,E[:,1],label="$E_y$",color="green",alpha=0.5)
    ax.plot(t,E[:,2],label="$E_z$",color="blue",alpha=0.5)

    plt.ylabel("electric field [a.u.]")
    plt.xlabel("time [ps]")
    plt.grid()
    plt.legend()

    plt.tight_layout()

    temp = os.path.splitext(options.output) 
    file = "{:s}.FFT{:s}".format(temp[0],temp[1])
    print("\tSaving plot to {:s}".format(file))
    plt.savefig(file)

def Ef_plot(Ef,data,options):

    t,E,En,f= compute(Ef,data,options)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(t,f,label="$f_{env} \\times E_{amp}$",color="black")
    ax.plot(t,En,label="$|E|$",color="gray",alpha=0.5)
    ax.plot(t,E[:,0],label="$E_x$",color="red",alpha=0.5)
    ax.plot(t,E[:,1],label="$E_y$",color="green",alpha=0.5)
    ax.plot(t,E[:,2],label="$E_z$",color="blue",alpha=0.5)

    plt.ylabel("electric field [a.u.]")
    plt.xlabel("time [ps]")
    plt.grid()
    plt.legend()

    plt.tight_layout()

    print("\tSaving plot to {:s}".format(options.output))
    plt.savefig(options.output)

    plt_clean()

def prepare_parser():
    """set up the script input parameters"""

    parser = argparse.ArgumentParser(description="Plot the electric field E(t) into a pdf file.")

    parser.add_argument(
        "-i", "--input", action="store", type=str,
        help="input file", default="input.xml"
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output file ", default="Efield.pdf"
    )
    parser.add_argument(
        "-n", "--n_steps", action="store", type=int,
        help="max. number of steps",
    )
    # parser.add_argument(
    #     "-t", "--t_max", action="store", type=float,
    #     help="max time",
    # )
    parser.add_argument(
        "-dt", "--time_spacing", action="store", type=float,
        help="max time",default=1
    )
    parser.add_argument(
        "-u", "--unit", action="store", type=str,
        help="unit",default="picosecond"
    )
       
    options = parser.parse_args()

    options.t_max = options.time_spacing * options.n_steps

    return options

def get_data(options):

    print("\tReading json file")
    # # Open the JSON file and load the data
    # with open(options.input) as f:
    #     info = json.load(f)

    data = xmlet.parse(options.input).getroot()

    efield = None
    for element in data.iter():
        if element.tag == "efield":
            efield = element
            break

    data     = {}
    keys     = ["amp",          "freq",    "phase",   "peak","sigma"]
    families = ["electric-field","frequency","undefined","time", "time"  ]
    
    for key,family in zip(keys,families):

        data[key] = None
        
        element = efield.find(key)

        if element is not None:
            #value = ast.literal_eval(element.text)
            text =  element.text
            try :
                value = text.split('[')[1].split(']')[0].split(',')
                value = [ float(i) for i in value ]
                if len(value) == 1:
                    value = float(value)
                else :
                    value = np.asarray(value)
            except :
                value = float(text)
            
            try :
                unit = element.attrib["units"]
                if unit is None :
                    unit = "atomic_unit"
            except:
                unit = "atomic_unit"

            # print(key,value,unit)

            value = convert(value,family,unit,"atomic_unit")
            data[key] = value

    return data

def main():
    """main routine"""

    # prepare/read input arguments
    print("\tReading script input arguments")
    options = prepare_parser()

    data = get_data(options)

    Ef = ElectricField( amp=data["amp"],\
                        phase=data["phase"],\
                        freq=data["freq"],\
                        peak=data["peak"],\
                        sigma=data["sigma"])

    # plot of the E-field
    Ef_plot(Ef,data,options)

    # plot of the E-field FFT
    # FFT_plot(Ef,data,options)

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()