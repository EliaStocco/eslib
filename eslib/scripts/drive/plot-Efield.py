#!/usr/bin/env python3
import argparse
from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as xmlet
import os
#import ast 
from eslib.tools import convert
from eslib.formatting import esfmt
from eslib.show import show_dict

description = "Plot the electric field E(t) into a pdf file."

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

def Ef_plot(Ef:ElectricField,data:dict,output):

    unit = "femtosecond"
    factor = convert ( 1 , "time" , "atomic_unit" , unit)
    t = np.arange(0,data["total_steps"])*data["timestep"] * factor
    tt = t * convert ( 1 , "time" , unit , "atomic_unit" )
    E = np.zeros( (len(t),3))    
    E = Ef.Efield(tt)
    f = Ef.Eenvelope(tt) * np.linalg.norm(data["amp"])
    En = np.linalg.norm(E,axis=1)

    fig, ax = plt.subplots(figsize=(10,6))

    f  = convert(f ,"electric-field","atomic_unit","v/ang")
    E  = convert(E ,"electric-field","atomic_unit","v/ang")
    En = convert(En,"electric-field","atomic_unit","v/ang")

    ax.plot(t,f,label="$f_{env} \\times E_{amp}$",color="black")
    ax.plot(t,En,label="$|E|$",color="gray",alpha=0.5)
    ax.plot(t,E[:,0],label="$E_x$",color="red",alpha=0.5)
    ax.plot(t,E[:,1],label="$E_y$",color="green",alpha=0.5)
    ax.plot(t,E[:,2],label="$E_z$",color="blue",alpha=0.5)

    plt.ylabel("electric field [V/ang]")
    plt.xlabel("time [fs]")
    plt.grid()
    plt.legend()

    plt.tight_layout()

    print("\n\tSaving plot to {:s}".format(output))
    plt.savefig(output)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def extract(keys,families,scope):

    data = {}
    for key,family in zip(keys,families):

        data[key] = None
        
        element = scope.find(key)

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

def get_data(options):

    print("\tReading json file")
    # # Open the JSON file and load the data
    # with open(options.input) as f:
    #     info = json.load(f)

    data = xmlet.parse(options.input).getroot()

    efield = None
    simulation = None
    dynamics = None
    for element in data.iter():
        if element.tag == "efield":
            efield = element
        if element.tag == "simulation":
            simulation = element
        if element.tag in  ["dynamics","driven_dynamics"]:
            dynamics = element

    data     = {}
    keys     = [ "amp"           , "freq"     , "phase"    , "peak", "sigma" ]
    families = [ "electric-field", "frequency", "undefined", "time", "time"  ]
    
    data0 = extract(keys,families,efield)
    data1 = extract(["timestep",],["time"],dynamics)
    data2 = extract(["total_steps"],["number"],simulation )

    return {**data0,**data1,**data2}

def prepare_parser(description):
    """set up the script input parameters"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", action="store", type=str,help="input file", default="input.xml")
    parser.add_argument("-o", "--output", action="store", type=str,help="output file ", default="Efield.pdf")
    parser.add_argument("-t", "--t_max", action="store", type=float,help="max time")
    parser.add_argument("-dt", "--time_spacing", action="store", type=float,help="max time",default=1)
    parser.add_argument("-u", "--unit", action="store", type=str,help="unit",default="picosecond")
    return parser

@esfmt(prepare_parser,description)
def main(args):

    data = get_data(args)
    print("\tData:")
    show_dict(data,string="\t",width=15)
    Ef = ElectricField( amp=data["amp"],\
                        phase=data["phase"],\
                        freq=data["freq"],\
                        peak=data["peak"],\
                        sigma=data["sigma"])

    # plot of the E-field
    Ef_plot(Ef,data,args.output)

    factor = convert(1,"time","atomic_unit","femtosecond")
    period = (2*np.pi) / data['freq']
    P1 = factor * period * np.floor( data["peak"] / period )
    P2 = factor * period * np.ceil( data["peak"] / period )

    print("\n\tPossible choices for the peak:")
    print("\t - {:f} fs".format(P1))
    print("\t - {:f} fs".format(P2))

    # plot of the E-field FFT
    # FFT_plot(Ef,data,options)

    return 0 


if __name__ == "__main__":
    main()