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
from eslib.classes.efield import ElectricField

description = "Plot the electric field E(t) into a pdf file."

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