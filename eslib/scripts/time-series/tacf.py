#!/usr/bin/env python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from eslib.mathematics import tacf
from eslib.plot import hzero
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.input import str2bool
from eslib.formatting import esfmt
from eslib.classes.tcf import TimeAutoCorrelation

#---------------------------------------#
# Description of the script's purpose
description = "Compute the time autocorrelation function (TACF) of a dipole time series."

alpha = 0.5

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # I/O
    parser.add_argument("-i" , "--input"      , **argv, required=True , type=str     , help="txt/npy input file")
    parser.add_argument("-o" , "--output"     , **argv, required=False, type=str     , help="txt/npy output file (default: %(default)s)", default='tacf.npy')
    # Calculations
    parser.add_argument("-dt", "--time_step"  , **argv, required=False, type=float   , help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-d" , "--derivative" , **argv, required=False, type=str2bool, help="compute derivative of the input data (default: %(default)s)", default=False)
    parser.add_argument("-n" , "--normalize" , **argv, required=False, type=str2bool, help="whether to normalize the autocorrelation (default: %(default)s)", default=False)
    # parser.add_argument("-b" , "--blocks"     , **argv, required=False, type=int     , help="number of blocks (default: %(default)s)", default=10)
    parser.add_argument("-ac", "--axis_corr"  , **argv, required=False, type=int     , help="axis along compute autocorrelation (default: %(default)s)", default=1)
    parser.add_argument("-as", "--axis_sum"  , **argv, required=False, type=int     , help="axis along compute the sum (default: %(default)s)", default=2)
    parser.add_argument("-rm", "--remove_mean", **argv, required=False, type=str2bool, help="whether to remove the mean (default: %(default)s)", default=True)
    parser.add_argument("-m" , "--method"     , **argv, required=False, type=str     , help="method (default: %(default)s)", default='class', choices=['class','function'])
    # Plot
    parser.add_argument("-p" , "--plot"       , **argv, required=False, type=str     , help="output file for the plot (default: %(default)s)", default=None)
    parser.add_argument("-tm", "--tmax"       , **argv, required=False, type=float   , help="max time in TACF plot [fs] (default: %(default)s)", default=None)
    parser.add_argument("-f" , "--fit"        , **argv, required=False, type=str2bool, help="whether to fit the TACF with an exponential (default: %(default)s)", default=True)
    # Window and padding
    parser.add_argument("-w"   , "--window"   , **argv, required=False, type=str     , help="window type (default: %(default)s)", default='hanning', choices=['none','barlett','blackman','hamming','hanning','kaiser'])
    parser.add_argument("-wt"  , "--window_t" , **argv, required=False, type=int     , help="time span of the window [fs] (default: %(default)s)", default=10)
    return parser

#---------------------------------------#
# Define the exponential function to fit
def sin_exp(x, tau, omega, phi,A):
    """
    Function to compute the exponential of x divided by tau.

    Parameters:
        x (float): The input value.
        tau (float): The time constant.
        omega (float): The angular frequency.
        phi (float): The phase.

    Returns:
        float: The exponential of -x/tau.

    This function calculates the exponential of x divided by tau,
    where x is the input value, tau is the time constant, omega is
    the angular frequency, and phi is the phase. The function
    multiplies the sine of the product of omega and x plus phi
    with the exponential of -x/tau.
    """
    # Compute the sine of the product of omega and x plus phi
    sin_term = np.sin(omega * x + phi)

    # Compute the exponential of -x/tau
    exp_term = np.exp(-x / tau)

    # Multiply the sine term with the exponential term
    return A*sin_term * exp_term
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    # assert args.derivative == True, "If derivative == False there is a bug."

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    args.input = str(args.input)
    data:np.ndarray = PhysicalTensor.from_file(file=args.input).to_data()
    print("done")
    print("\tdata shape: ",data.shape)

    # #------------------#
    # if args.blocks > 0 :
    #     print("\n\tN. of blocks: ", args.blocks)
    #     print("\tBuilding blocks ... ",end="")
    #     data = reshape_into_blocks(data,args.blocks)# .T
    #     print("done")
    #     print("\tdata shape: ",data.shape)

    #------------------#
    if args.remove_mean:
        print("\n\tRemoving mean ... ",end="")
        data -= np.mean(data,axis=args.axis_corr,keepdims=True)
        print("done")

    #------------------#
    if args.derivative:
        print("\n\tComputing the derivative ... ",end="")
        data = np.gradient(data,axis=args.axis_corr)/args.time_step
        print("done")
        print("\tdata shape: ",data.shape)    

    #------------------#
    print("\n\tComputing the autocorrelation function ... ", end="")
    if args.method == "function":
        autocorr = tacf(data)
    else:
        obj = TimeAutoCorrelation(data)
        autocorr = obj.tcf(axis=args.axis_corr,normalize=args.normalize)
    print("done")
    print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    if not args.normalize and not args.derivative:
        print("\n\tChecking that the autocorrelation at t=0 is equal to the fluctuation ... ",end="")
        tmp = np.var(data,axis=args.axis_corr)
        assert np.allclose(tmp,autocorr[:,0]), "Something is wrong with the normalization."
        print("done")
        print("\tEverything okay!")

    #------------------#
    if args.axis_sum is not None:
        print("\n\tComputing the sum along the axis {:d} ... ".format(args.axis_sum),end="")
        autocorr = np.sum(autocorr,axis=args.axis_sum)
        print("done")
        print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    if autocorr.ndim == 1:
        print("\tReshaping data  ... ", end="")
        autocorr = np.atleast_2d(autocorr)# .T
        
        print("done")
        print("\tdata shape: ",data.shape)

    #------------------#
    print("\n\tComputing the average over the trajectories:")
    mean = autocorr.mean(axis=0)
    std = autocorr.std(axis=0)
    x = np.arange(autocorr.shape[1])*args.time_step
    print("\t   x shape: ",x.shape)
    print("\tmean shape: ",mean.shape)
    print("\t std shape: ",std.shape)

    raw_autocorr = autocorr.copy()

    #------------------#
    if args.fit:
        print("\n\tFitting the autocorrelation function with an exponential ... ",end="")

        # Fit the data to the exponential function, providing the standard deviation as sigma
        popt, pcov = curve_fit(f=sin_exp, 
                            xdata=x, 
                            ydata=mean,
                            sigma=std,
                            p0=[1,1,1,1])

        yfit = sin_exp(x,popt[0],0,popt[2],popt[3])
        print("done")
        print("\ttau: {:.0f}fs".format(popt[0]))

    #------------------#
    if args.plot is not None:
        width = 8 if args.window == "none" else 12
        nplots = 2 if args.window == "none" else 3
        fig, axes = plt.subplots(nplots, 1, figsize=(10, width),sharex=True)

        for i in range(raw_autocorr.shape[0]):
            y = raw_autocorr[i]
            axes[0].plot(x,y, label=f'{i+1}')

        axes[1].plot(x,mean,color="blue",label='$f$')
        ylow,yhigh = mean - std, mean + std
        axes[1].fill_between(x,ylow,yhigh, color='gray', alpha=alpha, label='$f\\pm\\sigma$')
        # ylow,yhigh = mean - 2*std, mean + 2*std
        # axes[1].fill_between(x,ylow,yhigh, color='gray', alpha=alpha/2., label='$f\\pm2\\sigma$')

        if args.fit:
            axes[1].plot(x,yfit,color="red",label='$e^{-x/\\tau}$',linewidth=1)

        if args.fit:
            # Add text box with the value of a variable
            textbox_text = "$\\tau\\approx{:.0f}$fs".format(popt[0]) 
            # plt.text(0.1, 0.05, textbox_text,  ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            axes[1].annotate(textbox_text, xy=(0.5, 0.95), xycoords='axes fraction', fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', alpha=1,edgecolor="black"))
        axes[1].legend(loc="upper right",facecolor='white', framealpha=1,edgecolor="black")

                

    #------------------#
    if args.window != "none" :
        print("\n\tApplying the '{:s}' window ... ".format(args.window),end="")
        #Define window to be used with DCT below.
        func = getattr(np, args.window)
        window = np.zeros(raw_autocorr.shape[args.axis_corr])
        M = int(args.window_t / args.time_step)
        window[:M] = func(2*M)[M:]
        # window = np.atleast_2d(window)
        print("done")
        print("\twindow shape: ",window.shape)

        autocorr = raw_autocorr * window

        
        # raw_autocorr = np.atleast_2d(raw_autocorr)

        if args.plot is not None:
            # raw TACF
            axes[2].plot(x,mean,color="blue",label='$f$')
            # ylow,yhigh = mean - std, mean + std
            # axes[2].fill_between(x,ylow,yhigh, color='gray', alpha=alpha)
            # cleaned TACF
            # autocorr = autocorr * window
            mean = autocorr.mean(axis=0)
            std = autocorr.std(axis=0)
            axes[2].plot(x,mean,color="red",label='$f_{\\rm cleaned}$')
            ylow,yhigh = mean - std, mean + std
            axes[2].fill_between(x,ylow,yhigh, color='gray', alpha=alpha, label='$f_{\\rm cleaned}\\pm\\sigma$')
            axes[2].legend(loc="upper right",facecolor='white', framealpha=1,edgecolor="black")
            # window
            axes[2].plot(x,window,  color="black",label='window')


    #------------------#
    if args.plot is not None:
        print("\n\tSaving plot to file '{:s}'... ".format(args.plot), end="")
        for ax in axes:
            ax.grid()
            if args.tmax is not None:
                ax.set_xlim(0,min(args.tmax,x.max()))
            hzero(ax)
            # ax.set_xlim(1,x.max())
            # ax.set_xscale("log")
        axes[-1].set_xlabel('time [fs]')
        plt.suptitle("Time Autocorrelation Function")
        plt.tight_layout()
        plt.savefig(args.plot)
        print("done")
    

    #------------------#
    print("\n\tConverting autocorr into PhysicalTensor ... ", end="")
    autocorr = PhysicalTensor(autocorr)
    print("done")
    print("\tautocorr shape: ",autocorr.shape)

    #------------------#
    print("\tSaving TACF to file '{:s}' ... ".format(args.output), end="")
    autocorr.to_file(file=args.output)
    print("done")

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()