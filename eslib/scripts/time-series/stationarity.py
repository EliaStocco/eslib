#!/usr/bin/env python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format
from eslib.input import slist
from eslib.plot import hzero

matplotlib.use('QtAgg')
# Disable all warnings
warnings.filterwarnings('ignore')
#---------------------------------------#
# Description of the script's purpose
description = "Check if a time series is stationary."


#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="input file ")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n" , "--name"        , **argv, required=False, type=slist, help="name for the new info/array (default: %(default)s)", default=None)
    parser.add_argument("-sl" , "--significance_level"        , **argv, required=False, type=float, help="significance level (default: %(default)s)", default=0.05)
    parser.add_argument("-N" , "--interval"  , **argv,type=int, help="interval (default: %(default)s", default=100)  
    parser.add_argument("-o" , "--output", **argv,type=str, help="output folder for plots (default: %(default)s)", default=None)
    # parser.add_argument("-of", "--output_format", **argv,type=str, help="output format for np.savetxt (default: %(default)s)", default=float_format)
    return parser# .parse_args()

# def check_stationarity_adfuller(ts):
#     dftest = adfuller(ts)
#     adf = dftest[0]
#     pvalue = dftest[1]
#     critical_value = dftest[4]['5%']
#     if (pvalue < 0.05) and (adf < critical_value):
#         return True
#     else:
#         return False

# https://neptune.ai/blog/arima-sarima-real-world-time-series-forecasting-guide
def is_stationary_adf(series: np.ndarray, significance_level: float = 0.05) -> bool:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.

    Args:
        series (np.ndarray): The time series to test stationarity for.
        significance_level (float, optional): The significance level for the test. Defaults to 0.05.

    Returns:
        bool: True if the time series is stationary, False otherwise.
    """
    # Run the Augmented Dickey-Fuller test
    result = adfuller(series)

    # Extract ADF test statistic and critical value
    adf  = result[0]
    critical_value = result[4]['5%']

    # Extract the p-value from the test
    p_value = result[1]

    # Check if the p-value is below the significance level and ADF test statistic is less than critical value
    return (p_value < significance_level) and (adf < critical_value), p_value

def is_stationary_kpss(series: np.ndarray, significance_level: float = 0.05) -> bool:
    """
    Check if a time series is stationary using the KPSS test.

    Args:
        series (np.ndarray): The time series to test stationarity for.
        significance_level (float, optional): The significance level for the test. Defaults to 0.05.

    Returns:
        bool: True if the time series is stationary, False otherwise.
    """
    # Run the KPSS test
    result = kpss(series)

    # Extract the p-value from the test
    p_value = result[1]

    # Check if the p-value is below the significance level
    return p_value >= significance_level, p_value
    
# def check_stationarity_kpss(ts):
#     dftest = kpss(ts)
#     adf = dftest[0]
#     pvalue = dftest[1]
#     critical_value = dftest[4]['5%']
#     if (pvalue < 0.05) and (adf < critical_value):
#         return True
#     else:
#         return False
    
# def check_stationarity_kpss(timeseries):
#     print("Results of KPSS Test:")
#     kpsstest = kpss(timeseries, regression="c", nlags="auto")
#     kpss_output = pd.Series(
#         kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
#     )
#     for key, value in kpsstest[3].items():
#         kpss_output["Critical Value (%s)" % key] = value
#     print(kpss_output)

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    print("\tn. of structures: {:d}".format(len(atoms)))

    print("\n\tAvailable info: ",atoms.get_keys("info"))

    #---------------------------------------#
    # summary
    print("\n\tSummary of the atomic structures: ")
    df = atoms.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n","\n\t\t"))
    print()

    #---------------------------------------#
    if args.name is None:
        args.name = atoms.get_keys()

    #---------------------------------------#
    tmp = df.set_index("key")
    keys = [ a  for a in df["key"] if a in args.name and tmp.at[a,"numeric"] ]
    print("\n\tConsidered keys: ",keys)

    #---------------------------------------#
    print("\n\tChecking stationarity ... ")
    keys = np.array(keys)    
    timeseries = {k:atoms.get(k) for k in keys}
    adf = [is_stationary_adf(timeseries[k],significance_level=args.significance_level)[0] for k in keys]
    kpss = [is_stationary_kpss(timeseries[k],significance_level=args.significance_level)[0] for k in keys]
    df = pd.DataFrame({"key":keys,"adf":adf,"kpss":kpss})
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n","\n\t\t"))
    print()

    #---------------------------------------#
    # output directory
    if args.output is not None:
        os.makedirs(args.output,exist_ok=True)

        print("\n\tPreparing plots:")
        for k in keys:
            file = os.path.join(args.output, "{:s}.pdf".format(k))
            print(" - \t{:s} ... ".format(k),end="")
            ts = timeseries[k]
            interval = args.interval
            N = int(len(ts)/interval)
            arrays = [ts[i*interval:] for i in range(N)]
            results = pd.DataFrame({
                "index" : [i*interval for i in range(N)],
                "adf" : [is_stationary_adf(arr,significance_level=args.significance_level)[1] for arr in arrays],
                "kpss" : [is_stationary_kpss(arr,significance_level=args.significance_level)[1] for arr in arrays],
            })

            results.to_csv(os.path.join(args.output,"{:s}.csv".format(k)),index=False)

            fig, ax = plt.subplots()
            ax.plot(results['index'], results['adf'], color='blue', label='ADF')
            ax.plot(results['index'], results['kpss'], color='red', label='KPSS')
            ax.set_ylabel('p-values')
            ax.set_xlabel('index')
            hzero(ax,args.significance_level,label="sig. level",color="black")
            # ax.set_title('Results for {:s}'.format(k))
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            print(" saving plot to '{:s}' ... ".format(file),end="")
            plt.savefig(file)
            print("done")
    
    return 0    

#---------------------------------------#
if __name__ == "__main__":
    main()

