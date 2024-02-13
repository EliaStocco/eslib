# from elia.classes import MicroState
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

def visualize_datasets(datasets:dict,variable:str,folder:str="images"):

    if variable is None:
        warnings.warn("Datasets not visualizable for this kind of variable, i.e. '{:s}'".format(variable))

    if not os.path.exists(folder):
        print("\tCreating folder '{:s}'".format(folder))
        os.mkdir(folder)

    for k in datasets.keys():

        d = datasets[k]
        N = len(d)
        tmp = d[0][variable].numpy()
        quantity = np.zeros((N,*tmp.shape))
        for n in range(N):
            quantity[n] = d[n][variable].numpy()

        filename = "{:s}/{:s}.{:s}.txt".format(folder,variable,k)
        np.savetxt(filename,quantity)

        
        if len(tmp.shape) == 0:
            print("scalar")

            


        elif len(tmp.shape) == 1:
            
            n = tmp.shape[0]
            fig, axes = plt.subplots(ncols=n,figsize=(5*n,5))
            for n,ax in enumerate(axes):
                arr = quantity[:,n]
                bins = np.histogram_bin_edges(arr, bins='scott')
                ax.hist(arr,color='navy',bins=20)
                ax.grid()

            plt.title("{:s} ({:s} dataset, {:d} points)".format(variable,k,N))
            plt.tight_layout()
            filename = "{:s}/{:s}.{:s}.pdf".format(folder,variable,k)
            plt.savefig(filename)

        else :
            raise ValueError("It's not possible to print this kind of variable")
        
    return
        

    fig, ax = plt.subplots(figsize=(10,6))

    factor = unit_to_user("time","picosecond",1)
    time = time*factor
    ax.plot(time,normalized_occupations)

    # plt.title('LiNbO$_3$ (NVT@$20K$,$\\Delta t = 1fs$,T$=20-50ps$,$\\tau=10fs$)')
    ax.set_ylabel("$A^2_s\\omega^2_s / \\left( 2 N \\right)$ with $N=E_{harm}\\left(t\\right)$")
    ax.set_xlabel("time (ps)")
    ax.set_xlim(min(time),max(time))
    ylim = ax.get_ylim()
    #ax.set_ylim(0,ylim[1])
    ax.set_yscale("log")

    plt.grid()
    plt.tight_layout()
    plt.savefig("{:s}/dataset.pdf".format(folder))

    
    pass
