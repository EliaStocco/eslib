#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from eslib.formatting import esfmt, eslog
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

#---------------------------------------#
# Description of the script's purpose
description = "Short time FFT."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="one of the csv produced by 'dataset4molecules-pairs.py'")
    parser.add_argument("-k" , "--keyword"      , **argv, required=True , type=str, help="keyword for the dipole")
    # parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="output folder")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # data
    with eslog(f"Reading data fram file '{args.input}'"):
        df = pd.read_csv(args.input)
        
        # Sort by that column
        df = df.sort_values(by="time")

        # Check that values are unique
        if not df["time"].is_unique:
            raise ValueError("Column 'time' contains duplicate values.")

        # Optionally reset the index if you need a clean DataFrame
        df = df.reset_index(drop=True)

    #------------------#
    with eslog("\nExtracting dipoles"):
        i_keys = [f"{args.keyword}_{n}_i" for n in range(3)]
        j_keys = [f"{args.keyword}_{n}_j" for n in range(3)]
        
        mu_i = np.asarray(df[i_keys])
        mu_j = np.asarray(df[j_keys])
    assert mu_i.shape == mu_j.shape, "Dipoles must have the same shape"
    print(f"\t dipole.shape: {mu_i.shape}")
    
    with eslog("\nExtracting distances"):
        distances = df["distance"].to_numpy()
    assert distances.ndim == 1, "Distances must be a 1D array"
    print(f"\t distances.shape: {distances.shape}")
    
    #------------------#
    
    x = mu_i[:,0]
    t = np.linspace(0, 2 * np.pi, len(x))
    x = np.sin(10*t) + np.cos(3*t)# + x  # add some noise
    
    # T_x, N = 1 / 20, 1000  # 20 Hz sampling rate for 50 s signal
    # t_x = np.arange(N) * T_x  # time indexes for signal
    # f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # varying frequency
    # x = np.sin(2*np.pi*np.cumsum(f_i)*T_x) # the signal
    
    x = mu_i[:,0]
    x -= np.mean(x)  # remove DC offset
    T_x = 1
    N = len(x)  # number of samples in the signal
    
    g_std = 8  # standard deviation for Gaussian window in samples
    w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(w, hop=10, fs=1/T_x, mfft=200, scale_to='magnitude')
    
    


    Sx = SFT.stft(x)  # perform the STFT
    
    import matplotlib.pyplot as plt
    # N = len(mu_i)
    fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
    # t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    # ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gaussian window, " +
    #             rf"$\sigma_t={g_std*SFT.T}\,$s)")
    # ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
    #             rf"$\Delta t = {SFT.delta_t:g}\,$s)",
    #         ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
    #             rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
    #         xlim=(t_lo, t_hi))
    im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                    extent=SFT.extent(N), cmap='viridis')
    # ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
    fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
    # Shade areas where window slices stick out to the side:
    # for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
    #                 (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
    #     ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
    # for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
    #     ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
    # ax1.legend()
    fig1.tight_layout()
    plt.show()
    
    return
            
#---------------------------------------#
if __name__ == "__main__":
    main()


