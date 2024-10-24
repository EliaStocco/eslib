#!/usr/bin/env python

from __future__ import division, print_function

import argparse
# from numba import jit
import random
from itertools import combinations

import numpy as np
from asap3.analysis.rdf import RadialDistributionFunction
from ase.io import read, write
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import energy_distance, entropy, wasserstein_distance
from sklearn.metrics import log_loss, pairwise, pairwise_distances


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', type=str, nargs='+', required=True,
                        help='files with the frames (any format ASE reads)')
    parser.add_argument('--rmax', '-r', type=float, default=9.0,
                        help='max. radius')
    parser.add_argument('--nbins', '-n', type=int, default=200,
                        help='number of bins')
    parser.add_argument('--set_lattice', '-sl', type=float, nargs='+',
                        help='set lattice parameters (in angstrom)')
    parser.add_argument('--from_frame', '-ff', type=int, default=0,
                        help='start from frame nr. <n>')
    parser.add_argument('--to_frame', '-tf', type=int,
                        help='end at frame nr. <n>')
    parser.add_argument('--jump', '-j', type=int, default=1,
                        help='jump frames')
    parser.add_argument('--units', '-u', type=str, default='angstrom', choices=['angstrom', 'bohr'],
                        help='coordinates units')
    parser.add_argument('--nsample', '-ns', type=int, required=True,
                        help='number of samples')
    parser.add_argument('--ntries', '-nt', type=int, default=100,
                        help='number of tries')
    args = parser.parse_args()
    return args


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def show_matrix(A):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ms = ax.matshow(A[...], cmap='inferno')
    cbar = fig.colorbar(ms)
    plt.show()


@jit(parallel=True)
def sum_dist(pairs, dist):
    result = 0
    for i1, i2 in pairs:
        result += dist[i1, i2]
    return result


def main():
    
    args = read_arguments()
    
    file_list  = args.files
    r_max      = args.rmax
    n_bins     = args.nbins
    lattice    = args.set_lattice
    ini_frame  = args.from_frame
    end_frame  = args.to_frame
    jump       = args.jump
    units      = args.units
    n_sample   = args.nsample
    n_tries    = args.ntries
    
    if units == 'bohr':
        raise NotImplemented
        unit_conv = 0.529177
    
    if end_frame == None:
        print('will read frames: %i::%i' % (ini_frame, jump))
        slice_ = '%i::%i' % (ini_frame, jump)
    else:
        print('will read frames: %i:%i:%i' % (ini_frame, end_frame, jump))
        slice_ = '%i:%i:%i' % (ini_frame, end_frame, jump)
    
    rdfs = []
    all_frames = None
    O, H = 8, 1
    x = np.arange(n_bins) * r_max / n_bins
    dx = x[1] - x[0]
    eps = 1e-3
    
    for f in file_list:
        frames = read(f, index=slice_)
        print('read file %s with %i frames' % (f, len(frames)))
        for frame in frames:
            if lattice is not None:
                frame.set_cell(lattice)
                frame.set_pbc((True, True, True))
            rdf = RadialDistributionFunction(frame, r_max, n_bins)
            rdf_OO = rdf.get_rdf(elements=(O,O)) + eps
            rdf_OH = rdf.get_rdf(elements=(O,H)) + eps
            rdf_HH = rdf.get_rdf(elements=(H,H)) + eps
            rdfs.append( np.concatenate((rdf_OO, rdf_OH, rdf_HH)) )
            rdfs[-1] *= dx / rdfs[-1].sum()
        if all_frames == None:
            all_frames = frames
        else:
            all_frames += frames

    del rdf, rdf_OO, rdf_OH, rdf_HH
    
    n_frames = len(rdfs)
    rdfs = np.array(rdfs)
    
    print('total number of frames is %i' % n_frames)
    
    # draw candidate populations first
    print('sampling %i frames for each %i try from all %i read frames' % (n_sample, n_tries, n_frames))
    population = np.zeros([n_tries, n_sample], dtype=int)
    for i in range(n_tries):
        population[i, ...] = random.sample(range(n_frames), n_sample)
    
    # keep only candidates to save memory and compute time
    uniq_population, uniq_indices = np.unique(population, return_inverse=True)
    n_uniq_population = len(uniq_population)
    rdfs = rdfs[uniq_population]
    print('keeping only %i unique frames' % (n_uniq_population))
    
    metrics = {'energy'       : energy_distance, 
               'wasserstein'  : wasserstein_distance,
               'jensenshannon': jensenshannon, 
               'logloss'      : log_loss,
               'cosine'       : 'cosine' }
    optimal = {'energy'       : np.argmax,
               'wasserstein'  : np.argmax,
               'jensenshannon': np.argmax,
               'logloss'      : np.argmax, 
               'cosine'       : np.argmin }
    
    my_metric = 'jensenshannon'
    
    metric = metrics[my_metric]
    optima = optimal[my_metric]
    
    print('calculating pairwise distances with metric: %s' % my_metric)
    dist = pairwise_distances(rdfs, metric=metric, n_jobs=-1)
    del rdfs
    
    # find best samples
    print('finding optimal %i samples searching for %s' % (n_sample, optima.__name__))
    loss = np.zeros(n_tries)
    for i in range(n_tries):
        pair_indices = np.array(list(combinations(population[i, ...], 2)))
        pair_uniq_indices = np.array([[np.where(uniq_population == i1), np.where(uniq_population == i2)] for (i1, i2) in pair_indices]).reshape(-1,2)
        #for i1, i2 in pair_uniq_indices:
            #loss[i] += dist[i1, i2]
        loss[i] = sum_dist(pair_uniq_indices, dist)
        #loss[i] = dist[pair_uniq_indices].sum()
        loss[i] /= n_sample
        print("try %i: loss: %f" % (i, loss[i]))

    best = optima(loss)
    print('best\'s loss: %f' % loss[best])
    print(population[best, ...])
    best_frames = []
    for i in population[best, ...]:
        best_frames.append(all_frames[i])
    print('\n')
    
    write('best_sample.xyz', best_frames)
    #show_matrix(dist)


if __name__ == '__main__':
    main()
    
