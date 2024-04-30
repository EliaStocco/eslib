from ase.io import read, write
from ase import Atoms
import numpy as np
from eslib.classes.normal_modes import NormalModes
from eslib.geometry import angle_between_vectors
from icecream import ic
from ase.geometry import distance
import pandas as pd
from ase.neighborlist import neighbor_list
from eslib.tools import max_pbc_distance, relative_vectors
from eslib.plot import histogram

# stdau:Atoms = read("../start.au.extxyz")
std:Atoms = read("std.ang.extxyz")
inv:Atoms = read("inv.ang.extxyz")

direction = np.asarray( inv.get_positions() - std.get_positions() ).flatten()
dist = np.linalg.norm(direction)
direction /= dist
ic(dist," ang")

# # Compute interatomic distances
std_inv = distance(std,inv,permute=True)
ic(std_inv)

# # Print distances
# print("Interatomic distances [std]:")
# for i, j, dist in distances:
#     print("Distance between atom {} and atom {}: {:.2f} Å".format(i, j, dist))

mpd = max_pbc_distance(std.get_cell())
std_info = neighbor_list('ijdD',std,mpd)
std_rel_pos = std_info[3]

std_dist = relative_vectors(std)
inv_dist = relative_vectors(inv)

histogram(inv_dist['d'],"inv.pdf")
histogram(std_dist['d'],"std.pdf")

inv_dist.to_csv('inv.csv')
std_dist.to_csv('std.csv')

inv_dist['x'] = -inv_dist['x']
inv_dist['y'] = -inv_dist['y']
inv_dist['z'] = -inv_dist['z']

new_indices = [ f'{i}{j}' for i,j in zip(std_dist['i'],std_dist['j']) ]
std_dist.set_index(pd.Index(new_indices), inplace=True)

new_indices = [ f'{i}{j}' for i,j in zip(inv_dist['i'],inv_dist['j']) ]
inv_dist.set_index(pd.Index(new_indices), inplace=True)

# search for correspective pairs
std_vec = np.asarray([std_dist["x"],std_dist["y"],std_dist["z"]]).T

min_indices = np.zeros(len(std_dist))
k = 0 
for index,row in inv_dist.iterrows():
    inv_vec = np.asarray([inv_dist.at[index,"x"],inv_dist.at[index,"y"],inv_dist.at[index,"z"]]).flatten()
    min_indices[k] = np.linalg.norm( std_vec - inv_vec , axis=1).argmin()
    k += 1
ic(min_indices)


# Diffusion
std2 = std.copy()
std2.set_positions(std.get_positions()+(direction*2*dist).reshape(-1,3))

std_std2 = distance(std,std2,permute=True)
ic(std_std2)

write('diffused.std.extxyz',std2)