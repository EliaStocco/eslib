# pylint: disable=import-error
#import _rdfs_fort as fortran_rdfs # type: ignore
from ase import Atoms
from ase.cell import Cell
import numpy as np

def fortran_intermolecular_rdfs_fixed_cell(rdf,posA,posB,rmin,rmax,cell,invcell,partitionA,partitionB,massA,massB)->None:                
    try:
        import eslib.fortran._rdfs_fort as RDF  # type: ignore
    except:
        import eslib.fortran.rdf._rdfs_fort as RDF  # type: ignore
    RDF.intermolecularrdf(rdf,posA,posB,rmin,rmax,cell,invcell,partitionA,partitionB,massA,massB)
    
def fortran_rdfs_fixed_cell(rdf,posA,posB,rmin,rmax,cell,invcell,massA,massB)->None:                
    """
    Calculate the radial distribution function (RDF) using the Fortran implementation.

    Args:
        rdf (numpy.ndarray): The input RDF array to be updated.
        posA (numpy.ndarray): The positions of atom A.
        posB (numpy.ndarray): The positions of atom B.
        rmin (float): The minimum radius for the RDF calculation.
        rmax (float): The maximum radius for the RDF calculation.
        cell (numpy.ndarray): The cell parameters.
        invcell (numpy.ndarray): The inverse of the cell parameters.
        massA (float): The mass of atom A.
        massB (float): The mass of atom B.
    """
    try:
        import eslib.fortran._rdfs_fort as RDF  # type: ignore
    except:
        import eslib.fortran.rdf._rdfs_fort as RDF  # type: ignore
    RDF.updateqrdffixedcell(rdf,posA,posB,rmin,rmax,cell,invcell,massA,massB)

def fortran_rdfs_variable_cell(rdf,posA,posB,rmin,rmax,cell,invcell,massA,massB)->None:                
    """
    Calculate the radial distribution function (RDF) using the Fortran implementation.

    Args:
        rdf (numpy.ndarray): The input RDF array to be updated.
        posA (numpy.ndarray): The positions of atom A.
        posB (numpy.ndarray): The positions of atom B.
        rmin (float): The minimum radius for the RDF calculation.
        rmax (float): The maximum radius for the RDF calculation.
        cell (numpy.ndarray): The cell parameters.
        invcell (numpy.ndarray): The inverse of the cell parameters.
        massA (float): The mass of atom A.
        massB (float): The mass of atom B.
    """
    try:
        import eslib.fortran._rdfs_fort as RDF  # type: ignore
    except:
        import eslib.fortran.rdf._rdfs_fort as RDF  # type: ignore
    RDF.updateqrdfvariablecell(rdf,posA,posB,rmin,rmax,cell,invcell,massA,massB)
    
def fortran_interatomic_distances(atoms:Atoms)->np.ndarray:
    try:
        import eslib.fortran._rdfs_fort as RDF  # type: ignore
    except:
        import eslib.fortran.rdf._rdfs_fort as RDF  # type: ignore
    N = atoms.get_global_number_of_atoms()
    cell = np.asarray(atoms.get_cell()).T
    invCell = np.linalg.inv(cell)  
    distances = np.array((N,N), order="F")
    positions = np.zeros((N,3), order="F")
    positions[:,:] = atoms.get_positions()
    RDF.interatomicdistances(positions, cell, invCell, N, distances)
    return distances