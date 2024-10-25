# pylint: disable=import-error
#import _rdfs_fort as fortran_rdfs # type: ignore

def fortran_rdfs(rdf,posA,posB,rmin,rmax,cell,invcell,massA,massB)->None:                
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
    RDF.updateqrdf(rdf,posA,posB,rmin,rmax,cell,invcell,massA,massB)