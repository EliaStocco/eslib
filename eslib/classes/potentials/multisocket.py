
import numpy as np
from ase.calculators.socketio import SocketIOCalculator
from ase.calculators.calculator import Calculator
from ase import Atoms
from typing import List, TypeVar, Any, Dict, Set

T = TypeVar('T', bound='MultiSocket')

class MultiSocket(Calculator):
    
    implemented_properties:List[str] = ['energy', 'free_energy', 'forces', 'stress']
    supported_changes:Set[str] = {'positions', 'cell'}
    system_changes:List[str] = ['positions', 'cell']
    results:Dict[str,Any]
    
    drivers:List[SocketIOCalculator]

    def __init__(self:T,log,ports,unixsockets)->None:
        """
        Initialize the MultiSocket class.

        Parameters
        ----------
        log : bool
            Enable logging.
        ports : List[int]
            List of TCP ports to connect to.
        unixsockets : List[str]
            List of Unix socket paths to connect to.

        Returns
        -------
        None
        """
        Calculator.__init__(self)
        self.results = {}
        
        assert len(ports) == len(unixsockets), "ports and unixsockets must have the same length"
        N = len(ports)
        self.drivers = [None]*N
        for n,(port,unixsocket) in enumerate(zip(ports,unixsockets)):
            tmp = SocketIOCalculator(port=port,unixsocket=unixsocket,log=log)
            self.drivers[n] = tmp
            
    def __enter__(self:T):
        """
        Enter the runtime context for the MultiSocket instance.

        Returns
        -------
        MultiSocket
            The instance of the MultiSocket class.
        """
        return self

    def __exit__(self:T, *args):
        """
        Exit the runtime context for the MultiSocket instance.

        Close all the socket connections, and all the related resources.

        Parameters
        ----------
        *args
            Ignored.

        Returns
        -------
        None
        """
        for driver in self.drivers:
            driver.close()
            
    def calculate(self, atoms: Atoms = None, properties=None, system_changes=None):
        """
        Calculate properties.

        Parameters
        ----------
        atoms : Atoms, optional
            Atomic system to be calculated. Defaults to None.
        properties : list, optional
            Properties to calculate. Defaults to None.
        system_changes : list, optional
            System changes. Defaults to None.

        Returns
        -------
        None
        """
        Calculator.calculate(self, atoms)
        self.results = {}

        # Compute potential and forces
        Natoms = atoms.get_global_number_of_atoms()
        
        # Populate results
        self.results["energy"]      = 0
        self.results["free_energy"] = 0
        self.results["forces"]      = np.zeros((Natoms, 3))
        self.results["stress"]      = np.zeros(6)
        
        for driver in self.drivers:
            
            driver.calculate(atoms,system_changes=self.system_changes)
            
            e  = driver.results["energy"]
            fe = driver.results["free_energy"]
            f  = driver.results["forces"]
            s  = driver.results["stress"]
            
            self.results["energy"]      += e
            self.results["free_energy"] += fe
            self.results["forces"]      += f
            self.results["stress"]      += s if s is not None else 0
            
        pass
            
        
            
    