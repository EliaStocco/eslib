from ase import Atoms
from eslib.classes.models.mace_model import MACEModel
from ase.calculators.calculator import Calculator, all_changes
import threading  # Replace multiprocess with threading
from typing import List, TypeVar, Dict, Tuple
from ase.calculators.socketio import SocketClient
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # ThreadPoolExecutor for better thread management

T = TypeVar('T', bound='SocketsPoolMACE')

@dataclass
class BatchedModel:
    
    batch_size:int = field(init=False)    
    have_atoms:List[bool] = field(init=False)
    list_atoms:List[Atoms] = field(init=False)
    ready:bool = field(init=False)
    single_results:List[Dict[str,np.ndarray]] = field(init=False)
    exit:bool = field(init=False,default=False)
    
    def __post_init__(self):    
        self.have_atoms = [False]*self.batch_size
        self.ready = [False]*self.batch_size
        self.single_results = [None]*self.batch_size
        self.list_atoms = [None]*self.batch_size
        
    def master(self):
        """Master task running in parallel."""
        self.ready = False
        while not self.exit:
            
            while not all(self.have_atoms) and not self.exit:
                self.ready = False
                continue
            
            if self.exit:
                break
            
            results = self.mace_calculator.compute(self.list_atoms)
            
            # Do things
            
            if self.exit:
                break  
            
            self.ready = True         
        
        return
            

@dataclass
class SingleCalculator(Calculator):
    
    batched_model:BatchedModel
    index:int
    
    def calculate(self, atoms:Atoms=None, properties=None, system_changes=all_changes)->None:
        
        super().__init__()
        
        self.batched_model.list_atoms[self.index] = atoms
        self.batched_model.have_atoms[self.index] = True
        
        while not self.batched_model.ready:
            continue
        
        if self.batched_model.single_results is None \
            or self.batched_model.single_results[self.index] is None \
                or len(self.batched_model.single_results[self.index].keys()) == 0 :
                    raise ValueError("no results found")
        
        self.results = self.batched_model.single_results[self.index]
        self.batched_model.have_atoms[self.index] = False
        self.batched_model.ready[self.index] = False
        
    
@dataclass
class SocketsPoolMACE(BatchedModel):
    
    ports:List[int]
    unixsockets:List[str]
    socket_client:str
    log:str
    mace_calculator:MACEModel
    
    drivers:List[SocketClient] = field(init=False)
    calculators:List[SingleCalculator] = field(init=False)
    
    def __post_init__(self):
        self.batch_size = len(self.unixsockets)
        
        self.drivers = [None]*self.batch_size
        self.calculators = [None]*self.batch_size
        for n,(port,unixsocket) in enumerate(zip(self.ports,self.unixsockets)):
            if self.socket_client in ["extra","extras","eslib"]:
                from eslib.drivers.socketextras import SocketClientExtras
                self.drivers[n] = SocketClientExtras(port=port,unixsocket=unixsocket,log=None)
            else:
                self.drivers[n] = SocketClient(port=port,unixsocket=unixsocket,log=None)

            self.calculators[n] = SingleCalculator(self,n)
            self.calculators[n].implemented_properties = self.mace_calculator.implemented_properties
        
        super().__post_init__()
            
    @staticmethod
    def _run_single(task: Tuple[Atoms, SocketClient, Calculator, bool]):
        """Helper function for threading."""
        atoms, driver, calc, use_stress = task
        atoms.calc = calc
        driver.run(atoms, use_stress=use_stress)

    @staticmethod
    def _run_master_task(instance: "SocketsPoolMACE"):
        """Run the master task."""
        instance.master()

    def run(self, atoms: Atoms, use_stress: bool = False):
        """Run all drivers in parallel on the given atoms."""
        N = len(self.drivers)
        if N != len(self.calculators):
            raise ValueError("Number of drivers must match the number of calculators.")

        # Prepare arguments for threading
        tasks = [(atoms.copy(), self.drivers[n], self.calculators[n], use_stress) for n in range(N)]

        # Use ThreadPoolExecutor for threading
        with ThreadPoolExecutor(max_workers=N + 1) as executor:  # Add one for the master task
            
            # Submit all individual tasks to the executor
            futures = [executor.submit(self._run_single, task) for task in tasks]
            master = executor.submit(self._run_master_task, self)  # Submit the master task to the executor
            futures.append(master)
            # Wait for all futures to finish
            for future in futures:
                future.result()  # This will block until all tasks are complete
