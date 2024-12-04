from ase import Atoms
from eslib.classes.models.mace_model import MACEModel
from ase.calculators.calculator import Calculator, all_changes
import threading  # Replace multiprocess with threading
from typing import List, TypeVar, Dict, Tuple
from ase.calculators.socketio import SocketClient
from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

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
            
            results:Dict[str,np.ndarray] = self.mace_calculator.compute(self.list_atoms,raw=True)
            for n,_ in enumerate(self.single_results):
                self.single_results[n] = {}
                for key in results.keys():
                    value:np.ndarray = np.take(results[key],axis=0,indices=n)
                    self.single_results[n][key] = value if value.size > 1 else float(value)
            
            if self.exit:
                break  
            
            self.ready = True         
        
        return
            

@dataclass
class SingleCalculator(Calculator):
    
    batched_model:BatchedModel
    index:int
    
    def __post_init__(self):
        super().__init__()
    
    def calculate(self, atoms:Atoms=None, properties=None, system_changes=all_changes)->None:
        
        super().__init__()
        
        self.batched_model.list_atoms[self.index] = atoms
        self.batched_model.have_atoms[self.index] = True
        
        while not self.batched_model.ready:
            self.results = None
            continue
        
        if self.batched_model.single_results is None \
            or self.batched_model.single_results[self.index] is None \
                or len(self.batched_model.single_results[self.index].keys()) == 0 :
                    raise ValueError("no results found")
        
        self.results = deepcopy(self.batched_model.single_results[self.index])
        self.batched_model.have_atoms[self.index] = False
        # self.batched_model.ready = False
        
    
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

    # def run(self, atoms: Atoms, use_stress: bool = False):
    #     """Run all drivers in parallel on the given atoms."""
    #     N = len(self.drivers)
    #     if N != len(self.calculators):
    #         raise ValueError("Number of drivers must match the number of calculators.")

    #     # Prepare arguments for threading
    #     tasks = [(atoms.copy(), self.drivers[n], self.calculators[n], use_stress) for n in range(N)]

    #     # Use ThreadPoolExecutor for threading
    #     with ThreadPoolExecutor(max_workers=N + 1) as executor:  # Add one for the master task
            
    #         # Submit all individual tasks to the executor
    #         futures = [executor.submit(self._run_single, task) for task in tasks]
    #         # master = executor.submit(self._run_master_task, self)  # Submit the master task to the executor
    #         # futures.append(master)
    #         # Wait for all futures to finish
    #         for future in futures:
    #             future.result()  # This will block until all tasks are complete
    
    

    # def run(self, atoms: Atoms, use_stress: bool = False):
    #     """Run all drivers in parallel on the given atoms using threading."""
        
    #     N = len(self.drivers)
    #     if N != len(self.calculators):
    #         raise ValueError("Number of drivers must match the number of calculators.")

    #     # Prepare tasks
    #     tasks = [(atoms.copy(), self.drivers[n], self.calculators[n], use_stress) for n in range(N)]

    #     # Create a thread for the master task to run in parallel
    #     master_thread = threading.Thread(target=self.master)
    #     master_thread.start()  # Start the master thread
        
    #     # Create and start threads for each task
    #     threads:List[threading.Thread] = []
    #     for task in tasks:
    #         thread = threading.Thread(target=self._run_single, args=(task,))
    #         thread.daemon = False
    #         threads.append(thread)
    #         thread.start()  # Start the thread
        
    #     # Wait for all threads (tasks) to finish
    #     for thread in threads:
    #         thread.join()  # Block until the thread finishes

    #     # Wait for the master thread to finish
    #     master_thread.join()  # Block until the master thread finishes
    
    def run(self, atoms: Atoms, use_stress: bool = False):
        """Run all drivers in parallel on the given atoms using ThreadPoolExecutor."""
        
        N = len(self.drivers)
        if N != len(self.calculators):
            raise ValueError("Number of drivers must match the number of calculators.")

        # Prepare tasks
        tasks = [(atoms.copy(), self.drivers[n], self.calculators[n], use_stress) for n in range(N)]
        
        # Create a thread pool executor
        with ThreadPoolExecutor(max_workers=N + 1) as executor:  # Add one for the master task
            # Submit the master task
            master_future = executor.submit(self.master)
            
            # Submit the tasks for each driver
            task_futures = [executor.submit(self._run_single, task) for task in tasks]
            
            # Wait for all tasks to complete
            for future in as_completed(task_futures + [master_future]):
                try:
                    future.result()  # This will raise any exceptions if they occurred
                except Exception as e:
                    print(f"Error in thread execution: {e}")