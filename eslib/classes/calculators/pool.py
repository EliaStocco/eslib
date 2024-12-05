import threading
import time
import numpy as np
from typing import List, Dict, Tuple, TypeVar
from copy import deepcopy
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from eslib.classes.models.mace_model import MACEModel
from ase.calculators.socketio import SocketClient
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

WAITING_TIME = 0.001

T = TypeVar('T', bound='SocketsPoolMACE')

@dataclass
class BatchedModel:
    batch_size: int
    have_atoms: List[bool] = field(init=False)
    list_atoms: List[Atoms] = field(init=False)
    ready: bool = field(init=False)
    single_results: List[Dict[str, np.ndarray]] = field(init=False)
    exit: bool = field(init=False, default=False)
    mace_calculator: MACEModel
    
    def __post_init__(self):
        self.have_atoms = [False] * self.batch_size
        self.ready = False
        self.single_results = [None] * self.batch_size
        self.list_atoms = [None] * self.batch_size

    def master(self):
        """Master task running in parallel (infinite cycle)."""
        self.ready = False
        print("\tEntering master", flush=True)
        while not self.exit:
            print("\tWaiting for all atoms", flush=True)
            while not all(self.have_atoms) and not self.exit:
                time.sleep(WAITING_TIME)
            
            if self.exit:
                break
            
            print("\tCalling network", flush=True)
            results: Dict[str, np.ndarray] = self.mace_calculator.compute(self.list_atoms, raw=True)
            print("\tDispatching results", flush=True)
            
            for n, _ in enumerate(self.single_results):
                self.single_results[n] = {}
                for key in results.keys():
                    value: np.ndarray = np.take(results[key], axis=0, indices=n)
                    self.single_results[n][key] = value if value.size > 1 else float(value)
            
            # Reset the have_atoms list after dispatching the results
            self.have_atoms = [False] * self.batch_size
            
            if self.exit:
                break
            
            # Set the ready flag to True, allowing the workers to proceed
            self.ready = True
            time.sleep(WAITING_TIME)
        
        print("\tExiting master", flush=True)

@dataclass
class SingleCalculator(Calculator):
    batched_model: List[BatchedModel]
    index: int

    def __post_init__(self):
        super().__init__()

    def calculate(self, atoms: Atoms = None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        
        self.batched_model[0].list_atoms[self.index] = atoms
        
        # Wait until the batched model is ready
        while not self.batched_model[0].ready:
            self.batched_model[0].have_atoms[self.index] = True
            time.sleep(WAITING_TIME)
        
        self.results = self.batched_model[0].single_results[self.index]
        if self.results is None or len(self.results.keys()) == 0:
            raise ValueError("No results found")
        
        # Reset have_atoms flag after fetching results
        self.batched_model[0].have_atoms[self.index] = False

@dataclass
class SocketsPoolMACE:
    ports: List[int]
    unixsockets: List[str]
    socket_client: str
    log: str
    mace_calculator: MACEModel
    batched_model: BatchedModel = field(init=False)
    drivers: List[SocketClient] = field(init=False)
    calculators: List[SingleCalculator] = field(init=False)
    
    def __post_init__(self):
        self.batch_size = len(self.unixsockets)
        self.batched_model = BatchedModel(self.batch_size, self.mace_calculator)
        
        self.drivers = [None] * self.batch_size
        self.calculators = [None] * self.batch_size
        
        for n, (port, unixsocket) in enumerate(zip(self.ports, self.unixsockets)):
            self.drivers[n] = SocketClient(port=port, unixsocket=unixsocket, log=None,timeout=10)
            self.calculators[n] = SingleCalculator(batched_model=[self.batched_model], index=n)
            self.calculators[n].implemented_properties = self.mace_calculator.implemented_properties

    @staticmethod
    def _run_single(task: Tuple[Atoms, SocketClient, SingleCalculator, bool]) -> None:
        atoms, driver, calc, use_stress = task
        print(f"\n\tRunning thread {calc.index}", flush=True)
        atoms.calc = calc
        driver.run(atoms,use_stress=use_stress)

    def run(self, atoms: Atoms, use_stress: bool = False):
        """Run all drivers in parallel on the given atoms using threading."""
        N = len(self.drivers)
        if N != len(self.calculators):
            raise ValueError("Number of drivers must match the number of calculators.")

        # Prepare tasks
        tasks = [(atoms.copy(), self.drivers[n], self.calculators[n], use_stress) for n in range(N)]

        # Start the master thread (infinite cycle)
        master_thread = threading.Thread(target=self.batched_model.master)
        master_thread.start()

        # Create and start threads for each task
        threads: List[threading.Thread] = [master_thread]
        for task in tasks:
            thread = threading.Thread(target=self._run_single, args=(task,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads (tasks) to finish
        for thread in threads:
            thread.join()

