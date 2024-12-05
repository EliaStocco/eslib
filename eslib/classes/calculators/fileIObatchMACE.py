import os
import numpy as np
from typing import Dict
from ase.io import read
from eslib.classes.models.mace_model import MACEModel
from eslib.io_tools import save2json

def check_exit():
    if os.path.exists("EXIT"):
        exit(0)

class FileIOBatchedMACE:
    
    def __init__(self,folders:list[str],model:str): 
        self.folders = folders
        self.model = MACEModel.from_file(model)
        # self.ready = [False]*len(self.folders)
        
    def run(self):
        
        N = len(self.folders)
        atoms = [None]*N
        ready = [False]*N
        single_results = [None]*N
        ifiles = [ f"{folder}/input.extxyz" for folder in self.folders ]
        ofiles = [ f"{folder}/output.json" for folder in self.folders ]
        
        while True: # iterations
            
            check_exit()
            
            for n in range(N):
                ready[n] = False
            
            while not all(ready):
                
                for n,(folder,ifile) in enumerate(zip(self.folders,ifiles)):
                    if os.path.exists(folder):
                        if os.path.exists(ifile):
                            ready[n] = True
                        
                check_exit()
                   
            for n,file in zip(ifiles):
                atoms[n] =  read(file,format="extxyz",index=0)
                os.remove(file)
                
            check_exit()            
            results: Dict[str, np.ndarray] = self.model.compute(atoms, raw=True)
            
            for n, _ in enumerate(single_results):
                single_results[n] = {}
                for key in results.keys():
                    value: np.ndarray = np.take(results[key], axis=0, indices=n)
                    single_results[n][key] = value if value.size > 1 else float(value)
                    
            for ofile,res in zip(ofiles,single_results):
                save2json(ofile,res)
            
             