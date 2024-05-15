from ase import Atoms
from typing import List, Dict
from dataclasses import dataclass, field
import numpy as np
from mace.tools import torch_geometric, torch_tools
from mace.cli.elia_configs import make_dataloader
from mace.modules.models import get_model
from eslib.classes.io import pickleIO
from warnings import warn
from eslib.tools import add_info_array, reshape_info_array
import torch
import json
import os

@dataclass
class MACEModel(pickleIO):
    default_dtype: str
    device: str
    model: str
    model_type: str
    batch_size: int
    charges_key: str
    _initialized:bool = field(init=False)
    _attributes_dict: Dict[str, str] = field(init=False)
    _folder: str = field(init=False)

    def to_pickle(self,*argv,**kwargs):
        # self._initialized = False
        # if not reinit:
        # try:
        #     super().to_pickle(*argv,**kwargs)
        # except:
        self._initialized = False
        super().to_pickle(*argv,**kwargs)
        # else:
        #     self._initialized = False
        #     super().to_pickle(*argv,**kwargs)

    def __post_init__(self):
        self._attributes_dict = self.to_dict()
        self._initialized = False
        self._folder = os.getcwd()
        self.initialize()

    def to_dict(self) -> Dict[str, str]:
        # Get all fields of the dataclass
        fields = self.__dataclass_fields__

        # Store attribute values in a dictionary for initialized fields only
        attributes_dict = {field_name: getattr(self, field_name) 
                           for field_name in fields 
                           if field_name in self.__dict__}
        
        return attributes_dict
    
    def to_json(self,file):
        with open(file,"w") as ffile:
            json.dump(obj=self._attributes_dict,fp=ffile,indent=4)

    def initialize(self):     
        if not self._initialized:   
            torch_tools.set_default_dtype(self.default_dtype)
            self.device = torch_tools.init_device([self.device])
            try :
                model_path = self.model
                self.network = get_model(model_path=model_path,
                                    model_type=self.model_type,
                                    device=self.device)
            except:
                model_path = "{:s}/{:s}".format(self._folder,self.model)
                self.network = get_model(model_path=model_path,
                                    model_type=self.model_type,
                                    device=self.device)
            self.network = self.network.to(self.device)  # shouldn't be necessary but seems to help with CUDA problems
            for param in self.network.parameters():
                param.requires_grad = False
            self._initialized = True

    def compute(self,traj:List[Atoms],prefix:str="",raw:bool=False,**argv):
        self.initialize()
        data_loader:torch_geometric.dataloader.DataLoader = \
            make_dataloader(atoms_list=traj,
                            model=self.network,
                            batch_size=self.batch_size,
                            charges_key=self.charges_key)
        warnings = []
        outputs:Dict[str,np.ndarray] = dict()
        for batch in data_loader:
            batch = batch.to(self.device)
            results:Dict[str,torch.Tensor] = self.network(batch.to_dict(), compute_stress=False)

            for k in results.keys():
                if k not in self.network.implemented_properties:
                    if k not in warnings:
                        warnings.append(k)
                        warn("{:s} not in `implemented_properties`".format(k))
                else:
                    data = torch_tools.to_numpy(results[k])
                    if k not in outputs:
                        outputs[k] = data
                    else:
                        outputs[k] = np.append(outputs[k],data)

        new_outputs = dict()
        shapes = dict()
        for k in outputs.keys():
            new_outputs[prefix+k] = outputs[k]
            shapes[prefix+k] = self.network.implemented_properties[k]

        if raw:
            return reshape_info_array(traj,new_outputs,shapes)[0]
        else:
            return add_info_array(traj,new_outputs,shapes)
    
    def summary(self, string="\t"):
        args = {
            "path": self.model,
            "type": self.model_type,
            "device": self.device,            
            "batch size": self.batch_size,
            "charges key": self.charges_key,
            "dtype" : self.default_dtype,
            "properties": list(self.network.implemented_properties.keys())
        }

        # Determine the length of the longest key
        max_key_length = max(len(key) for key in args.keys())

        for k, v in args.items():
            # Align the output based on the length of the longest key
            print("{:s}{:<{width}}: {}".format(string, k, v, width=max_key_length))

        
