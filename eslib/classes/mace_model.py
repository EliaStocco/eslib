from ase import Atoms
from typing import List, Dict, Any, Optional
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
from mace.modules.models.general import MACEBaseModel

@dataclass
class MACEModel(pickleIO):
    default_dtype: str
    device: str
    model_path: str
    # model_type: str
    batch_size: int
    charges_key: str
    dR:bool
    to_diff_props:List
    rename_props:Dict[str,Any]

    # to_diff_props=["dipole"]
    # rename_dR={"dipole_dR":"BEC"}

    # _initialized:bool = field(init=False)
    # _attributes_dict: Dict[str, str] = field(init=False)
    # _folder: str = field(init=False)

    # def to_pickle(self,*argv,**kwargs):
    #     # self._initialized = False
    #     # if not reinit:
    #     # try:
    #     #     super().to_pickle(*argv,**kwargs)
    #     # except:
    #     self._initialized = False
    #     super().to_pickle(*argv,**kwargs)
    #     # else:
    #     #     self._initialized = False
    #     #     super().to_pickle(*argv,**kwargs)

    # def __post_init__(self):
    #     # self._attributes_dict = self.to_dict()
    #     # self._initialized = False
    #     # self._folder = os.getcwd()
    #     self.initialize()

    # def to_dict(self) -> Dict[str, str]:
    #     # Get all fields of the dataclass
    #     fields = self.__dataclass_fields__

    #     # Store attribute values in a dictionary for initialized fields only
    #     attributes_dict = {field_name: getattr(self, field_name) 
    #                        for field_name in fields 
    #                        if field_name in self.__dict__}
        
    #     return attributes_dict
    
    # def to_json(self,file):
    #     with open(file,"w") as ffile:
    #         json.dump(obj=self._attributes_dict,fp=ffile,indent=4)

    def __post_init__(self):     
        # if not self._initialized:   
        torch_tools.set_default_dtype(self.default_dtype)
        self.device = torch_tools.init_device([self.device])
        # try :
        # model_path = self.model_path
        self.network :MACEBaseModel = torch.load(f=self.model_path, map_location=self.device)
        # self.network = get_model(model_path=model_path,
        #                     model_type=self.model_type,
        #                     device=self.device)
        # except:
        #     model_path = "{:s}/{:s}".format(self._folder,self.model)
        #     self.network = get_model(model_path=model_path,
        #                         model_type=self.model_type,
        #                         device=self.device)
        self.network = self.network.to(self.device)  # shouldn't be necessary but seems to help with CUDA problems
        for param in self.network.parameters():
            param.requires_grad = False
        # self._initialized = True
        if self.dR:
            new_prop = get_d_prop_dR(self.to_diff_props,type(self.network),self.rename_props)
            self.network.implemented_properties = {**self.network.implemented_properties,**new_prop}

        self.network.set_prop()

    def compute(self,traj:List[Atoms],prefix:str="",raw:bool=False,**argv):
        # self.initialize()
        torch_tools.set_default_dtype(self.default_dtype)
        data_loader:torch_geometric.dataloader.DataLoader = \
            make_dataloader(atoms_list=traj,
                            model=self.network,
                            batch_size=self.batch_size,
                            charges_key=self.charges_key)
        warnings = []
        outputs:Dict[str,np.ndarray] = dict()
        for batch in data_loader:
            batch:torch_geometric.batch.Batch = batch.to(self.device)
            data:Dict[str,torch.Tensor] = batch.to_dict()
            results:Dict[str,torch.Tensor] = self.network(data, compute_stress=False)
            if self.dR: 
                results = add_derivatives(self,results,data)
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
            "path": self.model_path,
            # "type": self.model_type,
            "device": self.device,            
            "batch size": self.batch_size,
            "charges key": self.charges_key,
            "dtype" : self.default_dtype,
            "derivatives" : self.dR,
            "properties": list(self.network.implemented_properties.keys())
        }

        # Determine the length of the longest key
        max_key_length = max(len(key) for key in args.keys())

        for k, v in args.items():
            # Align the output based on the length of the longest key
            print("{:s}{:<{width}}: {}".format(string, k, v, width=max_key_length))

def compute_dielectric_gradients(
    dielectric: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    d_dielectric_dr = []
    for i in range(dielectric.shape[-1]):
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones((dielectric.shape[0], 1)).to(dielectric.device)
        ]
        gradient = torch.autograd.grad(
            outputs=[dielectric[:, i].unsqueeze(-1)],  # [n_graphs, 3], [n_graphs, 9]
            inputs=[positions],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=True,  # Make sure the graph is not destroyed during training
            create_graph=True,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )
        d_dielectric_dr.append(gradient[0])
    d_dielectric_dr = torch.stack(d_dielectric_dr, dim=1)
    if gradient[0] is None:
        return torch.zeros((positions.shape[0], dielectric.shape[-1], 3))
    return d_dielectric_dr

def add_derivatives(model:MACEModel,output:Dict[str,torch.Tensor],data:Dict[str,torch.Tensor])->Dict[str,torch.Tensor]:
    for prop in model.to_diff_props:
        array = compute_dielectric_gradients(
            dielectric=output[prop],
            positions=data["positions"],
        )
        name = "{:s}_dR".format(prop)
        name = name if name not in model.rename_props else model.rename_props[name]

        # # Determine the number of axes: (atom, positions coord, output coord)
        # num_axes = tmp.dim()

        # # Permute the axis order
        # permuted_order = list(range(num_axes))
        # permuted_order = [permuted_order[-1]] + permuted_order[:-1]  # Move last axis to the front
        # permuted_tensor = tmp.permute(permuted_order)

        # (output coord, atom, positions coord)
        output[name] = array
        if output[prop].shape[1] == 3:
            for n,i in enumerate(["x","y","z"]):
                output["{:s}{:s}".format(name,i)] = array[:,:,n]

    return output

def add_natoms(info):
    shape = info[1]
    if type(shape) == int:
        return (info[0], ("natoms",3,shape))
    else:
        return (info[0], ("natoms",3,) + shape)

def get_d_prop_dR(props:List,basecls:MACEBaseModel,rename:Dict[str,str])->Dict[str,Any]:
    # der = [None]*len(props)
    der = {}
    ip = basecls.implemented_properties
    for prop in props:
        if prop not in ip:
            raise ValueError("'{:s}' is not an implemented property of the parent class {}.".format(prop,basecls))
        name ="{:s}_dR".format(prop)
        name = name if name not in rename else rename[name]
        der[name] = add_natoms(ip[prop])
        if ip[prop][1] == 3:
            for n,i in enumerate(["x","y","z"]):
                der["{:s}{:s}".format(name,i)] = (ip[prop][0], ("natoms",3))
    return der