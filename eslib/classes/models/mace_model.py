from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar
from warnings import warn

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from mace.cli.elia_configs import make_dataloader
from mace.modules.models.general import MACEBaseModel
from mace.tools import torch_geometric, torch_tools

from eslib.classes.models import eslibModel
from eslib.tools import add_info_array, reshape_info_array

T = TypeVar('T', bound='MACEModel')

#---------------------------------------#
@dataclass
class MACEModel(eslibModel):
    """Class for loading and using MACE models as an `ase.Calculator` or in you custom scripts.

    An instance of this class can be saved to a `*.pickle` file but for its correct functioning you need to provide the correct filepath of the MACE model (`*.pt`), i.e. `self.model_path`.

    The class will load from `self.model_path` the network every time it is itself loaded (within `MACEModel.from_file` or `MACEModel.from_pickle` ).

    For this reason, be sure that `self.model_path` is accessible.
    
    If you change the location of the `*.pickle` file:
     - move the `*.pt` MACE model file if `self.model_path` is a relative path;
     - leave the `*.pt` MACE model file  where it is if `self.model_path` is an absolute path;
     
    Be sure that `self.model_path` is a relative path if you are planning to move the `*.pickle` file to another machine.
    """

    #------------------------------------#
    # Attributes
    #------------------------------------#

    #------------------#
    # Attributes initialized by `dataclass`
    default_dtype: str                                          # `torch` default data type used for computation for computation (e.g., 'float32', 'float64').
    device       : torch.device                                 # `torch` device used for computation (e.g., 'cpu', 'cuda').
    model_path   : str                                          #  path to the MACE (torch.nn.Module) model file.
    batch_size   : int                                          # `batch_size` is the size of the batch of structures used when computing properties.
    charges_key  : str                                          # `charges_key` is the key of the charges in the MACE model.
    dR           : bool                                         #  a boolean that indicates whether to compute spatial derivatives of properties.
    to_diff_props: List[str]                                    #  a list of properties to compute the spatial derivatives of.
    rename_props : Dict[str, Any]                               #  a dictionary that maps the names of properties to new names.

    #------------------#
    # Attributes initialized in `__post_init__`
    implemented_properties: Dict[str, Any] = field(init=False)  #  a dictionary that maps property names to their corresponding functions in the MACE model.
    network: torch.nn.Module = field(init=False)                # `MACE model.

    #------------------------------------#
    # Methods for `dataclass` initialization
    #------------------------------------#

    #------------------#
    def __post_init__(self:T) -> None:
        """Initialize MACEModel object."""
        Calculator.__init__(self)
        self._set_defaults()
        self._load_model()
        self._set_properties()

    #------------------------------------#
    # Methods for `pickle` read/write
    #
    # The `eslibModel.to_pickle` and `eslibModel.from_pickle` methods are overwritten 
    # to properly save and load the `self.network`. 
    #------------------------------------#

    #------------------#
    def __getstate__(self):
        """Get the state for pickling, excluding the network."""
        state = self.__dict__.copy()
        # Remove the model from the state
        state['network'] = None
        return state

    def __setstate__(self:T, state):
        """Set the state from unpickling, initializing the network to None."""
        self.__dict__.update(state)
        # Initialize model to None; it will be set later
        self.network = None
        return

    def __post__from_pickle__(self:T) -> None:
        """Load the network from a `*.pt` file and save it to `self.network`."""
        self._set_defaults()
        self._load_model()
        # The following line should not be necessary and indeed it will be kept commented
        # obj._set_properties()
        return 
    
    #------------------------------------#
    # Methods for initialization
    # used both in `__post_init__` and `from_pickle`
    #------------------------------------#

    #------------------#
    def _set_defaults(self:T) -> None:
        """Set default values for the model."""
        torch_tools.set_default_dtype(self.default_dtype)
        self.device = torch_tools.init_device(self.device)
        # if torch.cuda.is_available():
        #     self.device = torch_tools.init_device(self.device)
        # else:
        #     self.device = torch_tools.init_device("cpu")
        return

    #------------------#
    def _load_model(self:T) -> None:
        """Load the torch.nn.Module from file."""
        try:
            self.network: MACEBaseModel = torch.load(f=self.model_path, map_location=self.device)
        except Exception as e:
            raise ValueError(f"Could not load model from {self.model_path}.") from e
        self.network.to(self.device)  # Ensure model is on the specified device
        for param in self.network.parameters():
            param.requires_grad = False
        return
    
    #------------------#
    def _set_properties(self:T) -> None:
        """Set the `implemented_properties` of the model."""
        if self.dR:
            new_prop = get_d_prop_dR(self.to_diff_props, type(self.network), self.rename_props)
            self.network.implemented_properties = {**self.network.implemented_properties, **new_prop}
        self.network.set_prop()
        self.implemented_properties = self.network.implemented_properties


    #------------------#
    def to(self:T, device: str, dtype: str=None) -> None:
        """
        Sets the device for the model.

        Args:
            device (str): The device to set for the model.

        Returns:
            None
        """
        self.device = torch_tools.init_device(device)
        if self.device == "cuda" and torch.cuda.is_available():
            device = torch.device(self.device)
        else:
            device = torch.device("cpu")
        self.network = self.network.to(device)  # Ensure model is on the specified device

        if dtype is not None:
            self.default_dtype = dtype
            torch_tools.set_default_dtype(self.default_dtype)

    #------------------------------------#
    # Overloading `ase.Calculator.calculate` method
    #------------------------------------#

    #------------------#
    def calculate(self:T, atoms:Atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate the results for the given atoms.

        This method is necessary when using `ase`.
        
        Args:
            atoms (Atoms, optional): The atoms for which to calculate the results. Defaults to None.
            properties (list, optional): The properties to calculate. Defaults to None.
            system_changes (str, optional): The changes in the system. Defaults to all_changes.
        
        Returns:
            None
        
        Raises:
            AssertionError: If the shape of a result is not (1,).
        """
        super().calculate(atoms, properties, system_changes)
        results:Dict[str,np.ndarray] = self.compute([atoms],raw=True)
        for k in results.keys():
            assert results[k].shape[0] == 1, f"Invalid shape for '{k}'. Expected (1,), got {results[k].shape}"
            results[k] = results[k][0]
        # [ a.shape for _,a in results.items() ]
        self.results = results

    #------------------------------------#
    # Method to evaluate the model fpr different structures
    #------------------------------------#

    #------------------#
    def compute(self:T, traj: List[Atoms], prefix: str = "", raw: bool = False, **kwargs) -> Any:
        """
        Compute properties for a trajectory.

        Args:
            traj (List[Atoms]): List of ASE Atoms objects representing the trajectory.
            prefix (str, optional): Prefix to add to property names. Defaults to "".
            raw (bool, optional): If True, return raw numpy arrays. Defaults to False.
            **kwargs: Additional arguments to pass to compute function.

        Returns:
            Any: Computed properties.
        """
        # If only one structure is provided, convert it to a list
        # Nconfig = len(traj)

        # Set default dtype
        self._set_defaults()
        self.network.to(self.device)
        
        actual_device = str(next(self.network.parameters()).device).lower()
        if torch.cuda.is_available() and "cuda" not in actual_device:
            warn(f"CUDA is available but used device is {actual_device} (`self.device` is {self.device}).\n\tLet's move to CUDA.")
            self.device = torch_tools.init_device("cuda")
            self.network.to(self.device)
            actual_device = str(next(self.network.parameters()).device).lower()
            if "cuda" not in actual_device:
                warn(f"Something weird is going on.\n\tUsed device is {actual_device} and `self.device` is {self.device}.")
            # self.network = self.network.to("cuda")

        # Create data loader
        data_loader: torch_geometric.dataloader.DataLoader = make_dataloader(
            atoms_list=traj,
            model=self.network,
            batch_size=min(self.batch_size, len(traj)),  # Use the actual number of structures
            charges_key=self.charges_key,
        )

        # Initialize lists to store warnings and outputs
        warnings = []
        outputs: Dict[str, np.ndarray] = dict()

        # Iterate over batches
        for batch in data_loader:
            batch: torch_geometric.batch.Batch = batch.to(self.device)
            data: Dict[str, torch.Tensor] = batch.to_dict()

            # Compute properties for the current batch
            results: Dict[str, torch.Tensor] = self.network(data, compute_stress=False, **kwargs)

            # If derivatives are requested, add them to the results
            if self.dR:
                results = add_derivatives(self, results, data)

            # Process the results
            for k in results.keys():
                if k not in self.implemented_properties:
                    # If a property is not implemented, add a warning
                    if k not in warnings:
                        warnings.append(k)
                        warn(f"{k} not in `implemented_properties`")
                else:
                    # Convert the tensor to numpy array and append to the outputs
                    if results[k] is not None:
                        data = torch_tools.to_numpy(results[k])
                        if k not in outputs:
                            outputs[k] = data
                        else:
                            outputs[k] = np.append(outputs[k], data)
                    else:
                        data = np.nan

        # Prepare the outputs for return
        new_outputs = {prefix + k: outputs[k] for k in outputs.keys()}
        shapes = {prefix + k: self.implemented_properties[k] for k in outputs.keys()}

        # If `raw` is True, return the reshaped numpy arrays (no ASE Atoms objects)
        # Otherwise, add the properties to the ASE Atoms objects and return the trajectory
        if raw:
            # `raw` is True: return the reshaped numpy arrays (no ASE Atoms objects)
            return reshape_info_array(traj, new_outputs, shapes)[0]
        else:
            # `raw` is False: add the properties to the ASE Atoms objects and return the trajectory
            return add_info_array(traj, new_outputs, shapes)

    #------------------------------------#
    # Overloading `eslibModel.summary` method
    #------------------------------------#

    #------------------#
    def summary(self:T, string: str = "\t") -> None:
        """Print summary of the model."""
        super().summary(string=string)
        args = {
            "path": self.model_path,
            "device": self.device,
            "batch size": self.batch_size,
            "charges key": self.charges_key,
            "dtype": self.default_dtype,
            "derivatives": self.dR,
            "properties": list(self.implemented_properties.keys())
        }

        # Determine the length of the longest key
        max_key_length = max(len(key) for key in args.keys())+1

        for k, v in args.items():
            # Align the output based on the length of the longest key
            print("\t{:s}{:<{width}}: {}".format(string, k, v, width=max_key_length))
        # super().summary(string=string)
        pass

#---------------------------------------#
def compute_dielectric_gradients(dielectric: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Compute the spatial derivatives of dielectric tensor.

    Args:
        dielectric (torch.Tensor): Dielectric tensor.
        positions (torch.Tensor): Atom positions.

    Returns:
        torch.Tensor: Spatial derivatives of dielectric tensor.
    """
    # dielectric = dielectric[:,0:2]
    d_dielectric_dr = []
    for i in range(dielectric.shape[-1]):
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones((dielectric.shape[0], 1)).to(dielectric.device)
        ]
        gradient = torch.autograd.grad(
            outputs=[dielectric[:, i].unsqueeze(-1)],
            inputs=[positions],
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        d_dielectric_dr.append(gradient)
    d_dielectric_dr = torch.stack(d_dielectric_dr, dim=1)
    if gradient is None:
        return torch.zeros((positions.shape[0], dielectric.shape[-1], 3))
    return d_dielectric_dr

#---------------------------------------#
def add_derivatives(model: MACEModel, output: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Add spatial derivatives to the output dictionary.

    Args:
        model (MACEModel): MACEModel object.
        output (Dict[str, torch.Tensor]): Output dictionary.
        data (Dict[str, torch.Tensor]): Data dictionary.

    Returns:
        Dict[str, torch.Tensor]: Output dictionary with spatial derivatives.
    """
    for prop in model.to_diff_props:
        array = compute_dielectric_gradients(
            dielectric=output[prop], # [:,0][np.newaxis,:] --> should give only BECx
            positions=data["positions"],
        )
        name = "{:s}_dR".format(prop)
        name = name if name not in model.rename_props else model.rename_props[name]

        
        if output[prop].shape[1] == 3:
            output[name] = array.permute(0, 2, 1)
            for n, i in enumerate(["x", "y", "z"]):
                output["{:s}{:s}".format(name, i)] = array[:, n, :]
            # del output[name] # This could be deleted if working with i-PI
        else:
            output[name] = array

    return output

#---------------------------------------#
def add_natoms(info: Any) -> Any:
    """Add natoms information to the property information.

    Args:
        info (Any): Property information.

    Returns:
        Any: Property information with natoms added.
    """
    shape = info[1]
    if isinstance(shape, int):
        return (info[0], ("natoms", 3, shape))
    else:
        return (info[0], ("natoms", 3,) + shape)

#---------------------------------------#
def get_d_prop_dR(props: List[str], basecls: MACEBaseModel, rename: Dict[str, str]) -> Dict[str, Any]:
    """Get properties and their derivatives.

    Args:
        props (List[str]): List of properties.
        basecls (MACEBaseModel): Base class.
        rename (Dict[str, str]): Dictionary to rename properties.

    Returns:
        Dict[str, Any]: Dictionary of properties and their derivatives.
    """
    der = {}
    ip = basecls.implemented_properties
    for prop in props:
        if prop not in ip:
            raise ValueError("'{:s}' is not an implemented property of the parent class {}.".format(prop, basecls))
        name = "{:s}_dR".format(prop)
        name = name if name not in rename else rename[name]
        der[name] = add_natoms(ip[prop])
        if ip[prop][1] == 3:
            for n, i in enumerate(["x", "y", "z"]):
                der["{:s}{:s}".format(name, i)] = (ip[prop][0], ("natoms", 3))
    return der
