from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import torch
from warnings import warn
from mace.tools import torch_geometric, torch_tools
from mace.cli.elia_configs import make_dataloader
from mace.modules.models.general import MACEBaseModel
from eslib.tools import add_info_array, reshape_info_array
from eslib.classes.models.eslibModel import eslibModel

#---------------------------------------#
@dataclass
class MACEModel(eslibModel,Calculator):
    """Class for loading and using MACE models."""

    #------------------#
    default_dtype: str
    device: str
    model_path: str
    batch_size: int
    charges_key: str
    dR: bool
    to_diff_props: List[str]
    rename_props: Dict[str, Any]
    implemented_properties:Dict[str, Any] = field(init=False)

    #------------------#
    def to(self, device: str, dtype: str=None) -> None:
        """
        Sets the device for the model.

        Args:
            device (str): The device to set for the model.

        Returns:
            None
        """
        self.device = torch_tools.init_device(device)
        self.network = self.network.to(self.device)  # Ensure model is on the specified device

        if dtype is not None:
            self.default_dtype = dtype
            torch_tools.set_default_dtype(self.default_dtype)

    #------------------#
    def __post_init__(self) -> None:
        """Initialize MACEModel object."""
        Calculator.__init__(self)
        torch_tools.set_default_dtype(self.default_dtype)
        self.device = torch_tools.init_device(self.device)
        self.network: MACEBaseModel = torch.load(f=self.model_path, map_location=self.device)
        self.network = self.network.to(self.device)  # Ensure model is on the specified device
        for param in self.network.parameters():
            param.requires_grad = False
        if self.dR:
            new_prop = get_d_prop_dR(self.to_diff_props, type(self.network), self.rename_props)
            self.network.implemented_properties = {**self.network.implemented_properties, **new_prop}
        self.network.set_prop()
        self.implemented_properties = self.network.implemented_properties

    def calculate(self, atoms:Atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        results:Dict[str,np.ndarray] = self.compute([atoms],raw=True)
        for k in results.keys():
            assert results[k].shape[0] == 1, f"Invalid shape for '{k}'. Expected (1,), got {results[k].shape}"
            results[k] = results[k][0]
        # [ a.shape for _,a in results.items() ]
        self.results = results

    #------------------#
    def compute(self, traj: List[Atoms], prefix: str = "", raw: bool = False, **kwargs) -> Any:
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
        torch_tools.set_default_dtype(self.default_dtype)

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
                if k not in self.network.implemented_properties:
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
        shapes = {prefix + k: self.network.implemented_properties[k] for k in outputs.keys()}

        # If `raw` is True, return the reshaped numpy arrays (no ASE Atoms objects)
        # Otherwise, add the properties to the ASE Atoms objects and return the trajectory
        if raw:
            # `raw` is True: return the reshaped numpy arrays (no ASE Atoms objects)
            return reshape_info_array(traj, new_outputs, shapes)[0]
        else:
            # `raw` is False: add the properties to the ASE Atoms objects and return the trajectory
            return add_info_array(traj, new_outputs, shapes)

    #------------------#
    def summary(self, string: str = "\t") -> None:
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
        max_key_length = max(len(key) for key in args.keys())

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
            dielectric=output[prop],
            positions=data["positions"],
        )
        name = "{:s}_dR".format(prop)
        name = name if name not in model.rename_props else model.rename_props[name]

        output[name] = array
        if output[prop].shape[1] == 3:
            for n, i in enumerate(["x", "y", "z"]):
                output["{:s}{:s}".format(name, i)] = array[:, n, :]

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
