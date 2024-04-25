from glob import glob
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils

from mace.calculators.mace import get_model_dtype

from mace.modules.models import get_model, energy_models, dipole_models
from ase import Atoms
from mace.calculators import MACECalculator
from mace.modules.models import BaseEnergyClass

from dataclasswa import dataclass

default_properties = BaseEnergyClass.implemented_properties

@dataclass
class LennerdJonesWall(Calculator):

    

class LennerdJonesWalls(Calculator):
    """
    """

    def __init__(
        self,
        instructions_path: Dict[str, str],
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model_type = model_type

        #--------------------------------------------#

        if "model_path" in kwargs:
            print("model_path argument deprecated, use model_paths")
            model_paths = kwargs["model_path"]

        if isinstance(model_paths, str):
            # Find all models that satisfy the wildcard (e.g. mace_model_*.pt)
            model_paths_glob = glob(model_paths)
            if len(model_paths_glob) == 0:
                raise ValueError(f"Couldn't find MACE model files: {model_paths}")
            model_paths = model_paths_glob
        elif isinstance(model_paths, Path):
            model_paths = [model_paths]
        if len(model_paths) == 0:
            raise ValueError("No mace file names supplied")
        self.num_models = len(model_paths)

        #--------------------------------------------#

        self.models = [ get_model(model_path,model_type,device) for model_path in model_paths]

        if hasattr(self.models[0], 'implemented_properties'):
            self.internal_implemented_properties = self.models[0].implemented_properties
        else:
            # Handle the case when the attribute doesn't exist
            # For example, you can set it to an empty set
            self.internal_implemented_properties = dict()


        if len(model_paths) > 1:
            print(f"Running committee mace with {len(model_paths)} models")

            if self.model_type in energy_models:
                self.internal_implemented_properties.add(["energies", "energy_var", "forces_comm", "stress_var"])
            
            for prop in self.internal_implemented_properties:
                if prop in default_properties: continue
                self.internal_implemented_properties.add([f'{prop}_var'])

        self.implemented_properties = { **self.internal_implemented_properties, **default_properties}

        #--------------------------------------------#

        for model in self.models:
            model.to(device)  # shouldn't be necessary but seems to help with GPU
        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        assert np.all(
            r_maxs == r_maxs[0]
        ), "committee r_max are not all the same {' '.join(r_maxs)}"
        self.r_max = float(r_maxs[0])

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )
        self.charges_key = charges_key
        model_dtype = get_model_dtype(self.models[0])
        if default_dtype == "":
            print(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            print(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.models = [model.double() for model in self.models]
            elif default_dtype == "float32":
                self.models = [model.float() for model in self.models]
        torch_tools.set_default_dtype(default_dtype)
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        pass
        
    # pylint: disable=dangerous-default-value
    def calculate(self, atoms:Atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        # predict + extract data
        outputs = {}
        for prop in self.implemented_properties:
            outputs[prop] = []

        if self.model_type in energy_models:
            batch = next(iter(data_loader)).to(self.device)
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])
            outputs["node_energies"] = []
            compute_stress = True
        else:
            compute_stress = False

        batch_base = next(iter(data_loader)).to(self.device)
        for model in self.models:
            batch = batch_base.clone()
            out = model(batch.to_dict(), compute_stress=compute_stress)
            if self.model_type in energy_models:
                outputs["energy"].append(out["energy"].detach().cpu().item())
                outputs["free_energy"].append(out["energy"].detach().cpu().item())
                outputs["forces"].append(out["forces"].detach().cpu().numpy())
                outputs["node_energies"].append(
                    (out["node_energy"] - node_e0).detach().cpu().numpy()
                )
                if out["stress"] is not None:
                    outputs["stress"].append(out["stress"].detach().cpu().numpy())
            else:
                pass
            
            for prop in self.internal_implemented_properties:
                if prop in default_properties: continue
                array = out[prop].detach().cpu().numpy()
                if np.isscalar(self.internal_implemented_properties[prop][1]):
                    array = array[0]
                outputs[prop].append(array)

        self.results = {}
        # convert units
        if self.model_type in energy_models:
            energies = np.array(outputs["energy"]) * self.energy_units_to_eV
            self.results["energy"] = np.mean(energies)
            self.results["free_energy"] = self.results["energy"]
            forces = np.array(outputs["forces"]) * (
                self.energy_units_to_eV / self.length_units_to_A
            )
            self.results["forces"] = np.mean(forces, axis=0)
            self.results["node_energy"] = (
                np.mean(np.array(outputs["node_energies"]), axis=0)
                * self.energy_units_to_eV
            )
            if len(outputs["stress"]) > 0:
                stress = np.mean(np.array(outputs["stress"]), axis=0) * (
                    self.energy_units_to_eV / self.length_units_to_A**3
                )
                self.results["stress"] = full_3x3_to_voigt_6_stress(stress)[0]
            if self.num_models > 1:
                self.results["energies"] = energies
                self.results["energy_var"] = np.var(energies)
                self.results["forces_var"] = np.var(forces, axis=0)
        else:
            self.results["energy"]      = 0.
            self.results["free_energy"] = 0.
            self.results["forces"]      = np.zeros(atoms.get_positions().shape)
            self.results["stress"]      = np.zeros(6)

        for prop in self.internal_implemented_properties:
            if prop in default_properties: continue
            self.results[prop] = np.mean(np.array(outputs[prop]), axis=0)
            if self.num_models > 1:
                self.results[f'{prop}_var'] = np.var(np.array(outputs[prop]), axis=0)
    
        pass