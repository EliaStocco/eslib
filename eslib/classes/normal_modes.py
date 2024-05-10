import numpy as np
from copy import copy
from itertools import product
# import xarray as xr
from eslib.functions import get_one_file_in_folder, nparray2list_in_dict
from .io import pickleIO
from warnings import warn
from eslib.units import *
from eslib.tools import convert
import pandas as pd
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.classes.physical_tensor import *
from typing import List, Dict
import pint
import os
import warnings
# Disable all UserWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class NormalModes(pickleIO):

    # To DO :
    # - replace ref with as ase.Atoms and then initialize masses with that

    def to_pickle(self, file):
        pint.get_application_registry()
        super().to_pickle(file)

    @classmethod
    def from_pickle(cls, file_path: str):
        from eslib.units import ureg
        pint.set_application_registry(ureg)
        return super().from_pickle(file_path)
     

    def __init__(self,Nmodes:int,Ndof:int=None,ref:Atoms=None):

        # Nmodes
        self.Nmodes = int(Nmodes)
        if Ndof is None:
            Ndof = Nmodes
        self.Ndof = int(Ndof)

        # Natoms
        self.Natoms = int(self.Ndof / 3)

        self.dynmat = PhysicalTensor(np.full((self.Ndof,self.Ndof),np.nan), dims=('dof-a', 'dof-b'))

        empty = PhysicalTensor(np.full((self.Ndof,self.Nmodes),np.nan,dtype=np.complex128), dims=('dof', 'mode'))
        self.eigvec = empty.copy()
        self.mode   = empty.copy()
        # self.non_ortho_modes = empty.copy()

        self.eigval = PhysicalTensor(np.full(self.Nmodes,np.nan), dims=('mode')) 
        self.masses = PhysicalTensor(np.full(self.Ndof,np.nan), dims=('dof'))

        
        # if ref is not None:
        #     self.set_reference(ref)
        # else:
        #     self.reference = Atoms()
        #     self.masses = None

        self.set_reference(ref)

        pass

    def set_reference(self,ref:Atoms):
        if ref is None:
            self.reference = Atoms()
            # self.masses = PhysicalTensor(np.full(self.Ndof,np.nan), dims=('dof'))
        else:
            # print("setting reference")
            self.reference = Atoms( positions=ref.get_positions(),\
                                    cell=ref.get_cell(),\
                                    symbols=ref.get_chemical_symbols(),\
                                    pbc=ref.get_pbc())
            masses  = [mass for mass in ref.get_masses() for _ in range(3)]
            masses  = np.asarray(masses)
            masses *= convert(1,"mass","dalton","atomic_unit")
            self.masses = PhysicalTensor(masses, dims=('dof'))
            
        
    # def __repr__(self) -> str:
    #     line = "" 
    #     line += "{:<10s}: {:<10d}\n".format("# modes",self.Nmodes)  
    #     line += "{:<10s}: {:<10d}\n".format("# dof",self.Ndof)  
    #     line += "{:<10s}: {:<10d}\n".format("# atoms",self.Natoms)  
    #     return line
    
    def to_dict(self)->dict:
        return nparray2list_in_dict(vars(self))

    def to_folder(self,folder,prefix):

        outputs = {
            "dynmat" : {
                "data" : self.dynmat,
                "header" : "Dynamical Matrix"
            },
            "eigvec" : {
                "data" : self.eigvec,
                "header" : "Eigenvectors"
            },
            "mode" : {
                "data" : self.mode,
                "header" : "Normal Modes"
            },
            "eigval" : {
                "data" : self.eigval,
                "header" : "Eigenvalues"
            }
        }
        for key,output in outputs.items():
            # file = output["file"]
            file = os.path.normpath("{:s}/{:s}.{:s}.txt".format(folder,prefix,key))
            data:PhysicalTensor = output["data"]
            if 'mode' in data.dims and len(data.dims) == 2:
                if data.dims[0] ==  'mode':
                    second_dim = [dim for dim in data.dims if dim != 'mode'][0]
                    data = data.swap_dims({second_dim: 'mode'})
            data = remove_unit(data)[0].to_numpy()
            header = output["header"]
            with open(file,"w") as ffile:
                np.savetxt(ffile,data,header=header)


    @classmethod
    def from_folder(cls,folder=None,ref=None):    

        file = get_one_file_in_folder(folder=folder,ext=".mode")
        tmp = np.loadtxt(file)

        self = cls(tmp.shape[0],tmp.shape[1])    

        # masses
        # I should remove this
        file = get_one_file_in_folder(folder=folder,ext=".masses")
        self.masses[:] = np.loadtxt(file)

        # ortho mode
        file = get_one_file_in_folder(folder=folder,ext=".mode")
        self.mode[:,:] = np.loadtxt(file)

        # eigvec
        file = get_one_file_in_folder(folder=folder,ext=".eigvec")
        self.eigvec[:,:] = np.loadtxt(file)

        # # hess
        # file = get_one_file_in_folder(folder=folder,ext="_full.hess")
        # self.hess = np.loadtxt(file)

        # eigval
        file = get_one_file_in_folder(folder=folder,ext=".eigval")
        self.eigval[:] = np.loadtxt(file)

        # dynmat 
        # I should remove this. it's useless
        file = get_one_file_in_folder(folder=folder,ext=".dynmat")
        self.dynmat[:,:] = np.loadtxt(file)

        if ref is not None:
            self.set_reference(ref)

        # mode
        # self.mode[:,:] = diag_matrix(self.masses,"-1/2") @ self.eigvec
        self.eigvec2modes(_test=False)

        # proj
        # self.proj[:,:] = self.eigvec.T @ diag_matrix(self.masses,"1/2")
        # self.eigvec2proj()

        return self   
    
    def set_dynmat(self,dynmat,mode="phonopy"):
        _dynmat = np.asarray(dynmat)
        if mode == "phonopy":
            # https://phonopy.github.io/phonopy/setting-tags.html
            # _dynmat = []
            N = _dynmat.shape[0]
            dynmat = np.full((N,N),np.nan,dtype=np.complex64) 
            for n in range(N):
                row = np.reshape(_dynmat[n,:], (-1, 2))
                dynmat[n,:] = row[:, 0] + row[:, 1] * 1j
            self.dynmat = PhysicalTensor(dynmat, dims=('dof-a', 'dof-b'))
        else:
            raise ValueError("not implemented yet")
        pass

    def set_modes(self,modes):
        self.mode.values = PhysicalTensor(modes, dims=('dof', 'mode'))
        self.mode /= norm_by(self.mode,"dof")
        
    def set_eigvec(self,band,mode="phonopy"):
        if mode == "phonopy":
            N = self.Nmodes
            eigvec = np.full((N,N),np.nan,dtype=np.complex64)
            for n in range(N):
                f = band[n]["eigenvector"]
                f = np.asarray(f)
                f = f[:,:,0] + 1j * f[:,:,1]
                eigvec[:,n] = f.flatten()
            self.eigvec[:,:] = PhysicalTensor(eigvec, dims=('dof', 'mode'))
        else:
            raise ValueError("not implemented yet")
        pass

    def set_eigval(self,eigval):
        self.eigval[:] = PhysicalTensor(eigval, dims=('mode'))
    
    def set_force_constants(self,force_constant):
        Msqrt = diag_matrix(self.masses,exp="-1/2")
        MsqrtLeft  = PhysicalTensor(Msqrt, dims=('dof-A','dof-a'))
        MsqrtRight = PhysicalTensor(Msqrt, dims=('dof-B','dof-b'))
        Phi = PhysicalTensor(force_constant, dims=('dof-a', 'dof-b'))
        self.dynmat = dot(dot(MsqrtLeft,Phi,'dof-A'),MsqrtRight,'dof-B')
        pass

    def diagonalize(self):
        dm = remove_unit(self.dynmat)[0]
        dm = 0.50 * (dm + dm.T)
        eigval,eigvec = np.linalg.eigh(dm)
        self.eigvec.values = eigvec
        self.eigval.values = eigval
        self.eigvec2modes(_test=False)
        self.sort()

    def eigvec2modes(self,_test:bool=True):
        self.non_ortho_mode = self.eigvec.copy()
        for i in range(self.non_ortho_mode.sizes['dof']):
            index = {'dof': i}
            self.non_ortho_mode[index] = self.eigvec[index] / np.sqrt(self.masses[index])
        if _test:
            test = self.non_ortho_mode / norm_by(self.non_ortho_mode,"dof")
            if not np.allclose(test.data,self.mode.data):
                raise ValueError('some coding error')
        # self.old_mode = self.mode.copy()
        self.mode = self.non_ortho_mode / norm_by(self.non_ortho_mode,"dof")
        pass
    
    def build_supercell_displacement(self,size,q):

        q = np.asarray(q)

        values = [None]*len(size)
        for n,a in enumerate(size):
            values[n] = np.arange(a)
        r_point = list(product(*values))
        
        size = np.asarray(size)
        N = size.prod()
        supercell = NormalModes(self.Nmodes,self.Ndof*N)
        supercell.masses[:] = np.asarray(list(self.masses)*N)
        # supercell.eigvec.fill(np.nan)
        for i,r in enumerate(r_point):
            kr = np.asarray(r) / size @ q
            phase = np.exp(1.j * 2 * np.pi * kr )
            # phi = int(cmath.phase(phase)*180/np.pi)
            # ic(k,r,phi)
            supercell.eigvec[i*self.Ndof:(i+1)*self.Ndof,:] = ( self.eigvec * phase).real
                
        if np.isnan(supercell.eigvec).sum() != 0:
            raise ValueError("error")
        
        supercell.eigvec /= np.linalg.norm(supercell.eigvec,axis=0)
        supercell.eigval = self.eigval.copy()
        
        raise ValueError("Elia Stocco, this is a message for yourself of the past. Check again this script, please!")
        supercell.eigvec2modes()
        # supercell.eigvec2proj()

        return supercell
    
    def nmd2cp(self,A:PhysicalTensor)->Atoms:
        """Normal Modes Displacements to Cartesian Positions (nmd2cp)."""
        D = self.nmd2cd(A)
        P = self.cd2cp(D)
        return P
    
    def ed2cp(self,A:PhysicalTensor)->Atoms:
        """eigenvector displacements to cartesian positions (ed2cp)."""
        B = self.ed2nmd(A)
        D = self.nmd2cd(B)
        return self.cd2cp(D)
        
    def ed2nmd(self,A:PhysicalTensor)->PhysicalTensor:
        """eigenvector displacements to normal modes displacements (ed2nd).
        Convert the coeffients ```A``` [length x mass^{-1/2}] of the ```eigvec``` into the coeffients ```B``` [length] of the ```modes```."""
        invmode = inv(self.mode)
        for dim in ["dof","mode"]:
            test = rbc(invmode,self.mode,dim)
            if np.any(test.imag != 0.0):
                warn("'test' matrix should be real.")
            if not np.allclose(test.to_numpy(),np.eye(len(test))):
                warn("problem with inverting 'mode' matrix.")
        M = self.masses * atomic_unit["mass"] # PhysicalTensor(self.masses,dims=("dof")) * atomic_unit["mass"]
        Msqrt = np.sqrt(M)
        B = dot(invmode,1./Msqrt * dot(self.eigvec,A,"mode"),"dof")
        return remove_unit(B)[0]
    
    def nmd2cd(self,coeff:PhysicalTensor)->Atoms:
        """Normal Modes Displacements to Cartesian Displacements (nmd2cd).
        Return the cartesian displacements as an ```ase.Atoms``` object given the displacement [length] of the normal modes"""
        displ = dot(self.mode,coeff,"mode")
        displ = displ.to_numpy().real
        pos = self.reference.get_positions()
        displ = displ.reshape(pos.shape)
        structure = self.reference.copy()
        displ = displ.reshape((-1,3))
        structure.set_positions(displ)
        return structure
    
    def cd2cp(self,displ:Atoms)->Atoms:
        """cartesian displacements to cartesian positions (cd2cp).
        Return the cartesian positions as an ```ase.Atoms``` object given the cartesian displacement."""
        structure = self.reference.copy()
        structure.set_positions(structure.get_positions()+displ.get_positions())
        return structure

    def project(self,trajectory:List[Atoms],warning="**Warning**")->Dict[str,PhysicalTensor]:       

        #-------------------#
        # reference position
        ref = trajectory[0] if self.reference is None else self.reference

        #-------------------#
        # positions -> displacements
        trajectory = AtomicStructures(trajectory)
        q:np.ndarray = trajectory.get_array("positions") - ref.get_positions()
        q = q.reshape(len(q),-1)
        q *= atomic_unit["length"]

        #-------------------#
        # velocities
        if trajectory.is_there("velocities"):
            try :
                v = trajectory.get_array("velocities") # trajectory.call(lambda e: e.arrays["velocities"])
                v = v.reshape(len(v),-1)
            except:
                warn("velocities not found, setting them to zero.")
                v = np.zeros(q.shape)
        else:
            v = np.zeros(q.shape)

        v *= atomic_unit["velocity"]

        #-------------------#
        # building xarrays
        q = PhysicalTensor(q, dims=('time','dof')) 
        v = PhysicalTensor(v, dims=('time','dof')) 

        #-------------------#
        # eigvec
        # Rename the 'time' dimension to 'new_time' and 'space' dimension to 'new_space'
        eigvec = self.eigvec.copy() * atomic_unit["dimensionless"]
        A = self.eigvec.rename({'mode': 'mode-a', 'dof': 'dof'})
        B = self.eigvec.rename({'mode': 'mode-b', 'dof': 'dof'})
        test = A.dot(B,dim="dof")
        if test.shape != (self.Nmodes,self.Nmodes):
            raise ValueError("wrong shape")
        if np.square(test - np.eye(self.Nmodes)).sum() > 1e-8:
            raise ValueError("eigvec is not orthogonal")
        
        # _mode = np.asarray(self.mode.real.copy())
        # np.round( _mode.T @ _mode, 2)
        # _eigvec = np.asarray(eigvec.real)
        
        #-------------------#
        # masses
        M = self.masses * atomic_unit["mass"] # PhysicalTensor(self.masses,dims=("dof")) * atomic_unit["mass"]
        Msqrt = np.sqrt(M)

        #-------------------#
        # proj
        proj = eigvec.T * Msqrt #np.linalg.inv(Msqrt * eigvec)
        # mode = proj / norm_by(proj,"dof")
        # # mode,_ = set_unit(mode,atomic_unit["dimensionless"])
        # if not np.allclose(mode.data.magnitude,self.mode.data):
        #     raise ValueError("conflict between 'eigvec' and 'mode'")
        
        #-------------------#
        # proj should be real
        if np.any(proj.imag != 0.0):
            warn("'proj' matrix should be real --> discarding its imaginary part.")
        proj = proj.real


        # #-------------------#
        # # Normal Modes should be real
        # save = self.mode.copy()
        # if np.any(self.mode.imag != 0.0):
        #     warn("'mode' matrix should be real --> discarding its imaginary part.")
        # # do it anyway
        # self.mode = self.mode.real

        #-------------------#
        # create the projection operator onto the normal modes
        # proj = self.mode.T * atomic_unit["dimensionless"]

        #-------------------#
        # simple test
        if not check_dim(q,'[length]'):
            raise ValueError("displacements have the wrong unit")
        if not check_dim(v,'[length]/[time]'):
            raise ValueError("velocities have the wrong unit")
        if not check_dim(proj,'[mass]**0.5'):
            raise ValueError("projection operator has the wrong unit")

        #-------------------#
        # project positions and velocities
        # pint is not compatible with np.tensordot
        # we need to remove the unit and set them again
        # q,uq    = remove_unit(q)
        # v,uv    = remove_unit(v)
        # proj,up = remove_unit(proj)

        # qn = proj.dot(q,dim="dof")
        # vn = proj.dot(v,dim="dof")

        # qn   = set_unit(qn,uq*up)
        # vn   = set_unit(vn,uv*up)
        # proj = set_unit(vn,up)

        qn = dot(proj,q,"dof")
        vn = dot(proj,v,"dof")

        # #-------------------#
        # # masses
        # m = proj.dot(self.masses,dim="dof").dot(proj.T,dim="dof")
        # m = set_unit(m,atomic_unit["mass"])

        #-------------------#
        # vib. modes eigenvalues
        w2 = PhysicalTensor(self.eigval, dims=('mode')) 
        w2 = set_unit(w2,atomic_unit["frequency"]**2)
        
        #-------------------#
        # energy: kinetic, potential and total
        #
        # H = 1/2 M V^2 + 1/2 M W^2 X^2
        #   = 1/2 M V^2 + 1/2 K     X^2
        #   = 1/2 M ( V^2 + W^2 X^2 )
        #
        # K = 0.5 * m * vn*vn.conjugate()      # kinetic
        # U = 0.5 * m * w2 * qn*qn.conjugate() # potential
        K = 0.5 * np.square(vn) # vn*vn.conjugate()      # kinetic
        U = 0.5 * w2 * np.square(qn) # qn*qn.conjugate() # potential
        if not check_dim(K,'[energy]'):
            raise ValueError("the kinetic energy has the wrong unit: ",get_unit(K))
        if not check_dim(U,'[energy]'):
            raise ValueError("the potential energy has the wrong unit: ",get_unit(U))

        # if np.any( remove_unit(U)[0] < threshold ):
        if not all_positive(U):
            print("\t{:s}: negative potential energies!".format(warning),end="\n\t")
        # if np.any( remove_unit(K)[0] < threshold ):
        if not all_positive(K):
            print("\t*{:s}:negative kinetic energies!".format(warning),end="\n\t")
        
        energy = U + K
        if not check_dim(energy,'[energy]'):
            raise ValueError("'energy' has the wrong unit")
        else:
            energy = set_unit(energy,atomic_unit["energy"])
            if not check_dim(energy,'[energy]'):
                raise ValueError("'energy' has the wrong unit")
            
        # if np.any( energy < 0 ):
        #     raise ValueError("negative energies!")
            
        #-------------------#
        # amplitudes of the vib. modes
        mode, unit = remove_unit(self.mode)
        invmode = inv(mode)
        invmode = set_unit(invmode,1/unit)
        for dim in ["dof","mode"]:
            test = rbc(invmode,mode,dim)
            if np.any(test.imag != 0.0):
                warn("'test' matrix should be real.")
            if not np.allclose(test.to_numpy(),np.eye(len(test))):
                warn("problem with inverting 'mode' matrix.")

        displacements = dot(invmode,q,"dof").real
        if not check_dim(displacements,"[length]"):
            raise ValueError("'displacements' has the wrong unit.")
        
        B = dot(invmode,1./Msqrt * dot(self.eigvec,qn,"mode"),"dof")
        if not np.allclose(B,displacements):
            warn("'B' and 'displacements' should be equal.")

        # AtoB = rbc(invmode,1./Msqrt * self.eigvec,"dof")
        # B2 = dot(AtoB,qn,"mode")
        # if not np.allclose(B,B2):
        #     warn("'B' and 'B2' should be equal.")

        # vv = 1/np.sqrt(w2) * vn
        # A2 = ( np.square(qn) + np.square(vv) )
        # A  = np.sqrt(A2)
        # amplitude  = dot(dot(self.mode,1./np.sqrt(M) ,"mode"),eigvec,"dof")* A
        # if not check_dim(amplitude,'[length]'):
        #     raise ValueError("'amplitude' have the wrong unit")
        
        # amplitude_ = np.sqrt( energy / ( 0.5 * M * w2 ) )
        # if not np.allclose(amplitude,amplitude_):
        #     raise ValueError("inconsistent value")

        #-------------------#
        # check how much the trajectory 'satisfies' the equipartition theorem
        equipartition = energy.shape[energy.dims.index("mode")] * energy / energy.sum("time")
        if not check_dim(equipartition,'[]'):
            raise ValueError("'equipartition' has the wrong unit")
        
        #-------------------#
        # occupations (in units of hbar)
        occupation = energy / np.sqrt(w2)
        occupation /= atomic_unit["action"]
        if not check_dim(occupation,'[]'):
            raise ValueError("'occupation' has the wrong unit")
        
        #-------------------#
        w = np.sqrt(w2)
        if not check_dim(w,"1/[time]"):
            raise ValueError("'w' has the wrong unit")
        
        #-------------------#
        # phases (angles variables of the harmonic oscillators, i.e. the vib. modes)
        phases = np.arctan2(-vn, w*qn)
        if not check_dim(phases,"[]"):
            raise ValueError("'phases' has the wrong unit")

        #-------------------#
        # output
        out = {
            "energy"        : energy,
            "kinetic"       : K,
            "potential"     : U,
            "displacements" : displacements,
            "equipartition" : equipartition,
            "occupation"    : occupation,
            # "phases"        : phases
        }

        # self.mode = save

        return out
    
    def Zmodes(self,Z:PhysicalTensor)->PhysicalTensor:
        """Compute the Born Effective Charges of each Normal Mode."""
        correction = Z.data.reshape((-1,3,3)).mean(axis=0)
        Z -= np.tile(correction,int(Z.shape[0]/3)).T
        INV = False
        if INV:
            invmode = inv(self.mode)
            dZdN = dot(Z,invmode,dim="dof").real
        else:
            dZdN = dot(Z,self.mode,dim="dof").real
        norm = norm_by(dZdN,"dir")
        dZdN = xr.concat([dZdN, PhysicalTensor(norm, dims='mode')], dim='dir')
        return remove_unit(dZdN)[0]

    def sort(self,criterion="value"):
        if criterion == "value":
            sorted_indices = np.argsort(self.eigval)
        elif criterion == "absolute":
            sorted_indices = np.argsort(np.absolute(self.eigval))
        else:
            raise ValueError("not implemented yet")
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, (np.ndarray, PhysicalTensor)):
                if len(attr_value.shape) == 1:
                    setattr(self, attr_name, attr_value[sorted_indices])
                elif len(attr_value.shape) == 2:
                    setattr(self, attr_name, attr_value[:, sorted_indices])
        return
    
    def get_characteristic_spring_constants(self):
        Nleft  = inv(self.mode.rename({"dof": "dof-a","mode":"mode-a"}))
        Nright = self.mode.rename({"dof": "dof-b","mode":"mode-b"})
        D = self.dynmat# .rename({"mode":"mode-a","mode":"mode-b"})
        M = dot(dot(Nleft,D,"dof-a"),Nright,"dof-b")
        dM = np.diagonal(M).real
        return dM
    
    def get_characteristic_scales(self):
        """Returns a `pandas.DataFrame` with the characteristic scales of the normal modes, intended as quantum harmonic oscillators.
        
        The scales depend on:
            - `hbar`: the reduced Planck constant (`hbar`=1 in a.u.)
            - `w`: the angular frequency of the mode
            - `m`: the characteristic mass  of the mode

        The provided scales are computed as follows:
            - frequency: 2pi/w
            - energy: hw
            - time: 2pi/w
            - length: sqrt(h/mw)
            - spring constant: mw^2
        """
        hbar = 1
        scales = pd.DataFrame(columns=["angular frequency","frequency","energy","time","mass","length","spring constant"],index=np.arange(self.Nmodes))
        w2 = self.eigval.to_numpy()
        w = corrected_sqrt(w2)
        scales["angular frequency"] = w
        scales["frequency"] = scales["angular frequency"] / (2*np.pi)
        scales["energy"] = hbar * scales["angular frequency"]
        scales["time"] = 1. / scales["frequency"]
        scales["spring constant"]     = self.get_characteristic_spring_constants()
        scales["mass"] = scales["spring constant"] / (scales["angular frequency"]**2)
        scales["length"] = corrected_sqrt( hbar / (scales["mass"]*scales["angular frequency"]))
        
        return scales
    
    def potential_energy(self,structure:List[Atoms]):
        """Compute the harmonic energy of list of atomic structures."""
        results = self.project(structure)
        # assert np.linalg.norm((results['energy'] - results['potential']).to_numpy()) < 1e-8
        potential = results['potential'].to_numpy()
        return potential.sum(axis=0)
    
    def get(self,name):
        data = getattr(self,name)
        return remove_unit(data)[0].to_numpy()