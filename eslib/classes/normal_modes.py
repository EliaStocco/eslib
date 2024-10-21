import pint
import os
import numpy as np
from copy import deepcopy
from itertools import product
from warnings import warn
import pandas as pd
from ase import Atoms
from typing import List, Dict, TypeVar

from eslib.classes.io import pickleIO
from eslib.functional import unsafe, improvable
from eslib.tools import is_sorted_ascending, cart2frac, w2_to_w
from eslib.formatting import warning
from eslib.classes.physical_tensor import *
from eslib.units import *
from eslib.tools import convert
from eslib.functions import get_one_file_in_folder #, nparray2list_in_dict
from eslib.classes.atomic_structures import AtomicStructures

T = TypeVar('T', bound='NormalModes')

class NormalModes(pickleIO):

    # To DO :
    # - replace ref with as ase.Atoms and then initialize masses with that
    
    # parameters
    Ndof     :int
    Nmodes   :int
    Natoms   :int
    size     :List[int]
    
    # matrices (atomic units)
    dynmat   :PhysicalTensor
    eigvec   :PhysicalTensor
    mode     :PhysicalTensor
    eigval   :PhysicalTensor
    masses   :PhysicalTensor
    
    # reference structure (bohr)
    reference:Atoms
    

    def to_pickle(self:T, file)->None:
        pint.get_application_registry()
        super().to_pickle(file)

    @classmethod
    def from_pickle(cls, file_path: str):
        from eslib.units import ureg
        pint.set_application_registry(ureg)
        return super().from_pickle(file_path)
     

    def __init__(self:T,Nmodes:int,Ndof:int=None,ref:Atoms=None):

        # Nmodes
        self.Nmodes = int(Nmodes)
        if Ndof is None:
            Ndof = Nmodes
        self.Ndof = int(Ndof)
        
        self.size = [1,1,1]

        # Natoms
        self.Natoms = int(self.Ndof / 3)

        # dynamical matrix
        self.dynmat = PhysicalTensor(np.full((self.Ndof,self.Ndof),np.nan), dims=('dof-a', 'dof-b'))

        # normal modes and eigenvectors
        empty = PhysicalTensor(np.full((self.Ndof,self.Nmodes),np.nan+1j*np.nan,dtype=np.complex128), dims=('dof', 'mode'))
        self.eigvec = empty.copy()
        self.mode   = empty.copy()
        
        # eigenvalues and masses
        self.eigval = PhysicalTensor(np.full(self.Nmodes,np.nan), dims=('mode')) 
        self.masses = PhysicalTensor(np.full(self.Ndof,np.nan), dims=('dof'))

        # reference structure
        self.set_reference(ref)

        pass

    def set_reference(self:T,ref:Atoms):
        """
        Sets the reference structure for the NormalModes object.
        Attention: the reference structure should be in atomic units (bohr).

        Parameters:
            ref (Atoms): The reference structure. If None, an empty Atoms object is used.

        Returns:
            None
        """
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
            
        
    def __repr__(self) -> str:
        line = "Parameters:\n" 
        line += "  {:<10s}: {:<10d}\n".format("atoms",self.Natoms)  
        line += "  {:<10s}: {:<10d} (atoms x 3)\n".format("dof",self.Ndof)  
        line += "  {:<10s}: {:<10d}\n".format("modes",self.Nmodes)   
        line += "  {:<10s}: {}\n".format("size",self.size)                  
        line += "Matrices:"
        for mat in ["dynmat","eigvec","mode","eigval","masses"]:
            line += "\n  {:<10s}: {}".format(mat,vars(self)[mat].shape)
        line += "\nReference structure (bohr/a.u.):"
        line += "\n  {:<10s}: {}".format("reference", self.reference)  
        return line
    
    # def to_dict(self)->dict:
    #     return nparray2list_in_dict(vars(self))

    # @unsafe
    def to_folder(self:T,folder,prefix):

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
        # file = get_one_file_in_folder(folder=folder,ext=".masses")
        # self.masses[:] = np.loadtxt(file)

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
    
    # @unsafe
    def set_dynmat(self:T,dynmat,mode="phonopy"):
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

    # @unsafe
    def set_modes(self:T,modes):
        self.mode.values = PhysicalTensor(modes, dims=('dof', 'mode'))
        self.mode /= norm_by(self.mode,"dof")
        
    # @unsafe
    def set_eigvec(self:T,band,mode="phonopy"):
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

    # @unsafe
    def set_eigval(self:T,eigval):
        self.eigval[:] = PhysicalTensor(eigval, dims=('mode'))
    
    @unsafe
    def set_force_constants(self:T,force_constant):
        Msqrt = diag_matrix(self.masses,exp="-1/2")
        MsqrtLeft  = PhysicalTensor(Msqrt, dims=('dof-A','dof-a'))
        MsqrtRight = PhysicalTensor(Msqrt, dims=('dof-B','dof-b'))
        Phi = PhysicalTensor(force_constant, dims=('dof-a', 'dof-b'))
        self.dynmat = dot(dot(MsqrtLeft,Phi,'dof-A'),MsqrtRight,'dof-B')
        # tmp  = np.allclose(self.dynmat,self.dynmat.T)
        pass

    # @unsafe
    def diagonalize(self:T,symmetrize:bool=True):
        """
        Diagonalize the dynamical matrix to compute eigenvalues and eigenvectors.
        
        Parameters:
            symmetrize (bool): Flag to symmetrize the dynamical matrix.
        
        Returns:
            None
        """
        dm = remove_unit(self.dynmat)[0]
        if symmetrize:
            dm = 0.50 * (dm + dm.T)
        eigval,eigvec = np.linalg.eigh(dm)
        self.eigvec.values = eigvec
        self.eigval.values = eigval
        self.eigvec2modes(_test=False)
        self.sort()
        pass

    def copy(self:T)->T:
        """
        Return a deep copy of the current object.

        Returns:
            NormalModes: A deep copy of the current object.
        """
        return deepcopy(self)

    def check(self:T,threshold=1e-4,**argv):
        """
        Check if the eigenvalues, eigenvectors, and normal modes of the current NormalModes object match those of another NormalModes object within a specified threshold.
        
        Parameters:
            threshold (float): The threshold value for the comparison.
            **argv: Additional keyword arguments.
        """
        tmp = self.copy()
        tmp.diagonalize(symmetrize=False,**argv)
        if np.linalg.norm(tmp.eigval.data-self.eigval.data)/len(self.eigval.data) > threshold:  warn("Eigenvalues do not match")
        if np.linalg.norm(tmp.eigvec.data-self.eigvec.data)/len(self.eigvec.data) > threshold:  warn("Eigenvectors do not match")
        if np.linalg.norm(tmp.mode.data-self.mode.data)    /len(self.mode.data)   > threshold:  warn("Normal modes do not match")

    # @unsafe
    def eigvec2modes(self:T,_test:bool=True):
        """
        Convert the eigenvectors to the normal modes.

        Args:
            _test (bool, optional): Whether to perform a test to check if the conversion is correct. Defaults to True.

        Raises:
            ValueError: If the test fails and the conversion is not correct.

        Returns:
            None
        """
        self.non_ortho_mode = self.eigvec.copy()
        for i in range(self.non_ortho_mode.sizes['dof']):
            index = {'dof': i}
            self.non_ortho_mode[index] = self.eigvec[index] / np.sqrt(self.masses[index])
        # self.old_mode = self.mode.copy()
        self.mode = self.non_ortho_mode / norm_by(self.non_ortho_mode,"dof")
        if _test:
            test = self.non_ortho_mode / norm_by(self.non_ortho_mode,"dof")
            if not np.allclose(test.data,self.mode.data):
                raise ValueError('some coding error')
        pass
    
    # @unsafe
    def build_supercell_displacement(self:T,size,q,info:dict)->T:

        q = np.asarray(q)

        values = [None]*len(size)
        for n,a in enumerate(size):
            values[n] = np.arange(a)
        # r_point = list(product(*values))
        
        size = np.asarray(size)
        # N = size.prod()
        
        cell = np.asarray(info['supercell']['lattice'])
        sc = np.asarray([a['coordinates']  for a in info['supercell']['points']])
        sym = np.asarray([a['symbol']  for a in info['supercell']['points']])
        ref = Atoms(cell=cell,scaled_positions=sc,symbols=sym)
        
        # mapping from supercell to supercell
        index = np.asarray([a['reduced_to']  for a in info['supercell']['points']]) -1
        v,i = np.unique(index,return_index=True)
        assert np.allclose(v,i), "some coding error"
        # mapping from supercell to unit cell
        _,indexuc = np.unique(index,return_inverse=True)
        
        cell = np.asarray(info['unit_cell']['lattice'])
        sc = np.asarray([a['coordinates']  for a in info['unit_cell']['points']])
        sym = np.asarray([a['symbol']  for a in info['unit_cell']['points']])
        prim = Atoms(cell=cell,scaled_positions=sc,symbols=sym)
        
        assert np.allclose(prim.positions , ref.positions[i,:]), "wrong mapping from supercell to unit cell"
        
        N = ref.get_global_number_of_atoms()
        delta = np.zeros((N,3))
        for i in range(N):
            delta[i] = ref.positions[i] - ref.positions[index[i]]
            
        displ = cart2frac(prim.get_cell(),delta) 
        
        ref_au = ref.copy()
        ref_au.positions *= convert(1,"length","angstrom","atomic_unit")
        ref_au.cell *= convert(1,"length","angstrom","atomic_unit")
        supercell = NormalModes(self.Nmodes,self.Ndof*size.prod(),ref=ref_au)
        supercell.size = size.tolist()
        
        for i,r in enumerate(displ):
            kr = np.asarray(r) / size @ q
            phase = np.exp(1.j * 2 * np.pi * kr )
            # phi = int(cmath.phase(phase)*180/np.pi)
            # ic(k,r,phi)
            j = indexuc[i]
            supercell.eigvec[3*i:3*i+3,:] = self.eigvec[3*j:3*j+3,:] * phase
            
        assert not np.any(np.isnan(supercell.eigvec.data)), "some coding error"
        
        # # supercell.masses[:] = np.asarray(list(self.masses)*N)
        # # supercell.eigvec.fill(np.nan)
        # for i,r in enumerate(r_point):
        #     kr = np.asarray(r) / size @ q
        #     phase = np.exp(1.j * 2 * np.pi * kr )
        #     # phi = int(cmath.phase(phase)*180/np.pi)
        #     # ic(k,r,phi)
        #     supercell.eigvec[i*self.Ndof:(i+1)*self.Ndof,:] = ( self.eigvec * phase).real
                
        # if np.isnan(supercell.eigvec).sum() != 0:
        #     raise ValueError("error")
        
        supercell.eigvec /= np.linalg.norm(supercell.eigvec,axis=0)# *size.prod())
        supercell.eigval = self.eigval.copy()
        
        # raise ValueError("Elia Stocco, this is a message for yourself of the past. Check again this script, please!")
        supercell.eigvec2modes()
        # supercell.eigvec2proj()

        return supercell
    
    def nmd2cp(self:T,A:PhysicalTensor)->Atoms:
        """Normal Modes Displacements to Cartesian Positions (nmd2cp)."""
        D = self.nmd2cd(A)
        P = self.cd2cp(D)
        return P
    
    def ed2cp(self:T,A:PhysicalTensor)->Atoms:
        """eigenvector displacements to cartesian positions (ed2cp)."""
        B = self.ed2nmd(A)
        D = self.nmd2cd(B)
        return self.cd2cp(D)
        
    def ed2nmd(self:T,A:PhysicalTensor)->PhysicalTensor:
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
    
    def nmd2cd(self:T,coeff:PhysicalTensor)->Atoms:
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
    
    def cd2cp(self:T,displ:Atoms)->Atoms:
        """cartesian displacements to cartesian positions (cd2cp).
        Return the cartesian positions as an ```ase.Atoms``` object given the cartesian displacement."""
        structure = self.reference.copy()
        structure.set_positions(structure.get_positions()+displ.get_positions())
        return structure

    # @improvable
    def project(self:T,trajectory:List[Atoms])->Dict[str,PhysicalTensor]:    

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
        
        # #-------------------#
        # # proj should be real
        # if np.any(proj.imag != 0.0):
        #     warn("'proj' matrix should be real --> discarding its imaginary part.")
        # # Maybe this should remain complex for Gamma modes
        # proj = proj.real


        # #-------------------#
        # # Maybe this should remain complex for Gamma modes
        # # Normal Modes should be real
        # save = self.mode.copy()
        # if np.any(self.mode.imag != 0.0):
        #     warn("'mode' matrix should be real --> discarding its imaginary part.")
        # # do it anyway
        # self.mode = self.mode.real

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
        qn = dot(proj,q,"dof")
        vn = dot(proj,v,"dof")

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
        # K = 0.5 * np.square(vn) # vn*vn.conjugate()      # kinetic
        # U = 0.5 * w2 * np.square(qn) # qn*qn.conjugate() # potential
        K = 0.5 * np.absolute(vn)**2 # vn*vn.conjugate()      # kinetic
        U = 0.5 * w2 * np.absolute(qn)**2 # qn*qn.conjugate() # potential
        if not check_dim(K,'[energy]'):
            raise ValueError("the kinetic energy has the wrong unit: ",get_unit(K))
        if not check_dim(U,'[energy]'):
            raise ValueError("the potential energy has the wrong unit: ",get_unit(U))

        # if np.any( remove_unit(U)[0] < threshold ):
        if not all_positive(U):
            print("\t{:s}: negative potential energies!".format(warning),end="\n\t")
        # if np.any( remove_unit(K)[0] < threshold ):
        if not all_positive(K):
            print("\t{:s}: negative kinetic energies!".format(warning),end="\n\t")
        
        energy = U + K
        if not check_dim(energy,'[energy]'):
            raise ValueError("'energy' has the wrong unit")
        else:
            energy = set_unit(energy,atomic_unit["energy"])
            if not check_dim(energy,'[energy]'):
                raise ValueError("'energy' has the wrong unit")
            
        #-------------------#
        # amplitudes of the vib. modes
        mode, unit = remove_unit(self.mode)
        if mode.shape[0] == mode.shape[1]:
            # square matrix
            invmode = inv(mode)
            invmode = set_unit(invmode,1/unit)
            for dim in ["dof","mode"]:
                test = rbc(invmode,mode,dim)
                if np.any(test.imag != 0.0):
                    warn("'test' matrix should be real.")
                if not np.allclose(test.to_numpy(),np.eye(len(test))):
                    warn("problem with inverting 'mode' matrix.")
                    
            displacements = dot(invmode,q,"dof").real
            
            B = dot(invmode,1./Msqrt * dot(self.eigvec,qn,"mode"),"dof")
            if not np.allclose(B,displacements):
                warn("'B' and 'displacements' should be equal.")
            
        else:
            # rectagular matrix
            _A = rbc(mode,mode,"dof") # mode.T @ mode
            _Atmp:np.ndarray = remove_unit(_A)[0].data
            assert np.allclose(np.diag(_Atmp),1), "The matrix should contain normalized vectors."
            assert np.allclose(_Atmp,_Atmp.T), "The matrix should be symmetric."
            del _Atmp
            _b = dot(mode,q,"dof") # rbc(mode,q,"dof")
            displacements = np.linalg.solve(_A.data,_b.data)

        if not check_dim(displacements,"[length]"):
                raise ValueError("'displacements' has the wrong unit.")

        #-------------------#
        # check how much the trajectory 'satisfies' the equipartition theorem
        equipartition = energy.shape[energy.dims.index("mode")] * energy / energy.sum("time")
        if not check_dim(equipartition,'[]'):
            raise ValueError("'equipartition' has the wrong unit")
        
        #-------------------#
        w = w2_to_w(w2)
        if not check_dim(w,"1/[time]"):
            raise ValueError("'w' has the wrong unit")
        
        #-------------------#
        # occupations (in units of hbar)
        occupation = energy / w
        occupation /= atomic_unit["action"]
        if not check_dim(occupation,'[]'):
            raise ValueError("'occupation' has the wrong unit")
        
        # #-------------------#
        # # phases (angles variables of the harmonic oscillators, i.e. the vib. modes)
        # phases = np.arctan2(-vn, w*qn)
        # if not check_dim(phases,"[]"):
        #     raise ValueError("'phases' has the wrong unit")

        #-------------------#
        energy_summary = [None]*3
        energy_summary[0] = K.sum(axis=0)
        energy_summary[1] = U.sum(axis=0)
        energy_summary[2] = energy.sum(axis=0)
        energy_summary = np.asarray(energy_summary)
        
        # output
        out = {
            "total-energy"  : energy_summary,
            "energy"        : energy,
            "kinetic"       : K,
            "potential"     : U,
            "displacements" : displacements,
            "equipartition" : equipartition,
            "occupation"    : occupation,
            # "phases"        : phases
        }

        return out
    
    def Zmodes(self:T,Z:PhysicalTensor)->PhysicalTensor:
        """Compute the Born Effective Charges of each Normal Mode."""
        # correction = Z.data.reshape((-1,3,3)).mean(axis=0)
        # Z -= np.tile(correction,int(Z.shape[0]/3)).T
        # INV = True
        # if INV:
        M = self.mode.data.real
        # invM = np.linalg.inv(np.asarray(M))
        Z -= Z.mean(axis=0)
        dZdN = (np.asarray(Z).T @ M).T
        norm = np.sqrt(dZdN[:,0]**2 + dZdN[:,1]**2 + dZdN[:,2]**2 )
        out = np.zeros((dZdN.shape[0],4))
        out[:,0:3] = dZdN
        out[:,3] = norm
        return out
        # invmode = inv(self.mode)
        # dZdN = dot(Z,invmode,dim="dof").real
        # else:
        #     dZdN = dot(Z,self.mode.real,dim="dof")
        # norm = norm_by(dZdN,"dir")
        # dZdN = xr.concat([dZdN, PhysicalTensor(norm, dims='mode')], dim='dir')
        # return remove_unit(dZdN)[0]

    def sort(self:T,criterion="value"):
        values = w2_to_w(self.eigval.data)
        if criterion == "value":
            sorted_indices = np.argsort(values)
        elif criterion == "absolute":
            sorted_indices = np.argsort(np.absolute(values))
        else:
            raise ValueError("not implemented yet")
        if not is_sorted_ascending(sorted_indices):
            known_vars = ["dynmat","eigvec","mode","eigval"]
            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, (np.ndarray, PhysicalTensor)):
                    if attr_name in ["eigval"]:
                        setattr(self, attr_name, attr_value[sorted_indices])
                    elif attr_name in ["eigvec","mode","non_ortho_mode"]:
                        setattr(self, attr_name, attr_value[:, sorted_indices])
                    # elif attr_name in []:
                    #     setattr(self, attr_name, attr_value[sorted_indices, sorted_indices])
                    elif attr_name in ["masses","dynmat"]:
                        pass
                    else:
                        raise ValueError("Coding error: I don't know how to sort the attribute '{}'.".format(attr_name))
    
    @unsafe
    def get_characteristic_spring_constants(self:T):
        Nleft  = inv(self.mode.rename({"dof": "dof-a","mode":"mode-a"}))
        Nright = self.mode.rename({"dof": "dof-b","mode":"mode-b"})
        D = self.dynmat# .rename({"mode":"mode-a","mode":"mode-b"})
        M = dot(dot(Nleft,D,"dof-a"),Nright,"dof-b")
        dM = np.diagonal(M).real
        return dM
    
    @improvable
    def get_characteristic_scales(self:T):
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
    
    @unsafe
    def potential_energy(self:T,structure:List[Atoms]):
        """Compute the harmonic energy of list of atomic structures."""
        results = self.project(structure)
        # assert np.linalg.norm((results['energy'] - results['potential']).to_numpy()) < 1e-8
        potential = results['potential'].to_numpy()
        return potential.sum(axis=0)
    
    def get(self:T,name):
        data = getattr(self,name)
        return remove_unit(data)[0].to_numpy() 