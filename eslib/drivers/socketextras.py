import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
import json
import ase.units as units
from ase.calculators.socketio import SocketClient, SocketClosed

class FormatExtras:
    """Class to properly format extras string such that they will be compatible with i-PI."""

    @staticmethod
    def format(name:str,array:np.ndarray,atoms:Atoms)->np.ndarray:
        if name.lower() in ["bec","dipole_dr"]:
            return FormatExtras.format_bec(name,array,atoms)
        return name, array
    
    @staticmethod
    def format_bec(name:str,array:np.ndarray,atoms:Atoms)->np.ndarray:
        """Format Born Effective Charge Tensors."""
        array = np.asarray(array)        # (natoms,3,3)  -->  (atom index   ,pos. coord.   ,dipole coord.)
        array = np.moveaxis(array, 2, 0) # (3,natoms,3)  -->  (dipole coord., atom index   , pos. coord. )
        array = np.moveaxis(array, 1, 2) # (3,3,natoms)  -->  (dipole coord., pos. coord.  , atom index  )
        array = array.reshape((3,-1))    # (3,3xnatoms)  -->  (dipole coord., all coord.                 ) with R1x R1y R1z R2x R2y R2z ....
        array = array.T                  # (3xnatoms,3)  -->  (all coord.   , dipole coord.              )
        return "BEC",array

class ProtocolExtras:

    def ipi_sendforce(self, energy, forces, virial,morebytes=np.zeros(1, dtype=np.byte)):
        assert np.array([energy]).size == 1
        assert forces.shape[1] == 3
        assert virial.shape == (3, 3)

        self.log(' sendforce')
        self.sendmsg('FORCEREADY')  # mind the units
        self.send(np.array([energy / units.Ha]), np.float64)
        natoms = len(forces)
        self.send(np.array([natoms]), np.int32)
        self.send(units.Bohr / units.Ha * forces, np.float64)
        self.send(1.0 / units.Ha * virial.T, np.float64)
        # We prefer to always send at least one byte due to trouble with
        # empty messages.  Reading a closed socket yields 0 bytes
        # and thus can be confused with a 0-length bytestring.
        # morebytes = json.dumps(morebytes)
        self.send(np.int32(len(morebytes)), np.int32)
        # self.send(morebytes, np.byte)
        self.socket.send(morebytes.encode("utf-8"))
        return

class SocketClientExtras(SocketClient):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.protocol.sendforce = lambda  energy, forces, virial, morebytes : ProtocolExtras.ipi_sendforce(self.protocol, energy, forces, virial,morebytes)

    def calculate(self, atoms:Atoms, use_stress:bool):
        # We should also broadcast the bead index, once we support doing
        # multiple beads.
        self.comm.broadcast(atoms.positions, 0)
        self.comm.broadcast(np.ascontiguousarray(atoms.cell), 0)

        energy = 0
        forces = np.zeros_like(atoms.positions)
        virial = np.zeros((3, 3))

        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            if use_stress:
                stress = atoms.get_stress(voigt=False)
                virial = -atoms.get_volume() * stress
            else:
                virial = np.zeros((3, 3))
        except:
            pass        

        extras = {}
        calc:Calculator = atoms.calc
        for name in calc.implemented_properties:
            if name in ['energy', 'free_energy', 'node_energy', 'forces', 'stress']: continue
            array:np.ndarray = calc.get_property(name=name,atoms=atoms)
            name,array = FormatExtras.format(name,array,atoms)
            extras[name] = array.tolist()            
        extras = json.dumps(extras)

        return energy, forces, virial, extras
    
    def irun_rank0(self, atoms, use_stress=True):
        # For every step we either calculate or quit.  We need to
        # tell other MPI processes (if this is MPI-parallel) whether they
        # should calculate or quit.
        try:
            while True:
                try:
                    msg = self.protocol.recvmsg()
                except SocketClosed:
                    # Server closed the connection, but we want to
                    # exit gracefully anyway
                    msg = 'EXIT'

                if msg == 'EXIT':
                    # Send stop signal to clients:
                    self.comm.broadcast(np.ones(1, bool), 0)
                    # (When otherwise exiting, things crashed and we should
                    # let MPI_ABORT take care of the mess instead of trying
                    # to synchronize the exit)
                    return
                elif msg == 'STATUS':
                    self.protocol.sendmsg(self.state)
                elif msg == 'POSDATA':
                    assert self.state == 'READY'
                    cell, icell, positions = self.protocol.recvposdata()
                    atoms.cell[:] = cell
                    atoms.positions[:] = positions

                    # User may wish to do something with the atoms object now.
                    # Should we provide option to yield here?
                    #
                    # (In that case we should MPI-synchronize *before*
                    #  whereas now we do it after.)

                    # Send signal for other ranks to proceed with calculation:
                    self.comm.broadcast(np.zeros(1, bool), 0)
                    energy, forces, virial, extras = self.calculate(atoms, use_stress)

                    self.state = 'HAVEDATA'
                    yield
                elif msg == 'GETFORCE':
                    assert self.state == 'HAVEDATA', self.state                
                    if extras is None or extras is {}:
                        extras = np.zeros(1, dtype=np.byte)
                    self.protocol.sendforce(energy, forces, virial, morebytes=extras)
                    self.state = 'NEEDINIT'
                elif msg == 'INIT':
                    assert self.state == 'NEEDINIT'
                    bead_index, initbytes = self.protocol.recvinit()
                    self.bead_index = bead_index
                    self.bead_initbytes = initbytes
                    self.state = 'READY'
                else:
                    raise KeyError('Bad message', msg)
        finally:
            self.close()
