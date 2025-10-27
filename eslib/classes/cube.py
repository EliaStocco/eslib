import io
import pytest
import numpy as np
from ase.io.cube import read_cube
from dataclasses import dataclass, field
from typing import Tuple, Union

@dataclass
class CubeData:
    """
    Class to handle volumetric cube data (Hartree potential, density, etc.)
    """
    data: np.ndarray # 3D dimensional data of a scalar or vector field
    origin: np.ndarray # always in real space
    vectors: np.ndarray # three 3D lattice vectors
    space:str = field(default="real") # must be "real" or "reciprocal"

    #------------------#
    # initialization and construction
    #------------------#
    def __post_init__(self):
        if self.space not in ("real", "reciprocal"):
            raise ValueError(f"space must be 'real' or 'reciprocal', got '{self.space}'")
    
    @classmethod
    def from_file(cls, filename: str):
        """
        Factory method to create a CubeData instance from a cube file.
        Only stores data, origin, and vectors.
        """
        with open(filename, "r") as f:
            cube_dict = read_cube(f)  # ignore atoms

        data = np.asarray(cube_dict['data'])
        origin = np.array(cube_dict['origin'])
        spacing = np.array(cube_dict['spacing'])  # shape (3,3)

        # Calculate the full lattice vectors of the cube
        # Each vector = voxel spacing * number of voxels along that axis
        vectors = np.array([spacing[i] * data.shape[i] for i in range(3)])

        return cls(data, origin, vectors)
    
    def to_file(self,filename:str,format="npy"):
        if format == "npy":
            np.save(filename,self.data)
        else:
            raise ValueError(f"Format '{format}' is not supported.")
        
    #------------------#
    # properties
    #------------------#
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def spacing(self):
        """
        Returns the voxel spacing vectors along each axis.
        Each vector is the displacement of one voxel along that axis.
        """
        # Number of voxels along each axis
        shape = np.array(self.data.shape)
        
        # Spacing = total vector / number of voxels along that axis
        return np.array([self.vectors[i] / shape[i] for i in range(3)])
    
    @property
    def coordinates(self) -> np.ndarray:
        """
        Cartesian coordinates of each voxel in the cube.

        Returns
        -------
        coords : np.ndarray
            Array of shape (nx, ny, nz, 3) with the Cartesian coordinates
            of each voxel in real or reciprocal space depending on `self.space`.
        """
        nx, ny, nz = self.data.shape
        # Fractional coordinates along each axis (0 to 1)
        frac_x = np.linspace(0, 1, nx, endpoint=False)
        frac_y = np.linspace(0, 1, ny, endpoint=False)
        frac_z = np.linspace(0, 1, nz, endpoint=False)

        # 3D fractional grid
        FX, FY, FZ = np.meshgrid(frac_x, frac_y, frac_z, indexing='ij')

        # Convert to Cartesian coordinates: r = fx*a1 + fy*a2 + fz*a3
        a1, a2, a3 = self.vectors
        coords = FX[..., None] * a1 + FY[..., None] * a2 + FZ[..., None] * a3

        return coords
    
    @property
    def is_scalar(self) -> bool:
        """
        Return True if the data represents a scalar field.
        """
        return self.data.ndim == 3

    #------------------#
    # summary/debugging 
    #------------------#
    def summary(self, title: str = "CubeData summary", prefix: str = "") -> str:
        """
        Generate a summary of the CubeData, including shape, origin,
        lattice vectors, their lengths, edge densities, and basic statistics
        of the data (min, max, mean).

        Parameters
        ----------
        title : str
            Title to display at the top of the summary.
        prefix : str
            String to prepend at the beginning of each line (default: "").

        Returns
        -------
        summary_str : str
            The full summary as a string.
        """
        buffer = io.StringIO()
        print(f"\n{prefix}" + "="*60, file=buffer)
        print(f"{prefix}{title}", file=buffer)
        print(f"{prefix} - space: {self.space}", file=buffer)
        print(f"{prefix} - shape: {self.data.shape}", file=buffer)
        print(f"{prefix} - origin (always real space): {self.origin}", file=buffer)
        print(f"{prefix} - lattice vectors (cartesian coordinates):", file=buffer)
        for i, vec in enumerate(self.vectors):
            length = np.linalg.norm(vec)
            runit = "Å" if self.space == "real" else "Å⁻¹"
            kunit = "Å⁻¹" if self.space == "real" else "Å"
            n_points = self.data.shape[i]
            print(f"\n{prefix}\t - vector {i}: {vec}", file=buffer)
            print(f"{prefix}\t\t - length ({runit}): {length:.5f}", file=buffer)
            print(f"{prefix}\t\t - number of points along axis: {n_points}", file=buffer)
            print(f"{prefix}\t\t - edge density ({kunit}): {n_points / length:.5f}", file=buffer)
            print(f"{prefix}\t\t - edge spacing ({runit}): {length/n_points:.5f}", file=buffer)
        print(f"\n{prefix}" + "="*60, file=buffer)
        return buffer.getvalue()

    def __eq__(self, other:'CubeData')->bool:
        """Check if two CubeData instances have the same space, data, origin, and vectors."""
        if not isinstance(other, CubeData):
            return NotImplemented

        # Check that space is the same
        if self.space != other.space:
            return False

        # Check arrays using allclose for floating point data
        arrays_equal = (
            np.allclose(self.data, other.data, rtol=1e-8, atol=1e-12) and
            np.allclose(self.origin, other.origin, rtol=1e-8, atol=1e-12) and
            np.allclose(self.vectors, other.vectors, rtol=1e-8, atol=1e-12)
        )

        return arrays_equal

    #------------------#
    # mathematical operations
    #------------------#
    def kspace(self) -> 'CubeData':
        """
        Compute the Fourier transform of the cube data and return
        a CubeData instance in reciprocal (k-) space.

        The 'vectors' attribute contains the 3x3 reciprocal lattice basis.
        """
        if self.space != "real":
            raise ValueError("CubeData is not in real space.")

        # FFT of the data (unitary normalization)
        data_k = np.fft.fftn(self.data, norm="ortho")

        # Real-space lattice vectors (3x3)
        a1, a2, a3 = self.vectors

        # Reciprocal lattice vectors
        volume = np.dot(a1, np.cross(a2, a3))
        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume
        b3 = 2 * np.pi * np.cross(a1, a2) / volume
        vectors_k = np.array([b1, b2, b3])

        return CubeData(data=data_k, origin=self.origin.copy(), vectors=vectors_k, space="reciprocal")

    def rspace(self) -> 'CubeData':
        """
        Transform the CubeData from reciprocal (k-) space back to real space.
        """
        if self.space != "reciprocal":
            raise ValueError("CubeData is not in reciprocal space.")

        data_real = np.fft.ifftn(self.data, norm="ortho").real

        # The original real-space vectors should be stored somewhere;
        # if not, you could invert the reciprocal lattice:
        a1, a2, a3 = self.vectors
        volume = np.dot(a1, np.cross(a2, a3))
        r1 = 2 * np.pi * np.cross(a2, a3) / volume
        r2 = 2 * np.pi * np.cross(a3, a1) / volume
        r3 = 2 * np.pi * np.cross(a1, a2) / volume
        vectors_real = np.array([r1, r2, r3])  # optional if you want exact inverse

        return CubeData(data=data_real, origin=self.origin.copy(), vectors=vectors_real, space="real")
    
    def integrate(self, axis:Union[int,Tuple[int]]=(0,1,2)) -> np.ndarray:
        """
        Integrate the cube data along one or more axes, taking into account voxel spacing.

        Parameters
        ----------
        axis : int or tuple/list of int
            Axis or axes along which to integrate (0=x, 1=y, 2=z)

        Returns
        -------
        np.ndarray
            Integrated array with the specified axis/axes removed.
            Shape: remaining axes for scalar or vector fields.
        """
        # Convert axis to tuple if it is a single int
        if isinstance(axis, int):
            axes = (axis,)
        elif isinstance(axis, (tuple, list)):
            axes = tuple(axis)
        else:
            raise ValueError("axis must be int or tuple/list of ints.")

        # Compute the product of voxel lengths along the axes
        voxel_vol = np.prod([np.linalg.norm(self.spacing[a]) for a in axes])

        return np.sum(self.data, axis=axes) * voxel_vol
    
#------------------#
# pytest for CubeData
#------------------#
def test_cubedata_basic():
    # Random data
    nx, ny, nz = 10, 12, 8
    data = np.random.rand(nx, ny, nz)
    origin = np.array([0.0, 0.0, 0.0])
    vectors = np.array([[10.0, 0.0, 0.0],
                        [0.0, 12.0, 0.0],
                        [0.0, 0.0, 8.0]])

    cube = CubeData(data, origin, vectors)

    # Test properties
    assert cube.is_scalar
    assert cube.shape == (nx, ny, nz)
    coords = cube.coordinates
    assert coords.shape == (nx, ny, nz, 3)

    # Test summary returns a string
    summary_str = cube.summary()
    assert isinstance(summary_str, str)
    assert "shape" in summary_str

    # Test integrate
    total = cube.integrate(axis=(0,1,2))
    assert np.isscalar(total)

    # Test kspace/rspace round-trip
    cube_k = cube.kspace()
    assert cube_k.space == "reciprocal"
    cube_r = cube_k.rspace()
    assert cube_r.space == "real"
    assert cube_r == cube, "They differ"
    
#------------------#
# Pytest for CubeData with triclinic lattice
#------------------#
def test_cubedata_triclinic():
    # Random data
    nx, ny, nz = 6, 7, 5
    data = np.random.rand(nx, ny, nz)
    origin = np.array([0.0, 0.0, 0.0])

    # Triclinic lattice vectors
    vectors = np.array([[5.0, 1.0, 0.0],
                        [0.5, 6.0, 1.0],
                        [0.0, 1.0, 4.0]])

    cube = CubeData(data, origin, vectors)

    # Check shape and scalar property
    assert cube.shape == (nx, ny, nz)
    assert cube.is_scalar

    # Check coordinates
    coords = cube.coordinates
    assert coords.shape == (nx, ny, nz, 3)

    # Integrate along z-axis
    integral_z = cube.integrate(axis=2)
    assert integral_z.shape == (nx, ny)

    # k-space transform
    cube_k = cube.kspace()
    assert cube_k.space == "reciprocal"
    # Check that vectors are 3x3
    assert cube_k.vectors.shape == (3, 3)

    # r-space transform back
    cube_r = cube_k.rspace()
    assert cube_r.space == "real"
    # Should have same shape as original
    assert cube_r.data.shape == cube.data.shape

#------------------#
if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
