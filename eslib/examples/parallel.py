import os
import numpy as np
from eslib.classes.trajectory import AtomicStructures
from concurrent.futures import ProcessPoolExecutor
from icecream import ic

# Create output directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("data-tmp", exist_ok=True)

NBINS = 1000
Efields = ["0.10"]
z_axis = np.array([0, 0, 1])

def process_file(efield, n):
    """Process each (efield, n) pair and save the result."""
    file = f"pos/E={efield}/nve.n={n}.h5"
    ic(file)

    # Load the trajectory data and process positions
    traj = AtomicStructures.from_file(file=file)
    pos = traj.get("positions")
    sh = pos.shape
    pos = pos.reshape((sh[0], -1, 3, 3))  # snapshots, molecules, atom in molecule, xyz

    # Extract atomic positions
    O = pos[:, :, 0, :]  # Oxygen positions
    H1 = pos[:, :, 1, :]  # First hydrogen positions
    H2 = pos[:, :, 2, :]  # Second hydrogen positions

    # Compute the vector and normalize it
    vector = ((H1 - O) + (H2 - O)) / 2
    vector = vector / np.linalg.norm(vector, axis=2, keepdims=True)
    ic(vector.shape)

    # Save the computed vectors to a temporary file
    np.save(f"data-tmp/vectors.E={efield}.n={n}.npy", vector)
    return vector

def main():
    # Parallel processing for each (efield, n) pair
    for efield in Efields:
        all_vectors = [None] * 10

        # Using ProcessPoolExecutor to parallelize the processing of different 'n' values
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, efield, n) for n in range(10)]

            for i, future in enumerate(futures):
                all_vectors[i] = future.result()

        # Save the accumulated results for the current efield
        np.save(f"data/vectors.E={efield}.npy", np.asarray(all_vectors))

if __name__ == "__main__":
    main()
