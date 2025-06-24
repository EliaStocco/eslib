import torch
import math

def compute_ewald_sum(positions, charges, box_length, alpha, real_cutoff, recip_cutoff, device='cpu'):
    """
    Compute the Coulomb interaction energy between charged particles in a cubic periodic box
    using Ewald summation.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Cartesian coordinates of N particles in 3D.
    charges : torch.Tensor, shape (N,)
        Electric charges of each particle.
    box_length : float
        Length of one side of the cubic simulation box.
    alpha : float
        Ewald damping/splitting parameter controlling the decay of real-space terms.
    real_cutoff : float
        Cutoff radius for real-space sum. Only pairs closer than this contribute in real space.
    recip_cutoff : float
        Cutoff on squared reciprocal lattice vector magnitude |k|^2 for reciprocal-space sum.
    device : str, optional (default 'cpu')
        Device for tensor computation ('cpu' or 'cuda').

    Returns
    -------
    energy : torch.Tensor (scalar)
        Total Coulomb energy computed by Ewald summation.

    Notes
    -----
    - Assumes a cubic box with periodic boundary conditions.
    - Positions should be within [0, box_length).
    - Charges can be positive or negative.
    - Real and reciprocal cutoffs control accuracy and performance.
    """
    N = positions.shape[0]
    volume = box_length ** 3

    # Real-space sum
    energy_real = torch.tensor(0., device=device)
    for i in range(N):
        for j in range(i+1, N):
            rij = positions[i] - positions[j]
            # Apply minimum image convention for PBC
            rij = rij - box_length * torch.round(rij / box_length)
            r2 = torch.dot(rij, rij)
            if r2 < real_cutoff ** 2:
                r = torch.sqrt(r2)
                erfc_term = torch.erfc(alpha * r)
                energy_real += charges[i] * charges[j] * erfc_term / r

    # Reciprocal-space sum
    max_k = int(math.ceil(math.sqrt(recip_cutoff)))
    energy_recip = torch.tensor(0., device=device, dtype=torch.float64)
    for nx in range(-max_k, max_k+1):
        for ny in range(-max_k, max_k+1):
            for nz in range(-max_k, max_k+1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                k_vec = (2 * math.pi / box_length) * torch.tensor([nx, ny, nz], device=device, dtype=torch.float64)
                k2 = torch.dot(k_vec, k_vec)
                if k2 <= recip_cutoff:
                    # Structure factor S(k) = sum_j q_j exp(-i k · r_j)
                    phase = -1j * torch.matmul(positions.double(), k_vec)
                    S_k = torch.sum(charges.double() * torch.exp(phase))
                    contrib = (torch.exp(-k2 / (4 * alpha ** 2)) / k2) * (S_k.abs() ** 2)
                    energy_recip += contrib

    energy_recip = (2 * math.pi / volume) * energy_recip

    # Self-interaction correction
    energy_self = -alpha / math.sqrt(math.pi) * torch.sum(charges.double() ** 2)

    # Total energy
    energy_total = energy_real.double() + energy_recip + energy_self
    return energy_total.float()

def example_usage():
    """
    Example of using the compute_ewald_sum function.
    Creates random charges and positions in a box and computes the Coulomb energy.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    N = 10
    box_length = 10.0

    # Random positions in box [0, box_length)
    positions = torch.rand(N, 3, device=device) * box_length

    # Random charges with zero net charge (optional)
    charges = torch.randn(N, device=device)
    charges -= charges.mean()  # Neutralize total charge

    # Ewald parameters
    alpha = 0.5
    real_cutoff = 5.0
    recip_cutoff = 20.0

    energy = compute_ewald_sum(positions, charges, box_length, alpha, real_cutoff, recip_cutoff, device=device)
    print(f"Ewald Coulomb energy: {energy.item():.6f} (in arbitrary units)")

if __name__ == "__main__":
    example_usage()
