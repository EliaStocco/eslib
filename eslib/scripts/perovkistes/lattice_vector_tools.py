"""Shared lattice-cell helpers for the perovskite analysis scripts."""

from __future__ import annotations

from itertools import permutations

import numpy as np


PERMUTATIONS = tuple(permutations(range(3)))
SIGN_CHOICES = ((1.0, 1.0, 1.0), (1.0, 1.0, -1.0),
                (1.0, -1.0, 1.0), (1.0, -1.0, -1.0))


def cellpar_from_cell(cell: np.ndarray) -> np.ndarray:
    """Return ``a, b, c, alpha, beta, gamma`` for a row-vector cell matrix."""
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    metric = cell @ cell.T
    lengths = np.sqrt(np.diag(metric))
    if np.any(lengths <= 0.0):
        raise ValueError("cell contains a zero-length lattice vector")
    cosines = (
        metric[1, 2] / (lengths[1] * lengths[2]),
        metric[0, 2] / (lengths[0] * lengths[2]),
        metric[0, 1] / (lengths[0] * lengths[1]),
    )
    angles = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return np.concatenate((lengths, angles))


def _cellpar_from_metric(metric: np.ndarray) -> np.ndarray:
    lengths = np.sqrt(np.diag(metric))
    cosines = (
        metric[1, 2] / (lengths[1] * lengths[2]),
        metric[0, 2] / (lengths[0] * lengths[2]),
        metric[0, 1] / (lengths[0] * lengths[1]),
    )
    return np.concatenate((lengths, np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))))


def _metric_from_cellpar(cellpar: np.ndarray) -> np.ndarray:
    a, b, c, alpha, beta, gamma = np.asarray(cellpar, dtype=float)
    alpha, beta, gamma = np.deg2rad((alpha, beta, gamma))
    return np.array(
        ((a * a, a * b * np.cos(gamma), a * c * np.cos(beta)),
         (a * b * np.cos(gamma), b * b, b * c * np.cos(alpha)),
         (a * c * np.cos(beta), b * c * np.cos(alpha), c * c)),
        dtype=float,
    )


def canonical_transform(
    cellpar: np.ndarray,
    *,
    length_tolerance: float = 1.0e-4,
    angle_tolerance: float = 1.0e-3,
    right_angle_tolerance: float = 1.0e-2,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Return a vector permutation and sign convention for one cell.

    The shortest/medium/longest-vector convention fixes labels.  Equivalent
    signed bases are then selected with an all-acute rhombohedral convention.
    For an orthorhombic cell with alpha observably obtuse and beta/gamma right
    angles, that convention is preserved instead.
    """
    value = np.asarray(cellpar, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("one finite six-component cell-parameter row is required")
    if np.any(value[:3] <= 0.0) or np.any((value[3:] <= 0.0) | (value[3:] >= 180.0)):
        raise ValueError("cell parameters are outside their physical ranges")
    if min(length_tolerance, angle_tolerance, right_angle_tolerance) <= 0.0:
        raise ValueError("all canonicalization tolerances must be positive")
    metric = _metric_from_cellpar(value)
    if np.linalg.eigvalsh(metric)[0] <= 0.0:
        raise ValueError("cell metric is not positive definite")

    def permuted(order):
        return _cellpar_from_metric(metric[np.ix_(order, order)])

    def order_key(candidate):
        return (
            tuple(np.rint(candidate[:3] / length_tolerance).astype(np.int64)),
            tuple(np.rint(candidate[3:] / angle_tolerance).astype(np.int64)),
            tuple(candidate),
        )

    order = min(((order, permuted(order)) for order in PERMUTATIONS), key=lambda item: order_key(item[1]))[0]
    ordered_metric = metric[np.ix_(order, order)]
    unsigned = _cellpar_from_metric(ordered_metric)
    offset = unsigned[3:] - 90.0
    if offset[0] > right_angle_tolerance and np.all(np.abs(offset[1:]) <= right_angle_tolerance):
        return order, SIGN_CHOICES[0]

    def sign_key(signs):
        signed_metric = ordered_metric * np.outer(signs, signs)
        angles = _cellpar_from_metric(signed_metric)[3:]
        return (
            int(np.count_nonzero(angles > 90.0 + angle_tolerance)),
            tuple(np.rint(angles / angle_tolerance).astype(np.int64)),
            tuple(angles),
        )

    return order, min(SIGN_CHOICES, key=sign_key)


def canonical_cellpar(cellpar: np.ndarray, **kwargs) -> np.ndarray:
    """Return canonical cell parameters without changing the physical metric."""
    order, signs = canonical_transform(cellpar, **kwargs)
    metric = _metric_from_cellpar(cellpar)[np.ix_(order, order)]
    return _cellpar_from_metric(metric * np.outer(signs, signs))


def signed_permutation_matrix(
    order: tuple[int, int, int], signs: tuple[float, float, float]
) -> np.ndarray:
    """Return ``A`` such that ``A @ cell`` reorders and signs cell rows."""
    matrix = np.zeros((3, 3), dtype=float)
    for new_axis, (old_axis, sign) in enumerate(zip(order, signs)):
        matrix[new_axis, old_axis] = sign
    return matrix


def alignment_transform(cell: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
    """Build a fixed signed-basis and proper-rotation transform for one run.

    For an original row-vector cell ``H``, the returned matrices satisfy
    ``H_reference = A @ H @ R``.  ``A`` relabels the lattice-vector rows and
    ``R`` is a proper Cartesian rotation.  Cartesian row vectors, including
    positions and dipoles, are aligned as ``v_reference = v @ R``.
    """
    cell = np.asarray(cell, dtype=float)
    cellpar = cellpar_from_cell(cell)
    order, signs = canonical_transform(cellpar, **kwargs)
    signs = tuple(signs)
    basis = signed_permutation_matrix(order, signs) @ cell
    # A simultaneous sign reversal leaves the lattice metric unchanged.  Choose
    # it, when needed, so the reference mapping is a proper rotation.
    if np.linalg.det(basis) < 0.0:
        signs = tuple(-sign for sign in signs)
        basis = signed_permutation_matrix(order, signs) @ cell
    reference = np.linalg.cholesky(basis @ basis.T)
    rotation = np.linalg.solve(basis, reference)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-8):
        raise RuntimeError("could not construct an orthogonal cell-alignment rotation")
    if np.linalg.det(rotation) <= 0.0:
        raise RuntimeError("cell-alignment transform is not a proper rotation")
    return {
        "order": np.asarray(order, dtype=int),
        "signs": np.asarray(signs, dtype=float),
        "A": signed_permutation_matrix(order, signs),
        "R": rotation,
        "reference_cell": reference,
    }


def permute_components(values: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    """Relabel lengths, angles, or their standard deviations using ``order``."""
    pair_to_angle = {(1, 2): 3, (0, 2): 4, (0, 1): 5}
    index = (*order,
             pair_to_angle[tuple(sorted((order[1], order[2])))],
             pair_to_angle[tuple(sorted((order[0], order[2])))],
             pair_to_angle[tuple(sorted((order[0], order[1])))])
    return np.asarray(values)[..., index]
