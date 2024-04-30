#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:18:04 2021

@author: george

Tools for transforming into the Eckart frame, see
https://en.wikipedia.org/wiki/Eckart_conditions
DOI: 10.1063/1.1929739
DOI: 10.1063/1.4870936
"""

from __future__ import print_function
from __future__ import division
import numpy as np


def qua2mat(q):
    r"""Return the rotation matrix corresponding to the unit
    quaternion q. The matrix is transposed relative to
    https://en.wikipedia.org/wiki/Rotation_matrix
    so that the rotated vector is given by x @ A. We use
    the convention that the first element of the quaternion
    vector is its scalar part, and the remaining three are
    the imaginary components.

    Parameters
    ----------
    q : ndarray, shape (4,) or (M,4)
        A quaternion or stack of M quaternions. The algorithm
        performs normalisation automatically if needed.

    Returns
    -------
    rotmat : ndarray, shape (3,3) or (M,3,3)
        A rotation matrix or stack of M rotation matrices.

    """

    q0 = q[..., :1]  # Real part of the quaternion(s)
    iq = q[..., 1:]  # Imaginary part of the quaternion(s)
    # Cross-product of the imaginary part of the quaternion
    rotmat = iq[..., None, :] * iq[..., :, None]
    # A view of the main diagonal
    diag_view = np.einsum('...ii -> ...i', rotmat)
    # Squared norm of the imaginary part
    imtrace = np.sum(diag_view, axis=-1)
    diag_view -= np.expand_dims(imtrace, axis=-1)
    rotmat -= (q0[..., None, :] *
               np.cross(np.eye(3), iq[..., None, :], axisb=-1))

    rotmat *= 2
    normq2 = np.sum(q*q, axis=-1)
    rotmat /= normq2[..., None, None]
    rotmat += np.eye(3)

    return rotmat


def principal_rot(angle, axis="X"):
    r"""
    Return matrix for rotating about one of the principal
    axes ('X', 'Y' or 'Z').
    """
    axdict = dict(X=0, Y=1, Z=2)
    try:
        ind = axdict[axis.upper()]
    except:
        raise RuntimeError("'axis' must be one of 'X', 'Y', 'Z'")

    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.diag(3*[c, ])
    mat[ind,ind] = 1.0
    i = (ind+1)%3
    j = (ind+2)%3
    mat[i,j] = -s
    mat[j,i] = s
    return mat


def euler2mat(angles, order='ZXZ'):
    r"""
    Return the matrix corresponding to rotations about axes
    as specified in `order`.
    """

    assert len(angles) == 3
    mats = []
    for angle, axis in zip(angles, order):
        mats.append(principal_rot(angle, axis=axis))
    return np.linalg.multi_dot(mats)


class EckartFrame(object):
    r"""
    Rotate a molecule about its center of mass, minimising the
    mass-weighted square norm Sum[ m_k | X_k - R x_k |^2 ],
    enforcing rotational Eckart conditions according to
    the algorithm from 10.1063/1.4870936
    """

    def __init__(self, mass=[1.0, ]):
        r"""
        Initialise the Eckart frame

        Parameters
        ----------
        m : ndarray of shape (N,), OPTIONAL
            Atomic masses. Default = [1.0, ]

        """
        self.mass = np.expand_dims(
            np.atleast_1d(mass).copy(), axis=-1)
        assert self.mass.ndim == 2
        self.mtot = np.sum(self.mass)
        self.sqm = np.sqrt(self.mass)
        self.sqm_outer = self.sqm * self.sqm[:, 0]
        self.sqm_outer /= np.sum(self.mass)

    def shift_to_com(self, x):
        r"""
        Shift the input configuration to the centre-of-mass frame

        Parameters
        ----------
        x : ndarray, shape (N,3) or (M,N,3)
            Atomic configuration comprised of N atoms
            (or a stack of M such configurations).

        Returns
        -------
        shifted : ndarray, shape (N,3) or (M,N,3)
            The shifted configuration
        com : ndarray, shape (3,) or (M,3)
            The CoM of the original configuration in the lab frame

        """
        CoM = np.sum(self.mass*x, axis=-2)/self.mtot
        return x - CoM[..., None, :], CoM

    def conditions(self, x, xref):
        r"""
        Returns the sum of cross-products that is zero when the
        Eckart conditions are satisfied.

        Parameters
        ----------
        x, xref : ndarrays, shape (N,3) or (M,N,3)

        Returns
        -------
        ans : ndarray, shape (3,) or (M,3)

        """

        # Shift instantaneous and reference to com of
        # instantaneous configuration
        xs, com = self.shift_to_com(x)
        mx = xs*self.sqm
        xe = xref - com[..., None, :]
        mxe = xe*self.sqm
        return np.cross(mxe, mx).sum(axis=-2)

    def get_cmat(self, mx, mxe):
        r"""
        Construct the matrix C from (24) in 10.1063/1.4870936

        Parameters
        ----------
        mx, mxe : ndarray, shape (N,3) or (M,N,3)
            Mass-weighted instantaneous and reference
            configurations in a reference frame with
            the CoM of x at the origin

        Returns
        -------
        cmat : ndarray, shape (4,4) or (M,4,4)
            The matrix whose eigenvectors extremise the Eckart norm

        """
        # Calculate sum and difference coordinates
        x_sum = mxe + mx
        x_dif = mxe - mx

        # Calculate the upper triangle of the C-matrix
        cmat = np.zeros(x_sum.shape[:-2] + (4, 4), dtype=float)
        # Diagonal elements
        x2_sum = np.sum(x_sum**2, axis=-2)
        x2_dif = np.sum(x_dif**2, axis=-2)
        diag_view = np.einsum('...ii -> ...i', cmat)
        diag_view[..., 0] = np.sum(x2_dif, axis=-1)
        diag_view[..., 1:] = np.sum(x2_sum[..., None, :], axis=-1)
        diag_view[..., 1:] += x2_dif
        diag_view[..., 1:] -= x2_sum
        # Off-diagonal
        cmat[..., 0, 1] = np.sum(x_sum[..., 1]*x_dif[..., 2] -
                                 x_dif[..., 1]*x_sum[..., 2], axis=-1)
        cmat[..., 0, 2] = np.sum(x_sum[..., 2]*x_dif[..., 0] -
                                 x_dif[..., 2]*x_sum[..., 0], axis=-1)
        cmat[..., 0, 3] = np.sum(x_sum[..., 0]*x_dif[..., 1] -
                                 x_dif[..., 0]*x_sum[..., 1], axis=-1)

        cmat[..., 1, 2] = np.sum(x_dif[..., 0]*x_dif[..., 1] -
                                 x_sum[..., 0]*x_sum[..., 1], axis=-1)
        cmat[..., 1, 3] = np.sum(x_dif[..., 0]*x_dif[..., 2] -
                                 x_sum[..., 0]*x_sum[..., 2], axis=-1)
        cmat[..., 2, 3] = np.sum(x_dif[..., 1]*x_dif[..., 2] -
                                 x_sum[..., 1]*x_sum[..., 2], axis=-1)
        return cmat

    def grad_cmat(self, mx, mxe):
        r"""
        Construct the derivative of the C-matrix w.r.t. the
        mass-weighted instantaneous and reference configuration

        Parameters
        ----------
        mx, mxe : ndarray, shape (N,3) or (M,N,3)
            Mass-weighted instantaneous and reference
            configurations in a reference frame with CoM
            of x at the origin

        Returns
        -------
        dcmat_dx : ndarray, shape (N,3,4,4) or (M,N,3,4,4)
            Derivative of the C-matrix w.r.t. mass-weighted
            instantaneous configuration
        dcmat_dxe : ndarray, shape (N,3,4,4) or (M,N,3,4,4)
            Derivative of the C-matrix w.r.t. mass-weighted
            reference configuration

        """
        xi_sum = mxe + mx
        xi_dif = mxe - mx
        dcmat_dx = np.zeros(xi_sum.shape+(4, 4))
        dcmat_dxe = np.zeros_like(dcmat_dx)
        dcmat_dxis = np.zeros_like(dcmat_dx)
        dcmat_dxid = np.zeros_like(dcmat_dx)

        xs, ys, zs = xi_sum[..., 0], xi_sum[..., 1], xi_sum[..., 2]
        xd, yd, zd = xi_dif[..., 0], xi_dif[..., 1], xi_dif[..., 2]

        # ---- Calculate the derivative w.r.t. xi_sum ---- #
        # x-derivative
        gradx = dcmat_dxis[..., 0, :, :]

        gradx[..., 0, 2] = -zd
        gradx[..., 2, 0] = gradx[..., 0, 2]
        gradx[..., 0, 3] = yd
        gradx[..., 3, 0] = gradx[..., 0, 3]

        gradx[..., 1, 2] = -ys
        gradx[..., 2, 1] = gradx[..., 1, 2]
        gradx[..., 1, 3] = -zs
        gradx[..., 3, 1] = gradx[..., 1, 3]

        gradx[..., 2, 2] = 2*xs
        gradx[..., 3, 3] = gradx[..., 2, 2]

        # y-derivative
        grady = dcmat_dxis[..., 1, :, :]

        grady[..., 0, 1] = zd
        grady[..., 1, 0] = grady[..., 0, 1]
        grady[..., 0, 3] = -xd
        grady[..., 3, 0] = grady[..., 0, 3]

        grady[..., 1, 1] = 2*ys
        grady[..., 1, 2] = -xs
        grady[..., 2, 1] = grady[..., 1, 2]

        grady[..., 2, 3] = -zs
        grady[..., 3, 2] = grady[..., 2, 3]

        grady[..., 3, 3] = grady[..., 1, 1]

        # z-derivative
        gradz = dcmat_dxis[..., 2, :, :]

        gradz[..., 0, 1] = -yd
        gradz[..., 1, 0] = gradz[..., 0, 1]
        gradz[..., 0, 2] = xd
        gradz[..., 2, 0] = gradz[..., 0, 2]

        gradz[..., 1, 1] = 2*zs
        gradz[..., 1, 3] = -xs
        gradz[..., 3, 1] = gradz[..., 1, 3]

        gradz[..., 2, 2] = gradz[..., 1, 1]
        gradz[..., 2, 3] = -ys
        gradz[..., 3, 2] = gradz[..., 2, 3]

        # ---- Calculate the derivative w.r.t. xi_dif ---- #
        # x-derivative
        gradx = dcmat_dxid[..., 0, :, :]

        gradx[..., 0, 0] = 2*xd

        gradx[..., 0, 2] = zs
        gradx[..., 2, 0] = gradx[..., 0, 2]
        gradx[..., 0, 3] = -ys
        gradx[..., 3, 0] = gradx[..., 0, 3]

        gradx[..., 1, 1] = gradx[..., 0, 0]
        gradx[..., 1, 2] = yd
        gradx[..., 2, 1] = gradx[..., 1, 2]
        gradx[..., 1, 3] = zd
        gradx[..., 3, 1] = gradx[..., 1, 3]

        # y-derivative
        grady = dcmat_dxid[..., 1, :, :]

        grady[..., 0, 0] = 2*yd
        grady[..., 0, 1] = -zs
        grady[..., 1, 0] = grady[..., 0, 1]
        grady[..., 0, 3] = xs
        grady[..., 3, 0] = grady[..., 0, 3]

        grady[..., 1, 2] = xd
        grady[..., 2, 1] = grady[..., 1, 2]

        grady[..., 2, 2] = grady[..., 0, 0]
        grady[..., 2, 3] = zd
        grady[..., 3, 2] = grady[..., 2, 3]

        # z-derivative
        gradz = dcmat_dxid[..., 2, :, :]

        gradz[..., 0, 0] = 2*zd
        gradz[..., 0, 1] = ys
        gradz[..., 1, 0] = gradz[..., 0, 1]
        gradz[..., 0, 2] = -xs
        gradz[..., 2, 0] = gradz[..., 0, 2]

        gradz[..., 1, 3] = xd
        gradz[..., 3, 1] = gradz[..., 1, 3]

        gradz[..., 2, 3] = yd
        gradz[..., 3, 2] = gradz[..., 2, 3]

        gradz[..., 3, 3] = gradz[..., 0, 0]

        # ---- Calculate the derivative w.r.t. x ---- #
        dcmat_dx[:] = dcmat_dxis - dcmat_dxid  # type: ignore
        dcmat_dx[:] -= 2*np.einsum('sk,...kjtu->...sjtu',
                                   self.sqm_outer, dcmat_dxis)

        # ---- Calculate the derivative w.r.t. xref ---- #
        dcmat_dxe[:] = dcmat_dxis + dcmat_dxid

        return dcmat_dx, dcmat_dxe

    def hess_ip_cmat(self, mx, mxe, q):
        r"""
        Construct the hessian of the C-matrix w.r.t. the
        mass-weighted instantaneous and reference configuration
        and contract it twice with the 4-vector q.

        Parameters
        ----------
        mx, mxe : ndarray, shape (N,3) or (M,N,3)
            Mass-weighted instantaneous and reference
            configurations in a reference frame with
            CoM of x at the origin
        q : ndarray, shape(4) or (M,4)
            A unit quaternion with which to contract the hessian

        Returns
        -------
        d2cmat_dx2, d2cmat_drdx, d2cmat_dr2 : ndarrays,
        shape (N,3,N,3) or (M,N,3,N,3)
            Double derivatives of the C-matrix w.r.t mass-weighted
            input, input and reference, and reference
            configurations, contracted with the quaternion q.

        """

        N = mx.shape[-2]

        # We make use of the sparsity and symmetry of the
        # second derivative matrix to compute the inner product

        # Contributions complementary to the CoM
        # (non-zero only for derivatives w.r.t. same atom)
        qr, qi, qj, qk = (q[..., i:i+1] for i in range(4))
        qr2 = 2*qr*qr
        qi2 = 2*qi*qi
        qj2 = 2*qj*qj
        qk2 = 2*qk*qk

        qir = 2*qi*qr
        qjr = 2*qj*qr
        qkr = 2*qk*qr

        qij = 2*qi*qj
        qik = 2*qi*qk
        qjk = 2*qj*qk

        d2cmat_dxe2 = np.zeros(q.shape[:-1]+(N, 3, N, 3))
        hessrr = np.einsum('...ijij->...ij', d2cmat_dxe2)
        hessrr[..., 0] = qr2 + qi2 + qj2 + qk2
        hessrr[..., 1] = hessrr[..., 0]
        hessrr[..., 2] = hessrr[..., 0]

        d2cmat_dx2 = d2cmat_dxe2.copy()

        d2cmat_dxedx = np.zeros_like(d2cmat_dxe2)
        hessrx = np.einsum('...ijik->...ijk', d2cmat_dxedx)
        hessrx[..., 0, 0] = -qi2 + qj2 + qk2 - qr2
        hessrx[..., 0, 1] = -2*(qij + qkr)
        hessrx[..., 0, 2] = -2*(qik - qjr)

        hessrx[..., 1, 0] = -2*(qij - qkr)
        hessrx[..., 1, 1] = qi2 - qj2 + qk2 - qr2
        hessrx[..., 1, 2] = -2*(qjk + qir)

        hessrx[..., 2, 0] = -2*(qik + qjr)
        hessrx[..., 2, 1] = -2*(qjk - qir)
        hessrx[..., 2, 2] = qi2 + qj2 - qk2 - qr2

        # Contributions involving the CoM
        # (couple different atoms)
        w = 2*self.sqm_outer

        qmat = np.zeros(q.shape[:-1]+(3, 3))
        qmat[..., 0, 0] = (-qj2 - qk2)[..., 0]
        qmat[..., 0, 1] = (qij + qkr)[..., 0]
        qmat[..., 0, 2] = (qik - qjr)[..., 0]
        qmat[..., 1, 0] = (qij - qkr)[..., 0]
        qmat[..., 1, 1] = (-qi2 - qk2)[..., 0]
        qmat[..., 1, 2] = (qjk + qir)[..., 0]
        qmat[..., 2, 0] = (qik + qjr)[..., 0]
        qmat[..., 2, 1] = (qjk - qir)[..., 0]
        qmat[..., 2, 2] = (-qi2 - qj2)[..., 0]

        d2cmat_dxedx += np.einsum('ij,...kl->...ikjl', w, qmat)

        w2 = np.einsum('ij,jk->ik', w, w)
        qmat[..., 0, 1] = qij[..., 0]
        qmat[..., 1, 0] = qij[..., 0]
        qmat[..., 0, 2] = qik[..., 0]
        qmat[..., 2, 0] = qik[..., 0]
        qmat[..., 1, 2] = qjk[..., 0]
        qmat[..., 2, 1] = qjk[..., 0]

        d2cmat_dx2 += np.einsum('ij,...kl->...ikjl', 2*w-w2, qmat)

        return d2cmat_dx2, d2cmat_dxedx, d2cmat_dxe2

    def norm2(self, x, xref):
        r"""
        Rotate the input configuration into the Eckart frame

        Parameters
        ----------
        x, xref : ndarray, shape (N,3) or (M,N,3)

        Returns
        -------
        xrot : ndarray, shape (N,3) or (M,N,3)
            Rotated molecular configuration(s).
        rotmat : ndarray, shape (3,3) or (M,3,3)
            The matrices that accomplish the rotation,
            `xrot = x @ rotmat`.
        norm2 : float or ndarray, shape (M,)
            Mass-weighted square norm Sum[ m_k | X_k - x_k |^2 ],
            where X is the reference configuration and
            x is the Eckart-rotated input.

        """
        # Shift config to CoM
        x, com = self.shift_to_com(x)
        xe = xref - com[..., None, :]
        mx = x*self.sqm  # mass-weighted config
        mxe = xe*self.sqm
        cmat = self.get_cmat(mx, mxe)
        w, v = np.linalg.eigh(cmat, UPLO='U')
        norm2 = w[..., 0]
        if norm2.ndim == 0:
            norm2 = norm2.item()
        q = v[..., 0]
        q[..., 1:] *= -1  # take inverse of quaternion
        rotmat = qua2mat(q)
        xrot = np.einsum('...ki,...ij->...kj', x, rotmat)
        return xrot + com[..., None, :], rotmat, norm2

    def norm2grad(self, x, xref):
        r"""
        Calculate the analytical derivative of the square norm
        with respect to the instantaneous and reference configurations.

        Parameters
        ----------
        x, xref : ndarrays, shape (N,3) or (M,N,3)

        Returns
        -------
        xrot : ndarray, shape (N,3) or (M,N,3)
            Rotated molecule.
        rotmat : ndarray, shape (3,3) or (M,3,3)
            Matrices that accomplish the rotation via x @ rotmat.
        norm2 : float or ndarray, shape (M,)
            Mass-weighted square norm
        jacx, jacr : ndarrays, shape (3*N,) or (M,3*N)
            Derivatives of the mass-weighted square norm w.r.t.
            the instantaneous and reference configurations.

        """

        x, com = self.shift_to_com(x)
        xe = xref - com[..., None, :]
        # mass-weighted config
        mx = x*self.sqm
        mxe = xe*self.sqm
        cmat = self.get_cmat(mx, mxe)
        # Diagonalise the matrix
        w, v = np.linalg.eigh(cmat, UPLO='U')
        norm2 = w[..., 0]
        if norm2.ndim == 0:
            norm2 = norm2.item()
        q = v[..., 0]  # we only want the eigenvector with the lowest e-value
        qinv = q.copy()
        qinv[..., 1:] *= -1
        rotmat = qua2mat(qinv)
        xrot = np.einsum('...ki,...ij->...kj', x, rotmat)
        dcmat_dx, dcmat_dxe = self.grad_cmat(mx, mxe)
        jacx = np.einsum('...i,...ij,...j->...',
                         q[..., None, None, :], dcmat_dx,
                         q[..., None, None, :])
        jacr = np.einsum('...i,...ij,...j->...',
                         q[..., None, None, :], dcmat_dxe,
                         q[..., None, None, :])
        for arr in [jacr, jacx]:
            arr *= self.sqm
            arr.shape = arr.shape[:-2]+(-1,)

        return xrot+com[..., None, :], rotmat, norm2, jacx, jacr

    def norm2hess(self, x, xref):
        r"""
        Calculate the analytical derivative and hessian
        of the square norm to the instantaneous and reference configurations

        Parameters
        ----------
        x, xref : ndarray, shape (N,3) or (M,N,3)

        Returns
        -------
        xrot : ndarray, shape (N,3) or (M,N,3)
            Rotated molecular configuration(s).
        rotmat : ndarray, shape (3,3) or (M,3,3)
            Matrices that accomplish the rotation via x @ rotmat.
        norm2 : float or ndarray, shape (M,)
            Mass-weighted square norm
        jacx, jacr : ndarrays, shape (3*N,) or (M,3*N)
            Derivatives of the mass-weighted square norm w.r.t.
            the reference and instantaneous configurations
            respectively
        hessxx, hessxr, hessrr : ndarrays, shape (..., 3*N, 3*N)
            Double derivatives of the mass-weighted square norm

        """
        x, com = self.shift_to_com(x)
        xe = xref - com[..., None, :]
        # mass-weighted config
        mx = x*self.sqm
        mxe = xe*self.sqm
        N = x.shape[-2]

        # ------ Calculate Eckart transform ------ #
        cmat = self.get_cmat(mx, mxe)
        # Diagonalise the matrix
        w, v = np.linalg.eigh(cmat, UPLO='U')
        norm2 = w[..., 0]
        if norm2.ndim == 0:
            norm2 = norm2.item()
        q = v[..., 0]
        qinv = q.copy()
        qinv[..., 1:] *= -1
        rotmat = qua2mat(qinv)
        xrot = np.einsum('...ki,...ij->...kj', x, rotmat)

        # ------ Calculate Jacobian ------ #
        dcmat_dx, dcmat_dxe = self.grad_cmat(mx, mxe)
        jacx = np.einsum('...i,...ij,...j->...',
                         q[..., None, None, :], dcmat_dx,
                         q[..., None, None, :])
        jacr = np.einsum('...i,...ij,...j->...',
                         q[..., None, None, :], dcmat_dxe,
                         q[..., None, None, :])
        for arr in [jacr, jacx]:
            arr *= self.sqm
            arr.shape = arr.shape[:-2]+(-1,)

        # ------ Calculate Hessian ------- #
        hessxx, hessrx, hessrr = self.hess_ip_cmat(mx, mxe, q)
        old_shape = hessxx.shape
        for arr in (hessxx, hessrx, hessrr):
            arr.shape = dcmat_dxe.shape[:-4]+(3*N, 3*N)

        # First derivative of C-matrix and eigenvector
        dcmat_dxe.shape = dcmat_dxe.shape[:-4]+(-1, 4, 4)
        dcdxeq = np.einsum('...ij,...j',
                           dcmat_dxe, v[..., None, :, 0])
        dcmat_dx.shape = dcmat_dxe.shape
        dcdxq = np.einsum('...ij,...j',
                          dcmat_dx, v[..., None, :, 0])
        # Weighted eigenvectors, shape (M,4,3)
        qw = (v[..., 1:] /
              np.sqrt(w[..., 1:] - w[..., :1])[..., None, :])
        # ... dotted to give shape (M,3*N,3)
        dcdxeqq = np.einsum('...jk,...j',
                            qw[..., None, :, :], dcdxeq)
        dcdxqq = np.einsum('...jk,...j',
                           qw[..., None, :, :], dcdxq)
        #
        hessrr -= 2*np.einsum('...ik,...jk->...ij',
                              dcdxeqq, dcdxeqq)
        hessrx -= 2*np.einsum('...ik,...jk->...ij',
                              dcdxeqq, dcdxqq)
        hessxx -= 2*np.einsum('...ik,...jk->...ij',
                              dcdxqq, dcdxqq)
        # Mass weighting:
        for hess in (hessrr, hessrx, hessxx):
            rhess = np.reshape(hess, old_shape)
            rhess *= self.sqm
            rhess *= self.sqm[..., None, None]

        return (xrot + com[..., None, :], rotmat, norm2,
                jacx, jacr, hessxx, hessrx, hessrr)

    def align(self,x,xref):
        r"""
        Align some atomic configurations 'x' along the Eckart frame defined by 'xref'.
        
        Parameters
        ----------
        x    : ndarray, shape (M,N,3)
        xref : ndarray, shape (N,3)

        Returns
        -------
        xrot: ndarray, shape (M,N,3)
            Rotated and shifted molecular configuration(s).
        shift: ndarray, shape (M,3)
            Shift vectors
        rotmat: ndarray, shape (M,3,3)
            Rotation matrices
        """
        # get the c.o.m. 'com' of the coordinates 'x'
        xs, com = self.shift_to_com(x)
        # xs + com[..., None, :] == x --> True
        # get the c.o.m. 'refcom' of 'xref'
        xref, refcom = self.shift_to_com(xref)
        # rotate 'x' into the Eckart frame
        xrot, rotmat, _  = self.norm2(xs,xref)
        # put the c.o.m. of 'xrot' to zero
        # and then shift 'xrot' s.t. its c.o.m. is the same of 'xref'
        # sm    = -com   [..., None, :] # shift minus      
        # sp    = +refcom[..., None, :] # shift plus
        # shift = sm + sp   
        com = com[..., None, :] 
        xrot += com
        return xrot, com, rotmat

if __name__ == "__main__":

    def test_eckart(m, x, xref):
        eck = EckartFrame(m)
        print("Constraint function before rotation:")
        print(eck.conditions(x, xref))
        xrot, rotmat, norm2 = eck.norm2(x, xref)
        print("Constraint function after rotation:")
        print(eck.conditions(xrot, xref))  # Should be all zero
        print("Mass-weighted square norm:")
        print(norm2)
        print("Explicit:")
        print(np.sum(eck.mass*(xref-xrot)**2, axis=(-1, -2)))
        print("\n\n")

    np.set_printoptions(precision=3, suppress=True)
    print("Testing Eckart rotation")
    print("Single structure:")
    xref = np.array([[0.0, 0.0, 0.06615931],
                     [0.0, 0.75813308, -0.52499806],
                     [0.0, -0.75813308, -0.52499806]])
    x = xref.copy()
    x[0, 0:2] += 0.5
    shift = 1.0
    xref += shift
    x += shift
    m = np.array([15.99491502]+2*[1.00782522])
    test_eckart(m, x, xref)

    print("Try a bunch:")
    # Random rotation matrix to re-orient the reference
    rotmat = np.array(
        [[-0.74854572, -0.0837799,  0.65776913],
          [0.58108933,  0.39494159,  0.7115872],
          [-0.31939709,  0.91487817, -0.24694823]])
    xref = np.stack((xref, np.dot(xref, rotmat)))
    x = np.stack((x, x+np.random.uniform(-0.5, 0.5, size=x.shape)))
    test_eckart(m, x, xref)

    print("Hydronium:")
    xref = np.asarray([
        [-0.4826100501190497826797809466370,
          -0.1016665972237958126545009918118,
          -0.9105748657342833674022131162928],
        [0.86229426524710128809658726822818,
          0.31093727956030681180976671384997,
          -0.3463985580518358364798814363894],
        [-0.0406318400928327369547865544064,
          -0.5930572466063595005181241504033,
          0.91617383274285868477448957492015],
        [-0.0213633820274226997326660892895,
          0.02418204263762236544410555438844,
          0.02147347252025687963650213418987]
    ])
    x = np.asarray([
        [-0.45507457790102101569118531187996,
          -0.19789612476283194308912527503708,
          -1.88352961464804846691833972727181],
        [0.85163383292011818337385875565815,
          0.22365921742564565111166530186892,
          -0.36283325050737386119337202217139],
        [-0.05559921239360114675331914213530,
          -0.52130041761787648812997986169648,
          0.93006015747818460148721442237729],
        [-0.72148358242863856668036781627507,
          0.03122335651169665104998074411924,
          0.01992994616160740500854231527228],
    ])
    m = np.array(3*[1.00782522]+[15.99491502])
    test_eckart(m, x, xref)
