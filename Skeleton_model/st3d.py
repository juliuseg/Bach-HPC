#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:30:17 2019

@author: vand@dtu.dk
"""

import numpy as np
import scipy.io
import scipy.ndimage
from joblib import Parallel, delayed

#% STRUCTURE TENSOR 3D
#% The function is made parallel by s204427


def compute_derivative(volume, sigma, order):
    return scipy.ndimage.gaussian_filter(volume, sigma, order=order, mode='nearest')

def compute_tensor_component(name, Vx, Vy, Vz, rho):
    if name == "Jxx":
        return scipy.ndimage.gaussian_filter(Vx**2, rho, mode='nearest')
    elif name == "Jyy":
        return scipy.ndimage.gaussian_filter(Vy**2, rho, mode='nearest')
    elif name == "Jzz":
        return scipy.ndimage.gaussian_filter(Vz**2, rho, mode='nearest')
    elif name == "Jxy":
        return scipy.ndimage.gaussian_filter(Vx * Vy, rho, mode='nearest')
    elif name == "Jxz":
        return scipy.ndimage.gaussian_filter(Vx * Vz, rho, mode='nearest')
    elif name == "Jyz":
        return scipy.ndimage.gaussian_filter(Vy * Vz, rho, mode='nearest')

def structure_tensor(volume, sigma, rho, n_jobs=1):
    """
    Fully parallel version of structure tensor for 3D image data.
    Computes spatial derivatives and structure tensor components in parallel.
    
    Args:
        volume: 3D numpy array
        sigma: noise smoothing scale
        rho: integration scale
        n_jobs: number of parallel threads (default -1 = all)
    
    Returns:
        Structure tensor as a (6, N) array
    """
    # Step 1: Compute smoothed spatial derivatives in parallel
    orders = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # x, y, z
    Vx, Vy, Vz = Parallel(n_jobs=n_jobs)(
        delayed(compute_derivative)(volume, sigma, order) for order in orders
    )

    # Step 2: Compute structure tensor components in parallel
    components = ["Jxx", "Jyy", "Jzz", "Jxy", "Jxz", "Jyz"]
    tensors = Parallel(n_jobs=n_jobs)(
        delayed(compute_tensor_component)(comp, Vx, Vy, Vz, rho) for comp in components
    )

    # Step 3: Stack and flatten into (6, N)
    S = np.vstack([comp.ravel() for comp in tensors])
    return S

def eig_special(S, full=False):
    """ Eigensolution for symmetric real 3-by-3 matrices
    Arguments:
        S: an array with shape (6,N) containing structure tensor
        full: a flag indicating that all three eigenvalues should be returned
    Returns:
        val: an array with shape (3,N) containing sorted eigenvalues
        vec: an array with shape (3,N) containing eigenvector corresponding to 
            the smallest eigenvalue. If full, vec has shape (6,N) and contains 
            all three eigenvectors 
    More:        
        An analytic solution of eigenvalue problem for real symmetric matrix,
        using an affine transformation and a trigonometric solution of third
        order polynomial. See https://en.wikipedia.org/wiki/Eigenvalue_algorithm
        which refers to Smith's algorithm https://dl.acm.org/citation.cfm?id=366316
    Author: vand@dtu.dk, 2019
    """    
    # TODO -- deal with special cases, decide treatment of full (i.e. maybe return 2 for full)
    # computing eigenvalues
    s = S[3]**2 + S[4]**2 + S[5]**2 # off-diagonal elements
    q = (1/3)*(S[0]+S[1]+S[2]) # mean of on-diagonal elements
    p = np.sqrt((1/6)*(np.sum((S[0:3] - q)**2, axis=0) + 2*s)) # case p==0 treated below 
    p_inv = np.zeros(p.shape)
    p_inv[p!=0] = 1/p[p!=0] # to avoid division by 0
    B = p_inv * (S - np.outer(np.array([1,1,1,0,0,0]),q))  # B represents a 3-by-3 matrix, A = pB+2I   
    d = B[0]*B[1]*B[2] + 2*B[3]*B[4]*B[5] - B[3]**2*B[2]\
            - B[4]**2*B[1] - B[5]**2*B[0] # determinant of B
    phi = np.arccos(np.minimum(np.maximum(d/2,-1),1))/3 # min-max to ensure -1 <= d/2 <= 1 
    val = q + 2*p*np.cos(phi.reshape((1,-1))+np.array([[2*np.pi/3],[4*np.pi/3],[0]])) # ordered eigenvalues

    # computing eigenvectors -- either only one or all three
    if full:
        l = val
    else:
        l=val[0]
            
    u = S[4]*S[5]-(S[2]-l)*S[3]
    v = S[3]*S[5]-(S[1]-l)*S[4]
    w = S[3]*S[4]-(S[0]-l)*S[5]
    vec = np.vstack((u*v, u*w, v*w)) # contains one or three vectors
   
    # normalizing -- depends on number of vectors
    if full: # vec is [x1 x2 x3 y1 y2 y3 z1 z2 z3]
        vec = vec[[0,3,6,1,4,7,2,5,8]] # vec is [v1, v2, v3]
        l = np.sqrt(np.vstack((np.sum(vec[0:3]**2,axis=0), np.sum(vec[3:6]**2,\
                axis=0), np.sum(vec[6:]**2, axis=0))))
        vec = vec/l[[0,0,0,1,1,1,2,2,2]] # division by 0 should not occur
    else: # vec is [x1 y1 z1] = v1
        # vec = vec/np.sqrt(np.sum(vec**2, axis=0))
        norms = np.sqrt(np.sum(vec**2, axis=0))
        norms[norms == 0] = 1  # avoid division by zero
        vec = vec / norms
    return val,vec
