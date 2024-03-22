# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:53:19 2023
Compute ET rate based on theoretical models
@author: zhumj
"""
from qutip import *
import numpy as np
import random
'''
functions for generating random diagonals
'''
def rand_diag(N):
    '''
    generate a N*N matrix with [0,1) uniform diagonal elements which sum up to 1

    Parameters
    ----------
    N : int
        dimension of the matrix

    Returns
    -------
    diag_mat : np array

    '''
    eps = np.random.rand(N-1)
    diag_mat = np.zeros([N,N])
    sum_d = 0
    for i in range(N-1):
        di = (1 - eps[i]**( 1/(N-i-1) )) * (1-sum_d) 
        diag_mat[i][i] = di
        sum_d += di
    diag_mat[N-1][N-1] = 1 - sum_d
    return diag_mat
def random_pure_diag(N):
    #pure state
    diag_mat = np.zeros([N,N])
    index = random.randint(0,N-1)
    diag_mat[index][index] = 1
    return diag_mat


def random_N_index(N,n):
    #get n non-repeating integers from 0~N-1
    tot_sample = list(range(0,N))
    index = random.sample(tot_sample, n)
    return index
def random_fm_diag(N,n):
    #state with fixed rank n
    non_zero_ele = np.diag(rand_diag(n))
    #assigne these elements
    posi = random_N_index(N,n)
    diag_mat = np.zeros([N,N])
    for i in range(len(posi)):
        diag_mat[posi[i]][posi[i]] = non_zero_ele[i]
    return diag_mat
def CUE_mat(N):
    '''
    Generate random unitary from circular unitary ensumble

    Parameters
    ----------
    N : int
         dimension of the matrix

    Returns
    -------
    umat : np array

    '''
    #
    rmat =  np.random.randn(N,N) + (0+1j)*np.random.randn(N,N)
    umat, _ = np.linalg.qr(rmat)
    return umat
def random_U(Ns,Nq):
    '''
    Generate random unitary quantum operator compatible with qutip

    Parameters
    ----------
    Ns : int
         dimension of each spin (Hilbert space)
    Nq : int
         number of qubits/ (subsystem)

    Returns
    -------
    Qutip operator

    '''
    N = Ns**Nq
    Nstruc = 2*[Nq*[Ns]]
    U = CUE_mat(N); 
    return Qobj(U,dims = Nstruc)

def random_pure(Ns,Nq,ket=False):
    N = Ns**Nq
    a = np.random.rand(N); b = np.random.rand(N)
    vec = (a+1j*b); nket = Qobj(vec/np.linalg.norm(vec))
    if ket:
        nket.dims = 2*[Nq*[Ns]]
        return nket
    else:
        rho = nket*nket.dag()
        rho.dims = 2*[Nq*[Ns]]
        return rho
        
    
    
def random_dm(Ns,Nq,mtype='random',n=0,sep=False):
    '''
    Generate random density matrix with specified features

    Parameters
    ----------
    Ns : int
        dimension of each spin (Hilbert space)
    Nq : int
        number of qubits/ (subsystem)
    mtype : str, optional
        Type of density matrix, allowed types:
            random: completely random
            pure: pure state
            rk: with assigned rank
            . The default is 'random'.
    n : int,
        rank of density matrix, apply if type = 'rk'
    sep : bool, 
        Generate separable states if True. The default is False.

    Returns
    -------
    dm : Qutip operator

    '''
    N = Ns**Nq
    Nstruc = 2*[Nq*[Ns]]
    if mtype == 'random':
        D = rand_diag(N)
    elif mtype == 'pure':
        D = random_pure_diag(N)
    elif mtype == 'rk':
        D = random_fm_diag(N,n)
    else:
        print('incorrect type of matrix')
    U = CUE_mat(N); 
    if sep:
        mat = D
    else:
        mat = U@D@np.conjugate(U.transpose())
    dm = Qobj(mat,dims = Nstruc)
    return dm

def ent_neg(rho,pt_index):
    '''
    Compute entanglement negativity

    Parameters
    ----------
    rho : qutip operator
        density matrix 
    pt_index : list
        index of sub Hilbert space to be transposed
    Returns
    -------
    N : float

    '''
    mask = np.zeros(len(rho.dims[0]))
    mask[pt_index] = 1
    rho_pt = partial_transpose(rho, mask)
    #N = ((rho_pt.dag() * rho_pt).sqrtm().tr().real - 1)/2.0
    min_e = np.min(rho_pt.eigenenergies())
    if min_e < 0:
        return -2*min_e
    else:
        return 0
def Linear_E_norm(rho):
    '''
    Compute normalized linear entropy

    Parameters
    ----------
    rho : qutip operator
        density matrix 

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    N = rho.shape[0]
    return N/(N-1) * entropy_linear(rho)
