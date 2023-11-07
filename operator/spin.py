# -*- coding: utf-8 -*-
"""
Operators on spin space
functions: sx, sy, sz, sry, zero_op, sI, phid, phiup, spin_state   
@author: zhumj
"""
import numpy as np
from qutip import *
from qutip.qip.operations import ry
def summary():
    print("____________________________________________________________________")
    print("function: sx,sy,sz")
    print("generate the sigmax,y,z operator acting on the ith (python index) spin 1/2 in the system of N ions")
    print("____________________________________________________________________")
    print("function: up,down")
    print("generate the sigma+- operator acting on the ith (python index) spin 1/2 in the system of N ions")
    print("____________________________________________________________________")
    print("function: zero_op")
    print("generate zero operator on N ion spin space")
    print("____________________________________________________________________")
    print("function: sI")
    print("generate identity operator on N ion spin space")
    print("____________________________________________________________________")
    print("function: phid")
    print("construct a state with all N ions in spin down")
    print("____________________________________________________________________")
    print("function: phiup")
    print("construct a state with all N ions in spin up")
# spin eigenstates, 0 for eigenvalue 1 and 1 for eigenvalue of -1
sxket_0 = 1/np.sqrt(2) * ( basis(2,0) + basis(2,1) )
sxket_1 = 1/np.sqrt(2) * ( basis(2,0) - basis(2,1) )

syket_0 = 1/np.sqrt(2) * ( basis(2,0) + 1j*basis(2,1) )
syket_1 = 1/np.sqrt(2) * ( basis(2,0) - 1j*basis(2,1) )

szket_0 = basis(2,0)
szket_1 = basis(2,1)

spin_dic = {'x0': sxket_0, 'x1': sxket_1,
            'y0': syket_0, 'y1': syket_1,
            'z0': szket_0, 'z1': szket_1} 

#functions to use

def sx(N=1,i=0):
    '''
    generate the sigmax operator acting on the ith (python index) spin 1/2
    in the system of N ions
    Input: 
        N: int
            number of ions in the system, N > 1
        i: int 
            python index of the ion that the operator acts on, from 0 to N-1
    Output:
        Qutip Operator    
    '''
    op_list = [qeye(2)]*N
    op_list[i] = sigmax()
    for m in range(N):
        if m == 0:
            opsx = op_list[m]
        else:
            opsx = tensor(opsx, op_list[m])
    return opsx
def sy(N=1,i=0):
    '''
    generate the sigmay operator acting on the ith (python index) spin 1/2
    in the system of N ions
    Input: 
        N: int
            number of ions in the system, N > 1
        i: int 
            python index of the ion that the operator acts on, from 0 to N-1
    Output:
        Qutip Operator    
    '''
    op_list = [qeye(2)]*N
    op_list[i] = sigmay()
    for m in range(N):
        if m == 0:
            opsy = op_list[m]
        else:
            opsy = tensor(opsy, op_list[m])
    return opsy
def sz(N=1,i=0):
    '''
    generate the sigmaz operator acting on the ith (python index) spin 1/2
    space in the system of N ions
    Input: 
        N: int
            number of ions in the system, N > 1
        i: int 
            python index of the ion that the operator acts on, from 0 to N-1
    Output:
        Qutip Operator    
    '''
    op_list = [qeye(2)]*N
    op_list[i] = sigmaz()
    for m in range(N):
        if m == 0:
            opsz = op_list[m]
        else:
            opsz = tensor(opsz, op_list[m])
    return opsz
def sry(N=1,i=0,phi=0):
    '''
    generate the y rotation operator acting on the ith (python index) spin 1/2
    space in the system of N ions

    Parameters
    ----------
    N: int
        number of ions in the system, N > 1
    i: int 
        python index of the ion that the operator acts on, from 0 to N-1
    phi : float
        angle for rotation [rad]

    Output:
        Qutip Operator    
    '''
    op_list = [qeye(2)]*N
    op_list[i] = ry(phi)
    for m in range(N):
        if m == 0:
            opsry = op_list[m]
        else:
            opsry = tensor(opsry, op_list[m])
    return opsry
def up(N=1,i=0):
    '''
    generate the sigma+ operator acting on the ith (python index) spin 1/2
    in the system of N ions
    Input: 
        N: int
            number of ions in the system
        i: int 
            python index of the ion that the operator acts on
    Output:
        Qutip Operator    
    '''
    return 0.5*(sx(N,i)+1j*sy(N,i))
def down(N=1,i=0):
    '''
    generate the sigma- operator acting on the ith (python index) spin 1/2
    in the system of N ions
    Input: 
        N: int
            number of ions in the system
        i: int 
            python index of the ion that the operator acts on
    Output:
        Qutip Operator    
    '''
    return 0.5*(sx(N,i)-1j*sy(N,i))
def zero_op(N=1):
    """
    generate zero operator on N ion spin space
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system, N > 1

    Returns
    -------
    Qutip Operator

    """
    mat = qzero(2)
    for i in range(N-1):
        mat = tensor(mat,qzero(2))
    return mat    
def sI(N=1):
    """
    generate identity operator on N ion spin space
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system, N > 1

    Returns
    -------
    mat : Qutip Operator

    """
    if N == 1:
        Iden = qeye(2)
    else:    
        Iden = qeye(2)
        for i in range(N-1):
            Iden = tensor(Iden,qeye(2))
    return Iden    
def phid(N):
    """
    construct a state with all N ions in spin down
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system, N > 1

    Returns
    -------
    Qutip ket

    """
    istate = basis(2,1)
    for i in range(N-1):
        istate = tensor(istate, basis(2, 1))
    return istate
def phiup(N):
    """
    construct a state with all N ions in spin up
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system, N > 1

    Returns
    -------
    Qutip ket
    """
    istate = basis(2,0)
    for i in range(N-1):
        istate = tensor(istate, basis(2, 0))
    return istate
def spin_state(config = ['z0']):
    """
    construct a direct product state with all N ions, the spin
    of each ion can be specified as x,y,z (+1-1)eigenstate
    input(N)
    Parameters
    ----------
    config: list of str
        specify the spin configuration, each string is in form 'i0', 'i1'
        with i being x,y,z, 0 for eigenvalue 1 (up ) and 1 for eigenvalue -1 
        (down)
        for instance, ['x0','z1'] gives the direct product state of
        1/sqrt[2] (|0>+|1>) and |1>
    Returns
    -------
    Qutip ket
    """
    N = len(config)
    if N == 1:
        isket = spin_dic[config[0]]
    else:    
        isket = spin_dic[config[0]]
        for i in range(1,N):
            isket = tensor(isket,spin_dic[config[i]]) 
    return isket
def ammt_measure(state,N):
    '''
    Compute total angulare momentum vector for a ensamble of N spin 1/2

    Parameters
    ----------
    state : Qutip ket/density matrix
        state of the spin system
    N: int
        Number of spins in the system 
    Returns
    -------
    np array of float, [<Sx>,<Sy>,<Sz>]
    '''  
    ammt = np.zeros(3) 
    for i in range(N):
        sx_e = 0.5*expect(sx(N,i),state)
        sy_e = 0.5*expect(sy(N,i),state)
        sz_e = 0.5*expect(sz(N,i),state)
        ammt += np.array([sx_e,sy_e,sz_e])
    return ammt
def MSD(state,N):
    '''
    Compute MSD mean spin direction for a N spin system 

    Parameters
    ----------
    state : Qutip ket/density matrix
        state of the spin system
    N: int
        Number of spins in the system 

    Returns
    -------
    np array of float, [nx,ny,nz]
    
    '''
    ammt = ammt_measure(state,N)
    return ammt/np.linalg.norm(ammt)
def Jn_operator(n_vec,N):
    '''
    Compute operator Jn acting on spin-boson system

    Parameters
    ----------
    n_vec : np.array
        unit vector specifying the direction of Jn
    N: int
        Number of spins in the system 
    p_I : Qutip operator
        Identity operator of phonon space
    Returns
    -------
    float

    '''
    Jx = zero_op(N)
    Jy = zero_op(N)
    Jz = zero_op(N)
    for i in range(N):
        Jx += sx(N,i)
        Jy += sy(N,i)
        Jz += sz(N,i)
    Jn = 0.5*(n_vec[0]*Jx + n_vec[1]*Jy+ n_vec[2]*Jz)
    return Jn