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
def spin_state(N=1,config=[0]):
    """
    construct a state with all N ions, the spin
    of each ion can be specified 
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system, N > 1
    config: list of int
        specify the spin configuration, 0 for up and 1 for down 
    Returns
    -------
    Qutip ket
    """
    if N == 1:
        isket = fock(2,config[0])
    else:    
        isket = fock(2,config[0])
        for i in range(1,N):
            isket = tensor(isket,fock(2,config[i])) 
    return isket        