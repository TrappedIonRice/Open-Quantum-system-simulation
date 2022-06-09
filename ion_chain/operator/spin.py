# -*- coding: utf-8 -*-
"""
Operators on spin space
functions: sx, sy, sz, zero_op, sI, phid, phiup   
@author: zhumj
"""
import numpy as np
from qutip import *
def summary():
    print("____________________________________________________________________")
    print("function: sx,sy,sz")
    print("generate the sigmax,y,z operator acting on the ith (python index) spin 1/2 in the system of N ions")
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
def sx(N,i):
    '''
    generate the sigmax operator acting on the ith (python index) spin 1/2
    in the system of N ions
    Input: 
        N: int
            number of ions in the system
        i: int 
            python index of the ion that the operator acts on
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
def sy(N,i):
    '''
    generate the sigmay operator acting on the ith (python index) spin 1/2
    in the system of N ions
    Input: 
        N: int
            number of ions in the system
        i: int 
            python index of the ion that the operator acts on
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
def sz(N,i):
    '''
    generate the sigmaz operator acting on the ith (python index) spin 1/2
    in the system of N ions
    Input: 
        N: int
            number of ions in the system
        i: int 
            python index of the ion that the operator acts on
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
def zero_op(N):
    """
    generate zero operator on N ion spin space
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system

    Returns
    -------
    Qutip Operator

    """
    mat = qzero(2)
    for i in range(N-1):
        mat = tensor(mat,qzero(2))
    return mat    
def sI(N):
    """
    generate identity operator on N ion spin space
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system

    Returns
    -------
    mat : Qutip Operator

    """
    if N == 1:
        Iden = qeye(2)
    else:    
        Iden = qeye(N)
        for i in range(N-1):
            Iden = tensor(Iden,qeye(N))
    return Iden    
def phid(N):
    """
    construct a state with all N ions in spin down
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system

    Returns
    -------
    Qutip ket

    """
    istate = basis(2,0)
    for i in range(N-1):
        istate = tensor(istate, basis(2, 0))
    return istate
def phiup(N):
    """
    construct a state with all N ions in spin down
    input(N)
    Parameters
    ----------
    N : int
        number of ions in the system

    Returns
    -------
    Qutip ket
    """
    istate = basis(2,1)
    for i in range(N-1):
        istate = tensor(istate, basis(2, 1))
    return istate