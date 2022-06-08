# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 14:52:25 2022

@author: zhumj
"""
import numpy as np
from qutip import *
"""
Opeartors on spin space
"""
def list_func():
    print("____________________________________________________________________")
    print("function: sx,sy,sz")
    print("generate the sigmax,y,z operator acting on the ith (python index) spin 1/2 in the system of N ions")
    print("Input: (N,i)") 
    print("N: int, number of ions in the system")
    print("i: int, python index of the ion that the operator acts on")
    print("Output: Qutip Operator")
    print("____________________________________________________________________")
    print("function: zero_op")
    print("generate zero operator on N ion spin space")
    print("Input: (N)") 
    print("N: int, number of ions in the system")
    print("Output: Qutip Operator")
    print("____________________________________________________________________")
    print("function: sI")
    print("generate identity operator on N ion spin space")
    print("Input: (N)")
    print("N: int, number of ions in the system")
    print("Output: Qutip Operator")
    print("____________________________________________________________________")
    print("function: phid")
    print("construct a state with all N ions in spin down")
    print("Input: (N)")
    print("N: int, number of ions in the system")
    print("Output: Qutip ket")
    print("____________________________________________________________________")
    print("function: phiup")
    print("construct a state with all N ions in spin up")
    print("Input: (N)")
    print("N: int, number of ions in the system")
    print("Output: Qutip ket")
def sx(N,i):
    '''
    generate the sigmax operator acting on the ith (python index) spin 1/2
    in the system of N ions
    Input: 
        N: int, number of ions in the system
        i: int, python index of the ion that the operator acts on
    Output: Qutip Operator    
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
    #generate the sigmay operator acting on the ith (python index) spin 1/2
    # in the system of N ions
    op_list = [qeye(2)]*N
    op_list[i] = sigmay()
    for m in range(N):
        if m == 0:
            opsy = op_list[m]
        else:
            opsy = tensor(opsy, op_list[m])
    return opsy
def sz(N,i):
    #generate the sigmaz operator acting on the ith (python index) spin 1/2
    # in the system of N ions
    op_list = [qeye(2)]*N
    op_list[i] = sigmaz()
    for m in range(N):
        if m == 0:
            opsz = op_list[m]
        else:
            opsz = tensor(opsz, op_list[m])
    return opsz
def zero_op(N):
    #generate zero operator on N ion spin space for initialization 
    mat = qzero(2)
    for i in range(N-1):
        mat = tensor(mat,qzero(2))
    return mat    
def sI(N):
    #compute the identity operator acting on the spin states
    if N == 1:
        Iden = qeye(2)
    else:    
        Iden = qeye(N)
        for i in range(N-1):
            Iden = tensor(Iden,qeye(N))
    return Iden    
def phid(N):
    #construct initial state, spin down
    #compute initial spin state
    istate = basis(2,0)
    for i in range(N-1):
        istate = tensor(istate, basis(2, 0))
    return istate
def phiup(N):
    #compute initial spin state
    istate = basis(2,1)
    #print(istate)
    for i in range(N-1):
        istate = tensor(istate, basis(2, 1))
        #print(istate)
    return istate