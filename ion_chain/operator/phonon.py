# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:41:57 2022

@author: zhumj
"""
import numpy as np
from qutip import *
"""
Opeartors on phonon space
"""
def list_func():
    print("____________________________________________________________________")
    print("function: up")
    print("generate the creation ladder operator acting on the mth (python index) ion of system of N ions")
    print("Input: (m,clevel,N)")
    print("m: int, python index of the ion that the operator acts on")
    print("clevel: int, cut off  level of the harmonic ocsillator eigenenergy")
    print("N: int, number of ions in the system")
    print("Output: Qutip Operator")
    print("____________________________________________________________________")
    print("function: down")
    print("generate the annihilation ladder operator acting on the mth (python index) ion of system of N ions")
    print("Input: (m,clevel,N)")
    print("m: int, python index of the ion that the operator acts on")
    print("clevel: int, cut off  level of the harmonic ocsillator eigenenergy")
    print("N: int, number of ions in the system")
    print("Output: Qutip Operator")
    print("____________________________________________________________________")
    print("function: zero_op")
    print("generate zero operator on the phonon space")
    print("Input: (clevel,N)")
    print("clevel: int, cut off  level of the harmonic ocsillator eigenenergy")
    print("N: int, number of ions in the system")
    print("Output: Qutip Operator")
    print("____________________________________________________________________")
    print("function: pI")
    print("generate identity operator on N ion spin space")
    print("Input: (clevel,N)")
    print("clevel: int, cut off  level of the harmonic ocsillator eigenenergy")
    print("N: int, number of ions in the system")
    print("Output: Qutip Operator")
    print("____________________________________________________________________")
    print("function: phip")
    print("compute phonon state of the system with state energy specified by nlist")
    print("Input: (N,clevel,nlist)")
    print("clevel: int, cut off  level of the harmonic ocsillator eigenenergy")
    print("nlist, list of int, the specfic fock states that each ion is in")
    print("N: int, number of ions in the system")
    print("Output: Qutip ket")
def up(m,clevel,N):
    #input m is the index of ion being calculated, (python index) 
    #clevel is the cut off  level of the harmonic ocsillator eigenenergy, N
    #is the number of ions in the system
    if N == 1:
        lup = create(clevel)
    else:
        op_list = [qeye(clevel)]*N
        op_list[m] = create(clevel)
        for j in range(N):
            if j == 0:
                lup = op_list[j]
            else:
                lup = tensor(lup,op_list[j])
    return lup
def down(m,clevel,N):
    #input m is the index of ion being calculated, (python index) 
    #clevel is the cut off  level of the harmonic ocsillator eigenenergy, N
    #is the number of ions in the system
    if N == 1:
        ldown = destroy(clevel)
    else:
        op_list = [qeye(clevel)]*N
        op_list[m] = destroy(clevel)
        for j in range(N):
            if j == 0:
                ldown = op_list[j]
            else:
                ldown = tensor(ldown,op_list[j])
    return ldown   
def zero_op(clevel,N):
    #generate zero operator on N ion phonon space with 
    # cut off at clevel for initialization 
    mat = qzero(clevel)
    for i in range(N-1):
        mat = tensor(mat,qzero(clevel))
    return mat    
def phip(clevel,N,nlist):
    #compute initial phonon state with state energy specified by nlist
    istate = basis(clevel,nlist[0])
    for i in range(1,N):
        istate = tensor(istate, basis(clevel,nlist[i]))
    return istate
def pI(clevel,N):
    #compute the identity operator acting on the phonon states
    Iden = qeye(clevel)
    for i in range(N-1):
        Iden = tensor(Iden,qeye(clevel))
    return Iden    
def ini_dmt(clevel,N):
    dmta = np.zeros((clevel,clevel))
    for i in range(clevel):
        dmta[i,i] = np.exp(-(i+1)/n)
    dmta = dmta/np.trace(dmta)
    dmt0 = np.copy(dmta)
    dmt0[clevel-1,clevel-1] = 0
    dmta[clevel-1,clevel-1] = 1 - np.trace(dmt0)
    return Qobj(dmta/np.trace(dmta))
