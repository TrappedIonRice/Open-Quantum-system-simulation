# -*- coding: utf-8 -*-
"""
Operators on phonon space
functions: up, down, zero_op, pI, phip, inip_thermal   
@author: zhumj
"""
import numpy as np
from qutip import *
def summary():
    print("____________________________________________________________________")
    print("function: up")
    print("generate the creation ladder operator acting on the mth (python index) ion of system of N ions")
    print("____________________________________________________________________")
    print("function: down")
    print("generate the annihilation ladder operator acting on the mth (python index) ion of system of N ions")
    print("____________________________________________________________________")
    print("function: zero_op")
    print("generate zero operator on the phonon space")
    print("____________________________________________________________________")
    print("function: pI")
    print("generate identity operator on N ion spin space")
    print("____________________________________________________________________")
    print("function: phip")
    print("compute phonon state of the system with state energy specified by nlist")
    print("____________________________________________________________________")
    print("function: inip_thermal") 
    print("generate the initial density operator for the phonon space of 1 ion composed with pure states with population following a thermal distribution ")
def up(m,clevel,N):
    '''
    generate the creation ladder operator acting on the mth (python index) ion of system of N ions
    Input: (m,clevel,N)
    Parameters
    ----------
    m : int
        python index of the ion that the operator acts on
    clevel : list of int
        cut off level of each phonon space 
    N : int
       total number of phonon spaces

    Returns
    -------
    Qutip Operator

    '''
    if N == 1: 
        lup = create(clevel[0])
    else:
        for j in range(N):
            if j == m:
                nextop = create(clevel[j])
            else:
                nextop = qeye(clevel[j])
            if j == 0:
                lup = nextop
            else:
                lup = tensor(lup,nextop)
    return lup
def down(m,clevel,N):
    '''
    generate the annihilation ladder operator acting on the mth (python index) ion of system of N ions
    Input: (m,clevel,N)
    Parameters
    ----------
    m : int
        python index of the ion that the operator acts on
    clevel :  list of int
        cut off level of each phonon space 
    N : int
        total number of phonon spaces
    
    Returns
    -------
    Qutip Operator

    '''
    if N == 1: 
        ldown = destroy(clevel[0])
    else:
        for j in range(N):
            if j == m:
                nextop = destroy(clevel[j])
            else:
                nextop = qeye(clevel[j])
            if j == 0:
               ldown = nextop
            else:
               ldown = tensor(ldown,nextop)
    return ldown
    
def zero_op(clevel,N):
    '''
    generate the zero operator acting on the system of N ions
    Input: (clevel,N)
    Parameters
    ----------
    clevel :  list of int
        cut off level of each phonon space 
    N : int
        total number of phonon spaces
    
    Returns
    -------
    Qutip Operator

    '''
    mat = qzero(clevel[0])
    for i in range(N-1):
        mat = tensor(mat,qzero(clevel[i+1]))
    return mat    
def phip(clevel,N,nlist):
    '''
    compute phonon state of the system with state energy specified by nlist
    Input: (clevel,N,nlist)
    Parameters
    ----------
    clevel :  list of int
        cut off level of each phonon space 
    N : int
        total number of phonon spaces
    nlist : list of int
        the specfic fock states that each ion is in

    Returns
    -------
    Qutip Ket

    '''
    #compute initial phonon state with state energy specified by nlist
    istate = basis(clevel[0],nlist[0])
    for i in range(1,N):
        istate = tensor(istate, basis(clevel[i],nlist[i]))
    return istate
def state_measure(clevel,N,slevel,m=0):
    '''
    generate the operator to measure a single phonon state population for a specified
    phonon space
    clevel :  list of int
        cut off level of each phonon space 
    N : int
        total number of phonon spaces
    slevel: int
        phonon state level to be measured    
    m: index of phonon space for measurement, default as 0
    '''
    m_ket = fock(clevel[m],slevel)
    dm_h = m_ket*m_ket.dag()
    if N == 1: 
        hm_op =  dm_h
    else:
        for j in range(N):
            if j == m:
                nextop = dm_h
            else:
                nextop = qeye(clevel[j])
            if j == 0:
               hm_op  = nextop
            else:
               hm_op  = tensor(hm_op ,nextop)
    return hm_op 
def pI(clevel,N):
    '''
    generate the identity operator acting on the system of N ions
    Input: (clevel,N)
    Parameters
    ----------
    clevel : list of int
        cut off level of each phonon space 
    N : int
       total number of phonon spaces
    
    Returns
    -------
    Qutip Operator

    '''
    Iden = qeye(clevel[0])
    for i in range(N-1):
        Iden = tensor(Iden,qeye(clevel[i+1]))
    return Iden    

def p_thermal(clevel,wm,Etot):
    '''
    generate the probability distribution following a canonical distrbution 
    with kT = Etot, harmonic energy frequency wm
    input(clevel,N,wm,Etot)
    Parameters
    ----------
    clevel : int
        cut off level of phonon space
    wm : float
        energy frequency of the phonon states MHz
    Etot : float
        total energy of the ion

    Returns
    -------
    np array, each element is the probability of a correponding fock state

    '''
    pdis = np.array([])
    for i in range(clevel):
        pdis = np.append(pdis,np.exp(-(i+0.5)*wm/Etot))
    pdis = pdis/np.sum(pdis)
    return pdis     
def inip_thermal(clevel,wm,Etot,ket=False):
    '''
    generate the initial density matirx/pure quantum state ket for a single phonon space 
    with population following a thermal distribution 
    input(clevel,N,wm,Etot)
    Parameters
    ----------
    clevel : int
        cut off level of phonon space
    wm : float
        energy frequency of the phonon states
    Etot : float
        total energy of the ion
    ket: bool, default as false
        if true, output state as ket for a pure superposition of fock states
        if false, output the usual density matrix used for thermal state
    Returns
    -------
    Qutip Operator

    '''
    pdis0 = p_thermal(clevel,wm,Etot)
    if ket:
        for n in range(clevel):
            if n == 0:
                pket = np.sqrt(pdis0[0])*fock(clevel,0)
            else:
                pket = pket + np.sqrt(pdis0[n])*fock(clevel,n)
        return pket    
   
    else:
        dmta = np.zeros((clevel,clevel))
        for n in range(clevel):
            dmta[n,n] = pdis0[n] 
        return Qobj(dmta)
def inik_thermal(clevel,wm,Etot):
    '''
    generate a pure quantum state ket for a single phonon space 
    with population following a thermal distribution 
    input(clevel,N,wm,Etot)
    Parameters
    ----------
    clevel : int
        cut off level of phonon space
    wm : float
        energy frequency of the phonon states
    Etot : float
        total energy of the ion
    
    Returns
    -------
    Qutip Operator
    
    '''
    pdis0 = p_thermal(clevel,wm,Etot)
   