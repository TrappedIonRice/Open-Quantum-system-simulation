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
def up(m=0,cutoff=[2],N=1):
    '''
    generate the creation ladder operator acting on the mth phonon mode of a system of N modes
    Input: (m,cutoff,N)
    Parameters
    ----------
    m : int
        python index of the ion that the operator acts on
    cutoff : list of int
        cut off level of each phonon space 
    N : int
       total number of phonon spaces

    Returns
    -------
    Qutip Operator

    '''
    if N == 1: 
        lup = create(cutoff[0])
    else:
        for j in range(N):
            if j == m:
                nextop = create(cutoff[j])
            else:
                nextop = qeye(cutoff[j])
            if j == 0:
                lup = nextop
            else:
                lup = tensor(lup,nextop)
    return lup
def down(m=0,cutoff=[2],N=1):
    '''
    generate the annihilation ladder operator acting on the mth phonon mode of a system of N modes
    Input: (m,cutoff,N)
    Parameters
    ----------
    m : int
        python index of the ion that the operator acts on
    cutoff :  list of int
        cut off level of each phonon space 
    N : int
        total number of phonon spaces
    
    Returns
    -------
    Qutip Operator

    '''
    if N == 1: 
        ldown = destroy(cutoff[0])
    else:
        for j in range(N):
            if j == m:
                nextop = destroy(cutoff[j])
            else:
                nextop = qeye(cutoff[j])
            if j == 0:
               ldown = nextop
            else:
               ldown = tensor(ldown,nextop)
    return ldown
    
def zero_op(cutoff,N):
    '''
    generate the zero operator acting on the system of N modes
    Input: (cutoff,N)
    Parameters
    ----------
    cutoff :  list of int
        cut off level of each phonon space 
    N : int
        total number of phonon spaces
    
    Returns
    -------
    Qutip Operator

    '''
    mat = qzero(cutoff[0])
    for i in range(N-1):
        mat = tensor(mat,qzero(cutoff[i+1]))
    return mat    
def phip(cutoff=[2],N=1,nlist=[0]):
    '''
    compute phonon state of the system with phonon number specified by nlist
    Input: (cutoff,N,nlist)
    Parameters
    ----------
    cutoff :  list of int
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
    istate = basis(cutoff[0],nlist[0])
    for i in range(1,N):
        istate = tensor(istate, basis(cutoff[i],nlist[i]))
    return istate
def state_measure(cutoff=[2],N=1,meas_level=1,m=0):
    '''
    generate the operator to measure a single phonon state population for a specified
    phonon space
    cutoff :  list of int
        cut off level of each phonon space 
    N : int
        total number of phonon spaces
    slevel: int
        phonon state level to be measured    
    m: index of phonon space for measurement, default as 0
    '''
    m_ket = fock(cutoff[m],meas_level)
    dm_h = m_ket*m_ket.dag()
    if N == 1: 
        hm_op =  dm_h
    else:
        for j in range(N):
            if j == m:
                nextop = dm_h
            else:
                nextop = qeye(cutoff[j])
            if j == 0:
               hm_op  = nextop
            else:
               hm_op  = tensor(hm_op ,nextop)
    return hm_op 
def pI(cutoff,N):
    '''
    generate the identity operator acting on the system of N ions
    Input: (cutoff,N)
    Parameters
    ----------
    cutoff : list of int
        cut off level of each phonon space 
    N : int
       total number of phonon spaces
    
    Returns
    -------
    Qutip Operator

    '''
    Iden = qeye(cutoff[0])
    for i in range(N-1):
        Iden = tensor(Iden,qeye(cutoff[i+1]))
    return Iden    
  
def p_thermal(cutoff,nbar):
    '''
    generate the probability distribution following a canonical distrbution 
    with kT = Etot, harmonic energy frequency wm
    input(cutoff,N,wm,Etot)
    Parameters
    ----------
    cutoff : int
        cut off level of phonon space
    nbar : float
        average phonon number of the thermal state

    Returns
    -------
    np array, each element is the probability of a correponding fock state

    '''
    pdis = np.array([])
    for i in range(cutoff):
        pdis = np.append(pdis,(1/nbar + 1)**(-i))
    pdis = pdis/np.sum(pdis)
    return pdis    
def inip_thermal(cutoff=2,nbar=1,ket=False):
    '''
    generate the initial density matirx/pure quantum state ket for a single phonon space 
    with population following a thermal distribution 
    input(cutoff,N,wm,Etot)
    Parameters
    ----------
    cutoff : int
        cut off level of phonon space
    nbar : float
        average phonon number of the thermal state
    ket: bool, default as false
        if true, output state as ket for a pure superposition of fock states
        if false, output the usual density matrix used for thermal state
    Returns
    -------
    Qutip Operator

    '''
    pdis0 = p_thermal(cutoff,nbar)
    if ket:
        for n in range(cutoff):
            if n == 0:
                pket = np.sqrt(pdis0[0])*fock(cutoff,0)
            else:
                pket = pket + np.sqrt(pdis0[n])*fock(cutoff,n)
        return pket    
   
    else:
        dmta = np.zeros((cutoff,cutoff))
        for n in range(cutoff):
            dmta[n,n] = pdis0[n] 
        return Qobj(dmta)
   