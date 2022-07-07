# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022

@author: zhumj
 Construct Hamiltonian in reasonate rotating frame for the 3 ion open qunatum system used to simulate excitation transition between 2 molecules
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
from astropy.io import ascii
from astropy.table import Table
from qutip import *
import ion_chain.ising.ising_ps as iscp
#subfunction
def w(f):
    #convert frequency Hz to angular frequency rad/s
    return 2*np.pi*f
def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('___________________________________________________________________')
    print('U')
    print('compute the activation energy to reasonance conformation')
    print('___________________________________________________________________')
    print('Htot')
    print('construct Hamiltonian and collpase operators of the system in the reasonant rotating frame Parameters using a single mode, or double mode system')
    print('___________________________________________________________________')
    print('rho_ini')
    print('Construct initial density matrix according to a thermal distribution')

def Htot(J23, E2, E3, ion0 ):
    '''
    construct Hamiltonian and collpase operators of the system in the reasonant rotating frame
   
    Parameters 
    ----------
    J12 : float
       coupling between ion1 and ion2 [kHz]
    E1 : float
       site energy ion1 [kHz]  
    E2 : float
       site energy ion2 [kHz]      
    ion0: ions class object
        the object that represent the system to be simulated

    Returns
    -------
    H
        Qutip operator
    clist : list
        list of Qutip operators required by qutip solver
    '''
    Np = ion0.N #of ions to be considered for phonon space
    Ns = ion0.N-1 #of ions to be considered for spin space
    pcut = ion0.pcut
    dm = ion0.dmlist()
    #spin phonon coupling
    term1 =  tensor(spin.zero_op(Ns),phon.zero_op(pcut,Np)) 
    emat = iscp.Transmode(Np,ion0.fz,ion0.fx)
    coeff = iscp.eta(ion0.wmlist())
   # term1 = tensor(coeff*emat[2,0]*(spin.sz(Ns,0) + spin.sz(Ns,1)),
                   #(phon.up(2, ion0.pcut, Np)+phon.down(2, ion0.pcut, Np)))
             
    for i in range(Ns):
        subop = tensor(spin.zero_op(Ns),phon.zero_op(pcut,Np))
        for m in range(Np):
            eta_im = coeff[m]*emat[m,i]#(1/np.sqrt(3))
            subop = (subop +
                     0.5 * eta_im* ion0.Omega() * tensor(spin.sz(Ns,i),(phon.up(m, ion0.pcut, Np)+phon.down(m, ion0.pcut, Np)))) 
        term1 = term1 + subop  
    term2 = tensor(spin.zero_op(Ns),phon.zero_op(pcut,Np))
    for m in range(Np):
        term2 = term2 + dm[m]*tensor(spin.sI(Ns),phon.up(m, ion0.pcut, ion0.N)*phon.down(m, ion0.pcut, ion0.N))
    #phonnic mode
    sop3 = tensor(spin.up(Ns,0)*spin.down(Ns,1),phon.pI(pcut,Np))
    term3 = w(J23) * (sop3+sop3.dag())
    #vibrational harmonic oscillator potential
    term4 = (w(E2) * tensor(spin.sz(Ns,0),phon.pI(pcut,Np))+
             w(E3) * tensor(spin.sz(Ns,1),phon.pI(pcut,Np)))
    #coherent coupling of the donor and acceptor states
    c1 =  tensor(spin.sI(Ns), phon.down(1, ion0.pcut, ion0.N))
    c2 =  tensor(spin.sI(Ns), phon.down(2, ion0.pcut, ion0.N))
    #collapse operator
    H = term1-term2+term3+term4
    clist = []
    clist.append(np.sqrt(w(ion0.gamma[0])*(1+ion0.n_bar()))*c1)
    clist.append(np.sqrt(w(ion0.gamma[0])*ion0.n_bar())*c1.dag())
    clist.append(np.sqrt(w(ion0.gamma[1])*(1+ion0.n_bar()))*c2)
    clist.append(np.sqrt(w(ion0.gamma[1])*ion0.n_bar())*c2.dag())
    return H, clist
def rho_ini(ion0,single_mode):
    '''
    Construct initial density matrix according to a thermal distribution

    Parameters
    ----------
    ion0: ions class object
       the object that represent the system to be simulated
    single_mode : bool
       use COM mode only if true

    Returns
    -------
    Qutip operator

    '''
    isket = tensor(fock(2,0),fock(2,1)) # ion 1 in excited state
    ini_sdm = isket*isket.dag()
    pho0 = tensor(phon.inip_thermal(ion0.pcut,1000*w(ion0.wmlist()[0]),ion0.Etot),
                  phon.inip_thermal(ion0.pcut,1000*w(ion0.wmlist()[0]),ion0.Etot),
                  phon.inip_thermal(ion0.pcut,1000*w(ion0.wmlist()[0]),ion0.Etot))
    #dmat = fock(ion0.pcut,0)*fock(ion0.pcut,0).dag()
    #pho0 = tensor(dmat,dmat,dmat)
    rho0 = tensor(ini_sdm,pho0)
    return rho0    
        
    
    
