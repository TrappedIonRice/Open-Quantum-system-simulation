# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022
Compute the Hamiltonian in reasonant interaction frame of 3 ion open qunatum system
used to simulate excitation transition between 2 sites
function: Htot
@author: zhumj
"""

import matplotlib.pyplot as plt
import numpy as np
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
from qutip import *
from  ion_chain.ising.ion_system import *
#subfunction

def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('___________________________________________________________________')
    print('Htot')
    print('construct Hamiltonian and collpase operators for 3 ion excitation transfer in the reasonant rotating frame Parameters ')
    print('___________________________________________________________________')
    print('rho_ini')
    print('Construct initial density matrix according to a thermal distribution')

def Htot(J12, E1, E2, Vx, ion0, config):
    '''
    construct Hamiltonian in reasonant rotating frame and collpase operators of 3 ion system 
    used for simulating excitation transfer between 2 sites
    Input
    ----------
    J12 : float
       coupling between ion1 and ion2 [kHz]
    E1 : float
       site energy ion1 [kHz]  
    E2 : float
       site energy ion2 [kHz]      
    Vx：
       rabi rate Omegax [kHz] 
    
    ion0: ions class object
        the object that represent the system to be simulated
    config: integer
        the configuration of the chain, if config = 0, cool the ion on the side
        if config = 1, cool the ion in the center.
    Returns
    -------
    H
        Qutip operator
        Hamiltonian in reasonant rotating frame
    clist : list
        list of Qutip operators required by qutip solver
        collapse operators to describe coupling to the evironment
    '''
    Np = ion0.N #of ions to be considered for phonon space
    Ns = ion0.N-1 #of ions to be considered for spin space
    pcut = ion0.pcut
    dm = ion0.dmlist()
    #spin phonon coupling
    term1 =  tensor(spin.zero_op(Ns),phon.zero_op(pcut,Np)) 
    emat = ion0.Transmode()
    coeff = eta(ion0.wmlist())
    if config == 0:
        ilist = [0,1]
    else:
        ilist = [0,2]             
    sindex = 0 #this index is used for spin operators    
    for i in ilist:
        subop = tensor(spin.zero_op(Ns),phon.zero_op(pcut,Np))
        for m in range(Np):
            #print(i,m)
            #print(emat[m,i])
            eta_im = coeff[m]*emat[m,i]#(1/np.sqrt(3))
            subop = (subop +
                     0.5 * eta_im* ion0.Omega() * 
                     tensor(spin.sz(Ns,sindex),(phon.up(m, ion0.pcut, Np)+phon.down(m, ion0.pcut, Np))
                                                         )
                     ) 
        term1 = term1 + subop
        sindex = sindex + 1
    #extra term from the transformation to special frame
    term2 = tensor(spin.zero_op(Ns),phon.zero_op(pcut,Np))
    for m in range(Np):
        term2 = term2 + dm[m]*tensor(spin.sI(Ns),phon.up(m, ion0.pcut, ion0.N)*phon.down(m, ion0.pcut, ion0.N))
    #phonnic mode
    sop3 = tensor(spin.up(Ns,0)*spin.down(Ns,1),phon.pI(pcut,Np))
    term3 = fr_conv(J12,'hz') * (sop3+sop3.dag())
    #vibrational harmonic oscillator potential
    term4 = (fr_conv(E1,'hz') * tensor(spin.sz(Ns,0),phon.pI(pcut,Np))+
             fr_conv(E2,'hz') * tensor(spin.sz(Ns,1),phon.pI(pcut,Np)))
    #coherent coupling of the donor and acceptor states
    term5 = (fr_conv(Vx,'hz') * 
             tensor(spin.sx(Ns,0)+spin.sx(Ns,1),phon.pI(pcut,Np)))
    H = term1-term2+term3+term4+term5
    #collapse operator
    clist = []
    i = config
    for m in range(Np):
        cm = tensor(spin.sI(Ns), phon.down(m, ion0.pcut, ion0.N))
        clist.append(emat[m,i]*np.sqrt(fr_conv(ion0.gamma[m],'hz')*(1+ion0.n_bar()))*cm)
        clist.append(emat[m,i]*np.sqrt(fr_conv(ion0.gamma[m],'hz')*ion0.n_bar())*cm.dag())
    return H, clist

def ereasonance(ion0,nmax1,nmax2):
    '''
    Compute the expected deltaE of resonance specified by nmax1 for rock mode 
    and nmax2 for tilt mode

    Parameters
    ----------
    ion0 : ions class object
        the object that represent the system to be simulated
    nmax1 : int
        maximum integer multiple of rock detuning 
    nmax2 : int
        maximum integer multiple of tilt detuning 
    Returns
    -------
    rock_el: np array, reasonace energy difference caused by overlapping between rock 
    energy levels
    tilt_el: reasonace energy difference by overlapping between tilt 
    energy levels
    mix_el: reasonace energy difference by overlapping between rock 
    energy levels
    mix_mn: list of list, each sublist is the combination of m,n that produces the 
    the reasonance energy
    '''
    rock_f = -ion0.dmlist()[0]/(2*np.pi)
    tilt_f = -ion0.dmlist()[1]/(2*np.pi)
    mix_el = np.array([])
    rock_el = np.arange(0,nmax1+1,1)*rock_f
    tilt_el = np.arange(0,nmax2+1,1)*tilt_f
    mix_mn = np.array([])
    for i in range(1,nmax2+1):
        temp_tilt = tilt_f*i
        for j in range(1,nmax1+1):
            mix_el = np.append(mix_el, temp_tilt+j*rock_f)
            mix_mn = np.append(mix_mn,{j,i})
            mix_el = np.append(mix_el, np.abs(temp_tilt-j*rock_f))
            mix_mn = np.append(mix_mn,{-j,i})
    umix_el, uindex = np.unique(mix_el, return_index=True)
    #print(uindex)
    umix_mn = mix_mn[uindex]    
    return rock_el/2, tilt_el/2, umix_el/2,umix_mn     
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
    pho0 = tensor(phon.inip_thermal(ion0.pcut[0],fr_conv(ion0.fx,'khz'),ion0.Etot),
                  phon.inip_thermal(ion0.pcut[1],fr_conv(ion0.fx,'khz'),ion0.Etot),
                  phon.inip_thermal(ion0.pcut[2],fr_conv(ion0.fx,'khz'),ion0.Etot))
    #dmat = fock(ion0.pcut,0)*fock(ion0.pcut,0).dag()
    #pho0 = tensor(dmat,dmat,dmat)
    rho0 = tensor(ini_sdm,pho0)
    return rho0    
        
    
    
