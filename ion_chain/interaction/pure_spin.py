# -*- coding: utf-8 -*-
"""
Construct spin interaction Hamiltonian for single-site electron transfer systems 
and double site-excitation transfer systems

@author: zhumj
"""

import numpy as np
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.operator.spin_phonon as sp_op
from  Qsim.ion_chain.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: single_site")
    print("Compute Hamiltonian for single site spin interaction between two ionic states")
    print("function: single_site")
    print("Compute Hamiltonian for spin interaction of a two site system ")
def single_site(Omegax=0, Omegaz=0, ion0=None):
    '''
    Compute Hamiltonian for single site spin interaction 
    between the two ionic states for each enabled ion site, 
    with no coupling between the sites

    Parameters
    ----------
    Omegax : float/list of float 
        coupling coefficient between the doner and acceptor state for each site [kHz]
    Omegaz : float/list of float
        energy difference between the doner and acceptor state for each site  [kHz]
    ion0 : ion class object
    
    Returns
    -------
    Qutip operator

    '''
    Ns = ion0.df_spin()
    if Ns == 1:
        Hspin = (2 * np.pi) * (Omegax * tensor(spin.sx(1,0),sp_op.p_I(ion0))  
         -0.5 * Omegaz * tensor(spin.sz(1,0),sp_op.p_I(ion0))) 
    else:
        Hspin = (2 * np.pi) * (Omegax[0] * tensor(spin.sx(Ns,0),sp_op.p_I(ion0))  
         -0.5 * Omegaz[0] * tensor(spin.sz(Ns,0),sp_op.p_I(ion0)))
        for i in range(1,Ns):
            Hspin = Hspin + (2 * np.pi) * (Omegax[i] * tensor(spin.sx(Ns,i),sp_op.p_I(ion0))  
             -0.5 * Omegaz[i] * tensor(spin.sz(Ns,i),sp_op.p_I(ion0)))
    return Hspin
def double_site(J12=0, E1=0, E2=0, Vx=0, ion0=None):
    '''
    Compute Hamiltonian for spin interaction of a two site system 
    Parameters
    ----------
    J12 : float
       coupling between ion1 and ion2 [kHz]
    E1 : float
       site energy ion1 [kHz]  
    E2 : float
       site energy ion2 [kHz]      
    Vx：float
       rabi rate Omegax for a single site [kHz] 
    ion0: ion class object
    Returns
    -------
    Qutip operator

    '''
    Ns = 2
    #coupling between two sites
    sop = tensor(spin.up(Ns,0)*spin.down(Ns,1),sp_op.p_I(ion0))
    term1 = fr_conv(J12,'hz') * (sop+sop.dag())
    #site energy difference
    term2 = (fr_conv(E1,'hz') * tensor(spin.sz(Ns,0),sp_op.p_I(ion0))+
             fr_conv(E2,'hz') * tensor(spin.sz(Ns,1),sp_op.p_I(ion0)))
    #coupling between donor/ accpetor states for each site
    term3 = (fr_conv(Vx,'hz') * 
             tensor(spin.sx(Ns,0)+spin.sx(Ns,1),sp_op.p_I(ion0)))
    return term1+term2+term3