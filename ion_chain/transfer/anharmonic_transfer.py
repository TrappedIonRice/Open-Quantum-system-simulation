# -*- coding: utf-8 -*-
"""
Compute the complete time-dependent Hamiltonian with anharmonic terms
for  3 ion open qunatum system, under resonant condition which couples
axial tilt and radial rock mode.  
The laser field is only coupled to the ion on the side.
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.ion_chain.transfer.exci_operators as exop
import Qsim.ion_chain.interaction.spin_phonon as Isp
import Qsim.ion_chain.interaction.pure_spin as Is
from  Qsim.ion_chain.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: H_ord")
    print("Genearte the time-dependent Hamiltonian for 1 site electron tranfer with anhormonic terms in ordinary interaction frame")

'''
function to use
''' 
def H_ord(Omegax, Omegaz, ion0,spterm = True, ah_term=False,ah_op=0):
    '''
    Compute the complete time-dependent Hamiltonian and collapse operators for the 3 ion open qunatum system
    used to simulate electron of a single site with anharmonicity in ordinary interaction frame 
    Input:
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0: ions class object
        the object that represent the system to be simulated
    spterm: bool:
        if spin-phonon interacion term will be included
    an_term: bool
        if anharmonic terms will be included
    an_op: Qutip operator
        anharmonic coupling operator
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver, this list represents the time-dependent 
        Hamiltonian of the system in ordinary frame
    Hargd : dictionary
        dictionary that records the value of coefficients for time dependent functions
    '''
    #phonnic mode
    H_s =  Is.single_site(Omegax, Omegaz, ion0) 
    if ah_term:
        term_a = ah_op + ah_op.dag()
    else:    
        term_a = 0
    H0 = H_s+term_a
    Heff = [H0] + Isp.H_td(ion0,0) + Isp.H_td(ion0,1)
    H_arg = Isp.H_td_arg(ion0)
    if spterm:
        return Heff, H_arg
    else:
        return H0