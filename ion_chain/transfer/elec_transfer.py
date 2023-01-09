# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022

@author: zhumj
Construct Hamiltonian in reasonate rotating frame for the 2 ion open qunatum system used to simulation electron transition between acceptor and donor state
"""

import numpy as np
import Qsim.ion_chain.interaction.spin_phonon as Isp
import Qsim.ion_chain.interaction.pure_spin as Is
from  Qsim.ion_chain.ion_system import *
from qutip import *
#subfunction
def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('___________________________________________________________________')
    print('U')
    print('compute the activation energy to reasonance conformation')
    print('___________________________________________________________________')
    print('H_res')
    print('Genearte time-independent Hamiltonian for 2 state electron transfer system in resonant interaction frame')
    print('___________________________________________________________________')
    print('H_ord')
    print('Genearte the time-dependent Hamiltonian for 2 state electron transfer system in ordinary interaction frame')

def U(Omegaz,lambda0):
    '''
    compute the activation energy to reasonance conformation
    Parameters
    ----------
    Omegaz : float
        rabi frequency due to coupling to magenetic field, energy splitting between
        the donor and acceptor state
    lambda0 : float
       reorgonization energy, output of Lambda

    Returns
    -------
    float [J/10**6]

    '''
    return (Omegaz - lambda0)**2 / (4*lambda0)
def H_res(Omegax, Omegaz, ion0):
    '''
    Genearte time-independent Hamiltonian for 2 state electron transfer system in resonant interaction frame

    Parameters
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0 : ion class object

    Returns
    -------
    Heff: Qutip operator
        Effective Hamiltonian in resonant frame
    '''
    H_s =  Is.single_site(Omegax, Omegaz, ion0) 
    Heff = H_s+ Isp.H_res(ion0)
    return Heff
def H_ord(Omegax, Omegaz, ion0):
    '''
    Genearte the time-dependent Hamiltonian for 2 state electron transfer system in ordinary interaction frame,
    in the format required by the Qutip solver (string method) 

    Parameters
    ----------
    Omegax : float 
        coupling coefficient between the doner and acceptor state [kHz]
    Omegaz : float
        energy difference between the doner and acceptor state  [kHz]
    ion0 : ion class object

    Returns
    -------
    Heff: list
        time-dependent Hamiltonian list required by the solver
    H_arg: dictionary
        dic of argument parameters
    '''
    H_s =  Is.single_site(Omegax, Omegaz, ion0) 
    Heff = [H_s] + Isp.H_td(ion0,0) + Isp.H_td(ion0,1)
    H_arg = Isp.H_td_arg(ion0)
    return Heff, H_arg

        
    
    
