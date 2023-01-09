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
import Qsim.ion_chain.interaction.spin_phonon as Isp
import Qsim.ion_chain.interaction.pure_spin as Is
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
from qutip import *
from  Qsim.ion_chain.ion_system import *
#subfunction

def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('___________________________________________________________________')
    print('H_res')
    print('Genearte time-independent Hamiltonian for 2 site excitation transfer system in resonant interaction frame')
    print('___________________________________________________________________')
    print('H_ord')
    print('Genearte the time-dependent Hamiltonian for 2 site excitation transfer system in ordinary interaction frame')

def H_ord(J12, E1, E2, Vx, ion0):
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
    Returns
    -------
    Heff: list
        time-dependent Hamiltonian list required by the solver
    H_arg: dictionary
        dic of argument parameters
    '''
    H_s =  Is.double_site(J12, E1, E2, Vx, ion0)
    Heff = [H_s]+ Isp.H_td(ion0,0) + Isp.H_td(ion0,1)
    H_arg = Isp.H_td_arg(ion0)
    return Heff, H_arg
def H_res(J12, E1, E2, Vx, ion0):
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
    Returns
    -------
    Heff
        Qutip operator
        Hamiltonian in reasonant rotating frame
    '''   
    H_s =  Is.double_site(J12, E1, E2, Vx, ion0)
    Heff = H_s+ Isp.H_res(ion0)
    return Heff

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

        
    
    
