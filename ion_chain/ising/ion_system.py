# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022

@author: zhumj
Compute electron transition rate in quantum region for 2 body system
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
    print('Class:')
    print('___________________________________________________________________')
    print('ions')
    print('class used to store the parameters of a 2 ion system for electron transition simulation')
class ions:
    '''
    this class is used to store the parameters of a 2 ionsystem used for 
    electron transition simulation
    Parameters
    g : float
        state adaptive dipole force, correpsonds to eta*Omega in the ion system [kHz]   
    Omegax : float
        Rabi frequency, correpond to electronic coupling strength V [kHz]    
    N : int
        number of ions in the system
    pcut : int
        cut off level of the harmonic ocsillator eigenenergy
    fz : float
        axial frequency of the ion trap, [MHz]
    fx : float
        transverse frequency of the ion trap, [MHz]
    fr: float
         red side band frequency [kHz]     
    fb: float
         blue side band frequency [kHz] 
    phase: float
        spin phase phis [rad]    
    Etot: float
        total energy of one ion    
    '''
    '''
    default value of parameters
    '''
    delta = 20
    fx = 2 
    fz = 1
    fb = 10
    fr = 10
    phase = np.pi/2
    Omegax =  0.1 * delta  
    gamma = 0.1 * delta
    Etot = 0.5* w(delta) 
    N = 2 
    pcut = 15 #cutoff of phonon energy
    def list_para(self):
        '''
        list all physical parameters of the system

        '''
        print('________________________________________________________________')
        print('number of ions', self.N)
        print('phonon cutoff ', self.pcut)
        print('avearge phonon number ', np.round(self.n_bar(),3))
        print('axial COM (Confining) frequency ',np.round(self.fz,2),' [MHz]')
        print('transverse COM (Confining) frequency ',np.round(self.fx,2), '[MHz]')
        print('detuning delta (measured as deviation from transverse COM freq) ',np.round(self.delta,2)," [kHz]")
        print('red side band rabi frequency ', np.round(self.fr,2),' [kHz]')
        print('blue side band rabi frequency ', np.round(self.fb,2),' [kHz]')
        print('spin phase phis',np.round(self.phase*180/np.pi,2))
        print('(input in rad but displayed in degs)')
        print('electronic coupling strength, or rabi frequency Omega_x ',np.round(self.Omegax,2), ' [kHz]')
        print('state adaptive dipole force g, or eta*Omega ', np.round(self.g(),2),' [kHz]')
        print('cooling rate ', np.round(self.gamma,2)," [kHz]") 
    def Lambda(self):
        '''
        Compute the reorgonization energy lambda, which also correponds to energy
        splitting that leads to the maximum transfer rate
        Returns
        -------
        float [J/10**6]
        '''
        return self.g**2/self.delta
    def wmlist(self):
        '''
        compute the detuning from the two modes
        Returns
        -------
        float [J/10**6]
        '''
        wlist0 = 2*np.pi*iscp.Transfreq(self.N,self.fz,self.fx)*self.fz*1000
        mu = 2*np.pi*(1000*self.fx + self.delta)
        return mu - wlist0
    def n_bar(self):
        '''
        compute the average phonon number for a given set of phonon states
        ----------
        Returns
        -------
        float, no unit

        '''
        return 1/(np.exp(w(self.delta)/self.Etot)-1)
    def w0(self):
        '''
        compute the time scale of the system, defined as 2pi*delta
        ----------
        Returns
        -------
        float, no unit
        '''
        return w(self.delta)
    def g(self):
        return np.sqrt(iscp.Omega(self.fr,self.fx)*iscp.Omega(self.fb,self.fx))*iscp.eta(self.fx)/1000
    
    
