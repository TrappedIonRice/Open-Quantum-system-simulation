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
    print('Functions:')
    print('___________________________________________________________________')
    print('U')
    print('compute the activation energy to reasonance conformation')
    print('___________________________________________________________________')
    print('Htot')
    print('construct Hamiltonian and collpase operators of the system in the reasonant rotating frame Parameters using a single mode, or double mode system')
    print('___________________________________________________________________')
    print('rho_ini')
    print('Construct initial density matrix according to a thermal distribution')
class ions:
    '''
    this class is used to store the parameters of a 2 ionsystem used for 
    electron transition simulation
    Parameters
    g : float
        state adaptive dipole force, correpsonds to Ita*Omega in the ion system [kHz]   
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
    Etot: float
        total energy of one ion    
    '''
    '''
    default value of parameters
    '''
    delta = 10 
    fx = 2 
    fz = 1 
    Omegax =  0.1 * delta 
    g = 2 * delta 
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
        print('electronic coupling strength, or rabi frequency Omega_x ',np.round(self.Omegax,2), ' [kHz]')
        print('state adaptive dipole force g, or Ita*Omega ', np.round(self.g,2),' [kHz]')
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
def Htot(Omegaz, ion0 , single_mode):
    '''
    construct Hamiltonian and collpase operators of the system in the reasonant rotating frame
    Parameters using a single mode, or double mode system
    ----------
    Omegaz : float
        rabi frequency due to coupling to magenetic field, energy splitting between
        the donor and acceptor state, [kHz]
    ion0: ions class object
        the object that represent the system to be simulated
    single_mode : bool
        use COM mode only if true
    Returns
    -------
    H
        Qutip operator
    clist : list
        list of Qutip operators required by qutip solver
    '''
    wm = ion0.wmlist()
    if single_mode:
        term1 = (wm[0]* 
                 tensor(spin.sI(1), phon.up(1, ion0.pcut, 1)*phon.down(1, ion0.pcut, 1)))
        #phonnic mode
        term2 = (-1)*0.5 * w(Omegaz) * tensor(spin.sz(1,0),phon.pI(ion0.pcut,1))
        #vibrational harmonic oscillator potential
        term3 = w(ion0.Omegax) * tensor(spin.sx(1,0),phon.pI(ion0.pcut,1)) 
        #coherent coupling of the donor and acceptor states
        term4 = 0.5*w(ion0.g)*tensor(spin.sz(1,0),(phon.up(1, ion0.pcut, 1)+phon.down(1, ion0.pcut, 1)))   
        #a linear potential in the reaction coordinate z
        c0 =  tensor(spin.sI(1), phon.down(1, ion0.pcut, 1))
        #collapse operator
    else:    
        termp1 = wm[0]*phon.up(0, ion0.pcut, ion0.N)*phon.down(0, ion0.pcut, ion0.N)
        termp2 = wm[1]*phon.up(1, ion0.pcut, ion0.N)*phon.down(1, ion0.pcut, ion0.N)
        term1 =  tensor(spin.sI(1),termp1+termp2)
        #phonnic mode
        term2 = (-1)*0.5 * w(Omegaz) * tensor(spin.sz(1,0),phon.pI(ion0.pcut,ion0.N))
        #vibrational harmonic oscillator potential
        term3 = w(ion0.Omegax) * tensor(spin.sx(1,0),phon.pI(ion0.pcut,ion0.N)) 
        #coherent coupling of the donor and acceptor states
        term4 = 0.5*w(ion0.g)*tensor(spin.sz(1,0),(phon.up(0, ion0.pcut, ion0.N)+phon.down(0, ion0.pcut, ion0.N)))   
        #a linear potential in the reaction coordinate z
        c0 =  tensor(spin.sI(1), phon.down(0, ion0.pcut, ion0.N))
        #collapse operator
    H = (term1+term2+term3+term4) 
    clist = []
    clist.append(np.sqrt(w(ion0.gamma)*(1+ion0.n_bar()))*c0)
    clist.append(np.sqrt(w(ion0.gamma)*ion0.n_bar())*c0.dag())
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
    ini_sdm = Qobj([[1,0], [0,0]])
    if single_mode:
        rho0 = tensor(ini_sdm,phon.inip_thermal(ion0.pcut,ion0.wmlist()[0],ion0.Etot))
    else:
        pho0 = tensor(phon.inip_thermal(ion0.pcut,ion0.wmlist()[0],ion0.Etot),
                      phon.inip_thermal(ion0.pcut,ion0.wmlist()[1],ion0.Etot))
        rho0 = tensor(ini_sdm,pho0)
    return rho0    
        
    
    
