# -*- coding: utf-8 -*-
"""
used to generate eigenenergy diagrams for excitation transfer prototype simulator
@author: zhumj
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
from  Qsim.ion_chain.ion_system import *
from scipy.optimize import curve_fit
oplist = [spin.sz(2,0),spin.sz(2,1)]

def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('___________________________________________________________________')
    print('function:elevel')
    print('Compute energy levels of a given mode ')
    print('___________________________________________________________________')
    print('function:energy_diagram_2d')
    print('Compute eigenenergy for excitation transfer system given displacement of rocking mode. ')
'''
subfunctions
'''

def sort(states):
    #state is the output eigenstates
    #last E is the energy of last  
    #sort states in order downdown downup updown upup
    eige = states[0]; estate = states[1]
    selist = np.zeros(4) #sorted eigenenergy
    s1 = expect(oplist[0],estate); s2 = expect(oplist[1],estate)
    ssum = s1+s2
    maxi = np.argmax(ssum); mini = np.argmin(ssum)
    selist[0] = eige[mini] #down down
    selist[3] = eige[maxi] #up up
    index0 = [0,1,2,3]
    index0.remove(mini) 
    index0.remove(maxi)
    reverse = False
    # do tr
    for i in index0:
        if s1[i] < 0:
            selist[1] = eige[i] #down up 
        else:
            selist[2] = eige[i] #updown 
    return selist                
def parabola(x,a,b,c):
    #parabola fit for the eigenenergy
    return a*x**2 + b*x + c
def vertex(coef):
    a = coef[0]; b = coef[1]; c = coef[2]
    #compute the vertex of parabola 
    return [-b/(2*a), -b**2/(4*a) + c]
def intersect(coef):
    #compute the intersection of the two eigenenergy curves
    a = coef[0]; b = coef[1]; c = coef[2]
    delta0 = np.sqrt(b**2-4*a*c)
    root1 = (-b-delta0)/(2*a)
    root2 = (-b+delta0)/(2*a)
    if root1  > -20 and root1 <20:
        root = root1
    else:
        root = root2
    return root    

def Hspin(J12, E1, E2, V, x, ion0):
    '''
    construct Hamiltonian  of the system in the reasonant rotating frame with
    classical oscillator approximation by considering displacement as a classical quantity
    and neglecting momentum terms, used to compute eigenenergy 
    ----------
    J12 : float
       coupling between ion1 and ion2 [kHz]
    E1 : float
       site energy ion1 [kHz]  
    E2 : float
       site energy ion2 [kHz] 
    x: list of float
        displacement from equilibrium
    ion0: ions class object
        the object that represent the system to be simulated
    Returns
    -------
    H
        Qutip operator


    '''
    Np = ion0.N #of ions to be considered for phonon space
    Ns = ion0.N-1 #of ions to be considered for spin space
    dm = ion0.dmlist()
    #spin phonon coupling
    term1 =  spin.zero_op(Ns)
    for i in range(Ns):
        subop = spin.zero_op(Ns)
        for m in range(Np):
            subop = (subop +
                     0.5 *np.sqrt(2) * ion0.g(i,m) * x[m] * spin.sz(Ns,i))
        term1 = term1 + subop 
    #print(term1)    
    term2 = spin.zero_op(Ns)
    for m in range(Np):
        term2 = term2 + 0.5*dm[m]*((x[m])**2-1)
    #print(term2)
    #phonnic mode
    sop3 = spin.up(Ns,0)*spin.down(Ns,1)
    term3 = fr_conv(J12,'hz') * (sop3+sop3.dag())
    #vibrational harmonic oscillator potential
    term4 = (fr_conv(E1,'hz') * spin.sz(Ns,0)+
             fr_conv(E2,'hz') * spin.sz(Ns,1))
    term5 = V*(spin.sx(Ns,0)+spin.sx(Ns,1))
    H = term1-term2+term3+term4+term5
    return H
'''
functions to use
'''
def elevel(ion0,E0,nlev,m,shift=0):
    '''
    Compute energy levels of a given mode 

    Parameters
    ----------
    ion0: ions class object
       the object that represent the system to be simulated
    E0: float
        zero point of eigenenergy
    nlev: int
        number of levels to be computed    
    m: int
        index of mode used for computing energy levels
    shift: float
        an additional energy shift add to all splittings, can be 
        used for ploting mixed energy levels
    Returns
    -------
    ellist: python list
        list of energy levels

    '''
    deltae = np.abs(ion0.dmlist()[m]/(2*np.pi))
    ellist = np.array([])
    for i in range(nlev):
        if i == 0:
            elev = E0
        else:
            elev = elev + deltae + shift
        ellist = np.append(ellist,elev)
    return ellist    
def eig_energy(ion0,J12,E1,E2,V,x,m,s_type):
    eta_m = 1/np.sqrt(2) * (ion0.g(0,m)-ion0.g(1,m))
    term1 = np.sqrt( np.square( 2*np.pi*(E1-E2) + eta_m*x) + (2*np.pi*J12)**2)
    sum_d = np.sum(ion0.dmlist())/2
    if s_type == 0:
        eig = term1 - ion0.dmlist()[m]/2*( x**2) + sum_d
    else:
        eig = -term1 - ion0.dmlist()[m]/2*(x**2) + sum_d
    return eig
  
def energy_diagram_2d(ion0,J12,E1,E2,V,xarray,method = 0, m=None):
    '''
    compute eigenenergy for excitation transfer system given displacement of
    rocking mode.

    Parameters
    ----------
    ion_sys : ion_ system class object
        the ion system for computation
    J12 : float
       coupling between ion1 and ion2 [kHz]
    E1 : float
       site energy ion1 [kHz]  
    E2 : float
       site energy ion2 [kHz]      
    Vx：
       rabi rate Omegax [kHz] 
    xplot : np array 
        displacement (in unit [X0]) of tilt mode for computation

    Returns
    -------
    array of eigenenergy [kHz] for downup and updown

    '''
    eiged = np.array([])
    eigeu = np.array([])
    factor = 2*np.pi
    if method == 0:
        for x in xarray:
            tempx = [0,0,x]
            Heff = Hspin(J12,E1,E2,V,tempx,ion0)
            state = Heff.eigenstates()
            neige = sort(state)
            eiged = np.append(eiged,neige[1]/factor)
            eigeu = np.append(eigeu,neige[2]/factor)
    else:
        for x in xarray:
            eiged = np.append(eiged,eig_energy(ion0,J12,E1,E2,V,x,m,1)/factor)
            eigeu = np.append(eigeu,eig_energy(ion0,J12,E1,E2,V,x,m,0)/factor)
    return eiged, eigeu              
def energy_diagram_3d(ion_sys,J12,E1,E2,V,xrarray,xtarray):
    '''
    compute eigenenergy for excitation transfer system 

    Parameters
    ----------
    ion_sys : ion_ system class object
        the ion system for computation
    J12 : float
       coupling between ion1 and ion2 [kHz]
    E1 : float
       site energy ion1 [kHz]  
    E2 : float
       site energy ion2 [kHz]      
    Vx：
       rabi rate Omegax [kHz] 
    xplot : np array 
        displacement [X0] of tilt mode for computation

    Returns
    -------
    array of eigenenergy [kHz] for downup and updown

    '''
    scale = X0(ion_sys.wmlist())
    xrlist = xrarray*scale[0]; rsize = np.size(xrlist) 
    xtlist = xrarray*scale[1]; tsize = np.size(xtlist)
    eiged = np.zeros([rsize,tsize])
    eigeu = np.zeros([rsize,tsize])
    factor = 2*np.pi
    progress = 0; total_step = rsize*tsize; report = total_step//10
    for i in range(rsize):
        for j in range(tsize):
            tempx = [xrlist[i],xtlist[j],0]
            Heff = Hspin(J12,E1,E2,V,tempx,ion_sys)
            state = Heff.eigenstates()
            neige = sort(state)
            eiged[i,j] = neige[1]/factor
            eigeu[i,j] = neige[2]/factor
            if progress%report == 0:
                print(100*np.round(progress/total_step,3), '% completed')
            progress = progress + 1 
    return eiged, eigeu              
