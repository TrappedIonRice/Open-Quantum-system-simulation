# -*- coding: utf-8 -*-
"""
Compute the complete time-dependent Hamiltonian for  3 ion open qunatum system
used to simulate excitation transition between 2 sites
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
from  Qsim.ion_chain.ising.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: Htot")
    print("Genearte the time-dependent Hamiltonian for 2 site excitation tranfer in ordinary interaction frame")
'''
subfunctions
'''    

def Him(ion0,atype,i,m,sindex):
    '''
    Compute H with index i,m for time dependent part 
    Input: 
       
        atype, phonon opeartor type, 0 for destroy, 1 for create
        pcut, int, cut off  level of the harmonic ocsillator eigenenergy
        i, python index 
        m, pytho index
        sindex, index of spin operator 
    Output:
        Hamiltonina H im, Qobj
    '''
    N = ion0.N
    pcut = ion0.pcut
    coeff = ion0.Omega()   
    wlist = ion0.wmlist()
    emat = ion0.Transmode()
    if atype == 0:
        opa = phon.down(m,pcut,N)
    else:
        opa = phon.up(m,pcut,N)
    H = tensor(spin.sz(N-1,sindex),opa)
    eta_im = eta(wlist[m])*emat[m,i]
    return coeff*eta_im*H 
def tstring(N,atype):
    #generate the string list for time dependent part
    mstring = []
    fstring = []
    for mindex in range(1,N+1):
        newm = "m" + str(mindex)
        mstring.append(newm)
        if atype == 1:
            fstring.append('cos(t * u) * exp(t * ' + newm +")")
        else:
            fstring.append('cos(t * u) * exp(-1 * (t * ' + newm +"))")
    return mstring, fstring   
def argdic(N,atype,wlist,mu):     
    #generate the arg list for solving time dependent SE
    #wlist is the list of eigenfrequencies, mu is the frequency of the laser
    adic = {"u":mu}
    slist, fs = tstring(N,atype) 
    for i in range(N):
        adic[slist[i]] = wlist[i]
    return adic    
def Htd(ion0,atype,config): 
    '''
    Compute the list of H correponding to time dependent part of H of the
    system as input for qutip solver
    Input: 
        ion0, ion class object
        atype, phonon opeartor type, 0 for destroy, 1 for create
        config: integer
            the configuration of the chain, if config = 0, cool the ion on the side
            if config = 1, cool the ion in the center.
    '''
    N = ion0.N; pcut =ion0.pcut
    delta = ion0.delta
    Hlist = []
    wlist0 = 1j*ion0.wmlist() * 2000* np.pi #this is used to compute deltam in kHz
    mu = (1000*ion0.wmlist()[ion0.delta_ref] + delta)* 2* np.pi #kHz 
    Hstr, Hexpr = tstring(N,atype) #kHz
    Harg = argdic(N,atype,wlist0,mu)
    if config == 0:
        ilist = [0,1]
    else:
        ilist = [0,2]               
    #compute the mth element by summing over i for Him for destroy operators
    for m in range(N):
        sindex = 0 #this index is used for spin operators  
        subH = tensor(spin.zero_op(N-1),phon.zero_op(pcut,N))
        for i in ilist:
            subH = subH + Him(ion0,atype,i,m,sindex)
            sindex = sindex + 1
        Hlist.append([subH,Hexpr[m]]) 
    return Hlist, Harg
'''
function to use
''' 
def Htot(J12, E1, E2, Vx, ion0, config):
    '''
    Compute the complete time-dependent Hamiltonian and collapse operators for the 3 ion open qunatum system
    used to simulate excitation transition between 2 sites, and the collpase operators
    Input:
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
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver, this list represents the time-dependent 
        Hamiltonian of the system in ordinary frame
    Hargd : dictionary
        dictionary that records the value of coefficients for time dependent functions
    clist : list
        list of Qutip operators required by qutip solver
        collapse operators to describe coupling to the evironment
    '''
    Np = ion0.N #of ions to be considered for phonon space
    Ns = ion0.N-1 #of ions to be considered for spin space
    pcut = ion0.pcut
    Hlistd,Hargd = Htd(ion0,0,config)
    Hlistu,Hargu = Htd(ion0,1,config)
    #phonnic mode
    sop3 = tensor(spin.up(Ns,0)*spin.down(Ns,1),phon.pI(pcut,Np))
    term3 = fr_conv(J12,'hz') * (sop3+sop3.dag())
    #vibrational harmonic oscillator potential
    term4 = (fr_conv(E1,'hz') * tensor(spin.sz(Ns,0),phon.pI(pcut,Np))+
             fr_conv(E2,'hz') * tensor(spin.sz(Ns,1),phon.pI(pcut,Np)))
    #coherent coupling of the donor and acceptor states
    term5 = (fr_conv(Vx,'hz') * 
             tensor(spin.sx(Ns,0)+spin.sx(Ns,1),phon.pI(pcut,Np)))
    H0 = term3+term4+term5
    #collapse operator
    clist = []
    i = config
    emat = ion0.Transmode()
    for m in range(Np):
        cm = tensor(spin.sI(Ns), phon.down(m, ion0.pcut, ion0.N))
        clist.append(emat[m,i]*np.sqrt(fr_conv(ion0.gamma[m],'hz')*(1+ion0.n_bar()))*cm)
        clist.append(emat[m,i]*np.sqrt(fr_conv(ion0.gamma[m],'hz')*ion0.n_bar())*cm.dag())
    Heff = [H0] + Hlistd + Hlistu    
    return Heff, Hargd, clist