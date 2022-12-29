# -*- coding: utf-8 -*-
"""
Compute the complete time-dependent Hamiltonian for the 2 ion open qunatum system (ordinary interaction frame)
used to simulation electron transition between acceptor and donor state
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
from  ion_chain.ising.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: Htot")
    print("Genearte the time-dependent Hamiltonian for 2 state electron transfer in ordinary interaction frame")
'''
subfunctions
'''    

def Him(ion0,atype,i,m):
    '''
    Compute H with index i,m for time dependent part 
    Input: 
        ion0, ion class object
        atype, phonon opeartor type, 0 for destroy, 1 for create
        i, ion index 
        m, phonon mode index
    Output:
        component of Hamiltonian H im, Qobj
    '''
    pcut = ion0.pcut
    coeff = ion0.Omega()    
    wlist = ion0.wmlist() #kHz
    emat = ion0.Transmode()
    if atype == 0:
        opa = phon.down(m,pcut,ion0.N)
    else:
        opa = phon.up(m,pcut,ion0.N)
    H = tensor(spin.sz(1,0),opa)
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
def Htd(ion0,atype): 
    '''
    Compute the list of H correponding to time dependent part of H of the
    system as input for qutip solver
    Input: 
        ion0, ion class object
        atype, phonon opeartor type, 0 for destroy, 1 for create
    '''
    N = ion0.N; pcut =ion0.pcut
    fx = ion0.fx; delta = ion0.delta
    Hlist = []
    wlist0 = 1j*ion0.wmlist()* (2000*np.pi) #this is used to compute deltam in kHz
    mu = (1000*fx + delta)* 2* np.pi #kHz 
    Hstr, Hexpr = tstring(N,atype) #kHz
    Harg = argdic(N,atype,wlist0,mu)
    #compute the mth element by summing over i for Him for destroy operators
    #in this specfic case, i=0 since only 1 spin dgf has been used 
    #iniH = tensor(spin.zero_op(1),phon.zero_op(pcut,N))
    for m in range(N):
        Hlist.append([Him(ion0,atype,0,m),Hexpr[m]]) 
    return Hlist, Harg
'''
function to use
''' 
def Htot(Omegaz,ion0):
    '''
    GGenearte the time-dependent Hamiltonian for 2 state electron transfer in ordinary interaction frame
    in the format required by the Qutip solver (string method) and collapse operators
    input(Omegaz,ion0)
    Parameters
    ----------
    Omegaz : float
        rabi frequency due to coupling to magenetic field, energy splitting between
        the donor and acceptor state, [kHz]
    ion0: ions class object
        contains all parameters of the ion-chain system
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
    '''
    term2 = -0.5 * 2 * np.pi*(Omegaz) * tensor(spin.sz(1,0),phon.pI(ion0.pcut,ion0.N))
    term3 = 2 * np.pi*(ion0.Omegax) * tensor(spin.sx(1,0),phon.pI(ion0.pcut,ion0.N)) 
    H0 = term2+term3
    Hlistd,Hargd = Htd(ion0,0)
    Hlistu,Hargu = Htd(ion0,1)
    Heff = [H0] + Hlistd + Hlistu
    c0 =  tensor(spin.sI(1), phon.down(0, ion0.pcut, ion0.N))
    clist = []
    clist.append(np.sqrt( 2 * np.pi * ion0.gamma*(1+ion0.n_bar()))*c0)
    clist.append(np.sqrt( 2 * np.pi * ion0.gamma*ion0.n_bar())*c0.dag())
    return Heff, Hargd, clist