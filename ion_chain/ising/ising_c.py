# -*- coding: utf-8 -*-
"""
Compute the complete Hamiltonian for the ising coupling system
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
from  ion_chain.ising.ion_system import *
import ion_chain.ising.ising_ps as isc
def summary():
    print("____________________________________________________________________")
    print("function: Htot")
    print("Genearte the total Hamiltonian in the format required by the Qutip solver (string method), with ising coupling constructed only with sx and magentic field coupled with sz")
'''
subfunctions
'''    
def sigma_phi(N,i,phase):
    return np.cos(phase)*spin.sx(N,i) + np.sin(phase)*spin.sy(N,i)
def Him(ion0,atype,i,m):
    '''
    Compute H with index i,m for time dependent part 
    Input: 
        ion0, ion class object
        i, python index 
        m pytho index
    Output:
        Hamiltonina H im, Qobj
    '''
    coeff = ion0.Omega()/(2*np.pi)   
    emat = ion0.Transmode()
    if atype == 0:
        opa = phon.down(m,ion0.pcut,ion0.N)
    else:
        opa = phon.up(m,ion0.pcut,ion0.N)
    H = tensor(sigma_phi(ion0.N,i,ion0.phase),opa)
    eta_im = eta(ion0.wmlist()[m])*emat[m,i]
    return 2* np.pi*coeff*eta_im*H 
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
        ion0, ion system class object
        atype, phonon opeartor type, 0 for destroy, 1 for create
    '''
    N = ion0.N; pcut =ion0.pcut
    delta = ion0.delta
    Hlist = []
    wlist0 = 1j*np.array(ion0.Transfreq())* ion0.fz * 2000* np.pi #this is used to compute deltam in kHz
    mu = (1000*ion0.fx + delta)* 2* np.pi #kHz 
    Hstr, Hexpr = tstring(N,atype) #kHz
    Harg = argdic(N,atype,wlist0,mu)
    #compute the mth element by summing over i for Him for destroy operators
    for m in range(N):
        subH = tensor(spin.zero_op(N),phon.zero_op(pcut,N))
        for i in range(N): 
            subH = subH + Him(ion0,atype,i,m)
        Hlist.append([subH,Hexpr[m]]) 
    return Hlist, Harg
'''
function to use
''' 
def Htot(H0,ion0):
    '''
    Genearte the total Hamiltonian in the format required by the Qutip solver (string method),
    with ising coupling constructed only with sx and magentic field coupled with sz
    input(H0,fr,fb,N,fz,fx,delta,clevel)
    Parameters
    ----------
    H0 : qutip operator
       time independent part of the Hamiltonian
    ion0: ions class object
        contains all parameters of the ion-chain system
    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver
    Hargd : dictionary
      dictionary that records the value of coefficients for time dependent functions
    '''
    Hlistd,Hargd = Htd(ion0,0)
    Hlistu,Hargu = Htd(ion0,1)
    Heff = [H0] + Hlistd + Hlistu
    return Heff, Hargd