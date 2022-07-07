# -*- coding: utf-8 -*-
"""
Compute the complete Hamiltonian for the 2 ion open qunatum system
 used to simulation electron transition between acceptor and donor state
function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon
import ion_chain.ising.ising_ps as isc
def summary():
    print("____________________________________________________________________")
    print("function: Htot")
    print("Genearte the total Hamiltonian in the format required by the Qutip solver (string method), with ising coupling constructed only with sx and magentic field coupled with sz")
'''
subfunctions
'''    

def Him(fr,fb,N,fz,fx,pcut,atype,i,m,phase):
    '''
    Compute H with index i,m for time dependent part 
    Input: 
        fr, red side band rabi-frequency [kHz]
        fb, blue side band rabi-frequency [kHz]
        N, int, #of ions in the system
        fz, axial frequency of the ion trap, MHz
        fx, transverse frequency of the ion trap, MHz
        delta, detuning in kHz 
        opa, phonon opeartor type, 0 for destroy, 1 for create
        pcut, int, cut off  level of the harmonic ocsillator eigenenergy
        i, python index 
        m pytho index
    Output:
        Hamiltonina H im, Qobj
    '''
    coeff = np.sqrt(isc.Omega(fr,fx)*isc.Omega(fb,fx))/(2*np.pi*1000)    
    wlist = isc.Transfreq(N,fz,fx)*fz #MHz
    emat = isc.Transmode(N,fz,fx)
    if atype == 0:
        opa = phon.down(m,pcut,N)
    else:
        opa = phon.up(m,pcut,N)
    H = tensor(spin.sz(1,0),opa)
    eta_im = isc.eta(wlist[m])*emat[m,i]
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
def Htd(ion0,atype,): 
    '''
    Compute the list of H correponding to time dependent part of H of the
    system as input for qutip solver
    Input: 
        fr, red side band rabi-frequency [kHz]
        fb, blue side band rabi-frequency [kHz]
        N, int, #of ions in the system
        fz, axial frequency of the ion trap
        fx, transverse frequency of the ion trap
        delta, detuning in kHz
        pcut, int, cut off  level of the harmonic ocsillator eigenenergy
        opa, phonon opeartor type, 0 for destroy, 1 for create
    '''
    fr = ion0.fr; fb=ion0.fb
    N = ion0.N; pcut =ion0.pcut
    fz = ion0.fz; fx = ion0.fx
    phase = ion0.phase; delta = ion0.delta
    Hlist = []
    wlist0 = 1j*np.array(isc.Transfreq(N,fz,fx))* fz * 2000* np.pi #this is used to compute deltam in kHz
    mu = (1000*fx + delta)* 2* np.pi #kHz 
    Hstr, Hexpr = tstring(N,atype) #kHz
    Harg = argdic(N,atype,wlist0,mu)
    #compute the mth element by summing over i for Him for destroy operators
    subH = tensor(spin.zero_op(1),phon.zero_op(pcut,N))
    for m in range(N):
        subH = subH + Him(fr,fb,N,fz,fx,pcut,atype,0,m,phase)
        Hlist.append([subH,Hexpr[m]]) 
    return Hlist, Harg
'''
function to use
''' 
def Htot(H0,ion0):
    '''
    Genearte the total Hamiltonian in the format required by the Qutip solver (string method)
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