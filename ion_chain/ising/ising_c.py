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
import ion_chain.ising.ising_ps as isc
def summary():
    print("____________________________________________________________________")
    print("function: Htot")
    print("Genearte the total Hamiltonian in the format required by the Qutip solver (string method), with ising coupling constructed only with sx and magentic field coupled with sz")
'''
subfunctions
'''    
def Him(fr,fb,N,fz,fx,pcut,atype,i,m):
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
    coeff = -0.5j*np.sqrt(isc.Omega(fr,fx)*isc.Omega(fb,fx))/(2*np.pi*1000)    
    wlist = isc.Transfreq(N,fz,fx)*fz #MHz
    emat = isc.Transmode(N,fz,fx)
    if atype == 0:
        opa = phon.down(m,pcut,N)
    else:
        opa = phon.up(m,pcut,N)
    H = tensor(spin.sx(N,i),opa)
    ita_im = isc.Ita(wlist[m])*emat[m,i]
    return 2* np.pi*coeff*ita_im*H 
def tstring(N,atype):
    #generate the string list for time dependent part
    dmstring = []
    fstring = []
    for mindex in range(1,N+1):
        newdm = "dm" + str(mindex)
        dmstring.append(newdm)
        if atype == 0:
            fstring.append('exp(t * ' + newdm +")")
        else:
            fstring.append('exp(-1 * (t * ' + newdm +"))")
    return dmstring, fstring   
def argdic(N,atype,wlist):     
    #generate the arg list for solving time dependent SE
    #wlist is the list of eigenfrequencies
    adic = {}
    slist, fs = tstring(N,atype) 
    for i in range(N):
        adic[slist[i]] = wlist[i]
    return adic    
def Htd(fr,fb,N,fz,fx,delta,pcut,atype): 
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
    Hlist = []
    wlist0 = np.array(isc.Transfreq(N,fz,fx))* fz * 2000* np.pi #this is used to compute deltam in kHz
    mu = (1000*fx + delta)* 2* np.pi #kHz
    wlist2 = 1j*(mu-wlist0) #kHz 
    Hstr, Hexpr = tstring(N,atype) #kHz
    Harg = argdic(N,atype,wlist2)
    #compute the mth element by summing over i for Him for destroy operators
    for m in range(N):
        subH = tensor(spin.zero_op(N),phon.zero_op(pcut,N))
        for i in range(N): 
            subH = subH + Him(fr,fb,N,fz,fx,pcut,atype,i,m)
        if atype == 1:
            subH = -1*subH
        Hlist.append([subH,Hexpr[m]]) 
    return Hlist, Harg
'''
function to use
''' 
def Htot(H0,fr,fb,N,fz,fx,delta,clevel):
    '''
    Genearte the total Hamiltonian in the format required by the Qutip solver (string method),
    with ising coupling constructed only with sx and magentic field coupled with sz
    input(H0,fr,fb,N,fz,fx,delta,clevel)
    Parameters
    ----------
    H0 : qutip operator
       time independent part of the Hamiltonian
    fr : float
        red side band rabi-frequency [kHz]
    fb : float
        blue side band rabi-frequency [kHz]
    N : TYPE
        number of ions in the system, N > 1
    fz : float
        axial frequency of the ion trap, [MHz]
    fx : float
        transverse frequency of the ion trap, [MHz]
    delta : float
        detuning, defined as the deviation from transverse COM frequency [kHz]
    clevel : int
        cut off  level of the harmonic ocsillator eigenenergy

    Returns
    -------
    Heff : list
        list of Qutip Operator and string expressions for time dependent functions, 
        format required by the Qutip solver
    Hargd : dictionary
      dictionary that records the value of coefficients for time dependent functions
    '''
    Hlistd,Hargd = Htd(fr,fb,N,fz,fx,delta,clevel,0)
    Hlistu,Hargu = Htd(fr,fb,N,fz,fx,delta,clevel,1)
    Heff = [H0] + Hlistd + Hlistu
    return Heff, Hargd
