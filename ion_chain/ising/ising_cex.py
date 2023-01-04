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
import Qsim.ion_chain.transfer.exci_operators as exop
from  Qsim.ion_chain.ising.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: Htot")
    print("Genearte the time-dependent Hamiltonian for 2 site excitation tranfer in ordinary interaction frame")
'''
subfunctions
'''    

      
def Him(ion0,atype,i,m,sindex,mindex,df):
    '''
    Compute H with index i,m for time dependent part 
    Input: 
       
        atype, phonon opeartor type, 0 for destroy, 1 for create
        pcut, int, cut off  level of the harmonic ocsillator eigenenergy
        i, ion index 
        m, phonon space index
        sindex, index to construct spin operator
        mindex, index to construct phonon operator
        mode, phonon space that couples to the laser, 0: axial, 1: radial
    Output:
        Hamiltonina H im, Qobj
    '''
    coeff = ion0.Omega()   
    wlist = ion0.wmlist()[df]
    #set coefficient constants according to the coupling degree of freedom
    if df == 0:
        emat = ion0.Axialmode()
    else:
        emat = ion0.Transmode()
    p_opa = exop.p_ladder(ion0,df,mindex,atype)
    H = tensor(spin.sz(ion0.df_spin(),sindex),p_opa)
    eta_im = eta(wlist[m])*emat[m,i]
    return coeff*eta_im*H 
def tstring(N,atype):
    #generate the string list for time dependent part
    mstring = []
    fstring = []
    for mi in range(1,N+1):
        newm = "m" + str(mi)
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
    for argi in range(N):
        adic[slist[argi]] = wlist[argi]
    return adic    
def Htd(ion0,atype,df): 
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
    N = ion0.N
    Np = exop.pnum(ion0,df)
    delta = ion0.delta
    Hlist = []
    wlist0 = 1j*ion0.wmlist()[df] * 2000* np.pi #this is used to compute deltam in kHz
    mu = (1000*ion0.wmlist()[df][ion0.delta_ref] + delta)* 2* np.pi #kHz 
    Hstr, Hexpr = tstring(N,atype) #kHz
    Harg = argdic(N,atype,wlist0,mu)            
    #compute the mth element by summing over i for Him for destroy operators
    mindex = 0 #this index is used for phonon operators
    for m in exop.ph_list(ion0,df):
        sindex = 0 #this index is used for spin operators
        subH = tensor(spin.zero_op(ion0.df_spin()),exop.p_zero(ion0))
        for i in ion0.laser_couple:
            subH = subH + Him(ion0,atype,i,m,sindex,mindex,df)
            sindex = sindex + 1
        mindex = mindex+1
        Hlist.append([subH,Hexpr[m]]) 
    return Hlist, Harg
'''
function to use
''' 
def Htot(J12, E1, E2, Vx, ion0, df):
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
    Ns = ion0.df_spin() #of ions to be considered for spin space
    Hlistd,Hargd = Htd(ion0,0,df)
    Hlistu,Hargu = Htd(ion0,1,df)
    #phonnic mode
    pI= exop.p_I(ion0)
    sop3 = tensor(spin.up(Ns,0)*spin.down(Ns,1),pI)
    term3 = fr_conv(J12,'hz') * (sop3+sop3.dag())
    #vibrational harmonic oscillator potential
    term4 = (fr_conv(E1,'hz') * tensor(spin.sz(Ns,0),pI)+
             fr_conv(E2,'hz') * tensor(spin.sz(Ns,1),pI))
    #coherent coupling of the donor and acceptor states
    term5 = (fr_conv(Vx,'hz') * 
             tensor(spin.sx(Ns,0)+spin.sx(Ns,1),pI))
    H0 = term3+term4+term5
    #collapse operator
    clist = []
    emat = ion0.Transmode()
    mindex = 0
    for m in exop.ph_list(ion0,df):
        cm = tensor(spin.sI(Ns), exop.p_ladder(ion0,df,mindex,0))
        clist.append(emat[m,ion0.coolant[0]]*np.sqrt(fr_conv(ion0.gamma[m],'hz')*(1+ion0.n_bar()))*cm)
        clist.append(emat[m,ion0.coolant[0]]*np.sqrt(fr_conv(ion0.gamma[m],'hz')*ion0.n_bar())*cm.dag())
        mindex = mindex + 1
    Heff = [H0] + Hlistd + Hlistu    
    return Heff, Hargd, clist