# -*- coding: utf-8 -*-
"""

function: Htot
@author: zhumj
"""
import numpy as np
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.transfer.exci_operators as exop
from  Qsim.ion_chain.ion_system import *
def summary():
    print("____________________________________________________________________")
    print("function: H_td")
    print("Genearte the time-dependent Hamiltonian for laser-ion interaction")
'''
subfunctions
'''    
def sigma_phi(N,i,phase):
    return np.cos(phase)*spin.sx(N,i) + np.sin(phase)*spin.sy(N,i)
      
def Him_ord(ion0,atype,i,m,sindex,mindex,i_type=0):
    '''
    Compute the i,m th component for time independent part of ion-laser interaction 
    Hamiltonian in ordinary frame, which discribes the coupling between ion i and mode m
    Input: 
       
        atype: int
            phonon opeartor type, 0 for destroy, 1 for create
        i: int
            ion index 
        m: int
            phonon space index
        sindex: int
            index to construct spin operator
        mindex: int
            index to construct phonon operator
        i_type: int default as 0
            type of interaction, set to 1 for ising interactions
            
    Output:
        Hamiltonina H im, Qobj
    '''
    #set coefficient constants according to the coupling degree of freedom
    p_opa = exop.p_ladder(ion0,mindex,atype)
    if i_type == 1:
        s_oper = sigma_phi(ion0.df_spin(),sindex,ion0.phase)
    else:    
        s_oper = spin.sz(ion0.df_spin(),sindex)
    H = tensor(s_oper,p_opa)
    return ion0.g(i,m)*H 

def tstring(N,atype):
    '''
    Generate the list of time depedent expression for the Hamiltonian 

    Parameters
    ----------
    N : int
        total number of ions in the trapped ion system
    atype : int
        type of phonon operators, 0 for down, 1 for up

    Returns
    -------
    mstring : list of string
        list of parameters
    fstring : list of string
        list of time dependent expressions to be used 

    '''
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
def H_td_arg(ion0):    
    '''
    Generate an argument dictonary to map parameters in time-dependent 
    expressions to their actual values
    Parameters
    ----------
    ion0: ion class object
    Returns
    -------
    adic : dictionary
        argument dictonary
    '''
    #generate the arg list for solving time dependent SE
    #wlist is the list of eigenfrequencies, mu is the frequency of the laser
    adic = {"u":ion0.mu()}
    slist, fs = tstring(ion0.N,0)
    wlist0 = 1j*ion0.wmlist() * 2000* np.pi #this is used to compute deltam in kHz
    for argi in range(ion0.N):
        adic[slist[argi]] = wlist0[argi]
    return adic 

def H_td(ion0,atype,i_type = 0): 
    '''
    Compute the list of H correponding to time-dependent Hamiltonian for ion-lasesr
    interaction with a single drive as a input for qutip solver
    Input: 
        ion0, ion class object
        atype: int
            phonon opeartor type, 0 for destroy, 1 for create
        i_type: int default as 0
            type of interaction, set to 1 for ising interactions    
    '''
    Hstr, Hexpr = tstring(ion0.N,atype) #kHz generate time depedent part for all modes and select 
                                      # modes of interest           
    #compute the mth element by summing over i for Him for destroy operators
    Hlist = []
    mindex = 0 #this index is used for phonon operators
    for m in exop.ph_list(ion0):
        sindex = 0 #this index is used for spin operators
        subH = tensor(spin.zero_op(ion0.df_spin()),exop.p_zero(ion0))
        for i in ion0.laser_couple:
            subH = subH + Him_ord(ion0,atype,i,m,sindex,mindex,i_type)
            sindex = sindex + 1
        mindex = mindex+1
        Hlist.append([subH,Hexpr[m]]) 
    return Hlist
'''
Resonant interaction frame
'''
def Him_res(ion0, i,m,sindex,mindex):
    '''
    Compute the i,m th component for ion-laser interaction  Hamiltonian in resonant frame, 
    which discribes the coupling between ion i and mode m
    Input: 
        i, ion index 
        m, phonon space index
        sindex, index to construct spin operator
        mindex, index to construct phonon operator
    Output:
        Hamiltonina H im, Qobj
    '''
    #set coefficient constants according to the coupling degree of freedom
    p_opa = exop.p_ladder(ion0,mindex,0) + exop.p_ladder(ion0,mindex,1) 
    H = tensor(spin.sz(ion0.df_spin(),sindex),p_opa)
    return 0.5*ion0.g(i,m)*H 
def H_res(ion0):
    '''
    Compute the time-independent Hamiltonian e for ion-lasesr
    interaction with a single drive in resonant fram
    Input: 
        ion0, ion class object
    '''
    term1 = tensor(spin.zero_op(ion0.df_spin()),exop.p_zero(ion0)) #laser-ion interaction term 
    term2 = tensor(spin.zero_op(ion0.df_spin()),exop.p_zero(ion0)) #compensation for change of interaction frame
    mindex = 0 #this index is used for phonon operators
    for m in exop.ph_list(ion0):
        sindex = 0 #this index is used for spin operators
        for i in ion0.laser_couple:
            term1 = term1 + Him_res(ion0,i,m,sindex,mindex)
            sindex = sindex + 1
        term2 = (term2 + ion0.dmlist()[m]
                 *tensor(spin.sI(ion0.df_spin()),
                         exop.p_ladder(ion0,mindex,1)*exop.p_ladder(ion0,mindex,0)))    
        mindex = mindex+1
    return term1-term2
        
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