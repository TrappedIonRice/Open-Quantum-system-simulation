# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:53:19 2023
Compute ET rate based on theoretical models
@author: zhumj
"""
from qutip import *
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def FC_matrix(cutoff,gf):
    #compute Franck-Condon matrix in fock state representation
    a = phon.down(0,[cutoff],1)
    aop = (gf*(a.dag() - a)).expm()
    return aop
def plot_FC_coef(cutoff, gf, n, tot_points):
    #plot FC coefficient for a given initial fock state n 
    mat = FC_matrix(cutoff,gf)
    fcplot = np.abs(mat[n,n:tot_points+n].reshape(tot_points))**2
    splot = np.arange(0, tot_points,1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(splot,fcplot,'*',markersize=12)
    plt.xlabel(r'$\nu''$',fontsize = 16)
    plt.ylabel(r'$FC$',fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xticks(np.arange(0, 13,1),fontsize = 16)
    #plt.xlim(1,12)
    #plt.legend(fontsize = 15)
    plt.grid()
    plt.show()
    print(fcplot)
def FC_sum(cutoff,E_split,gf,nbar,state_type='thermal'):
    '''
    Compute the sum of FC coefficient for a given energy splitting,
    weighted by thermal distribution 

    Parameters
    ----------
    cutoff : int
        cutoff of SHO operator
    E_split : float
        effective energy splitting
    gf : float
        effective s-p coupling
    nbar: float
        average phonon number of the system
    state_type: str
        specify the type of initial state, 
         can be 'thermal' or 'fock'
    Returns
    -------
    Float

    '''
    if state_type == 'thermal':
        pdist = phon.p_thermal(cutoff,nbar)
    elif state_type == 'fock':
        pdist = np.zeros(cutoff)
        pdist[nbar] = 1 
    else:
        print('Incorrect specification of state type, allowed types are thermal, fock')
        return 0
    aop = FC_matrix(cutoff,gf)
    result = 0
    for i in range(cutoff):
        if i + E_split >= cutoff:
            break
        result +=  pdist[i]*(np.abs(aop[i,i + E_split]))**2
    return result
    
def ET_rate_point_2D_norm(nx_d,ny_d,nx_a,ny_a,V_fac,pdist_x,pdist_y,dmat_x,dmat_y):
    # compute the ET rate k given a set {nx_d,ny_d,nx_a,ny_a}
    fc = (np.abs(dmat_x[nx_a,nx_d] * dmat_y[ny_a,ny_d] ))**2
    result = (2*np.pi)**2*V_fac**2*pdist_x[nx_d]*pdist_y[ny_d]*fc
    return result
    
def ET_rate_point_2D(nx_d,ny_d,nx_a,ny_a,V_fac,pdist_x,pdist_y,dmat_x,dmat_y):
    # compute the ET rate k given a set {nx_d,ny_d,nx_a,ny_a}
    fc = (np.abs(dmat_x[nx_a,nx_d] * dmat_y[ny_a,ny_d] ))**2
    result = (2*np.pi)*(2*np.pi*V_fac)**2*pdist_x[nx_d]*pdist_y[ny_d]*fc
    return result
    
def ET_rate_dist_2D(p_cut,n_cut,omega_x,omega_y,gx,gy,V_fac,nbar_x,nbar_y):
    '''
    calculate transfer rate in the perturbation regime for ET system with 2 modes

    Parameters
    ----------
    p_cut : int
        cutoff for each phonon space
    n_cut : int
        cutoff for fock state used for computing energy
    omega_x : float
        frequency of x mode
    omega_y : float
        frequency for y mode
    gx : float
        normalized sp coupling for x mode (gx/omega_x)
    gy : float
        normalized sp coupling for y mode (gy/omega_y)
    V_fac : float
        coefficient for sigma_x 
    nbar_x : float
        average phonon numebr for x mode
    nbar_y : TYPE
        average phonon numebr for y mode

    Returns
    -------
    result_dic : dic
        keys are resonant DeltaE, values are transfer rate

    '''
    # store results in a hash map {Delta E : k}
    result_dic = {}; 
    # displacement opeartor for x,y mode
    dmat_x = displace(p_cut,-gx); dmat_y = displace(p_cut,-gy)
    # thermal distribution for x,y mode
    pdist_x = phon.p_thermal(p_cut,nbar_x); pdist_y = phon.p_thermal(p_cut,nbar_y)
    for nx_d, ny_d, nx_a, ny_a in product(range(n_cut), repeat=4):
        #check if the donor energy is smaller than the acceptor (difference to be compensate by E)
        DeltaE = (nx_a-nx_d)*omega_x + (ny_a-ny_d)*omega_y
        if  (DeltaE>0 and pdist_x[nx_d]*pdist_y[ny_d]>0 ):
            new_k = ET_rate_point_2D(nx_d,ny_d,nx_a,ny_a,V_fac,pdist_x,pdist_y,dmat_x,dmat_y)
            result_dic[DeltaE] = result_dic.get(DeltaE, 0) + new_k
    return result_dic
'''
def check_first_order_trans(ni,nf,j):
    #given a pair of arrays ni=[ni_1,ni_2],nf=[nf_1,nf_2] and a indice j, check if VAET
    #ni[j]->nf[j] can be induced by first order coupling (a+a^\dag)
    mask = np.ones_like(ni, dtype=bool) ; mask[j] = False
    if np.array_equal(ni[mask], nf[mask]):
        result = ni[j]*float(nf[j]-ni[j]+1) + nf[j]*float(nf[j]-ni[j]-1)
    else:
        result = 0
    return result
    
def VAET_2mode_point(ni,nf,gfac,pdist):
    #given a pair of arrays ni=[ni_1,ni_2],nf=[nf_1,nf_2], parameters Vfac, g=[g1,g2], pdist=[pdist1,pdist2] 
    #(without prefactors)
    result = 0
    for j in range(2):
        result += gfac[j]**2 * check_first_order_trans(ni,nf,j)
    return pdist[0][ni[0]]*pdist[1][ni[1]]*result 
'''
def VAET_2mode_point_norm(ni,nf,gfac,pdist):
    # notice this function is used for valid pairs of ni,nf
    #given a pair of arrays ni=[ni_1,ni_2],nf=[nf_1,nf_2], parameters Vfac, g=[g1,g2], pdist=[pdist1,pdist2] 
    #(without prefactors)
    result = 0
    for j in range(2):
        result += gfac[j]**2 * (ni[j]*float(nf[j]-ni[j]+1==0) + nf[j]*float(nf[j]-ni[j]-1==0))
    return pdist[0][ni[0]]*pdist[1][ni[1]]*result

def VAET_2mode_point(ni,nf,gfac,pdist):
    # notice this function is used for valid pairs of ni,nf
    #given a pair of arrays ni=[ni_1,ni_2],nf=[nf_1,nf_2], parameters Vfac, g=[g1,g2], pdist=[pdist1,pdist2] 
    #(without prefactors)
    result = 0
    for j in range(2):
        result += (2*np.pi*gfac[j])**2 * (ni[j]*float(nf[j]-ni[j]+1==0) + nf[j]*float(nf[j]-ni[j]-1==0))
    return pdist[0][ni[0]]*pdist[1][ni[1]]*result

def Lorentz_norm(gamma, E,E0):
    return (gamma/(2*np.pi))/( (gamma/2)**2 + (E-E0)**2/(2*np.pi) )
    
def Lorentz(gamma, E,E0):
    return ((2*np.pi*gamma)/(2*np.pi))/( 2*np.pi*(2*np.pi*gamma/2)**2 + (2*np.pi*(E-E0))**2 )
    
def Lorentz_VAET(gamma, E,E0,V):
    epsilon = np.sqrt(E**2/4 + V**2)
    return ((2*np.pi*gamma)/(2*np.pi))/( 2*np.pi*(2*np.pi*gamma/2)**2 + (2*np.pi*(2*epsilon-E0))**2)
    
def VAET_rate_dist_2D_Lor_norm(p_cut,n_cut,omega,V_fac,gfac,nbar,gamma,Eplot):
    kplot = np.zeros(np.shape(Eplot))
    # thermal distribution for x,y mode
    pdist = [phon.p_thermal(p_cut,nbar[0]), phon.p_thermal(p_cut,nbar[1])]
    # generate all possible initial states
    x, y = np.meshgrid(np.arange(n_cut), np.arange(n_cut))
    ni_mat = np.column_stack((x.ravel(), y.ravel()))
    dif_mat = np.array([[1,0],[0,1]]) # exchange spin energy E to gain 1 phonon for mode 1/2
    #dif_mat_1 = np.array([[0,1],[0,-1]]) # gain/loss 1 phonon for mode 2
    for ni in ni_mat:
        for k in range(2):
            nf = ni + dif_mat[k]
            #print(VAET_2mode_point(ni,nf,gfac,pdist))
            #on resonance 2E = \omega
            pre_fac =  (2*np.pi) * (V_fac)**2 / omega[k]**2
            kplot += (pre_fac*VAET_2mode_point_norm(ni,nf,gfac,pdist)
                          *Lorentz_norm(gamma[k], Eplot, omega[k]))
    return kplot

def VAET_rate_dist_2D_Lor(p_cut,n_cut,omega,V_fac,gfac,nbar,gamma,Eplot):
    kplot = np.zeros(np.shape(Eplot))
    # thermal distribution for x,y mode
    pdist = [phon.p_thermal(p_cut,nbar[0]), phon.p_thermal(p_cut,nbar[1])]
    # generate all possible initial states
    x, y = np.meshgrid(np.arange(n_cut), np.arange(n_cut))
    ni_mat = np.column_stack((x.ravel(), y.ravel()))
    dif_mat = np.array([[1,0],[0,1]]) # exchange spin energy E to gain 1 phonon for mode 1/2
    #dif_mat_1 = np.array([[0,1],[0,-1]]) # gain/loss 1 phonon for mode 2
    for ni in ni_mat:
        for k in range(2):
            nf = ni + dif_mat[k]
            #print(VAET_2mode_point(ni,nf,gfac,pdist))
            #on resonance 2E = \omega
            pre_fac =  (2*np.pi) * (2*np.pi*V_fac)**2 / (2*np.pi*omega[k])**2
            kplot += (pre_fac*VAET_2mode_point(ni,nf,gfac,pdist)
                          *Lorentz_VAET(gamma[k], Eplot, omega[k],V_fac))
    return kplot

def ET_rate_dist_2D_Lor_norm(p_cut,n_cut,omega,g,V_fac,nbar,gamma,Eplot,n_dep=False):
    '''
    calculate transfer rate in the perturbation regime for ET system with 2 modes

    Parameters
    ----------
    p_cut : int
        cutoff for each phonon space
    n_cut : int
        cutoff for fock state used for computing energy
    omega : list of float
        frequency of [x,y] mode
    g : list of float
        normalized sp coupling for [x,y] mode (gx/omega_x,gy/omega_y)
    V_fac : float
        coefficient for sigma_x 
    nbar : list of float
        average phonon numebr for [x,y] mode
    gamma: list of float
        dissipation rate for [x,y] mode
    Eplot: np array
        array of Delta E used for plot
    n_dep: bool
        if true, include the fock number of acceptor state
    Returns
    -------
    np array: transfer rate evaluated at each point of Eplot

    '''
    # result transfer rate plot
    kplot = np.zeros(np.shape(Eplot))
    # displacement opeartor for x,y mode
    dmat_x = displace(p_cut,-g[0]); dmat_y = displace(p_cut,-g[1])
    # thermal distribution for x,y mode
    pdist_x = phon.p_thermal(p_cut,nbar[0]); pdist_y = phon.p_thermal(p_cut,nbar[1])
    for nx_d, ny_d, nx_a, ny_a in product(range(n_cut), repeat=4):
        #check if the donor energ[1] is smaller than the acceptor (difference to be compensate by E)
        DeltaE = (nx_a-nx_d)*omega[0] + (ny_a-ny_d)*omega[1]
        if  (DeltaE>0 and pdist_x[nx_d]*pdist_y[ny_d]>0 ):
            new_k = ET_rate_point_2D_norm(nx_d,ny_d,nx_a,ny_a,V_fac,pdist_x,pdist_y,dmat_x,dmat_y)
            if n_dep:
                kplot += new_k*Lorentz_norm(nx_a * gamma[0] + ny_a * gamma[1], Eplot,DeltaE)/(2*np.pi)
            else:
                kplot += new_k*Lorentz_norm(gamma[0] + gamma[1], Eplot,DeltaE)/(2*np.pi)
    return kplot    
        
def ET_rate_dist_2D_Lor(p_cut,n_cut,omega,g,V_fac,nbar,gamma,Eplot,n_dep=False):
    '''
    calculate transfer rate in the perturbation regime for ET system with 2 modes

    Parameters
    ----------
    p_cut : int
        cutoff for each phonon space
    n_cut : int
        cutoff for fock state used for computing energy
    omega : list of float
        frequency of [x,y] mode
    g : list of float
        normalized sp coupling for [x,y] mode (gx/omega_x,gy/omega_y)
    V_fac : float
        coefficient for sigma_x 
    nbar : list of float
        average phonon numebr for [x,y] mode
    gamma: list of float
        dissipation rate for [x,y] mode
    Eplot: np array
        array of Delta E used for plot
    n_dep: bool
        if true, include the fock number of acceptor state
    Returns
    -------
    np array: transfer rate evaluated at each point of Eplot

    '''
    # result transfer rate plot
    kplot = np.zeros(np.shape(Eplot))
    # displacement opeartor for x,y mode
    dmat_x = displace(p_cut,-g[0]); dmat_y = displace(p_cut,-g[1])
    # thermal distribution for x,y mode
    pdist_x = phon.p_thermal(p_cut,nbar[0]); pdist_y = phon.p_thermal(p_cut,nbar[1])
    for nx_d, ny_d, nx_a, ny_a in product(range(n_cut), repeat=4):
        #check if the donor energ[1] is smaller than the acceptor (difference to be compensate by E)
        DeltaE = (nx_a-nx_d)*omega[0] + (ny_a-ny_d)*omega[1]
        if  (DeltaE>0 and pdist_x[nx_d]*pdist_y[ny_d]>0 ):
            new_k = ET_rate_point_2D(nx_d,ny_d,nx_a,ny_a,V_fac,pdist_x,pdist_y,dmat_x,dmat_y)
            if n_dep:
                kplot += new_k*Lorentz(nx_a * gamma[0] + ny_a * gamma[1], Eplot,DeltaE)
            else:
                kplot += new_k*Lorentz(gamma[0] + gamma[1], Eplot,DeltaE)
    return kplot

def ET_rate_Fermi(cutoff,E_split,g_fac,V_fac,nbar,state_type='thermal'):
    '''
    Compute normalized electron transfer rate 2pi k / omega_0
    based on Fermi golden rule, assuming V<<gamma.
    This model assumes Delta E = n omega_0
    g,V factors are in terms of hbar\omega
    Parameters
    ----------
    cutoff : int
        cutoff of SHO operator
    E_split : int
        Energy splitting factor, equals the number of 
        SHO energy levels required to compensate the energy gap
    gf : float
        effective s-p coupling factor
    Vf : float
        Site coupling factor, coefficient for sigma_x
    nbar : float
        Initial average phonon number
    state_type: str
        specify the type of initial state, 
         can be 'thermal' or 'fock'
    Returns
    -------
    Float
    '''
    if state_type == 'fock' and not(isinstance(nbar,int)):
        print('fock state phonon number must be integer')
        return 0
    else:
        return (2*np.pi)**2*V_fac**2*FC_sum(cutoff,E_split,g_fac,nbar,state_type)
def plot_ET_rate_Fermi(E_start,E_end,p_cutoff,g_fac,V_fac,nbar,state_type='thermal'):
    '''
    Plot normalized electron transfer rate 2pi k / omega_0
    based on Fermi golden rule for a set of energy splittings, 
    assuming V<<gamma.
    This function also assumes Delta E = n omega_0
    g,V factors are in terms of hbar\omega

    Parameters
    ----------
    E_start : int
        starting point of energy splitting, equals the number of 
        SHO energy levels required to compensate the energy gap
    E_end : int
        ending point of energy splitting, equals the number of 
        SHO energy levels required to compensate the energy gap
    cutoff : int
        cutoff of SHO operator
    gf : float
        effective s-p coupling factor
    Vf : float
        Site coupling factor, coefficient for sigma_x
    nbar : float
        Initial average phonon number
    state_type: str
        specify the type of initial state, 
         can be 'thermal' or 'fock'
    Returns
    -------
    None.

    '''
    if state_type == 'fock' and not(isinstance(nbar,int)):
        print('fock state phonon number must be integer')
        return 0
    else:
        rate_list = []
        splot = np.arange(E_start, E_end,1)
        for s in splot:
            rate_list.append(ET_rate_Fermi(p_cutoff,s,g_fac,V_fac,nbar,state_type))
        fig = plt.figure(figsize=(8, 6))
        plt.plot(splot,rate_list,'*',markersize=12)
        plt.xlabel(r'$\Delta E [\hbar\omega_0]$',fontsize = 16)
        plt.ylabel(r'$2 \pi k / \omega_0$',fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xticks(np.arange(0, E_end,1),fontsize = 16)
        #plt.xlim(1,12)
        #plt.legend(fontsize = 15)
        plt.grid()
        plt.show()

def ET_2level(t,V,gamma,nu,gf,cutoff):
    '''
    

    Parameters
    ----------
    t : float
        time for evaluation
    V : float
        Site coupling strength in 2pi kHz, coefficient for sigma_x
    gamma : float
        dissipation rate in 2pi kHz
    nu : int
        initial vibrational state
    gf : float
        effective s-p coupling factor
    cutoff : int
        cutoff of SHO operator
 
 

    Returns
    -------
    pd : TYPE
        DESCRIPTION.

    '''
    FC_coeff = FC_matrix(cutoff,gf)[0,nu]
    d = np.sqrt(gamma**2 - 4*V**2 * FC_coeff+0j)
    pd = np.exp(-gamma*t) * (np.cosh(d*t/2) + gamma/d * np.sinh(d*t/2))**2
    return pd


def ET_rate_QA(V,gamma,nu,gf,cutoff):
    '''
    

    Parameters
    ----------
    V : float
        Site coupling strength in 2pi kHz, coefficient for sigma_x
    gamma : float
        dissipation rate in 2pi kHz
    nu : int
        initial vibrational state
    gf : float
        effective s-p coupling factor
    cutoff : int
        cutoff of SHO operator
 
 

    Returns
    -------
    kQA : TYPE
        DESCRIPTION.

    '''
    FC_coeff = FC_matrix(cutoff,gf)[0,nu]
    eta = gamma/(np.abs(V)*np.sqrt((FC_coeff+0j)**2))
    kQA = gamma * (1 + eta**2)/(1 + (1/2)*eta**4)
    return kQA
        