# -*- coding: utf-8 -*-
"""
Compute basic physical quantities of ising coupling system 
Generate the Hamiltonian under pure spin approximation
functions: lc, Ita, Omega, Axialfreq, Axialmode, Transfreq, Transmode, Jt, plotj, HBz, Hps
@author: zhumj
"""
import numpy as np
from qutip import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon

def summary():
    print("____________________________________________________________________")
    print("function: lc")
    print("compute the length scale of the system, in SI")
    print("____________________________________________________________________")
    print("function: Ita")
    print("Compute single ion Lamb-Dicke parameter for the transvers COM mode.")
    print("____________________________________________________________________")
    print("function: Omega")
    print("compute side band rabi-rate")
    print("____________________________________________________________________")
    print("function: Axiafreq")
    print("compute the eigenfrequencies of Axial oscillation, multiply by wz to get real frequency [Hz]")
    print("____________________________________________________________________")
    print("function: Axiamode")
    print("compute the eigenmodes of Axial oscillation")
    print("____________________________________________________________________")
    print("function: Transfreq")
    print("compute the eigenfrequencies of transverse oscillation, multiply by wz to get real frequency [Hz]")
    print("____________________________________________________________________")
    print("function: Transmode")
    print("compute the eigenmodes of transverse oscillation")
    print("____________________________________________________________________")
    print("function: Jt")
    print("Compute Ising coupling matrix J")
    print("____________________________________________________________________")
    print("function: plotj")
    print("visiualize the matrix elements in Jt")
    print("____________________________________________________________________")
    print("function: HBz")
    print("compute the Hamiltonian coupled with z magnetic field    ")
    print("____________________________________________________________________")
    print("function: Hps")
    print("Compute Hamiltonian under a pure spin approximation, with ising coupling constructed only with sx and magentic field coupled with sz")
#Define phyiscal constants of the system
h = 6.62607015 * 10**(-34) / (2*np.pi)
MYb171 = 0.171 / (6.02214076*10**23) #mass of Yb ion, kg
qe = 1.60218 * 10**(-19) #charge of electron, C
eps0 = 8.85418781 * 10**(-12) #vacuum dielectric constant,SI
Dk = np.sqrt(2)*2*np.pi / (355*10**(-9)) 
#difference in wavevector projection of the 2 lasers m-1
R = (h*Dk**2) / (2*MYb171) #recoil frequency constant, SI 
'''
subfunctions
'''
def w(f):
    #transfer to the radial frequency in Hz
    #input fz (MHZ) 
    return 2*np.pi*10**6 * f
def X0(f0):
    #compute the characteristic length scale of the motional mode, [m]
    #input in MHz
    return np.sqrt(h / (2*MYb171* w(f0)))
#Compute Transverse and Axial Matrix, eigen modes of the system
def mask_p(m,N0):
    #generate a mask array for calculating mth diagonal element
    mlist = np.ones(N0)
    mlist[m] = 0
    return mlist
#compute equilirium positions
def E_index(m,N):
    #generate the indexes for summation for N ions and we are considering the 
    # mth ion, m,N must be intergers and N>m >=2 
    e_list = np.zeros(N)
    e_list[0:m-1] = -1
    e_list[m:N] = 1
    return e_list
def ef_term(xm,xi):
    #compute the term of electric force with respect to xi, xi is the position
    #of another ion, xm is the mth index of the position vector
    return (1/(xm-xi))**2
def E_eq(x):
    #generate the list of equations to be solved, x is the position vector of 
    #the system(list)
    eqlist = []
    for m in range(len(x)):
        elist = E_index(m+1,len(x))
        new_eq = x[m]
        for i in range(len(x)):
            if elist[i] != 0:
                new_eq =  new_eq + elist[i] * ef_term(x[m],x[i])
        eqlist.append(new_eq)
    return eqlist                    
def E_position(N,fz,scale):
    '''
    Compute the equilibrium position of the 1D ion chain centered at 0 
    Input: 
        fz, axial frequency of the ion trap
        N, number of ions in the system
    Output:
        np array of N elements representing the equilibrium position of each ion
    '''
    x0 = np.zeros(N)
    if N % 2 == 0:
        x0[0:int(N/2)] = np.arange(-N/2,0)
        x0[int(N/2):N] = np.arange(1,(N/2)+1)
    else:
        x0 = np.arange((-N-1)/2,(N-1)/2+1)
    esol = fsolve(E_eq, x0)
    if scale:
        return np.array(lc(fz)*esol)
    else:
        return np.array(esol)
#print(E_position(1, 3))
def Aele(N0,m,n,epos):
    #compute of matrix A at index m 
    ele_val = 0 
    if m == n:
        mlist = mask_p(m,N0)
        for p in range(N0):
            if mlist[p] != 0:
                ele_val = ele_val + 2/(np.abs(epos[m]-epos[p]))**3
        ele_val = ele_val + 1        
    else:
        ele_val = -2/(np.abs(epos[m]-epos[n]))**3    
    return ele_val
def Amatrix(N,fz):
    #compute the tensor A which determines the axial oscillation
    #fz, axial frequency of the ion trap
    eposition = E_position(N, fz,False) 
    Amat = np.zeros((N,N))
    for m in range(N):
        for n in range(N):
            Amat[m,n] = Aele(N,m,n,eposition)
    return Amat
def Tmatrix(N,fz,fx):
    #compute the tensor B that determines transverse oscillation
    #fx, transverse frequency of the ion trap
    Amat = Amatrix(N,fz)
    Tmat = (0.5+(fx/fz)**2) * np.identity(N) - 0.5*Amat
    return Tmat
'''
functions to use
'''
def lc(fz):
    '''
    compute the length scale of the system, in SI
    input(fz)
    Parameters
    ----------
    fz : flaot
        axial frequency of the ion trap, [MHz]

    Returns
    -------
    float, unit SI

    '''
    return (qe**2/ (4*np.pi * eps0 * MYb171 * w(fz)**2))**(1/3)
def Ita(fx):
    '''
    Compute single ion Lamb-Dicke parameter for the transvers COM mode.
    input(fx)
    Parameters
    ----------
    fx : float
        transverse frequency of the ion trap, [MHz]

    Returns
    -------
    float, unit SI

    '''
    return Dk * X0(fx) 
def Omega(fs,fx):
    '''
    Compute side band rabi-rate
    Input(fs,fx)
    Parameters
    ----------
    fs : float
        sideband rabi frequency, [kHz]
    fx : float
        transverse frequency of the ion trap, [MHz]

    Returns
    -------
    float, unit SI

    '''
    return 2*np.pi*10**3*fs / Ita(fx)  

def Axialfreq(N,fz):
    '''
    compute the eigenfrequencies of axial oscillation, multiply by wz to get real frequency [Hz]
    input(N,fz)
    Parameters
    ----------
    N : int
        number of ions in the system
    fz : float
        axial frequency of the ion trap, [MHz]

    Returns
    -------
    np array object, each index is an axial eigenfrequency

    '''
    e_val = np.linalg.eig(Amatrix(N,fz))[0]
    return np.sqrt(e_val)
def Axialmode(N,fz):
    '''
    compute the eigenmodes of axial oscillation 
    input(N,fz)
    Parameters
    ----------
    N : int
        number of ions in the system
    fz : float
        axial frequency of the ion trap, [MHz]

    Returns
    -------
    np array object that represents N by N matrix, each row is an axial eigenmode

    '''
    e_array = np.linalg.eig(Amatrix(N,fz))[1]
    return np.transpose(e_array)
def Transfreq(N,fz,fx):
    '''
    compute the eigenfrequencies of transverse oscillation, multiply by wz to get real frequency [Hz]
    input(N,fz,fx)
    Parameters
    ----------
    N : int
        number of ions in the system
    fz : float
        axial frequency of the ion trap, [MHz]
    fx : float
        transverse frequency of the ion trap, [MHz]

    Returns
    -------
    np array object, each index is an transverse eigenfrequency

    '''
    e_val = np.linalg.eig(Tmatrix(N,fz,fx))[0]
    #check if the matrix is positive-definite
    if np.min(e_val) < 0:
        print("Negtive transverse frequency, the system is unstable")
    return np.sqrt(e_val)
def Transmode(N,fz,fx):
    '''
    compute the eigenmode of transverse oscillation
    input(N,fz,fx)
    Parameters
    ----------
    N : int
        number of ions in the system
    fz : float
        axial frequency of the ion trap, [MHz]
    fx : float
        transverse frequency of the ion trap, [MHz]

    Returns
    -------
    np array object that represents N by N matrix, each row is an transverse eigenmode

    '''
    e_array = np.linalg.eig(Tmatrix(N,fz,fx))[1]
    return np.transpose(e_array)
#Compute Ising coupling matrix J
def Jt(fr,fb,N,fz,fx,delta0):
    '''
    Compute Ising coupling matrix J
    Input: (fr,fb,N,fz,fx,delta0)
    Parameters
    ----------
    fr : float
        red side band rabi-frequency [kHz]
    fb : float
        blue side band rabi-frequency [kHz]
    N : TYPE
        number of ions in the system
    fz : float
        axial frequency of the ion trap, [MHz]
    fx : float
        transverse frequency of the ion trap, [MHz]
    delta0 : float
        detuning, defined as the deviation from transverse COM frequency [kHz]

    Returns
    -------
    np array object that represents N by N matrix J

    '''

    wx = w(fx); wz = w(fz); sdelta = delta0*2*np.pi*10**3
    Omegar = Omega(fr,fx); Omegab = Omega(fb,fx)
    Omega0 = Omegar*Omegab
    nfreq = Transfreq(N,fz,fx);emat = Transmode(N,fz,fx)
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                eij = 0
                for m in range (N):
                    numer = R*Omega0 * emat[m,i] * emat[m,j]
                    demon = (sdelta+wx)**2 - (wz*nfreq[m])**2
                    eij = eij + (numer/demon)
                J[i,j] = eij/(2*np.pi*10**3)    
    return J
def plotj(J):
    '''
    visiualize the matrix elements in J
    Parameters
    ----------
    J : np array
        np array object that represents N by N matrix J, output of function Jt

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ndim = np.shape(J)[0]
    plot_c = np.arange(1,ndim+1)
    xx, yy = np.meshgrid(plot_c,plot_c)
    xcoord = xx.ravel(); ycoord = yy.ravel()
    ax1.bar3d(xcoord, ycoord, np.zeros_like(xcoord), 1, 1, J.ravel(), shade=True)
    ax1.set_title('Coupling matrix J')
    ax1.set_xlabel('i index')
    ax1.set_ylabel('j index')
def HBz(N,B0):
    '''
    compute the Hamiltonian due to coupling with z magnetic field
    input(N,B0)
    Parameters
    ----------
    N : int
        number of ions in the system
    B0 : float
        effective field strength

    Returns
    -------
    Qutip operator

    '''   
    H = spin.zero_op(N)
    for i in range(N):
        H = H + B0 * spin.sz(N,i) 
    return 2*np.pi*H    
def Hps(J,N,B0):
    '''
    Compute Hamiltonian under a pure spin approximation, with ising coupling constructed only with sx and magentic field coupled with sz
    input(J,N,B0)
    Parameters
    ----------
    J : np array
        np array object that represents N by N matrix J, output of function Jt
    N : int
        number of ions in the system    
    B0 : float
        effective field strength

    Returns
    -------
    Qutip operator

    ''' 
    H = spin.zero_op(N)
    for i in range(1,N):
        submat = spin.zero_op(N)
        for j in range(i):
            submat = submat + J[i,j]*spin.sx(N,i)*spin.sx(N,j)
        H = H + submat
    return 2*np.pi*H +  HBz(N,B0)
 
