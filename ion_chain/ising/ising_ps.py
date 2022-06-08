# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 16:38:22 2022

@author: zhumj
"""
import numpy as np
from qutip import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import ion_chain.operator.spin as spin
import ion_chain.operator.phonon as phon

'''
Compute basic physical quantities of ising coupling system and generate the 
Hamiltonian under pure spin approximation
'''
def list_func():
    timer = 0
    while timer == 0:
        print("type the corresponding index to see the description for functions in this package")
        print("index                           function")
        print("0                               summary for all functions")
        print("1                               lc")
        print("2                               Ita")
        print("3                               Omega")
        print("4                               Axiafreq")
        print("5                               Axiamode")
        print("6                               Transfreq")
        print("7                               Transmode")
        print("8                               Jt")
        print("9                               plotj")
        print("10                              HBz")
        print("11                              Hps")
        index = int(input("Enter function index: "))
        if (index == 0):
            print("summarys for all functions in this module")
            print("1                               lc")
            print("compute the length scale of the system, in SI")
            print("2                               Ita")
            print("Compute single ion Lamb-Dicke parameter for the transvers COM mode.")
            print("3                               Omega")
            print("compute side band rabi-rate")
            print("4                               Axiafreq")
            print("compute the eigenfrequencies of Axial oscillation, multiply by wz to get real frequency [Hz]")
            print("5                               Axiamode")
            print("compute the eigenmodes of Axial oscillation")
            print("6                               Transfreq")
            print("compute the eigenfrequencies of transverse oscillation, multiply by wz to get real frequency [Hz]")
            print("7                               Transmode")
            print("compute the eigenmodes of transverse oscillation")
            print("8                               Jt")
            print("Compute Ising coupling matrix J")
            print("9                               plotj")
            print("visiualize the matrix elements in Jt")
            print("10                              HBz")
            print("compute the Hamiltonian coupled with z magnetic field    ")
            print("11                              Hps")
            print("Compute Hamiltonian under a pure spin approximation, with ising coupling constructed only with sx and magentic field coupled with sz")
        if (index == 1): 
            print("____________________________________________________________________")
            print("function: lc")
            print("compute the length scale of the system, in SI")
            print("Input: (fz)")
            print("fz: float, axial frequency of the ion trap, MHz")
            print("Output: float, unit SI")
        if (index == 2): 
            print("____________________________________________________________________")
            print("function: Ita")
            print("Compute single ion Lamb-Dicke parameter for the transvers COM mode.")
            print("Input: (fx)")
            print("fx:  float, transverse frequency of the ion trap, MHz")
            print("Output: float, unit SI")
        if (index == 3): 
            print("____________________________________________________________________")
            print("function: Omega")
            print("compute side band rabi-rate")
            print("Input: (fs,fx)")
            print("fs:  sideband rabi frequency, kHz")
            print("fx:  float, transverse frequency of the ion trap, MHz")
            print("Output: float  unit SI")
        if (index == 4):     
            print("____________________________________________________________________")
            print("function: Axiafreq")
            print("compute the eigenfrequencies of Axial oscillation, multiply by wz to get real frequency [Hz]")
            print("Input: (N,fz)")
            print("N: int, number of ions in the system")
            print("fz: float, axial frequency of the ion trap, MHz")
            print("Output: np array object, each index is an axial eigenfrequency")
        if (index == 5):      
            print("____________________________________________________________________")
            print("function: Axiamode")
            print("compute the eigenmodes of Axial oscillation")
            print("Input: (N,fz)")
            print("N: int, number of ions in the system")
            print("fz: float, axial frequency of the ion trap, MHz")
            print("N: int, number of ions in the system")
            print("Output: np array object that represents N by N matrix, each row is an axial eigenmode")
        if (index == 6): 
            print("____________________________________________________________________")
            print("function: Transfreq")
            print("compute the eigenfrequencies of transverse oscillation, multiply by wz to get real frequency [Hz]")
            print("Input: (N,fz,fx)")
            print("N: int, number of ions in the system")
            print("fz: float, axial frequency of the ion trap, MHz")
            print("fx: float, transverse frequency of the ion trap, MHz")
            print("N: int, number of ions in the system")
            print("Output: np array object, each index is an transverse eigenfrequency")
        if (index == 7):     
            print("____________________________________________________________________")
            print("function: Transmode")
            print("compute the eigenmodes of transverse oscillation")
            print("Input: (N,fz,fx)")
            print("N: int, number of ions in the system")
            print("fz: float, axial frequency of the ion trap, MHz")
            print("fx: float, transverse frequency of the ion trap, MHz")
            print("Output: np array object that represents N by N matrix, each row is an transverse eigenmode")
        if (index == 8): 
            print("____________________________________________________________________")
            print("function: Jt")
            print("Compute Ising coupling matrix J")
            print("Input: (fr,fb,N,fz,fx,delta)")
            print("fr, float, red side band rabi-frequency [kHz]")
            print("fb, float, blue side band rabi-frequency [kHz]")
            print("N: int, number of ions in the system")
            print("fz: float, xial frequency of the ion trap, MHz")
            print("fx: float, transverse frequency of the ion trap, MHz")
            print("delta: float, detuning, kHz")
            print("Output: np array object that represents N by N matrix J")
        if (index == 9): 
            print("____________________________________________________________________")
            print("function: plotj")
            print("visiualize the matrix elements in Jt")
            print("Input: (J)")
            print("J: np array object that represents N by N matrix J, output of function Jt")
        if (index == 10): 
            print("____________________________________________________________________")
            print("function: HBz")
            print("compute the Hamiltonian coupled with z magnetic field    ")
            print("Input: (N,B0)")
            print("N: int, number of ions in the system")
            print("B0: float, effective field strength")
            print("Output: Qutip operator")    
        if (index == 11): 
            print("____________________________________________________________________")
            print("function: Hps")
            print("Compute Hamiltonian under a pure spin approximation, with ising coupling constructed only with sx and magentic field coupled with sz")
            print("Input: (J,N,B0)")
            print("J: np array object that represents N by N matrix J, output of function Jt")
            print("N: int, number of ions in the system")
            print("B0: float, effective field strength")
            print("Output: Qutip operator")
        cont = int(input("Enter 0 to quit the help section, 1 to look at other functions: "))    
        if (cont == 0):
            break
#Define phyiscal constants of the system
h = 6.62607015 * 10**(-34) / (2*np.pi)
MYb171 = 0.171 / (6.02214076*10**23) #mass of Yb ion, kg
qe = 1.60218 * 10**(-19) #charge of electron, C
eps0 = 8.85418781 * 10**(-12) #vacuum dielectric constant,SI
Dk = np.sqrt(2)*2*np.pi / (355*10**(-9)) 
#difference in wavevector projection of the 2 lasers m-1
R = (h*Dk**2) / (2*MYb171) #recoil frequency constant, SI 
def w(f):
    #transfer to the radial frequency in Hz
    #input fz0 (MHZ) 
    return 2*np.pi*10**6 * f
def lc(fz0):
    #compute the length scale of the system, SI
    #input in MHz
    return (qe**2/ (4*np.pi * eps0 * MYb171 * w(fz0)**2))**(1/3)
def X0(f0):
    #compute the characteristic length scale of the motional mode, [m]
    #input in MHz
    return np.sqrt(h / (2*MYb171* w(f0)))
def Ita(fx0):
    #single ion Lamb-Dicke parameter for the transvers COM mode. SI
    #input in MHz
    return Dk * X0(fx0) 
def Omega(fs,fx0):
    #compute side band rabi-rate
    #input rabi frequency fs in kHz, fx in MHz
    return 2*np.pi*10**3*fs / Ita(fx0)  
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
def E_position(N,fz0,scale):
    '''
    Compute the equilibrium position of the 1D ion chain centered at 0 
    Input: 
        fz0, axial frequency of the ion trap
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
        return np.array(lc(fz0)*esol)
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
def Amatrix(N,fz0):
    #compute the tensor A which determines the axial oscillation
    #fz0, axial frequency of the ion trap
    eposition = E_position(N, fz0,False) 
    Amat = np.zeros((N,N))
    for m in range(N):
        for n in range(N):
            Amat[m,n] = Aele(N,m,n,eposition)
    return Amat
def Tmatrix(N,fz0,fx0):
    #compute the tensor B that determines transverse oscillation
    #fx0, transverse frequency of the ion trap
    Amat = Amatrix(N,fz0)
    Tmat = (0.5+(fx0/fz0)**2) * np.identity(N) - 0.5*Amat
    return Tmat
def Axiafreq(N,fz0):
    #compute the eigenfrequencies of Axial oscillation
    #multiply by wz to get real frequency [Hz]
    e_val = np.linalg.eig(Amatrix(N,fz0))[0]
    return np.sqrt(e_val)
def Axiamode(N,fz0):
    #compute the eigenmodes of Axial oscillation
    e_array = np.linalg.eig(Amatrix(N,fz0))[1]
    return np.transpose(e_array)
def Transfreq(N,fz0,fx0):
    #compute the eigenfrequencies of transverse oscillation
    #multiply by wz to get the real frequency [Hz]
    e_val = np.linalg.eig(Tmatrix(N,fz0,fx0))[0]
    #check if the matrix is positive-definite
    if np.min(e_val) < 0:
        print("Negtive transverse frequency, the system is unstable")
    return np.sqrt(e_val)
def Transmode(N,fz0,fx0):
    #compute the eigenmodes of transverse oscillation
    e_array = np.linalg.eig(Tmatrix(N,fz0,fx0))[1]
    return np.transpose(e_array)
#Compute Ising coupling matrix J
def Jt(fr0,fb0,N,fz0,fx0,delta0):
    '''
    Compute Ising coupling matrix J
    Input: 
        fr, red side band rabi-frequency [kHz]
        fb, blue side band rabi-frequency [kHz]
        N, int, #of ions in the system
        fz, axial frequency of the ion trap
        fx, transverse frequency of the ion trap
        delta, detuningm kHz
    Output:
        np array that gives J
    '''
    wx = w(fx0); wz = w(fz0); sdelta = delta0*2*np.pi*10**3
    Omegar = Omega(fr0,fx0); Omegab = Omega(fb0,fx0)
    Omega0 = Omegar*Omegab
    nfreq = Transfreq(N,fz0,fx0);emat = Transmode(N,fz0,fx0)
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
def plotj(Jt):
    #visiualize the matrix elements in Jt
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ndim = np.shape(Jt)[0]
    plot_c = np.arange(1,ndim+1)
    xx, yy = np.meshgrid(plot_c,plot_c)
    xcoord = xx.ravel(); ycoord = yy.ravel()
    #print(Jt.ravel())
    ax1.bar3d(xcoord, ycoord, np.zeros_like(xcoord), 1, 1, Jt.ravel(), shade=True)
    ax1.set_title('Coupling matrix J')
    ax1.set_xlabel('i index')
    ax1.set_ylabel('j index')
def HBz(N,B0z):
    #compute the Hamiltonian coupled with z magnetic field    
    H = spin.zero_op(N)
    for i in range(N):
        H = H + B0z * spin.sz(N,i) 
    return 2*np.pi*H    
def Hps(J0,N,B0z):
    #compute the Hamiltonian under pure spin approximation     
    H = spin.zero_op(N)
    for i in range(1,N):
        submat = spin.zero_op(N)
        for j in range(i):
            submat = submat + J0[i,j]*spin.sx(N,i)*spin.sx(N,j)
        H = H + submat
    return 2*np.pi*H +  HBz(B0z)
 