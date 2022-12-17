# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022

@author: zhumj
basic physical parameters of 1D N ion system with multiple lasers 
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from scipy.optimize import fsolve
#________________________________________________________________________
'''
Define phyiscal constants of the system
'''
h = 6.62607015 * 10**(-34) / (2*np.pi)
MYb171 = 0.171 / (6.02214076*10**23) #mass of Yb ion, kg
qe = 1.60218 * 10**(-19) #charge of electron, C
eps0 = 8.85418781 * 10**(-12) #vacuum dielectric constant,SI
Dk = np.sqrt(2)*2*np.pi / (355*10**(-9)) 
#difference in wavevector projection of the 2 lasers [m-1]
R = (h*Dk**2) / (2*MYb171) #recoil frequency constant, SI 
def fr_conv(f,unit):
    '''
    convert frequency to radial frequency 
    Parameters
    ----------
    f : float
        input frequency, in Hz, kHz, or MHz
    unit : str
        specify the unit, can be 'hz', 'khz', 'mhz'
    Returns
    -------
    f_out : float
        output radial frequency, unit of 2pi Hz

    '''
    factor = {'hz':0,'khz':3,'mhz':6}
    f_out = 2*np.pi * f*10**(factor[unit])
    return f_out
'''
subfunction
'''
def X0(f0):
    #compute the characteristic length scale of the motional mode, [m]
    #input in MHz
    return np.sqrt(h / (2*MYb171* fr_conv(f0,'mhz')))
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
def E_position(N,fz,scale,com_scale):
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
        x0 = np.arange((-N-1)/2+1,(N-1)/2+1)
    esol = fsolve(E_eq, x0)
    if scale:
        return np.array(com_scale*esol)
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
    eposition = E_position(N, fz,False,0) 
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
functions to use for this module
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
    return (qe**2/ (4*np.pi * eps0 * MYb171 * fr_conv(fz,'mhz')**2))**(1/3)
def eta(f):
    '''
    Compute single ion Lamb-Dicke parameter for the a transvers mode.
    input(fx)
    Parameters
    ----------
    fx : float
        transverse frequency of the ion trap, [MHz]

    Returns
    -------
    float, unit SI

    '''
    return Dk * X0(f) 
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
    return fr_conv(fs,'khz') / eta(fx)  
def summary():
    '''
    give a summary of all functions and classes defined in this module
    '''
    print('Class:')
    print('___________________________________________________________________')
    print('ions')
    print('class used to store the parameters of a 2 ion system for electron transition simulation')
    print("____________________________________________________________________")
    print("function: lc")
    print("compute the length scale of the system, in SI")
    print("____________________________________________________________________")
    print("function: eta")
    print("Compute single ion Lamb-Dicke parameter for the transvers COM mode.")
    print("____________________________________________________________________")
    print("function: Omega")
    print("compute side band rabi-rate")
class ions:
    '''
    this class is used to store the parameters of a 2 ionsystem used for 
    electron transition simulation
    Parameters
    g : float
        state adaptive dipole force, correpsonds to eta*Omega in the ion system [kHz]   
    Omegax : float
        Rabi frequency, correpond to electronic coupling strength V [kHz]    
    N : int
        number of ions in the system
    n_laser: int
        number of laser drives applied    
    pcut : int
        cut off level of the harmonic ocsillator eigenenergy
    fz : float
        axial frequency of the ion trap, [MHz]
    fx : float
        transverse frequency of the ion trap, [MHz]
    fr: float
         red side band frequency [kHz]     
    fb: float
         blue side band frequency [kHz] 
    phase: float
        spin phase phis [rad]    
    Etot: float
        total energy of one ion    
    '''
    '''
    default value of parameters
    '''
    N = 2  #number of ions
    n_laser = 1
    delta = 20 #detuning 
    fx = 2 
    fz = 1
    fb = 10
    fr = 10
    phase = 0
    Omegax =  0.1 * 20
    gamma = [0.1 * 20, 0.1*20]
    Etot = fr_conv(0.217*2,'khz')
    pcut = [15,15] #cutoff of phonon energy for distinctive modes
    delta_ref = 1 #reference frequency index, 0 for com frequency
    def list_para(self):
        '''
        list basic physical parameters of the system

        '''
        print('________________________________________________________________')
        print('number of ions', self.N)
        print('number of laser drives applied', self.n_laser)
        print('phonon cutoff ', self.pcut)
        print('avearge phonon number ', np.round(self.n_bar(),4))
        print('axial COM (Confining) frequency ',np.round(self.fz,2),' [MHz]')
        print('transverse COM (Confining) frequency ',np.round(self.fx,2), '[MHz]')
        print('detuning delta (measured as deviation from transverse COM freq) ',np.round(self.delta,2)," [kHz]")
        print('red side band rabi frequency ', np.round(self.fr,2),' [kHz]')
        print('blue side band rabi frequency ', np.round(self.fb,2),' [kHz]')
        print('spin phase phis',np.round(self.phase*180/np.pi,2))
        print('(input in rad but displayed in degs)')
        print('phononic eigenfrequency', np.round(self.wmlist(),2),'MHz')
        print('cooling rate ', np.round(self.gamma,2)," [kHz]") 
        print('detuning frequency index: ', self.delta_ref)
        print('detuning from eigenfrequency',np.round(self.dmlist()/(2*np.pi),2),'kHz')
    def list_elec_para(self):
        '''
        list parameters used for electron transfer
        '''
        print('________________________________________________________________')
        print('electronic coupling strength, or rabi frequency Omega_x ',np.round(self.Omegax,2), ' [kHz]')
        print('state adaptive dipole force g, or eta*Omega ', np.round(self.g()/(2*np.pi),2),' [kHz]')
    def plot_freq(self):
        '''
        visualize eigenfreqencies and laser frequency
        '''
        wmlist  = self.wmlist()*1000
        lab0 = r'$f_{com}$ = ' + str(np.round(wmlist[2],1)) + 'kHz'
        lab1 = r'$f_{tilt}$ = ' + str(np.round(wmlist[1],1)) + 'kHz'
        lab2 = r'$f_{rock}$ = ' + str(np.round(wmlist[0],1)) + 'kHz'
        froc =  [wmlist[2], wmlist[2]]
        ftil =  [wmlist[1], wmlist[1]]
        fcom =  [wmlist[0], wmlist[0]]
        ylist = [0,1]
        title = r'$\delta = $' + str(self.delta) + 'kHz, reference: '+str(self.delta_ref)
        plt.figure(0)
        plt.plot(froc,ylist,label = lab0)
        plt.plot(ftil,ylist,label = lab1)
        plt.plot(fcom,ylist,label = lab2)
        if self.n_laser == 1:
            las = wmlist[self.delta_ref]+self.delta    
            labl = r'$f_{laser}$ = ' + str(np.round(las ,1)) + 'kHz'
            flas =  [las, las]
            plt.plot(flas,ylist,'--',label = labl)
        else:    
            for i in range(self.n_laser):    
                las = wmlist[self.delta_ref[i]]+self.delta[i]    
                labl = r'$f_{laser}$ = ' + str(np.round(las ,1)) + 'kHz'
                flas =  [las, las]
                plt.plot(flas,ylist,'--',label = labl)
        plt.ylim(0,1)
        plt.title(title)
        plt.xlabel(r'frequecny kHz')
        plt.grid(b=None, which='major', axis='x', color = 'blue', linestyle = '--')
        plt.legend()
    def Equi_posi(self):
        '''
        compute the equilibrium position of 1-D ion-chain

        Returns
        -------
        np array object, each index a equilibirum position

        '''
        return E_position(self.N,self.fz,False,0) 
    def Axialfreq(self):
        '''
        compute the eigenfrequencies of axial oscillation, multiply by wz to get real frequency [Hz]
        input(N,fz)
       
        Returns
        -------
        np array object, each index is an axial eigenfrequency

        '''
        e_val = np.linalg.eig(Amatrix(self.N,self.fz))[0]
        order = np.argsort(e_val)
        e_val = e_val[order]
        return np.sqrt(e_val)
    def Axialmode(self):
        '''
        compute the eigenmodes of axial oscillation 
        input(N,fz)
        Returns
        -------
        np array object that represents N by N matrix, each row is an axial eigenmode

        '''
        e_val = np.linalg.eig(Amatrix(self.N,self.fz))[0]
        order = np.argsort(e_val)
        e_array = np.linalg.eig(Amatrix(self.N,self.fz))[1][order]
        return np.transpose(e_array)
    def Transfreq(self):
        '''
        compute the eigenfrequencies of transverse oscillation, multiply by wz to get real frequency [Hz]
        input(N,fz,fx)
        Returns
        -------
        np array object, each index is an transverse eigenfrequency

        '''
        e_val = np.linalg.eig(Tmatrix(self.N,self.fz,self.fx))[0]
        order = np.argsort(e_val)
        e_val = e_val[order]
        #check if the matrix is positive-definite
        if np.min(e_val) < 0:
            print("Negtive transverse frequency, the system is unstable")
        return np.sqrt(e_val)
    def Transmode(self):
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
        e_val = np.linalg.eig(Tmatrix(self.N,self.fz,self.fx))[0]
        order = np.argsort(e_val)
        e_array = np.linalg.eig(Tmatrix(self.N,self.fz,self.fx))[1][order]
        return np.transpose(e_array)    
    def wmlist(self):
        '''
        compute transverse eigenfrequencies of the system
        Returns
        -------
        np array of float MHz in increasing order
        '''
        return self.Transfreq()*self.fz
    def dmlist(self):
        '''
        compute the detuning from the two modes, decreasing order
        Returns
        -------
        float 2pi kHz (angular)
        '''
        wlist0 = self.wmlist()*1000
        if self.n_laser == 1:
            mu = (wlist0[self.delta_ref] + self.delta)
            dm = 2*np.pi*(mu - wlist0)
        else:
            dm = np.array([])
            for i in range(self.n_laser):
                mu = (wlist0[self.delta_ref[i]] + self.delta[i])
                dm = np.append(dm,2*np.pi*(mu - wlist0))
        return dm
    def n_bar(self):
        '''
        compute the average phonon number for a given set of phonon states
        ----------
        Returns
        -------
        float, no unit

        '''
        return 1/(np.exp(1000*(2*np.pi)*(self.fx)/self.Etot)-1)
    def w0(self):
        '''
        compute the time scale of the system, defined as 2pi*delta
        ----------
        Returns
        -------
        float, 2pi kHz
        '''
        return (2*np.pi)*np.abs(self.delta)
    def Omega(self):
        '''
        compute the rabi rate of the system
        ----------
        Returns
        -------
        float,2pi kHz
        '''
        return np.sqrt(Omega(self.fr,self.fx)*Omega(self.fb,self.fx))/1000
    #functions defined to compute electron transfer
    def g(self):
        
        '''
        compute the state adaptive dipole force g
        ----------
        Returns
        -------
        float, 2pi kHz
        '''
        emat = self.Transmode()
        coeff = eta(self.wmlist())*self.Omega()
        garray = np.array([])
        for i in range(self.N):
            garray = np.append(garray,coeff[i]*emat[i,0])
        return garray
    def expeak(self):
        '''
        compute the expected peak frequencies
        '''
        print(np.round(np.abs(self.dmlist())/(4*np.pi),2))
    def Lambda(self):
        '''
        Compute the reorgonization energy lambda, which also correponds to energy
        splitting that leads to the maximum transfer rate
        Returns
        -------
        float [J/10**6]
        '''
        return self.g()**2/np.abs(self.w0())