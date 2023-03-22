# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022

@author: zhumj
A class that contains basic physical and computational parameters of 1-D N ion system with 1 laser drive
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
Dk = np.sqrt(2)*2*np.pi / (355*10**(-9)) #effective wave vector 355nm laser drive
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
    float, unit of m

    '''
    return (qe**2/ (4*np.pi * eps0 * MYb171 * fr_conv(fz,'mhz')**2))**(1/3)
def eta(f):
    '''
    Compute single ion Lamb-Dicke parameter for vibrational eigenmode correspond to
    eigenfrequency f
    input(f)
    Parameters
    ----------
    f : float
       eigenfrequency [MHz]

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
    float, unit [2pi Hz]

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
    this class is used to store the parameters of a N ion system couples to a single
    laser drive in Axial or Radial direction, assuming the two radial directions are equivalent
    Parameters  
    N : int
        number of ions in the system
    pcut : int
        cut off level of the harmonic ocsillator eigenenergy
    fz : float
        axial frequency of the ion trap, [MHz]
    fx : float
        transverse frequency of the ion trap, [MHz]
    fr: float
         effective red side band frequency Omega * dK * X_0(fx) [kHz],      
    fb: float
         effective blue side band frequency [kHz] 
    phase: float
        spin phase phis [rad]    
    active_phonon: list of list of int
        index phonon space that will be considered
        initialized in form [[axial]]/[[radial]] if only 1 vibration direction is considered
        initialized in form [[axial],[radial]] if both vibrational degrees of freedoms are considered.
        0 always means com mode
        #for instance, [[1,2],[0,2]] means consider the tilt, rock mode for axial motion
        #com, rock mode for 1 radial modtion
    pcut: list of list of int
        cutoff of phonon space size for each phonon space being considered, 
        initialized in form [[axial]]/[[radial]] if only 1 vibration direction is considered
        initialized in form [[axial],[radial]] if both vibrational degrees of freedoms are considered.
        Note for the sublist, first index is always for com mode
        Needs to be consistent with active_phonon
    coolant: list of int
        index of coolant    
    df_laser: int   
        vibrational degree of freedom coupled to the laser, 0 for axial, 1 for radial
    laser_couple: list of int
        ion index that couples to the laser, for instance [0,1] means couple to ion 0, 1
    delta_ref: int 
        reference frequency index for defining laser detuning, 0 for com mode
    
    Etot: float
        total energy of one ion    
    '''
    '''
    default value of parameters
    '''
    N = 2  #number of ions
    delta = 20 #detuning 
    fx = 2 
    fz = 1
    fb = 10
    fr = 10
    phase = 0
    gamma = [0.1 * 20, 0.1*20]
    active_phonon = [[0,1,2]]  #index phonon space that will be considered, if more than 1 space,
    pcut = [[2,2,2]] #cutoff of phonon energy for distinctive modes, first index is always for com mode
    coolant = [2] #index of coolant
    df_laser = 1 #vibrational degree of freedom coupled to the laser, 0 for axial, 1 for radial
    laser_couple = [0,1] #ion index that couples to the laser, for instance [0,1] means couple to ion 0, 1
    delta_ref = 0 #reference frequency index, 0 for com frequency
    def list_para(self):
        '''
        list basic physical parameters of the system

        '''
        Coup_dic = {0:'Axial', 1:'Transverse (Radial)'}
        freqdic = {'0':'COM freq','1':'tilt freq','2':'rock freq'}
        print('________________________________________________________________')
        print('********************Setup of the Trap************************')
        print('number of ions', self.N)
        print('axial COM (Confining) frequency ',np.round(self.fz,2),' [MHz]')
        print('transverse COM (Confining) frequency ',np.round(self.fx,2), '[MHz]')
        print('index of ions that couple to the laser field: ',self.laser_couple)
        print('Axial vibrational eigenfrequency', np.round(self.Axialfreq()*self.fz,2),'MHz')
        print('Transverse(Radial) vibrational eigenfrequency', np.round(self.Transfreq()*self.fz,2),'MHz')
        print('                                                                 ')
        print('********************Parameters of Laser Drive************************')
        print('Vibrational degree of freedom couples to the laser: '+ Coup_dic[self.df_laser])
        print('detuning delta (measured as deviation from transverse'+freqdic[str(self.delta_ref)]
              ,np.round(self.delta,2)," [kHz]")
        print('detuning from eigenfrequency',np.round(self.dmlist()/(2*np.pi),2),'kHz')
        print('detuning frequency index: ', self.delta_ref)
        print('red side band rabi frequency ', np.round(self.fr,2),' [kHz]')
        print('blue side band rabi frequency ', np.round(self.fb,2),' [kHz]')
        print('Estimated spin-phonon coupling strength:', np.round(self.g(0,0)/(2*np.pi),2),' [kHz]')
        print('                                                                 ')
        print('********************Config of Numeric Calculation************************')
        print('index of phonon space included in simulation: ',self.active_phonon )
        print('corresonding phonon space cutoff ', self.pcut)
        print('spin phase phis',np.round(self.phase*180/np.pi,2))
        print('(input in rad but displayed in degs)')
        print('cooling rate ', np.round(self.gamma,2)," [kHz]") 
    def check_phonon(self):
        '''
        Check the consistency in set up of phonon space

        '''
        print('_____________________________________________________________')
        print('Checking phonon space setup')
        checker = 1
        if (len(self.pcut)==len(self.active_phonon)):
            for i in range(len(self.pcut)):
                if len(self.pcut[i]) != len(self.active_phonon[i]):
                    checker = 0
        else:
            checker = 0
        if checker == 0:    
            print('Inconsistency between active phonon space and assigned phonon cutoff')
        else:
            print('consistent')
        print('_____________________________________________________________')    
    def df_spin(self):
        '''
        number of spin degree of freedom to be considered

        '''
        return len(self.laser_couple)
    def df_phonon(self):
        '''
        output parameteres to construct phonon space
        Returns
        -------
        ph_space : list 
            the first element is the number of degree of freedoms considered.
            the second second element is list in which each element is the 
            number of phonon space to be considered for a specfic degreea of freedom
            for instance [2, [3, 3]] means 2 degreee of freedom to be considered
            and 3 phonon spaces for each degree of freedom

        '''
        ph_space = [len(self.active_phonon)]
        ph_N  = []
        for ph_i in range(len(self.active_phonon)):
            ph_N.append(len(self.active_phonon[ph_i]))
        ph_space.append(ph_N)
        return ph_space
    
    def plot_freq(self):
        '''
        visualize eigenfreqencies and laser frequency
        '''
        if self.N == 3:
            wmlist  = self.wmlist()*1000
            lab0 = r'$f_{com}$ = ' + str(np.round(wmlist[0],1)) + 'kHz'
            lab1 = r'$f_{tilt}$ = ' + str(np.round(wmlist[1],1)) + 'kHz'
            lab2 = r'$f_{rock}$ = ' + str(np.round(wmlist[2],1)) + 'kHz'
            froc =  [wmlist[2], wmlist[2]]
            ftil =  [wmlist[1], wmlist[1]]
            fcom =  [wmlist[0], wmlist[0]]
            ylist = [0,1]
            title = r'$\delta = $' + str(np.round(self.delta,2)) + 'kHz, reference: '+str(self.delta_ref)
            plt.figure(0)
            plt.plot(fcom,ylist,label = lab0)
            plt.plot(ftil,ylist,label = lab1)
            plt.plot(froc,ylist,label = lab2)
            las = wmlist[self.delta_ref]+self.delta    
            labl = r'$f_{laser}$ = ' + str(np.round(las ,1)) + 'kHz'
            flas =  [las, las]
            plt.plot(flas,ylist,'--',label = labl)   
            plt.ylim(0,1)
            plt.title(title)
            plt.xlabel(r'frequecny kHz')
            plt.grid(b=None, which='major', axis='x', color = 'black', linestyle = '--')
            plt.legend()
            plt.show()
        else:
            print('current module only enables plotting frequency diagram for 3 ion system')
    def plot_all_freq(self):
        '''
        plot all eigenfrequencies of the sustem
        '''
        lab_dic = {0:'com',1:'tilt',2:'rock'}
        if self.N == 3:
            wmlist  = self.Axialfreq()*self.fz*1000
            ylist = [0,1]
            plt.figure(0)
            for m in range(3):
                lab =  r'$f_{'+lab_dic[m]+'}$ = ' + str(np.round(wmlist[m],1)) + 'kHz'
                fplot =  [wmlist[m], wmlist[m]]
                plt.plot(fplot ,ylist,'r-',label = lab) 
            wmlist  = self.Transfreq()*self.fz*1000
            plt.figure(0)
            for m in range(3):
                lab =  r'$f_{'+lab_dic[m]+'}$ = ' + str(np.round(wmlist[m],1)) + 'kHz'
                fplot =  [wmlist[m], wmlist[m]]
                plt.plot(fplot ,ylist,'b-',label = lab)     
            plt.ylim(0,1)
            plt.xlabel(r'frequecny kHz')
            plt.grid(b=None, which='major', axis='x', color = 'black', linestyle = '--')
            plt.legend()
        else:
            print('current module only enables plotting frequency diagram for 3 ion system')
    def plot_N_freq(self):
        '''
        plot all eigenfrequencies of N ion system
        '''
        wmlist  = self.Axialfreq()*self.fz*1000
        ylist = [0,1]
        plt.figure(0)
        for m in range(self.N):
            fplot =  [wmlist[m], wmlist[m]]
            if m ==0:
                lab = 'Axial COM' 
                plt.plot(fplot ,ylist,'r-',label = lab) 
            else:
                plt.plot(fplot ,ylist,'r-') 
        wmlist  = self.Transfreq()*self.fz*1000
        for m in range(self.N):
            fplot =  [wmlist[m], wmlist[m]]
            if m ==0:
                lab = 'Radial' 
                plt.plot(fplot ,ylist,'b-',label = lab) 
            else:
                plt.plot(fplot ,ylist,'b-') 
        plt.title('Axial COM: '+str(np.round(self.fz,2))+' MHz, '+ 
                  'Radial COM: '+str(self.fx)+' MHz')        
        plt.ylim(0,1)
        plt.xlabel(r'frequecny kHz')
        plt.grid(b=None, which='major', axis='x', color = 'black', linestyle = '--')
        plt.legend()
    def alpha(self):
        '''
        compute anisotropy coefficient of the trap
        Returns
        -------
        float, unit of 1

        '''
        return (self.fz/self.fx)**2 
    def l0(self):
        '''
        compute the chracteristic length scale of the system

        Returns
        -------
        float, unit of m
        '''
        return lc(self.fz)
    def Equi_posi(self):
        '''
        compute the equilibrium position of 1-D ion-chain

        Returns
        -------
        np array object, each index a equilibirum position

        '''
        return E_position(self.N,self.fz,False,0) 
    def Amatrix(self):
        #compute the tensor A which determines the axial oscillation
        #fz, axial frequency of the ion trap
        eposition = E_position(self.N, self.fz,False,0) 
        Amat = np.zeros((self.N,self.N))
        for m in range(self.N):
            for n in range(self.N):
                Amat[m,n] = Aele(self.N,m,n,eposition)
        return Amat
    def Tmatrix(self):
        #compute the tensor B that determines transverse oscillation
        #fx, transverse frequency of the ion trap
        Amat = self.Amatrix()
        Tmat = (0.5+(self.fx/self.fz)**2) * np.identity(self.N) - 0.5*Amat
        return Tmat
    def C(self,m,n,p):
        '''
        Compute the anharmonic tensor in the classical Lagrangian 

        Parameters
        ----------
        m, n, p : int
        python index from 0~N-1

        Returns
        -------
        float
        tensor element m, n, p
        '''
        if (m==n) and (n==p):
            Cmnp = 0
            for q in range(self.N):
                if q!=m:
                    Cmnp = (Cmnp + 
                            np.sign(q-m)/(self.Equi_posi()[q]-self.Equi_posi()[m])**4)
        elif (m!=n) and (n!=p) and (m!=p):
            Cmnp = 0
        else:
            #exchange variable if m=n is not statisfied 
            if m == p:
                p0 = n ; m0 = m 
            elif m == n: 
                p0 = p ; m0 = m 
            else:
                p0 = m ; m0 = p 
            #only needs m,p to compute this     
            Cmnp = -1*np.sign(p0-m0) / (self.Equi_posi()[p0]-self.Equi_posi()[m0])**4
        return Cmnp
    def D(self,p0,q,r):
        '''
        Compute the anharmonic tensor for mode-mode coupling   
        
        Parameters
        ----------
        p0, q, r : int
        python index from 0~N-1

        Returns
        -------
        float
        tensor element p, q, r
        '''
        #implement the summation over 3 index
        Dpqr = 0
        N0 = self.N
        Amat = self.Axialmode()
        #this matrix adjust the sign of the eigenmodes such that all eigenvectors
        #has positive sign at index N (or N-1 in terms of python index)
        sigs = np.sign(Amat[:,N0-1])
        sigmat = np.zeros([N0,N0])
        for i in range(N0):
            sigmat[i,i] = sigs[i] 
        Amat = np.transpose(np.matmul(np.transpose(Amat), sigmat))
        for l in range(N0):
            for m0 in range(N0):
                for n0 in range(N0):
                    nterm = self.C(l,m0,n0) * Amat[p0,l] * Amat[q,m0] * Amat[r,n0]
                    Dpqr = Dpqr + nterm
        return Dpqr 
    def plot_D(self,non_zero=True):
        '''
        Plot absolute value of anharmonic coefficients D_mnp

        Parameters
        ----------
        non_zero: bool 
            default is True, if True, only plot non-zero coefficeints     

        Returns
        -------
        None.

        '''
        Dplot = {}
        for i in range(self.N ):
            for j in range(self.N):
                for k in range(self.N ):
                    Dvalue = np.abs(self.D(i,j,k))
                    if non_zero:
                        if Dvalue>1e-5:
                            Dplot[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=np.abs(Dvalue)
                    else:
                        Dplot[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=np.abs(Dvalue)
        names = list(Dplot.keys())
        values = list(Dplot.values())
        plt.bar(range(len(Dplot)), values, tick_label=names)  
        plt.ylabel(r'$|D_{mnp}|$',fontsize = 13)      
        plt.xticks(fontsize = 13)  
        plt.yticks(fontsize = 13)  
        plt.xlabel('Mode index mnp: m,n for radial, p for axial',fontsize = 13)
        plt.grid()       
        plt.show()
    def ah_freq(self,mode_index,ftype):
        '''
        compute the oscillating frequency of the anharmonic term m n p

        Parameters
        ----------
        m : int
            index of radial mode 1
        n : int
            index of radial mode
        p : int
            index of axial mode 
        ftype: int
            determines the type of frequency to be computed
            0 for -
            1 for +
        Returns
        -------
        frequency in kHz

        '''
        [m,n,p] = mode_index
        efaxial = self.Axialfreq()*self.fz*1000
        efradial = self.Transfreq()*self.fz*1000
        if ftype==0:
            f_ah = efradial[m]-efradial[n]-efaxial[p]
        else:
            f_ah = efradial[m]+efradial[n]-efaxial[p]
        return f_ah 
    def plot_ah_freq(self,non_zero=True):
        '''
        Plot absolute value of anharmonic coupling freq

        Parameters
        ----------
        non_zero: bool 
            default is True, if True, only plot non-zero coefficeints     

        Returns
        -------
        None.

        '''
        afplot1 = {}; afplot2 = {}
        #compute axial, radial freq
        efaxial = self.Axialfreq()*self.fz*1000
        efradial = self.Transfreq()*self.fz*1000
        for i in range(self.N ):
            for j in range(self.N):
                for k in range(self.N ):
                    Dvalue = np.abs(self.D(i,j,k))
                    af1 = np.abs(efradial[i]-efradial[j]-efaxial[k])
                    af2 = np.abs(efradial[i]+efradial[j]-efaxial[k])
                    if non_zero:
                        if Dvalue>1e-5:
                            afplot1[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=af1
                            afplot2[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=af2
                    else:
                        afplot1[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=af1
                        afplot2[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=af2
        names = list(afplot1.keys())
        #plot Delta-
        plt.figure()
        values1 = list(afplot1.values())
        plt.bar(range(len(afplot1)), values1, tick_label=names)  
        plt.title(r'$|\omega_n-\omega_m-\nu_p|$',fontsize = 12)
        plt.ylabel(r'$|\Delta^-|$, [kHz]',fontsize = 13)      
        plt.xticks(fontsize = 13)  
        plt.yticks(fontsize = 13)  
        plt.xlabel('Mode index mnp: m,n for radial, p for axial',fontsize = 13)
        plt.grid()       
        plt.show()
        #plot Delta+
        plt.figure()
        values2 = list(afplot2.values())
        plt.bar(range(len(afplot1)), values2, tick_label=names)  
        plt.title(r'$|\omega_n+\omega_m-\nu_p|$',fontsize = 12)
        plt.ylabel(r'$|\Delta^+|$, [kHz]',fontsize = 13)      
        plt.xticks(fontsize = 13)  
        plt.yticks(fontsize = 13)  
        plt.xlabel('Mode index mnp: m,n for radial, p for axial',fontsize = 13)
        plt.grid()       
        plt.show()
    def Axialfreq(self):
        '''
        compute the eigenvalue of axial eigenmode matrix, multiply by fz to get real frequency
        Returns
        -------
        np array object, each index is an eigenvalue for axial mode [unit of 1]
        The eigenvalues are arranged in an increasing order, such that the first 
        one correpond to COM mode frequency
        '''
        e_val = np.linalg.eig(self.Amatrix())[0]
        order = np.argsort(e_val)
        e_val = e_val[order]
        return np.sqrt(e_val)
    def Axialmode(self):
        '''
        compute the eigenmodes of axial oscillation 
        Returns
        -------
        np array object that represents N by N matrix, each row is an axial eigenmode
        The eigenmode are arranged in an increasing order of eigenvalues, such that the first 
        one correpond to COM mode

        '''
        e_val,e_array = np.linalg.eig(self.Amatrix())
        order = np.argsort(e_val)
        return (np.transpose(e_array))[order]
    def Transfreq(self):
        '''
        compute the eigenvalue of transverse eigenmode matrix, multiply by fz to get real frequency [MHz]
        Returns
        -------
        np array object, each index is an eigenvalue for Transverse(Radial) mode [unit of 1]
        The eigenvalues are arranged in an decreasing order, such that the first 
        one correpond to COM mode frequency

        '''
        e_val = np.linalg.eig(self.Tmatrix())[0]
        order = np.argsort(e_val)
        e_val = e_val[order][::-1]
        #check if the matrix is positive-definite
        if np.min(e_val) < 0:
            print("Negtive transverse frequency, the system is unstable")
            return np.sqrt(e_val+0j)
        else:
            return np.sqrt(e_val)
    def Transmode(self):
        '''
        compute the eigenmode of transverse oscillation

        Returns
        -------
        np array object that represents N by N matrix, each row is an Transverse (Radial) eigenmode
        The eigenmode are arranged in an decreasing order of eigenvalues, such that the first 
        one correpond to COM mode

        '''
        e_val, e_array= np.linalg.eig(self.Tmatrix())
        order = np.argsort(e_val)[::-1]
        return np.transpose(e_array)[order]    
    def wmlist(self):
        '''
        compute axial , transverse eigenfrequencies of the system
        Returns
        -------
        np array of float, index 0 is always for com mode, [MHz]
        '''
        if self.df_laser == 0:
            wlsit = self.Axialfreq()*self.fz
        else:   
            wlsit = self.Transfreq()*self.fz
        return wlsit
    def mu(self):
        '''
        compute band laser frequency mu defined as  omega_{Lb}- omega{hf}
        -------
        float 2pi kHz (angular)
        '''
        wlist0 = self.wmlist()*1000
        mu = 2*np.pi*(wlist0[self.delta_ref] + self.delta)
        return mu
    def dmlist(self):
        '''
        compute the detuning from eigenmodes with specified direction, increasing order
        Returns
        -------
        np array of float, index 0 is always for com mode. 2pi kHz (angular)
        '''
        dm = self.mu()-2*np.pi*(self.wmlist()*1000)
        return dm
    def Omega(self):
        '''
        compute the effective rabi frequency of the laser
        ----------
        Returns
        -------
        float,2pi kHz
        '''
        return np.sqrt(Omega(self.fr,self.fx)*Omega(self.fb,self.fx))/1000
    def g(self,i,m):
        '''
        Compute the laser-ion coupling strength between ion i and mode m
        Parameters
        ----------
        i : int
            ion index
        m : int
            eigenmode index

        Returns
        -------
        g : float
            [2pi kHz]
        '''
        
        if self.df_laser == 0:
            emat = self.Axialmode()
        else:   
            emat = self.Transmode()
        
        coeff = eta(self.wmlist()[m])*self.Omega()*emat[m,i]
        return coeff
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
    def epsilon(self):
        '''
        Compute the anharmonic coefficient epsilon = sqrt(hbar/(2 m fz)/(4l))
        -------
        Returns
        -------
        float unit of 1

        '''
        sigma0 = np.sqrt(0.5*h / (MYb171*fr_conv(self.fz,'mhz')))
        return  sigma0 / (4*self.l0())
    def ah_couple(self, mode_index,real_unit=False):
        '''
        Compute the anharmonic coupling strength for index m , n, p

        Parameters
        ----------
        mode_index: list of python index [m n p]
        m,n for transverse modes, p for axial mode
        real_unit: bool
            default as False
            if True, compute coefficients in unit of kHz
            if false, compute coefficients in unit of fz
        Returns
        -------
        float, anharmonic coupling strength, [unit 1]
        multiply fz to get coupling strength in Hz
        or real frequency in kHz

        '''
        [m,n,p] = mode_index
        tfreq = (self.Transfreq())**2; afreq = (self.Axialfreq())**2
        freq_factor = (tfreq[m]*tfreq[n]*afreq[p])**0.25
        if real_unit:
            ah_coef = self.fz*1000*(-3*self.epsilon()*self.D(m,n,p)/freq_factor)
        else:
            ah_coef = -3*self.epsilon()*self.D(m,n,p)/freq_factor
        return ah_coef
    def plot_ah_c(self,non_zero=True):
        '''
        Plot absolute value of anharmonic coupling coefficients
    
        Parameters
        ----------
        non_zero: bool 
            default is True, if True, only plot non-zero coefficeints     
    
        Returns
        -------
        None.
    
        '''
        ahplot = {}
        for i in range(self.N ):
            for j in range(self.N):
                for k in range(self.N ):
                    ah_value = np.abs(self.ah_couple([i,j,k],True))
                    if non_zero:
                        if ah_value>1e-5:
                            ahplot[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)] = ah_value 
                    else:
                        ahplot[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)] = ah_value 
        names = list(ahplot.keys())
        values = list(ahplot.values())
        plt.bar(range(len(ahplot)), values, tick_label=names)  
        plt.ylabel(r'$|C_{ah}|$, [kHz]',fontsize = 13)      
        plt.xticks(fontsize = 13)  
        plt.yticks(fontsize = 13)  
        plt.xlabel('Mode index mnp: m,n for radial, p for axial',fontsize = 13)
        plt.grid()       
        plt.show()