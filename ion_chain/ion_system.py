# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:47:46 2022

@author: zhumj, gp
class ions(): a class that contains basic physical and computational parameters of 1-D N ion system with 1 laser drive
class Laser(): a class that contains physical properties of the laser field (amplitude, frequency, phase and momentum)
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
def fr_conv(f,unit):
    '''
    convert frequency to radial frequency 
    Parameters
    ----------
    f : float
        input frequency, in Hz, kHz, or MHz
    unit : str
        specify the unit, can be 'Hz', 'kHz', 'MHz'
    Returns
    -------
    f_out : float
        output radial frequency, unit of 2pi Hz

    '''
    factor = {'Hz':0,'kHz':3,'MHz':6}
    f_out = 2*np.pi * f*10**(factor[unit])
    return f_out
'''
subfunction
'''
def X0(f0):
    #compute the characteristic length scale of the motional mode, [m]
    #input in MHz
    return np.sqrt(h / (2*MYb171* fr_conv(f0,'MHz')))
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
def E_position(N,fz,scale=False,com_scale=1):
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
def compute_three_tensor(N,c_func):
    #compute all elements of a rank 3 tensor and store them in array
    tensor = np.zeros((N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                tensor[i][j][k] = c_func(i,j,k)
    return tensor
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
    return (qe**2/ (4*np.pi * eps0 * MYb171 * fr_conv(fz,'MHz')**2))**(1/3)


def efreq(ion0,laser0):
    '''
    extract eigenfrequency [MHz] in a given laser direction
    
    Parameters
    ----------
    ion0 : ions class object
    laser0 : Laser class object

    Returns
    -------
    np array

    '''
    if laser0.wavevector == 0: 
        freq = ion0.axial_freq 
    elif laser0.wavevector == 1:   
        freq = ion0.radial_freq
    else:   
        freq = ion0.radial_freq2
    return freq

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
    this class is used to store the parameters of a N ion system couples to a 
    single laser drive in Axial or Radial direction, assuming the two radial 
    directions are equivalent object attributes to be set directly
    ion_config:
    N : int
        number of ions in the system
    fz : float
        axial frequency of the ion trap, [MHz]
    fx : float
        transverse frequency of the ion trap, [MHz]
    V_mod : list of float
        Modulation Amplitude for parameteric amplification, [V]
    f_mod: list of float
        Modulation Frequency for parameteric amplification, [kHz]
    d_T: float
        Trap dimension parameter, [um] 
    laser_config:
    Omega_eff: float
         effective laser_dipole Rabi frequency Omega * dK * X_0(fx) [kHz],
    laser_couple: list of int
        ion index that couples to the laser, for instance [0,1] means couple to 
        ion 0, 1
    phase: float
        spin phase phis [rad]
        
    numeric config
    active_spin: list of int
        index of ionic spin space to be considered 
    active_phonon: list of list of int
        Index of phonon space to be be considered.
        Initialized in form [[axial]]/[[radial]] if only 
        1 vibrational degree of freedom is considered. 
        Initialized in form [[axial],[radial]] if both 
        vibrational degrees of freedoms are considered. 
        (0 always means com mode)
        For instance, [[1,2],[0,2]] means consider the tilt, 
        rock mode for axial motion com, rock mode for 1 radial motion.
    active_df_motion: list of int
        motional degrees of freedom being considered, required only in case of 
        including both radial degrees of freedoms
        Radial: 1 for x, 2 for y
        Axial: 0 for z
    pcut: list of list of int
        Cutoff of phonon space size for each phonon space to be considered, 
        needs to be consistent with active_phonon. 
        Initialized in form [[axial]]/[[radial]] if only
        1 vibration degree of freedom is considered. 
        Initialized in form [[axial],[radial]] if both 
        vibrational degrees of freedoms are considered.
    
    cooling config
    coolant: list of int
        index of coolant    
        1 for radial.
    important class method:
            
    '''
    def __init__(self,
                 trap_config = {'N':2,'fx':2,'fz':1},
                 numeric_config = {'active_spin':[0,1], 'active_phonon':[[0,1]],'pcut' : [[5,5]]},
                 cooling_config = {'coolant' : [1]},
                 para_mod_config = {'f_mod':[0],'V_mod':[0],'d_T':200}
                 ):
        
        '''
        initialize a ions class object with given parameters
        Parameters
        ----------
        trap_config : dict, optional
            parameters for trap configuration. 
            The default is {'N':2,'fx':2,'fz':1,'f_mod':0,'V_mod':0}:
            N = 2 ions, fx = 2MHz, fz = 1MHz, f_mod = 0kHz, V_mod = 0V, d_T = 200 um
        numeric_config : dict, optional
            parameters for phonon space configuration. 
            The default is {'active_phonon':[[0,1]],'pcut' : [[5,5]]}.
            consider all 2 modes in axial/radial direction and set cutoff level at 5
        cooling_config : dict optional
            parameters for sympathetic cooling configuration. 
            The default is {coolant' : [1]}.
        para_mod_config : dict optional
            parameters for parametric modulation of trap frequency
            The default is {'f_mod':[0],'V_mod':[0],'d_T':200}.

        Returns
        -------
        None.

        '''
        self.update_all(trap_config, numeric_config, cooling_config,para_mod_config)        
        print('Ions class object initialized.')
    def list_para(self):
        '''
        list basic physical parameters of the trapped ion chain 

        '''
        print('________________________________________________________________')
        print('********************Setup of the Trap************************')
        print('number of ions', self.N)
        print('Axial COM (Confining) frequency ',np.round(self.fz,2),' [MHz]')
        print('Radial COM (Confining) frequency ',np.round(self.fx,2), '[MHz]')
        print('Axial vibrational eigenfrequency', np.round(self.axial_freq,2),'MHz')
        print('Radial (Transverse) vibrational eigenfrequency', np.round(self.radial_freq,2),'[MHz]')
        print('Modulation Amplitude for parameteric amplification: ',np.round(self.V_mod,2), '[V]')
        print('Modulation Frequency for parameteric amplification: ',np.round(self.f_mod,2), '[MHz]')
        print('Trap dimension parameter: ', self.d_T, '[um]')
        
        print('                                                                 ')
        print('********************Config of Numeric Calculation************************')
        print('index of spin space included in simulation: ',self.active_spin )
        print('index of phonon space included in simulation: ',self.active_phonon )
        print('corresonding phonon space cutoff ', self.pcut)
        #print('index of corresponding motional degrees of freedom', self.active_df_motion)
        print('********************Config of Cooling************************')
        print('Coolant index ', self.coolant)
        print('********************Config of Trap Modulation************************')

        print(' Modulation Amplitude', np.round(self.V_mod,2)," [V]") 
        print(' Modulation Frequency', np.round(self.f_mod,2)," [kHz]")
        print(' Trap dimension parameter', np.round(self.d_T,2)," [um]")

    def update_all(self, trap_config = None, numeric_config=None, cooling_config=None,
                   para_mod_config = None,
                   print_text = True) :
        self.update_trap(trap_config, print_text)
        self.update_numeric(numeric_config, print_text)
        self.update_cooling(cooling_config, print_text)
        self.update_PM(para_mod_config, print_text)
    def update_trap(self,trap_config = None, print_text = True):
        '''
        Set trap parameters and compute all related attributes. 
        (see pdf document for details)
        Parameters
        ----------
        trap_config : dict, optional
           Parameters for trap configuration. The default is None.
        print_text : bool, optional
            If true, print a message after updating parameters. 
            The default is True.

        Returns
        -------
        None.

        '''
        if trap_config != None:
            self.N = trap_config['N']  #set trap parameters
            self.fx = trap_config['fx']
            self.fz = trap_config['fz']
        #compute and update Eigenmodes and Eigenfrequencies
        #compute normalized equilibrium position of the ions
        self.equi_posi = self.Equi_posi() 
        #compute axial/radial elastic tensor
        self.a_matrix = self.A_matrix(); self.r_matrix = self.R_matrix()
        #compute axial/radial eigenvalues
        self.axial_eval = self.Axial_eval(); self.radial_eval = self.Radial_eval()
        self.axial_mode = self.Axial_mode(); self.radial_mode = self.Radial_mode()
        #compute axial/radial eigenfrequencies in  (MHz)
        self.axial_freq = self.fz * np.sqrt(self.axial_eval)
        #check if the radial tensor is positive definite
        if np.min(self.radial_eval) < 0:
            print('=================================================')
            print('---------------------WARNING---------------------')
            print("Negative radial-eigenvalues, the ion configuration is unstable")
            print('=================================================')
            self.radial_freq = self.fz * np.sqrt(self.radial_eval+0j)
        else:
            self.radial_freq = self.fz * np.sqrt(self.radial_eval)
        
        #compute anharmonic coupling tensor C, D
        self.ah_epsilon_val = self.ah_epsilon()
        self.ah_C_tensor = compute_three_tensor(self.N,self.ah_C)
        self.ah_D_tensor = compute_three_tensor(self.N,self.ah_D)
        if print_text:
            print('Trap coefficients updated')
            print('Anharmonic coefficients updated')
    def update_numeric(self, numeric_config = None, print_text = True):
        '''
        Set phonon parameters and compute all related attributes. 
        (see pdf document for details)
        Parameters
        ----------
        numeric_config : dict, optional
           Parameters for phonon space configuration. The default is None.
        print_text : bool, optional
          If true, print a message after updating parameters. The default is True.

       Returns
       -------
       None.

        '''
        if numeric_config != None:
            self.active_spin = numeric_config['active_spin']  
            self.active_phonon = numeric_config['active_phonon']
            #self.active_df_motion = numeric_config['active_df_motion']
            self.pcut = numeric_config['pcut']
        self.df_spin = len(self.active_spin)
        if print_text:
            self.check_phonon()
            print('Phonon space parameters updated')
    def update_cooling(self, cooling_config = None, print_text = True):
        '''
        Set laser parameters and compute all related attributes. 
        (see pdf document for details)
        Parameters
        ----------
        cooling_config : dict, optional
           Parameters for laser configuration. The default is None.
        print_text : bool, optional
          If true, print a message after updating parameters. The default is True.

       Returns
       -------
       None.

        '''
        if cooling_config != None:
            self.coolant=  cooling_config['coolant']
        if print_text:
            print('Cooling parameters updated')
    def update_PM(self, para_mod_config = None, print_text = True):
        '''
        Set parameters for parametric modulation of trap frequency
        Parameters
        ----------
       para_mod_config : dic, optional
            Parameters for parametric modulation. The default is None.
       print_text : bool, optional
         If true, print a message after updating parameters. The default is True.

        Returns
        -------
        None.

        '''
        if para_mod_config != None:
           self.f_mod = para_mod_config['f_mod']
           self.V_mod = para_mod_config['V_mod']
           self.d_T = para_mod_config['d_T']
        if print_text:
            print('Trap parametric modulation updated')
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
            print('Phonon space setups are consistent')
        print('_____________________________________________________________')    

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
    def A_matrix(self):
        #compute the tensor A which determines the axial oscillation
        #fz, axial frequency of the ion trap
        Amat = np.zeros((self.N,self.N))
        for m in range(self.N):
            for n in range(self.N):
                Amat[m,n] = Aele(self.N,m,n,self.equi_posi)
        return Amat
    def R_matrix(self):
        #compute the tensor B that determines transverse oscillation
        #fx, transverse frequency of the ion trap
        Amat = self.a_matrix
        Tmat = (0.5+(self.fx/self.fz)**2) * np.identity(self.N) - 0.5*Amat
        return Tmat
    def Axial_eval(self):
        '''
        compute the eigenvalue of axial elastic tensor
        Returns
        -------
        np array object, each index is an eigenvalue for axial mode [unit of 1]
        The eigenvalues are arranged in an increasing order, such that the first 
        one corresponds to COM mode frequency
        '''
        e_val = np.linalg.eig(self.a_matrix)[0]
        order = np.argsort(e_val)
        e_val = e_val[order]
        return e_val
    def Axial_mode(self):
        '''
        compute the eigenmodes of axial oscillation 
        Returns
        -------
        np array object that represents N by N matrix, each row is an axial eigenmode
        The eigenmode are arranged in an increasing order of eigenvalues, such that the first 
        one correpond to COM mode

        '''
        e_val,e_array = np.linalg.eig(self.a_matrix)
        order = np.argsort(e_val)
        return (np.transpose(e_array))[order]
    def Radial_eval(self):
        '''
        compute the eigenvalue of transverse/radial elastic tensor
        Returns
        -------
        np array object, each index is an eigenvalue for Transverse(Radial) mode [unit of 1]
        The eigenvalues are arranged in an decreasing order, such that the first 
        one correpond to COM mode frequency

        '''
        e_val = np.linalg.eig(self.r_matrix)[0]
        order = np.argsort(e_val)
        e_val = e_val[order][::-1]
        return e_val
    def Radial_mode(self):
        '''
        compute the eigenmode of radial oscillation

        Returns
        -------
        np array object that represents N by N matrix, each row is an Transverse (Radial) eigenmode
        The eigenmode are arranged in an decreasing order of eigenvalues, such that the first 
        one correpond to COM mode

        '''
        e_val, e_array= np.linalg.eig(self.r_matrix)
        order = np.argsort(e_val)[::-1]
        return np.transpose(e_array)[order]    
    '''
    _________________________________________________________________________
    This part computes anharmonic coupling coefficients
    '''
    def ah_C(self,m,n,p):
        '''
        Compute the anharmonic tensor C elements for classical Lagrangian 

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
                            np.sign(q-m)/(self.equi_posi[q]-self.equi_posi[m])**4)
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
            Cmnp = -1*np.sign(p0-m0) / (self.equi_posi[p0]-self.equi_posi[m0])**4
        return Cmnp
    def ah_D(self,p0,q,r):
        '''
        Compute the anharmonic tensor D element for mode-mode coupling   
        
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
        Amat = self.axial_mode
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
                    nterm = self.ah_C_tensor[l][m0][n0] * Amat[p0,l] * Amat[q,m0] * Amat[r,n0]
                    Dpqr = Dpqr + nterm
        return Dpqr 
    def ah_epsilon(self):
        '''
        Compute the anharmonic coefficient epsilon = sqrt(hbar/(2 m fz)/(4l))
        -------
        Returns
        -------
        float unit of 1

        '''
        sigma0 = np.sqrt(0.5*h / (MYb171*fr_conv(self.fz,'MHz')))
        return  sigma0 / (4*self.l0())
    def ah_couple(self, mode_index, real_unit=False):
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
        tfreq = self.radial_eval; afreq = self.axial_eval
        freq_factor = (tfreq[m]*tfreq[n]*afreq[p])**0.25
        if real_unit:
            ah_coef = self.fz*1000*(-3*self.ah_epsilon_val*self.ah_D_tensor[m][n][p]/freq_factor)
        else:
            ah_coef = -3*self.ah_epsilon_val*self.ah_D_tensor[m][n][p]/freq_factor
        return ah_coef
    def plot_ah_c(self,non_zero=True,real_freq = True):
        '''
        Plot absolute value of anharmonic coefficients D_mnp

        Parameters
        ----------
        non_zero: bool 
            default is True, if True, only plot non-zero coefficeints     
        real_freq: bool 
            default is True, if True, plot real anharmonic coupling coefficients in kHz 
                             if False, plot elements of tensor D
        Returns
        -------
        None.

        '''
        ah_plot = {}
        for i in range(self.N ):
            for j in range(self.N):
                for k in range(self.N ):
                    if real_freq:
                        ah_val = np.abs(self.ah_couple([i,j,k],True))
                    else:
                        ah_val = np.abs(self.ah_D_tensor[i][j][k])
                    if non_zero:
                        if ah_val>1e-5:
                            ah_plot[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=np.abs(ah_val)
                    else:
                        ah_plot[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=np.abs(ah_val)
        names = list(ah_plot.keys())
        values = list(ah_plot.values())
        plt.bar(range(len(ah_plot)), values, tick_label=names) 
        if real_freq:
            plt.ylabel(r'$|C_{ah}|$, [kHz]',fontsize = 13)      
        else:
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
        efaxial = self.axial_freq*1000
        efradial = self.radial_freq*1000
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
        for i in range(self.N ):
            for j in range(self.N):
                for k in range(self.N ):
                    Dvalue = np.abs(self.ah_D_tensor[i][j][k])
                    af1 = np.abs(self.ah_freq(self,[i,j,k],0))
                    af2 = np.abs(self.ah_freq(self,[i,j,k],1))
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

    def expeak(self):
        '''
        compute the expected peak frequencies
        '''
        print(np.round(np.abs(self.detuning)/(4*np.pi),2))
    def Lambda(self):
        '''
        Compute the reorgonization energy lambda, which also correponds to energy
        splitting that leads to the maximum transfer rate
        Returns
        -------
        float [J/10**6]
        '''
        return self.g()**2/np.abs(self.w0())
    def PA_coef(self,df_mod,m,pa_index=0):
        '''
        Compute the coupling strength due to parameteric amplification,
        2eV / d_t^2
        df_mod: int
            modulation direction, 0 for axial, 1 for radial 
        m:
            mode index 
        pa_index: int
            index of pa component index, used for multiple wave component case
            default as 0
        Returns
        -------
        float [2pi kHz]

        '''
        if df_mod == 0:
            freq = self.axial_freq
        else:
            freq = self.radial_freq
            #compute everything in SI
        numer = qe * self.V_mod[pa_index]
        demoni = ( MYb171 * fr_conv(freq[m],'MHz' ) * (self.d_T*1e-6)**2 )
        return (numer/demoni)/1000 #convert SI to 2pi kHz
    
    def plot_freq(self,show_axial = False, laser_list=[]):
        '''
        visualize eigenfreqencies and laser frequency for an ion system

        Parameters
        ----------
        show_axial : bool, optional
            If true, plot all axial frequencies. The default is False.
        laser_list : list of class Laser objects, optional
            List of laser drives to be plotted. The laser freuqencies will 
            not be plotted if this list is empty. The default is [].
        Returns
        -------
        None.

        '''
        lab_dic = {0:'com',1:'tilt',2:'rock'}
        wmlist1  = self.axial_freq*1000; wmlist2  = self.radial_freq*1000
        ylist = [0,1]
        if self.N == 3 or self.N == 2:
            for m in range(self.N):
                if m == 0:
                    lw = 4
                else:
                    lw = 2
                lab =  r'$f_{'+lab_dic[m]+'}$ = ' + str(np.round(wmlist2[m],1)) + 'kHz' 
                plt.plot([wmlist2[m], wmlist2[m]] ,ylist,'b-',label = lab, linewidth= lw)
                if show_axial:
                    lab =  r'$f_{'+lab_dic[m]+'}$ = ' + str(np.round(wmlist1[m],1)) + 'kHz' 
                    plt.plot([wmlist1[m], wmlist1[m]] ,ylist,'r-',label = lab, linewidth= lw) 
        else:
            for m in range(self.N): 
                if m ==0:
                    lab = 'Radial COM' 
                    plt.plot([wmlist2[m], wmlist2[m]] ,ylist,'b-',label = lab, linewidth= 4) 
                    if show_axial:
                        lab = 'Axial COM' 
                        plt.plot([wmlist1[m], wmlist1[m]] ,ylist,'r-',label = lab, linewidth= 4) 
                else:
                    plt.plot([wmlist2[m], wmlist2[m]] ,ylist,'b-',linewidth = 2)
                    if show_axial:
                        plt.plot([wmlist1[m], wmlist1[m]] ,ylist,'r-',linewidth= 2) 
            plt.title('Axial COM: '+str(np.round(self.fz,2))+' MHz, '+ 
                      'Radial COM: '+str(self.fx)+' MHz')   
        #plot laser frequency
        if laser_list != []:
            for laser0 in laser_list:
                las = laser0.mu
                labl = r'$f_{L}$ = ' + str(np.round(las ,1)) + 'kHz'
                flas = [las, las]
                plt.plot(flas,ylist,'--',label = labl, linewidth = 4)   
        plt.ylim(0,1)
        plt.xlabel(r'frequecny kHz')
        plt.grid(visible=True, which='major', axis='x', color = 'black', linestyle = '--')
        plt.legend()
        plt.show()
        
class Ions_asy(ions):
    '''
    A subclass of class ions for asymmetric confinment in radial directions
    '''
    def __init__(self,
                 trap_config = {'N':2,'fx':2,'fz':1,'offset':0},
                 numeric_config = {'active_spin':[0,1], 'active_phonon':[[0,1]],'pcut' : [[5,5]]},
                 cooling_config = {'coolant' : [1]},
                 para_mod_config = {'f_mod':[0],'V_mod':[0],'d_T':200}
                 ):
        super().__init__(trap_config,numeric_config,cooling_config,para_mod_config)
    def update_trap(self,trap_config = None, print_text = True):
        '''
        Set trap parameters and compute all related attributes. 
        with a new parameter: freq_offset,
        specifying the offset in [kHz] of the radial eigenfrequncies

        '''
        super().update_trap(trap_config,print_text)
        if trap_config != None:
            self.freq_offset = trap_config['offset']
        self.radial_freq1 = self.radial_freq  #first set of frequencies 
        self.radial_freq2 =  self.radial_freq + self.freq_offset/1000
    def list_para(self):
        '''
        list basic physical parameters of the trapped ion chain 

        '''
        print('________________________________________________________________')
        print('********************Setup of the Trap************************')
        print('number of ions', self.N)
        print('Axial COM (Confining) frequency ',np.round(self.fz,2),' [MHz]')
        print('Radial COM (Confining) frequency ',np.round(self.fx,2), '[MHz]')
        print('Axial vibrational eigenfrequency', np.round(self.axial_freq,2),'MHz')
        print('Radial (Transverse) vibrational eigenfrequency (1st set)', np.round(self.radial_freq1,3),'[MHz]')
        print('Radial (Transverse) vibrational eigenfrequency (2nd set)', np.round(self.radial_freq2,3),'[MHz]')
        print('Modulation Amplitude for parameteric amplification: ',np.round(self.V_mod,2), '[V]')
        print('Modulation Frequency for parameteric amplification: ',np.round(self.f_mod,2), '[MHz]')
        print('Trap dimension parameter: ', self.d_T, '[um]')
        
        print('                                                                 ')
        print('********************Config of Numeric Calculation************************')
        print('index of phonon space included in simulation: ',self.active_phonon )
        print('corresonding phonon space cutoff ', self.pcut)
        print('********************Config of Cooling************************')
        print('Coolant index ', self.coolant)
        print('********************Config of Trap Modulation************************')

        print(' Modulation Amplitude', np.round(self.V_mod,2)," [V]") 
        print(' Modulation Frequency', np.round(self.f_mod,2)," [kHz]")
        print(' Trap dimension parameter', np.round(self.d_T,2)," [um]") 
    def plot_freq(self, show_axial = False, show_neg =  False , laser_list=[]):
        '''
        visualize eigenfreqencies and laser frequency for an ion system

        Parameters
        ----------
        show_axial : bool, optional
            If true, plot all axial frequencies. The default is False.
        laser_list : list of class Laser objects, optional
            List of laser drives to be plotted. The laser freuqencies will 
            not be plotted if this list is empty. The default is [].
        Returns
        -------
        None.

        '''
        ylist = [0,1] 
        wmlist_a  = self.axial_freq*1000; 
        wmlist_rx  = self.radial_freq1*1000; wmlist_ry  = self.radial_freq2*1000
        for m in range(self.N): 
            if m ==0:
                lab = 'Radial(x) COM' 
                plt.plot([wmlist_rx[m], wmlist_rx[m]] ,ylist,'b-',label = lab, linewidth= 4) 
                lab = 'Radial(y) COM' 
                plt.plot([wmlist_ry[m], wmlist_ry[m]] ,ylist,'g-',label = lab, linewidth= 4) 
                if show_axial:
                    lab = 'Axial COM' 
                    plt.plot([wmlist_a[m], wmlist_a[m]] ,ylist,'r-',label = lab, linewidth= 4) 
                if show_neg:
                    plt.plot([-wmlist_rx[m], -wmlist_rx[m]] ,ylist,'b-',linewidth= 4)
                    plt.plot([-wmlist_ry[m], -wmlist_ry[m]] ,ylist,'g-',linewidth= 4) 
            else:
                plt.plot([wmlist_rx[m], wmlist_rx[m]] ,ylist,'b-',linewidth = 2)
                plt.plot([wmlist_ry[m], wmlist_ry[m]] ,ylist,'g-',linewidth = 2)
                if show_axial:
                    plt.plot([wmlist_a[m], wmlist_a[m]] ,ylist,'r-',linewidth= 2) 
                if show_neg:
                    plt.plot([-wmlist_rx[m], -wmlist_rx[m]] ,ylist,'b-',linewidth= 2)
                    plt.plot([-wmlist_ry[m], -wmlist_ry[m]] ,ylist,'g-', linewidth= 2)
        plt.title('Axial COM: '+str(np.round(self.fz,2))+' MHz, '+ 
                  'Radial COM: '+str(self.fx)+' MHz' + ', Offset: ' +str(self.freq_offset)+ 'kHz')  
 
        #plot laser frequency
        if laser_list != []:
            for laser0 in laser_list:
                las = laser0.mu
                labl = r'$f_{L}$ = ' + str(np.round(las ,1)) + 'kHz'
                flas = [las, las]
                plt.plot(flas,ylist,'--',label = labl, linewidth = 4)   
        plt.ylim(0,1)
        plt.xlabel(r'frequecny kHz')
        plt.grid(visible=True, which='major', axis='x', color = 'black', linestyle = '--')
        plt.legend()
        plt.show()
class Laser():
    def __init__(self,
                 config = {'Omega_eff':10,'wavevector':1,'Dk':2*2*np.pi / (355*10**(-9)),
                           'laser_couple':[0,1],'mu':1e3,'phase':0},
                 ):
        '''
        Initialize laser class object with given parameters
        Parameters
        ----------
        laser_config : dict, optional
            parameters for laser configuration.
            The default is {'Omega_eff':10,'f_laser':1,'laser_couple':[0,1],
                            'delta':20, 'delta_ref':0,'phase':0}.
            Omega_eff = 10 kHz (Effective Rabi frequency)
            wavevector = 1 (Laser drive in radial direction, 0 for axial direction)
                          In case of an asymmetric setup, use 1 for x, 2 for y (offset)
            laser_couple = [0,1] (Laser coupled to all two ions)
            phase  = 0 rad (spin phase)
            Dk = np.sqrt(2)*2*np.pi / (355*10**(-9)) (Effective wavenumber)
        Returns
        -------
        None.

        '''
        self.update(config)
        print('Lasers class object initialized.')
    def update(self, config = None, print_text = True):
        '''
        Set laser parameters and compute all related attributes. 
        (see pdf document for details)
        Parameters
        ----------
        config : dict, optional
           Parameters for laser configuration. The default is None.
        print_text : bool, optional
          If true, print a message after updating parameters. The default is True.

       Returns
       -------
       None.

        '''
        if config != None:
            self.Omega_eff = config['Omega_eff']  
            self.wavevector = config['wavevector']
            self.laser_couple = config['laser_couple']
            self.mu = config['mu']
            self.phase = config['phase']
            self.Dk = config['Dk'] 
        self.R = self.Recoil_freq()#recoil frequency constant, SI 
        if print_text:
            print('Laser parameters updated')
    def Recoil_freq(self):
        '''
        comupte recoil frequency in 2pi Hz
        '''
        return (h*self.Dk**2) / (2*MYb171) 
    def detuning(self,ion0):
        '''
        compute detuning [2pi kHz] in a given laser direction
        
        Parameters
        ----------
        ion0 : ions class object
        laser0 : Laser class object

        Returns
        -------
        np array

        '''
        dfreq = fr_conv(self.mu, 'Hz') - fr_conv(efreq(ion0,self), 'kHz')
        return dfreq
    def eta(self,f):
        '''
        Compute Lamb-Dicke coefficient for vibrational eigenmode with
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
        return self.Dk * X0(f) 

    def Omega(self,ion0):
        '''
        compute real rabi frequency [2pi kHz] in a given laser direction
        
        Parameters
        ----------
        ion0 : ions class object
        laser0 : Laser class object

        Returns
        -------
        float

        '''
        if self.wavevector == 0:
            f_scale = ion0.fz
        elif self.wavevector == 1:
            f_scale = ion0.fx
        else:
            f_scale = ion0.radial_freq2[0]
        return fr_conv(self.Omega_eff,'Hz') / self.eta(f_scale)
    def list_para(self):
        '''
        list basic physical parameters of the laser drive 

        '''
        Coup_dic = {0:'Axial', 1:'Transverse (Radial x)', 2:'Transverse (Radial y)'}
        freqdic = {'0':'COM freq','1':'tilt freq','2':'rock freq'}
        print('                                                                 ')
        print('********************Parameters of Laser Drive************************')
        print('Vibrational degree of freedom couples to the laser: '+ Coup_dic[self.wavevector])
        print('index of ions that couple to the laser field: ',self.laser_couple)
        print('Effective rabi frequency ', np.round(self.Omega_eff,2),' [kHz]')
        print('Effective laser frequency ', np.round(self.mu,2),' [kHz]')
        print('Laser phase phis',np.round(self.phase*180/np.pi,2))
        print('(input in rad but displayed in degs)')



