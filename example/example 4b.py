# -*- coding: utf-8 -*-
"""
Simulate the time evolution of excitation transfer at different deltaE 
with multi cores parallel coumputation,
this code enables using different cutoff for different deltaE.
Note: this script must be saved before running if any changes have been made
(a strange feature of multiprocessing package)
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.ion_chain.transfer.exci_transfer as extrans
import Qsim.ion_chain.transfer.exci_operators as exop
import Qsim.ion_chain.transfer.multi_core as mcs
from  Qsim.ion_chain.ion_system import *
from astropy.io import ascii
from astropy.table import Table
import multiprocessing as mp
from tqdm import tqdm
import datetime
#%% initialize parameters of ion-chain system 
'''
set parameters of the system
'''    
ion_sys = ions() 
ion_sys.delta_ref = 2
ion_sys.N = 3
ion_sys.delta = -100
fsb = 50
ion_sys.fr = fsb; ion_sys.fb = fsb
ion_sys.phase = np.pi/2
ion_sys.df_laser = 1 #couple to Radial vibrational modes
ion_sys.gamma = [0,0,10]
ion_sys.laser_couple = [0,1] #laser applied to ion 1,2
ion_sys.coolant = [2] #ion 3 as coolant
ion_sys.list_para() #print parameters
ion_sys.plot_freq()
J23 = 1
E3 = 0
V = 0
print('coupling strength between ion 1 and 2', J23, ' kHz *h')
print('site energy of ion 2 ', E3, ' kHz *h')
tscale = J23
#configure time scale for computation
end = 0.5    #end of simulation time
tplot = np.arange(0,end,0.01)/tscale
times = tplot/tscale
#%% configure multi-core schema
'''
Note here deltaE is defined as 2*(E2-E3), when implementing, we need E2/2
'''
#for this specific example, we only compute 4 points to save computation time
Elist = [100,200,300,400]
print('number of total points: ',np.size(Elist))
#%% configure parallel computation task
ncores = 4 # number of cores to use
tdict = mcs.generate_task(ncores,Elist)
print('task dictionary', tdict) 
#%%  define task function
ion_sys.active_phonon = [[1,2]] #consider com, tilt, and 
pcut_cri = 200 # critirion to change phonon cut off 
pcut_list = [[[3,6]],[[3,8]]] #phonon cutoff to be used for different parameters
def spin_evolution(task,Earray):
    '''
    solve time evolution for a single energy splitting
    Parameters
    ----------
    task : string 
        task name
    Elist : np array
        input site energy difference for the simulation task
    Returns
    -------
        task: task name
        sresult: a list of arrays that represents updown state population
        evolution at different deltaE
    '''
    sresult = []
    for E2 in Earray[0]:
        if E2 <= pcut_cri:
            ion_sys.pcut = pcut_list[0]
        else:
            ion_sys.pcut = pcut_list[1]
        oplist = [exop.spin_measure(ion_sys,[0,1]),
                  exop.spin_measure(ion_sys,[1,0])]
        elist = oplist
        rho0 = exop.rho_thermal(ion_sys)
        H0 = extrans.H_res(J23,E2/2,E3,V,ion_sys)
        clist1 = exop.c_op(ion_sys)
        result = mesolve(H0,rho0,times,clist1,elist,progress_bar=True,options=Options(nsteps=100000))
        sresult.append(result.expect[0])
    return {task:sresult}   
#%% parallel computation
if __name__ == '__main__':
    print('start parallel computing')
    print('number of cores used:',ncores,'/',mp.cpu_count())
    start_t = datetime.datetime.now() #record starting time
    pool = mp.Pool(ncores)
    results = [pool.apply_async(spin_evolution, args=(ntask, nEarray)) for ntask, nEarray in tdict.items()]
    pool.close()
    result_list_tqdm = [] #generate progress bar
    for result in tqdm(results):
        result_list_tqdm.append(result.get()) 
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("time consumed " + "{:.2f}".format(elapsed_sec) + "s")    
    sevl = [] #combine all the results consistent with deltaE input
    for i in range(ncores):
        sevl = sevl + result_list_tqdm[i][str(i)]
    print('________________________________________________________________')   
    print('all computation completed')
    #save data and plot
    ndata = Table()
    ndata['time'] = tplot
    filename = ('3ionexci_J='+str(J23)+'_d='+str(ion_sys.delta)+'_'+
                'fsb='+str(ion_sys.fr)+'.csv')  
    plt.figure(0)
    for i in range(np.size(Elist)):
        E0 = Elist[i]
        ndata[str(E0)] = sevl[i]
        plt.plot(tplot,ndata[str(E0)],label=r'$\Delta E=$'+str(E0)+'kHz')
    plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 14)
    plt.ylabel(r'$P_{\uparrow\downarrow}$',fontsize = 14)
    plt.grid()   
    plt.yticks(fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.legend(fontsize=13)
    plt.show()    
    ascii.write(ndata, filename, format ="csv",  overwrite=True) 
    print('saved to file')
