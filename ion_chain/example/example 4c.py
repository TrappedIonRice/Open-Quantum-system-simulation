# -*- coding: utf-8 -*-
"""
Simulate the time evolution of excitation transfer at different dissipation gamma
with multi cores parallel coumputation,
Note: this script must be saved before running if any changes have been made
(a strange feature of multiprocessing package)
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.ion_chain.transfer.exci_transfer as extrans
import Qsim.ion_chain.transfer.multi_core as mcs
import Qsim.ion_chain.transfer.exci_operators as exop
from  Qsim.ion_chain.ising.ion_system import *
from astropy.io import ascii
from astropy.table import Table
import multiprocessing as mp
from tqdm import tqdm
import datetime
#%% initialize parameters of ion-chain system 
'''
set parameters of the system, simulate the evolution at a type 1 resonance DeltaE = 100
'''    
ion_sys = ions() 
ion_sys.delta_ref = 2
ion_sys.N = 3
ion_sys.delta = -100
fsb = 50
ion_sys.fr = fsb; ion_sys.fb = fsb
ion_sys.phase = np.pi/2
ion_sys.plot_freq()
ion_sys.gamma = [0,0,0]
J23 = 1
E2 = 100
E3 = 0
V = 0
print('coupling strength between ion 1 and 2', J23, ' kHz *h')
print('site energy of ion 2 ', E3, ' kHz *h')
configuration = 0 #0 for side cooling
tscale = J23
#configure time scale for computation
end = 1   #end of simulation time
tplot = np.arange(0,end,0.01)/tscale
times = tplot/tscale
ion_sys.list_para() #print parameters
#%% configure multi-core schema
'''
Note here deltaE is defined as 2*(E2-E3), when implementing, we need E2/2
'''
#for this specific example, we only compute 4 points to save computation time
glist = [1,5,10,15]
print('number of total points: ',np.size(glist))
#%%
ncores = 4 # number of cores to use
tdict = mcs.generate_task(ncores,glist)
print('task dictionary', tdict) 
#%%  define task function
ion_sys.active_phonon = [[1,2]] #consider com, tilt, and 
pcut_cri = 5 # critirion to change phonon cut off 
pcut_list = [[[3,6]],[[3,8]]] #phonon cutoff to be used for different parameters
def spin_evolution_g(task,Garray):
    '''
    solve time evolution for a single energy splitting
    Parameters
    ----------
    task : string 
        task name
    Garray : np array
        input site energy to be computed for the task

    Returns
    -------
        task: task name
        sresult: a list of arrays that represents updown state population
        evolution at different deltaE
    '''
    sresult = []
    for g in Garray[0]:
        if g <= pcut_cri:
            ion_sys.pcut = pcut_list [0]
        else: 
            ion_sys.pcut = pcut_list [1]
        ion_sys.gamma = [0,0,g]
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
    results = [pool.apply_async(spin_evolution_g, args=(ntask, ngarray)) for ntask, ngarray in tdict.items()]
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
                'fsb='+str(ion_sys.fr)+'_E='+str(E2)+'.csv')   
    plt.figure(0)
    for i in range(np.size(glist)):
        gamma = glist[i]
        ndata[str(gamma)] = sevl[i]
        plt.plot(tplot,ndata[str(gamma)],label=r'$\gamma =$'+str(gamma)+'kHz')
    plt.xlabel(r'$\omega_0t/(2\pi)$',fontsize = 14)
    plt.ylabel(r'$P_{\uparrow\downarrow}$',fontsize = 14)
    plt.grid()   
    plt.yticks(fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.legend(fontsize=13)
    plt.show()    
    ascii.write(ndata, filename, format ="csv",  overwrite=True) 
    print('saved to file')
