# -*- coding: utf-8 -*-
"""
Spyder Editor

Compute the time evolution of 8-level open quantum system with a specified
rabi-frequency
"""
#from IPython.display import Image
import sys
import datetime
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import Qsim.operator.spin as spin
import Qsim.operator.phonon as phon
import Qsim.ion_chain.transfer.elec_transfer as etrans
from  Qsim.ion_chain.ion_system import *
import Qsim.operator.spin_phonon as sp_op
import Qsim.ion_chain.interaction.dissipation as disp

import Qsim.ion_chain.interaction.spin_phonon as sp_ph
import Qsim.auxiliay_function.data_fitting as fit
import os
from scipy.optimize import curve_fit
import pandas as pd
from lmfit import Model


qutip.settings.has_mkl = False  # to avoid OS error in my PC

# %%
# define constants

# energy levels
# 2-> bracket state 1 [3/2] 1/2
# 1-> 2D3/2
# 0-> 2S1/2

# spontaneous emission
G21 = 4*0.018 *2* np.pi 
G20=4*0.982 *2*np.pi # in 2pi* MHz and this sets the approx timescale in which we want to compute everything else.
h = 1  # plank constant with /2pi

# %%
# ladder operators
ncutoff=15
a=destroy(ncutoff)
phon_eye=phon.pI([ncutoff],1)
spin_eye=Qobj(np.eye(3,3))
full_eye=tensor(spin_eye,phon_eye)




# %%
# Compute collapse operators
def col_o(index, ele):
    '''
    construct collapse operator by setting the ij element
    '''
    cm = np.zeros((3, 3))
    cm[index[0], index[1]] = ele
    nc = Qobj(cm)
    return nc

clist = []
ilist1 = [[1, 2]]
ilist2 = [[0, 2]]
for iindex in ilist1:
    clist.append(col_o(iindex, np.sqrt(G21)))
for iindex in ilist2:
    clist.append(col_o(iindex, np.sqrt(G20)))

clistWithPh=[tensor(Qobj(clist[i]),phon_eye ) for i in range(len(clist))]

#additional heating from 297
k297=2*np.pi/(297*10**-9)
m171=171*1.67*10**-27
wrecoil=(6.626*10**-34/(2*np.pi))*k297**2/(2*m171)
wtrap=3*10**6*2*np.pi;
eta=np.sqrt(wrecoil/wtrap)
motionalG21=eta*np.sqrt(G21)
motionalG20=eta*np.sqrt(G20)

# separate jump operators for heating and cooling from scattered 297
# heating297_c= tensor(col_o(ilist2[0], motionalG20), a.dag() )
# heating935_c= tensor(col_o(ilist1[0], motionalG21), a.dag() ) 
# cooling297_c= tensor(col_o(ilist2[0], motionalG20), a )
# cooling935_c= tensor(col_o(ilist1[0], motionalG21), a )

# clistWithPh.append(heating935_c)
# clistWithPh.append(heating297_c)
# clistWithPh.append(cooling297_c)
# clistWithPh.append(cooling935_c)


    

# set initial density matrix
n0=1
spin_rho0 = np.zeros((3, 3));
spin_rho0[0, 0] = 1
phon_rho0=np.zeros((ncutoff,ncutoff))
phonarr=phon.p_thermal(ncutoff,n0)
for i in range(ncutoff):
    phon_rho0[i,i]=phonarr[i]

rho0 =tensor( Qobj(spin_rho0),Qobj(phon_rho0))

for c in clistWithPh:
    print(c)


# laser and detuning parameters

det935=-0.0*2*np.pi
det435=0.*2*np.pi
O435=0.0125*2*np.pi
# Psat for 20umx20um profile is 12.647nW
O935=0.051*2*np.pi



# compute Hamiltonian

#function of te
# works
def CSBC_te_funcWithCGsigns(det_935,det_435,O_935,O_435,rho_0,times_,clist_):

# On1 = f * G369 * np.sin(alpha) * np.cos(beta);
# O0 = f * G369 * np.cos(alpha);
# O1 = f * G369 * np.sin(alpha) * np.cos(beta);

    hm = np.zeros((3, 3))
    # diagonal elements
    hm[0, 0] = det_435;
    hm[1, 1] = det_935;
    hm[2, 2] = 0;
    # offdiagonal
    hm[1, 2]= O_935/2
    hm[2, 1]= O_935/2
    
    hm_base=tensor(Qobj(hm),phon_eye)
    
    hm_lower=np.zeros((3,3))
    hm_lower[0,1]=O_435/2
    RSB_spinlower=tensor(Qobj(hm_lower),a.dag()) # sigma * adagger
    RSB_spinupper=RSB_spinlower.dag()
    
    H=hm_base + RSB_spinlower + RSB_spinupper
    #print(H.isherm)
    res=mesolve(H, rho_0, times_, clist_, [], progress_bar=True)
    return res

# in progress. Doesn't work properly yet
def PSBC_te_funcWithCGsigns(det_935,det_435,O_935,O_435,rho_0,pulsetime_435, pulsetime_935, clistWithPh_, Npulses_):

# On1 = f * G369 * np.sin(alpha) * np.cos(beta);
# O0 = f * G369 * np.cos(alpha);
# O1 = f * G369 * np.sin(alpha) * np.cos(beta);

    hm = np.zeros((3, 3))
    # diagonal elements
    hm[0, 0] = det_435;
    hm[1, 1] = det_935;
    hm[2, 2] = 0;
    # offdiagonal
    hm[1, 2]= O_935/2
    hm[2, 1]= O_935/2
    
    hm_base=tensor(Qobj(hm),phon_eye)
    
    hm_lower=np.zeros((3,3))
    hm_lower[0,1]=O_435/2
    RSB_spinlower=tensor(Qobj(hm_lower),a.dag()) # sigma * adagger
    RSB_spinupper=RSB_spinlower.dag()
    
    H=hm_base+RSB_spinlower+RSB_spinupper
    
    emptytensor=tensor(Qobj(np.zeros((3,3))),Qobj(np.zeros((ncutoff,ncutoff))))
    rhotemp=rho_0
    for j in range(Npulses_):
        res1=mesolve(H, rhotemp, pulsetime_435, [emptytensor], [], progress_bar=True)
        rhotemp=res1.states[-1]
        res2=mesolve(H, rhotemp, pulsetime_935, clistWithPh_, [], progress_bar=True)
        rhotemp=res2.states[-1]
        # now just to bring it to the solver.Result datatype in the end.
        if j==(Npulses_-1):
            res2=mesolve(H, rhotemp, np.array([0]), clistWithPh_, [], progress_bar=True)
    return res2 


# works

def reservoirEngineering_te_funcWithCGsigns(det_935,det_435RSB,det_435BSB,O_935,O_435RSB,O_435BSB, rho_0,times_,clist_):

# On1 = f * G369 * np.sin(alpha) * np.cos(beta);
# O0 = f * G369 * np.cos(alpha);
# O1 = f * G369 * np.sin(alpha) * np.cos(beta);

    hm = np.zeros((3, 3))
    # diagonal elements
    hm[0, 0] = det_435RSB; # for now only using the detuning from the RSB
    hm[1, 1] = det_935;
    hm[2, 2] = 0;
    # offdiagonal
    hm[1, 2]= O_935/2
    hm[2, 1]= O_935/2
    
    hm_base=tensor(Qobj(hm),phon_eye)
    
    hm_RSB=np.zeros((3,3))
    hm_BSB=np.zeros((3,3))
    hm_RSB[0,1]=O_435RSB/2
    hm_BSB[0,1]=O_435BSB/2
    
    RSB_spinlower=tensor(Qobj(hm_RSB),a.dag()) # sigma * a_dag
    RSB_spinupper=RSB_spinlower.dag() # sigma_dag * a
    
    BSB_spinlower=tensor(Qobj(hm_BSB),a) # sigma * a
    BSB_spinupper=BSB_spinlower.dag() # sigma_dag * a_dag
    
    
    H=hm_base + RSB_spinlower + RSB_spinupper + BSB_spinlower + BSB_spinupper
    res=mesolve(H, rho_0, times_, clist_, [], progress_bar=True)
    return res


# compute time evolution
T=4000 # total pumping time in microseconds
steps=int(G21*T*0.25) # constant is multiplier factor
times = np.linspace(0, T, steps)  # time is in microseconds

#%% Single te evolution

# comment out only required res_single command to observe respective evolution.

#CSBC
res_single= CSBC_te_funcWithCGsigns(det935,det435,O935,O435,rho0,times,clistWithPh)

#Pulsed SBC (requires improvement, do not use)
Npulses0=10
pulse435times=np.linspace(0,np.pi*10**3/(O435),800)
pulse935times=np.linspace(0,50,800)
#res_single=PSBC_te_funcWithCGsigns(det935,det435,O935,O435,rho0,pulse435times,pulse935times,clistWithPh,Npulses0)

#Reservoir engineering -squuezed state
r=1.45
O435RSB=O435
O435BSB=np.tanh(r)*O435RSB
det435RSB=0
det435BSB=det435RSB
#res_single= reservoirEngineering_te_funcWithCGsigns(det935,det435RSB,det435BSB,O935,O435RSB,O435BSB, rho0,times,clistWithPh)


#%% extracting average phonon number for single te

# the usual to look at phonons
navg=np.array([np.sum(np.diag(res_single.states[i].ptrace(1))*np.arange(ncutoff)) for i in range(len(times))]) 
# navg=np.sum(np.diag(res_single.states[-1].ptrace(1))*np.arange(ncutoff))
# print(navg)


# plt.figure(2)
# ax=plt.axes()
# ax.set_yscale('log')
# n_final=np.array([res.states[-1].ptrace(1)[j,j] for j in range(ncutoff)])

# plt.plot(range(ncutoff),n_final, label='Final')
# plt.ylabel('$p(n)$')
# plt.xlabel('$n$')
# title='$\Omega_{{435}}=$ {0:.3f} kHz, $\Omega_{{935}}=$ {1:.3f} kHz,\n $\Delta_{{435}}=$ {2:.3f} kHz, $\Delta_{{935}}=$ {3:.3f} kHz, $\gamma/2\pi$= {4:.3f} kHz, n0={5:.3f}'\
#     .format(O435*1000/(2*np.pi),O935*1000/(2*np.pi),det435*1000/(2*np.pi),det935*1000/(2*np.pi), np.abs(coolingrate*1000/(2*np.pi)), n0)

# plt.title(title)
# plt.legend()
# plt.show()

#%%
#Plotting for single evolution


# single time trace
plt.clf()
plt.figure(1)
plt.plot(times,navg)
plt.ylabel('$\overline{n}$')
plt.xlabel('$t(\mu s)$')
coolingrate=fit.et_decay_integrate(times,navg) # in units of ms^-1
title='$\Omega_{{435}}=$ {0:.3f} kHz, $\Omega_{{935}}=$ {1:.3f} kHz,\n $\Delta_{{435}}=$ {2:.3f} kHz, $\Delta_{{935}}=$ {3:.3f} kHz, $\gamma/2\pi$= {4:.3f} kHz, n0={5:.3f}'\
    .format(O435*1000/(2*np.pi),O935*1000/(2*np.pi),det435*1000/(2*np.pi),det935*1000/(2*np.pi), np.abs(coolingrate*1000/(2*np.pi)), n0)

plt.title(title)
plt.show()

# plot initial and final phonon distributions
plt.figure(2)
ax=plt.axes()
ax.set_yscale('log')
n_init=np.array([res_single.states[0].ptrace(1)[j,j] for j in range(ncutoff)])
n_final=np.array([res_single.states[-1].ptrace(1)[j,j] for j in range(ncutoff)])

plt.plot(range(ncutoff),n_init, label='Initial')
plt.plot(range(ncutoff),n_final, label='Final')
plt.ylabel('$p(n)$')
plt.xlabel('$n$')
title='$\Omega_{{435}}=$ {0:.3f} kHz, $\Omega_{{935}}=$ {1:.3f} kHz,\n $\Delta_{{435}}=$ {2:.3f} kHz, $\Delta_{{935}}=$ {3:.3f} kHz, $\gamma/2\pi$= {4:.3f} kHz, n0={5:.3f}'\
    .format(O435*1000/(2*np.pi),O935*1000/(2*np.pi),det435*1000/(2*np.pi),det935*1000/(2*np.pi), np.abs(coolingrate*1000/(2*np.pi)), n0)

plt.title(title)
plt.legend()
plt.show()


# plot spin state evolution across 435 transition
spin0pop=np.array([res_single.states[i].ptrace(0)[0,0] for i in range (len(times))])


plt.figure(3)
plt.plot(times,spin0pop)
plt.ylabel('$P(0)$')
plt.xlabel('$t(\mu s)$')
title='$\Omega_{{435}}=$ {0:.3f} kHz, $\Omega_{{935}}=$ {1:.3f} kHz,\n $\Delta_{{435}}=$ {2:.3f} kHz, $\Delta_{{935}}=$ {3:.3f} kHz, n0={4:.3f}'\
    .format(O435*1000/(2*np.pi),O935*1000/(2*np.pi),det435*1000/(2*np.pi),det935*1000/(2*np.pi), n0)

plt.title(title)
plt.show()



#%% Simulation of cooling by varying detuning of 935

det935_arr=np.arange(-0.010,0.010,0.005)*2*np.pi # MHz times 2pi
navg_arr=[]
res_arr=[]
coolingrate_arr=[]

for ii in range(len(det935_arr)):
    res_arr.append(CSBC_te_funcWithCGsigns(det935_arr[ii],det435,O935,O435,rho0,times,clistWithPh))
    navg_arr.append(np.array([np.sum(np.diag(res_arr[ii].states[i].ptrace(1))*np.arange(ncutoff)) for i in range(len(times))]))
    coolingrate_arr.append(fit.et_decay_integrate(times,navg_arr[ii]))

#%% plots for diff 935 detuning
plt.figure()
plt.plot(np.array(det935_arr)*1000/(2*np.pi),np.array(coolingrate_arr)*1000/(2*np.pi))
plt.ylabel(r'$\gamma/(2\pi)$ (kHz)')
plt.xlabel(r'$\Delta_{{935}}$ (kHz)')
title0='$\Omega_{{435}}=$ {0:.3f} kHz, $\Omega_{{935}}=$ {1:.3f} kHz,\n $\Delta_{{435}}=$ {2:.3f} kHz, $\gamma/2\pi$= {3:.3f} kHz, n0={4:.3f}'\
    .format(O435*1000/(2*np.pi),O935*1000/(2*np.pi),det435*1000/(2*np.pi), np.abs(coolingrate*1000/(2*np.pi)), n0)
plt.title(title0)
plt.grid(visible=True)
plt.show()

#%% Simulation of CSBC by varying power of 935


O935_arr=np.arange(0.081,0.62,0.05)*2*np.pi # MHz times 2pi
navg_arr=[]
res_arr=[]
coolingrate_arr=[]

for ii in range(len(O935_arr)):
    res_arr.append(CSBC_te_funcWithCGsigns(det935,det435,O935_arr[ii],O435,rho0,times,clistWithPh))
    navg_arr.append(np.array([np.sum(np.diag(res_arr[ii].states[i].ptrace(1))*np.arange(ncutoff)) for i in range(len(times))]))
    coolingrate_arr.append(fit.et_decay_integrate(times,navg_arr[ii]))

#%% plots for 935 power variation
plt.clf()
plt.figure()
plt.plot(np.array(O935_arr)*1000/(2*np.pi),np.array(coolingrate_arr)*1000/(2*np.pi))
plt.ylabel(r'$\gamma/(2\pi)$ (kHz)')
plt.xlabel(r'$\Omega_{{935}}$ (kHz)')
title0='$\Omega_{{435}}=$ {0:.3f} kHz, $\Delta_{{935}}=$ {1:.3f} kHz,\n $\Delta_{{435}}=$ {2:.3f} kHz, $\gamma/2\pi$= {3:.3f} kHz, n0={4:.3f}'\
    .format(O435*1000/(2*np.pi),det935*1000/(2*np.pi),det435*1000/(2*np.pi), np.abs(coolingrate*1000/(2*np.pi)), n0)
plt.title(title0)
plt.show()

# time traces
plt.figure()
for ii in range(len(O935_arr)):
    plt.plot(times,navg_arr[ii], label='$\Omega_{{935}}=$ {0:.3f} kHz'.format(O935_arr[ii]*1000/(2*np.pi)))
    
plt.legend()
plt.ylabel('$\overline{n}$')
plt.xlabel('$t(\mu s)$')
title0='$\Omega_{{435}}=$ {0:.3f} kHz, $\Delta_{{935}}=$ {1:.3f} kHz,\n $\Delta_{{435}}=$ {2:.3f} kHz, n0={3:.3f}'\
    .format(O435*1000/(2*np.pi),det935*1000/(2*np.pi),det435*1000/(2*np.pi), n0)
plt.title(title0)
plt.grid(visible=True)
plt.show()
    
    

#%% exporting dataset for single te


filedirec=r"Z:\gp31\Lab Rice\Projects in Progress\AM\ET simulations\sim_datasets\2024-2-1\\"

finalrho=res_single.states
# change name according to command
simType=r"CSBC_435_"

rhofile=filedirec+simType+datetime.datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
rhotimefile=filedirec+simType+"time_arr_"+datetime.datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+".csv"


qsave(finalrho,rhofile)
np.savetxt(rhotimefile,times)

print("Saved rho data to "+rhofile+".qu")
print("Saved time series to "+rhotimefile+".csv")


#%% exporting data set for multiple te

filedirec=r"Z:\gp31\Lab Rice\Projects in Progress\AM\ET simulations\sim_datasets\2024-2-1\\"

scan_arr=det935_arr # to be changed according to scan, eg. of scanning parameter

for i in range (len(scan_arr)):
    if i==0:
        finalrho_arr=res_arr[i].states#RamseyTime(farr[i],rho0,times,dataindex)
    else:
        finalrho_arr=np.vstack((finalrho_arr,res_arr[i].states))
   # print("f={0:.3f} completed.".format(farr[i]))

# change name according to command
simType=r"CSBC_435_multiple_"

multiple_rhofile=filedirec+simType+datetime.datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
multiple_rhotimefile=filedirec+simType+"time_arr_"+datetime.datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+".csv"
multiple_rhoscanfile=filedirec+simType+"det935_arr_"+datetime.datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+".csv"

qsave(finalrho_arr,multiple_rhofile)
np.savetxt(multiple_rhotimefile,times)
np.savetxt(multiple_rhoscanfile,scan_arr)

print("Saved rho data to "+multiple_rhofile+".qu")
print("Saved time series to "+multiple_rhotimefile)
print("Saved scan series to "+multiple_rhoscanfile)

#%% loading dataset for single te
rho_file=r'Z:\gp31\Lab Rice\Projects in Progress\AM\ET simulations\sim_datasets\2024-2-1\CSBC_435_Feb_03_2024_11_44_17'
imported_rho=qload(rho_file)
time_file=r'Z:\gp31\Lab Rice\Projects in Progress\AM\ET simulations\sim_datasets\2024-2-1\CSBC_435_time_arr_Feb_03_2024_11_44_18.csv'
imported_time=np.loadtxt(time_file)
print(r"Loaded rho file: "+rho_file+r'.qu \n Loaded time file: '+time_file)

# reload back rho_file and time_file as needed

#%% loading data set for multiple te


multiple_rho_file=r'Z:\gp31\Lab Rice\Projects in Progress\AM\ET simulations\sim_datasets\2024-2-1\CSBC_435_multiple_Feb_03_2024_12_16_01'
imported_multiple_rho=qload(multiple_rho_file)
multiple_rho_time_file=r'Z:\gp31\Lab Rice\Projects in Progress\AM\ET simulations\sim_datasets\2024-2-1\CSBC_435_multiple_time_arr_Feb_03_2024_12_16_01.csv'
imported_multiple_rho_time=np.loadtxt(multiple_rho_time_file)
multiple_rho_scan_file=r'Z:\gp31\Lab Rice\Projects in Progress\AM\ET simulations\sim_datasets\2024-2-1\CSBC_435_multiple_det935_arr_Feb_03_2024_12_16_01.csv'
imported_multiple_rho_scan=np.loadtxt(multiple_rho_scan_file)

print(r"Loaded rho file: "+multiple_rho_file+r'.qu \n Loaded time file: '+multiple_rho_time_file \
      +r'\n Loaded scan file: '+multiple_rho_scan_file)

# reload back rho_file, time_file, and scan_file  as needed









