# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:28:18 2022
@author: zhumj
Compute basic quantities of the anharmonic coupling terms 
"""

from  Qsim.ion_chain.ion_system import *
import numpy as np
import matplotlib.pyplot as plt
#%%
ion0 = ions()
ion0.N=3
#verify C is invariant under the exchange of two indexes
print('test with index 1,1,0')
print(ion0.C(1,1,0))
print(ion0.C(1,0,1))
print(ion0.C(0,1,1))
print('_____________________________________________')
#varify that Cmmn = -Cmnn
print('test with index 1,0,0')
print(ion0.C(1,0,0))
print(ion0.C(0,0,1))
print(ion0.C(0,1,0))
print('_____________________________________________')
#%% test tensor D
#compare result with theoretical calculation in Marquet's paper eq(25)
#also varify D is symmetric under exchange of two indexes
#note the python index is different from what is used in the paper
print('a 3 ion system')
ion0.N=3
print('compute D233')
print('theoretical result: ', -3/np.sqrt(2)*(4/5)**(4/3))
print('numeric result:') 
print(ion0.D(1,2,2))
print(ion0.D(2,1,2))
print(ion0.D(2,2,1))
print('_____________________________________________')
print('compute D222')
print('theoretical result: ', -1/np.sqrt(2)*(4/5)**(1/3))
print('numeric result:',ion0.D(1,1,1)) 
print('_____________________________________________')
print('a 2 ion system')
ion0.N=2
print('compute D222')
print('theoretical result: ', -2**(1/6))
print('numeric result:',ion0.D(1,1,1)) 
#%% plot all combinations of non-zero D 
ion0.N=3
Dplot = {}
N = ion0.N 
for i in range(N):
    for j in range(N):
        for k in range(N):
            Dvalue = np.abs(ion0.D(i,j,k))
            if Dvalue>1e-5:
                Dplot[str(i+1)+'\n'+str(j+1)+'\n'+str(k+1)]=np.abs(Dvalue)
names = list(Dplot.keys())
values = list(Dplot.values())
plt.bar(range(len(Dplot)), values, tick_label=names)  
plt.ylabel(r'$|D_{mnp}|$',fontsize = 13)      
plt.xticks(fontsize = 13)  
plt.yticks(fontsize = 13)  
plt.xlabel('Mode index mnp: m,n for radial, p for axial',fontsize = 13)
plt.grid()
#%% Check the result with PRL 119 
#compute coupling frequency for mnp = 332
#set the frequency such that wa = 2wb is satisfied
ion0.N=3
ion0.fx = 3.5
ion0.fz = (20/63)**0.5 * ion0.fx
print('axial confining frequency ', ion0.fz) 
#%%
z0 = (5*qe**2 / (16*np.pi * eps0 * MYb171 * fr_conv(ion0.fz,'mhz')**2))**(1/3)
wa = ion0.Axialfreq()[1]*fr_conv(ion0.fz,'mhz')
wb = ion0.Transfreq()[2]*fr_conv(ion0.fz,'mhz')
th_result = 9* fr_conv(ion0.fz,'mhz') * np.sqrt(h/(MYb171*wa*wb**2))/(10*z0)
print('_____________________________________________')
print('theoretical result',th_result)
print('numeric result',ion0.ah_couple(2,2,1))
#%% plot anharmonic coupling coefficient for mode 332 with fx ranging from 2-4MHz
#fx, fz satisfy the reasonant condition for this mode
anc_plot = np.array([])
fxplot = np.arange(2,4,0.01) 
for fx in fxplot:
    ion0.fx = fx
    ion0.fz = (20/63)**0.5 * fx
    anc_plot = np.append(anc_plot, ion0.ah_couple(2,2,1)*ion0.fz*10**6) 
    #multiply fz in Hz to get real coupling in unit of frequency
#%%
plt.plot(fxplot,anc_plot)
plt.xlabel('Radial frequency, [MHz]',fontsize = 13) 
#plt.ylabel(r'Anharmonic Coupling, $[\omega_z]$',fontsize = 13)
plt.ylabel(r'Anharmonic Coupling, Hz',fontsize = 13)
plt.xticks(fontsize = 12)    
plt.yticks(fontsize = 12)   
plt.grid()   