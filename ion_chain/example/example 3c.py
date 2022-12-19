# -*- coding: utf-8 -*-
"""
plot the energy diagram for excitation transfer Hamiltonian 
under semi-classical approximation
@author: zhumj
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import ion_chain.transfer.exci_transfer as extrans
import ion_chain.eigendiagram.exci_diagram as eigd
from  ion_chain.ising.ion_system import *
from scipy.optimize import curve_fit
#%%
'''
parameters of the system, in this example, we compute the energydiagram at type 1
reasonance at Delta E = 1*delta_rock
'''  
ion_sys = ions()  
ion_sys.N = 3
ion_sys.delta = -100
ion_sys.delta_ref = 0
ion_sys.phase = np.pi/2
fsb = 50
ion_sys.fr = fsb; ion_sys.fb = fsb
ion_sys.list_para() #print parameters
ion_sys.plot_freq()
J12 = 1; 
E2 = 0;V = 0
E1 = 100
#%%                
xplot = np.arange(-10,10,0.1) #rocking mode displacement 
eige2,eige3 = eigd.energy_diagram_2d(ion_sys,J12,E1/2,E2,V,xplot)  #compute eigenenergy  
#%%
plt.plot(xplot,eige2,label = r'$\downarrow\!\!\!\!\uparrow$')
plt.plot(xplot,eige3,label = r'$\uparrow\!\!\!\!\downarrow$')
plt.plot(xplot,eige2,'b')
plt.plot(xplot,eige3,'r')
plt.xlabel(r'Displacement x, $X_0$')
plt.title(r'$\Delta E=$'+str(E1)+'kHz')
plt.ylabel('Energy [kHz]')
plt.legend(fontsize=15)
plt.grid()
#%% fit parabola to extract parameters
c0 = int(np.size(xplot)/2 - 1) 
coef2, pcov2=curve_fit(eigd.parabola,xplot,eige2,p0=[10,xplot[c0],eige2[c0]],
                     bounds=([0,-1000,np.min(eige2)], [1000,1000,0]),maxfev=5000)
v2 = eigd.vertex(coef2)
vlabel2 = 'vertex coord1: ' + str(np.round(v2[0],1)) + ' x0, ' + str(np.round(v2[1],1)) + ' kHz'
plt.plot(xplot,eige2,'+')
plt.plot(v2[0],v2[1],'rx',label=vlabel2)
plt.plot(xplot,eigd.parabola(xplot,coef2[0],coef2[1],coef2[2]),label='parabola fit1')
coef3, pcov3=curve_fit(eigd.parabola,xplot,eige3,p0=[10,xplot[c0],eige3[c0]],
                     bounds=([0,-1000,np.min(eige3)], [1000,1000,0]),maxfev=5000)
v3 = eigd.vertex(coef3)
vlabel3 = 'vertex coord2: ' + str(np.round(v3[0],1)) + ' x0, ' + str(np.round(v3[1],1)) + ' kHz'
plt.plot(xplot,eige3,'+')
plt.plot(v3[0],v3[1],'kx',label=vlabel3)
plt.plot(xplot,eigd.parabola(xplot,coef3[0],coef3[1],coef3[2]),label='parabola fit 2')
#compute intersection
xi = eigd.intersect(coef2-coef3); yi = eigd.parabola(xi,coef3[0],coef3[1],coef3[2])
ilabel = 'intersection: ' + str(np.round(xi,1)) + ' x0, ' + str(np.round(yi,1)) + ' kHz'
#plot
plt.plot(xi,yi,'b*',label = ilabel)
plt.title(r'$\Delta E=$'+str(E1)+'kHz',fontsize = 13)
plt.xlabel(r'Displacement x, $X_0$',fontsize = 13)
plt.ylabel('Energy [kHz]',fontsize = 13)
plt.legend()
plt.grid(True)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.legend(fontsize=10)
plt.show()
#%%plot energy level
Nlev = 10
min1 = np.min(eige2); min2 = np.min(eige3)
elevel1 = eigd.elevel(ion_sys,min1,Nlev,0)
elevel2 = eigd.elevel(ion_sys,min2,Nlev,0) #0 for rocking mode
plt.figure(0)
plt.plot(xplot,eige2,label = r'$\downarrow\!\!\!\!\uparrow$')
plt.plot(xplot,eige3,label = r'$\uparrow\!\!\!\!\downarrow$')
for i in range(Nlev):
    xcoor = [-10,10]
    ycoor1 = [elevel1[i],elevel1[i]]
    ycoor2 = [elevel2[i],elevel2[i]]
    plt.plot(xcoor,ycoor1,'r')
    plt.plot(xcoor,ycoor2,'k--', linewidth=3)
plt.plot(xi,yi,'b*',label = ilabel)   
plt.xlabel(r'Displacement x, $X_0$',fontsize = 13)
plt.ylabel('Energy [kHz]',fontsize = 13)
tilt_lab = str(np.round(ion_sys.dmlist()[1]/(2*np.pi),2))
plt.title(r'$\Delta E=$'+str(np.round(2*E1,2))+'kHz '+ r'$\delta_{rock}=$'+str(ion_sys.delta)+' kHz '
          + r'$\delta_{tilt}=$'+tilt_lab+' kHz')
plt.ylim(-1000,1000)
plt.legend(fontsize=10)
plt.yticks(fontsize = 13)
plt.xticks(fontsize = 13)
plt.grid()
plt.show()
print('difference between vertexes ', np.round(np.abs(v2[1]-v3[1]),1) )
print('distance between minimum and intersection', np.round(np.abs(xi-xplot[np.argmin(eige3)]),3))