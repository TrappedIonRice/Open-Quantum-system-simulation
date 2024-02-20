# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:33:34 2024

@author: abhim


Cooling 171
"""


import matplotlib.pyplot as plt
import numpy as np
#from IPython.display import Image
from qutip import *
import sys
import datetime


qutip.settings.has_mkl = False  # to avoid OS error in my PC
plt.close('all')
# %%
# define constants
# damping
GYb = 19.7 * 2 * np.pi # in MHz and this sets the timescale in which we want to compute everything else.
Geg = GYb
G = GYb
# laser constants
# unused
Dd = -2105 * 2 * np.pi;
Dg = 12600 * 2 * np.pi
# Rabi Frequency
#f1 =0.1 # gives s=0.02
s=4.25
f1=np.sqrt(s/2) # sqrt(s/2)
f2=np.sqrt(s/2*1/3)

# %%
# Compute collapse operators
def col_o(index, ele):
    '''
    construct collapse operator by setting the ij element
    '''
    cm = np.zeros((8, 8))
    cm[index[0], index[1]] = ele
    nc = Qobj(cm)
    return nc


clist = []
ilist1 = [[0, 5], [2, 5], [3, 5], [0, 6], [1, 6], [3, 6],
          [0, 7], [1, 7], [2, 7]]
ilist2 = [[1, 4], [2, 4], [3, 4]]
for iindex in ilist1:
    clist.append(col_o(iindex, np.sqrt(Geg / 3)))
for iindex in ilist2:
    clist.append(col_o(iindex, np.sqrt(G / 3)))
# set initial density matrix
idm = np.zeros((8, 8));
idm[2, 2] = 1
rho0 = Qobj(idm)
# %%
# oplot = np.arange(-100,101,1)
sslist = []
# laser frequency
del1=0*2*np.pi
del2=0*2*np.pi
DzG=6.178*2*np.pi#11.35*2*np.pi#6.178*2*np.pi
DzE=DzG/3
# detune

#polarization
#alpha=np.arccos(1/np.sqrt(3)) ; # angle btw linear polarization plane and B field, includes applied that generates +-8.33 Zeeman shift and B_earth=0.5G.
alpha=np.pi/4 #  2023/5/5 config 1 where B field makes an angle of 45deg with the polarization vector.
beta=np.pi/4; #angle for equal projection of sigma+ and sigma- in linear polarization: x= - (sigma_+ - sigma_-)/sqrt(2)
O1_n1 = f1 * GYb*np.sin(alpha)*np.cos(beta);
O1_0 = f1 * GYb*np.cos(alpha);
O1_1 = f1 * GYb*np.sin(alpha)*np.sin(beta);
h = 1  # plank, hbar actually 

O2_n1 = f2 * GYb*np.sin(alpha)*np.cos(beta);
O2_0 = f2 * GYb*np.cos(alpha);
O2_1 = f2 * GYb*np.sin(alpha)*np.sin(beta);




# compute Hamiltonian

#function of te
#def te_funcWithCGsigns_SS(del1,del2,rho0,times,clist,f1,f2):
def te_funcWithCGsigns_SS(del1,del2,clist,f1,f2,alpha,beta):

    O1_n1 = f1 * GYb*np.sin(alpha)*np.cos(beta)/np.sqrt(3);
    O1_0 = -1* f1 * GYb*np.cos(alpha)/np.sqrt(3);
    O1_1 = f1 * GYb*np.sin(alpha)*np.sin(beta)/np.sqrt(3);

    O2_n1 = f2 * GYb*np.sin(alpha)*np.cos(beta)/np.sqrt(3);
    O2_0 = -1*f2 * GYb*np.cos(alpha)/np.sqrt(3);
    O2_1 = f2 * GYb*np.sin(alpha)*np.sin(beta)/np.sqrt(3);

    hm = np.zeros((8, 8))
    # diagonal elements
    hm[0, 0] = del2;
    hm[1, 1] = del1-DzG;
    hm[2, 2] = del1;
    hm[3, 3] = del1+DzG;
    hm[4, 4] = 0.000;
    hm[5, 5] = DzE;
    hm[6, 6] = 0.000;
    hm[7, 7] = -DzE;
    # 1st row
    hm[0, 5] = O2_n1 / 2;
    hm[5, 0] = O2_n1 / 2
    hm[0, 6] = O2_0 / 2;
    hm[6, 0] = O2_0 / 2
    hm[0, 7] = O2_n1 / 2;
    hm[7, 0] = O2_n1 / 2
    
    # 2th row
    hm[1, 4] = 1*O1_1 / 2;   #zeroed to null detection transition and get pure pumping
    hm[4, 1] = 1*O1_1 / 2;
   
    # 3th row
    hm[2, 4] = 1*O1_0 / 2;
    hm[4, 2] = 1*O1_0 / 2;
    
    # 4th row
    hm[3, 4] = 1*O1_n1 / 2;
    hm[4, 3] = 1*O1_n1 / 2;
  
    H = Qobj(hm)
    #print(isherm(H))
    # compute time evolution
    # times = np.linspace(0,10/GYb,1000)
    #res = mesolve(H, rho0, times, clist, [])
    res=steadystate(H,clist)
    return res

def te_func(Dg,Dn1,D0,D1,Dd,D1e,D0e,Dn1e,On1,O0,O1,rho0,times,clist):

    hm = np.zeros((8, 8))
    # diagonal elements
    hm[0, 0] = Dg;
    hm[1, 1] = Dn1;
    hm[2, 2] = D0;
    hm[3, 3] = D1
    hm[4, 4] = Dd;
    hm[5, 5] = D1e;
    hm[6, 6] = D0e;
    hm[7, 7] = Dn1e
    # 1st row
    hm[0, 5] = O1 / 2;
    hm[5, 0] = O1 / 2
    hm[0, 6] = O1 / 2;
    hm[6, 0] = O0 / 2
    hm[0, 7] = On1 / 2;
    hm[7, 0] = On1 / 2
    # 2th row
    hm[1, 4] = On1 / 2;
    hm[4, 1] = On1 / 2
    hm[1, 6] = On1 / 2;
    hm[6, 1] = On1 / 2
    hm[1, 7] = O0 / 2;     #Only these pi transitions have -ve CG coeff no nulled by polarization component
    hm[7, 1] = O0 / 2
    # 3th row
    hm[2, 4] = O0 / 2;
    hm[4, 2] = O0 / 2
    hm[2, 5] = O1 / 2;
    hm[5, 2] = O1 / 2
    hm[2, 7] = On1 / 2;
    hm[7, 2] = On1 / 2
    # 4th row
    hm[3, 4] = O1 / 2;
    hm[4, 3] = O1 / 2
    hm[3, 5] = O0 / 2;
    hm[5, 3] = O0 / 2
    hm[3, 6] = O1 / 2;
    hm[6, 3] = O1 / 2
    H = Qobj(hm)
    # compute time evolution
    # times = np.linspace(0,10/GYb,1000)
    res = mesolve(H, rho0, times, clist, [])
    return res

def te_detuning(offs):
    offs = offs * 2 * np.pi;
    Dm1 = -6.178 * 2 * np.pi + offs;
    Dmm1 = 6.178 * 2 * np.pi + offs
    # detune
    Dn1 = Dm1;
    D0 = offs;
    D1 = Dmm1
    Dn1e = Dm1 / 3  # why does the excited frame also have an offset?
    D0e = 0;
    D1e = Dmm1 / 3
    res = te_funcWithCGsigns(Dg, Dn1, D0, D1, Dd, D1e, D0e, Dn1e, rho0, times, clist,f)
    return res

def RamseyTime(f,initialstate, timearr):
    # halfflip=np.zeros((8,8),dtype=complex)
    # rotangle=np.pi/4
    # halfflip[0,0]=np.cos(rotangle)
    # halfflip[2,2]=np.cos(rotangle)
    # halfflip[0, 2]=-1j*np.sin(rotangle)
    # halfflip[2, 0]=-1j*np.sin(rotangle)
    # halfflip[3:, 3:]=np.diag(np.ones(5))
    # halfflip=Qobj(halfflip)
    # seg1rho=halfflip*initialstate*halfflip.dag()
    # teresult=te_funcWithCGsigns(Dg,Dn1,D0,D1,Dd,D1e,D0e,Dn1e,seg1rho,timearr,clist,f)
    teresult=te_funcWithCGsigns(Dg,Dn1,D0,D1,Dd,D1e,D0e,Dn1e,initialstate,timearr,clist,f)
    # timeevolvedrho=teresult.states[-1] # just taking the last state of the time evolution.
    # seg2=halfflip*timeevolvedrho*halfflip.dag()
    seg2=teresult.states[-1]
    return seg2    
    
#te
T=21 # total pumping time in microseconds
steps=int(GYb*T)*10
times = np.linspace(0, T, steps)  # time is in microseconds
#result=te_funcWithCGsigns(Dg,Dn1,D0,D1,Dd,D1e,D0e,Dn1e,On1,O0,O1,rho0,times,clist)
finalrhoarr=[]


# plots for different scans

delta_arr=np.linspace(-80,80,100)*2*np.pi;
alpha_arr=np.linspace(-np.pi,np.pi,100)
beta_arr=np.linspace(-np.pi,np.pi,100)
s_arr=np.linspace(0.01,5,21)
f1_arr=np.sqrt(s_arr/2)

L=len(delta_arr)
#L=len(alpha_arr)
#L=len(beta_arr)
#L=len(f1_arr)


for i in range (L):
    finalrhoarr.append(te_funcWithCGsigns_SS(delta_arr[i],delta_arr[i],clist,f1,f2,alpha,beta))
    #finalrhoarr.append(te_funcWithCGsigns_SS(del1,del2,clist,f1,f2,alpha,beta_arr[i]))
    #finalrhoarr.append(te_funcWithCGsigns_SS(del1,del2,clist,f1_arr[i],f1_arr[i]/3,alpha,beta))


finalrhoarr=np.array(finalrhoarr)

# def RamseyTimeArr(f):
#     return RamseyTime(f,rho0,times)

#finalrhoarr=parallel_map(RamseyTimeArr,farr, progress_bar=True)

labels = ['g', 'm1', '0', '1', 'e', 'e1', 'e0', 'em1']


plt.figure()

for i in range(8):
    plt.plot(delta_arr/(2*np.pi),np.abs(finalrhoarr[:,i,i]),label=labels[i] )
    #plt.plot(alpha_arr,np.abs(finalrhoarr[:,i,i]),label=labels[i] )
    #plt.plot(beta_arr,np.abs(finalrhoarr[:,i,i]),label=labels[i] )
    #plt.plot(s_arr,np.abs(finalrhoarr[:,i,i]),label=labels[i] )



plt.plot(delta_arr/(2*np.pi),np.sum(np.abs([finalrhoarr[:,j,j] for j in range(8)]) )*np.ones(L)/L, label='Total')
#plt.plot(alpha_arr,np.sum(np.abs([finalrhoarr[:,j,j] for j in range(8)]) )*np.ones(L)/L, label='Total')
#plt.plot(beta_arr,np.sum(np.abs([finalrhoarr[:,j,j] for j in range(8)]) )*np.ones(L)/L, label='Total')
#plt.plot(s_arr,np.sum(np.abs([finalrhoarr[:,j,j] for j in range(8)]) )*np.ones(L)/L, label='Total')


plt.xlabel(r'$\delta$ (MHz)')
#plt.xlabel(r'$\alpha$ (rad)')
#plt.xlabel(r'$\beta$ (rad)')
#plt.xlabel('s')

plt.ylabel(r'Population of $\rho_{ss}$ ')
#plt.ylim([0, 0.17])
plt.legend()
plt.grid(visible=True)

#%% fitting 2lvl absorption profile to 4lvl absoprtion profile for level at index=ind

ind=4
a0=np.max(np.abs(finalrhoarr[:,ind,ind]))
g0=GYb/(2*np.pi)

def lorentz(x,a,s):
    #a=a0
    g=g0
    return a*0.5*s/(s+1)*1/(1+(2*x/(g*np.sqrt(1+s)))**2)


from scipy.optimize import curve_fit

plt.figure()

popt, pcov = curve_fit(lorentz,delta_arr/(2*np.pi),np.abs(finalrhoarr[:,ind,ind]),
                               p0=[0.04,0.01], bounds=([0,0.001],[0.5,20]))


afit,sfit=popt
plt.plot(delta_arr/(2*np.pi),np.abs(finalrhoarr[:,ind,ind]),label=labels[ind] )
plt.plot(delta_arr/(2*np.pi),lorentz(delta_arr/(2*np.pi),afit,sfit), label=r'fit [a,s]='+str(popt)) 
plt.title(r' Two lvl fit model: $a\frac{s}{2(1+s)}$ $\frac{1}{1+(\frac{2\delta}{(\Gamma\sqrt{1+s}})^2}$ '+' '+' Sim input s='+str(s))

print(popt)
plt.legend()
plt.xlabel(r'$\delta$ (MHz)')
plt.ylabel(r'Population')
plt.grid(visible=True)

#%%

finalrhoarr=[]
popt_arr=[]

L1=len(delta_arr)
L2=len(s_arr)
for i in range(L2):
    finalrhoarr_temp=[]
    for j in range(L1):
        ele=te_funcWithCGsigns_SS(delta_arr[j],delta_arr[j],clist,f1_arr[i],f1_arr[i]/3,alpha,beta)
        finalrhoarr_temp.append(ele)
    finalrhoarr_temp=np.array(finalrhoarr_temp)   
    finalrhoarr.append(finalrhoarr_temp)
    popt,pcov=curve_fit(lorentz,delta_arr/(2*np.pi),np.abs(finalrhoarr_temp[:,ind,ind]),
                                   p0=[0.04,0.01], bounds=([0,0.0001],[0.5,20]))
    popt_arr.append(popt)

finalrhoarr=np.array(finalrhoarr)   
popt_arr=np.array(popt_arr)
#%%
plt.figure()
plt.plot(s_arr,popt_arr[:,0])
plt.title('2lvl fit a vs input s')
plt.grid(visible=True)
#plt.xlim([0,[:,1])])


#rho_gg vs f arr
# finalrhoarrtemp=finalrhoarr.reshape((len(farr),8,8))
# finalrhoggarr=np.absolute(finalrhoarrtemp[:,0,0])        #extracting only ground state
# plt.figure(0)
# plt.clf()
# plt.plot(x,1-finalrhoggarr) #plotting net rhoee from detection including mixed states
# plt.plot(x,1-farr/f0)
# plt.xlabel(r'x($\mu$m)')
# plt.ylabel(r'$\rho_{ee}$ mF=-1,0,+1 ')

# qsave(finalrhoarr,"Ramseyscan_"+datetime.now().strftime("%b %d %Y %H:%M:%S"))

# sys.exit()
#########################################

#%%
#rho_gg vs detuning
'''
offsetlist=np.linspace(0,50,5)
times = np.linspace(0, 247 / (GYb), int(2475))  # time is in microseconds
rho2Darr=np.zeros((len(offsetlist),len(times)))

plt.figure()
for i in range(len(offsetlist)):
    result = te_detuning(offsetlist[i])
    for j in range(len(times)):
        rho2Darr[i,j]=np.absolute(result.states[j][0,0])
    print("{1:0.3f}us : rho_gg= {0:0.3f} for detuning= {2:0.2f} MHz\n".format(rho2Darr[i,-1],times[-1],offsetlist[i]))
    plt.plot(times,rho2Darr[i,:],'--',label=r"$\Delta$={0:0.2f}MHz".format(offsetlist[i]))

plt.xlabel('$\mu$s')
plt.ylabel(r'$\rho_{gg}$')
plt.title(r'$\rho_{gg}$ vs $\Delta_{detuning}$')
plt.grid()
plt.legend()
plt.show()

'''
# for standard runs

'''
def plot_p(tresult, index, p):

    # extract and plot the population labeled by lindex
    # Input:
    #     tresust: solution generated by mesolve function
    #     index: int, the matrix index that represents the selected state
    #     p: bool, determines whether a solution should be plotted
    # Return:
    #     rhos, a numpy array of selected state population
 
    rhos = np.array([])
    for i in range(len(tresult.states)):
        rhos = np.append(rhos, np.absolute(result.states[i][index, index]))
    if p == 1:
        plt.plot(times, rhos, '--', label=labels[index])
        # if index==0:
        #     np.savetxt('localOP_te_20us_f_01_data.csv', np.array([times,rhos]).transpose(), delimiter=',')
        print(r'rho_gg at {0:.2f}us = {1:.4f}'.format(times[-1],rhos[-1]))
    return rhos


plt.figure(0)
rhotot = np.zeros(len(result.states))  # total population, check if it equals 1
etot = np.zeros(len(result.states))  # total population of excited state
plist = np.zeros(8)
plist[[0, 1, 2, 3, 4, 5 ]] = 1  # change the index to decide whether a solution should be plotted
#plist[[0, ]] = 1
#nrho = plot_p(result, 0, plist[0]) # temp way to just print rho_gg

plt.figure(0)
for i in range(8):
    nrho = plot_p(result, i, plist[i])
    rhotot = rhotot + nrho
    if i >= 4:
        etot = etot + nrho
plt.plot(times, rhotot, label='total population')
plt.plot(times, etot, '--', label='excited states')
#plt.plot(times, 1-np.exp(-times/8.23), label='1/e data')
plt.title('state population')
plt.xlabel(r'$t [\mu s$]')
plt.ylabel(r'$\rho$')
plt.grid()
plt.legend()
plt.show()
'''
