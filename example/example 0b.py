# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 21:35:22 2023

@author: zhumj
"""

from  Qsim.ion_chain.ion_system import *
#%% setup a 3 ion system  
ion1 = ions()
ion1.N = 3
ion1.fx = 4 #set radial trapping freq MHz
ion1.fz = 2 #set axial trapping freq MHz
#%%
print('radial elastic tensor B')
print(ion1.Tmatrix())
print('axial elastic tensor A')
print(ion1.Amatrix())
print('sqrt of eigenvalues for radial elastic tensor B ')
print(ion1.Transfreq())
print('sqrt of eigenvalues for axial elastic tensor A ')
print(ion1.Axialfreq())
print('radial eigenfrequency in MHz: com, tilt, rock ')
print(ion1.Transfreq()*ion1.fz)
print('corresponding radial eigenmodes')
print(ion1.radial_mode[0],'\n',ion1.radial_mode[1],'\n',ion1.radial_mode[2])
print('axial eigenfrequency in MHz: com, tilt, rock')
print(ion1.Axialfreq()*ion1.fz)
print('corresponding axial eigenmodes')
print(ion1.axial_mode[0],'\n',ion1.axial_mode[1],'\n',ion1.axial_mode[2])
