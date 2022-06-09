# -*- coding: utf-8 -*-
"""
This example intends to show how to import modules and read the documentation of functions in this package
"""
#%%
#import the spin module from ion_chain.operator subpackage and call it "spin" for further usage
import ion_chain.operator.spin as spin
#%%
#print the discription of the package and list the name of all functions
print(spin.__doc__)
#%%
#read the summary for all functions in spin
spin.summary()
#%%
#print the documentation for function sx
help(spin.sx)
