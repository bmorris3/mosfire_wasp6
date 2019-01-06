# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 13:02:00 2014

@author: bmmorris
"""

from glob import glob
import numpy as np
from matplotlib import pyplot as plt

models = glob('../../../../../uw/classes/astr521/hw1/dat/uk??v.dat')
#models = glob('../../../../../uw/classes/astr521/hw1/dat/uk??iv.dat')

def getspectrum(filename):
    input_file = open(filename,'r').read().splitlines()[3:]
    #Each of the 131 flux.dat files contains 3 header lines with a # in the first
    #column, and several columns of data. The first column is wavelength in
    #Angstrom, always 1150 - 10620A in steps of 5A. The second column is
    #F(lambda) for the relevant spectrum, normalised to unity at 5556A. The third
    #column is the rms flux error on the same scale. Columns 4-N (Nmax=10)
    #contain the components used, in no particular order.
    wavelength = []
    flux = []
    for line in input_file:
        splitline = line.split()
        wavelength.append(float(splitline[0]))
        flux.append(float(splitline[1]))
    return [np.array(wavelength)*1e-4,np.array(flux)/np.max(np.array(flux))]

fig, ax = plt.subplots(figsize=(14,14))
for model in models:
    w, f = getspectrum(model)
    ax.plot(w, f, label=model.split('/')[-1])
ax.legend()
plt.show()