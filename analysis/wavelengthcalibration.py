# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 08:11:31 2014

@author: bmmorris
"""

import numpy as np
from numpy import linalg as LA
import pyfits
import math
from matplotlib import pyplot as plt
from scipy import ndimage, optimize
from time import sleep
from re import split
wavelengthListPath = 'wavelengths.list'	
def trackLine(axisA,plots=False):
	'''Find spectroscopic line centroids using the tracking technique written for tracking stars
	   with differential photometry'''
	axisADeriv = np.diff(axisA)	 ## Find the differences between each pixel intensity and
	derivMinAind = np.where(axisADeriv == min(axisADeriv[len(axisADeriv)/2:len(axisADeriv)]))[0][0] ## Minimum in the derivative
	derivMaxAind = np.where(axisADeriv == max(axisADeriv[0:len(axisADeriv)/2]))[0][0] ## Maximum in the derivative

	indMax = np.argmax(axisADeriv)
	fitPlots = 'off'
	def quadraticFit(derivative,ext):
		rangeOfFit = 1
		#lenDer = len(derivative)/1
		if ext == "max":
			#indExtrema = np.argmax(derivative[:lenDer])
			indExtrema = np.argmax(derivative[:])
		if ext == "min":
			#indExtrema = np.argmin(derivative[lenDer:])+lenDer
			indExtrema = np.argmin(derivative[:])
			
		fitPart = derivative[indExtrema-rangeOfFit:indExtrema+rangeOfFit+1]
		if len(fitPart) == 3:
			stackPolynomials = [0,0,0]
			for i in range(0,len(fitPart)):
				vector = [i**2,i,1]
				stackPolynomials = np.vstack([stackPolynomials,vector])
			xMatrix = stackPolynomials[1:,:]
			from numpy import linalg as LA
			xMatrixInv = LA.inv(xMatrix)

			estimatedCoeffs = np.dot(xMatrixInv,fitPart)

			a_fit = estimatedCoeffs[0]#-0.05
			b_fit = estimatedCoeffs[1]#0.5
			c_fit = estimatedCoeffs[2]#0.1
			d_fit = -b_fit/(2.*a_fit)
			extremum = d_fit+indExtrema-rangeOfFit
		else: 
			extremum = indExtrema
		return extremum
	
	extremumA = quadraticFit(axisADeriv,ext="max")
	extremumB = quadraticFit(axisADeriv,ext="min")
	centroid = (extremumA+extremumB)/2.
	
	if plots:
		plt.plot(axisADeriv,'k.-')
		plt.axvline(ymin=0,ymax=1,x=extremumA,color='b')
		plt.axvline(ymin=0,ymax=1,x=extremumB,color='b')
		plt.axvline(ymin=0,ymax=1,x=centroid,color='r',linewidth=2)
		plt.show()
	return centroid



raw = open(wavelengthListPath,'r').read().splitlines() 

wavelengths_linelist = []
intensities_linelist = []
for line in raw:				## Load the lab-produced spectrum for the Ne lamp
	if len(split(' ',line)[3]) == 8: 
		wavelengths_linelist.append(float(split(' ',line)[3]))
		intensity = split(' ',line)[4]
		string = ''.join(c for c in intensity if c.isdigit())
		intensities_linelist.append(float(string))
	else: 
		wavelengths_linelist.append(float(split(' ',line)[2]))
		intensity = split(' ',line)[3]
		string = ''.join(c for c in intensity if c.isdigit())
		intensities_linelist.append(float(string))

wavelengths_linelist = np.array(wavelengths_linelist)
intensities_linelist = np.array(intensities_linelist)

wavelengths = np.linspace(np.min(wavelengths_linelist), np.max(wavelengths_linelist), 1e3)
intensities = np.zeros_like(wavelengths)

# Merge the linelist intensities with the zeros list
wavelengths = np.concatenate((wavelengths, wavelengths_linelist))
intensities = np.concatenate((intensities, intensities_linelist))[np.argsort(wavelengths)]
wavelengths = np.sort(wavelengths)

kbandrange = [19200, 24000] # angstroms
inkband = (wavelengths > kbandrange[0])*(wavelengths < kbandrange[1])
wavelengths = wavelengths[inkband]
intensities = intensities[inkband]


from glob import glob
paths = ['/local/tmp/mosfire/2014sep18_analysis/m140918_000'+str(i)+'shifted.fits' for i in range(4,7)]
lampimage = np.sum([pyfits.getdata(imagepath) for imagepath in paths],axis=0)
lampspec = np.sum(lampimage, axis=0)
lampspec *= np.max(intensities)/np.max(lampspec)
columnrange = np.arange(len(lampspec))
brightestline_bounds = [2055, 2075]
nextbrightestline_bounds = [1545, 1565]

brightestline_position = trackLine(lampspec[brightestline_bounds[0]:brightestline_bounds[1]]) +\
                                   brightestline_bounds[0]
nextbrightestline_position = trackLine(lampspec[nextbrightestline_bounds[0]:nextbrightestline_bounds[1]]) +\
                                       nextbrightestline_bounds[0]
# From http://www2.keck.hawaii.edu/inst/mosfire/data/MosfireArcs/neon_K.id.pdf
brightestline_wavelength = 23643
nextbrightestline_wavelength = 22537

def fitfunc(p, cols):
    return p[0] + p[1]*cols

def errfunc(p, cols):
    '''
    Swap order here of the lines to flip the wavelength solution
    '''
#    return fitfunc(p, cols) - np.array([nextbrightestline_wavelength, brightestline_wavelength])
    return fitfunc(p, cols) - np.array([brightestline_wavelength, nextbrightestline_wavelength])

initP = [20000, -1]
bestp = optimize.leastsq(errfunc, initP, args=(np.array([brightestline_position, nextbrightestline_position])))[0]
#plt.axvline(brightestline_position)
#plt.axvline(nextbrightestline_position)
wavelengthsolution = fitfunc(bestp, columnrange) # Save wavelength solution in microns
np.save('notebooks/outputs/wavelengthsoln.npy', wavelengthsolution*1e-4)

plt.axvline(brightestline_wavelength, color='r', ls='--')
plt.axvline(nextbrightestline_wavelength, color='r', ls='--')
plt.plot(wavelengths, intensities, label='lab')
plt.plot(wavelengthsolution, lampspec, label='lamp')
plt.legend()
plt.show()