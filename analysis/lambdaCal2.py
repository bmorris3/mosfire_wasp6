'''
Created on Feb 4, 2013

Compute transformations from column number to wavelength bin for MOSFIRE spectral data
using Ne calibration lamps.

This script does not take into account the slanting of the spectral lines in the data
from MOSFIRE, which slightly decreases the spectral resolution of the data when it is handled
as we have handled it here. For the purposes of the November HAT-P-32 data, we decided that 
this wasn't worth fussing about.

URL for Keck website to find line lists for Neon and Argon lamps:
 http://www2.keck.hawaii.edu/inst/mosfire/wavelength_calibration.html

@author: bmmorris
'''
import numpy as np
from numpy import linalg as LA
import pyfits
import math
from matplotlib import pyplot as plt
from scipy import ndimage, optimize
from time import sleep
from re import split

'''INPUTS & SETTINGS'''
wavelengthListPath = 'wavelengths.list'	## Path to line list for the relevant lamp (URL for MOSFIRE lamp links is above)
minimumWavelengthInBand = 1.5 			## Set the wavelength limits of the band (eg: K-band min = 1.5, K-band max = 2.45)
maximumWavelengthInBand  = 2.45
path = '/local/tmp/mosfire/2014sep18_analysis/m140918_00' ## Path to lamp images
minLampIndex = 4 ## lamp indices to cycle through (by concatenating these variables as strings to the "path" string above)
maxLampIndex = 6
'''Input the pixel ranges for two specific Neon lines from the lamp to track in the "MaxBounds" lists,
Also put in the rowBounds -- the bounding pixels rows encompassing each slit that we will analyze.
'''
rowBounds = [[0,2020]] 	## Rows enclosing each slit
firstMaxBounds = [[261,273]] ## Columns enclosing the brightest line

secondWavelengthMin = 2.16	## Wavelength ange on which to look for the second peak in the lab spectrum
secondWavelengthMax = 2.18
secondMaxBounds = [[1550,1560]] ## Columns enclosing the second brightest line
smoothConst=1.0 ## Constant that adjusts degree of gaussian smooth on the lamp images
plots = True

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


def interpolate(referenceX,x,y):
	'''Create a grid with the sampling of referenceX that extends to surround x, then median bin x onto that grid
	   Assumes referenceX and x are sorted from least to greatest'''
	#print 'interp'
	refSpacing = (referenceX[1] - referenceX[0])
	testGrid = np.array(referenceX)
	while testGrid[0] > x[0]:	## Extend grid towards min(x)
		#print 'extend towards 0'
		testGrid = np.concatenate([np.arange(testGrid[0]-(10.*refSpacing),testGrid[0],refSpacing),testGrid])
	while testGrid[-1] < x[-1]:	## Extend grid towards max(x)
		#print 'extend away from 0'
		testGrid = np.concatenate([testGrid,np.arange(testGrid[-1]+refSpacing,testGrid[-1]*1.5,refSpacing)])
	#print 'final testgrid:',testGrid
	interpedXs = np.zeros_like(testGrid)
	interpedYs = np.zeros_like(testGrid)
	for i in range(0,len(testGrid)):
		#print testGrid[i]
		acceptXs = (testGrid[i] + refSpacing/2. >= x)*(testGrid[i] - refSpacing/2. < x)
		if any(acceptXs) == False:
			interpedYs[i] = 0
			interpedXs[i] = testGrid[i]
		else:
			interpedYs[i] = np.mean(y[acceptXs])
			interpedXs[i] = testGrid[i]#np.mean(x[acceptXs])	
	nonZero =  (interpedXs >= referenceX[0])*(interpedXs <= referenceX[-1]) # (interpedYs != 0)*
	return interpedXs[nonZero].astype(np.float64),interpedYs[nonZero].astype(np.float64)


raw = open(wavelengthListPath,'r').read().splitlines() 

wavelengths = []
intensities = []
for line in raw:				## Load the lab-produced spectrum for the Ne lamp
	if len(split(' ',line)[3]) == 8: 
		wavelengths.append(float(split(' ',line)[3]))
		intensity = split(' ',line)[4]
		string = ''.join(c for c in intensity if c.isdigit())
		#print string
		#print split(' ',line)[3]
		intensities.append(float(string))
	else: 
		wavelengths.append(float(split(' ',line)[2]))
		intensity = split(' ',line)[3]
		string = ''.join(c for c in intensity if c.isdigit())
		#print string
		#print split(' ',line)[3]
		intensities.append(float(string))

wavelengths = np.array(wavelengths)*1e-4		## convert to angstroms
intensities = np.array(intensities)
wavelengthsFine = np.arange(np.min(wavelengths),np.max(wavelengths),1e-4)	## Create Kernel before interpolation

longWavelengths = np.concatenate([wavelengths,wavelengthsFine])
longIntensities = np.concatenate([intensities,np.zeros_like(wavelengthsFine)])

longIntensities = longIntensities[np.argsort(longWavelengths)]	## put zeros in at even intervals
longWavelengths = longWavelengths[np.argsort(longWavelengths)]	## so that lab spectrum isn't just peaks

wavelengths = longWavelengths
intensities = longIntensities

intensities = intensities[(wavelengths > minimumWavelengthInBand)*(wavelengths < maximumWavelengthInBand)] 
wavelengths = wavelengths[(wavelengths > minimumWavelengthInBand)*(wavelengths < maximumWavelengthInBand)]	## Restrict view to wavelengths we sampled
intensities /= np.max(intensities)	## Normalize intensities (because they start out in strange units)


'''Load lamp spectrum images, stack them up for best signal:noise (not that it really matters)'''
lampImage = np.zeros_like(pyfits.getdata(path+str(4).zfill(2)+'shifted.fits'))
for i in range(minLampIndex,maxLampIndex):
	lampImage +=  pyfits.getdata(path+str(i).zfill(2)+'shifted.fits')#np.load(path+str(i)+'.npy')


'''
Load "firstMaxBounds" and "secondMaxBounds", each of which are bounding columns on either side of
prominent lamp peak features which we will use to match the lab spectrum to the lamp spectrum. 
Take the center of the line in the lamp spectra in both bounds (first and second), and solve for
a linear transformation (multiplication by an amplitude, addition of a constant) that can be applied
to the column numbers to transform them into wavelength bins.
'''
lampImage = np.fliplr(lampImage)
scalingParam = []
offsetParam = []
nrows, ncolumns = lampImage.shape
for i in range(0,len(rowBounds)):
	cols = np.arange(0,ncolumns,dtype=float)
	colSums = np.sum(lampImage[rowBounds[i][0]:rowBounds[i][1],:], axis=0)
	colSums /= np.max(colSums) 	## Normalize

	cols1 = cols[firstMaxBounds[i][0]:firstMaxBounds[i][1]]#*0.35/1000 + 1.7
	cols2 = cols[secondMaxBounds[i][0]:secondMaxBounds[i][1]]#*0.35/1000 + 1.7
	colSums1 = np.sum(lampImage[rowBounds[i][0]:rowBounds[i][1],firstMaxBounds[i][0]:firstMaxBounds[i][1]], axis=0)
	colSums2 = np.sum(lampImage[rowBounds[i][0]:rowBounds[i][1],secondMaxBounds[i][0]:secondMaxBounds[i][1]], axis=0)

	colSums1 = ndimage.gaussian_filter1d(colSums1,sigma=smoothConst,order=0)	## Smooth the lamp spectra
	colSums2 = ndimage.gaussian_filter1d(colSums2,sigma=smoothConst,order=0)	## before finding line centroids

	firstMax = trackLine(colSums1) + cols1[0]	## Find line centroids 
	secondMax = trackLine(colSums2) + cols2[0]

	plt.plot(cols1,colSums1,'k.-')
	plt.plot(cols2,colSums2,'k.-')
	plt.axvline(ymin=0,ymax=1,x=firstMax)
	plt.axvline(ymin=0,ymax=1,x=secondMax)
	plt.show()

	firstWavelength = wavelengths[intensities == np.max(intensities)][0]## First lab spectrum line to track is the brightest line
	secondWavelengthRange = (wavelengths > secondWavelengthMin)*(wavelengths < secondWavelengthMax)	## Indicate lab spectrum range of second line to track  
	secondWavelength = wavelengths[secondWavelengthRange][intensities[secondWavelengthRange] == np.max(intensities[secondWavelengthRange])][0]

	scalingFactor = (firstWavelength-secondWavelength)/(firstMax-secondMax)
	cols *= scalingFactor

	newFirstMax = firstMax*(firstWavelength-secondWavelength)/(firstMax-secondMax)
	offset = firstWavelength-newFirstMax
	cols += offset
	
	finalFirstMax = cols[(np.max(colSums) == colSums)][0]
	print scalingFactor, firstMax - finalFirstMax
	scalingParam.append(scalingFactor)
	offsetParam.append(offset)
	if plots:
		a,=plt.plot(wavelengths,intensities*2,'k',linewidth=2)
		b,=plt.plot(cols,colSums,'r-',linewidth=2)
		plt.title('Lamp spectra-matching')
		plt.ylabel('Normalized intensity')
		plt.xlabel('Wavelength ($\AA$)')
		plt.legend((a,b),('Lab spectrum','Lamp spectrum'))
		plt.show()

np.save('wavelengthCalibrationParams.npy',np.vstack([scalingParam,offsetParam]))
