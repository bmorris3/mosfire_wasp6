'''
Created on Jan 22, 2013

Classes and methods for photometry with MOSFIRE.

@author: bmmorris
'''
import numpy as np
from numpy import linalg as LA
import pyfits
import math
from matplotlib import pyplot as plt
from scipy import ndimage, optimize
from time import sleep
## Set some photometry settings
smoothConst = 2
gain = 2.15
class fluxObject(object):
	def __init__(self,stars,paths,Nchannels):
		#self.exposureNumber = -1
		self.pathList = paths
		self.fluxesDict = {}#np.zeros([len(stars),len(paths)])
		self.errorsDict = {}#np.zeros([len(stars),len(paths)])
		self.poissonRawCountsDict = {}
		self.psfDict = {}
		self.tagDict = {}
		for i in range(0,len(stars)):
			#self.fluxesDict[str(stars[i]['number'])] = np.zeros_like(paths)
			#self.errorsDict[str(stars[i]['number'])] = np.zeros_like(paths)
			self.fluxesDict[str(stars[i]['number'])] = np.zeros([len(paths),Nchannels])
			self.errorsDict[str(stars[i]['number'])] = np.zeros([len(paths),Nchannels])
			self.psfDict[str(stars[i]['number'])] = np.zeros_like(paths)
			self.poissonRawCountsDict[str(stars[i]['number'])] = np.zeros_like(paths)
			self.tagDict[str(stars[i]['number'])] = stars[i]['tag']
	def rawFluxInput(self,starNumber,exposureNumber,fluxInput,errorInput,psfInput):
		#self.fluxesDict[str(starNumber)][exposureNumber] = np.float(fluxInput)
		#self.errorsDict[str(starNumber)][exposureNumber] = np.float(errorInput)
		self.fluxesDict[str(starNumber)][exposureNumber,:] = fluxInput
		self.errorsDict[str(starNumber)][exposureNumber,:] = errorInput
		#self.poissonRawCountsDict[str(starNumber)][exposureNumber] = np.float(poissonRawCounts)
		self.psfDict[str(starNumber)][exposureNumber] = np.float(psfInput)
#	def wholeRawFluxInput(self,starNumber,fluxInput,errorInput):
#		self.fluxesDict[str(starNumber)] = fluxInput
#		self.errorsDict[str(starNumber)] = errorInput
	def rawFlux(self):
		return self.fluxesDict
	def rawError(self):
		return self.errorsDict
	def rawCounts(self):
		return self.poissonRawCountsDict
	def psf(self):
		return self.psfDict
	def tags(self):
		return self.tagDict
#	def noiseToSignal(self,lightCurve):
#		'''Calculate the noise:signal ratio, which gives the minimum achievable rms scatter on the
#		   out-of-transit portions of the light curve'''
#		## gain = 2.15
#		sigmaComparisonStarsSquared = np.zeros_like(self.pathList,dtype=float)
#		signalComparisonStars = np.zeros_like(self.pathList,dtype=float)
#		for star in self.tags():
#			if self.tags()[star] == 'comparison':
#				sigmaComparisonStarsSquared += np.array(self.rawError()[star],dtype=float)**2
#				signalComparisonStars += np.array(self.rawCounts()[star],dtype=float)
#			elif self.tags()[star] == 'target':
#				sigmaTargetStar = np.array(self.rawError()[star],dtype=float)
#				signalTargetStar = np.array(self.rawCounts()[star],dtype=float)
#		sigmaComparisonStars = np.sqrt(sigmaComparisonStarsSquared)
#		N = np.sqrt((sigmaComparisonStars/signalComparisonStars)**2 + (sigmaTargetStar/signalTargetStar)**2)
#		return N/lightCurve
#		

#	def noiseToSignal(self,lightCurve):
#		'''Calculate the noise:signal ratio, which gives the minimum achievable rms scatter on the
#		   out-of-transit portions of the light curve'''
#		## gain = 2.15
#		sigmaComparisonStarsSquared = np.zeros_like(self.pathList,dtype=float)
#		signalComparisonStars = np.zeros_like(self.pathList,dtype=float)
#		for star in self.tags():
#			if self.tags()[star] == 'comparison':
#				sigmaComparisonStarsSquared += np.array(self.rawError()[star],dtype=float)**2
#				signalComparisonStars += np.array(self.rawCounts()[star],dtype=float)
#			elif self.tags()[star] == 'target':
#				sigmaTargetStar = np.array(self.rawError()[star],dtype=float)
#				signalTargetStar = np.array(self.rawCounts()[star],dtype=float)
#		sigmaComparisonStars = np.sqrt(sigmaComparisonStarsSquared)
#		N = np.sqrt((sigmaComparisonStars/signalComparisonStars)**2 + (sigmaTargetStar/signalTargetStar)**2)
#		return N/lightCurve


class photometryObject(object):
	'''Class for managing photometry of MOSEFIRE data'''
	
	def __init__(self,pathInput,pathSumInput):
		'''Initialize a phot object. pathInput is the list of paths to npy pickles to use for photometry.'''
		self.path = pathInput
		self.pathSum = pathSumInput
		self.starNumber = -1
		self.starDictionaryList = []
		#self.masterFlat = flatField
	def paths(self):
		'''Returns paths to the input npy pickle files used in the current phot object'''
		return self.path
	def pathsSum(self):
		'''Returns paths to the summed (rather than subtracted) nods used in the current phot object'''
		return self.pathSum

	def addStars(self,starTag,pxlBounds,psfMasks,wavelengthCalibParams,plots=False):
		'''Create a dictionary of stars with their pixel regions'''
		self.starNumber += 1
		starDict = {} 			## Create a dictionary for this star
		starDict['tag'] = starTag
		starDict['pxlBounds'] = pxlBounds
		starDict['psfMasks'] = psfMasks
		starDict['number'] = self.starNumber
		starDict['wavelengthParams'] = wavelengthCalibParams
		#starDict['bgBounds'] = bgBounds
		self.starDictionaryList.append(starDict)
	def starDictionaries(self):
		return self.starDictionaryList

	def colBounds(self,starNumber,lowerBound,upperBound):
		self.starDictionaries()[starNumber]['colBounds'] = [lowerBound,upperBound]

	def trackStar(self,image,pxlBounds,rootInd,plots=False,returnCentroidsOnly=False):
		'''Track the centroids of the left and right (A and B, positive and negative) images of the star located between
		   rows lowerPxlBound and upperPxlBound. Returns the lists of left and right centroid positions'''
		#if plots==True: import matplotlib; matplotlib.interactive(True);fig = plt.figure(figsize=(10,10))	
		[lowerPxlBound,upperPxlBound] = pxlBounds
		rowPxls = np.arange(lowerPxlBound,upperPxlBound)
		## Numpy pickle dictionary keys (from organizeImages/AminusB2.py): 
		##       image=nodSubtractedImage,time=expTime,path=currentPath,rootInd=rawObj.path2ind(currentPath),nod=nodName)
		sumRows = np.sum(image[lowerPxlBound:upperPxlBound,:],axis=1) ## Sum up cropped image along rows
		#rootInd =  np.load(path)['rootInd']
		rawMedianOfsumRows = np.median(sumRows) 	## Subtract sumRows by it's median before inversion
		sumRowsMedianSubtracted = sumRows - rawMedianOfsumRows
		
		## The sum of the rows shows the positive and negative image of the star.
		## Measure halfway between max row and min row, invert the min row's half of the image
	
		leftHalfIndices = (0.5*(rowPxls[(sumRows == np.min(sumRows))]+rowPxls[(sumRows == np.max(sumRows))]) < rowPxls)
		rightHalfIndices = (0.5*(rowPxls[(sumRows == np.min(sumRows))]+rowPxls[(sumRows == np.max(sumRows))]) >= rowPxls)
		if np.mean(sumRowsMedianSubtracted[leftHalfIndices]) > np.mean(sumRowsMedianSubtracted[rightHalfIndices]):
			positiveHalfIndices = leftHalfIndices
			negativeHalfIndices = rightHalfIndices
			negativeOffset = len(positiveHalfIndices)
			positiveOffset = 0
		else:
			positiveHalfIndices = rightHalfIndices
			negativeHalfIndices = leftHalfIndices
			positiveOffset = len(leftHalfIndices)
			negativeOffset = 0
		#sumRowsInverted = np.copy(sumRowsMedianSubtracted)
		sumRowsInverted = np.copy(sumRows)
		sumRowsInverted[negativeHalfIndices] *= -1 
		sumRowsInverted += rawMedianOfsumRows 		## Add back the median background of the raw image		

		## Use derivative of sumRows to locate stellar centroid
		sumRowsInvertedSmoothed = ndimage.gaussian_filter(sumRowsInverted,sigma=smoothConst,order=0)	## Smooth out sumRows
		derivativeOfsumRows = np.diff(sumRowsInvertedSmoothed)											## ...take derivative

		## Do rough matrix-algebra quadratic fit to the extrema of the derivative
		positiveHalfMax,positiveHalfMaxInd = quadFit(rowPxls,derivativeOfsumRows,positiveHalfIndices[1:],"max",positiveOffset)
		positiveHalfMin,positiveHalfMinInd = quadFit(rowPxls,derivativeOfsumRows,positiveHalfIndices[1:],"min",positiveOffset)
		negativeHalfMax,negativeHalfMaxInd = quadFit(rowPxls,derivativeOfsumRows,negativeHalfIndices[1:],"max",negativeOffset)
		negativeHalfMin,negativeHalfMinInd = quadFit(rowPxls,derivativeOfsumRows,negativeHalfIndices[1:],"min",negativeOffset)
		## Find positive and negative stellar centroids, assign them to the left/right arrays
		positiveCentroid = 0.5*(positiveHalfMax+positiveHalfMin)
		negativeCentroid = 0.5*(negativeHalfMax+negativeHalfMin)
		positiveInd = 0.5*(positiveHalfMaxInd+positiveHalfMinInd)
		negativeInd = 0.5*(negativeHalfMaxInd+negativeHalfMinInd)
		if np.mean(sumRowsMedianSubtracted[leftHalfIndices]) < np.mean(sumRowsMedianSubtracted[rightHalfIndices]):
			leftCentroid = positiveCentroid
			rightCentroid = negativeCentroid
			centroidInd = positiveInd
		else:
			leftCentroid = negativeCentroid
			rightCentroid = positiveCentroid
			centroidInd = negativeInd
		if plots:	## Generate plots
			plt.clf()
			plt.axvline(ymin=0,ymax=1,x=leftCentroid,color='y',linewidth=2)
			plt.axvline(ymin=0,ymax=1,x=rightCentroid,color='b',linewidth=2)
			smooth,=plt.plot(rowPxls,sumRowsInvertedSmoothed,'r:',linewidth=3)
			deriv,=plt.plot(rowPxls[1:],derivativeOfsumRows,'b',linewidth=2)
			raw,=plt.plot(rowPxls,sumRows,'g',linewidth=2)
			plt.legend((raw,smooth,deriv),('Raw Counts','Smoothed, Inverted','Derivative'))
			plt.xlabel('Pixel Row')
			plt.ylabel('Counts')
			plt.draw()
		if returnCentroidsOnly==False: return rowPxls, sumRows, leftCentroid, rightCentroid, rootInd, centroidInd
		else: return leftCentroid, rightCentroid

	def trackStarGaussian(self,image,pxlBounds,rootInd,plots=False,returnCentroidsOnly=False):
		'''Track the centroids of the left and right (A and B, positive and negative) images of the star located between
		   rows lowerPxlBound and upperPxlBound. Returns the lists of left and right centroid positions'''
		#if plots==True: import matplotlib; matplotlib.interactive(True);fig = plt.figure(figsize=(10,10))	
		[lowerPxlBound,upperPxlBound] = pxlBounds
		rowPxls = np.arange(lowerPxlBound,upperPxlBound)
		## Numpy pickle dictionary keys (from organizeImages/AminusB2.py): 
		##       image=nodSubtractedImage,time=expTime,path=currentPath,rootInd=rawObj.path2ind(currentPath),nod=nodName)
		sumRows = np.sum(image[lowerPxlBound:upperPxlBound,:],axis=1) ## Sum up cropped image along rows
		#rootInd =  np.load(path)['rootInd']
		rawMedianOfsumRows = np.median(sumRows) 	## Subtract sumRows by it's median before inversion
		sumRowsMedianSubtracted = sumRows - rawMedianOfsumRows
		
		## The sum of the rows shows the positive and negative image of the star.
		## Measure halfway between max row and min row, invert the min row's half of the image

#		leftHalfIndices = (0.5*(rowPxls[(sumRows == np.min(sumRows))]+rowPxls[(sumRows == np.max(sumRows))]) < rowPxls)
#		rightHalfIndices = (0.5*(rowPxls[(sumRows == np.min(sumRows))]+rowPxls[(sumRows == np.max(sumRows))]) >= rowPxls)
		leftHalfIndices = (0.5*(rowPxls[(sumRows == np.min(sumRows))]+rowPxls[(sumRows == np.max(sumRows))]) > rowPxls)
		rightHalfIndices = (0.5*(rowPxls[(sumRows == np.min(sumRows))]+rowPxls[(sumRows == np.max(sumRows))]) <= rowPxls)
		if np.mean(sumRowsMedianSubtracted[leftHalfIndices]) > np.mean(sumRowsMedianSubtracted[rightHalfIndices]):
			positiveHalfIndices = leftHalfIndices
			negativeHalfIndices = rightHalfIndices
			negativeOffset = len(positiveHalfIndices)
			positiveOffset = 0
		else:
			positiveHalfIndices = rightHalfIndices
			negativeHalfIndices = leftHalfIndices
			positiveOffset = len(leftHalfIndices)
			negativeOffset = 0
		#sumRowsInverted = np.copy(sumRowsMedianSubtracted)
		sumRowsInverted = np.copy(sumRows)
		sumRowsInverted[negativeHalfIndices] *= -1 
		sumRowsInverted += rawMedianOfsumRows 		## Add back the median background of the raw image		

		## Use derivative of sumRows to locate stellar centroid
		sumRowsInvertedSmoothed = ndimage.gaussian_filter(sumRowsInverted,sigma=smoothConst,order=0)	## Smooth out sumRows
		derivativeOfsumRows = np.diff(sumRowsInvertedSmoothed)											## ...take derivative

		## Do rough matrix-algebra quadratic fit to the extrema of the derivative
		x = rowPxls[leftHalfIndices]
		y = sumRowsInverted[leftHalfIndices]-np.min(sumRowsInverted[leftHalfIndices])
		#print fitgaussian(y)#,initialParameters=(max(y),x[y == max(y)],5.0))
		fineSample = np.arange(0,len(y),1e-2)
		#bestFit = gaussian(*fitgaussian(y))(np.arange(0,len(y)))
		bestFit = gaussian(*fitgaussian(y))(fineSample)
		leftCentroid = fitgaussian(y)[1] + x[0]
		if plots == True:
			plt.clf()
			plt.plot(fineSample+x[0],bestFit,'r')
			plt.plot(x,y,'bo')
			plt.draw()
		centroidInd = fitgaussian(y)[1]

		x = rowPxls[rightHalfIndices]
		y = sumRowsInverted[rightHalfIndices]-np.min(sumRowsInverted[rightHalfIndices])
		#print fitgaussian(y)
		fineSample = np.arange(0,len(y),1e-2)
		#bestFit = gaussian(*fitgaussian(y))(np.arange(0,len(y)))
		bestFit = gaussian(*fitgaussian(y))(fineSample)
		rightCentroid = fitgaussian(y)[1] + x[0]
		#print leftCentroid,rightCentroid
		plt.show()

		return rowPxls, sumRows, leftCentroid, rightCentroid, rootInd, centroidInd

	def phot(self,image,imageSum,pxlBounds,colBounds,leftCentroid,rightCentroid,psfMasks,plots=False):
		'''Do photometry with the results returned by self.trackStar()
		   Inputs: 	image 		- A-B nod subtracted image
		   			imageSum 	- A+B nod ADDED image '''
		## Sum  up all rows
		#image /= self.masterFlat
		#imageSum /= self.masterFlat
		rowPxls = np.arange(pxlBounds[0],pxlBounds[1])
		sumRows = np.sum(image[pxlBounds[0]:pxlBounds[1],colBounds[0]:colBounds[1]],axis=1)		
		psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
		midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
		leftHalfIndices = (midPoint > rowPxls)
		rightHalfIndices = (midPoint <= rowPxls)
	
		if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): set = [leftHalfIndices,float(leftCentroid)]
		else: set = [rightHalfIndices,float(rightCentroid)]
		indices = set[0]
		maskedOut = ((set[1]+psfMaskRight < rowPxls[indices]) + (set[1]-psfMaskLeft > rowPxls[indices]))
		sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))
		rowBit = rowPxls[indices][maskedOut]
		sumRowsBit = sumRows[indices][maskedOut]	## Cropped bit of "sumRows" for background

		## Subtract median of column intensities	
		sumColsBit = np.median(image[pxlBounds[0]:pxlBounds[1],colBounds[0]:colBounds[1]][indices,:][maskedOut],axis=0)
		imageColCorrected = image[pxlBounds[0]:pxlBounds[1],colBounds[0]:colBounds[1]] - np.meshgrid(sumColsBit,rowPxls)[0]

		skyLineCorrectedSum = imageSum[pxlBounds[0]:pxlBounds[1],colBounds[0]:colBounds[1]] + np.meshgrid(sumColsBit,rowPxls)[0]
		imageColCorrectedUncertainty = skyLineCorrectedSum

		## Rename previously defined variables with new column sky line corrected version
		sumRows = np.sum(imageColCorrected,axis=1)
		sumRowsBit = sumRows[indices][maskedOut]
		## calculate sigma spread of PSF in the positive half:
		psf = scipyCalcSigma(rowPxls[indices],sumRows[indices])
		
		rowBitFiltered, sumRowsBitFiltered = meanFilter(rowBit,sumRowsBit,4) ## Mean filter the background pixels
		counts,bestfit = removeLinearTrend(rowPxls[indices],sumRows[indices],rowBitFiltered,sumRowsBitFiltered) ## remove linear trend

		if plots:
			plt.clf()
			plt.plot(rowPxls[indices],counts,'b',linewidth=2)
			plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
			plt.plot(rowBit,sumRowsBit,'b.')
			plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
			plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
			plt.draw()		
			plt.show()
		rawFlux = np.sum(counts[sourceRange])
		rawError = np.sqrt(np.sum(imageColCorrectedUncertainty[indices][sourceRange]))
		
		return rawFlux,rawError,psf

	def photLite(self,image,imageSum,pxlBounds,leftCentroid,rightCentroid,psfMasks,plots=False,diagnostics=False):
		'''Do photometry with the results returned by self.trackStar()
		   Inputs: 	image 		- A-B nod subtracted image
		   			imageSum 	- A+B nod ADDED image '''
		## Sum up all rows
		rowPxls = np.arange(pxlBounds[0],pxlBounds[1])
		sumRows = np.sum(image,axis=1)

		psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
		midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
		leftHalfIndices = (midPoint > rowPxls)
		rightHalfIndices = (midPoint <= rowPxls)
		if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): set = [leftHalfIndices,float(leftCentroid)]
		else: set = [rightHalfIndices,float(rightCentroid)]
		indices = set[0]
		maskedOut = ((set[1]+psfMaskRight < rowPxls[indices]) + (set[1]-psfMaskLeft > rowPxls[indices]))
		sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))
		rowBit = rowPxls[indices][maskedOut]
		#sumRowsBit = sumRows[indices][maskedOut]	## Cropped bit of "sumRows" for background

		## Subtract median of column intensities	
		#sumColsBit = np.median(image[indices,:][maskedOut],axis=0)
		sumColsBit = np.mean(image[indices,:][maskedOut],axis=0)
		imageColCorrected = image - np.meshgrid(sumColsBit,rowPxls)[0]
				
		## To find the error, take the square root of the signal, then square their sums, or equivalently, the square root of the sum:
		skyLineCorrectedSum = imageSum #+ np.meshgrid(sumColsBit,rowPxls)[0]
		
		print 'imageSum',np.sum(imageSum[indices,:][sourceRange])/209.
		print np.shape(imageSum[indices][sourceRange])
		
		imageColCorrectedUncertainty = skyLineCorrectedSum

		## Rename previously defined variables with new column sky line corrected version
		sumRows = np.sum(imageColCorrected,axis=1)
		sumRowsBit = sumRows[indices][maskedOut]
		## calculate sigma spread of PSF in the positive half:
		psf = scipyCalcSigma(rowPxls[indices],sumRows[indices])
		
		rowBitFiltered, sumRowsBitFiltered = meanFilter(rowBit,sumRowsBit,4) ## Mean filter the background pixels
		counts,bestfit = removeLinearTrend(rowPxls[indices],sumRows[indices],rowBitFiltered,sumRowsBitFiltered) ## remove linear trend

#		import matplotlib.cm as cm
#		fig = plt.figure(figsize=(16,8))
#		ax1 = fig.add_subplot(411)
#		ax2 = fig.add_subplot(412)
#		ax3 = fig.add_subplot(413)
#		ax4 = fig.add_subplot(414)
#		a1 = ax1.imshow(image[indices,:],interpolation='nearest',cmap=cm.Greys_r)
#		a1.set_clim([-150,150])
#		ax1.set_title('Raw image')
#		a2 = ax2.imshow(image[maskedOut],interpolation='nearest',cmap=cm.Greys_r)
#		a2.set_clim([-150,150])
#		ax2.set_title('Background')
#		a3 = ax3.imshow(np.meshgrid(sumColsBit,rowPxls[indices])[0],interpolation='nearest',cmap=cm.Grecys_r)
#		a3.set_clim([-150,150])
#		ax3.set_title('Sky-lines for removal')
#		a4 = ax4.imshow(imageColCorrected[indices],interpolation='nearest',cmap=cm.Greys_r)
#		a4.set_clim([-150,150])
#		ax4.set_title('Sky-line corrected')
#		plt.savefig('/Users/bmmorris/Desktop/leftOverSubtraction.eps',bbox_inches='tight')
#		plt.show()

#		fig = plt.figure()
#		ax1 = fig.add_subplot(211)
#		ax2 = fig.add_subplot(212, sharex=ax1)
#		ax1.set_title('Stellar flux profile (binned in spatial direction)')
#		ax1.set_xlabel('Pixel')
#		ax1.set_ylabel('Raw Flux')
#		ax1.plot(rowPxls,sumRows,'k',linewidth=1)
#		ax1.plot(rowPxls[indices][sourceRange],sumRows[indices][sourceRange],'b',linewidth=2)
#		#plt.plot(rowBit,sumRowsBit,'b.')
#		ax1.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k',ls=':')
#		ax1.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k',ls=':')
#		
#		ax2.plot(rowPxls,sumRows,'k',linewidth=1)
#		ax2.plot(rowPxls[indices][sourceRange],sumRows[indices][sourceRange],'b',linewidth=2)
#		#plt.plot(rowBit,sumRowsBit,'b.')
#		ax2.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k',ls=':')
#		ax2.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k',ls=':')
#		ax2.set_title('ZOOMED Stellar flux profile')
#		ax2.set_xlabel('Pixel')
#		ax2.set_ylabel('Raw Flux')
#		plt.tight_layout(pad=2, w_pad=2, h_pad=2)
#		plt.draw()		
#		plt.show()

		if plots:
			plt.clf()
			plt.plot(rowPxls[indices],counts,'b',linewidth=2)
			plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
			plt.plot(rowBit,sumRowsBit,'b.')
			plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
			plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
			plt.draw()		
			plt.show()
		rawFlux = np.sum(counts[sourceRange])
		rawError = np.sqrt(np.sum(imageColCorrectedUncertainty[indices][sourceRange]))
		if diagnostics:
			finalBgStd = np.std(imageColCorrected[maskedOut]) ## After nod subtraction and column sky-line correction
			finalBgSqSum = np.sum(np.power(imageColCorrected[maskedOut],2))
			skyLineBgStd = np.std(sumColsBit)
			skyLineBgSqSum = np.sum(np.power(sumColsBit,2))
			sumColumns = np.sum(imageColCorrected[indices][sourceRange],axis=0)
			return rawFlux, rawError, psf, finalBgSqSum, finalBgStd, skyLineBgSqSum, skyLineBgStd, sumColumns
		else:
			return rawFlux,rawError,psf

	def photLite2(self,image,imageSum,star,leftCentroid,rightCentroid,plots=False):
		'''Do photometry with the results returned by self.trackStar()
		   Inputs: 	image 		- A-B nod subtracted image
		   			imageSum 	- A+B nod ADDED image '''
		## Sum up all rows
		image = image[star['pxlBounds'][0]:star['pxlBounds'][1],star['colBounds'][0]:star['colBounds'][1]]
		imageSum = imageSum[star['pxlBounds'][0]:star['pxlBounds'][1],star['colBounds'][0]:star['colBounds'][1]]
		pxlBounds = star['pxlBounds']
		psfMasks = star['psfMasks']
		rowPxls = np.arange(pxlBounds[0],pxlBounds[1])
		sumRows = np.sum(image,axis=1)

		psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
		midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
		leftHalfIndices = (midPoint > rowPxls)
		rightHalfIndices = (midPoint <= rowPxls)
	
		if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): set = [leftHalfIndices,float(leftCentroid)]
		else: set = [rightHalfIndices,float(rightCentroid)]
		indices = set[0]
		maskedOut = ((set[1]+psfMaskRight < rowPxls[indices]) + (set[1]-psfMaskLeft > rowPxls[indices]))
		sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))
		rowBit = rowPxls[indices][maskedOut]
		#sumRowsBit = sumRows[indices][maskedOut]	## Cropped bit of "sumRows" for background

		## Subtract median of column intensities	
		sumColsBit = np.median(image[indices,:][maskedOut],axis=0)
		imageColCorrected = image - np.meshgrid(sumColsBit,rowPxls)[0]
		## To find the error, take the square root of the signal, then square their sums, or equivalently, the square root of the sum:
		skyLineCorrectedSum = imageSum + np.meshgrid(sumColsBit,rowPxls)[0]
		imageColCorrectedUncertainty = skyLineCorrectedSum

		## Rename previously defined variables with new column sky line corrected version
		sumRows = np.sum(imageColCorrected,axis=1)
		sumRowsBit = sumRows[indices][maskedOut]
		## calculate sigma spread of PSF in the positive half:
		psf = scipyCalcSigma(rowPxls[indices],sumRows[indices])
		
		rowBitFiltered, sumRowsBitFiltered = meanFilter(rowBit,sumRowsBit,4) ## Mean filter the background pixels
		counts,bestfit = removeLinearTrend(rowPxls[indices],sumRows[indices],rowBitFiltered,sumRowsBitFiltered) ## remove linear trend

		if plots:
			plt.clf()
			plt.plot(rowPxls[indices],counts,'b',linewidth=2)
			plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
			plt.plot(rowBit,sumRowsBit,'b.')
			plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
			plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
			plt.draw()		
			plt.show()
		rawFlux = np.sum(counts[sourceRange])
		rawError = np.sqrt(np.sum(imageColCorrectedUncertainty[indices][sourceRange]))
		return rawFlux,rawError,psf

	def photLiteNoBG(self,image,imageSum,pxlBounds,leftCentroid,rightCentroid,psfMasks,plots=False,diagnostics=False):
		'''Do photometry with the results returned by self.trackStar()
		   Inputs: 	image 		- A-B nod subtracted image
		   			imageSum 	- A+B nod ADDED image 
		   			
		   DO NOT remove left-over skylines after nod-subtraction.'''
		## Sum up all rows
		rowPxls = np.arange(pxlBounds[0],pxlBounds[1])
		sumRows = np.sum(image,axis=1)

		psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
		midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
		leftHalfIndices = (midPoint > rowPxls)
		rightHalfIndices = (midPoint <= rowPxls)
		print leftCentroid,rightCentroid
		if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): set = [leftHalfIndices,float(leftCentroid)]
		else: set = [rightHalfIndices,float(rightCentroid)]
		indices = set[0]
		maskedOut = ((set[1]+psfMaskRight < rowPxls[indices]) + (set[1]-psfMaskLeft > rowPxls[indices]))
		sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))
		#rowBit = rowPxls[indices][maskedOut]
		#sumRowsBit = sumRows[indices][maskedOut]	## Cropped bit of "sumRows" for background

		## Subtract median of column intensities	
		#sumColsBit = np.median(image[indices,:][maskedOut],axis=0)
		#imageColCorrected = image - np.meshgrid(sumColsBit,rowPxls)[0]
		imageColCorrected = image
				
		## To find the error, take the square root of the signal, then square their sums, or equivalently, the square root of the sum:
		#skyLineCorrectedSum = imageSum + np.meshgrid(sumColsBit,rowPxls)[0]
		imageColCorrectedUncertainty = imageSum#skyLineCorrectedSum

		## Rename previously defined variables with new column sky line corrected version
		sumRows = np.sum(imageColCorrected,axis=1)
		#sumRowsBit = sumRows[indices][maskedOut]
		## calculate sigma spread of PSF in the positive half:
		psf = scipyCalcSigma(rowPxls[indices],sumRows[indices])
		
		#rowBitFiltered, sumRowsBitFiltered = meanFilter(rowBit,sumRowsBit,4) ## Mean filter the background pixels
		#counts,bestfit = removeLinearTrend(rowPxls[indices],sumRows[indices],rowBitFiltered,sumRowsBitFiltered) ## remove linear trend
		#bestfit = removeLinearTrend(rowPxls[indices],sumRows[indices],rowBitFiltered,sumRowsBitFiltered)[1] ## remove linear trend
		
		counts = sumRows[indices]
		if plots:
			plt.clf()
			plt.plot(rowPxls[indices],counts,'b',linewidth=2)
			#plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
			#plt.plot(rowBit,sumRowsBit,'b.')
			plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
			plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
			plt.draw()		
			plt.show()
		rawFlux = np.sum(counts[sourceRange])
		rawError = np.sqrt(np.sum(imageColCorrectedUncertainty[indices][sourceRange]))
		if diagnostics:
			finalBgStd = np.std(imageColCorrected[maskedOut]) ## After nod subtraction and column sky-line correction
			finalBgSqSum = np.sum(np.power(imageColCorrected[maskedOut],2))
			skyLineBgStd = 0#np.std(sumColsBit)
			skyLineBgSqSum = 0#np.sum(np.power(sumColsBit,2))
			return rawFlux, rawError, psf, finalBgSqSum, finalBgStd, skyLineBgSqSum, skyLineBgStd
		else:
			return rawFlux,rawError,psf

	def photLite3(self,image,imageSum,pxlBounds,bgBounds,leftCentroid,rightCentroid,psfMasks,plots=False,diagnostics=False):
		'''Do photometry with the results returned by self.trackStar()
		   Inputs: 	image 		- A-B nod subtracted image
		   			imageSum 	- A+B nod ADDED image 
		   			pxlBounds 	- boundaries of pixel indices (rows) on which each slit of interest falls
		   			bgBounds	- boundaries of pixel indices (rows) on which the background (non-source) falls (assume same as pxlBounds)
		   			leftCentroid - centroid of the star on the left half of the spatial direction (be it positive or negative)
		   			rightCentroid - centroid of the star on the right half of the spatial direction (be it positive or negative)
		   			psfMasks 	- (two element list) how many pixels on the left and right sides of the source to count as "source" pixels within the PSF
		   			plots 		- (boolean) show plots for each aperture photometry measurement
		   			diagnostics	- save all kinds of other diagnostic statistics or not
		   			'''
		'''Since the sky-lines need to be subtracted but the source region must use all available parts of the slit
		in the spatial direction, use part of the source region to subtract the sky lines. Label which pixels to use
		for sky line subtraction with new boundaries: "bgBounds"'''
		rowPxls = np.arange(pxlBounds[0],pxlBounds[1]) ## save indices of each row
		sumRows = np.sum(image,axis=1) ## Sum up all rows

		'''For each nod, the positive nod is either on the left or right half of the image
		in the spatial direction. Identify where it is, and divide the image into appropriate
		halves accordingly so as to handle only the positive nod.'''
		psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
		midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
		leftHalfIndices = (midPoint > rowPxls)
		rightHalfIndices = (midPoint <= rowPxls)
		if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): set = [leftHalfIndices,float(leftCentroid)]
		else: set = [rightHalfIndices,float(rightCentroid)]
		indices = set[0]	## indices is the half of the rows with the positive source
		maskedOut = ((bgBounds[0] < rowPxls)*(bgBounds[1] > rowPxls)) ## Define "background" pixels
		
		sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))	## Define source pixels
		rowBit = rowPxls
		sumRowsBit = sumRows[maskedOut]	## Cropped bit of "sumRows" for background

		''' Subtract median of column intensities from each column, to subtract out some of the
		left-over sky lines from poor nod subtractions.'''
		sumColsBit = np.median(image[maskedOut,:],axis=0)
		imageColCorrected = image - np.meshgrid(sumColsBit,rowPxls)[0]
		skyLineCorrectedSum = imageSum + np.meshgrid(sumColsBit,rowPxls)[0]	## Do the same process to the summed frames for error propagation
		imageColCorrectedUncertainty = skyLineCorrectedSum
		
		sumRows = np.sum(imageColCorrected,axis=1) ## Rename previously defined variables with new column sky line corrected version
		psf = scipyCalcSigma(rowPxls[indices],sumRows[indices]) ## calculate sigma spread of PSF for the positive source:

		counts = sumRows[indices]
		if plots:
			plt.clf()
			plt.plot(rowPxls,sumRows,'k',linewidth=1)
			#plt.plot(rowPxls[indices],counts,'b',linewidth=3)
			plt.plot(rowPxls[indices][sourceRange],counts[sourceRange],'b',linewidth=3)
			#plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
			plt.plot(rowPxls[maskedOut],sumRows[maskedOut],'ro',linewidth=2)
			#plt.plot(rowBit,sumRowsBit,'b.')
			plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
			plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
			plt.draw()		
			plt.show()
		rawFlux = np.sum(counts[sourceRange])	## The flux is the sum of the counts
		rawError = np.sqrt(np.sum(imageColCorrectedUncertainty[indices][sourceRange])) ## The error is the sqrt of the sum of the counts in the uncertainty frame
		if diagnostics:
			finalBgStd = np.std(imageColCorrected[maskedOut]) ## After nod subtraction and column sky-line correction
			finalBgSqSum = np.sum(np.power(imageColCorrected[maskedOut],2))
			skyLineBgStd = 0#np.std(sumColsBit)
			skyLineBgSqSum = 0#np.sum(np.power(sumColsBit,2))
			return rawFlux, rawError, psf, finalBgSqSum, finalBgStd, skyLineBgSqSum, skyLineBgStd
		else:
			return rawFlux,rawError,psf
	def photLite4(self,image,imageSum,pxlBounds,bgBounds,leftCentroid,rightCentroid,psfMasks,plots=False,diagnostics=False):
		'''Do photometry with the results returned by self.trackStar()
		   Inputs: 	image 		- A-B nod subtracted image
		   			imageSum 	- A+B nod ADDED image 
		   			pxlBounds 	- boundaries of pixel indices (rows) on which each slit of interest falls
		   			bgBounds	- boundaries of pixel indices (rows) on which the background (non-source) falls (assume same as pxlBounds)
		   			leftCentroid - centroid of the star on the left half of the spatial direction (be it positive or negative)
		   			rightCentroid - centroid of the star on the right half of the spatial direction (be it positive or negative)
		   			psfMasks 	- (two element list) how many pixels on the left and right sides of the source to count as "source" pixels within the PSF
		   			plots 		- (boolean) show plots for each aperture photometry measurement
		   			diagnostics	- save all kinds of other diagnostic statistics or not
		   			'''
		'''Since the sky-lines need to be subtracted but the source region must use all available parts of the slit
		in the spatial direction, use part of the source region to subtract the sky lines. Label which pixels to use
		for sky line subtraction with new boundaries: "bgBounds"'''
		rowPxls = np.arange(pxlBounds[0],pxlBounds[1]) ## save indices of each row
		sumRows = np.sum(image,axis=1) ## Sum up all rows

		'''For each nod, the positive nod is either on the left or right half of the image
		in the spatial direction. Identify where it is, and divide the image into appropriate
		halves accordingly so as to handle only the positive nod.'''
		psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
		midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
		leftHalfIndices = (midPoint > rowPxls)
		rightHalfIndices = (midPoint <= rowPxls)
		if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): 
			set = [leftHalfIndices,float(leftCentroid)]
		else: 
			set = [rightHalfIndices,float(rightCentroid)]
		indices = set[0]	## indices is the half of the rows with the positive source
		#maskedOut = ((bgBounds[0] < rowPxls)*(bgBounds[1] > rowPxls)) ## Define "background" pixels
		maskedOut = ((set[1]+psfMaskRight < rowPxls[indices]) + (set[1]-psfMaskLeft > rowPxls[indices]))
		
		sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))	## Define source pixels
		rowBit = rowPxls
		
		sumRowsBit = sumRows[maskedOut]	## Cropped bit of "sumRows" for background

		''' Subtract median of column intensities from each column, to subtract out some of the
		left-over sky lines from poor nod subtractions.'''
		sumColsBit = np.median(image[maskedOut,:],axis=0)
		imageColCorrected = image - np.meshgrid(sumColsBit,rowPxls)[0]
		#skyLineCorrectedSum = imageSum + np.meshgrid(sumColsBit,rowPxls)[0]	## Do the same process to the summed frames for error propagation
		#imageColCorrectedUncertainty = skyLineCorrectedSum
		UNCERTAINTY  = imageSum*(1 + 1./np.shape(imageSum[maskedOut,:])[0]**2)
		
		sumRows = np.sum(imageColCorrected,axis=1) ## Rename previously defined variables with new column sky line corrected version
		psf = scipyCalcSigma(rowPxls[indices],sumRows[indices]) ## calculate sigma spread of PSF for the positive source:

		counts = sumRows[indices]
		if plots:
			plt.clf()
			plt.plot(rowPxls,sumRows,'k',linewidth=1)
			#plt.plot(rowPxls[indices],counts,'b',linewidth=3)
			plt.plot(rowPxls[indices][sourceRange],counts[sourceRange],'b',linewidth=3)
			#plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
			plt.plot(rowPxls[indices][maskedOut],sumRows[indices][maskedOut],'ro',linewidth=2)
			#plt.plot(rowBit,sumRowsBit,'b.')
			plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
			plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
			plt.draw()		
			plt.show()
		rawFlux = np.sum(counts[sourceRange])	## The flux is the sum of the counts
		rawError = np.sqrt(np.sum(UNCERTAINTY)) ## The error is the sqrt of the sum of the counts in the uncertainty frame
		
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		ax1.imshow(image[indices][maskedOut,:])
		ax2.plot(np.mean(image[indices][maskedOut,:],axis=0))
		plt.show()
		
		
		if diagnostics:
			
			return rawFlux, rawError, psf
		else:
			return rawFlux,rawError,psf
	def photLite5(self,image,imageSum,pxlBounds,bgBounds,leftCentroid,rightCentroid,psfMasks,plots=False,diagnostics=False):
		'''Do photometry with the results returned by self.trackStar()
		   Inputs: 	image 		- A-B nod subtracted image
		   			imageSum 	- A+B nod ADDED image 
		   			pxlBounds 	- boundaries of pixel indices (rows) on which each slit of interest falls
		   			bgBounds	- boundaries of pixel indices (rows) on which the background (non-source) falls (assume same as pxlBounds)
		   			leftCentroid - centroid of the star on the left half of the spatial direction (be it positive or negative)
		   			rightCentroid - centroid of the star on the right half of the spatial direction (be it positive or negative)
		   			psfMasks 	- (two element list) how many pixels on the left and right sides of the source to count as "source" pixels within the PSF
		   			plots 		- (boolean) show plots for each aperture photometry measurement
		   			diagnostics	- save all kinds of other diagnostic statistics or not
		   			'''
		'''Since the sky-lines need to be subtracted but the source region must use all available parts of the slit
		in the spatial direction, use part of the source region to subtract the sky lines. Label which pixels to use
		for sky line subtraction with new boundaries: "bgBounds"'''
		rowPxls = np.arange(pxlBounds[0],pxlBounds[1]) ## save indices of each row
		sumRows = np.sum(image,axis=1) ## Sum up all rows

		'''For each nod, the positive nod is either on the left or right half of the image
		in the spatial direction. Identify where it is, and divide the image into appropriate
		halves accordingly so as to handle only the positive nod.'''
		psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
		midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
		leftHalfIndices = (midPoint > rowPxls)
		rightHalfIndices = (midPoint <= rowPxls)
		if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): 
			set = [leftHalfIndices,float(leftCentroid)]
		else: 
			set = [rightHalfIndices,float(rightCentroid)]
		indices = set[0]	## indices is the half of the rows with the positive source
		#maskedOut = ((bgBounds[0] < rowPxls)*(bgBounds[1] > rowPxls)) ## Define "background" pixels
		maskedOut = ((set[1]+psfMaskRight < rowPxls[indices]) + (set[1]-psfMaskLeft > rowPxls[indices]))
		
		sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))	## Define source pixels
		rowBit = rowPxls
		
		sumRowsBit = sumRows[indices][maskedOut]	## Cropped bit of "sumRows" for background

		''' Subtract median of column intensities from each column, to subtract out some of the
		left-over sky lines from poor nod subtractions.'''
		sumColsBit = np.median(image[indices][maskedOut,:],axis=0)
		imageColCorrected = image - np.meshgrid(sumColsBit,rowPxls)[0]
		#skyLineCorrectedSum = imageSum + np.meshgrid(sumColsBit,rowPxls)[0]	## Do the same process to the summed frames for error propagation
		#imageColCorrectedUncertainty = skyLineCorrectedSum
		UNCERTAINTY  = imageSum*(1 + 1./np.shape(imageSum[indices][maskedOut,:])[0]**2)
		
		sumRows = np.sum(imageColCorrected,axis=1) ## Rename previously defined variables with new column sky line corrected version
		psf = scipyCalcSigma(rowPxls[indices],sumRows[indices]) ## calculate sigma spread of PSF for the positive source:

		counts = sumRows[indices]
		if plots:
			plt.clf()
			plt.plot(rowPxls,sumRows,'k',linewidth=1)
			#plt.plot(rowPxls[indices],counts,'b',linewidth=3)
			plt.plot(rowPxls[indices][sourceRange],counts[sourceRange],'b',linewidth=3)
			#plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
			plt.plot(rowPxls[indices][maskedOut],sumRows[indices][maskedOut],'ro',linewidth=2)
			#plt.plot(rowBit,sumRowsBit,'b.')
			plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
			plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
			plt.draw()		
			plt.show()
		#plt.plot(counts[sourceRange])
		#plt.show()
		rawFlux = np.sum(counts[sourceRange])	## The flux is the sum of the counts
		rawError = np.sqrt(np.sum(UNCERTAINTY)) ## The error is the sqrt of the sum of the counts in the uncertainty frame
		
		if False:
			fig = plt.figure()
			ax1 = fig.add_subplot(211)
			ax2 = fig.add_subplot(212)
			ax1.imshow(imageColCorrected[indices][maskedOut,:])
			ax2.plot(np.mean(imageColCorrected[indices][maskedOut,:],axis=0),label='Corrected image')
			ax2.plot(np.mean(image[indices][maskedOut,:],axis=0),label='Uncorrected image')
			ax2.legend()
			plt.show()
			
		correctedBG = np.mean(imageColCorrected[indices][maskedOut,:],axis=0)
		if diagnostics:
			return rawFlux, rawError, psf,correctedBG
		else:
			return rawFlux,rawError,psf
	
	
	
	
	def photLite5psf(self,image,imageSum,pxlBounds,bgBounds,leftCentroid,rightCentroid,psfMasks,plots=False,diagnostics=False):
		'''Do photometry with the results returned by self.trackStar()
		   Inputs: 	image 		- A-B nod subtracted image
		   			imageSum 	- A+B nod ADDED image 
		   			pxlBounds 	- boundaries of pixel indices (rows) on which each slit of interest falls
		   			bgBounds	- boundaries of pixel indices (rows) on which the background (non-source) falls (assume same as pxlBounds)
		   			leftCentroid - centroid of the star on the left half of the spatial direction (be it positive or negative)
		   			rightCentroid - centroid of the star on the right half of the spatial direction (be it positive or negative)
		   			psfMasks 	- (two element list) how many pixels on the left and right sides of the source to count as "source" pixels within the PSF
		   			plots 		- (boolean) show plots for each aperture photometry measurement
		   			diagnostics	- save all kinds of other diagnostic statistics or not
		   			'''
		'''Since the sky-lines need to be subtracted but the source region must use all available parts of the slit
		in the spatial direction, use part of the source region to subtract the sky lines. Label which pixels to use
		for sky line subtraction with new boundaries: "bgBounds"'''
		rowPxls = np.arange(pxlBounds[0],pxlBounds[1]) ## save indices of each row
		sumRows = np.sum(image,axis=1) ## Sum up all rows

		'''For each nod, the positive nod is either on the left or right half of the image
		in the spatial direction. Identify where it is, and divide the image into appropriate
		halves accordingly so as to handle only the positive nod.'''
		psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
		midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
		leftHalfIndices = (midPoint > rowPxls)
		rightHalfIndices = (midPoint <= rowPxls)
		if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): 
			set = [leftHalfIndices,float(leftCentroid)]
		else: 
			set = [rightHalfIndices,float(rightCentroid)]
		indices = set[0]	## indices is the half of the rows with the positive source
		#maskedOut = ((bgBounds[0] < rowPxls)*(bgBounds[1] > rowPxls)) ## Define "background" pixels
		maskedOut = ((set[1]+psfMaskRight < rowPxls[indices]) + (set[1]-psfMaskLeft > rowPxls[indices]))
		
		sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))	## Define source pixels
		rowBit = rowPxls
		
		sumRowsBit = sumRows[indices][maskedOut]	## Cropped bit of "sumRows" for background

		''' Subtract median of column intensities from each column, to subtract out some of the
		left-over sky lines from poor nod subtractions.'''
		sumColsBit = np.median(image[indices][maskedOut,:],axis=0)
		imageColCorrected = image - np.meshgrid(sumColsBit,rowPxls)[0]
		#skyLineCorrectedSum = imageSum + np.meshgrid(sumColsBit,rowPxls)[0]	## Do the same process to the summed frames for error propagation
		#imageColCorrectedUncertainty = skyLineCorrectedSum
		UNCERTAINTY  = imageSum*(1 + 1./np.shape(imageSum[indices][maskedOut,:])[0]**2)
		
		sumRows = np.sum(imageColCorrected,axis=1) ## Rename previously defined variables with new column sky line corrected version
		psf = scipyCalcSigma(rowPxls[indices],sumRows[indices]) ## calculate sigma spread of PSF for the positive source:

		################
		## Single gaussian
#		p = [max(sumRows[indices]),rowPxls[sumRows[indices]==np.max(sumRows[indices])],3.0]
#		def gaussian(p,x):
#			height, center_x, width_x = p
#			"""Returns a gaussian function with the given parameters"""
#			return height*np.exp(-0.5*(((center_x-x)/float(width_x))**2))
#		def errfunc(p,x,y):
#			return gaussian(p,x) - y
#		bestFitP = optimize.leastsq(errfunc,p[:],args=(rowPxls[indices],sumRows[indices]))[0]
#		print bestFitP
#		bestFitPSF = gaussian(bestFitP,rowPxls[indices])
		################

		################
		## Double gaussian
		#twoP = [max(sumRows[indices]),rowPxls[sumRows[indices]==np.max(sumRows[indices])],3.0,\
		#	    max(sumRows[indices]),rowPxls[sumRows[indices]==np.max(sumRows[indices])],3.0]
		twoP = [max(sumRows[indices]),rowPxls[indices][sumRows[indices]==np.max(sumRows[indices])],3.0,0.0,\
			    5000.,rowPxls[indices][sumRows[indices]==np.max(sumRows[indices])],10.,0.0]
		
		
		def gaussian(p,x):
			height, center_x, width_x,y_offset = p
			"""Returns a gaussian function with the given parameters"""
			global condition
			condition = (np.abs(x - center_x) < 20)
			#y[condition] = 0.0
			return height*np.exp(-0.5*(((center_x-x)/float(width_x))**2)) #+ y_offset

		def maskedGaussian(p,x):
			height, center_x, width_x,y_offset = p
			"""Returns a gaussian function with the given parameters"""
			y= height*np.exp(-0.5*(((center_x-x)/float(width_x))**2)) + y_offset
			#y[(np.abs(x - center_x) < 15.0)] = 0.0 	## Don't fit within 15 rows of the centroid
			#y[(np.abs(x - center_x) < 20.0)*(x < center_x)] = 0.0
			return y
		
		def zeroOut(input,condition):
			input[condition] = 0
			return input
		
		#def errfunc(twoP,x,y):
		#	print twoP[0:3],twoP[3:6]
		#	return gaussian(twoP[0:3],x) + gaussian(twoP[3:6],x) - y
		def errfunc(twoP,x,y):
			if y==None:
				return gaussian(twoP[0:4],x) + zeroOut(maskedGaussian(twoP[4:8],x),condition)
			else:
				return gaussian(twoP[0:4],x) + zeroOut(maskedGaussian(twoP[4:8],x),condition) - y

		bestFitP = optimize.leastsq(errfunc,twoP[:],args=(rowPxls[indices],sumRows[indices]))[0]
		#print bestFitP
		#bestFitPSF = gaussian(bestFitP[0:4],rowPxls[indices]) + maskedGaussian(bestFitP[4:8],rowPxls[indices])
		bestFitPSF = errfunc(bestFitP,rowPxls[indices],None)
		secondGaussian = gaussian(bestFitP[4:8],rowPxls[indices])
		bestFitPSFmissing = errfunc(bestFitP,rowPxls,None)[rowPxls < midPoint]
		
		missingFlux = np.sum(bestFitPSFmissing)
		################		
		
#		bestFitPSF = psfFit(rowPxls[indices],sumRows[indices],initialParameters=[max(sumRows[indices]),rowPxls[sumRows[indices]==np.max(sumRows[indices])],3])
		counts = sumRows[indices]
		rawFlux = np.sum(counts[sourceRange])	## The flux is the sum of the counts
		rawError = np.sqrt(np.sum(UNCERTAINTY)) ## The error is the sqrt of the sum of the counts in the uncertainty frame
		#print 'missingFlux/flux',missingFlux/rawFlux
		
		awayFromCentroid = (rowPxls[indices] < bestFitP[1]-20)+(rowPxls[indices] > bestFitP[1]+20)
		
		if plots:
			#plt.clf()
			fig = plt.figure(figsize=(12,12))
			plt.plot(rowPxls,sumRows,'k',linewidth=2,label='Data')
			#plt.plot(rowPxls[indices],counts,'b',linewidth=3)
			#plt.plot(rowPxls[indices][sourceRange],counts[sourceRange],'b',linewidth=3)
			#plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
			#plt.plot(rowPxls[indices][maskedOut],sumRows[indices][maskedOut],'ro',linewidth=2)
			plt.plot(rowPxls[indices][awayFromCentroid],bestFitPSF[awayFromCentroid],'g',linewidth=2,label='Double gauss fit')
			plt.plot(rowPxls[rowPxls < midPoint],bestFitPSFmissing,'r',linewidth=2,label='Double gauss extrapolated')

			#plt.plot(rowPxls[indices], secondGaussian,'m',linewidth=2,label='Wing gauss fit only')
			#plt.plot(rowBit,sumRowsBit,'b.')
			plt.axvline(ymin=0,ymax=1,x=midPoint,linewidth=2,color='k',linestyle=':',label='Midpoint between centroids\nof opposite nods')
			plt.xlabel('Pixel row')
			plt.ylabel('Flux (Counts)')
			plt.legend()
			#plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
			#plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
			#plt.draw()		
			plt.show()

#		if plots:
#			plt.clf()
#			fig = plt.figure(figsize=(12,12))
#			plt.plot(rowPxls,sumRows,'k',linewidth=1)
#			#plt.plot(rowPxls[indices],counts,'b',linewidth=3)
#			plt.plot(rowPxls[indices][sourceRange],counts[sourceRange],'b',linewidth=3)
#			#plt.plot(rowPxls[indices],bestfit,'r',linewidth=2)
#			plt.plot(rowPxls[indices][maskedOut],sumRows[indices][maskedOut],'ro',linewidth=2)
#			plt.plot(rowPxls[indices],bestFitPSF,'g',linewidth=2)
#			plt.plot(rowPxls[indices], secondGaussian,'m',linewidth=2)
#			#plt.plot(rowBit,sumRowsBit,'b.')
#			plt.axvline(ymin=0,ymax=1,x=leftCentroid,linewidth=2,color='k')
#			plt.axvline(ymin=0,ymax=1,x=rightCentroid,linewidth=2,color='k')
#			plt.draw()		
#			plt.show()

		if False:
			fig = plt.figure()
			ax1 = fig.add_subplot(211)
			ax2 = fig.add_subplot(212)
			ax1.imshow(imageColCorrected[indices][maskedOut,:])
			ax2.plot(np.mean(imageColCorrected[indices][maskedOut,:],axis=0),label='Corrected image')
			ax2.plot(np.mean(image[indices][maskedOut,:],axis=0),label='Uncorrected image')
			ax2.legend()
			plt.show()
			
		correctedBG = np.mean(imageColCorrected[indices][maskedOut,:],axis=0)
		#print set[1]==rightCentroid
		if diagnostics and float(rightCentroid) == set[1]:
			#return rawFlux, rawError, psf,correctedBG
			return rawFlux, rawError, psf,missingFlux
		else:
			return rawFlux,rawError,psf,None

	
	def alignSpectra(self,image,rootInd,plots=False):
		'''Take a cross-correlation between each of the stars and line up their spectra in wavelength-space
		vOld features the old sky-line final background subtraction algorithm'''
		spectra = []
		for star in self.starDictionaries():
			pxlBounds = star['pxlBounds']
			psfMasks = star['psfMasks']
			[leftCentroid,rightCentroid] = self.trackStar(image, pxlBounds, rootInd, plots=False, returnCentroidsOnly=True)
		
			rowPxls = np.arange(pxlBounds[0],pxlBounds[1])
			sumRows = np.sum(image[pxlBounds[0]:pxlBounds[1],:],axis=1)		
			psfMaskLeft = psfMasks[0]; psfMaskRight = psfMasks[1]
			midPoint = 0.5*(float(leftCentroid)+float(rightCentroid))
			leftHalfIndices = (midPoint > rowPxls)
			rightHalfIndices = (midPoint <= rowPxls)
		
			if np.mean(sumRows[leftHalfIndices]) > np.mean(sumRows[rightHalfIndices]): set = [leftHalfIndices,float(leftCentroid)]
			else: set = [rightHalfIndices,float(rightCentroid)]
			indices = set[0]
			maskedOut = ((set[1]+psfMaskRight < rowPxls[indices]) + (set[1]-psfMaskLeft > rowPxls[indices]))
			sourceRange = ((set[1]+psfMaskRight >  rowPxls[indices]) * (set[1]-psfMaskRight < rowPxls[indices]))
	
			sumColsBit = np.median(image[pxlBounds[0]:pxlBounds[1],:][indices,:][maskedOut],axis=0)
			imageColCorrected = image[pxlBounds[0]:pxlBounds[1],:] - np.meshgrid(sumColsBit,rowPxls)[0]
			spectrum = np.sum(imageColCorrected[indices][sourceRange],axis=0)
			spectra.append(spectrum)

		p1,s1 = findBestRoll(spectra[0],spectra[1])
		p2,s2 = findBestRoll(spectra[0],spectra[2])
	
		nonZeroRegion = (s1 != 0)*(s2 != 0)

		colBounds = []
		colBounds.append(rollCol(0,nonZeroRegion))
		colBounds.append(rollCol(p1,nonZeroRegion))
		colBounds.append(rollCol(p2,nonZeroRegion))
		starNumber = 0
		
		scalingFactor = self.starDictionaries()[0]['wavelengthParams'][0] 	## Initialize self.wavelengths variable that stores the
		offset = self.starDictionaries()[0]['wavelengthParams'][1] 			## the scaled wavelengths in Angstroms
		self.wavelengths = scalingFactor*np.arange(0,2048)[nonZeroRegion] + offset
		
		for star in self.starDictionaries():
			star['colBounds'] = colBounds[starNumber]
			starNumber += 1

		if plots:
			plt.plot(self.wavelengths,spectra[0][nonZeroRegion])
			plt.plot(self.wavelengths,s1[nonZeroRegion])
			plt.plot(self.wavelengths,s2[nonZeroRegion])
			plt.title('Aligned spectra')
			plt.xlabel('Wavelength ($\AA$)')
			plt.ylabel('Flux')
			plt.show()

	def calibrateColumns(self,calibrationArchivePath,plots=False):
		colBounds = np.load(calibrationArchivePath)['colBounds']
		
		starNumber = 0
		for star in self.starDictionaries():
			star['colBounds'] = colBounds[starNumber]
			starNumber += 1
		self.wavelengths = np.load(calibrationArchivePath)['wavelengths']
		return self.wavelengths
	
	def alignedWavelengths(self):
		return self.wavelengths

def ut2jd(utshut):
	[date, Time] = utshut.split(';')
	[year, month, day] = date.split('-')
	[hour, minutes, sec] = Time.split(':')
	year = int(year); month = int(month); day = int(day)
	hour = int(hour); minutes = int(minutes); sec = float(sec)
	if month == 1 or month == 2: 
		month += 12
		year -= 1	
	a = year/100
	b = a/4
	c = 2-a+b
	d = day
	e = math.floor(365.25*(year+4716))
	f = math.floor(30.6001*(month+1))
	years = c+d+e+f-1524.5
	fracOfDay = (hour/24.) + (minutes/(24*60.)) + (sec/(24*60*60.))
	jd = years + fracOfDay
	return jd

def regressionScale(comparisonFlux,targetFlux,time,ingress,egress):
	'''Use a least-squares regression to stretch and offset a comparison star fluxes
	   to scale them to the relative intensity of the target star. Only do this regression
	   considering the out-of-transit portions of the light curve.'''
	outOfTransit = (time < ingress) + (time > egress)
	regressMatrix = np.vstack([comparisonFlux[outOfTransit], np.ones_like(targetFlux[outOfTransit])]).T
	m,c = np.linalg.lstsq(regressMatrix,targetFlux[outOfTransit])[0]
	scaledVector = m*comparisonFlux + c
	return scaledVector

def meanFilter(timeVector,fluxVector,sigmaThreshold):
	inThreshold = np.abs(fluxVector - np.mean(fluxVector)) < sigmaThreshold*np.std(fluxVector)
	return timeVector[inThreshold],fluxVector[inThreshold]

def removeLinearTrend(xVector,yVector,xVectorCropped,yVectorCropped,returnBestFitP=False):
	'''Fit a line to the set {xVectorCropped,yVectorCropped}, then remove that linear trend
	   from the full set {xVector,yVector}'''
	initP = [0.0,0.0]
	fitfunc = lambda p, x: p[0]*x + p[1]
	errfunc = lambda p, x, y: (fitfunc(p,x) - y)
	bestFitP = optimize.leastsq(errfunc,initP[:],args=(xVectorCropped,yVectorCropped))[0]
	if returnBestFitP==False:
		return yVector - fitfunc(bestFitP,xVector), fitfunc(bestFitP,xVector)
	else: 
		return yVector - fitfunc(bestFitP,xVector), fitfunc(bestFitP,xVector), bestFitP
	
def linearTrend(xVector,yVector,returnBestFitP=False):
	'''Fit a line to the set {xVectorCropped,yVectorCropped}, then remove that linear trend
	   from the full set {xVector,yVector}'''
	print 'linearTrend'
	initP = [0.0,0.0]
	fitfunc = lambda p, x: p[0]*x + p[1]
	errfunc = lambda p, x, y: (fitfunc(p,x) - y)
	bestFitP = optimize.leastsq(errfunc,initP[:],args=(xVector,yVector))[0]
	return bestFitP


 ## Source: prewritten scipy routines for Gaussian fitting: http://www.scipy.org/Cookbook/FittingData
def gaussian(height, center_x, width_x):
	"""Returns a gaussian function with the given parameters"""
	return lambda x: height*np.exp(-0.5*(((center_x-x)/float(width_x))**2))
def moments(data):
	"""Returns (height, x, y, width_x, width_y)
	the gaussian parameters of a 2D distribution by calculating its
	moments """
	total = data.sum()
	X = np.indices(data.shape)
	x = (X*data).sum()/total
	width_x = np.sqrt(np.abs((np.arange(data.size))**2*data).sum()/data.sum())
	height = data.max()
	return height, x, width_x
def fitgaussian(data,initialParameters=None):
	'''If initialParameters are input, let these be the guesses for the Amplitude, x-centroid and sigma of the gaussian'''
	"""Returns (height, x, y, width_x, width_y)
	the gaussian parameters of a 2D distribution found by a fit"""
	errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape))-data)
	if initialParameters==None:
		return optimize.leastsq(errorfunction, moments(data),maxfev=int(1e4))[0]
	else:
		return optimize.leastsq(errorfunction,initialParameters,maxfev=int(1e4))[0]

def scipyCalcSigma(xVector,yVector):
	'''Take data resembling a guassian and return the "sigma" of the best fit gaussian to the data'''
	return abs(fitgaussian(yVector)[2])

def psfFit(xVector,yVector,initialParameters=None):
	'''Take data resembling a guassian and return the "sigma" of the best fit gaussian to the data'''
	bestFitGaussian = gaussian(*fitgaussian(yVector,initialParameters=initialParameters))
	print fitgaussian(yVector,initialParameters=initialParameters)
	return bestFitGaussian(xVector)

def myRound(a, decimals=0):
	return np.around(a-10**(-(decimals+5)), decimals=decimals)

def quadFit(rows,derivative,condition,ext,offset):
	'''Rows -- pixel row numbers
	   Derivative -- derivative of sumRows
	   Condition -- indices of Rows and sumRows to consider (which half)
	   Offset -- include index offset for the second half of the indices'''
	rows = rows[condition]
	derivative = derivative[condition]
	if ext == "max": indExtrema = np.argmax(derivative)
	else: indExtrema = np.argmin(derivative)	## Else ext == "min" is assumed
		
	fitPart = derivative[indExtrema-1:indExtrema+2]
	if len(fitPart) == 3:
		stackPolynomials = [0,0,0]
		for i in range(0,len(fitPart)):
			vector = [i**2,i,1]
			stackPolynomials = np.vstack([stackPolynomials,vector])
		estimatedCoeffs = np.dot(LA.inv(stackPolynomials[1:,:]),fitPart)
		d_fit = -estimatedCoeffs[1]/(2.*estimatedCoeffs[0])
		extremum = d_fit+float(indExtrema)#+offset
	else: 
		extremum = indExtrema #+ offset
	return rows[myRound(extremum)],extremum

### Added for spectra-matching via cross-correlation

def appendBuffer(vector,buffer):
	return np.concatenate([np.zeros([buffer]),vector,np.zeros([buffer])])

def crossCorrelate(vector1,vector2):
	return np.sum(np.power(vector1-vector2,2))

def findBestRoll(vector0,vector1):
	'''Roll one vector over the other and return roll index where the cross-correlation is minimized,
	   and the vector in the best roll position'''
	#import matplotlib; matplotlib.interactive(True)
	buffer = 2048
	v0 = appendBuffer(vector0,460)
	v1 = appendBuffer(vector1,460)
	minCorrelation = 1e20
	for i in np.arange(-450,0,1):
		rolledSpectra = np.roll(v1,i)
		rolledSpectra *= np.mean(v0)/np.mean(rolledSpectra)
		relaventRegions = (rolledSpectra != 0)*(v1 != 0)
		correlation = crossCorrelate(rolledSpectra[relaventRegions],v0[relaventRegions])
		if correlation < minCorrelation:
			minCorrelation = correlation
			bestRoll = i
	return bestRoll,np.concatenate([np.roll(vector1,bestRoll)[:bestRoll],np.zeros([-1*bestRoll])])

def rollCol(bestRoll,nonZeroCondition):
	'''Given the best roll index and a boolean array indicating the non-zero entries of the rolled spectra,
	   return the column numbers of the final rolled, non-zero spectra'''
	rolledCols = np.roll(np.arange(0,2048),bestRoll)[nonZeroCondition]
	return [rolledCols[0],rolledCols[-1]]
