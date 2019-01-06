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

def trackStar(image,pxlBounds,rootInd,plots=False,returnCentroidsOnly=False):
    '''Track the centroids of the left and right (A and B, positive and negative) images of the star located between
       rows lowerPxlBound and upperPxlBound. Returns the lists of left and right centroid positions'''
    #if plots==True: import matplotlib; matplotlib.interactive(True);fig = plt.figure(figsize=(10,10))
    [lowerPxlBound,upperPxlBound] = pxlBounds
    rowPxls = np.arange(lowerPxlBound,upperPxlBound)
    ## Numpy pickle dictionary keys (from organizeImages/AminusB2.py):
    ##       image=nodSubtractedImage,time=expTime,path=currentPath,rootInd=rawObj.path2ind(currentPath),nod=nodName)
    sumRows = np.sum(image[lowerPxlBound:upperPxlBound,:],axis=1) ## Sum up cropped image along rows
    #rootInd =  np.load(path)['rootInd']
    rawMedianOfsumRows = np.median(sumRows)     ## Subtract sumRows by it's median before inversion
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
    sumRowsInverted += rawMedianOfsumRows         ## Add back the median background of the raw image

    ## Use derivative of sumRows to locate stellar centroid
    sumRowsInvertedSmoothed = ndimage.gaussian_filter(sumRowsInverted,sigma=smoothConst,order=0)    ## Smooth out sumRows
    derivativeOfsumRows = np.diff(sumRowsInvertedSmoothed)                                            ## ...take derivative

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
    if plots:    ## Generate plots
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

def trackStarFine(image,pxlBounds,rootInd,plots=False,returnCentroidsOnly=False):
    '''Track the centroids of the left and right (A and B, positive and negative) images of the star located between
       rows lowerPxlBound and upperPxlBound. Returns the lists of left and right centroid positions'''
    #if plots==True: import matplotlib; matplotlib.interactive(True);fig = plt.figure(figsize=(10,10))
    [lowerPxlBound,upperPxlBound] = pxlBounds
    rowPxls = np.arange(lowerPxlBound,upperPxlBound)
    ## Numpy pickle dictionary keys (from organizeImages/AminusB2.py):
    ##       image=nodSubtractedImage,time=expTime,path=currentPath,rootInd=rawObj.path2ind(currentPath),nod=nodName)
    sumRows = np.sum(image[lowerPxlBound:upperPxlBound,:],axis=1) ## Sum up cropped image along rows
    #rootInd =  np.load(path)['rootInd']
    rawMedianOfsumRows = np.median(sumRows)     ## Subtract sumRows by it's median before inversion
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
    sumRowsInverted += rawMedianOfsumRows         ## Add back the median background of the raw image

    ## Use derivative of sumRows to locate stellar centroid
    sumRowsInvertedSmoothed = ndimage.gaussian_filter(sumRowsInverted,sigma=smoothConst,order=0)    ## Smooth out sumRows
    derivativeOfsumRows = np.diff(sumRowsInvertedSmoothed)                                            ## ...take derivative

    ## Do rough matrix-algebra quadratic fit to the extrema of the derivative
    positiveHalfMax,positiveHalfMaxInd = quadFit(rowPxls,derivativeOfsumRows,positiveHalfIndices[1:],"max",positiveOffset)
    positiveHalfMin,positiveHalfMinInd = quadFit(rowPxls,derivativeOfsumRows,positiveHalfIndices[1:],"min",positiveOffset)
    negativeHalfMax,negativeHalfMaxInd = quadFit(rowPxls,derivativeOfsumRows,negativeHalfIndices[1:],"max",negativeOffset)
    negativeHalfMin,negativeHalfMinInd = quadFit(rowPxls,derivativeOfsumRows,negativeHalfIndices[1:],"min",negativeOffset)
    ## Find positive and negative stellar centroids, assign them to the left/right arrays
    #positiveCentroid = 0.5*(positiveHalfMax+positiveHalfMin)
    #negativeCentroid = 0.5*(negativeHalfMax+negativeHalfMin)
    positiveInd = 0.5*(positiveHalfMaxInd+positiveHalfMinInd)
    negativeInd = 0.5*(negativeHalfMaxInd+negativeHalfMinInd)

    # Fine measurement by interpolating
    negativeHalfMax_fine = np.interp(negativeHalfMaxInd, np.arange(sum(negativeHalfIndices)), rowPxls[negativeHalfIndices])
    negativeHalfMin_fine = np.interp(negativeHalfMinInd, np.arange(sum(negativeHalfIndices)), rowPxls[negativeHalfIndices])
    positiveHalfMax_fine = np.interp(positiveHalfMaxInd, np.arange(sum(positiveHalfIndices)), rowPxls[positiveHalfIndices])
    positiveHalfMin_fine = np.interp(positiveHalfMinInd, np.arange(sum(positiveHalfIndices)), rowPxls[positiveHalfIndices])
    negativeCentroid = 0.5*(negativeHalfMax_fine + negativeHalfMin_fine)
    positiveCentroid = 0.5*(positiveHalfMax_fine + positiveHalfMin_fine)

    if np.mean(sumRowsMedianSubtracted[leftHalfIndices]) < np.mean(sumRowsMedianSubtracted[rightHalfIndices]):
        leftCentroid = positiveCentroid
        rightCentroid = negativeCentroid
        centroidInd = positiveInd
    else:
        leftCentroid = negativeCentroid
        rightCentroid = positiveCentroid
        centroidInd = negativeInd
    if plots:    ## Generate plots
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


def phot(image, xCentroid, yCentroid, apertureRadius, plottingThings, annulusOuterRadiusFactor=2.8, annulusInnerRadiusFactor=1.40, ccdGain=1, sigmaclipping=False, plots=False, returnsubtractedflux=False):
    '''
    Method for aperture photometry, taken from the screwIRAF package. 
    
    Parameters
    ----------
    image : numpy.ndarray
        FITS image opened with PyFITS
    
    xCentroid : float
        Stellar centroid along the x-axis (determined by trackSmooth or equivalent)
                
    yCentroid : float
        Stellar centroid along the y-axis (determined by trackSmooth or equivalent)
                
    apertureRadius : float
        Radius in pixels from centroid to use for source aperture
                     
    annulusInnerRadiusFactor : float
        Measure the background for sky background subtraction fron an annulus from a factor of 
        `annulusInnerRadiusFactor` bigger than the `apertureRadius` to one a factor `annulusOuterRadiusFactor` bigger.
    
    annulusOuterRadiusFactor : float
        Measure the background for sky background subtraction fron an annulus a factor of 
        `annulusInnerRadiusFactor` bigger than the `apertureRadius` to one a factor `annulusOuterRadiusFactor` bigger.
                          
    ccdGain : float
        Gain of your detector, used to calculate the photon noise
    
    plots : bool
            If `plots`=True, display plots showing the aperture radius and 
            annulus radii overplotted on the image of the star
                   
    Returns
    -------
    rawFlux : float
        The background-subtracted flux measured within the aperture
    
    rawError : float
        The photon noise (limiting statistical) Poisson uncertainty on the measurement of `rawFlux`
    
    errorFlag : bool
        Boolean corresponding to whether or not any error occured when running oscaar.phot(). If an error occured, the flag is
        True; otherwise False.
               
     Core developer: Brett Morris (NASA-GSFC)
    '''
    if plots:
        [fig,subplotsDimensions,photSubplotsOffset] = plottingThings
        if photSubplotsOffset == 0: plt.clf()
    annulusRadiusInner = annulusInnerRadiusFactor*apertureRadius 
    annulusRadiusOuter = annulusOuterRadiusFactor*apertureRadius

    ## From the full image, cut out just the bit around the star that we're interested in
    imageCrop = image[xCentroid-annulusRadiusOuter+1:xCentroid+annulusRadiusOuter+2,yCentroid-annulusRadiusOuter+1:yCentroid+annulusRadiusOuter+2]
    [dimy,dimx] = imageCrop.shape
    XX, YY = np.meshgrid(np.arange(dimx),np.arange(dimy))    
    x = (XX - annulusRadiusOuter)**2
    y = (YY - annulusRadiusOuter)**2
    ## Assemble arrays marking the pixels marked as either source or background pixels
    sourceIndices = x + y <= apertureRadius**2
    skyIndices = (x + y <= annulusRadiusOuter**2)*(x + y >= annulusRadiusInner**2)

    if sigmaclipping:    
        clippedbackground = np.median(sigmaclip(imageCrop[skyIndices]))

        rawFlux = np.sum(imageCrop[sourceIndices] - clippedbackground)*ccdGain
        rawError = np.sqrt(np.sum(imageCrop[sourceIndices]*ccdGain) + ccdGain*clippedbackground) ## Poisson-uncertainty
    else:
        rawFlux = np.sum(imageCrop[sourceIndices] - np.median(imageCrop[skyIndices]))*ccdGain
        rawError = np.sqrt(np.sum(imageCrop[sourceIndices]*ccdGain) + np.median(ccdGain*imageCrop[skyIndices])) ## Poisson-uncertainty

#    fig = plt.figure()
#    cutoff = 4*np.std(imageCrop[skyIndices])
#    clipped = sigmaclip(imageCrop[skyIndices])
#    plt.hist(imageCrop[skyIndices],1000,facecolor='r',histtype='stepfilled',alpha=0.2)
#    plt.hist(clipped,1000,facecolor='w',histtype='stepfilled')
#    plt.axvline(ymin=0,ymax=1,x=np.median(imageCrop[skyIndices])+cutoff)
#    plt.axvline(ymin=0,ymax=1,x=np.median(imageCrop[skyIndices])-cutoff)
#    plt.axvline(ymin=0,ymax=1,x=np.median(imageCrop[skyIndices]),color='g')
#    plt.axvline(ymin=0,ymax=1,x=np.median(clipped),color='m')
#    plt.title('Difference in medians: %f' % ((np.median(imageCrop[skyIndices]) - np.median(clipped))/np.median(imageCrop[skyIndices])))
#    plt.show()

    if plots:
        def format_coord(x, y):
            ''' Function to also give data value on mouse over with imshow. '''
            col = int(x+0.5)
            row = int(y+0.5)
            try:
                return 'x=%i, y=%i, Flux=%1.1f' % (x, y, imageCrop[row,col])
            except:
                return 'x=%i, y=%i' % (x, y)
       
        med = np.median(imageCrop)
        dsig = np.std(imageCrop)
        
        ax = fig.add_subplot(subplotsDimensions+photSubplotsOffset+1)
        ax.imshow(imageCrop, cmap=cm.gray, interpolation="nearest",vmin = med-0.5*dsig, vmax =med+2*dsig)
       
        theta = np.arange(0,360)*(np.pi/180)
        rcos = lambda r, theta: annulusRadiusOuter + r*np.cos(theta)
        rsin = lambda r, theta: annulusRadiusOuter + r*np.sin(theta)
        ax.plot(rcos(apertureRadius,theta),rsin(apertureRadius,theta),'m',linewidth=4)
        ax.plot(rcos(annulusRadiusInner,theta),rsin(annulusRadiusInner,theta),'r',linewidth=4)
        ax.plot(rcos(annulusRadiusOuter,theta),rsin(annulusRadiusOuter,theta),'r',linewidth=4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Aperture')
        ax.set_xlim([-.5,dimx-.5])
        ax.set_ylim([-.5,dimy-.5])
        ax.format_coord = format_coord 
        plt.draw()
    # new feature for diagnostics
    if not returnsubtractedflux: 
        return [rawFlux, rawError, False]
    else: 
        if sigmaclipping: 
            return [rawFlux, rawError, False, clippedbackground]
        else: 
            return [rawFlux, rawError, False, np.median(imageCrop[skyIndices])]

def quadraticFit(derivative,ext):
    '''
    Find an extremum in the data and use it and the points on either side, fit
    a quadratic function to the three points, and return the x-position of the 
    apex of the best-fit parabola. 
    
    Called by oscaar.trackSmooth()
    
    Parameters
    ----------
    derivative : numpy.ndarray
       The first derivative of the series of points, usually calculated by np.diff()
                    
    ext : string 
        Extremum to look find. May be either "max" or "min"
    
    Returns
    -------
    extremum : float
        The (non-integer) index where the extremum was found
       
    '''
    rangeOfFit = 1
    lenDer = len(derivative)/2
    if ext == "max":
        indExtrema = np.argmax(derivative[:lenDer])
    elif ext == "min":
        indExtrema = np.argmin(derivative[lenDer:])+lenDer

    fitPart = derivative[indExtrema-rangeOfFit:indExtrema+rangeOfFit+1]
    if len(fitPart) == 3:
        stackPolynomials = np.zeros([3,3])
        for i in range(0,len(fitPart)):
            stackPolynomials[i,:] = [i**2,i,1.0]
        estimatedCoeffs = np.dot(LA.inv(stackPolynomials),fitPart)
        d_fit = -estimatedCoeffs[1]/(2.0*estimatedCoeffs[0])            #d_fit = -b_fit/(2.*a_fit)
        extremum = d_fit+indExtrema-rangeOfFit
    else: 
        extremum = indExtrema
    return extremum

def trackSmooth1D(image, est_x, smoothingConst, preCropped=False, zoom=20.0,plots=False):
    '''
    Method for tracking stellar centroids. 
    
    Parameters
    ---------- 
    image : numpy.ndarray
        FITS image read in by PyFITS

    est_x : float
        Inital estimate for the x-centroid of the star
    
    est_y : float
        Inital estimate for the y-centroid of the star
    
    smoothingConstant : float
        Controls the degree to which the raw stellar intensity profile will be smoothed by a Gaussian filter (0 = no smoothing)
    
    preCropped : bool
        If preCropped=False, image is assumed to be a raw image, if preCropped=True, image is assumed to be only the 
        portion of the image near the star
    
    zoom : int or float
        How many pixels in each direction away from the estimated centroid to consider when tracking the centroid. Be 
        sure to choose a large enough zoom value the stellar centroid in the next exposure will fit within the zoom
    
    plots : bool
        If plots=True, display stellar intensity profile in two axes and the centroid solution
                                
     Returns
     ------- 
     xCenter : float
         The best-fit x-centroid of the star

     yCenter : float
         The best-fit y-centroid of the star
     
     averageRadius : float
         Average radius of the SMOOTHED star in pixels
     
     errorFlag : bool
         Boolean corresponding to whether or not any error occured when running oscaar.trackSmooth(). If an 
         error occured, the flag is True; otherwise False.
                         
     Core developer: Brett Morris
     Modifications by: Luuk Visser, 2-12-2013
    '''
    '''If you have an interpolated grid as input, small inputs for smoothingConst
        it won't have any effect. Thus it has to be increased by the
        zoom factor you used to sub-pixel interpolate. 
        
        np.e seems to give nice smoothing results if frame is already cut out, you can 
        set preCropped to True, so the script won't cut a frame out again. '''
    #try:
    if preCropped:
        zoom = image.shape[0]/2
        #est_x, est_y = 0,0
        target = image ## Assume image is pre-cropped image of the star
    else:
        #smoothingConst *= zoom/20 
        target = image[est_x-zoom:est_x+zoom, :]   ## Crop image of just the target star
        
    #Save original (unsmoothed) data for plotting purposses

    target = ndimage.gaussian_filter(target, sigma=smoothingConst,order=0)
    
    ## Sum columns
#    axisA = np.sum(target,axis=0)   ## Take the sums of all values in each column,
    axisB = np.sum(target,axis=1)   ## then repeat for each row

 #   axisADeriv = np.diff(axisA)     ## Find the differences between each pixel intensity and
    axisBDeriv = np.diff(axisB)     ## the neighboring pixel (derivative of intensity profile)

#    lenaxisADeriv = len(axisADeriv)
#    lenaxisADeriv_2 = lenaxisADeriv/2
    lenaxisBDeriv = len(axisBDeriv)#    
    lenaxisBDeriv_2 = lenaxisBDeriv/2
    
#    derivMinAind = np.where(axisADeriv == min(axisADeriv[lenaxisADeriv_2:lenaxisADeriv]))[0][0] ## Minimum in the derivative
    derivMinBind = np.where(axisBDeriv == min(axisBDeriv[lenaxisBDeriv_2:lenaxisBDeriv]))[0][0] ## of the intensity plot

 #   derivMaxAind = np.where(axisADeriv == max(axisADeriv[0:lenaxisADeriv_2]))[0][0] ## Maximum in the derivative
    derivMaxBind = np.where(axisBDeriv == max(axisBDeriv[0:lenaxisBDeriv_2]))[0][0] ## of the intensity plot

#    extremumA = quadraticFit(axisADeriv,ext="max")
#    extremumB = quadraticFit(axisADeriv,ext="min")
    extremumC = quadraticFit(axisBDeriv,ext="max")
    extremumD = quadraticFit(axisBDeriv,ext="min")

#    averageRadius = (abs(derivMinAind-derivMaxAind)+ \
#        abs(derivMinBind-derivMaxBind))/4. ## Average diameter / 2
 #   axisAcenter = (extremumA+extremumB)/2.
    axisBcenter = (extremumC+extremumD)/2.
    
    xCenter = est_x-zoom+axisBcenter
#    yCenter = est_y-zoom+axisAcenter

    yCenter = 0
    if plots:
        fig = plt.figure()
        #print axisA
        #plt.plot(range(len(axisA)), axisA)
        plt.plot(range(len(axisB)), axisB)
        #plt.axvline(x=extremumA,ymin=0,ymax=1,color='b',linestyle=':',linewidth=1)
        #plt.axvline(x=extremumB,ymin=0,ymax=1,color='b',linestyle=':',linewidth=1)
        plt.axvline(x=extremumC,ymin=0,ymax=1,color='r',linestyle=':',linewidth=1)
        plt.axvline(x=extremumD,ymin=0,ymax=1,color='r',linestyle=':',linewidth=1)
#        plt.axvline(x=axisBcenter,ymin=0,ymax=1,color='r',linewidth=2)
       # plt.axvline(x=axisAcenter,ymin=0,ymax=1,color='b',linewidth=2)

        plt.show()
    return xCenter#, yCenter
#    except Exception:    ## If an error occurs:
#        print "An error has occured in oscaar.trackSmooth(), \n\treturning inital (x,y) estimate"
#        return [est_x, est_y, 1.0, True]
        



