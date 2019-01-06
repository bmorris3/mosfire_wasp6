
# coding: utf-8

import pyfits
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import photPack2
from astropy.time import Time

#wasp6paths_nodsub_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_', \
#                         i,'n.fits') for i in range(374,629,2)]
#wasp6paths_sum_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_',\
#                      i,'sum.fits') for i in range(374,629,2)]
wasp6paths_nodsub_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_', \
                         i,'n_nobadpxl.fits') for i in range(374,629,2)]
wasp6paths_sum_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_',\
                      i,'sum_nobadpxl.fits') for i in range(374,629,2)]
testimgpath = wasp6paths_nodsub_odd[8]
testimg = pyfits.getdata(testimgpath)
arcpath = '/local/tmp/mosfire/2014sep18_analysis/m140918_0005shifted.fits'
arcimage = pyfits.getdata(arcpath)
wavelengthsoln = np.load('wavelengthsoln.npy')

times = np.zeros(len(wasp6paths_nodsub_odd))
fluxes = np.zeros((len(wasp6paths_nodsub_odd), 2))
centroids = np.zeros((len(wasp6paths_nodsub_odd), 2))
airmass = np.zeros(len(wasp6paths_nodsub_odd))

targetbounds = [385, 445]
compbounds = [1390, 1460]
roughnodcentroids = [500, 1500] # Rough indices between A and B nods
apertureradius = 18#20#8
bg_o = 5      # Background box outer limit
bg_i = 1.5#2.5    # Background box inner limit

### Preparing channelshift() and bad pixel detections from badpixelsearch.py
rowlimits = [5, 2030]
collimits = [5, 2044]
bestshiftspath = '/local/tmp/mosfire/2014sep18_analysis/bestxshifts.npy'
bestxshifts = np.load(bestshiftspath)
oversamplefactor = 1
def channelshift(image):
    ydim, xdim = image.shape
    outputpaddingwidth = np.ceil(np.max(bestxshifts)/oversamplefactor)
    outputpadding = np.zeros((ydim, outputpaddingwidth))
    paddedimage = np.hstack([outputpadding, image, outputpadding])

    for row in range(1, ydim):
        paddedimage[row] = np.roll(paddedimage[row], int(bestxshifts[row]/oversamplefactor))
    return paddedimage

rowlimits = [5, 2030]
collimits = [5, 2044]
shapeimg = pyfits.getdata('/local/tmp/mosfire/2014sep18/m140918_0005.fits')[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]

def channelshift_coords(x,y,dims=np.shape(shapeimg)):
    image = np.zeros(dims)
    image[y,x] = 1
    ydim, xdim = image.shape
    outputpaddingwidth = np.ceil(np.max(bestxshifts)/oversamplefactor)
    outputpadding = np.zeros((ydim, outputpaddingwidth))
    paddedimage = np.hstack([outputpadding, image, outputpadding])

    for row in range(1, ydim):
        paddedimage[row] = np.roll(paddedimage[row], int(bestxshifts[row]/oversamplefactor))
    XX, YY = np.meshgrid(range(ydim),range(xdim))
    #coords = (XX[paddedimage == 1][0], YY[paddedimage == 1][0])
    coords = (np.arange(ydim)[np.sum(paddedimage,axis=0) == 1][0], np.arange(xdim)[np.sum(paddedimage,axis=1) == 1][0])
    return coords

badpxls_x = []
badpxls_y = []
badpxls_exp = []
for basepath in ['badpxlseven/','badpxlsodd/']:
    badpxls_x.append(np.array(map(float, open(basepath+'variablepxls_x.csv').read().splitlines())))
    badpxls_y.append(np.array(map(float, open(basepath+'variablepxls_y.csv').read().splitlines())))
    badpxls_exp.append(np.array(map(float, open(basepath+'variablepxls_exp.csv').read().splitlines())))
    
badpxls_x = np.concatenate(badpxls_x)
badpxls_y = np.concatenate(badpxls_y)
badpxls_exp = np.concatenate(badpxls_exp)
meaningfulinds = badpxls_exp != 1e10
badpxls_x = badpxls_x[meaningfulinds]
badpxls_y = badpxls_y[meaningfulinds]
badpxls_exp = badpxls_exp[meaningfulinds]

#################################################################################
# Expected transit time:
t0_expected = 2456918.887816  # JD
t0_roughfit = 2456918.8793039066
t14duration_expected = 0.1086 # days

Nbins = 10
times = np.zeros(len(wasp6paths_nodsub_odd))
apertureradii = [28]#np.arange(20,40)#[39]#np.arange(30,45)#np.arange(10, 70, 5)#np.arange(14, 25, 2)
chisquared_allbins = np.zeros(len(apertureradii),dtype=float)
# Fluxes/errors dimensions: 
# N time series, N stars, N spectral bins, N apertures
fluxes = np.zeros((len(wasp6paths_nodsub_odd), 2, Nbins, len(apertureradii)))
errors = np.zeros_like(fluxes)
centroids = np.zeros((len(wasp6paths_nodsub_odd), 2))
airmass = np.zeros(len(wasp6paths_nodsub_odd))
wavelengthbincenters = np.zeros(Nbins)
paddingbounds = [110, 2130]
spectralbinbounds = np.linspace(paddingbounds[0], paddingbounds[1], Nbins+1, dtype=int)

targetbounds = [385, 445]
compbounds = [1390, 1460]
roughnodcentroids = [500, 1500] # Rough indices between A and B nods
#apertureradius = 18#20#8
bg_o = 5      # Background box outer limit
bg_i = 1.5#2.5    # Background box inner limit
badpixelclip = 6.0 #sigma
badpxlincore = 0
#for i, imagepath in enumerate(wasp6paths_nodsub_odd[:1]):
for i, imagepath, imagesumpath in zip(range(len(wasp6paths_nodsub_odd)), wasp6paths_nodsub_odd, wasp6paths_sum_odd):
    imagenameindex = int(imagepath.split('/')[-1].split('.')[0].split('_')[1].replace('n',''))
    print i,'of',len(wasp6paths_nodsub_odd)
    image = pyfits.getdata(imagepath)#[:,paddingbounds[0]:paddingbounds[1]]
    header = pyfits.getheader(imagepath)
    imagesum = pyfits.getdata(imagesumpath)[:,paddingbounds[0]:paddingbounds[1]]
    times[i] = Time('2014-09-18 '+header['UTC'], scale='utc', format='iso').jd
    airmass[i] = header['AIRMASS']
    
    ## Check for bad pixels in this exposure
    if imagenameindex in badpxls_exp:
        correction_inds = badpxls_exp == imagenameindex
 
        # Replace bad pixels with the median of the nearest 10 pixels in the channel
        # if the median turns out to be within 2stddevs of the median of 
        # the entire image: this will make sure not to correct bad pixels near the
        # core of the PSF. 
        oldimage = np.copy(image)
        allmedian = np.median(image)
        allstd = np.std(image)
        for y_badpxl,x_badpxl in zip(badpxls_x[correction_inds], badpxls_y[correction_inds]):
            window = 10
            plotbadpxl = False
            windowmedian = np.median(image[x_badpxl-window:x_badpxl+window, y_badpxl])

            # windowmedian will return nan for pixels near the edges of images
            print windowmedian, windowmedian, 0.5*allstd, imagepath
            if np.abs(windowmedian - windowmedian) < 0.5*allstd and not np.isnan(windowmedian):
                #oldimage = np.copy(image)
                image[x_badpxl,y_badpxl] = windowmedian
                plotbadpxl = False
            elif np.isnan(windowmedian):
                plotbadpxl = False
            else: 
                badpxlincore += 1
                plotbadpxl = True
                
            if plotbadpxl:
                imgm = np.median(oldimage[x_badpxl-window:x_badpxl+window, y_badpxl-window:y_badpxl+window])
                imgstd = np.std(oldimage[x_badpxl-window:x_badpxl+window, y_badpxl-window:y_badpxl+window])
                imgN = 0.5
                fig, ax = plt.subplots(1,2,figsize=(14,8), sharex=True, sharey=True)
                ax[0].imshow(oldimage, interpolation='nearest', origin='lower', \
                             vmin=imgm-imgN*imgstd, vmax=imgm+imgN*imgstd)            #ys = np.arange(y_badpxl-window,y_badpxl+window)
                ax[1].imshow(image, interpolation='nearest', origin='lower', \
                             vmin=imgm-imgN*imgstd, vmax=imgm+imgN*imgstd)
                for axes in ax:
                    axes.axvline(y_badpxl,lw=2,color='white')
                    axes.axhline(x_badpxl,lw=2,color='white')
                    axes.set_ylim([x_badpxl-window,x_badpxl+window])
                    axes.set_xlim([y_badpxl-window,y_badpxl+window])
                plt.show()
    
    # crop image:
    image = image[:,paddingbounds[0]:paddingbounds[1]]
    if i==0: 
        sumimg = np.zeros_like(image)
    sumimg += image
pyfits.writeto('tmp/dimsourcesum.fits', sumimg, clobber=True) 

'''
Next idea: what if you can zero-out all of the background noise in the
summed image. Then you could make a scaled version of the summed- and zeroed-
frame to find the flux from the dim source in each frame
'''
#h, edges = np.histogram(testimg, 1e5)
#x = [0.5*(edges[i]+edges[i+1]) for i in range(0,len(edges)-1)]
#plt.semilogy(x,h)
#plt.axvline(np.percentile(testimg, 99))
#plt.axvline(np.percentile(testimg, 1))
#plt.show()
smoothimg = np.copy(testimg)
m = np.median(smoothimg)
s = np.std(smoothimg)
N = 1.0
bg = np.abs(m - smoothimg) < N*s
smoothimg[bg] = 0
pyfits.writeto('tmp/smoothedsum.fits',smoothimg, clobber=True)


#'''
#New idea: convolve a gaussian (with sigma=N pixels) with the signal in the summed
#image over each channel individually. This way we avoid spectral blending 
#across channels (by not convolving in 2D)
#'''
#
##def gauss(x, center, sigma):
##    return 1./np.sqrt(2*np.pi)/sigma * np.exp(-0.5*((x-center)/sigma)**2)
##kernel = np.arange(10)
##gaussian = gauss(kernel, 5, 2)
#
#from scipy.ndimage import filters
##from scipy.integrate import cumtrapz
##filtered = filters.gaussian_filter1d(testchannel,2)
#
#smoothimg = np.copy(testimg)
#for i in range(len(testimg[0,:])):
#    smoothimg[:,i] = filters.gaussian_filter1d(smoothimg[:,i], 2)
#
#fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
#testchannel = testimg[:,200]
#smoothchannel = smoothimg[:,200]
#ax[0].plot(testchannel)
#ax[1].plot(smoothchannel)
#ax[0].set_title(np.trapz(testchannel))
#ax[1].set_title(np.trapz(smoothchannel))
#print np.trapz(testchannel)/np.trapz(smoothchannel)
#plt.show()
#plt.imshow(sumimg)
#plt.show()