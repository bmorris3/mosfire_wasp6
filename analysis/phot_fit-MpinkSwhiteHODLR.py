
# coding: utf-8

## Photometry and Fitting

# Begin with the same photometry routine as `phot_multiexposures.ipynb`, which recovers the first few exposures before ingress that had different exposure times.
#
# This version of `phot_fit` is being written at Keck Observing HQ on Dec 14 in an attempt to come up with a new parameterization for the light curve that properly includes in the airmass as a multiplicative term.

#In[1]:

# get_ipython().magic(u'pylab inline')
import pyfits
import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 15
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import photPack2
from astropy.time import Time
import emcee
import george
from george import kernels

dophotometry = False
dofitting = False


## Do photometry or load pre-calculated photometry

#In[2]:

## Read in raw data, organize
rawch1 = np.genfromtxt('spitzer/wasp6_channel1.txt')
rawch2 = np.genfromtxt('spitzer/wasp6_channel2.txt')

# Load initial fitting parameters from spitzer/spitzer-fit-results.ipynb
spitzerinitparams = np.load('spitzer/max_lnp_params_201501301021.npy')

ch1 = {}
ch2 = {}
for rawdata, output in zip([rawch1, rawch2], [ch1, ch2]):
    for i, key, offset in zip(range(3), ['t', 'f', 'e'], [2450000.0, 0.0, 0.0]):
        output[key] = rawdata[:,i] + offset

for ch in [ch1, ch2]:
    ch['e'] = np.zeros_like(ch['f']) + np.std(ch['f'][int(0.66*len(ch['f'])):])

fig, ax = plt.subplots(1,2,figsize=(16,10), sharey='row', sharex='col')
times = np.concatenate([ch2['t'], ch1['t']])
lightcurve = np.concatenate([ch2['f'], ch1['f']])
lightcurve_errors = np.concatenate([ch2['e'], ch1['e']])
for i, ch in enumerate([ch2, ch1]):
    ax[i].errorbar(ch['t'], ch['f'], yerr=ch['e'], fmt='.', color='k', ecolor='gray')


ax[0].set_ylabel('Relative Flux')
#plt.show()


#In[3]:

t0_roughfit = 2456918.879303906
period = 3.361006
print (ch2['t'].mean() - t0_roughfit)/period
print (ch1['t'].mean() - t0_roughfit)/period
print np.log(np.std(ch2['f'][0.7*len(ch2['t']):])**2)
print np.log(np.std(ch1['f'][0.7*len(ch1['t']):])**2)


#In[4]:

if dophotometry:
    print 'Calculating photometry'

    wasp6paths_nodsub_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_',                              i,'n_nobadpxl.fits') for i in range(365,629,1)]
    wasp6paths_sum_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_',                          i,'sum_nobadpxl.fits') for i in range(365,629,1)]
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
    # for basepath in ['badpxlsodd/','badpxlseven/']:
    #     badpxls_x = np.array(map(float, open(basepath+'variablepxls_x.csv').read().splitlines()))
    #     badpxls_y = np.array(map(float, open(basepath+'variablepxls_y.csv').read().splitlines()))
    #     badpxls_exp = np.array(map(float, open(basepath+'variablepxls_exp.csv').read().splitlines()))

    badpxls_x = np.concatenate(badpxls_x)
    badpxls_y = np.concatenate(badpxls_y)
    badpxls_exp = np.concatenate(badpxls_exp)
    meaningfulinds = badpxls_exp != 1e10
    badpxls_x = badpxls_x[meaningfulinds]
    badpxls_y = badpxls_y[meaningfulinds]
    badpxls_exp = badpxls_exp[meaningfulinds]

    #################################################################################

    ## Galaxy image
    galaxyimage = np.load('/astro/users/bmmorris/git/research/keck/2014september/analysis/rightnod/galaxy/wholeframegalaxyimg.npy')
    galaxyimagesum = np.load('/astro/users/bmmorris/git/research/keck/2014september/analysis/rightnod/galaxy/wholeframegalaxysum.npy')

    # Expected transit time:
    t0_expected = 2456918.887816  # JD
    t0_roughfit = 2456918.8793039066
    t14duration_expected = 0.1086 # days

    Nbins = 8
    paddingbounds = [210, 2130]
    spectralbinbounds = np.linspace(paddingbounds[0], paddingbounds[1], Nbins+1, dtype=int)

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
    wavelengthbounds = []
    exposuredurs = np.zeros(len(wasp6paths_nodsub_odd))

    # Additional bad pixel correction in core of PSF:
    badpxlmap = np.load('/astro/users/bmmorris/git/research/keck/2014september/analysis/rightnod/badpxlincoremap.npy')
    def correctbp(image, badpxlmap, plots=False, copyimage=True, medianwindow=3):
        props = {'cmap':cm.Greys_r, 'origin':'lower', 'vmin':-1.7e6, 'vmax':1.7e6, 'interpolation':'nearest'}
        dims = np.shape(badpxlmap)
        XX, YY = np.meshgrid(np.arange(dims[1]), np.arange(dims[0]))
        if copyimage:
            correctedimage = image.copy()
        else:
            correctedimage = image

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(10,10), sharex=True, sharey=True)
            ax[0].imshow(image, **props)
            ax[0].plot(XX[badpxlmap], YY[badpxlmap], 'rx')

        for x, y in zip(XX[badpxlmap], YY[badpxlmap]):
            correctedimage[y, x] = np.median(np.concatenate([image[y, x-medianwindow:x],
                                                             image[y, x+1:x+medianwindow]]))

        if plots:
            ax[1].imshow(correctedimage, **props)
            ax[1].plot(XX[badpxlmap], YY[badpxlmap], 'rx')
            plt.show(block=True)
        return correctedimage

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
        if i % 50 == 0: print i, 'of', len(wasp6paths_nodsub_odd)
        imagenameindex = int(imagepath.split('/')[-1].split('.')[0].split('_')[1].replace('n',''))
        image = pyfits.getdata(imagepath)#[:,paddingbounds[0]:paddingbounds[1]]
        header = pyfits.getheader(imagepath)
        imagesum = pyfits.getdata(imagesumpath)[:,paddingbounds[0]:paddingbounds[1]]
        times[i] = Time('2014-09-18 '+header['UTC'], scale='utc', format='iso').jd
        airmass[i] = header['AIRMASS']
        exposuredurs[i] = header['TRUITIME']


        ## Add in galaxy correction frame
        image += galaxyimage
        imagesum += galaxyimagesum[:,paddingbounds[0]:paddingbounds[1]]

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
                if x_badpxl-window < 0:
                    x_badpxl = window        # Correction January 13, 2015
                windowmedian = np.median(image[x_badpxl-window:x_badpxl+window, y_badpxl])

                # windowmedian will return nan for pixels near the edges of images
                #print windowmedian, windowmedian, 0.5*allstd, imagepath
                if not np.isnan(windowmedian):
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
                    ax[0].imshow(oldimage, interpolation='nearest', origin='lower',                                  vmin=imgm-imgN*imgstd, vmax=imgm+imgN*imgstd)            #ys = np.arange(y_badpxl-window,y_badpxl+window)
                    ax[1].imshow(image, interpolation='nearest', origin='lower',                                  vmin=imgm-imgN*imgstd, vmax=imgm+imgN*imgstd)
                    for axes in ax:
                        axes.axvline(y_badpxl,lw=2,color='white')
                        axes.axhline(x_badpxl,lw=2,color='white')
                        axes.set_ylim([x_badpxl-window,x_badpxl+window])
                        axes.set_xlim([y_badpxl-window,y_badpxl+window])
                    plt.show()

        # crop image:
        image = correctbp(image, badpxlmap)[:,paddingbounds[0]:paddingbounds[1]]

        #image = image[:,paddingbounds[0]:paddingbounds[1]]

        for j in range(2):
            #target star is j=1
            leftcentroid, rightcentroid = photPack2.trackStar(image, [roughnodcentroids[j]-250,                                                               roughnodcentroids[j]+250], 0,                                                               plots=False, returnCentroidsOnly=True)
            if i % 2 == 0:
                centroids[i, j] = rightcentroid#leftcentroid
            else:
                centroids[i, j] = leftcentroid

            for k in range(Nbins):
                binimage = image[:, spectralbinbounds[k]:spectralbinbounds[k+1]]
                binimagesum = imagesum[:, spectralbinbounds[k]:spectralbinbounds[k+1]]
                wavelengthbincenters[k] = np.mean([wavelengthsoln[spectralbinbounds[k]], wavelengthsoln[spectralbinbounds[k+1]]])
                wavelengthbounds.append([wavelengthsoln[spectralbinbounds[k]], wavelengthsoln[spectralbinbounds[k+1]]])
                #print leftcentroid, rightcentroid
                midnod = np.mean([leftcentroid, rightcentroid])
                for l, apertureradius in enumerate(apertureradii):
                    background_upper = binimage[centroids[i, j]+bg_i*apertureradius:centroids[i, j]+bg_o*apertureradius,:]
                    background_lower = binimage[centroids[i, j]-bg_o*apertureradius:centroids[i, j]-bg_i*apertureradius,:]
                    background = np.concatenate([background_upper, background_lower])#np.hstack([background_upper, background_lower])
                    meanbackground = np.mean(background)
                    rowprofile = np.sum(binimage[centroids[i, j]-10*apertureradius:centroids[i, j]+10*apertureradius,:],axis=1)
                    withinaperture = binimage[centroids[i, j]-apertureradius:centroids[i, j]+apertureradius,:]
                    withinaperture_sum = binimagesum[centroids[i, j]-apertureradius:centroids[i, j]+apertureradius,:]

                    withinaperture_corrected = np.copy(withinaperture)
                    medianwindow = 5
                    corr_x = []
                    corr_y = []
                    lastlength = 0
                    withinaperture = withinaperture_corrected

                    fluxes[i, j, k, l] = (np.sum(withinaperture) - meanbackground*withinaperture.size)/exposuredurs[i]
                    errors[i, j, k, l] = (np.sqrt(np.sum(withinaperture_sum) + meanbackground*withinaperture.size))/exposuredurs[i]
        #plt.plot(np.sum(image[leftcentroid-apertureradius:leftcentroid+apertureradius,:],axis=1))
    #plt.show()

    np.save('photoutputs/fluxes.npy', fluxes)
    np.save('photoutputs/errors.npy', errors)
    np.save('photoutputs/times.npy', times)
    np.save('photoutputs/airmass.npy', airmass)
    np.save('photoutputs/wavelengthbincenters.npy', wavelengthbincenters)
    np.save('photoutputs/exposuredurs.npy', exposuredurs)
    np.save('photoutputs/wavelengthbounds.npy', wavelengthbounds)
else:
    print 'Loading pre-calculated photometry'
    fluxes = np.load('photoutputs/fluxes.npy')
    errors = np.load('photoutputs/errors.npy')
    times = np.load('photoutputs/times.npy')
    airmass = np.load('photoutputs/airmass.npy')
    wavelengthbincenters = np.load('photoutputs/wavelengthbincenters.npy')
    exposuredurs = np.load('photoutputs/exposuredurs.npy')
    wavelengthbounds = np.load('photoutputs/wavelengthbounds.npy')


#### Plot photometry

#In[5]:

lightcurve = fluxes[:, 1, :, 0]/fluxes[:, 0, :, 0]
lightcurve_errors = lightcurve*np.sqrt((errors[:, 1, :, 0]/fluxes[:, 1, :, 0])**2 + (errors[:, 0, :, 0]/fluxes[:, 0, :, 0])**2)
Nbins = np.shape(lightcurve)[1]
#oot = (times < t0_roughfit - t14duration_expected/2.0) +      (times > t0_roughfit + t14duration_expected/2.0)
mintimeint = int(np.min(times))
cmap = plt.cm.autumn
fig, ax = plt.subplots(1, figsize=(14,14))
for eachbin in range(len(lightcurve[0,:])):
    ax.errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.02,                 yerr=lightcurve_errors[:,eachbin], fmt='.', color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
    ax.set_xlabel('JD - %d' % mintimeint)
    ax.set_ylabel('Relative Flux')
    ax.grid()
#plt.show()


## Set up MCMC fit

# Parameters to **link**: $a/R_s,\; i, \; t_0, \; u_1, \; u_2$
#
# Parameters to **float everywhere**: $R_p/R_s, \; F_0, \; c_X$
#
# Parameters to **lock**: $P, \; e, \;\omega$
#
# | Index | 0 | 1 | 2 | 3-4 | 5-6 | 7-8 | 9-19 | 20-30 | 30-38 |
# |---    |---|---|---|---  |---  |---  |---   |---    |---    |
# | Param | a/R_s | $i$ | $t_0$ | $q_{1,2}$ (mos) | $q_{1,2}$ (ch1) | $q_{1,2}$ (ch2) | $R_p/R_s$ | $F_0$ | am |

#In[6]:

import sys
sys.path.append('/astro/users/bmmorris/Downloads/Fast_MA')
from ext_func.rsky import rsky
from ext_func.occultquad import occultquad

def get_lc(aRs, i, t0, q1, q2, p0, F0, e, w, period, eps, t):
    '''
    e - eccentricity
    aRs - "a over R-star"
    i - inclination angle in radians
    u1, u2 - quadratic limb-darkening coeffs
    p0 - planet to star radius ratio
    w - argument of periapse
    period - period
    t0 - midtransit (JD)
    eps - minimum eccentricity for Kepler's equation
    t - time array
    '''
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1 - 2*q2)
    r_s = 1.0
    npoints = len(t)
    #calculates separation of centers between the planet and the star
    z0 = rsky(e, aRs, i, r_s, w, period, t0, eps, t)
    #returns limb darkened model lightcurve
    mu_c = occultquad(z0, u1, u2, p0, npoints)
    return F0*mu_c

from scipy import optimize
aOverRs = 1./0.0932 # Jord`an et al 2013
RpOverRs = 0.1404   # Jord`an et al 2013
eccentricity = 0.0 # Husnoo 2012
inclination = 88.47*np.pi/180
q1 = 0.00001
q2 = 0.2
periapse = np.pi/2 # To match e=0, from Husnoo 2012
period = 3.36100239 # Nikolov 2015          #3.361006
mineccentricity = 1.0e-7
t0_roughfit = 2456918.8793039066

Nbins = np.shape(lightcurve)[1]
print Nbins
Nlightcurves = Nbins + 2
#aRs, i, t0, RpRs, LD, F0, am
paramlimits = [[8.0, 14.0],  #aRs
               [85*np.pi/180, 95.0*np.pi/180],
               [np.min(times), np.max(times)],
               [np.min(times), np.max(times)]] + \
               2*3*[[0, 1.0]] +\
               Nlightcurves*[[0.0, 0.3]] + \
               Nlightcurves*[[0.1, 10]] + \
               Nbins*[[0.0, 10.0]] + Nlightcurves*[[-13, -6]] +\
               Nbins*[[-15, -5]] + Nbins*[[0, 0.5]] # GP hyperparams

# paramlimits = dict(
# aRs = [8.0, 14.0],
#Inc = [85*np.pi/180, 95.0*np.pi/180],
# t0mos = [np.min(times), np.max(times)],
# t0spitz = [np.min(times), np.max(times)],
# LD = 2*3*[[0, 1.0]],
# RpRs = Nlightcurves*[[0.0, 0.3]],
# F0 = Nlightcurves*[[0.1, 10]],
# am = Nbins*[[0.0, 10.0]],
# GPw = Nlightcurves*[[-13, -6]],
# GPamp = Nbins*[[-15, -5]],
# GPsig = Nbins*[[0, 0.5]])

labels = ['aRs', 'inc', 't0mos', 't0spitz'] + 2*3*['LD'] + Nlightcurves*['RpRs'] +          Nlightcurves*['F0'] + Nbins*['am'] + Nlightcurves*['GPw'] + Nbins*['GPamp'] + Nbins*['GPsig']
lastp = 0

mosfire_meantimediff = np.median(np.diff(times))
ch1_meantimediff = np.median(np.diff(ch1['t']))
ch2_meantimediff = np.median(np.diff(ch2['t']))

def fine_lc(aRs, i, t0, q1, q2, p0, F0, e, w, period, eps, t, meantimediff):
    new_t = np.linspace(t.min() - 2*meantimediff, t.max() + 2*meantimediff, 5*len(t))
    return new_t, get_lc(aRs, i, t0, q1, q2, p0, F0, e, w, period, eps, new_t)

def binned_lc(aOverRs, inclination, t0_roughfit, q1, q2, RpOverRs, F0, am, eccentricity,
              periapse, period, eps, t, meantimediff, airmassvector=airmass):
    new_t, finemodel = fine_lc(aOverRs, inclination, t0_roughfit, q1, q2, RpOverRs,
                               F0, eccentricity, periapse, period, eps, t, meantimediff)
    exptime = t[1] - t[0]
    timebinedges = np.sort(np.concatenate([t - 0.5*exptime, t + 0.5*exptime]))
    d = np.digitize(new_t, timebinedges)
    binned_model = np.array([np.mean(finemodel[d == i]) for i in range(1, 2*len(t), 2)])
    if airmassvector is None:
        return binned_model
    else:
        return binned_model*(1 + (airmassvector - 1)/am)


#In[7]:

print labels, len(labels)


#In[15]:

print np.finfo('d').eps


#In[16]:

def genmodel(parameters, Nbins=Nbins):
    mosfiremodel = np.zeros_like(lightcurve)

    listparams = parameters.tolist()

    for eachbin in xrange(Nbins):
        mosfirelcparams = listparams[0:3] + listparams[4:6] +                     [parameters[10+eachbin], parameters[20+eachbin], np.exp(parameters[30+eachbin]), eccentricity,                      periapse, period, 1e-7, times, mosfire_meantimediff] # Fixed params

        mosfiremodel[:,eachbin] = binned_lc(*mosfirelcparams)

    spitzeram = [np.e] # placeholder argument, ignored
    ch1lcparams = listparams[0:2] + [parameters[3]] + listparams[6:8] +                   listparams[18:19] + listparams[28:29] + spitzeram +                   [eccentricity, periapse, period, 1e-7, ch1['t'], ch1_meantimediff]

    ch2lcparams = listparams[0:2] + [parameters[3]] + listparams[8:10]  +                   listparams[19:20] + listparams[29:30] + spitzeram +                   [eccentricity, periapse, period, 1e-7, ch2['t'], ch2_meantimediff]

    ch1model = binned_lc(*ch1lcparams, airmassvector=None)
    ch2model = binned_lc(*ch2lcparams, airmassvector=None)

    return mosfiremodel, ch1model, ch2model

kernellist = []
gp_objs = []
stimes = np.sort(times)
cosineperiod = 2*np.median(np.diff(stimes))
GPerrorfactor = 10*np.finfo('d').eps
for i in range(Nlightcurves):
    # For MOSFIRE light curves:
    if i < Nbins:
        kernellist.append(kernels.WhiteKernel(np.exp(-10.5)) +
                      1e-3*kernels.ExpSquaredKernel(120)*kernels.CosineKernel(cosineperiod))
        gp_objs.append(george.GP(kernellist[i], solver=george.HODLRSolver))
        gp_objs[i].compute(times, GPerrorfactor)

    # For Spitzer light curves:
    elif i == 8:
        kernellist.append(kernels.WhiteKernel(np.exp(-10.5)))
        gp_objs.append(george.GP(kernellist[i], solver=george.HODLRSolver))
        gp_objs[i].compute(ch1['t'], GPerrorfactor)
    elif i == 9:
        kernellist.append(kernels.WhiteKernel(np.exp(-10.5)))
        gp_objs.append(george.GP(kernellist[i], solver=george.HODLRSolver))
        gp_objs[i].compute(ch2['t'], GPerrorfactor)

def lnlike(theta, y_mos, yerr_mos, y_ch1, yerr_ch1, y_ch2, yerr_ch2):
    mosfiremodel, ch1model, ch2model = genmodel(theta)
    w = np.exp(theta[-Nlightcurves-2*Nbins:-2*Nbins])
    amp = np.exp(theta[-2*Nbins:-Nbins])
    sig = theta[-Nbins:]
    lnlikelihoodsum = 0

    for i in range(Nlightcurves):
        # For MOSFIRE light curves:
        if i < Nbins:
            gp_objs[i].kernel.pars = [w[i], amp[i], sig[i]]
            lnlikelihoodsum += gp_objs[i].lnlikelihood(y_mos[:,i] - mosfiremodel[:,i])

        # For Spitzer light curves:
        elif i == 8:
            gp_objs[i].kernel.pars = [w[i]]
            lnlikelihoodsum += gp_objs[i].lnlikelihood(ch1['f'] - ch1model)
        elif i == 9:
            gp_objs[i].kernel.pars = [w[i]]
            lnlikelihoodsum += gp_objs[i].lnlikelihood(ch2['f'] - ch2model)
    return lnlikelihoodsum
#ma2)))

def lnprior(theta, paramlimits=paramlimits):
    parameters = theta
    # If parameter is locked, limits are set to [0,0]. If parameter is not locked,
    # check that all values for that parameter are within the set limits. If they are,
    # return 0.0, else return -np.inf
    for i, limits in enumerate(paramlimits):
        if not ((limits[0] < parameters[i]) and (parameters[i] < limits[1])):
            return -np.inf
    return 0.0

def lnprob(theta, y_mos, yerr_mos, y_ch1, yerr_ch1, y_ch2, yerr_ch2):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, y_mos, yerr_mos, y_ch1, yerr_ch1, y_ch2, yerr_ch2)


#labels = ['aRs', 'i', 't0_mos', 't0_spitzer'] + ['q1mos','q2mos'] + ['q1_ch1','q2_ch1'] + ['q1_ch2','q2_ch2'] +\
#         Nlightcurves*['RpRs'] + Nlightcurves*['F0'] + Nbins*['am'] + Nlightcurves*['w']

scattercoeffs = np.array([0.01, 0.005, 0.0001, 0.0001, 0.02, 0.02, 0.005, 0.01,
                       0.01, 0.01] + Nlightcurves*[0.0001] + Nlightcurves*[0.0001] +
                      Nbins*[0.15] + Nlightcurves*[0.15] + Nbins*[1.5] + Nbins*[0.04])
# Load recent run (from fit_results-spitzewhitekernelall?)
spitzwhitekernelall_params = np.load('/local/tmp/mosfire/longchains/mosfirespitzer/max_lnp_params_201503040921.npy')
mossqexpcos_params = np.load('/local/tmp/mosfire/longchains/mosfirespitzer/max_lnp_params_mossqexpcos.npy')

# Combine these old best-fits, using white kernel hyperparams from the sqexpcos fit
initP = np.concatenate([spitzwhitekernelall_params[:-Nlightcurves], # all parameters except GP hyperparams
                        mossqexpcos_params[:Nbins], # white kernel hyperparams for mosfire
                        spitzwhitekernelall_params[-2:], # white kernel hyperparams for spitzer
                        mossqexpcos_params[Nbins:]]) # sqexpcos params for mosfire
print len(initP), len(scattercoeffs)

Nfreeparameters = len(initP)

ndim = Nfreeparameters
nwalkers = 2*Nfreeparameters if 2*Nfreeparameters % 2 == 0 else 2*Nfreeparameters + 1

pos = []
while len(pos) < nwalkers:
    trial = initP + scattercoeffs*np.random.randn(len(scattercoeffs))
    if np.isfinite(lnprior(trial)):
        pos.append(trial)


#In[17]:

print map("{0:.5f}".format, pos[0])


#### Show initial parameters

#In[18]:

mosfiremodel, ch1model, ch2model = genmodel(initP)
fig, ax = plt.subplots(1, 2, figsize=(14,14))
for eachbin in range(len(lightcurve[0,:])):
    ax[0].errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.02,                 yerr=lightcurve_errors[:,eachbin], fmt='.', color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
    ax[0].plot(times - mintimeint, mosfiremodel[:,eachbin] + eachbin*0.02, 'k')
    ax[0].set_xlabel('JD - %d' % mintimeint)
    ax[0].set_ylabel('Relative Flux')



for i, ch, model, phases in zip(range(2), [ch2, ch1], [ch2model, ch1model], [-182, -180]):
    ax[1].errorbar(ch['t'] - t0_roughfit - phases*period, ch['f'] + i*0.02, yerr=ch['e'], fmt='.', color='k', ecolor='gray')
    ax[1].plot(ch['t'] - t0_roughfit - phases*period, model + i*0.02, color='r', lw=2)

ax[0].grid()
ax[0].set_title('Init Params')
#plt.show()


#In[ ]:

mosfiremodel, ch1model, ch2model = genmodel(initP)
fig, ax = plt.subplots(1, 2, figsize=(14,14))
for eachbin in range(len(lightcurve[0,:])):
    ax[0].errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.02,                 yerr=lightcurve_errors[:,eachbin], fmt='.', color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
    ax[0].set_xlabel('JD - %d' % mintimeint)
    ax[0].set_ylabel('Relative Flux')

for i, ch, model, phases in zip(range(2), [ch2, ch1], [ch2model, ch1model], [-182, -180]):
    ax[1].errorbar(ch['t'] - t0_roughfit - phases*period, ch['f'] + i*0.02, yerr=ch['e'], fmt='.', color='k', ecolor='gray')

for p in pos:
    mosfiremodel, ch1model, ch2model = genmodel(p)

    for eachbin in range(len(lightcurve[0,:])):
        ax[0].plot(times - mintimeint, mosfiremodel[:,eachbin] + eachbin*0.02, 'k', lw=1)

    for i, ch, model, phases in zip(range(2), [ch2, ch1], [ch2model, ch1model], [-182, -180]):
        ax[1].plot(ch['t'] - t0_roughfit - phases*period, model + i*0.02, color='r', lw=1)


ax[0].grid()
ax[0].set_title('Init Params')
#plt.show()


#In[ ]:

if True:
    Nhours = 48
    Nsteps = int(Nhours*60*3.9) # 3.9 steps/min
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4,
                                    args=(lightcurve, lightcurve_errors, ch1['f'], ch1['e'], ch2['f'], ch2['e']))

    #y_mos, yerr_mos, y_ch1, yerr_ch1, y_ch2, yerr_ch2

    print 'ndim =', ndim
    print 'nwalkers =', nwalkers

    print "Running initial burn in"
    p0, _, _ = sampler.run_mcmc(pos, 5, storechain=False)
    sampler.reset()

    #pos = [p0[i] + 1e-2*np.random.randn(len(initP)) for i in range(nwalkers)]
    print "Running production chains"
    import datetime
    print 'Start time:', datetime.datetime.now()

    #p0, _, _ = sampler.run_mcmc(p0, Nsteps)
    chainpath = '/local/tmp/mosfire/longchains/mosfirespitzer/'
    f = open(chainpath+"MpinkSwhiteHODLR.dat", "w") #iterations=500 -> 42 MB for raw text
    f.write('#'+' '.join(labels)+'\n')
    f.close()
    for result in sampler.sample(p0, iterations=Nsteps, storechain=False):
        f = open(chainpath+"MpinkSwhiteHODLR.dat", "a")
        for k in range(result[0].shape[0]):
            f.write("{0} {1} {2}\n".format(k, result[1][k], " ".join(map(str,result[0][k]))))
        f.close()
    print 'End time:', datetime.datetime.now()


#In[27]:

burninfraction = 0.3
samples = sampler.chain[:, burninfraction*Nsteps:, :].reshape((-1, ndim))

#np.save('thirdchain20141210.npy', samples[::50,:])

import triangle

# trifig, ax = plt.subplots(Nfreeparameters, Nfreeparameters, figsize=(16, 16))
# fig2 = triangle.corner(samples[:, :], labels=labels,
#                        fig=trifig, plot_datapoints=False) # truths=[t0_expected, aOverRs, RpOverRs, scale, 1]
# plt.show()

def medplusminus(vector):
    return map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(vector, [16, 50, 84])))

print np.shape(samples)
for i, l in enumerate(labels):#range(len(samples[0,:])):
    v = np.percentile(samples[:,i], [16, 50, 84])
    print l, v[1], v[2]-v[1], v[1]-v[0]

# for p, l in zip(lastp, labels):
#     print l, p


#In[7]:

# model = genmodel(lastp)
# fig, ax = plt.subplots(1, figsize=(14,14))
# for eachbin in range(len(lightcurve[0,:])):
#     ax.errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.02, \
#                 yerr=lightcurve_errors[:,eachbin], fmt='.', color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
#     ax.plot(times - mintimeint, model[:,eachbin])
#     ax.set_xlabel('JD - %d' % mintimeint)
#     ax.set_ylabel('Relative Flux')
#     ax.grid()
# plt.show()


#In[ ]:

fig, ax = plt.subplots(1, figsize=(8,8))
Nhistbins = 50
sampleind = 10
n, edges = np.histogram(samples[:,sampleind], Nhistbins)
x = np.array([0.5*(edges[i] + edges[i+1]) for i in range(len(edges) - 1)])
x *= 180./np.pi if sampleind == 1 else 1.0

ax.plot(x, n)
ax.set_title(labels[sampleind])
#plt.show()


#In[ ]:

RpRs = samples[:,3:3+Nbins]
print np.shape(RpRs)
print np.median(RpRs, axis=0)#np.percentile(RpRs, 50, axis=0)

plt.plot(np.percentile(RpRs, 50, axis=0), color='k', lw=2)
plt.fill_between(range(Nbins),np.percentile(RpRs, 16, axis=0), np.percentile(RpRs, 84, axis=0), color='k', alpha=0.3)
#plt.show()


#In[ ]:

model = genmodel(np.mean(p0,axis=0))
fig, ax = plt.subplots(1, figsize=(14,14))
for eachbin in range(len(lightcurve[0,:])):
    ax.errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.025,                 yerr=lightcurve_errors[:,eachbin], fmt='.', color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
    ax.plot(times - mintimeint, model[:,eachbin]+ eachbin*0.025)
    ax.set_xlabel('JD - %d' % mintimeint)
    ax.set_ylabel('Relative Flux')
    ax.grid()
#plt.show()


#In[5]:

pwd


#In[27]:

#spectralbinbounds = np.linspace(paddingbounds[0], paddingbounds[1], Nbins+1, dtype=int)
#print spectralbinbounds

firstlines = '''
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical
'''

# Box format:
# centerx centery widthx widthy rot
with open('binregions.reg','w') as reg:
    for i in range(len(spectralbinbounds)-1):
        centerx =  0.5*(spectralbinbounds[i] + spectralbinbounds[i+1])
        centery = 2024/2
        widthx = spectralbinbounds[i+1] - spectralbinbounds[i]
        widthy = 2024
        angle = 0
        linewidth = 3
        wavelength = wavelengthbincenters[i]
        reg.write("box({0:f},{1:f},{2:f},{3:f},{4:f}) # width={5} text={{{6:.3f}}} \n".format(
                  centerx, centery, widthx, widthy, angle, linewidth, wavelength))


#In[ ]:



