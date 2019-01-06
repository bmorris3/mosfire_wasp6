
# coding: utf-8

## Sample image after `xshift`

# After nod subtraction, flat field correction, bad pixels (from [MOSFIRE webpage](https://code.google.com/p/mosfire/wiki/BadPIxelMask)) forced to zero

# In[1]:
#get_ipython().magic(u'matplotlib qt')
#get_ipython().magic(u'pylab inline')
import pyfits
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import photPack2
from astropy.time import Time

wasp6paths_nodsub_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_', \
                         i,'n.fits') for i in range(374,629,2)]
wasp6paths_sum_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_',\
                      i,'sum.fits') for i in range(374,629,2)]
testimgpath = wasp6paths_nodsub_odd[8]
testimg = pyfits.getdata(testimgpath)
arcpath = '/local/tmp/mosfire/2014sep18_analysis/m140918_0005shifted.fits'
arcimage = pyfits.getdata(arcpath)
wavelengthsoln = np.load('outputs/wavelengthsoln.npy')
#fig, ax = plt.subplots(1,2,figsize=(18,9))
#m = np.mean(testimg)
#s = np.std(testimg)
#ns = 0.15
#ax[0].imshow(testimg, origin='lower', interpolation='nearest', vmin=m-ns*s, vmax=m+ns*s, cmap='hot')#'brg'
#ax[0].set_title(testimgpath)
#ax[1].imshow(arcimage,             origin='lower', interpolation='nearest', vmin=m-ns*s, vmax=m+10*ns*s, cmap='hot')
#ax[1].set_title(arcpath)
## ax.set_xticks([])
## ax.set_yticks([])
#plt.show()

#plt.style.use('bmh')

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

# Expected transit time:
t0_expected = 2456918.887816  # JD
t0_roughfit = 2456918.8793039066
t14duration_expected = 0.1086 # days
#
#lightcurve = fluxes[:,1]/fluxes[:,0]
#
## Normalize light curve by out of transit flux
#oot = (times < t0_expected - t14duration_expected/2.0) +      (times > t0_expected + t14duration_expected/2.0)
#lightcurve /= np.mean(lightcurve[oot])
#

Nbins = 10
times = np.zeros(len(wasp6paths_nodsub_odd))
apertureradii = [28]#np.arange(20,40)#[39]#np.arange(30,45)#np.arange(10, 70, 5)#np.arange(14, 25, 2)
chisquared_allbins = np.zeros(len(apertureradii),dtype=float)
# Fluxes/errors dimensions: 
# N time series, N stars, N spectral bins, N apertures
fluxes = np.zeros((len(wasp6paths_nodsub_odd), 2, Nbins, len(apertureradii)))
errors = np.zeros_like(fluxes)
backgroundfluxes = np.zeros_like(fluxes)
centroids = np.zeros((len(wasp6paths_nodsub_odd), 2))
airmass = np.zeros(len(wasp6paths_nodsub_odd))
wavelengthbincenters = np.zeros(Nbins)
paddingbounds = [110, 2130]
spectralbinbounds = np.linspace(paddingbounds[0], paddingbounds[1], Nbins+1, dtype=int)

targetbounds = [385, 445]
compbounds = [1390, 1460]
roughnodcentroids = [500, 1500] # Rough indices between A and B nods
apertureradius = 28#20#8
bg_o = 5      # Background box outer limit
bg_i = 1.0#1.5#2.5    # Background box inner limit

#for i, imagepath in enumerate(wasp6paths_nodsub_odd[:1]):
for i, imagepath, imagesumpath in zip(range(len(wasp6paths_nodsub_odd)), wasp6paths_nodsub_odd, wasp6paths_sum_odd):
    image = pyfits.getdata(imagepath)[:,paddingbounds[0]:paddingbounds[1]]
    header = pyfits.getheader(imagepath)
    imagesum = pyfits.getdata(imagesumpath)[:,paddingbounds[0]:paddingbounds[1]]
    times[i] = Time('2014-09-18 '+header['UTC'], scale='utc', format='iso').jd
    airmass[i] = header['AIRMASS']
    
    for j in range(2):
        leftcentroid, rightcentroid = photPack2.trackStar(image, [roughnodcentroids[j]-250,                                                           roughnodcentroids[j]+250], 0,                                                           plots=False, returnCentroidsOnly=True)
        centroids[i, j] = leftcentroid
        for k in range(Nbins):
            binimage = image[:, spectralbinbounds[k]:spectralbinbounds[k+1]]
            binimagesum = imagesum[:, spectralbinbounds[k]:spectralbinbounds[k+1]]
            wavelengthbincenters[k] = np.mean([wavelengthsoln[spectralbinbounds[k]], wavelengthsoln[spectralbinbounds[k+1]]])
            #print leftcentroid, rightcentroid
            midnod = np.mean([leftcentroid, rightcentroid])
            background_upper = binimage[leftcentroid+bg_i*apertureradius:leftcentroid+bg_o*apertureradius,:]
            background_lower = binimage[leftcentroid-bg_o*apertureradius:leftcentroid-bg_i*apertureradius,:]
            #background_upper = binimage[leftcentroid+bg_i*apertureradius:midnod]#.ravel()
            #background_lower = binimage[:leftcentroid-bg_i*apertureradius]#.ravel()
            background = np.concatenate([background_upper, background_lower])#np.hstack([background_upper, background_lower])
            meanbackground = np.mean(background)
            rowprofile = np.sum(binimage[leftcentroid-10*apertureradius:leftcentroid+10*apertureradius,:],axis=1)
            withinaperture = binimage[leftcentroid-apertureradius:leftcentroid+apertureradius,:]
            withinaperture_sum = binimagesum[leftcentroid-apertureradius:leftcentroid+apertureradius,:]
            withinaperture_background = background_lower#background_upper

            if False:#j == 1:# and (i == 0 or i == 10):
                plt.plot(np.arange(len(rowprofile))+leftcentroid-10*apertureradius, rowprofile)
                plt.axhline(meanbackground*withinaperture.shape[1],                            color='k',ls='--')
                for v in [leftcentroid+bg_i*apertureradius, leftcentroid+bg_o*apertureradius,                          leftcentroid-bg_i*apertureradius, leftcentroid-bg_o*apertureradius]:
                    plt.axvline(v, ls=':', color='g')
                for v in [leftcentroid+apertureradius, leftcentroid-apertureradius]:
                    plt.axvline(v, ls=':', color='r')
                #plt.show()
            fluxes[i, j, k] = np.sum(withinaperture) - meanbackground*withinaperture.size
            errors[i, j, k] = np.sqrt(np.sum(withinaperture_sum) + meanbackground*withinaperture.size)
            backgroundfluxes[i, j, k] = np.sum(withinaperture_background)
    #plt.plot(np.sum(image[leftcentroid-apertureradius:leftcentroid+apertureradius,:],axis=1))


fig, ax = plt.subplots()
Nbins_hist = 50
ax.hist(backgroundfluxes[:,0,:,0].ravel()/1e7, Nbins_hist, histtype='stepfilled', alpha=0.4, label='Comp')
ax.hist(backgroundfluxes[:,1,:,0].ravel()/1e7, Nbins_hist, histtype='stepfilled', alpha=0.4, label='WASP-6')
ax.set_ylabel('Frequency')
ax.set_xlabel(r'Background intensity $\times 10^{-7}$')
ax.legend()
fig.savefig('plots/backgroundvariations.pdf')
plt.show()

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(backgroundfluxes[:,0,:,0]/1e7, fluxes[:,0,:,0]/1e7, '.')
ax[1].plot(backgroundfluxes[:,1,:,0]/1e7, fluxes[:,1,:,0]/1e7, '.')
for axes in ax: 
    axes.set_xlabel('Background Flux')
    axes.set_ylabel('Source Flux')
ax[0].set_title('All fluxes normalized by $10^{7}$')
fig.savefig('plots/backgroundtrends.pdf')
plt.show()


fig, ax = plt.subplots()
Nbins_hist = 50
ax.hist(backgroundfluxes[:,0,:,0].ravel()/fluxes[:,0,:,0].ravel(), Nbins_hist, histtype='stepfilled', alpha=0.4, label='Comp')
ax.hist(backgroundfluxes[:,1,:,0].ravel()/fluxes[:,1,:,0].ravel(), Nbins_hist, histtype='stepfilled', alpha=0.4, label='WASP-6')
ax.set_ylabel('Frequency')
#ax.set_xlabel(r'Background intensity $\times 10^{-7}$')

ax.set_xlabel(r'Background intensity')
ax.legend()
#fig.savefig('plots/backgroundvariations.pdf')
plt.show()


#
##print 'Out of transit stddev = %f' % (np.std(lightcurve[oot,0]))
#def chi2(v1, v2, err, Nfreeparams):
#    return np.sum( ((v1-v2)/err)**2 )/(len(v1) - Nfreeparams)
#
##fig.savefig('plots/lightcurve.pdf')
#
#### Quick fit
#import sys
#sys.path.append('/astro/users/bmmorris/Downloads/Fast_MA')
#from ext_func.rsky import rsky
#from ext_func.occultquad import occultquad
#def get_lc(e, aRs, i, u1, u2, p0, w, period, t0, eps, t):
#    '''
#    e - eccentricity
#    aRs - "a over R-star"
#    i - inclination angle in radians
#    u1, u2 - quadratic limb-darkening coeffs
#    p0 - planet to star radius ratio
#    w - argument of periapse
#    period - period
#    t0 - midtransit (JD)
#    eps - minimum eccentricity for Kepler's equation
#    t - time array
#    '''
#    r_s = 1.0
#    npoints = len(t)
#    z0 = rsky(e, aRs, i, r_s, w, period, t0, eps, t)   #calculates separation of centers between the planet and the star
#    mu_c = occultquad(z0, u1, u2, p0, npoints)   #returns limb darkened model lightcurve
#    return mu_c
#    
#from scipy import optimize
#aOverRs = 1./0.0932 # Jord`an et al 2013
#RpOverRs = 0.1404   # Jord`an et al 2013
#eccentricity = 0
#inclination = 88.47*np.pi/180
#u1 = 0.2
#u2 = 0.2
#periapse = np.pi/2
#period = 3.361006
#mineccentricity = 1.0e-7
#times_JD = times
#
#
#import matplotlib
#reload(matplotlib)
#
#for l, apertureradius in enumerate(apertureradii):
#    lightcurve = fluxes[:, 1, :, l]/fluxes[:, 0, :, l]
#    lightcurve_errors = lightcurve*np.sqrt((errors[:, 1, :, l]/fluxes[:, 1, :, l])**2 + (errors[:, 0, :, l]/fluxes[:, 0, :, l])**2)
#    oot = (times < t0_roughfit - t14duration_expected/2.0) +      (times > t0_roughfit + t14duration_expected/2.0)
#    
#    for eachbin in range(len(fluxes[0,0,:,0])):
#        lightcurve[:,eachbin] = lightcurve[:,eachbin]/np.mean(lightcurve[oot,eachbin])# + eachbin*0.02
#        lightcurve_errors[:,eachbin] = lightcurve_errors[:,eachbin]/np.mean(lightcurve[oot,eachbin])# + eachbin*0.02
#        
#    def quicklsfit(binindex):
#        def fitfunc(p):
#            #return get_lc(eccentricity, aOverRs, inclination, u1, u2, RpOverRs, periapse, period, t0, mineccentricity, times)
#            return get_lc(eccentricity, p[0], inclination, u1, u2, p[1], periapse, period, p[2], mineccentricity, times)   
#        
#        def errfunc(p):
#            return (fitfunc(p) - lightcurve[:,binindex])/lightcurve_errors[:,binindex]
#        
#        initp = [aOverRs, RpOverRs, t0_roughfit]
#        bestp = optimize.leastsq(errfunc, initp)[0]
#        return bestp, fitfunc(bestp)
#    def quicklsfit_bandintegrated():
#        def fitfunc(p):
#            #return get_lc(eccentricity, aOverRs, inclination, u1, u2, RpOverRs, periapse, period, t0, mineccentricity, times)
#            return p[3]*get_lc(eccentricity, p[0], inclination, u1, u2, p[1], \
#                   periapse, period, p[2], mineccentricity, times) + p[4]*airmass#/(1 + p[4]*airmass) #+ p[4]*airmass
#        
#        def errfunc(p):
#            return (fitfunc(p) - np.sum(lightcurve, axis=1)/len(lightcurve[0,:]))/np.sqrt(np.sum(lightcurve_errors**2, axis=1)/len(lightcurve[0,:]))
#        
#        initp = [aOverRs, RpOverRs, t0_roughfit, 1, 0.01]
#        bestp = optimize.leastsq(errfunc, initp)[0]
#        return bestp, fitfunc(bestp)
#    bestp_bandintegrated, bestfit_bandintegrated = quicklsfit_bandintegrated()
#    print 'bestp_bandintegrated',bestp_bandintegrated
##    fig, ax = plt.subplots(2, 1)
##    ax[0].plot(times, bestp_bandintegrated)
##    ax[0].plot(times, np.sum(lightcurve, axis=1)/len(lightcurve[0,:]), 'o')
##    ax[1].plot(times, np.sum(lightcurve, axis=1)/len(lightcurve[0,:]) - bestp_bandintegrated, 'o')
##    plt.show()
#    def quicklsfit_fixparams(binindex, bestp_bandintegrated=bestp_bandintegrated):
#        
#        def fitfunc(p, bestp_bandintegrated=bestp_bandintegrated):
#            #return get_lc(eccentricity, aOverRs, inclination, u1, u2, RpOverRs, periapse, period, t0, mineccentricity, times)
#            aOverRs_bandintegrated, RpOverRs_bandintegrated, t0_bandintegrated, scale_bandintegrated, airmass_bandintegrated = bestp_bandintegrated
#            return scale_bandintegrated*get_lc(eccentricity, p[0], inclination, u1, u2, p[1], \
#                   periapse, period, t0_bandintegrated, mineccentricity, times) + airmass_bandintegrated*airmass
#        
#        def errfunc(p):
#            return (fitfunc(p) - lightcurve[:,binindex])/lightcurve_errors[:,binindex]
#        
#        initp = [aOverRs, RpOverRs]#[aOverRs, RpOverRs, t0_roughfit]
#        bestp = optimize.leastsq(errfunc, initp)[0]
#        return bestp, fitfunc(bestp)
#
#    cmap = plt.cm.autumn
#    fig, ax = plt.subplots(1, figsize=(14,14))
#    mintimeint = int(np.min(times))
#    #ax.plot(times - mintimeint, lightcurve, '.')
#    lightcurve_allbins = np.zeros(len(times)*Nbins)
#    lightcurve_errors_allbins = np.zeros(len(times)*Nbins)
#    models_allbins = np.zeros(len(times)*Nbins)
#    for eachbin in range(len(lightcurve[0,:])):
#        #ax.errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.02, yerr=lightcurve_errors[:,eachbin], fmt='.')
#        #ax.errorbar((times - mintimeint)[np.invert(oot)], lightcurve[np.invert(oot),eachbin] + eachbin*0.02, \
#        #            yerr=lightcurve_errors[np.invert(oot),eachbin], fmt='.', color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
#        #ax.errorbar((times - mintimeint)[oot], lightcurve[oot,eachbin] + eachbin*0.02, \
#        #            yerr=lightcurve_errors[oot,eachbin], fmt='s', color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
#        ax.errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.02, \
#                    yerr=lightcurve_errors[:,eachbin], fmt='.', color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
#        bestp, bestfit = quicklsfit_fixparams(eachbin)#quicklsfit(eachbin)
#        print 'bestp',bestp
#        chisquared = chi2(lightcurve[:,eachbin], bestfit, lightcurve_errors[:,eachbin], 3)
#        lightcurve_allbins[eachbin*len(times):(eachbin+1)*len(times)] = lightcurve[:,eachbin]
#        lightcurve_errors_allbins[eachbin*len(times):(eachbin+1)*len(times)] = lightcurve_errors[:,eachbin]
#        models_allbins[eachbin*len(times):(eachbin+1)*len(times)] = bestfit
#        #fig, ax = plt.subplots(2,1,figsize=(12,10))
#        ax.plot(times - mintimeint,bestfit + eachbin*0.02, 'k', ls='--')
#        text = r'$\lambda = %.2f\mu m, \chi^2 = %.2f$' % (wavelengthbincenters[eachbin],\
#                                    chisquared)
#        ax.annotate(text, (np.max(times) - mintimeint + 2*np.mean(np.diff(times)), \
#                    1+eachbin*0.02),  xycoords='data', textcoords='data')
#    
#    chisquared_allbins[l] = chi2(lightcurve_allbins, models_allbins, lightcurve_errors_allbins, 2*Nbins)
#    ax.set_xlabel('JD - %d' % mintimeint)
#    ax.set_ylabel('Relative Flux')
#    ax.set_title('WASP-6 b  $\chi^2 = %.4f$' % chisquared_allbins[l])
#    
#    ax.grid()
#    fig.savefig('bins/bin%02d.pdf' % eachbin)
#    #plt.close()
#
#fig, ax = plt.subplots()
#ax.plot(apertureradii, chisquared_allbins)
#ax.set_xlabel('Aperture Radii')
#ax.set_ylabel('$\chi^2$')
#plt.show()
#for ap, c in zip(apertureradii, chisquared_allbins):
#    print 'Aperture=%2d   Chi^2=%2.4f' % (ap, c)
plt.show()

