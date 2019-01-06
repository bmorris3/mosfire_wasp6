import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 15
from matplotlib import pyplot as plt

import moar
reload(moar)

plotmosfirephot = False


scratchpath = '/astro/store/scratch/tmp/bmmorris/longchains/mosfirespitzer/'


## Load photometry
from moar import load

ch1, ch2 = load.spitzer('/astro/users/bmmorris/git/research/keck/'+
             '2014september/analysis/bothnods/spitzer/thirdPLD/'+
             'wasp6_channel1_binned.ascii',
             '/astro/users/bmmorris/git/research/keck/2014september/analysis/'+
             'bothnods/spitzer/thirdPLD/wasp6_channel2_binned.ascii', 
             plots=False)

fluxes, errors, times, airmass, wavelengthbincenters, exposuredurs, wavelengthbounds = load.mosfire(
             '/astro/users/bmmorris/git/research/keck/2014september/analysis/'+
             'bothnods/photoutputs', plots=False)

lightcurve = fluxes[:, 1, :, 0]/fluxes[:, 0, :, 0]
lightcurve_errors = lightcurve*np.sqrt((errors[:, 1, :, 0]/
                    fluxes[:, 1, :, 0])**2 + (errors[:, 0, :, 0]/
                    fluxes[:, 0, :, 0])**2)

Nbins = np.shape(lightcurve)[1]
mintimeint = int(np.min(times))
cmap = plt.cm.autumn
if plotmosfirephot: 
    fig, ax = plt.subplots(1, figsize=(14,14))
    for eachbin in range(len(lightcurve[0,:])):
        ax.errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.02,
                     yerr=lightcurve_errors[:,eachbin], fmt='.', color=cmap(
                     1 - eachbin / float(Nbins)), ecolor='gray')
        ax.set_xlabel('JD - %d' % mintimeint)
        ax.set_ylabel('Relative Flux')
        ax.grid()
    plt.show()

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
Nlightcurves = Nbins + 2

gpon = True
gpred = False
if gpon:
    if gpred: 
        paramlimits = [[0.07, 0.11],  #aRs
                       [-0.5, 0.5],
                       [np.min(times), np.max(times)],
                       [np.min(times), np.max(times)]] + \
                       2*3*[[0, 1.0]] +\
                       Nlightcurves*[[0.0, 0.3]] + \
                       Nlightcurves*[[0.1, 10]] + \
                       Nbins*[[0.0, 10.0]] + Nlightcurves*[[-20, 20]] + \
                       Nbins*[[-20, 20]] + Nbins*[[0, 2]] # GP hyperparams

    else: 
        paramlimits = [[0.07, 0.11],  #aRs
                       [-0.5, 0.5],
                       [np.min(times), np.max(times)],
                       [np.min(times), np.max(times)]] + \
                       2*3*[[0, 1.0]] +\
                       Nlightcurves*[[0.0, 0.3]] + \
                       Nlightcurves*[[0.1, 10]] + \
                       Nbins*[[0.0, 10.0]] + Nlightcurves*[[-20, 20]]#+ Nlightcurves*[[-13, -6]]# +\
                       #Nbins*[[-15, -5]] + Nbins*[[0, 0.5]] # GP hyperparams

else:
    paramlimits = [[0.07, 0.11],  #aRs
                   [-0.5, 0.5],
                   [np.min(times), np.max(times)],
                   [np.min(times), np.max(times)]] + \
                   2*3*[[0, 1.0]] +\
                   Nlightcurves*[[0.0, 0.3]] + \
                   Nlightcurves*[[0.1, 10]] + \
                   Nbins*[[0.0, 10.0]]#+ Nlightcurves*[[-13, -6]]# +\
                   #Nbins*[[-15, -5]] + Nbins*[[0, 0.5]] # GP hyperparams

labels = ['T14', 'b', 't0mos', 't0spitz'] + 2*3*['LD'] + \
          Nlightcurves*['RpRs'] + Nlightcurves*['F0'] + Nbins*['am'] +\
          Nlightcurves*['GPw'] #+ Nbins*['GPamp'] + Nbins*['GPsig']
lastp = 0

mosfire_meantimediff = np.median(np.diff(times))
ch1_meantimediff = np.median(np.diff(ch1['t']))
ch2_meantimediff = np.median(np.diff(ch2['t']))

from moar.fitting import binned_lc
def genmodel(parameters, Nbins=Nbins):
    mosfiremodel = np.zeros_like(lightcurve)

    listparams = parameters.tolist()

    for eachbin in xrange(Nbins):
        mosfirelcparams = listparams[0:3] + listparams[4:6] +\
                     [parameters[10+eachbin], parameters[20+eachbin], 
                      np.exp(parameters[30+eachbin]), eccentricity,
                      periapse, period, 1e-7, times, mosfire_meantimediff] 
        
        mosfiremodel[:,eachbin] = binned_lc(*mosfirelcparams, 
                                            airmassvector=airmass)
    
    spitzeram = [np.e] # placeholder argument, ignored
    ch1lcparams = listparams[0:2] + [parameters[3]] + listparams[6:8] + \
                  listparams[18:19] + listparams[28:29] + spitzeram + \
                  [eccentricity, periapse, period, 1e-7, ch1['t'], 
                   ch1_meantimediff]
    
    ch2lcparams = listparams[0:2] + [parameters[3]] + listparams[8:10]  + \
                  listparams[19:20] + listparams[29:30] + spitzeram + \
                  [eccentricity, periapse, period, 1e-7, ch2['t'], 
                   ch2_meantimediff]

    ch1model = binned_lc(*ch1lcparams, airmassvector=None)
    ch2model = binned_lc(*ch2lcparams, airmassvector=None)
    
    return mosfiremodel, ch1model, ch2model


def medsig(vector):
    median, medupper, medlower =  np.percentile(vector, [16, 50, 84])
    upper, lower = medupper-median, median - medlower
    return median, np.mean([upper, lower])
    
## Load initial positions

from moar.fitting import multiplyresiduals

getinitposition = False
if getinitposition:
    ## Path from the phot_fit-tbMpinkSwhitePLD3_nogeorge run
    from moar import fitting
    path = '/astro/store/scratch/tmp/bmmorris/longchains/mosfirespitzer/'+\
            'phot_fit-tbMpinkSwhitePLD3_nogeorge201503111238/shortchains.dat'
    lastpos_extracolumns = fitting.getlastchainstate(path, 8)
    # Remove extraneous columns
    lastpos = lastpos_extracolumns[:,:-Nlightcurves-2*Nbins]
    
    ## Estimate the right values for the red noise parameters based on initP
    initP = lastpos[0]
    mosfiremodel, ch1model, ch2model = genmodel(initP)
    
    initws = np.zeros(Nlightcurves)
    for eachbin in range(Nlightcurves):
        if eachbin < Nbins:
            residuals = lightcurve[:,eachbin] - mosfiremodel[:,eachbin]
    
        elif eachbin == 8:
            residuals = ch1['f'] - ch1model
    
        elif eachbin == 9:
            residuals = ch2['f'] - ch2model
        
        initws[eachbin] = np.log(np.std(multiplyresiduals(residuals))**2)
    
    ## White noise for white+red runs
    #whiteparams = np.zeros((len(lastpos), len(initws)))
    #for i in range(len(lastpos)):
    #    proposedstep = np.ones_like(initws)*1e10
    #    while not ((proposedstep > paramlimits[-Nlightcurves-2*Nbins][0]).all() and 
    #               (proposedstep < paramlimits[-Nlightcurves-2*Nbins][1]).all()):
    #        proposedstep = initws + 0.1*np.random.randn(len(initws))
    #    whiteparams[i,:] = proposedstep
    
    if gpon:
        # White noise for white only runs
        whiteparams = np.zeros((len(lastpos), len(initws)))
        for i in range(len(lastpos)):
            proposedstep = np.ones_like(initws)*1e10
            while not ((proposedstep > paramlimits[-Nlightcurves][0]).all() and 
                       (proposedstep < paramlimits[-Nlightcurves][1]).all()):
                proposedstep = initws + 0.1*np.random.randn(len(initws))
            whiteparams[i,:] = proposedstep
        if not gpred:
            pos = np.hstack([lastpos, whiteparams])#, redparams])
        else: 
            ## Load results from phot_fit-mossqexpcos.ipynb for red noise parameters
            mossqexpcos = np.load('max_lnp_params_mossqexpcos.npy')
            redparams = np.zeros((len(lastpos), len(mossqexpcos[Nbins:])))
            for i in range(len(lastpos)):
                proposedstep = -1e10*np.ones_like(mossqexpcos[Nbins:])
                #while proposedstep[]
                while not ((proposedstep[:Nbins] > paramlimits[-2*Nbins][0]).all() and 
                           (proposedstep[:Nbins] < paramlimits[-2*Nbins][1]).all() and 
                           (proposedstep[Nbins:] > paramlimits[-Nbins][0]).all() and 
                           (proposedstep[Nbins:] < paramlimits[-Nbins][1]).all()):
                    #proposedstep = mossqexpcos[Nbins:] + np.concatenate([0.7*np.random.randn(Nbins), 
                    proposedstep = mossqexpcos[Nbins:] + np.concatenate([4*np.random.randn(Nbins), 
                                    0.1*np.random.randn(Nbins)])
                redparams[i,:] = proposedstep
            redparams[:,:Nbins] = np.log(multiplyresiduals(np.exp(redparams[:,:Nbins])))
            pos = np.hstack([lastpos, whiteparams, redparams])
    else: 
        pos = lastpos

else: 
    path = '/astro/store/scratch/tmp/bmmorris/longchains/mosfirespitzer/'+\
            'phot_fit-tbMpinkSwhitePLD3_mult_corrected201503171042/chains.dat'
    pos = fitting.getlastchainstate(path, 8)

# Initialize fitting object from moar package
#from moar.fitting import Fit
stimes = np.sort(times)
cosineperiod = 2*np.median(np.diff(stimes))

## For this test, we'll just fit white noise parameters: 
updatedpos = []
for p in pos:
    updatedpos.append(p[:-2*Nbins])
pos = np.array(updatedpos)

from moar import plots
plots.initialwalkers(pos, genmodel, Nbins, times, lightcurve, 
                   lightcurve_errors, ch1, ch2, period, t0_roughfit)

Nfreeparameters = pos.shape[1]

ndim = Nfreeparameters
nwalkers = pos.shape[0]

if False:
    Nsteps = 1e6
    Nthreads = 4
    from moar.fitting import runemcee
    print 'runemcee():'
    runemcee(genmodel, paramlimits, Nlightcurves, Nbins, pos, Nsteps, 
                 Nthreads, nwalkers, ndim, __file__,
                 (times, ch1['t'], ch2['t'], lightcurve, 
                       lightcurve_errors, ch1['f'], ch1['e'], ch2['f'], 
                       ch2['e']),gpwhite=True, 
                 gpred=False, fitlc=True, cosineperiod=cosineperiod)    



