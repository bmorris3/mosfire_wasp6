# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 08:56:57 2015

@author: bmmorris
"""

import sys
sys.path.append('/astro/users/bmmorris/Downloads/Fast_MA')
from ext_func.rsky import rsky
from ext_func.occultquad import occultquad
import numpy as np
import emcee
import george
from george import kernels
import os
import datetime 

# Path to chain outputs
scratchpath = '/astro/store/scratch/tmp/bmmorris/longchains/mosfirespitzer/'

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

def T14b2aRsi(P, T14, b):
    '''
    Convert from duration and impact param to a/Rs and inclination
    '''
    i = np.arccos( ( (P/np.pi)*np.sqrt(1 - b**2)/(T14*b) )**-1 )
    aRs = b/np.cos(i)
    return aRs, i

def aRsi2T14b(P, aRs, i):
    b = aRs*np.cos(i)
    T14 = (P/np.pi)*np.sqrt(1-b**2)/aRs
    return T14, b

def reparameterized_lc(T14, b, t0, q1, q2, p0, F0, e, w, period, eps, t):
    '''
    Reparameterization of the transit light curve in `get_lc()` with
    duration (first-to-fourth contact) instead of a/R* and impact
    parameter instead of inclination
    '''
    aRs, i = T14b2aRsi(period, T14, b)
    return get_lc(aRs, i, t0, q1, q2, p0, F0, e, w, period, eps, t)

def fine_lc(T, b, t0, q1, q2, p0, F0, e, w, period, eps, t, meantimediff):
    new_t = np.linspace(t.min() - 2*meantimediff, t.max() + 
                        2*meantimediff, 5*len(t))
    #return new_t, get_lc(aRs, i, t0, q1, q2, p0, F0, e, w, period, eps, new_t)
    return new_t, reparameterized_lc(T, b, t0, q1, q2, p0, F0, e, 
                                     w, period, eps, new_t)
    
def binned_lc(T, b, t0, q1, q2, RpOverRs, F0, am, eccentricity, 
              periapse, period, eps, t, meantimediff, airmassvector):
    exptime = t[1] - t[0]
    # If difference between exposures is less than 10 seconds, don't re-bin LC
    new_t, finemodel = fine_lc(T, b, t0, q1, q2, RpOverRs, 
                               F0, eccentricity, periapse, period, eps, t, 
                               meantimediff)

    timebinedges = np.sort(np.concatenate([t - 0.5*exptime, t + 0.5*exptime]))
    d = np.digitize(new_t, timebinedges)
    binned_model = np.array([np.mean(finemodel[d == i]) for i in 
                             range(1, 2*len(t), 2)])
    if airmassvector is None:
        return binned_model
    else:
        return binned_model*(1 + (airmassvector - 1)/am)   

                
def multiplyresiduals(residuals, constant=1e4):
    '''
    Multiply the residuals by a constant factor to encourage numerical
    stability inside of george
    '''
    return constant*residuals

class Fit(object):
    def __init__(self, genmodel, paramlimits, Nlightcurves, Nbins, **kwargs):
        self.gpwhite = kwargs.get('gpwhite', True)
        self.gpred = kwargs.get('gpred', True)
        self.fitlc = kwargs.get('fitlc', True)
        #self.fitresiduals = kwargs.get('fitlc', True)
        self.cosineperiod = kwargs.get('cosineperiod', None)
        self.genmodel = genmodel
        self.paramlimits = paramlimits
        self.Nlightcurves = Nlightcurves
        self.Nbins = Nbins
        
        # Check that inputs are good
        self.validateinputs()
        
        # Choose appropriate log likelihood method
        if self.gpred and not self.gpwhite:
            self.lnlike = self.lnlike_red 
        elif not self.gpred and self.gpwhite:
            self.lnlike = self.lnlike_white
        else: 
            self.lnlike = self.lnlike_chi2
        
    def validateinputs(self):
        '''
        Check that the inputs to the Fit() object make sense.
        '''
        if self.gpred:
            assert not self.cosineperiod is None,('Need cosineperiod argument'+
                'if computing red noise model')
        if self.gpred:
            assert not self.gpwhite,('If solving for red noise, white noise' +
                'must also be solved for')

    def lnlike_chi2(self, theta, MOSFIREtimes, ch1times, ch2times, y_mos, 
                    yerr_mos, y_ch1, yerr_ch1, y_ch2, yerr_ch2):
        '''
        Compute ln-likelihood using chi^2 rather than george
        '''
        mosfiremodel, ch1model, ch2model = self.genmodel(theta)

        lnlikelihoodsum = 0.0
    
        for i in range(self.Nlightcurves):
            # For MOSFIRE light curves:
            if i < self.Nbins:
                lnlikelihoodsum += -0.5*np.sum((y_mos[:,i] - 
                                        mosfiremodel[:,i])**2/yerr_mos[:,i]**2)
            
            # For Spitzer light curves:
            elif i == 8:
                lnlikelihoodsum += -0.5*np.sum((y_ch1 - 
                                                    ch1model)**2/yerr_ch1**2)
            elif i == 9:
                lnlikelihoodsum += -0.5*np.sum((y_ch2 - 
                                                    ch2model)**2/yerr_ch2**2)
        return lnlikelihoodsum        

    def lnlike_red(self, theta, MOSFIREtimes, ch1times, ch2times, y_mos, 
                     yerr_mos, y_ch1, yerr_ch1, y_ch2, yerr_ch2):
        '''
        Compute ln-likelihood using george with red noise described by
        the sum of (1) a white noise kernal and (2) the product of a square
        exponential and a cosine kernel.
        '''
        mosfiremodel, ch1model, ch2model = self.genmodel(theta)

        w = np.exp(theta[-self.Nlightcurves:])
        amp = np.exp(theta[-2*self.Nbins:-self.Nbins])
        sig = theta[-self.Nbins:]
        lnlikelihoodsum = 0.0
    
        for i in range(self.Nlightcurves):
            # For MOSFIRE light curves:
            if i < self.Nbins:
                gpobj = george.GP(kernels.WhiteKernel(w[i]) + 
                    amp[i]*kernels.ExpSquaredKernel(sig[i]) *
                    kernels.CosineKernel(self.cosineperiod))
                gpobj.compute(MOSFIREtimes)
                lnlikelihoodsum += gpobj.lnlikelihood(
                        multiplyresiduals(y_mos[:,i] - mosfiremodel[:,i]))
            # For Spitzer light curves:
            elif i == 8:
                gpobj = george.GP(kernels.WhiteKernel(w[i]))
                gpobj.compute(ch1times)
                lnlikelihoodsum += gpobj.lnlikelihood(
                    multiplyresiduals(y_ch1 - ch1model))
            elif i == 9:
                gpobj = george.GP(kernels.WhiteKernel(w[i]))
                gpobj.compute(ch2times)
                lnlikelihoodsum += gpobj.lnlikelihood(
                    multiplyresiduals(y_ch2 - ch2model))
        return lnlikelihoodsum        

    def lnlike_white(self, theta, MOSFIREtimes, ch1times, ch2times, y_mos, 
                     yerr_mos, y_ch1, yerr_ch1, y_ch2, yerr_ch2):
        '''
        Compute ln-likelihood using george with only white noise
        '''
        mosfiremodel, ch1model, ch2model = self.genmodel(theta)

        w = np.exp(theta[-self.Nlightcurves:])
        lnlikelihoodsum = 0.0
    
        for i in range(self.Nlightcurves):
            # For MOSFIRE light curves:
            if i < self.Nbins:
                gpobj = george.GP(kernels.WhiteKernel(w[i]))
                gpobj.compute(MOSFIREtimes)
                lnlikelihoodsum += gpobj.lnlikelihood(
                        multiplyresiduals(y_mos[:,i] - mosfiremodel[:,i]))
            
            # For Spitzer light curves:
            elif i == 8:
                gpobj = george.GP(kernels.WhiteKernel(w[i]))
                gpobj.compute(ch1times)
                lnlikelihoodsum += gpobj.lnlikelihood(
                    multiplyresiduals(y_ch1 - ch1model))
            elif i == 9:
                gpobj = george.GP(kernels.WhiteKernel(w[i]))
                gpobj.compute(ch2times)
                lnlikelihoodsum += gpobj.lnlikelihood(
                    multiplyresiduals(y_ch2 - ch2model))
        return lnlikelihoodsum        

    def lnprior(self, theta):
        '''
        Check that proposal step is acceptable.
        '''
        #for i, limits in enumerate(self.paramlimits):
        #    if not ((limits[0] < parameters[i]) and (parameters[i] < limits[1])):
        if not ((self.paramlimits[:,0] < theta).all() and 
                (theta < self.paramlimits[:,1]).all()):
            return -np.inf
        return 0.0
    
    def lnprob(self, theta, MOSFIREtimes, ch1times, ch2times, y_mos, yerr_mos, 
               y_ch1, yerr_ch1, y_ch2, yerr_ch2):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, MOSFIREtimes, ch1times, ch2times, y_mos, 
                                yerr_mos, y_ch1, yerr_ch1, y_ch2, yerr_ch2)

    def runemcee(self, pos, Nsteps, Nthreads, nwalkers, ndim, filename, args):
        #Nsteps = 1e6#int(Nhours*60*3.9) # 3.9 steps/min
        pool = emcee.interruptible_pool.InterruptiblePool(processes=Nthreads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, a=2, 
                                        pool=pool, args=args)
                                        
        print 'ndim =', ndim
        print 'nwalkers =', nwalkers
    
        print "Running initial burn in"
        p0, _, _ = sampler.run_mcmc(pos, 1, storechain=False)
        sampler.reset()
        
        thisfilename = os.path.split(filename)[-1].split('.py')[0]
        outputdir = scratchpath+thisfilename+datetime.datetime.now().strftime('%Y%m%d%H%M')+'/'
        os.mkdir(outputdir)
    
        #pos = [p0[i] + 1e-2*np.random.randn(len(initP)) for i in range(nwalkers)]
        print "Running production chains"
        print 'Start time:', datetime.datetime.now()
        
        labels = map(str,range(len(pos[0])))
        with open(outputdir+"chains.dat", "w") as f1: #iterations=500 -> 42 MB for raw text
            f1.write('#'+' '.join(labels)+'\n')
    
        with open(outputdir+"acceptance.dat", "w") as f2:
            f2.write('# acceptancefraction \n')
    
        with open(outputdir+"acor.dat", "w") as f3:
            f3.write('# autocorrelation \n')
    
        for result in sampler.sample(p0, iterations=Nsteps, storechain=False):
            print 'mean acceptance fraction: {0}'.format(np.mean(sampler.acceptance_fraction))
            with open(outputdir+"chains.dat", "a") as f1:
                for k in range(result[0].shape[0]):
                    f1.write("{0} {1} {2}\n".format(k, result[1][k], " ".join(map(str,result[0][k]))))
    
            with open(outputdir+"acceptance.dat", "a") as f2:
                f2.write('{0}\n'.format(' '.join(map(str, sampler.acceptance_fraction))))
            
            try: 
                with open(outputdir+"acor.dat", "a") as f3:
                    f3.write('{0}\n'.format(' '.join(map(str, sampler.acor))))
            except:
                pass
            
        print 'End time:', datetime.datetime.now()