# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:55:19 2014

@author: bmmorris
"""
import numpy as np
from matplotlib import pyplot as plt

########################################
## Input light curve parameters
## http://adsabs.harvard.edu/abs/2014ApJ...784...44L
aOverRs = 1./0.0932 # Jord`an et al 2013
RpOverRs = 0.1404   # Jord`an et al 2013
eccentricity = 0.054
inclination = 88.47*np.pi/180
q1 = 0.00001
q2 = 0.2
periapse = np.pi/2
period = 3.361006
mineccentricity = 1.0e-7
t0_roughfit = 2456918.8793039066


u1 = 0.2
u2 = 0.2
periapse = np.pi/2
mineccentricity = 1.0e-7
t0 = t0_roughfit
b = aOverRs * np.cos(inclination)
T = period*np.sqrt(1-b**2)/(np.pi*aOverRs)
#aOverRs = period*np.sqrt(1-b**2)/(np.pi*T)

########################################
## Input data
from glob import glob
inputfiles = sorted(glob('../spitzer/wasp6_channel?.txt'))
#inputfiles += ['/astro/users/bmmorris/git/research/koi351/spitzertransits/p0spitzer.txt']
#inputfiles = ()
Ntransits = len(inputfiles)

transitdata = []
exposuredurations = []
for eachfile in inputfiles:
    rawdata = np.loadtxt(eachfile, skiprows=1)
    times, lc = rawdata[:,0], rawdata[:,1]
    times += 22450000.0
    #if np.mean(np.diff(times))*24*60 > 5:
    #    exposuredurations.append(1766./60)
    #else: 
    #    exposuredurations.append(58.9/60)
    exposuredurations.append(np.mean(np.diff(times))*24*60)
    transitdata.append([times, lc]) 

# Separate technique for the Spitzer light curve
#for eachfile in inputfiles[-1:]:
#    rawdata = np.loadtxt(eachfile, skiprows=1)
#    times, lc = rawdata[:,0], rawdata[:,1]
#    times += 0
#    exposuredurations.append(30.0/60)
#    transitdata.append([times, lc]) 



### Quick fit
import sys
sys.path.append('/astro/users/bmmorris/Downloads/Fast_MA')
from ext_func.rsky import rsky
from ext_func.occultquad import occultquad
def get_lc(e, aRs, i, u1, u2, p0, w, period, t0, eps, t):
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
    r_s = 1.0
    npoints = len(t)
    z0 = rsky(e, aRs, i, r_s, w, period, t0, eps, t)   #calculates separation of centers between the planet and the star
    mu_c = occultquad(z0, u1, u2, p0, npoints)   #returns limb darkened model lightcurve
    return mu_c

def convolved_lc(e, aRs, i, u1, u2, p0, w, period, t0, eps, t, exposureduration):
    originaltimes = t
    fine_times = np.linspace(t.min(), t.max(), 10*len(t))
    fine_lc = get_lc(e, aRs, i, u1, u2, p0, w, period, t0, eps, fine_times)
#    subsampled = binned_statistic(originaltimes, fine_lc)
#    meantimediff = np.mean(np.diff(x))
#    binedges = [i-0.5*meantimediff for i in range(len(x)+1)]
#    bincenters = [0.5*(binedges[i-1]+binedges[i]) for i in range(1,len(binedges))]
    convolved_lc = np.zeros(len(originaltimes))
    exptime = exposureduration/(60*24) #integration time in minutes->days
    for i, t_original in enumerate(originaltimes):
        withintimebin = (fine_times < t_original + 0.5*exptime)*\
                        (fine_times > t_original - 0.5*exptime)
        convolved_lc[i] = np.mean(fine_lc[withintimebin])
    return convolved_lc

def writetapfile(times, lightcurve, filename):
    spacer = ''
    f = open(filename, 'w')
    for time, flux in zip(times, lightcurve):
        f.write('%s%16.7f%s%16.7f\n' % (spacer, time, spacer, flux))
    f.close()

def writelightcurve(transitindex, Ntransits, times, lightcurve):
    spacer = ''
    if transitindex == 1:
        outputstring = '# transit{0:6d}'.format(transitindex)+'\n'
    else: 
        outputstring = ''
    for time, flux in zip(times, lightcurve):
        outputstring += '%s%16.7f%s%16.7f\n' % (spacer, time, spacer, flux)    
    if transitindex - 1 != Ntransits - 1:
        outputstring += '-1 -1\n'
    return outputstring

firstheaderlines = '''# TAP combination of setup parameters and light curves.  Adjust parameter setup matrix to
#  change the setup before loading this file as a "Transit File" in a new instance of TAP
#
'''
parametersheader = "#  transit  set               param                        value lock                      low_lim                       hi_lim  limited   prior=1_penalty=2            val            sig"
def writeoversampling(transitindex, exposureduration):
    oversampling = "#  transit  long_int    int_length_min  cadence_multiplier\n#{0:9d}         1{1:15.4f}             10\n#\n#\n".format(transitindex, exposureduration)
    return oversampling

def writeparams(transitindex, settoindex, period, b, RpRs, MidTransit, LinearLD, QuadraticLD, eccentricity, omega, F_oot, times, RpRsindex):
    duration = period*np.sqrt(1-b**2)/(np.pi*aOverRs)
    dataduration = times.max() - times.min()
    lowert0limit = times.min() - dataduration
    uppert0limit = times.max() + dataduration
    meant0limit = times.min() + 0.2*dataduration
    u1 = LinearLD
    u2 = QuadraticLD
    q1 = (u1+u2)**2
    q2 = 0.5*u1/(u1+u2)
    parameters = {
#                'transit': transitindex, \
#                'set': settoindex, \
                'Period': \
                    {'transit':transitindex, 'set':1, 'value': period, 'lock':1, 'low_lim':1e-1*period, 'hi_lim':1e1*period, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'b': \
                    {'transit':transitindex, 'set':1, 'value': b, 'lock':0, 'low_lim':-1, 'hi_lim':1, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'T': \
                    {'transit':transitindex, 'set':1, 'value': duration, 'lock':0, 'low_lim':1e-1*duration, 'hi_lim':1e1*duration, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Rp/R*': \
                    {'transit':transitindex, 'set':RpRsindex, 'value': RpRs, 'lock':0, 'low_lim':0.001, 'hi_lim':0.5, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Mid_Transit': \
                    {'transit':transitindex, 'set':settoindex, 'value': meant0limit, 'lock':0, 'low_lim':lowert0limit, 'hi_lim':uppert0limit, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Linear_LD': \
                    {'transit':transitindex, 'set':RpRsindex, 'value': q1, 'lock':0, 'low_lim':0, 'hi_lim':1, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Quadratic_LD': \
                    {'transit':transitindex, 'set':RpRsindex, 'value': q2, 'lock':0, 'low_lim':0, 'hi_lim':1, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Eccentricity': \
                    {'transit':transitindex, 'set':1, 'value': 0, 'lock':1, 'low_lim':0, 'hi_lim':1, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Omega': \
                    {'transit':transitindex, 'set':1, 'value': 0, 'lock':1, 'low_lim':0, 'hi_lim':2*np.pi, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'OOT_t^0': \
                    {'transit':transitindex, 'set':settoindex, 'value': F_oot, 'lock':0, 'low_lim':0.5, 'hi_lim':10.0, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'OOT_t^1': \
                    {'transit':transitindex, 'set':settoindex, 'value': 0, 'lock':0, 'low_lim':-0.01, 'hi_lim':0.01, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'OOT_t^2': \
                    {'transit':transitindex, 'set':settoindex, 'value': 0, 'lock':1, 'low_lim':-0.001, 'hi_lim':0.001, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Sigma_Red': \
                    {'transit':transitindex, 'set':settoindex, 'value': 0, 'lock':0, 'low_lim':0, 'hi_lim':1, 'limited':0, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Sigma_White': \
                    {'transit':transitindex, 'set':settoindex, 'value': 0.001, 'lock':0, 'low_lim':0, 'hi_lim':1, 'limited':0, 'priorpenalty':0, 'val':0, 'sig':0}, \
                'Delta_Light': \
                    {'transit':transitindex, 'set':settoindex, 'value': 0, 'lock':1, 'low_lim':0, 'hi_lim':1, 'limited':1, 'priorpenalty':0, 'val':0, 'sig':0}, \
                }
    outputstring = ''
    for row in parameters:
#        formattedline = '#{transit:9d}{set:5d}{Period:>20s}{b:>29.10f}{T:}{Rp/R*:}{Mid_Transit:}{Linear_LD:}{Quadratic_LD:}{Eccentricity:}{Omega:}{OOT_t^0:}{OOT_t^1:}{OOT_t^2:}{Sigma_Red:}{Sigma_White:}{Delta_Light:}'
        parameters[row]['param'] = row
        formattedline = ('#{transit:>9d}{set:>5d}{param:>20s}{value:>29.10f}'+\
                         '{lock:>5d}{low_lim:>29.10f}{hi_lim:>29.10f}{limited:>9d}'+\
                         '{priorpenalty:>20d}{val:15.5f}{sig:15.5f}\n').format(**parameters[row])
        outputstring += formattedline
    return outputstring

preamble = firstheaderlines
lightcurvesbody = ''
transitparams = parametersheader+'\n'
for transit in range(Ntransits):
    times, lc = transitdata[transit]
    exposureduration = exposuredurations[transit]
    preamble += writeoversampling(transit+1, exposureduration)
    F_oot = np.median(lc)
    settoindex = 1
#    if transit < Ntransits - 1: 
#        RpRsindex = 1
#    else:
    RpRsindex = Ntransits
    transitparams += writeparams(transit+1, transit+1, period, b, RpOverRs, t0,\
                                 u1, u2, eccentricity, periapse, F_oot, times, RpRsindex)
    #transitparams += writeparams(transit+1, settoindex, period, b, RpOverRs, t0, u1, u2, eccentricity, periapse, F_oot)
    lightcurvesbody += writelightcurve(transit+1, Ntransits, times, lc)
    plt.plot(times, lc, '.')
#    plt.plot(times, convolved_lc(eccentricity, aOverRs, \
#                inclination, u1, u2, RpOverRs, periapse, \
#                period, t0, mineccentricity, times[currenttransit]))

finaloutput = preamble+transitparams+lightcurvesbody
f = open(__file__.split('.')[0]+'_spitzer.ascii','w')
f.write(finaloutput)
f.close()
plt.show()


