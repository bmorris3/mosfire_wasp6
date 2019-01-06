# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:10:53 2015

@author: bmmorris
"""


import numpy as np
from matplotlib import pyplot as plt


def initialwalkers(pos, genmodel, Nbins, times, lightcurve, 
                       lightcurve_errors, ch1, ch2, period, t0_roughfit,
                       show=True):
    mosfiremodel, ch1model, ch2model = genmodel(pos[0])
    mintimeint = int(np.min(times))
    fig, ax = plt.subplots(1, 2, figsize=(14,14))
    cmap = plt.cm.autumn
    for eachbin in range(len(lightcurve[0,:])):
        ax[0].errorbar(times - mintimeint, lightcurve[:,eachbin] + eachbin*0.02,
                     yerr=lightcurve_errors[:,eachbin], fmt='.', 
                     color=cmap(1 - eachbin / float(Nbins)), ecolor='gray')
        ax[0].set_xlabel('JD - %d' % mintimeint)
        ax[0].set_ylabel('Relative Flux')
    
    for i, ch, model, phases in zip(range(2), [ch2, ch1], [ch2model, ch1model], 
                                    [-182, -180]):
        ax[1].errorbar(ch['t'] - t0_roughfit - phases*period, ch['f'] + i*0.02, 
                       yerr=ch['e'], fmt='.', color='k', ecolor='gray')
    
    for p in pos:
        mosfiremodel, ch1model, ch2model = genmodel(p)
    
        for eachbin in range(len(lightcurve[0,:])):
            ax[0].plot(times - mintimeint, mosfiremodel[:,eachbin] + 
                       eachbin*0.02, 'k', lw=1)
            
        for i, ch, model, phases in zip(range(2), [ch2, ch1], [ch2model, 
                                        ch1model], [-182, -180]):
            ax[1].plot(ch['t'] - t0_roughfit - phases*period, model + 
                       i*0.02, color='r', lw=1)
        
    ax[0].grid()
    ax[0].set_title('Init Params')
    
    if show:
        plt.show()

def lnpostprob(lnp, skipfactor=2, show=True, mode='nearest'):
    fig, ax = plt.subplots(figsize=(8,8))
    from scipy.ndimage import gaussian_filter1d
    ax.set_title('$\log \,p$')
    abbrv_lnp = lnp[::skipfactor]
    ax.plot(abbrv_lnp, 'k.', alpha=0.8)
    ax.plot(gaussian_filter1d(abbrv_lnp, 0.1*len(abbrv_lnp), mode=mode), 
            'r', lw=4)
    ax.plot(np.argmax(abbrv_lnp), np.max(abbrv_lnp), 'rs', markersize=10)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('$\log \,p$')
    [l.set_rotation(45) for l in ax.get_xticklabels()]
    print "Max ln(p): {0}".format(np.max(abbrv_lnp))
    if show:
        plt.show()
        

def acceptancerate(directory, show=True):
    acceptance = np.loadtxt(directory+'acceptance.dat')
    m, n = np.shape(acceptance)
    plt.plot(acceptance)
    plt.axhline(np.mean(acceptance[-1]), lw=3, ls='--', color='w')
    plt.ylabel('Acceptance Rate')
    plt.xlabel('Step')
    plt.title('$N_{{walkers}} = {0}$'.format(n))
    
    plt.ylim([-0.05, 1.05])
    #plt.ylim([0.1, 0.4])
    plt.annotate('$\mu_{{acc}} = {0:.3f}$'.format(np.mean(acceptance[-1])), 
                 (0.8, 0.8), textcoords='axes fraction', ha='right')
  
    if show:
        plt.show()