# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:33:04 2015

@author: bmmorris
"""

import numpy as np
import triangle
from matplotlib import pyplot as plt

def splitchain(infile, outfile, tossfraction=0.9):
    '''
    Take the last `savefraction` of file `infile`, save it as the
    smaller file `outfile`.
    '''
    with open(outfile, 'w') as out:
        with open(infile, 'r') as f:
            alllines = f.readlines()
            lastXpercent = int(tossfraction*len(alllines))
            shortlines = alllines[lastXpercent:]
            out.write(''.join(shortlines))

def loadchains(directory, file='chains.dat', burnin=0.0):
    '''
    Load chains in `directory` saved to text file `file`, eliminate
    burn in fraction `burnin`
    '''

    chains = np.loadtxt(directory+file)
    burnin = int(burnin*chains.shape[0])
    lnp = chains[burnin:, 1]
    samples = chains[burnin:, 2:]
    return lnp, samples
   
class emceesamples(object):
    def __init__(self, samples, labels, dtypes, Nbins, Nlightcurves):
        '''
        Input the samples, output from loadchains(), labels for each parameter
        and data types for each parameter according to the following format: 
        
        'o' = orbital parameter
        'l' = (L) limb darkening
        't' = transit parameters particular to each spectral bin
        'w' = white noise hyperparameters
        'r' = red noise hyperparameters
        'a' = airmass
        'R' = radius
        'F' = out of transit flux
        
        '''
        
        self.samples = samples
        self.labels = labels
        self.dtypes = dtypes
        self.Nbins = Nbins
        self.Nlightcurves = Nlightcurves
        self.white = None
        self.red = None
        
        self.getld()
        self.getRpRs()
        self.getF0()
        self.getorb()
        self.getam()
        if 'w' in self.dtypes:
            self.getwhite()
        if 'r' in self.dtypes:
            self.getred()
        
    def getwhite(self):
        whiteinds = [i for i in range(len(self.dtypes)) 
                     if self.dtypes[i] == 'w']
        self.white = self.samples[:,whiteinds]
        self.whitelabels = len(whiteinds)*['w']

    def getred(self):
        redinds = [i for i in range(len(self.dtypes)) 
                     if self.dtypes[i] == 'r']
        self.red = self.samples[:,redinds]

    def getld(self):
        ldinds = [i for i in range(len(self.dtypes)) 
                     if self.dtypes[i] == 'l']
        self.ld = self.samples[:,ldinds]
        self.ldlabels = [label for i, label in enumerate(self.labels) 
                          if i in ldinds]     

    def getorb(self):
        orbinds = [i for i in range(len(self.dtypes)) 
                     if self.dtypes[i] == 'o']
        self.orb = self.samples[:,orbinds]
        self.orblabels = [label for i, label in enumerate(self.labels) 
                          if i in orbinds]     

    def getRpRs(self):
        RpRsinds = [i for i in range(len(self.dtypes)) 
                     if self.dtypes[i] == 'R']
        self.RpRs = self.samples[:,RpRsinds]
        self.RpRslabels = [label for i, label in enumerate(self.labels) 
                          if i in RpRsinds]     

    def getF0(self):
        F0inds = [i for i in range(len(self.dtypes)) 
                     if self.dtypes[i] == 'F']
        self.F0 = self.samples[:,F0inds]
        self.F0labels = [label for i, label in enumerate(self.labels) 
                          if i in F0inds]     

    def getam(self):
        aminds = [i for i in range(len(self.dtypes)) 
                     if self.dtypes[i] == 'a']
        self.am = self.samples[:,aminds]
        self.amlabels = [label for i, label in enumerate(self.labels) 
                          if i in aminds]     

    def triangles(self, directory=None, wavelengths=None, show=False):
        '''
        Create triangle plots. If directory is not None, save plots in that 
        directory.
        '''
        
        if wavelengths is None:
            wavelengths = np.arange(self.Nlightcurves)
        
        # Orbital parameters 
        Norbparams = len(self.orblabels)
        trifig1, ax = plt.subplots(Norbparams, Norbparams, figsize=(10, 10))
        kwargs = dict(fig=trifig1, plot_datapoints=False, 
                      labels=self.orblabels)
        fig1 = triangle.corner(self.orb, **kwargs) 
        trifig1.suptitle('Orbital Parameters', size=20)
        if directory is not None:
            trifig1.savefig(directory+'triangle_orbit.png',bbox_inches='tight')
        if not show:
            plt.clf()
        
        # Plot Limb darkening parameters
        for i in range(0, len(self.ldlabels), 2):
            trifigLD, ax = plt.subplots(2, 2, figsize=(6, 6))
            kwargs = dict(fig=trifigLD, plot_datapoints=False, 
                          labels=self.ldlabels[i:i+2])
            fig2 = triangle.corner(self.ld[:,i:i+2], 
                                   labelspace=False, **kwargs) 
            trifigLD.suptitle('LD Parameters', size=20)
            if directory is not None:
                trifigLD.savefig(directory+'triangle_ld{0}.png'.format(i/2),
                                bbox_inches='tight')
            if not show:
                plt.clf()
                
        # Plot Limb darkening parameters 
        for i in range(0, len(self.ldlabels), 2):
            trifigLD, ax = plt.subplots(2, 2, figsize=(6, 6))
            kwargs = dict(fig=trifigLD, plot_datapoints=False, 
                          labels=self.ldlabels[i:i+2])
            fig2 = triangle.corner(self.ld[:,i:i+2], 
                                   labelspace=False, **kwargs) 
            trifigLD.suptitle('LD Parameters', size=20)
            if directory is not None:
                trifigLD.savefig(directory+'triangle_ld{0}.png'.format(i/2),
                                bbox_inches='tight')
            if not show:
                plt.clf()                
                
                
        # Plot RpRs, F0, white noise
        for i in range(len(self.RpRslabels)):
            if i < self.Nbins:
                trifig, ax = plt.subplots(4, 4, figsize=(6, 6))
                kwargs = dict(fig=trifig, plot_datapoints=False, 
                              labels=[self.RpRslabels[i], self.F0labels[i],
                                      self.whitelabels[i], self.amlabels[i]])
                testsamples = np.vstack([self.RpRs[:,i],
                                         self.F0[:,i],
                                         self.white[:,i],
                                         self.am[:,i]]).T
            else:
                trifig, ax = plt.subplots(3, 3, figsize=(6, 6))
                kwargs = dict(fig=trifig, plot_datapoints=False, 
                              labels=[self.RpRslabels[i], self.F0labels[i],
                                      self.whitelabels[i]])
                testsamples = np.vstack([self.RpRs[:,i],
                                                  self.F0[:,i],
                                                  self.white[:,i]]).T

            fig2 = triangle.corner(testsamples, labelspace=True, **kwargs) 
            trifig.suptitle('{0:.3f}$\mu m$'.format(wavelengths[i]), size=20)
            if directory is not None:
                trifig.savefig(directory+'triangle_RpRs{0}.png'.format(i/2),
                                bbox_inches='tight')
            if not show:
                plt.clf() 
                
        if show:
            plt.show()
        else:
            plt.clf()
        
        
        
        
        
        