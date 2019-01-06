# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 08:56:57 2015

@author: bmmorris
"""
import numpy as np
from matplotlib import pyplot as plt

def gelmanrubin(samples, **kwargs):
    '''
    The Gelman-Rubin (1992) statistic R-hat.
    
    Parameters
    ----------
    
    samples : array-like
        Array of MCMC links for each parameter
        
    plot : bool
        If `plot`=True, draw a bar plot of the R-hats

    labels : list of strings
        If `plot`=True, label each bar on the bar plot
        with the names in the list `labels`. Otherwise, 
        use indices as labels.
        
    Returns
    -------
    
    Rhat : float
        The Gelman-Rubin R-hat statistic, approaches unity after infinite
        steps of a well-mixed chain.
    
    '''
    n, m = np.shape(samples)
    Rhats = np.zeros(m)
    for j in range(m):
        individualchains = [samples[i:, j][::2*m] for i in range(2*m)]
        # W = mean of within-chain variance
        W = np.mean([np.var(chain, ddof=1) for chain in individualchains])
        # B = between chain variance
        B = n*np.var([np.mean(chain) for chain in individualchains], ddof=1)
    
        Vhat = W*(n-1)/n + B/n
        Rhats[j] = np.sqrt(Vhat/W)
        
    if kwargs.get('plot', False):
        if kwargs.get('labels', False):
            labels = kwargs.get('labels')
        else:
            labels = range(m)
        
        fig, ax = plt.subplots(1, figsize=(16,5))
        ax.bar(np.arange(len(labels))-0.5, Rhats, color='k')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, ha='right')
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        ax.set_ylim([0.9, np.max(Rhats)])
        ax.set_xlim([-1, len(labels)+1])
        ax.set_ylabel('$\hat{R}$')
        ax.set_title('Gelman-Rubin Statistic')
        plt.show()
    
    return Rhats

def chi2(v1, v2, err, Nfreeparams):
    return np.sum( ((v1-v2)/err)**2 )/(len(v1) - Nfreeparams)
    
def medplusminus(vector):
    '''
    Returns the 50%ile, the difference between the 84%ile and 50%ile, and 
    the difference between the 50%ile and the 16%ile, representing
    the median and the +/-1 sigma samples.

    Parameters
    ----------
    
    vector : array-like
            Vector of MCMC samples for one fitting parameter
        
    '''
    v = np.percentile(vector, [16, 50, 84])
    return v[1], v[2]-v[1], v[1]-v[0]
    
