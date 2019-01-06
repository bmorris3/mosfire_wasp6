# -*- coding: utf-8 -*-
"""
Created on Feb 3, 2015 by Brett Morris.

Retrieve limb-darkening parameters from the Claret et al. 2013 tables for 
specificed filters, and for stars of specified effective temperatures, surface
gravities and for 


Notes
-----

This script will download the Claret et al. 2013 table from Vizier at: 

   http://vizier.cfa.harvard.edu/viz-bin/asu-txt/?-source=J/A+A/552/A16/limb1-4

the complementary table to the Claret et al. 2013 paper:

New limb-darkening coefficients for PHOENIX/1D model atmospheres.
II. Calculations for 5000K ≤ Teff ≤ 10000K Kepler, CoRot, Spitzer, uvby,
UBVRIJHK, Sloan, and 2MASS photometric systems.
    Claret, A., Hauschildt, P.H., Witte, S.
   <Astron. Astrophys. 552, A16 (2013)>
   =2013A&A...552A..16C

http://adsabs.harvard.edu/abs/2013A%26A...552A..16C


Available filters: 
   S3, S2, S1, S4, z', Kp, u', C, B, I, H, K, J, g', R, U, V, i', Ks, b, H2, 
   J2, r', u, v, y

Implemented limb-darkening parameterizations: 
   linear, quadratic, logarithmic


Usage
-----

Get quadratic limb-darkening parameters for a star with effective temperature
5800 K and log g = 4.0, in the K band: 

>>> from claretld import quad
>>> print quad(5800, 4.0, 'K')
(0.13200000000000001, 0.183)

The first returned term is the linear term, the second is the quadratic term in
the two-parameter quadratic limb-darkening law.

@author: bmorris3
"""

import numpy as np
from urllib import urlopen
from StringIO import StringIO
from glob import glob
import cPickle

URL = 'http://vizier.cfa.harvard.edu/viz-bin/asu-txt/'+\
      '?-source=J/A+A/552/A16/limb1-4'
filename = 'claret2013limb1-4.pkl'

# If the Claret table has been downloaded already, load it. If not, download it
if len(glob(filename)) == 0:
    rawtable = urlopen(URL).read()
    cPickle.dump(rawtable, open(filename, 'wb'))
else: 
    rawtable = cPickle.load(open(filename, 'rb'))

table = np.genfromtxt(StringIO(rawtable), skip_header=42, dtype=None)

# Load numpy array into dictionary for readable code. 
# The keys are the column label.
d = {}
columnlabels = ['logg', 'Teff', 'Z', 'xi', 'Filt', 'Met', 'Mod',
        'u', 'Merit', 'a', 'b', 'c', 'd', 'e', 'f']
for i, column in enumerate(columnlabels):
    d[column] = table['f{0}'.format(i)]

# Check that filters requested are available filters
filters = ["S3", "S2", "S1", "S4", "z'", "Kp", "u'", "C", "B", "I", "H", "K",
           "J", "g'", "R", "U", "V", "i'", "Ks", "b", "H2", "J2", "r'", "u", 
           "v", "y"]

def getcloseindices(key, value):
    '''
    Given a dictionary `key` and a `value`, search for the indices within
    that key's vector that are closest to `value`
    '''
    return np.min(np.abs(d[key] - value)) == np.abs(d[key] - value)

def getclosestmodel(Teff, logg, filt, method='F'):
    '''
    All grid points have metallicities Z=0 (solar), microturbulence xi=2
    
    Parameters
    ----------
    filt : string
        Name of the filter used. 
        
    logg : float
        Logarithm of the stellar surface gravity.

    Teff : float
        Host star effective temperature.
       
    method : string, optional. Either 'F' or 'L'
        Method of computation: least-square or flux conservation
    '''
    if filt not in filters:
        raise ValueError(('Filter "{0}" is not in the list of available ' +
              'filters.').format(filt))
    else:
        closest = getcloseindices('logg', logg) * \
                  getcloseindices('Teff', Teff) * \
                  (d['Met'] == method.upper()) * (d['Filt'] == filt) 
        
        # If only one data row is closest: 
        if np.sum(closest) == 1:
            return np.arange(len(closest))[closest][0]
        
        # If two data rows are equally close, return both:
        elif np.sum(closest) == 2:
            return np.arange(len(closest))[closest]
        else:
            return []

def linear(*args):
    '''
    Linear limb-darkening law. 
    All grid points have metallicities Z=0 (solar), microturbulence xi=2.
    
    Parameters
    ----------
    filt : string
        Name of the filter used. 
        
    logg : float
        Logarithm of the stellar surface gravity.

    Teff : float
        Host star effective temperature.
       
    method : string, optional. Either 'F' or 'L'
        Method of computation: least-square or flux conservation
        
    Returns
    -------
    a : float
        The linear limb-darkening term for a quadratic law: 
        I(mu)/I(1) = 1 - u*(1 - mu)
    '''
    closestmodel = getclosestmodel(*args)

    # If two grid points are equally close, take the mean of the ld-parameters
    if type(closestmodel) == np.ndarray:
        return np.mean(d['u'][closestmodel])
    # If one grid point is the closest: 
    else: 
        return d['u'][closestmodel]

def quad(*args):
    '''
    Quadratic limb-darkening law.
    All grid points have metallicities Z=0 (solar), microturbulence xi=2.
    
    Parameters
    ----------
    filt : string
        Name of the filter used. 
        
    logg : float
        Logarithm of the stellar surface gravity.

    Teff : float
        Host star effective temperature.
       
    method : string, optional. Either 'F' or 'L'
        Method of computation: least-square or flux conservation
        
    Returns
    -------
    a, b : float, float
        The linear and quadratic limb-darkening terms for a quadratic law: 
        I(mu)/I(1) = 1 - a*(1 - mu) - b*(1 - mu)**2
    '''
    closestmodel = getclosestmodel(*args)

    # If two grid points are equally close, take the mean of the ld-parameters
    if type(closestmodel) == np.ndarray:
        return np.mean(d['a'][closestmodel]), np.mean(d['b'][closestmodel])
    # If one grid point is the closest: 
    else: 
        return d['a'][closestmodel], d['b'][closestmodel]
    
def logarithmic(*args):
    '''
    Logarithmic limb-darkening law.
    All grid points have metallicities Z=0 (solar), microturbulence xi=2.
    
    Parameters
    ----------
    filt : string
        Name of the filter used. 
        
    logg : float
        Logarithm of the stellar surface gravity.

    Teff : float
        Host star effective temperature.
       
    method : string, optional. Either 'F' or 'L'
        Method of computation: least-square or flux conservation
        
    Returns
    -------
    c, d : float, float
        The first and second limb-darkening terms for a logarithmic law: 
        I(mu)/I(1) = 1 - e*(1 - mu) - f*mu*ln(mu)
    '''
    closestmodel = getclosestmodel(*args)

    # If two grid points are equally close, take the mean of the ld-parameters
    if type(closestmodel) == np.ndarray:
        return np.mean(d['e'][closestmodel]), np.mean(d['f'][closestmodel])
    # If one grid point is the closest: 
    else: 
        return d['e'][closestmodel], d['f'][closestmodel]

def u2q(u1, u2, warnings=True):
    '''
    Convert the linear and quadratic terms of the quadratic limb-darkening
    parameterization -- called `u_1` and `u_2` in Kipping 2013 or `a` and `b` in 
    Claret et al. 2013 -- and convert them to `q_1` and `q_2` as described in
    Kipping 2013: 
    
    http://adsabs.harvard.edu/abs/2013MNRAS.435.2152K
    
    Parameters
    ----------
    u1 : float
        Linear component of quadratic limb-darkening
        
    u2 : float
        Quadratic component of quadratic limb-darkening    
        
    Returns
    -------
    (q1, q2) : tuple of floats
        Kipping (2013) style quadratic limb-darkening parameters
    '''
    q1 = (u1 + u2)**2
    q2 = 0.5*u1/(u1+u2)
    if warnings and (u1 < 0 or u2 < 0):
        print "WARNING: The quadratic limb-darkening parameters " + \
              "u1={0:.3f} or u2={0:.3f} violate Kipping's ".format(u1, u2) + \
              "conditions for a monotonically increasing or everywhere-" +\
              "positive intensity profile. Returning them as is."
    return q1, q2
    
def q2u(q1, q2):
    '''
    Convert the two parameter quadratic terms of the Kipping 2013 limb-
    darkening parameterization `q_1` and `q_2` to the standard linear and 
    quadratic terms of the quadratic limb-darkening parameterization of
    Claret et al. 2013 -- called `u_1` and `u_2` in Kipping 2013 or `a` and `b` in 
    Claret et al. 2013:
    
    http://adsabs.harvard.edu/abs/2013A%26A...552A..16C
    
    Parameters
    ----------
    q1 : float
        First component of Kipping 2013 quadratic limb-darkening
        
    q2 : float
        Second component of Kipping 2013 quadratic limb-darkening   
        
    Returns
    -------
    (u1, u2) : tuple of floats
        Claret et al. (2013) style quadratic limb-darkening parameters
    '''
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1-2*q2)
    return u1, u2