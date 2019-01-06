
# coding: utf-8

## Dim nod-nulling source

# While a dim source near the target has been identified, we need to figure out how to remove the dim sources that overlap in the nods. 

# In[1]:

#get_ipython().magic(u'pylab inline')
import numpy as np
import matplotlib
matplotlib.rcParams['path.simplify'] = False
matplotlib.rcParams['path.simplify_threshold'] = 1
from matplotlib import pyplot as plt
from scipy import optimize
import pyfits
import pyfits
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import photPack2
from astropy.time import Time

nodApath = '/local/tmp/mosfire/2014sep18/m140918_0400.fits'
nodBpath = '/local/tmp/mosfire/2014sep18/m140918_0401.fits'
masterflat = pyfits.getdata('/local/tmp/mosfire/2014sep18_analysis/masterflat.fits')
pathtooutputs = '/local/tmp/mosfire/2014sep18_analysis/'
badpixpath = '/astro/store/scratch/tmp/bmmorris/mosfire/2014sep18_analysis/badpix_10sep2012.fits'
badpixelmask = np.abs(pyfits.getdata(badpixpath) - 1)

nodA = np.zeros_like(pyfits.getdata(nodApath))#*badpixelmask/masterflat
nodB = np.zeros_like(pyfits.getdata(nodBpath))#*badpixelmask/masterflat

evenpaths = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18/m140918_', i,                                        '.fits') for i in range(374,629,2)]
oddpaths = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18/m140918_', i,                                        '.fits') for i in range(375,629,2)]

for i in evenpaths:
    addimage = pyfits.getdata(i)*badpixelmask/masterflat
    addimage[np.isnan(addimage)] = 0
    nodA += addimage
for i in oddpaths:
    addimage = pyfits.getdata(i)*badpixelmask/masterflat
    addimage[np.isnan(addimage)] = 0
    nodB += addimage

nodArows = np.sum(nodA, axis=1)
nodBrows = np.sum(nodB, axis=1)


# In[2]:

fig, ax = plt.subplots(2,1,figsize=(16,16))

ax[0].plot(nodArows)
ax[1].plot(nodBrows)

for axes in ax:
    axes.set_xlabel('Row')
    axes.set_ylabel('Counts')
    axes.set_xlim([100,1900])
    axes.set_ylim([1.5e9,2.5e9])

plt.show()


# In[3]:

def idlstylemedianfilter(array, kernel):
    '''
    This mimics the behavior of the MEDIAN(array, kernel) built-in method
    in IDL for one or two dimensional input arrays. Kernel must be odd. 
    '''
    smoothedarray = np.copy(array)
    assert kernel % 2 != 0, 'Must be an odd kernel'
    if len(np.shape(array)) == 1:
        for j in xrange(kernel/2, len(array)-kernel/2):
            smoothedarray[j] = np.median(array[j-kernel/2 : j+kernel/2 + 1])        
    else:
        for i in xrange(len(array[0,:])):
            row = array[:,i]
            for j in xrange(kernel/2, len(array[:,0])-kernel/2):
                smoothedarray[j,i] = np.median(row[j-kernel/2 : j+kernel/2 + 1])
    return smoothedarray

fig, ax = plt.subplots(2,1,figsize=(16,16))

rows = np.arange(len(nodArows))
threshold = 0.5
nodArows[np.isnan(nodArows)] = 0
nodBrows[np.isnan(nodBrows)] = 0
#mask = (np.abs(nodArows - np.mean(nodArows)) <  + threshold*np.std(nodArows))
mask = (np.abs(nodArows - np.mean(nodArows)) < threshold*np.std(nodArows))
ax[0].plot(rows[mask], nodArows[mask]/idlstylemedianfilter(nodArows[mask], 301))
ax[1].plot(rows[mask], nodBrows[mask]/idlstylemedianfilter(nodArows[mask], 301))

Ndegree = 10
#fitA = np.polyfit(rows, nodArows, Ndegree)
#fitB = np.polyfit(rows, nodBrows, Ndegree)

#ax[0].plot(rows, np.polyval(fitA, rows))
#ax[1].plot(rows, np.polyval(fitB, rows))
print (mask == 0).all()

for axes in ax:
    axes.set_xlabel('Row')
    axes.set_ylabel('Counts')
    axes.set_xlim([100,1900])
    #axes.set_ylim([1.5e9,1.8e9])
    axes.set_ylim([0.97,1.03])
plt.show()


# In[16]:

from scipy import ndimage

fig, ax = plt.subplots(1,figsize=(16,8))

rows = np.arange(len(nodArows))
threshold = 0.5
#nodArows[np.isnan(nodArows)] = 0
#nodBrows[np.isnan(nodBrows)] = 0
#mask = (np.abs(nodArows - np.mean(nodArows)) <  + threshold*np.std(nodArows))
#mask = (np.abs(nodArows - np.mean(nodArows)) < threshold*np.std(nodArows))
nodsubtracted = nodArows - nodBrows

dimsourcerows = np.arange(1750, 1840)
dimsourceflux = nodsubtracted[(rows <= dimsourcerows.max())*(rows >= dimsourcerows.min())]
ax.plot(dimsourcerows, dimsourceflux, 'r', lw=1)
righthalfmedian = np.median(nodsubtracted[(rows <= 1900)*(rows >= 1200)])
ax.axhline(righthalfmedian, ls='--', color='k', lw=2)

brightsourcerows = np.arange(1360, 1500)
brightsourceflux = nodsubtracted[(rows <= brightsourcerows.max())*(rows >= brightsourcerows.min())]
ax.plot(brightsourcerows, brightsourceflux, 'r', lw=2)
ax.fill_between(brightsourcerows, righthalfmedian, brightsourceflux, alpha=0.3, color='r')
ax.fill_between(dimsourcerows, righthalfmedian, dimsourceflux, alpha=0.3, color='r')

nulledsourcerows = np.arange(1550, 1650)
nulledsourceflux = nodsubtracted[(rows <= nulledsourcerows.max())*(rows >= nulledsourcerows.min())]
ax.plot(nulledsourcerows, nulledsourceflux, 'g', lw=2)
ax.fill_between(nulledsourcerows, righthalfmedian, nulledsourceflux)
print type(nulledsourceflux), np.std(nulledsourceflux), np.mean(nulledsourceflux)

ax.plot(rows, ndimage.gaussian_filter1d(nodsubtracted,0.5), 'o',         color='blue', markersize=4, mec='none')#/idlstylemedianfilter(nodsubtracted,1))

brightarea = np.trapz(brightsourceflux - righthalfmedian, x=brightsourcerows)
dimarea = -1*np.trapz(dimsourceflux - righthalfmedian, x=dimsourcerows)
nulledarea = -1*np.trapz(nulledsourceflux - righthalfmedian, x=nulledsourcerows)
print 'flux ratio between bright and dim (in red) = %f' % (dimarea/brightarea)
print 'Ratio of nulled star to clean star flux = %f' % (nulledarea/brightarea)

ax.set_xlabel('Row')
ax.set_ylabel('Counts')
ax.set_xlim([100,1900])
#axes.set_ylim([1.5e9,1.8e9])
ax.set_ylim([0.5e7,1.5e7])
ax.set_title('Nod subtracted frame')
plt.show()


# In[5]:

fig, ax = plt.subplots(2, 1,figsize=(16,10))
ax[0].plot(np.sum(masterflat,axis=1))
ax[0].set_xlim([100,1900])
ax[0].set_ylim([1900,2200])
ax[0].set_title('Master flat')

ax[1].plot(np.sum(np.abs(badpixelmask - 1),axis=1))
ax[1].set_xlim([100,1900])
ax[1].set_ylim([0, 80])
ax[1].set_title('Bad pixels per row')
plt.show()


# In[12]:


x = np.arange(10)
y = x**2
y2 = x**2 - 10
y3 = 0.5*x**2
y4 = 0.1*x**2
plt.plot(x, y)
plt.fill_between(x, y, 25, alpha=0.5, color='r')
plt.fill_between(x, y2, 25, alpha=0.5, color='g')
plt.fill_between(x, y3, 25, alpha=0.5, color='b')
plt.fill_between(x, y4, 25, alpha=0.5, color='m')
plt.show()


# In[7]:

np.isnan(nulledsourceflux).any()


# In[7]:



