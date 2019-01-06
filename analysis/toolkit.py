import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import pyfits
'''
Calibrations
============

First, we must make a master flat and dark, then do simple nod subtraction.

File prefix for the September observations is: m140918_????.fits

Flats are numbers 0011-0210. Flat darks are 0211-0260

WASP-6 exposures are from 0373 - 0630
'''

#pathtorawimages = '/astro/store/scratch/tmp/bmmorris/mosfire/2014sep18/m140918_'
pathtorawimages = '/local/tmp/mosfire/2014sep18/m140918_'
#pathtooutputs = '/astro/store/scratch/tmp/bmmorris/mosfire/2014sep18_analysis/'
pathtooutputs = '/local/tmp/mosfire/2014sep18_analysis/'
badpixpath = '/astro/users/bmmorris/git/research/keck/2014september/analysis/variablepixels.npy'#'/astro/store/scratch/tmp/bmmorris/mosfire/2014sep18_analysis/badpix_10sep2012.fits'
badpixelmask = np.load(badpixpath)#pyfits.getdata(badpixpath)

prefix = 'm140918_'
suffix = '.fits'
suffix_nodsubtracted = 'n.fits'
suffix_nodadded = 'sum.fits'
suffix_nonod = 'cor.fits'
suffix_nodadded_lincomb = 'nlincom.fits'
suffix_nodsubtracted_lincomb = 'sumlincom.fits'
suffix_nodsubtracted_nobadpxl = 'n_nobadpxl.fits'
suffix_nodadded_nobadpxl = 'sum_nobadpxl.fits'
suffix_arc = 'shifted.fits'
rawflatpaths = ["%s%04d%s" % (pathtorawimages, i, suffix) for i in range(11,211)]
rawflatdarkpaths = ["%s%04d%s" % (pathtorawimages, i, suffix) for i in range(211,261)]
#rawwasp6paths = ["%s%04d%s" % (pathtorawimages, i, suffix) for i in range(373,631)]
rawwasp6paths = ["%s%04d%s" % (pathtorawimages, i, suffix) for i in range(360,631)]
rawwasp6darkpaths = ["%s%04d%s" % (pathtorawimages, i, suffix) for i in range(956,967)]
masterflatdarkpath = pathtooutputs+'masterflatdark.fits'
masterflatpath = pathtooutputs+'masterflat.fits'
bestshiftspath = pathtooutputs+'bestxshifts.npy'

# Pixel limits -- ignore the pixels outside these bounds
rowlimits = [5, 2030]
collimits = [5, 2044]


# Toggle on and off computations
makemasterflatdark = False
makemasterflat = False
computexshifts = False
makenodsubtracted = False#True
makecorrectedframes = False
makeshiftedarcs = False
makenodsubtracted_linearcombination = False
makenodsubtracted_nobadpixels = True

# Open a single sample image to get its dimensions
dim1, dim2 = np.shape(pyfits.getdata(rawflatpaths[0]))

if makemasterflatdark:
    # Make a three dimensional cube of darks, then median along the time axis
    flatdarkcube = np.zeros((dim1, dim2, len(rawflatdarkpaths)))
    for i, imagepath in enumerate(rawflatdarkpaths):
        print 'Reading raw dark image', i, 'of', len(rawflatdarkpaths)
        flatdarkcube[:,:,i] = pyfits.getdata(imagepath)
    masterflatdark = np.median(flatdarkcube, axis=2)
    pyfits.writeto(masterflatdarkpath, masterflatdark, clobber=True)
#plt.imshow(masterdark,cmap='GnBu_r')
#plt.show()

if makemasterflat: 
    if not makemasterflatdark:
        masterflatdark = pyfits.getdata(masterflatdarkpath)
    # Make a three dimensional cube of darks, then median along the time axis
    flatcube = np.zeros((dim1, dim2, len(rawflatpaths)))
    for i, imagepath in enumerate(rawflatpaths):
        print 'Reading raw flat image', i, 'of', len(rawflatpaths)
        flatcube[:,:,i] = pyfits.getdata(imagepath) - masterflatdark
    # Take median of all images
    masterflat = np.median(flatcube, axis=2)
    # Median normalize the result
    masterflat /= np.median(masterflat)
    pyfits.writeto(masterflatpath, masterflat, clobber=True)

oversamplefactor = 1 # Interpolation density compared to original kernel
if computexshifts or makenodsubtracted or makeshiftedarcs:
    arcpaths = ['/local/tmp/mosfire/2014sep18/m140918_000'+str(i)+'.fits' for i in range(4, 7)]
    arcimage = np.sum([pyfits.getdata(arcpath) for arcpath in arcpaths], axis=0)
    ydim, xdim = arcimage.shape

if computexshifts:
    # Correct it with the bad pixel mask -- set all bad pixel values to zero:
    arcimage *= np.abs(badpixelmask-1)
    arcimage = arcimage[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
    
    datakernel = np.arange(xdim)
    finekernel = np.linspace(0, xdim, oversamplefactor*len(datakernel))
    #paddingones = np.ones(oversamplefactor*len(datakernel)/2)
    paddingones = np.ones(250)
    xshifts = range(-1*len(paddingones), len(paddingones))
    
    templaterow = arcimage[0,:]
    templaterow_interp = np.interp(finekernel, datakernel, templaterow)
    templaterow_interp /= np.median(templaterow_interp)
    templaterow_interp_pad = np.concatenate([paddingones, templaterow_interp, paddingones])
    bestxshifts = np.zeros(ydim)
    
    for row in range(1, ydim):
        print 'Interpolationg row', row, 'of', ydim
        currentrow = arcimage[row,:]
        currentrow_interp = np.interp(finekernel, datakernel, currentrow)
        currentrow_interp /= np.median(currentrow_interp)
        currentrow_interp_pad = np.concatenate([paddingones, currentrow_interp, paddingones])
        bestchi2 = 1e30
        bestxshift = 0
        for xshift in xshifts:
            rolled_currentrow = np.roll(currentrow_interp_pad, xshift)
            chi2 = np.sum((rolled_currentrow - templaterow_interp_pad)**2)
            if chi2 < bestchi2:
                bestchi2 = chi2
                bestxshift = xshift
        bestxshifts[row] = bestxshift
        #plt.plot(np.roll(currentrow_interp_pad,bestxshift))
        #plt.plot(templaterow_interp_pad)
        #plt.show()
    np.save(bestshiftspath, bestxshifts)
else: 
    bestxshifts = np.load(bestshiftspath)



def path2int(filepath, prefix='m140918_', suffix=suffix):
    '''
    Given a file at path `filepath`, split the path at `prefix` and `suffix`,
    to grab just the exposure index from the file name, and cast it to an int. 
    '''
    dirs, secondhalf = filepath.split(prefix)
    int_label = int(secondhalf.split(suffix)[0])
    return int_label

def nearestexposures(filepath):
    '''
    Given a file at path `filepath`, return a list with the two "neighboring" exposures -- 
    the ones immediately preceding and immediately following the file at filepath
    '''
    int_label = path2int(filepath)
    previous_exp = "%s%04d%s" % (pathtorawimages, int_label-1, suffix)
    next_exp = "%s%04d%s" % (pathtorawimages, int_label+1, suffix)
    return [previous_exp, next_exp]

def channelshift(image):
    ydim, xdim = image.shape
    outputpaddingwidth = np.ceil(np.max(bestxshifts)/oversamplefactor)
    outputpadding = np.zeros((ydim, outputpaddingwidth))
    paddedimage = np.hstack([outputpadding, image, outputpadding])

    for row in range(1, ydim):
        paddedimage[row] = np.roll(paddedimage[row], int(bestxshifts[row]/oversamplefactor))
    return paddedimage

if makenodsubtracted:
    if not makemasterflat:
        masterflat = pyfits.getdata(masterflatpath)
    
    # Take mean of exposure before and after current exposure, 
    # then subtracted current exposure by that mean.
    removebadpix = np.abs(badpixelmask - 1)

    for positiveexposurepath in rawwasp6paths[1:-2]:
        print 'Nod subtracting:',positiveexposurepath
        # Find previous and next images in the list
        negativeexposurepaths = nearestexposures(positiveexposurepath)
        # Take the mean of the previous and next images in the list
        meannegativeexposure = (np.mean([pyfits.getdata(negativeexp)/masterflat for negativeexp in negativeexposurepaths],\
                                        axis=0) * removebadpix)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
        # Subtract the current image by the mean opposite nod image:
        positiveexposure = (pyfits.getdata(positiveexposurepath) * removebadpix/masterflat)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
        # Force bad pixels to zero in the sum
        nodsubtractedimage =  channelshift(positiveexposure - meannegativeexposure)
        #nodaddedimage = channelshift(positiveexposure**2 + meannegativeexposure**2)
        nodaddedimage = channelshift(positiveexposure + meannegativeexposure)
        originalheader = pyfits.getheader(positiveexposurepath)
        # string of file name index for current image:
        currentindstr = "%04d" % path2int(positiveexposurepath) 
        pyfits.writeto(pathtooutputs+prefix+currentindstr+suffix_nodsubtracted, \
                       nodsubtractedimage, header=originalheader, clobber=True)         
        pyfits.writeto(pathtooutputs+prefix+currentindstr+suffix_nodadded, \
                       nodaddedimage, header=originalheader, clobber=True)          

if makenodsubtracted_nobadpixels:
    if not makemasterflat:
        masterflat = pyfits.getdata(masterflatpath)
    
    # Take mean of exposure before and after current exposure, 
    # then subtracted current exposure by that mean.
    removebadpix = np.abs(badpixelmask - 1)

    for positiveexposurepath in rawwasp6paths[1:-2]:
        print 'Nod subtracting:',positiveexposurepath
        # Find previous and next images in the list
        negativeexposurepaths = nearestexposures(positiveexposurepath)
        # Take the mean of the previous and next images in the list
        meannegativeexposure = np.mean([pyfits.getdata(negativeexp)/masterflat for negativeexp in negativeexposurepaths],\
                                        axis=0)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
        # Subtract the current image by the mean opposite nod image:
        positiveexposure = (pyfits.getdata(positiveexposurepath)/masterflat)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
        # Force bad pixels to zero in the sum
        nodsubtractedimage =  channelshift(positiveexposure - meannegativeexposure)
        #nodaddedimage = channelshift(positiveexposure**2 + meannegativeexposure**2)
        nodaddedimage = channelshift(positiveexposure + meannegativeexposure)
        originalheader = pyfits.getheader(positiveexposurepath)
        # string of file name index for current image:
        currentindstr = "%04d" % path2int(positiveexposurepath) 
        pyfits.writeto(pathtooutputs+prefix+currentindstr+suffix_nodsubtracted_nobadpxl, \
                       nodsubtractedimage, header=originalheader, clobber=True)         
        pyfits.writeto(pathtooutputs+prefix+currentindstr+suffix_nodadded_nobadpxl, \
                       nodaddedimage, header=originalheader, clobber=True)          


if makenodsubtracted_linearcombination:
    f = open('lincomblog.txt','w')
    if not makemasterflat:
        masterflat = pyfits.getdata(masterflatpath)
    
    # Take mean of exposure before and after current exposure, 
    # then subtracted current exposure by that mean.
    removebadpix = np.abs(badpixelmask - 1)

    for positiveexposurepath in rawwasp6paths[1:-2]: #rawwasp6paths[5:10]:
        print 'Nod subtracting:',positiveexposurepath
        # Find previous and next images in the list
        negativeexposurepaths = nearestexposures(positiveexposurepath)
        # Take the mean of the previous and next images in the list
        #meannegativeexposure = (np.mean([pyfits.getdata(negativeexp)/masterflat for negativeexp in negativeexposurepaths],\
        #                                axis=0) * removebadpix)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
        earlieroppositenod = (pyfits.getdata(negativeexposurepaths[0]) * \
                              removebadpix/masterflat)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
        lateroppositenod = (pyfits.getdata(negativeexposurepaths[1]) * \
                              removebadpix/masterflat)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]

        positiveexposure = (pyfits.getdata(positiveexposurepath) * \
                            removebadpix/masterflat)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]

        def fitfunc(p):
            return p[0]*earlieroppositenod + (1-p[0])*lateroppositenod
        def errfunc(p):
            if 0 > p[0] or 1 < p[0]:
                return 1000
            else: 
                return (np.sum(fitfunc(p),axis=0) - np.sum(positiveexposure, axis=0))/np.sqrt(np.sum(fitfunc(p),axis=0))
        
        initp = [0.5]
        bestp, success = optimize.leastsq(errfunc, initp)
        if bestp[0] < 0 or bestp[0] > 1 or success not in [1, 2, 3, 4]:
            bestp[0] = 0.5
        print bestp, sum(bestp), success
        f.write("%s\n" % bestp)
        meannegativeexposure = fitfunc(bestp)
        # Subtract the current image by the mean opposite nod image:
        # Force bad pixels to zero in the sum
        nodsubtractedimage =  channelshift(positiveexposure - meannegativeexposure)
        nodaddedimage = channelshift(positiveexposure + meannegativeexposure)
        originalheader = pyfits.getheader(positiveexposurepath)
        # string of file name index for current image:
        currentindstr = "%04d" % path2int(positiveexposurepath) 

#        fig, ax = plt.subplots(1,figsize=(9,9))
#        testimg = nodsubtractedimage
#        m = np.mean(testimg)
#        s = np.std(testimg)
#        ns = 0.15
#        ax.imshow(testimg, origin='lower', interpolation='nearest', vmin=m-ns*s, vmax=m+ns*s, cmap='hot')
#        ax.set_xticks([])
#        ax.set_yticks([])
#        ax.set_title(positiveexposurepath)
#        plt.show()

        pyfits.writeto(pathtooutputs+prefix+currentindstr+suffix_nodsubtracted_lincomb, \
                       nodsubtractedimage, header=originalheader, clobber=True)         
        pyfits.writeto(pathtooutputs+prefix+currentindstr+suffix_nodadded_lincomb, \
                       nodaddedimage, header=originalheader, clobber=True)      
    f.close()

    f = open('lincomblog.txt','r').readlines()
    p = [eval(line)[0] for line in f]
    p
    p = np.array(p)
    import numpy as np
    plt.hist(p,100)
    plt.show()


if makecorrectedframes:
    if not makemasterflat:
        masterflat = pyfits.getdata(masterflatpath)
    darkframe = np.zeros_like(pyfits.getdata(rawwasp6darkpaths[0]))
    for darkpath in rawwasp6darkpaths:
        darkframe += pyfits.getdata(darkpath)
    darkframe /= len(rawwasp6darkpaths)
    
    # Take mean of exposure before and after current exposure, 
    # then subtracted current exposure by that mean.
    removebadpix = np.abs(badpixelmask - 1)

    for positiveexposurepath in rawwasp6paths[1:-2]:
        # Subtract the current image by the mean opposite nod image:
        positiveexposure = ((pyfits.getdata(positiveexposurepath) - darkframe) * removebadpix/masterflat)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
        # Force bad pixels to zero in the sum
        shiftedimage =  channelshift(positiveexposure)
        #nodaddedimage = channelshift(positiveexposure**2 + meannegativeexposure**2)
        originalheader = pyfits.getheader(positiveexposurepath)
        # string of file name index for current image:
        currentindstr = "%04d" % path2int(positiveexposurepath) 
        pyfits.writeto(pathtooutputs+prefix+currentindstr+suffix_nonod, \
                       shiftedimage, header=originalheader, clobber=True)         
        

if makeshiftedarcs:
    removebadpix = np.abs(badpixelmask - 1)
    if not makemasterflat:
        masterflat = pyfits.getdata(masterflatpath)
    for arcpath in arcpaths:
        arc = pyfits.getdata(arcpath)*removebadpix/masterflat
        header = pyfits.getheader(arcpath)
        shiftedarc = channelshift(arc[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]])
        currentindstr = "%04d" % path2int(arcpath) 
        pyfits.writeto(pathtooutputs+prefix+currentindstr+suffix_arc, \
                       shiftedarc, header=header, clobber=True)    
## Now save copies of the raw images with the wavelength shifts in place:
#outputpaddingwidth = np.ceil(np.max(bestxshifts)/oversamplefactor)
#outputpadding = np.zeros((ydim, outputpaddingwidth))
#paddedarcimage = np.hstack([outputpadding, arcimage, outputpadding])
#
#testimage = pyfits.getdata(pathtorawimages+'0561'+suffix)[rowlimits[0]:rowlimits[1],collimits[0]:collimits[1]]
#paddedtestimage = np.hstack([outputpadding, testimage, outputpadding])
#
#for row in range(1, ydim):
#    paddedtestimage[row] = np.roll(paddedtestimage[row], int(bestxshifts[row]/oversamplefactor))
#plt.imshow(paddedtestimage,origin='lower')
#plt.show()
#
#for row in range(1, ydim):
#    paddedarcimage[row] = np.roll(paddedarcimage[row], int(bestxshifts[row]/oversamplefactor))
#plt.imshow(paddedarcimage,origin='lower')
#plt.show()

#plt.imshow(arcimage, interpolation='nearest', origin='lower')
#plt.show()

#wasp6paths_nodsub_odd = ["%s%04d%s" % ('/local/tmp/mosfire/2014sep18_analysis/m140918_', i, suffix_nodsubtracted) for i in range(374,630,2)]
##img_collimits = [200, 2136]
###img_rowlimits = [0, ]
##
###for imagepath in wasp6paths_nodsub_odd:
###    
#testimg = pyfits.getdata(wasp6paths_nodsub_odd[0])
##plt.imshow(testimg)
#fig, ax = plt.subplots(1)#,figsize=())
#plt.plot(np.sum(testimg, axis=1))
#plt.show()




