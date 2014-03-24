import os, glob, sys
import numpy as np
try:
    import astropy.io.fits as pyfits
except (ImportError):
    import pyfits
import matplotlib.pyplot as pl

import observate, attenuation
import scombine
import sfhutils as utils

filternamelist = ['galex_FUV','wfc3_uvis_f275w']
filterlist = observate.load_filters(filternamelist)                  
sfhfiles = glob.glob('./sfhs/*sfh')

objname = [os.path.basename(f).split('.')[0] for f in sfhfiles]
# In general it's good to avoid bin edges (like 100Myr in the example below...)
tl = ['present','100 Myr ago']
tval = [0.0,1e8]

# Generate a few basis files, one for present day, and
#  one for a lookback-time of 100Myr.  This is slooow
#present = scombine.generate_basis(sfhfiles[0], zmet = 1.0, imf_type = 0,  outroot = 'L0_tl100Myr', t_lookback = 0.0)
#
clobber = False
present, previous = scombine.generate_basis(sfhfiles[0], zmet = 1.0, imf_type = 0, t_lookback = tval, clobber = clobber)

# If the two bases were already created 
#present, previous = glob.glob('*fits')

# Make a list of 'combiners', where each uses a different spectral basis
combiners = [scombine.Combiner(basis, dust_law = attenuation.cardelli) for basis in [present, previous]]

# Initialize magnitude and spectral storage
all_mags = np.zeros([ len(sfhfiles), len(combiners), len(filterlist)]) 
all_spec = np.zeros([ len(sfhfiles), len(combiners), len(combiners[0].wave)])

pl.figure()
# Loop over sfhs and over the lookback times
for i,f in enumerate(sfhfiles):
    # Constant foreground dust affecting all stars of all ages.
    #   In principle this can be a vector if you want age dependent dust screen
    av = 0.1 
    for j, cb in enumerate(combiners):
        # Use the Dolphin 2002 differential extinction model, which is age dependent.
        #  NOTE: It is somewhat ambiguous what to do in the case of the
        #  larger lookback time basis.  There should probably be a
        #  lookback time parameter passed to sexAmodel
        #dav = combiners[0].sexAmodel(avmax = 1.0, f)
        dav= 0.1 #have stars additionly extincted by dust uniformly distributed from av to dav
        results = cb.combine(f, av = av, dav = dav, filterlist =filterlist)            
        all_mags[i,j,:] = results[2]
        all_spec[i,j,:] = results[0]
        pl.plot(cb.wave, results[0] * cb.wave, label = '{0} @ {1}'.format(objname[i], tl[j]))

pl.xlim(1e3, 1e4)
pl.ylim(1e-3,1e1)
pl.yscale('log')
pl.xlabel(r'wavelength ($\AA$)')
pl.ylabel(r'$\lambda$L$_\lambda$ (erg/s/cm$^2$ @ 10pc)')
pl.legend(loc = 'lower right')
pl.savefig('demo.png')
