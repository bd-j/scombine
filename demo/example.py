import os
import glob
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as pl

import observate
import attenuation
import pycombine
import sfhutils as utils

filternamelist = ['galex_FUV','wfc3_uvis_f275w']
filterlist = observate.load_filters(filternamelist)                  
sfhfiles = glob.glob('./sfhs/*sfh')
utils.skiprows = 0 #!!!important!!! for Jake's zcb files there's no header.  otherwise set skiprows to the number of header rows
utils.skiprows = 6

objname = [os.path.basename(f).split('.')[0] for f in sfhfiles]
tl = ['present','100 Myr ago']

# Generate a few basis files, one for present day, and
#  one for a lookback-time of 100Myr.  This is slooow
#present = pycombine.generate_basis(sfhfiles[0], zmet = 1.0, imf_type = 0, outroot = 'L0_tl0', t_lookback = 0.0)
#previous = pycombine.generate_basis(sfhfiles[0], zmet = 1.0, imf_type = 0, outroot = 'L0_tl100Myr', t_lookback = 1e8)

# If bases were already created 
present, previous = glob.glob('*fits')

# Make a list of 'combiners', where each uses a different spectral basis
combiners = [pycombine.Combiner(basis, dust_law = attenuation.cardelli) for basis in [present, previous]]

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
        pl.plot(cb.wave, results[0], label = '{0} @ {1}'.format(objname[i], tl[j]))

pl.xlim(1e3, 1e4)
pl.ylim(5e-19,5e-14)
pl.yscale('log')
pl.xlabel(r'wavelength ($\AA$)')
pl.ylabel(r'erg/s/cm$^2$/$\AA$ @ 10pc')
pl.legend(loc = 'lower right')
pl.savefig('demo.png')
