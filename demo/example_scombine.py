import os, glob, sys
import numpy as np
import matplotlib.pyplot as pl

"""
A quick test and demonstration of the algorithms.
"""

import fsps
from scombine.sfhutils import load_angst_sfh
from scombine.dust import sexAmodel
import scombine.bursty_sfh as bsp
from sedpy import attenuation, observate

# Instantiate the SPS object and make any changes to the parameters here
sps = fsps.StellarPopulation(zcontinuous=1)
sps.params['logzsol'] = -1.0

# Load the input SFH, and set any bursts if desired (set f_burst=0
# to not add bursts)
filename = 'sfhs/ddo71.lowres.ben.v1.sfh'
f_burst, fwhm_burst, contrast = 0.0, 0.05 * 1e9, 5
sfh = load_angst_sfh(filename)
sfh['t1'] = 10.**sfh['t1']
sfh['t2'] = 10.**sfh['t2']
sfh['sfr'][0] *=  1 - (sfh['t1'][0]/sfh['t2'][0])
sfh[0]['t1'] = 0.
mtot = ((sfh['t2'] - sfh['t1']) * sfh['sfr']).sum()

# choose lookback times to generate
lookback_time = [0, 1e8]

# generate a high temporal resolution SFH, with bursts if f_burst > 0
lt, sfr, tb = bsp.burst_sfh(sfh=sfh, fwhm_burst=fwhm_burst, f_burst=f_burst, contrast=contrast)
# get the interpolation weights.  This does not have to be run in
# general (it is run interior to bursty_sps) unless you are
# debugging or for plotting purposes
aw = bsp.sfh_weights(lt, sfr, 10**sps.ssp_ages, lookback_time=lookback_time)
# get the intrinsic spectra at the lookback_times specified.
wave, spec, mstar, _ = bsp.bursty_sps(lt, sfr, sps, lookback_time=lookback_time)
# get reddened spectra, Calzetti foreground screen
wave, red_spec, _, lir = bsp.bursty_sps(lt, sfr, sps,
                                        lookback_time=lookback_time,
                                        dust_curve=attenuation.calzetti,
                                        av=1, dav=0)
# get reddened spectra, SexA differntial extinction plus SMC
dav = sexAmodel(davmax=1.0, ages=10**sps.ssp_ages)
wave, red_spec, _, lir = bsp.bursty_sps(lt, sfr, sps,
                                        lookback_time=lookback_time,
                                        dust_curve=attenuation.smc,
                                        av=0, dav=dav)

# Get intrinsic spectrum including an age metallicity relation
def amr(ages, **extras):
    """This should take an array of ages (linear years) and return an array
    of metallicities (units of log(Z/Z_sun)
    """
    logz_array = -1.0 * np.ones_like(ages)
    return logz_array

wave, spec, mstar, _ = bsp.bursty_sps(lt, sfr, sps, lookback_time=lookback_time,
                                      logzsol=amr(10**sps.ssp_ages, sfh=sfh))


# Output plotting.
pl.figure()
for i,t in enumerate(lookback_time):
    pl.plot(wave, spec[i,:], label = r'$t_{{lookback}} = ${0:5.1f} Gyr'.format(t/1e9))
pl.legend()
pl.xlim(1e3,1e4)
pl.xlabel('wave')
pl.ylabel(r'$F_\lambda$')

fig, ax = pl.subplots(2,1)
for i,t in enumerate(lookback_time):
    ax[1].plot(10**sps.ssp_ages, aw[i,:], marker='o', markersize=2,
               label=r'$t_{{lookback}} = ${0:5.1f} Gyr'.format(t/1e9))
    mstring = 'm_formed({0:3.1f}Gyr)={1}, m_formed(total)={2}, m_formed({0:3.1f}Gyr)/m_formed(total)={3}'
    print(mstring.format(t/1e9, aw[i,:].sum(), mtot, aw[i,:].sum()/mtot))
    print('m_star({0:3.1f}Gyr)={1}'.format(t/1e9, mstar[i]))
ax[1].set_xlabel('SSP age - lookback time')
ax[1].set_ylabel('Mass')
ax[1].legend(loc = 'upper left')

ax[0].plot(lt, sfr, 'k')
ax[0].set_xlabel('lookback time')
ax[0].set_ylabel('SFR')
pstring = 'f$_{{burst}}={0:3.1f}$, fwhm$_{{burst}}=${1:3.0f}Myr, contrast ={2}'
ax[0].set_title(pstring.format(f_burst, fwhm_burst/1e6, contrast))
for t in lookback_time:
    ax[0].axvline(x = t, color = 'r', linestyle =':', linewidth = 5)
pl.show()
