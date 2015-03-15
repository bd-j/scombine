import sys
import numpy as np
from scipy import interpolate

from sfhutils import weights_1DLinear
from sedpy import attenuation

lsun, pc = 3.846e33, 3.085677581467192e18
lightspeed = 2.998e18 #AA/s
#value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )

def burst_sfh(fwhm_burst=0.05, f_burst=0.5, contrast=5,
              sfh=None, bin_res=10.):
    """
    Given a binned SFH as a numpy structured array, and burst
    parameters, generate a realization of the SFH at high temporal
    resolution. The output time resolution will be approximately
    fwhm_burst/12 unless no bursts are generated, in which case the
    output time resolution is the minimum bin width divided by
    bin_res.

    :param fwhm_burst: default 0.05
        the fwhm of the bursts to add, in Gyr.
        
    :param f_burst: default, 0.5
        the fraction of stellar mass formed in each bin that is formed
        in the bursts.
        
    :param contrast: default, 5
        the approximate maximum height or amplitude of the bursts
        above the constant background SFR.  This is only approximate
        since it is altered to preserve f_burst and fwhm_burst even
        though the number of busrsts is quantized.
        
    :param sfh: structured ndarray
        A binned sfh in numpy structured array format.  Usually the
        result of sfhutils.load_angst_sfh()
        
    :param bin_res: default 10
        Factor by which to increase the time resolution of the output
        grid, relative to the shortest bin width in the supplied SFH.

    :returns times:  ndarray of shape (nt)
        The output linear, regular temporal grid of lookback times.

    :returns sfr: ndarray of shape (nt)
        The resulting SFR at each time.

    :returns f_burst_actual:
        In case f_burst changed due to burst number discretezation.
        Shouldn't happen though.        
    """
    #
    a, tburst, A, sigma, f_burst_actual = [],[],[],[],[]
    for i,abin in enumerate(sfh):
     #   print('------\nbin #{0}'.format(i))
        res = convert_burst_pars(fwhm_burst = fwhm_burst, f_burst = f_burst, contrast = contrast,
                             bin_width = (abin['t2']-abin['t1']), bin_sfr = abin['sfr'])
        a += [res[0]]
        if len(res[1]) > 0:
            tburst += (res[1] + abin['t1']).tolist()
            A += len(res[1]) * [res[2]]
            sigma += len(res[1]) * [res[3]]
        #f_burst_actual += [res[4]]
    if len(sigma) == 0:
        #if there were no bursts, set the time resolution to be 1/bin_res of the
        # shortest bin width.
        dt = (sfh['t2'] - sfh['t1']).min()/(1.0 * bin_res)
    else:
        dt = np.min(sigma)/5. #make sure you sample the bursts reasonably well
    times = np.arange(np.round(sfh['t2'].max()/dt)) * dt
    #figure out which bin each time is in
    bins = [sfh[0]['t1']] + sfh['t2'].tolist()
    bin_num = np.digitize(times, bins) -1
    #calculate SFR from all components
    sfr = np.array(a)[bin_num] + gauss(times, tburst, A, sigma)
    
    return times, sfr, f_burst_actual

def smooth_sfh(sfh=None, bin_res=10., **kwargs):
    """Method to produce a smooth SFH from a given step-wise SFH, under the
    constraint that the  total integrated mass at the end of each 'step' is
    preserved.  Uses a cubic spline with a monotonicity constraint to obtain
    M_tot(t) which is then differentiated to produce the SFH.

    :param sfh:
        A stepwise SFH in the format produced by `load_angst_sfh()`,
        but with the time fields converted to linear time. A structured
        array.
       
    :param bin_res: default 10
        Factor by which to increase the time resolution of the output
        grid, relative to the shortest bin width in the supplied SFH.

    :returns times:  ndarray of shape (nt)
        The output linear, regular temporal grid of lookback times.

    :returns sfr:
        The SFR at `times`.
    """
    mtot = ((sfh['t2'] - sfh['t1']) * sfh['sfr'] ).sum()
    #flip time axis to be time since earliest point in SFH
    # instead of lookback time
    tmax = sfh['t2'].max()
    tt = tmax - sfh['t1'] 
    #tt[0] = sfh[-1]['t2']

    monospline = interpolate.PchipInterpolator(np.concatenate([[0],tt[::-1]]),
                                               mtot * np.concatenate([[0],sfh['mformed'][::-1]]))
    
    dt = (sfh['t2'] - sfh['t1']).min()/(1.0 * bin_res)
    times = tmax - np.arange(np.round(sfh['t2'].max()/dt)) * dt
    return tmax - times, monospline.derivative(der=1)(times)

def bursty_sps(lookback_time, lt, sfr, sps,
               safe=True,
               av=None, dav=None, nsplit=9,
               dust_curve=attenuation.cardelli):
    """
    Obtain the spectrum of a stellar poluation with arbitrary complex
    SFH at a given lookback time.  The SFH is provided in terms of SFR
    vs t_lookback. Note that this in in contrast to the normal
    specification in terms of time since the big bang. Interpolation
    of the available SSPs to the time sequence of the SFH is
    accomplished by linear interpolation in log t.  Highly oscillatory
    SFHs require dense sampling of the temporal axis to obtain
    accurate results.

    :param lookback_time: scalar or ndarray, shape (ntarg)
        The lookback time(s) at which to obtain the spectrum. In yrs.
        
    :param lt: ndarray, shape (ntime)
        The lookback time sequence of the provided SFH.  Assumed to
        have have equal linear time intervals.
        
    :param sfr: ndarray, shape (ntime)
        The SFR corresponding to each element of lt, in M_sun/yr.
        
    :param sps: fsps.StellarPopulation instance
        The fsps stellar population (with metallicty and IMF
        parameters set) to use for the SSP spectra.

    :param av: scalar or ndarray of shape (nspec)
        The attenuation at V band, in magnitudes, that affects all
        stars equally. Passed to redden()

    :param dav: scalar or ndarray of shape (nspec)
        The maximum differential attenuation, in V band
        magnitudes. Passed to redden()
        
    :returns wave: ndarray, shape (nwave)
        The wavelength array
        
    :returns int_spec: ndarray, shape(ntarg, nwave)
        The integrated spectrum at lookback_time, in L_sun/AA
        
    :returns aw: ndarray, shape(ntarg, nage)
        The total weights of each SSP spectrum for each requested
        lookback_time.  Useful for debugging.

    :returns lir: optional, ndarray, shape(ntarg)
        The total absorbed luminosity, in L_sun.  Only returned if the
        attenuation curve, dav, and av are not None.
        
    """
    
    dt = lt[1] - lt[0]
    assert np.all(np.isclose(np.diff(lt), dt))
    
    sps.params['sfh'] = 0 #make sure SSPs
    # get *all* the ssps
    if safe:
        wave, spec = sps.get_spectrum(peraa=True, tage=0) #slower, stabler way
    else:
        # slightly more dangerous fast way, requiring the up-to-date python-fsps
        zmet = sps.params['zmet']-1
        spec, mass, _ = sps.all_ssp_spec(peraa=True, update=True)
        spec = spec[:,:,zmet].T
        wave = sps.wavelengths
    ssp_ages = 10**sps.ssp_ages #in yrs
    
    # redden the SSP spectra
    spec, lir = redden(wave, spec, av=av, dav=dav,
                       dust_curve=dust_curve, nsplit =nsplit)
    
    target_lt = np.atleast_1d(lookback_time)
    #set up output
    int_spec = np.zeros( [ len(target_lt), len(wave) ] )
    aw = np.zeros( [ len(target_lt), len(ssp_ages) ] )

    for i,tl in enumerate(target_lt):
        valid = (lt > tl) #only consider time points in the past of this lookback time.
        #inds, weights = weights_1DLinear(np.log(ssp_ages), np.log(lt[valid] - tl))

        #augment the t_lookback array of the SFH with the SSP ages
        sfr_ssp = np.interp(ssp_ages, lt-tl, sfr, left=0.0, right=0.0)
        tmp_t = np.concatenate([ssp_ages, lt[valid]-tl])
        tmp_sfr = np.concatenate([sfr_ssp, sfr[valid]])
        #sort the augmented array by age
        order = tmp_t.argsort()
        tmp_t = tmp_t[order]
        tmp_sfr = tmp_sfr[order]
        # get weights to interpolate the log_t array
        inds, weights = weights_1DLinear(ssp_ages, tmp_t)
        # aggregate the weights for each ssp time index, after
        # accounting for SFR *dt
        tmp_dt = np.gradient(tmp_t)
        agg_weights = np.bincount( inds.flatten(),
                                   weights = (weights * tmp_sfr[:, None] *
                                              tmp_dt[:, None]).flatten(),
                                   minlength = len(ssp_ages) )

        
        #agg_weights = np.bincount( inds.flatten(),
        #                           weights = (weights * sfr[valid, None]).flatten(),
        #                           minlength = len(ssp_ages) ) * dt
        int_spec[i,:] = (spec * agg_weights[:,None]).sum(axis=0)
        aw[i,:] = agg_weights
    if lir is not None:
        lir_tot = (aw * lir[None,:]).sum(axis = -1)
        return wave, int_spec, aw, lir_tot
    else:
        return wave, int_spec, aw


def bursty_lf(lookback_time, lt, sfr, sps_lf):
    """
    Obtain the luminosity function of stars for an arbitrary complex
    SFH at a given lookback time.  The SFH is provided in terms of SFR
    vs t_lookback. Note that this in in contrast to the normal
    specification in terms of time since the big bang.

    :param lookback_time: scalar or ndarray, shape (ntarg)
        The lookback time(s) at which to obtain the spectrum. In yrs.
        
    :param lt: ndarray, shape (ntime)
        The lookback time sequence of the provided SFH.  Assumed to
        have have equal linear time intervals.
        
    :param sfr: ndarray, shape (ntime)
        The SFR corresponding to each element of lt, in M_sun/yr.
        
    :param sps_lf:
        Luminosity function information, as a dictionary.  The keys of
        the dictionary are 'bins', 'lf' and 'ssp_ages'

    :returns bins:
        The bins used to define the LF
        
    :returns int_lf: ndarray, shape(ntarg, nbin)
        The integrated LF at lookback_time, in L_sun/AA
        
    :returns aw: ndarray, shape(ntarg, nage)
        The total weights of each LF for each requested
        lookback_time.  Useful for debugging.
        
    """

    dt = lt[1] - lt[0]
    bins, lf, ssp_ages = sps_lf['bins'], sps_lf['lf'], 10**sps_lf['ssp_ages']
        
    target_lt = np.atleast_1d(lookback_time)
    #set-up output
    int_lf = np.zeros( [ len(target_lt), len(bins) ] )
    aw = np.zeros( [ len(target_lt), len(ssp_ages) ] )

    for i,tl in enumerate(target_lt):
        valid = (lt >= tl) #only consider time points in the past of this lookback time.
        inds, weights = weights_1DLinear(np.log(ssp_ages), np.log(lt[valid] - tl))
        #aggregate the weights for each ssp time index, after accounting for SFR
        agg_weights = np.bincount( inds.flatten(),
                                   weights = (weights * sfr[valid,None]).flatten(),
                                   minlength = len(ssp_ages) ) * dt
        int_lf[i,:] = (lf * agg_weights[:,None]).sum(axis = 0)
        aw[i,:] = agg_weights

    return bins, int_lf, aw

def gauss(x, mu, A, sigma):
    """
    Project the sum of a sequence of gaussians onto the x vector,
    using broadcasting.

    :param x: ndarray
        The array onto which the gaussians are to be projected.
        
    :param mu:
        Sequence of gaussian centers, same units as x.

    :param A:
        Sequence of gaussian normalization (that is, the area of the
        gaussians), same length as mu.
        
    :param sigma:
        Sequence of gaussian standard deviations or dispersions, same
        length as mu.

    :returns value:
       The value of the sum of the gaussians at positions x.
        
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A/(sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:,None] - mu)**2/(2 * sigma**2))
    return val.sum(axis = -1)


def convert_burst_pars(fwhm_burst = 0.05, f_burst=0.5, contrast=5,
                       bin_width=1.0, bin_sfr=1e9):

    """
    Perform the conversion from a burst fraction, width, and
    'contrast' to to a set of gaussian bursts stochastically
    distributed in time, each characterized by a burst time, a width,
    and an amplitude.  Also returns the SFR in the non-bursting mode.

    :param fwhm_burst: default 0.05
        The fwhm of the bursts to add, in Gyr.
        
    :param f_burst: default, 0.5
        The fraction of stellar mass formed in each bin that is formed
        in the bursts.
        
    :param contrast: default, 5
        The approximate maximum height or amplitude of the bursts
        above the constant background SFR.  This is only approximate
        since it is altered to preserve f_burst and fwhm_burst even
        though the number of busrsts is quantized.

    :param bin_width: default, 1.0
        The width of the bin in Gyr.

    :param bin_sfr:
        The average sfr for this time period.  The total stellar mass
        formed during this bin is just bin_sfr * bin_width.

    :returns a:
        The sfr of the non bursting constant component

    :returns tburst:
        A sequence of times, of length nburst, where the time gives
        the time of the peak of the gaussian burst
        
    :returns A:
        A sequence of normalizations of length nburst.  each A value
        gives the stellar mass formed in that burst.

    :returns sigma:
        A sequence of burst widths.  This is usually just
        fwhm_burst/2.35 repeated nburst times.
    """
    
    #print(bin_width, bin_sfr)
    width, mstar = bin_width, bin_width * bin_sfr
    if width < fwhm_burst * 2:
        f_burst = 0.0 #no bursts if bin is short - they are resolved
    #constant SF component
    a = mstar * (1 - f_burst) /width
    #determine burst_parameters
    sigma = fwhm_burst / 2.355
    maxsfr = contrast * a
    A = maxsfr * (sigma * np.sqrt(np.pi * 2))
    tburst = []
    if A > 0:
        nburst = np.round(mstar * f_burst / A)
        #recalculate A to preserve total mass formed in the face of burst number quntization
        if nburst > 0:
            A = mstar * f_burst / nburst
            tburst = np.random.uniform(0,width, nburst)
        else:
            A = 0
            a = mstar/width
    else:
        nburst = 0
        a = mstar/width
        
    
    #print(a, nburst, A, sigma)
    return [a, tburst, A, sigma]

#def redden_analytic(wave, spec, av = None, dav = None,
#                    dust_curve = None, wlo = 1216., whi = 2e4, **kwargs):
# Incorrect.  
#    k = dust_curve(wave)
#    alambda = av / (np.log10(av+dav)) * ( 10**(-0.4 * k * (av+dav)) - 10**(-0.4 * k * av))
#    spec_red = spec * alambda
#    return spec_red, None
        
def redden(wave, spec, av=None, dav=None, nsplit=9,
           dust_curve=None, wlo=1216., whi=2e4, **kwargs):
    
    """
    Redden the spectral components.  The attenuation of a given
    star is given by the model av + U(0,dav) where U is the uniform
    random function.  Extensive use of broadcasting.

    :param wave:  ndarray of shape (nwave)
        The wavelength vector.
    
    :param spec: ndarray of shape (nspec, nwave)
        The input spectra to redden. nspec is the number of spectra.
        
    :param av: scalar or ndarray of shape (nspec)
        The attenuation at V band, in magnitudes, that affects all
        stars equally.  Can be a scalar if its the same for all
        spectra or an ndarray to apply different reddenings to each
        spectrum.

    :param dav: scalar or ndarray of shape (nspec)
        The maximum differential attenuation, in V band magnitudes.
        Can be a scalar if it's the same for all spectra or an array
        to have a different value for each spectrum.  Stars are
        assumed to be affected by an random additional amount of
        attenuation uniformly distributed from 0 to dav.

    :param nsplit: (default 10.0)
        The number of pieces in which to split each spectrum when
        performing the integration over differntial attenuation.
        Higher nsplit ensures greater accuracy, especially for very
        large dav.  However, because of the broadcasting, large nsplit
        can result in memory constraint issues.

    :param dust_curve: function
        The function giving the attenuation curve normalized to the V
        band, \tau_lambda/\tau_v.  This function must accept a
        wavelength vector as its argument and return tau_lambda/tau_v
        at each wavelength.

    :returns spec: ndarray of shape (nwave, nspec)
        The attenuated spectra.

    :returns lir: ndarray of shape (nspec)
        The integrated difference between the unattenuated and
        attenuated spectra, for each spectrum. The integral is
        performed over the interval [wlo,whi].
        
    """

    if (av is None) and (dav is None):
        return spec, None
    if dust_curve is None:
        print('Warning:  no dust curve was given')
        return spec, None
    #only split if there's a nonzero dAv 
    nsplit = nsplit * np.any(dav > 0) + 1
    lisplit = spec/nsplit
    #enable broadcasting if av and dav aren't vectors
    #  and convert to an optical depth instead of an attenuation
    av = np.atleast_1d(av)/1.086
    dav = np.atleast_1d(dav)/1.086
    lisplit = np.atleast_2d(lisplit)
    #uniform distribution from Av to Av + dAv
    avdist = av[None, :] + dav[None,:] * ((np.arange(nsplit) + 0.5)/nsplit)[:,None]
    #apply it
    #print(avdist.shape)
    ee = (np.exp(-dust_curve(wave)[None,None,:] * avdist[:,:,None]))
    #print(avdist.shape, ee.shape, lisplit.shape)
    spec_red = (ee * lisplit[None,:,:]).sum(axis = 0)
    #get the integral of the attenuated light in the optical-
    # NIR region of the spectrum
    opt = (wave >= wlo) & (wave <= whi) 
    lir = np.trapz((spec - spec_red)[:,opt], wave[opt], axis = -1)
    return np.squeeze(spec_red), lir



def examples(filename = '/Users/bjohnson/Projects/angst/sfhs/angst_sfhs/gr8.lowres.ben.v1.sfh',
             lookback_time = [1e9, 10e9]):
    """
    A quick test and demonstration of the algorithms.
    """
    import matplotlib.pyplot as pl
    import fsps
    sps = fsps.StellarPopulation()
    from sfhutils import load_angst_sfh

    f_burst, fwhm_burst, contrast = 0.5, 0.05 * 1e9, 5
    sfh = load_angst_sfh(filename)
    sfh['t1'] = 10.**sfh['t1']
    sfh['t2'] = 10.**sfh['t2']
    sfh['sfr'][0] *=  1 - (sfh['t1'][0]/sfh['t2'][0])
    sfh[0]['t1'] = 0.
    mtot = ((sfh['t2'] - sfh['t1']) * sfh['sfr']).sum()
    lt, sfr, fb = burst_sfh(fwhm_burst = fwhm_burst, f_burst = f_burst, contrast = contrast, sfh = sfh)

    wave, spec, aw = bursty_sps(lookback_time, lt, sfr, sps)

    pl.figure()
    for i,t in enumerate(lookback_time):
        pl.plot(wave, spec[i,:], label = r'$t_{{lookback}} = ${0:5.1f} Gyr'.format(t/1e9))
    pl.legend()
    pl.xlim(1e3,1e4)
    pl.xlabel('wave')
    pl.ylabel(r'$F_\lambda$')

    fig, ax = pl.subplots(2,1)
    for i,t in enumerate(lookback_time):
        ax[1].plot(10**sps.log_age, aw[i,:], label = r'$t_{{lookback}} = ${0:5.1f} Gyr'.format(t/1e9), marker ='o', markersize = 2)
        print(aw[i,:].sum(), mtot, aw[i,:].sum()/mtot)
    ax[1].set_xlabel('SSP age - lookback time')
    ax[1].set_ylabel('Mass')
    ax[1].legend(loc = 'upper left')

    ax[0].plot(lt, sfr, 'k')
    ax[0].set_xlabel('lookback time')
    ax[0].set_ylabel('SFR')
    ax[0].set_title(r'f$_{{burst}}={0:3.1f}$, fwhm$_{{burst}}=${1:3.0f}Myr, contrast ={2}'.format(f_burst, fwhm_burst/1e6, contrast))
    for t in lookback_time:
        ax[0].axvline(x = t, color = 'r', linestyle =':', linewidth = 5)
    pl.show()
