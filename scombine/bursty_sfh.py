import sys
import numpy as np
from scipy import interpolate

from .sfhutils import weights_1DLinear
from .dust import redden
from sedpy import attenuation

lsun, pc = 3.846e33, 3.085677581467192e18
lightspeed = 2.998e18 #AA/s
#value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )


def burst_sfh(sfh=None, bin_res=50, fwhm_burst=0.05, f_burst=0.0, contrast=5,
              **extras):
    """Given a binned SFH as a numpy structured array, and burst parameters,
    generate a realization of the SFH at high temporal resolution. The output
    time resolution will be approximately fwhm_burst/12 unless no bursts are
    generated, in which case the output time resolution is the minimum bin
    width divided by bin_res.

    :param sfh: structured ndarray
        A binned sfh in numpy structured array format.  Usually the result of
        sfhutils.load_angst_sfh()

    :param bin_res: default 50
        Factor by which to increase the time resolution of the output grid,
        relative to the shortest bin width in the supplied SFH.
    
    :param fwhm_burst: default 0.05
        the fwhm of the bursts to add, in Gyr.
        
    :param f_burst: default, 0
        the fraction of stellar mass formed in each bin that is formed in the
        bursts.  If 0, no bursts are formed.
        
    :param contrast: default, 5
        the approximate maximum height or amplitude of the bursts above the
        constant background SFR.  This is only approximate since it is altered
        to preserve f_burst and fwhm_burst even though the number of busrsts is
        quantized.

    :returns times:  ndarray of shape (nt)
        The output linear, regular temporal grid of lookback times.

    :returns sfr: ndarray of shape (nt)
        The resulting SFR at each time.

    :returns tburst:
       The burst times        
    """
    a, tburst, A, sigma, f_burst_actual = [],[],[],[],[]
    for i,abin in enumerate(sfh):
        res = convert_burst_pars(fwhm_burst=fwhm_burst, f_burst=f_burst,
                                 contrast=contrast,
                                 bin_width=(abin['t2']-abin['t1']),
                                 bin_sfr=abin['sfr'])
        a += [res[0]]
        if len(res[1]) > 0:
            tburst += (res[1] + abin['t1']).tolist()
            A += len(res[1]) * [res[2]]
            sigma += len(res[1]) * [res[3]]
    if len(sigma) == 0:
        # If there were no bursts, set the time resolution to be
        # 1/bin_res of the shortest bin width. This is approximate.
        dt = (sfh['t2'] - sfh['t1']).min()/(1.0 * bin_res)
    else:
        # make sure you sample the bursts reasonably well
        dt = np.min(sigma) / 5. 
    times = np.arange(np.round(sfh['t2'].max()/dt)) * dt
    sfr = gauss(times, tburst, A, sigma)
    # Figure out which bin each time is in and add the SFR of that bin
    # to that time.
    bins = [sfh[0]['t1']] + sfh['t2'].tolist()
    bin_num = np.digitize(times, bins) - 1
    has_bin = bin_num >= 0
    sfr[has_bin] += np.array(a)[bin_num[has_bin]]
    
    return times, sfr, tburst


def tau_burst_sfh(fwhm_burst=0.05, f_burst=0.5, contrast=5, mass=0.0,
                  bin_res=50., tau=100e9, t_mass=0.0, tstart=13.7e9,
                  sftype='tau', gamma=1.4):
    """Given a binned SFH as a numpy structured array, construct an SFH that is
    composed of a smooth rising or falling SFH with superposed bursts, subject
    to a constraint on the total stellar mass formed.

    :param sftype:
        The SFH form for the smooth component.  One of:
        * ``linear rising``: SFR~t/tau
        * ``tau``: SFR~e^{-t/tau}
        * ``power-law``: SFR~(t/tau)^gamma
        
    :param tstart:
        Lookback time of the start of the SFH, in yrs (e.g.
        astropy.cosmology.WMAP9.lookback_time(10).value * 1e9)

    :param t_mass:
        The lookback time (in yrs) at which the total stellar mass formed is
        geven by ``mass``.  Must be less than tstart.
        
    :param mass:
        The total stellar mass formed by t_mass.

    :param tau:
        Timescale parameter

    :returns time:
        Lookback time in yrs

    :returns sfr:
        SFR (M_sun/yr) at the locations of time.

    :returns bursts:
        Two-element tuple composed of arrays of the lookback times and total
        stellar mass formed in each burst.
    """
    assert t_mass < tstart
    
    bin_width = tstart - t_mass
    bin_sfr = mass / bin_width # average SFR

    sigma_burst = fwhm_burst/2.35
    dt = sigma_burst/5.
    if dt == 0:
        dt = bin_width/bin_res
    times = np.arange(np.round(bin_width/dt)) * dt + t_mass

    normalized_times = (tstart - times) / tau

    if sftype == 'linear rising':
        sfr = normalized_times
        int_sfr = tau/2.0 * normalized_times[0]**2
    elif sftype == 'power-law':
        sfr = normalized_times**gamma
        int_sfr = (tau/(gamma+1)) * normalized_times[0]**(gamma+1)
    elif sftype == 'exponential rising':
        raise(NotImplementedError)
    elif sftype == 'delayed tau':
        raise(NotImplementedError)
    else:
        # Standard tau decline
        sfr = np.exp(-normalized_times)
        int_sfr = tau * (1.0 - np.exp(-normalized_times[0]))

    smooth_mass = mass * (1.0 - f_burst)
    sfr *= smooth_mass/int_sfr

    #start adding bursts
    burst_mass_total, A_burst, t_burst = 0.0, [], []
    while burst_mass_total < mass*f_burst:
        t_burst.append(np.random.uniform(t_mass, tstart))
        sfr_burst = np.interp(t_burst[-1], times, sfr) * contrast
        A_burst.append(sfr_burst * sigma_burst * np.sqrt(2*np.pi))
        burst_mass_total += A_burst[-1]
    t_burst = np.array(t_burst)
    A_burst = np.array(A_burst)
    # flip a coin to decide whether to go over or under the desired
    # burst mass
    if np.random.uniform(0,1) < 0.5:
        t_burst = t_burst[:-1]
        A_burst = A_burst[:-1]
    # Adjust burst amplitudes to get the correct burst_fraction (and
    # total mass) in the face of burst quantization
    A_burst *= mass*f_burst/A_burst.sum()
    sfr = sfr + gauss(times, t_burst, A_burst, sigma_burst)

    return times, sfr, (t_burst, A_burst)


def smooth_sfh(sfh=None, bin_res=10., **kwargs):
    """Method to produce a smooth SFH from a given step-wise SFH, under the
    constraint that the  total integrated mass at the end of each 'step' is
    preserved.  Uses a cubic spline with a monotonicity constraint to obtain
    M_tot(t) which is then differentiated to produce the SFH.

    :param sfh:
        A stepwise SFH in the format produced by `load_angst_sfh()`, but with
        the time fields converted to linear time. A structured array.
       
    :param bin_res: default 10
        Factor by which to increase the time resolution of the output grid,
        relative to the shortest bin width in the supplied SFH.

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

    pcint = interpolate.PchipInterpolator
    monospline = pcint(np.concatenate([[0],tt[::-1]]),
                       mtot * np.concatenate([[0],sfh['mformed'][::-1]]))
    
    dt = (sfh['t2'] - sfh['t1']).min()/(1.0 * bin_res)
    times = tmax - np.arange(np.round(sfh['t2'].max()/dt)) * dt
    return tmax - times, monospline.derivative(der=1)(times)


def bursty_sps(lt, sfr, sps, lookback_time=[0], logzsol=None,
               dust_curve=attenuation.cardelli, av=None, dav=None, nsplit=9,
               **extras):
    """Obtain the spectrum of a stellar poluation with arbitrary complex SFH at
    a given lookback time.  The SFH is provided in terms of SFR vs
    t_lookback. Note that this in in contrast to the normal specification in
    terms of time since the big bang. Interpolation of the available SSPs to
    the time sequence of the SFH is accomplished by linear interpolation in log
    t.  Highly oscillatory SFHs require dense sampling of the temporal axis to
    obtain accurate results.

    :param lt: ndarray, shape (ntime)
        The lookback time sequence of the provided SFH. 
        
    :param sfr: ndarray, shape (ntime)
        The SFR corresponding to each element of lt, in M_sun/yr.
        
    :param sps: fsps.StellarPopulation instance
        The fsps stellar population (with metallicty and IMF parameters set) to
        use for the SSP spectra.

    :param lookback_time: scalar or ndarray, shape (ntarg)
        The lookback time(s) at which to obtain the spectrum. In yrs.  Defaults
        to [0].

    :param av: scalar or ndarray of shape (nspec)
        The attenuation at V band, in magnitudes, that affects all stars
        equally. Passed to redden().

    :param dav: scalar or ndarray of shape (nspec)
        The maximum differential attenuation, in V band magnitudes. Passed to
        redden().

    :param logzsol: ndarray of shape (nspec)
        The metallicity in units of log(Z/Z_sun) corresponding to each SSP age.

    :returns wave: ndarray, shape (nwave)
        The wavelength array
        
    :returns int_spec: ndarray, shape(ntarg, nwave)
        The integrated spectrum at lookback_time, in L_sun/AA
        
    :returns mstar: ndarray, shape(ntarg, nage)
        The total stellar mass, excluding mass returned to the ISM by stellar
        evolution.
        
    :returns lir: ndarray, shape(ntarg)
        The total absorbed luminosity, in L_sun. 
    """
    # get *all* the ssps
    sps.params['sfh'] = 0  # make sure SSPs
    ssp_ages = 10**sps.ssp_ages  # in yrs
    if logzsol is None:
        wave, spec = sps.get_spectrum(peraa=True, tage=0)
        mass = sps.stellar_mass.copy()
    else:
        assert(sps._zcontinuous > 0)
        spec, mass = [], []
        for tage, logz in zip(ssp_ages/1e9, logzsol):
            sps.params['logzsol'] = logz
            spec.append(sps.get_spectrum(peraa=True, tage=tage)[1])
            mass.append(sps.stellar_mass)
        spec = np.array(spec)
        mass = np.array(mass)
        wave = sps.wavelengths
        
    # Redden the SSP spectra
    spec, lir = redden(wave, spec, av=av, dav=dav,
                       dust_curve=dust_curve, nsplit=nsplit)

    # Get interpolation weights based on the SFH
    target_lt = np.atleast_1d(lookback_time)
    aw = sfh_weights(lt, sfr, ssp_ages, lookback_time=target_lt, **extras)
    # Do the linear combination
    int_spec = (spec[None,:,:] * aw[:,:,None]).sum(axis=1)
    mstar = (mass[None,:] * aw).sum(axis=-1)
    if lir is not None:
        lir_tot = (lir[None,:] * aw).sum(axis = -1)
    else:
        lir_tot = 0
    return wave, int_spec, mstar, lir_tot


def bursty_lf(lt, sfr, sps_lf, lookback_time=[0], **extras):
    """Obtain the luminosity function of stars for an arbitrary complex SFH at
    a given lookback time.  The SFH is provided in terms of SFR vs
    t_lookback. Note that this in in contrast to the normal specification in
    terms of time since the big bang.

    :param lt: ndarray, shape (ntime)
        The lookback time sequence of the provided SFH.  Assumed to have have
        equal linear time intervals.

    :param sfr: ndarray, shape (ntime)
        The SFR corresponding to each element of lt, in M_sun/yr.

    :param sps_lf:
        Luminosity function information, as a dictionary.  The keys of the
        dictionary are 'bins', 'lf' and 'ssp_ages'

    :param lookback_time: scalar or ndarray, shape (ntarg)
        The lookback time(s) at which to obtain the spectrum. In yrs.

    :returns bins:
        The bins used to define the LF

    :returns int_lf: ndarray, shape(ntarg, nbin)
        The integrated LF at lookback_time, in L_sun/AA

    :returns aw: ndarray, shape(ntarg, nage)
        The total weights of each LF for each requested lookback_time.  Useful
        for debugging.
    """
    bins, lf, ssp_ages = sps_lf['bins'], sps_lf['lf'], 10**sps_lf['ssp_ages']        
    target_lt = np.atleast_1d(lookback_time)
    aw = sfh_weights(lt, sfr, ssp_ages, lookback_time=target_lt, **extras)
    int_lf = (lf[None,:,:] * aw[:,:,None]).sum(axis=1)
    return bins, int_lf, aw


def sfh_weights(lt, sfr, ssp_ages, lookback_time=0, renormalize=False,
                log_interp=False, **extras):
    """        
    :param lt: ndarray, shape (ntime)
        The lookback time sequence of the provided SFH.  Assumed to have have
        equal linear time intervals.
        
    :param sfr: ndarray, shape (ntime)
        The SFR corresponding to each element of lt, in M_sun/yr.

    :param ssp_ages: ndarray, shape (nage)
        The ages at which you want weights.  Linear yrs.

    :param lookback_time: scalar or ndarray, shape (ntarg)
        The lookback time(s) at which to obtain the spectrum. In yrs.

    :param log_interp: (default: False)
        Do the interpolation weights in log time
        
    :returns aw: ndarray, shape(ntarg, nage)
        The total weights of each LF for each requested lookback_time.  Useful
        for debugging.
    """
    dt = lt[1] - lt[0]
    target_lt = np.atleast_1d(lookback_time)
    aw = np.zeros( [ len(target_lt), len(ssp_ages) ] )
    for i,tl in enumerate(target_lt):
        valid = (lt >= tl) #only consider time points in the past of this lookback time.
        #augment the t_lookback array of the SFH with the SSP ages
        sfr_ssp = np.interp(ssp_ages, lt-tl, sfr, left=0.0, right=0.0)
        tmp_t = np.concatenate([ssp_ages, lt[valid]-tl])
        tmp_sfr = np.concatenate([sfr_ssp, sfr[valid]])
        #sort the augmented array by lookback time
        order = tmp_t.argsort()
        tmp_t = tmp_t[order]
        tmp_sfr = tmp_sfr[order]
        # get weights to interpolate the log_t array
        if log_interp:
            inds, weights = weights_1DLinear(np.log(ssp_ages), np.log(tmp_t), **extras)
        else:
            inds, weights = weights_1DLinear(ssp_ages, tmp_t, **extras)
        # aggregate the weights for each ssp time index, after
        # accounting for SFR *dt
        tmp_dt = np.gradient(tmp_t)
        agg_weights = np.bincount( inds.flatten(),
                                   weights = (weights * tmp_sfr[:, None] *
                                              tmp_dt[:, None]).flatten(),
                                   minlength = len(ssp_ages) )
        aw[i,:] = agg_weights
        if renormalize:
            aw[i,:] /= agg_weights.sum()
    return aw


def gauss(x, mu, A, sigma):
    """Project the sum of a sequence of gaussians onto the x vector, using
    broadcasting.

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
    if len(mu) == 0:
        return np.zeros_like(x)
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A/(sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:,None] - mu)**2/(2 * sigma**2))
    return val.sum(axis = -1)


def convert_burst_pars(fwhm_burst=0.05, f_burst=0.5, contrast=5,
                       bin_width=1.0, bin_sfr=1e9):
    """Perform the conversion from a burst fraction, width, and 'contrast' to
    to a set of gaussian bursts stochastically distributed in time, each
    characterized by a burst time, a width, and an amplitude.  Also returns the
    SFR in the non-bursting mode.

    :param fwhm_burst: default 0.05
        The fwhm of the bursts to add, in Gyr.
        
    :param f_burst: default, 0.5
        The fraction of stellar mass formed in each bin that is formed in the
        bursts.
        
    :param contrast: default, 5
        The approximate maximum height or amplitude of the bursts above the
        constant background SFR.  This is only approximate since it is altered
        to preserve f_burst and fwhm_burst even though the number of busrsts is
        quantized.

    :param bin_width: default, 1.0
        The width of the bin in Gyr.

    :param bin_sfr:
        The average sfr for this time period.  The total stellar mass formed
        during this bin is just bin_sfr * bin_width.

    :returns a:
        The sfr of the non bursting constant component

    :returns tburst:
        A sequence of times, of length nburst, where the time gives the time of
        the peak of the gaussian burst
        
    :returns A:
        A sequence of normalizations of length nburst.  each A value gives the
        stellar mass formed in that burst.

    :returns sigma:
        A sequence of burst widths.  This is usually just fwhm_burst/2.35
        repeated nburst times.
    """
    width, mstar = bin_width, bin_width * bin_sfr
    if width < fwhm_burst * 2:
        # No bursts if bin is short - they are resolved
        f_burst = 0.0 
    # Constant SF component
    a = mstar * (1 - f_burst) /width
    # Determine burst_parameters
    sigma = fwhm_burst / 2.355
    maxsfr = contrast * a
    A = maxsfr * (sigma * np.sqrt(np.pi * 2))
    tburst = []
    if A > 0:
        nburst = np.round(mstar * f_burst / A)
        # Recalculate A to preserve total mass formed in the face of
        # burst number quantization
        if nburst > 0:
            A = mstar * f_burst / nburst
            tburst = np.random.uniform(0,width, nburst)
        else:
            A = 0
            a = mstar/width
    else:
        nburst = 0
        a = mstar/width

    return [a, tburst, A, sigma]


def examples(filename='demo/sfhs/ddo75.lowres.ben.v1.sfh',
             lookback_time=[0.0, 1e9, 10e9]):
    """
    A quick test and demonstration of the algorithms.
    """
    import matplotlib.pyplot as pl
    import fsps
    from scombine.sfhutils import load_angst_sfh

    # Instantiate the SPS object and make any changes to the parameters here
    sps = fsps.StellarPopulation(zcontinuous=1)
    sps.params['logzsol'] = -1.0

    # Load the input SFH, and set any bursts if desired (set f_burst=0
    # to not add bursts)
    f_burst, fwhm_burst, contrast = 0.5, 0.05 * 1e9, 5
    sfh = load_angst_sfh(filename)
    sfh['t1'] = 10.**sfh['t1']
    sfh['t2'] = 10.**sfh['t2']
    sfh['sfr'][0] *=  1 - (sfh['t1'][0]/sfh['t2'][0])
    sfh[0]['t1'] = 0.
    mtot = ((sfh['t2'] - sfh['t1']) * sfh['sfr']).sum()
    
    # generate a high temporal resolution SFH, with bursts if f_burst > 0
    lt, sfr, tb = burst_sfh(sfh=sfh, fwhm_burst=fwhm_burst, f_burst=f_burst, contrast=contrast)
    # get the interpolation weights.  This does not have to be run in
    # general (it is run interior to bursty_sps) unless you are
    # debugging or for plotting purposes
    aw = sfh_weights(lt, sfr, 10**sps.ssp_ages, lookback_time=lookback_time)
    # get the intrinsic spectra at the lookback_times specified.
    wave, spec, mstar, _ = bursty_sps(lt, sfr, sps, lookback_time=lookback_time)
    # get reddened spectra, Calzetti foreground screen
    wave, red_spec, _, lir = bursty_sps(lt, sfr, sps, lookback_time=lookback_time,
                                        dust_curve=attenuation.calzetti, av=1, dav=0)
    # get reddened spectra, SexA differntial extinction plus SMC
    from scombine.dust import sexAmodel
    dav = sexAmodel(davmax=1.0, ages=10**sps.ssp_ages)
    wave, red_spec, _, lir = bursty_sps(lt, sfr, sps, lookback_time=lookback_time,
                                        dust_curve=attenuation.smc, av=1, dav=dav)
    
    # Get intrinsic spectrum including an age metallicity relation
    def amr(ages, **extras):
        """This should take an array of ages (linear years) and return an array
        of metallicities (units of log(Z/Z_sun)
        """
        logz_array = -1.0 * np.ones_like(ages)
        return logz_array
    wave, spec, mstar, _ = bursty_sps(lt, sfr, sps, lookback_time=lookback_time,
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
