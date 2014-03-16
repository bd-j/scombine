import scipy.interpolate as interpolate

times = np.arange(2000)* 1e7

def smooth_sfh(bin_sfh, times):
    mtot = ((10**bin_sfh['t2'] - 10**bin_sfh['t1']) * bin_sfh['sfr'] ).sum()
    tt = 10**bin_sfh[-1]['t2'] - 10**bin_sfh['t1'] 
    tt[0] = 10**bin_sfh[-1]['t2']
    
    monospline = interpolate.PchipInterpolator(np.concatenate([[0],tt[::-1]]),
                                               mtot * np.concatenate([[0],bin_sfh['mformed'][::-1]]))
    return monospline.derivative(times)



def smooth_spec(bin_sfh, zmet = 1.0, imf_type = 0, outroot = 'S0',  t_lookback = [0],
                   narr = 2000, t0 = 0.0005 * 1e9, clobber = False):

    """Method to produce an FSPS spectrum for a given stwp function SFH.
    Uses subprocess to call autosps from FSPS after generating a smooth SFH from a
    SFH file, which is a pretty hacky workaround.  Requires
    FSPS to be installed and $SPS_HOME to point to the correct fsps directory.

    :param bin_sfh:
        a numpy structured array from load_sfh
        
    :param zmet: (default: 1.0)
        Metallicity in units of solar (linear)
        
    :param imf_type: (default: 0)
        The IMF to use.  Defaults to Salpeter. see FSPS manual for details.
        
    :param narr:
        Number of linear time points to use for defining each bin in the sfh.dat
        user file.
        
    :param t0:
        Extra time padding to add to the beginning of each bin, to please FSPS.
        
    :param outroot:
        Output file name root.  Metallicity and IMF info will be automatically
        appended to this.
        
    :param t_lookback:
        lookback time (in yrs) at which to calculate the spectrum of each bin of SF.
        If it's before the beginning of a given bin then the spectrum and mass for
        that bin are set to zero.

    :param clobber: (deafult False)
        If False, then if the filename already exists do not recompute the basis and
        return the filename.  Otherwise, recompute the basis.
        
    :returns outname:
        A string giving the path and filename of the produced basis file.  The basis file
        consists of  an array of shape [NBIN+1, NWAVE+1] where the extra indices are to
        store the wavelength vector and the stellar mass vector.  Otherwise the data consists
        of the spectrum of each bin (top-hat SFH) at time t_lookback, in units of
        erg/s/cm**2/AA per M_sun/yr at 10pc distance.
    """

    outname = '{2}_{0}_z{1:.1f}.fits'.format(imfname[imf_type],np.log10(zmet), outroot)
    if os.path.exists(outname) and (clobber is False):
        return outname

    sfh = bin_sfh.copy()
    time = np.arange(narr) * 10**sfh[-1]['t2']/narr
    
    sfr = smooth_sfh(sfh, time)
    time += t0
    nlead = 20.
    leadin = t0/nlead
    
    # Produce and read an FSPS spectral file for arbitrary SFH
    a_gyr, logm, logl, logs, spec, wave, hdr = get_fsps_spectrum(time, sfr, zmet, imf_type, leadin)

    # If necessary build the output
    if i is 0:
        aspec = np.zeros([len(t_lookback), len(wave)])
        amass= np.zeros(len(t_lookback))
    
    # Interpolate the fsps spectra (and masses) to the present day
    #   (or to t_lookback), taking into account the non-zero begining of the SFH
    #   This is inefficient for multiple t_lookbacks...
    tprime = sfh[-1]['t2'] + t0 - t_lookback
    if tprime > 0:
        inds, weights = utils.weights_1DLinear( np.log(a_gyr), np.log(tprime / 1e9) )
        aspec = to_cgs * (( weights* (spec[inds].transpose(2,0,1)) ).sum(axis=2)).T
        amass = (weights * 10**logm[inds]).sum(axis=1)
