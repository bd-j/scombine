import glob
import numpy as np
import matplotlib.pyplot as pl

loglsol = 3.839e33
pc2cm = 3.086e18
magsphere = 4.*np.pi*100*pc2cm**2
skiprows = 0 #number of extra rows at the top of the SFH files

def load_angst_sfh(name, sfhdir = '', skiprows = skiprows):
    """Read a `match`-produced SFH file into a numpy structured array.

    :param name:
        string giving the name (and optionally the path) of the SFH file
    :param skiprows:
        number of header rows in the SFH file to skip
    """
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty), ('mformed',ty)])
    #fn = glob.glob("{0}*{1}*sfh".format(sfhdir,name))[0]
    fn = name
    data = np.loadtxt(fn, usecols = (0,1,2,3,6,12) ,dtype = dt, skiprows = skiprows)

    return data


def read_fsps(filename):
    """Read a .spec file produced by FSPS and return a number of arrays
    giving quantities of the stellar population.  These are:
    [age (in Gyr), log(Mstar), log(Lbol), log(SFR), spectrum, wavelength, header]

    :param filename:
        The file name of the FSPS output .spec file, including path
    :returns age:
        The age of the stellar population (in Gyr).  1D array
    :returns mstar:
        The log of the stellar mass (in solar units) at `age'.  1D array
    :returns lbol:
        The log of the bolometric luminosity at `age'.  1D array
    :returns sfr:
        The log of the SFR (M_sun/yr) at `age'.  1D array
    :returns spectrum:
        The spectrum of the stellar population at `age'.  (L_sun/AA), 2D array
    :returns wavelngth:
        The wavelength vector (AA).  1D array
    :returns header:
        String array of header information in the .spec file
    """
    age, logmass, loglbol, logsfr, spec, header =[],[],[],[],[], []
    with open(filename, "r") as f:
        line = f.readline()
        while line[0] is '#':
            header.append(line)
            line = f.readline()
        nt, nl = line.split()
        wave = [float(w) for w in f.readline().split()]
        for i in range(int(nt)):
            a, m, l, s = f.readline().split()
            age.append(10**float(a)/1e9)
            logmass.append(float(m))
            loglbol.append(float(l))
            logsfr.append(float(s))
            s = [float(l) for l in f.readline().split()]
            spec.append(s)
        f.close()

    return np.array(age), np.array(logmass), np.array(loglbol), np.array(logsfr), np.array(spec), np.array(wave), header



def weights_1DLinear(model_points, target_points):
    #well this is ugly.
    order = model_points.argsort()
    mod_sorted = model_points[order]
    ind_nearest = np.searchsorted(mod_sorted, target_points,side='left')
    
    maxind = mod_sorted.shape[0]-1
    edge = ( ind_nearest == 0) | ( ind_nearest == (maxind+1) )
    inds = (np.vstack([order[np.clip(ind_nearest, 0, maxind)],
                       order[np.clip(ind_nearest-1, 0, maxind)]])).T
    
    d1 = np.abs( model_points[inds[:,0]] - target_points )
    d2 = np.abs( model_points[inds[:,1]] - target_points )
    width = d1+d2
    width[edge] = 1
    weights = np.vstack([1-d1/width, 1-d2/width]).T
    weights[edge,:]=0.5

    return inds, weights



def plot_agespec(name, sfh, flux, rp, rebin, ylim = None):
    pl.figure(1)
    for i in xrange(rp['nrebin']):
        binned_flux = flux[:, rebin['start'][i]:rebin['end'][i]+1].sum(axis =1)
        label = '{0:3.2f} <log t<{1:4.2f}'.format(sfh['t1'][rebin['start'][i]], sfh['t2'][rebin['end'][i]])
        
        pl.plot(rp['wave'], binned_flux*(magsphere/loglsol), label = label, color = rebin['color'][i])

    pl.plot(rp['wave'], flux.sum(axis = 1)*(magsphere/loglsol), label = 'total', color = 'k')
    pl.title(name)
    pl.xlabel('wave')
    pl.ylabel(r'$L_{\lambda,bin}/M_{*,tot} (L_{\odot}/M_{\odot}/\AA)$')
    pl.yscale('log')
    pl.xscale('log')
    pl.xlim(2e3, 2e4)
    if ylim is not None: pl.ylim(ylim[0], ylim[1])
    pl.legend()
    pl.savefig('{0}_agespec.png'.format(name))
    pl.close()
