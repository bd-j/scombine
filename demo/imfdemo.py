import numpy as np
from scombine.imfvariance import *
mass = np.arange(0.08, 120, 0.005)
phis = salpeter(mass)

#set a mass range
minds = (mass >= 0.10) & (mass <= 100.0)

#number of stars
nstar = np.trapz(phis[minds], mass[minds])

#mass of stars
mstar = np.trapz((phis*mass)[minds], mass[minds])

#average mass
m_avg = mstar/nstar

#fraction of stars above 1 M_sun
gt1 = mass[minds] > 1.0
n_gt1 = np.trapz(phis[minds][gt1], mass[minds][gt1])
frac_gt1 = n_gt1/nstar

vals = (mass[minds].min(), mass[minds].max(), nstar, mstar, m_avg, frac_gt1)
print(' M_min={0}\n M_max={1}\n N_*={2}\n M_*={3}\n <M_*>={4}\n f(M>1)={5}'.format(*vals))

print('N(m>1|M=1e6) = {0}'.format(n_gt1 * 1e6/mstar))
