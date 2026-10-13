"""Shared settings for the two 1 degree configurations on the f19_g16 grid: iCESM1.3 (LR; Zhu et al. 2017, GRL,
doi:10.1002/2017GL073406) and CESM1.2. Run names, analysis windows (the last 20 years of each run), control spans,
and a loader for the annual Atlantic overturning streamfunction computed offline from monthly VVEL.
"""
import glob
import os
import numpy as np
import netCDF4 as nc

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
ATL = 1
DIR = DATA + '/casper_vvel/f19_mocz'
WIN = {'LGM': ('F19LGM', 981, 1000), 'PI': ('F19PI', 485, 504)}
CTRL = ('F19PI', 105, 504)
R18O = DIR + '/f19_r18o_%s.nc'   # mean seawater d18O of each analysis window

# CESM1.2 is the f19_g16 pair of the PMIP4 ensemble (Tierney et al. 2020; Zhu & Poulsen 2021), 900 years each,
# on the same gx1v6 ocean. Windows: last 20 years of each run. Control: PI years 101-900.
CONF = {'f19': dict(dir=DIR, LGM=WIN['LGM'], PI=WIN['PI'], CTRL=CTRL, label='CESM1.3 (f19_g16)'),
        'c12': dict(dir=DATA + '/casper_vvel/c12_mocz', LGM=('C12LGM', 881, 900),
                    PI=('C12PI', 881, 900), CTRL=('C12PI', 101, 900), label='CESM1.2 (f19_g16)')}
RUN_DIR = {c[s][0]: c['dir'] for c in CONF.values() for s in ('LGM', 'PI')}
R18O_ANY = {'F19LGM': R18O % 'F19LGM', 'F19PI': R18O % 'F19PI',
            'C12LGM': CONF['c12']['dir'] + '/c12_r18o_C12LGM.nc', 'C12PI': CONF['c12']['dir'] + '/c12_r18o_C12PI.nc'}


def load_annual(run, y0, y1):
    """(lat, z [m], psi[year, z, lat], years) for Atlantic psi, rows sorted by latitude."""
    root = RUN_DIR.get(run, DIR)
    fs = sorted(glob.glob('%s/mocz_%s_*_annual.nc' % (root, run)))
    assert fs, 'no annual psi for %s in %s' % (run, root)
    P, Y = [], []
    for f in fs:
        d = nc.Dataset(f)
        P.append(np.array(d.variables['MOC'][ATL]).astype('f8')); Y.append(np.array(d.variables['year'][:]))
        z = np.array(d.variables['z_t'][:]) / 100.0; tl = np.array(d.variables['TLAT_1D'][:]); d.close()
    psi, yrs = np.concatenate(P, axis=0), np.concatenate(Y)
    o = np.argsort(yrs); psi, yrs = psi[o], yrs[o]
    assert np.all(np.diff(yrs) == 1), '%s: years not contiguous' % run
    sel = (yrs >= y0) & (yrs <= y1)
    assert sel.sum() == y1 - y0 + 1, '%s: %d of %d years present' % (run, sel.sum(), y1 - y0 + 1)
    lat = np.asarray(tl[ATL]).ravel()[:psi.shape[-1]]; ok = np.isfinite(lat); oo = np.argsort(lat[ok])
    return lat[ok][oo], z, psi[sel][:, :, ok][:, :, oo], yrs[sel]
