#!/usr/bin/env python
"""Figure 1: Atlantic overturning streamfunction (latitude, depth) for HR (0.1 degree), LR (1 degree, f19_g16) and
CESM1.2 (1 degree, f19_g16); columns LGM, PI and LGM minus PI. Dots mark the points where the difference is NOT
significant (moving-block bootstrap of annual means, 2-year blocks, 4000 draws; Benjamini-Hochberg false discovery
rate q = 0.15). Input: the overturning files of each run. Output: fig1_overturning.png.
"""
import os, sys, warnings
import numpy as np, netCDF4 as nc
import ultraplot as uplt
from f19_common import load_annual
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
ATL = 1
FILES = {('ne120', 'LGM'): DATA + '/cesm/icesm13/archive/b.e13.Bi1850C5.ne120_t12.lgm.test.001/ocn/proc/mocz/mocz_1-15_all.nc',
         ('ne120', 'PI'):  DATA + '/casper_vvel/PI_ne120/mocz_1-20_all.nc'}
F19 = {('f19', 'LGM'): ('F19LGM', 981, 1000), ('f19', 'PI'): ('F19PI', 485, 504),
       ('c12', 'LGM'): ('C12LGM', 881, 900), ('c12', 'PI'): ('C12PI', 881, 900)}
# expected window-mean strength (Sv) and cell base (m) of each state
GATE = {('ne120', 'LGM'): (11.12, 2110), ('ne120', 'PI'): (18.55, 2794), ('f19', 'LGM'): (25.34, 2701),
        ('f19', 'PI'): (22.33, 3210), ('c12', 'LGM'): (24.41, 2603), ('c12', 'PI'): (23.73, 2820)}
ROWS = [('ne120', 'HR (0.1°)'), ('f19', 'LR (1°)'), ('c12', 'CESM1.2 (1°)')]
B, L, Q = 4000, 2, 0.15
DOTS = 'not significant'
MINUS = lambda v, f: (f % v).replace('-', '\u2212')
rng = np.random.default_rng(20260811)


def load_monthly_annual(path):
    d = nc.Dataset(path); moc = np.array(d.variables['MOC'][ATL]).astype('f8'); z = np.array(d.variables['z_t'][:]) / 100.0
    tl = np.array(d.variables['TLAT_1D'][:]); d.close()
    if tl.ndim == 3: tl = np.nanmean(tl, axis=0)
    assert moc.shape[0] % 12 == 0, '%s: %d records is not whole years' % (path, moc.shape[0])
    ann = np.nanmean(moc.reshape(moc.shape[0] // 12, 12, *moc.shape[1:]), axis=1)
    nlat = moc.shape[-1]; lat = np.asarray(tl[ATL]).ravel()[:nlat]
    valid = np.isfinite(lat); lat, ann = lat[valid], ann[:, :, valid]; o = np.argsort(lat)
    return lat[o], z, ann[:, :, o]


def strength_core(lat, z, psi):
    zm = np.where((z >= 500) & (z <= 3000))[0]; li = np.where((lat >= 20) & (lat <= 65))[0]
    sub = psi[np.ix_(zm, li)]; s = np.nanmax(sub); zi, _ = np.unravel_index(np.nanargmax(sub), sub.shape)
    return s, z[zm][zi]


def cell_base(lat, z, psi):
    li = np.where((lat >= 30) & (lat <= 45))[0]; p = np.nanmean(psi[:, li], axis=1); k0 = int(np.nanargmax(p))
    for k in range(k0, len(z) - 1):
        if p[k] >= 0 and p[k + 1] < 0:
            f = p[k] / (p[k] - p[k + 1]); return z[k] + f * (z[k + 1] - z[k])
    return np.nan


def mbb_weights(n):
    """Moving-block bootstrap as a (B, n) matrix of resampling weights for the mean."""
    Lb = min(L, n); starts = np.arange(0, n - Lb + 1); k = int(np.ceil(n / Lb))
    pick = rng.integers(0, len(starts), size=(B, k))
    idx = (starts[pick][:, :, None] + np.arange(Lb)[None, None, :]).reshape(B, -1)[:, :n]
    W = np.zeros((B, n)); np.add.at(W, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    assert np.allclose(W.sum(axis=1), n)
    return W / n


def fdr_mask(XL, XP, lat, z):
    """Significance of mean(XL) - mean(XP) per grid point, Benjamini-Hochberg at q = Q over the drawn domain."""
    nz, ny = XL.shape[1:]; F = nz * ny
    xl, xp = XL.reshape(len(XL), F), XP.reshape(len(XP), F)
    dom = ((z[:, None] <= 5000) & (lat[None, :] >= -34) & (lat[None, :] <= 70)).ravel()
    ok = dom & np.all(np.isfinite(xl), 0) & np.all(np.isfinite(xp), 0)
    WL, WP = mbb_weights(len(XL)), mbb_weights(len(XP))
    cols = np.where(ok)[0]; le = np.zeros(F); ge = np.zeros(F)
    for c0 in range(0, cols.size, 4000):
        cc = cols[c0:c0 + 4000]; D = WL @ xl[:, cc] - WP @ xp[:, cc]
        le[cc] = (D <= 0).sum(axis=0); ge[cc] = (D >= 0).sum(axis=0)
    p = np.full(F, np.nan); p[ok] = np.minimum(1.0, 2.0 * (np.minimum(le[ok], ge[ok]) + 1.0) / (B + 1.0))
    ps = np.sort(p[ok]); m = ps.size; below = np.where(ps <= Q * np.arange(1, m + 1) / m)[0]
    pcrit = ps[below[-1]] if below.size else -1.0
    sig = np.zeros(F, bool); sig[ok] = p[ok] <= pcrit
    return sig.reshape(nz, ny), m, int(sig.sum()), pcrit


M = {}
for key in [(r, s) for r, _ in ROWS for s in ('LGM', 'PI')]:
    if key in FILES:
        lat, z, ann = load_monthly_annual(FILES[key])
    else:
        lat, z, ann, yrs = load_annual(*F19[key])
    psi = np.nanmean(ann, axis=0); s, _ = strength_core(lat, z, psi); b = cell_base(lat, z, psi)
    gs, gb = GATE[key]
    if abs(s - gs) > 0.006 or abs(b - gb) > 1.0:
        sys.exit('STOP: %s window-mean strength %.3f Sv / cell base %.1f m, expected %.2f / %d' % (key, s, b, gs, gb))
    M[key] = dict(lat=lat, z=z, ann=ann, psi=psi, s=s, base=b)
    print('%s: %d years, %.2f Sv, base %.0f m [gate passed]' % (key, ann.shape[0], s, b), flush=True)

uplt.rc.update({'font.size': 7.5, 'title.size': 7.5, 'leftlabel.size': 8, 'toplabel.size': 8, 'abc.size': 8,
                'tick.labelsize': 7, 'label.size': 7.5, 'leftlabel.weight': 'bold', 'toplabel.weight': 'bold'})
fig, axs = uplt.subplots(nrows=3, ncols=3, figwidth=7.0, refaspect=1.55, sharey=True, sharex=True, wspace=1.2, hspace=3.4)
lev = np.arange(-24, 25, 2); dlev = np.arange(-12, 13, 1); mains, diffs = [], []
LATD, ZD = np.arange(-32.5, 70, 2.5), np.arange(125, 5000, 250)
for ri, (res, _) in enumerate(ROWS):
    Lg, P = M[(res, 'LGM')], M[(res, 'PI')]
    assert Lg['z'].shape == P['z'].shape and np.allclose(Lg['z'], P['z']), res
    if Lg['lat'].shape == P['lat'].shape and np.allclose(Lg['lat'], P['lat']):
        annP = P['ann']
    else:
        annP = np.stack([[np.interp(Lg['lat'], P['lat'], P['ann'][t, k], left=np.nan, right=np.nan)
                          for k in range(P['ann'].shape[1])] for t in range(P['ann'].shape[0])])
    diff = Lg['psi'] - np.nanmean(annP, axis=0)
    sig, m, nsig, pcrit = fdr_mask(Lg['ann'], annP, Lg['lat'], Lg['z'])
    print('%s LGM-PI: %d grid points tested, %d significant (%.1f%%) at FDR q=%.2f, p threshold %.2e'
          % (res, m, nsig, 100.0 * nsig / m, Q, pcrit), flush=True)
    for ci, dat in enumerate([Lg, P]):
        ax = axs[ri * 3 + ci]
        mm = ax.contourf(dat['lat'], dat['z'], dat['psi'], levels=lev, cmap='RdBu_r', extend='both')
        ax.contour(dat['lat'], dat['z'], dat['psi'], levels=[0], colors='k', lw=0.5)
        if np.isfinite(dat['base']): ax.axhline(dat['base'], color='k', ls='--', lw=0.8)
        ax.format(title='%.1f Sv, base %.0f m' % (dat['s'], dat['base'])); mains.append(mm)
    ax = axs[ri * 3 + 2]
    md = ax.contourf(Lg['lat'], Lg['z'], diff, levels=dlev, cmap='RdBu_r', extend='both'); diffs.append(md)
    jy = np.array([int(np.argmin(np.abs(Lg['lat'] - y))) for y in LATD]); kz = np.array([int(np.argmin(np.abs(Lg['z'] - d))) for d in ZD])
    S = sig[np.ix_(kz, jy)]; yy, dd = np.meshgrid(LATD, ZD)
    finite = np.isfinite(diff[np.ix_(kz, jy)]) & (dd <= 5000)
    S = (~S & finite) if DOTS == 'not significant' else S
    ax.scatter(yy[S], dd[S], s=1.6, c='k', marker='o', lw=0, alpha=0.9)
    print('  %s: %d of %d lattice points dotted (%s)' % (res, S.sum(), finite.sum(), DOTS), flush=True)
    ax.format(title='strength ' + MINUS(Lg['s'] - P['s'], '%+.1f') + ' Sv (' + MINUS(100.0 * (Lg['s'] - P['s']) / P['s'], '%+.0f') + '%)')
axs.format(toplabels=['LGM', 'PI', 'LGM − PI'], leftlabels=[r for _, r in ROWS], ylim=(5000, 0), xlim=(-34, 70),
           xlocator=20, ylocator=1000, xlabel='latitude (°N)', ylabel='depth (m)', abc='(a)', abcloc='ul')
fig.colorbar(mains[0], loc='b', cols=(1, 2), label='overturning streamfunction ψ (Sv)', length=0.92, width=0.09, ticks=4)
fig.colorbar(diffs[0], loc='r', rows=(1, 3), label='ψ, LGM − PI (Sv)', length=0.7, width=0.09, ticks=2)
p = f"{OUT}/fig1_overturning.png"; fig.savefig(p, dpi=400, bbox_inches='tight'); print(f"wrote {p}")
