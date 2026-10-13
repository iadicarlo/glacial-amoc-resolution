#!/usr/bin/env python
"""Figure 4 (c-f), first step, for LR (f19_g16) and HR (0.1 degree): the North Atlantic in March seen face on
(orthographic, native cells, classes of mixed-layer depth and sea ice, PI 1000 m mixed-layer contour, LGM ice edge),
saved as map underlays, and the PI Atlantic overturning of both models. Prints the centroids used to place the arrows.
Output: _fig4_natl_<conf>.png, _fig4_natl.json and fig4_psi_pi.npz, read by fig4_schematic.py.
"""
import json, os, sys, warnings
import numpy as np, xarray as xr, netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.collections import PolyCollection
from contourpy import contour_generator
from pop_native import grid, area, cyclic, map_cells
from f19_common import load_annual
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
B = DATA + '/casper_vvel'; LS = B + '/lr_suite'; ATL = 1
MLD = {('ne120', 'LGM'): B + '/MESACLIP_mld/LGM_HMXL_monthly_clim_0001-0011.nc',
       ('ne120', 'PI'): B + '/MESACLIP_mld/PI_ne120_HMXL_monthly_clim_0041-0060.nc',
       ('f19', 'LGM'): LS + '/f19_LGM_HMXL_monthly_clim_0981-1000.nc',
       ('f19', 'PI'): LS + '/f19_PI_HMXL_monthly_clim_0485-0504.nc'}
GRIDF = {('f19', 'LGM'): LS + '/f19_LGM_REGION_MASK.nc', ('f19', 'PI'): LS + '/f19_PI_REGION_MASK.nc'}
ICE = {'ne120': B + '/ne120_lgm_reduce/ne120_LGM_aice_hi_monthly_clim_0001-0015.nc',
       'f19': LS + '/f19_LGM_aice_monthly_clim_0981-1000.nc'}
MOCZ_PI = DATA + '/casper_vvel/PI_ne120/mocz_1-20_all.nc'
GATE = {'ne120': (18.55, 2794), 'f19': (22.33, 3210)}
LON0, LAT0, R = -38.0, 48.0, 6371000.0
XL, YL = (-0.72, 0.72), (-0.62, 0.78)                    # extent in units of the Earth radius
PROJ = ccrs.Orthographic(LON0, LAT0, globe=ccrs.Globe(semimajor_axis=R, semiminor_axis=R, ellipse=None))
C = dict(land='#e6dcc6', ocean='#d5e7f3', ice='#ffffff', mld5='#9e9ac8', mld10='#54278f', pi='#e8710a', edge='#000000')
MI = 2                                                     # March


def clean(a): a = np.asarray(a, 'f8'); return np.where(np.abs(a) < 1e10, a, np.nan)


def load_mld(key):
    ds = xr.open_dataset(MLD[key]); m = clean(ds['HMXL_monthly'].values[MI])
    if 'TLONG' in ds: lon, lat = ds['TLONG'].values, ds['TLAT'].values
    else:
        g = xr.open_dataset(GRIDF[key]); lon, lat = g['TLONG'].values, g['TLAT'].values; g.close()
    ds.close(); assert 300.0 < np.nanmax(m) < 7000.0, '%s: MLD is not metres' % (key,)
    return clean(lon), clean(lat), m


def strength_core(lat, z, psi):
    zm = np.where((z >= 500) & (z <= 3000))[0]; li = np.where((lat >= 20) & (lat <= 65))[0]
    return np.nanmax(psi[np.ix_(zm, li)])


def cell_base(lat, z, psi):
    li = np.where((lat >= 30) & (lat <= 45))[0]; p = np.nanmean(psi[:, li], axis=1); k0 = int(np.nanargmax(p))
    for k in range(k0, len(z) - 1):
        if p[k] >= 0 and p[k + 1] < 0:
            f = p[k] / (p[k] - p[k + 1]); return z[k] + f * (z[k + 1] - z[k])
    return np.nan


def ortho(lon, lat):
    lo, la = np.radians(np.asarray(lon) - LON0), np.radians(lat); p0 = np.radians(LAT0)
    return np.cos(la) * np.sin(lo), np.cos(p0) * np.sin(la) - np.sin(p0) * np.cos(la) * np.cos(lo)


meta = dict(lon0=LON0, lat0=LAT0, R=R, xlim=XL, ylim=YL, colors=C)
psi_out = {}
for k in ('f19', 'ne120'):
    lon, lat, mL = load_mld((k, 'LGM')); lonP, latP, mP = load_mld((k, 'PI'))
    ds = xr.open_dataset(ICE[k]); ice = clean(ds['aice_monthly'].values[MI])
    ilat = (ds['TLAT'] if 'TLAT' in ds else ds['tlat']).values; ilon = (ds['TLON'] if 'TLON' in ds else ds['tlon']).values; ds.close()
    assert ice.shape == mL.shape == mP.shape and np.nanmax(ice) > 2.0, k
    grid(latP, lonP); grid(ilat, ilon)
    xc, yc, xv, yv = grid(lat, lon, want_corners=True); A = area(lat)
    ocean = np.isfinite(mL)
    mP = np.where(ocean, mP, np.nan); ice = np.where(ocean, ice, np.nan)
    cls = np.full(mL.shape, -1, int)
    cls[ocean] = 0
    cls[ocean & (np.nan_to_num(ice) >= 15.0)] = 1
    cls[ocean & (mL >= 500.0)] = 2
    cls[ocean & (mL >= 1000.0)] = 3
    # only cells well inside the visible hemisphere (orthographic: invisible corners project to inf, map_cells drops them)
    cosc = (np.sin(np.radians(LAT0)) * np.sin(np.radians(yc)) +
            np.cos(np.radians(LAT0)) * np.cos(np.radians(yc)) * np.cos(np.radians(xc - LON0)))
    xo, yo = ortho(xc, yc)
    vis = (cosc > 0.05) & (xo > XL[0] - 0.03) & (xo < XL[1] + 0.03) & (yo > YL[0] - 0.03) & (yo < YL[1] + 0.03)
    fig = plt.figure(figsize=(2.4, 2.4 * (YL[1] - YL[0]) / (XL[1] - XL[0])), facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=PROJ)
    ax.set_xlim(XL[0] * R, XL[1] * R); ax.set_ylim(YL[0] * R, YL[1] * R)
    ax.set_facecolor(C['land']); ax.spines['geo'].set_visible(False)
    try: ax.background_patch.set_visible(False)
    except AttributeError: pass
    col = {0: C['ocean'], 1: C['ice'], 2: C['mld5'], 3: C['mld10']}
    for cv in (0, 1, 2, 3):
        sel = vis & (cls == cv)
        if not sel.any(): continue
        q, iq, ex, ie = map_cells(xv, yv, sel, LON0, PROJ)
        ax.add_collection(PolyCollection(q, facecolors=col[cv], edgecolors=col[cv], linewidths=0.05, antialiased=True,
                                         transform=ax.transData, rasterized=True, zorder=1 + cv))
        if ex: ax.add_collection(PolyCollection(ex, facecolors=col[cv], edgecolors=col[cv], linewidths=0.05,
                                                transform=ax.transData, rasterized=True, zorder=1 + cv))
        print('  %s class %d: %d cells' % (k, cv, len(q) + len(ex)), flush=True)
    cl, ca, cI, cP = cyclic(xc, yc, ice, mP)
    for f, lev, c, lw, ls in ((cI, 15.0, C['edge'], 0.6, '-'), (cP, 1000.0, C['pi'], 1.0, (0, (3, 1.6)))):
        for s in contour_generator(cl, ca, np.ma.masked_invalid(f)).lines(lev):
            s = s[np.all(np.isfinite(s), axis=1)]
            if len(s) < 2: continue
            x, y = ortho(s[:, 0], s[:, 1]); ok = (np.sin(np.radians(LAT0)) * np.sin(np.radians(s[:, 1])) +
                                                   np.cos(np.radians(LAT0)) * np.cos(np.radians(s[:, 1])) * np.cos(np.radians(s[:, 0] - LON0))) > 0
            x = np.where(ok, x, np.nan) * R; y = np.where(ok, y, np.nan) * R
            ax.plot(x, y, color=c, lw=lw, ls=ls, zorder=8, solid_capstyle='round', transform=ax.transData)
    p = '%s/_fig4_natl_%s.png' % (OUT, k); fig.savefig(p, dpi=500, facecolor='none'); plt.close(fig)   # outside the globe transparent, land keeps its colour
    print('wrote', p, flush=True)
    # centroids for the arrows (Atlantic sector 40-80N, 80W-20E)
    box = (yc >= 40) & (yc <= 80) & (((xc + 180) % 360 - 180) >= -80) & (((xc + 180) % 360 - 180) <= 20)
    for st, m in (('LGM', mL), ('PI', mP)):
        for thr in (500.0, 1000.0):
            s = box & (m >= thr)
            if s.any():
                w = A[s]; xs = ((xc[s] + 180) % 360) - 180
                print('  %s %s MLD >= %4.0f m: area %.3f e12 m2, centroid %.1fE %.1fN' % (k, st, thr, w.sum() * 1e-12,
                      np.sum(w * xs) / w.sum(), np.sum(w * yc[s]) / w.sum()), flush=True)
            else: print('  %s %s MLD >= %4.0f m: none' % (k, st, thr), flush=True)
    for lo_ in (-50.0, -30.0, -15.0):                       # southernmost LGM ice (>=15%) along three meridians
        s = box & (np.abs(((xc + 180) % 360 - 180) - lo_) < 1.0) & (np.nan_to_num(ice) >= 15.0)
        print('  %s LGM March ice reaches %.1fN at %.0fE' % (k, yc[s].min() if s.any() else np.nan, lo_), flush=True)
    # pre-industrial psi, gated
    if k == 'ne120':
        d = nc.Dataset(MOCZ_PI); moc = np.array(d.variables['MOC'][ATL]).astype('f8'); z = np.array(d.variables['z_t'][:]) / 100.0
        tl = np.array(d.variables['TLAT_1D'][:]); d.close()
        if tl.ndim == 3: tl = np.nanmean(tl, axis=0)
        plat = np.asarray(tl[ATL]).ravel()[:moc.shape[-1]]; psi = np.nanmean(moc, axis=0)
        ok = np.isfinite(plat); plat, psi = plat[ok], psi[:, ok]; o = np.argsort(plat); plat, psi = plat[o], psi[:, o]
    else:
        plat, z, ann, _ = load_annual('F19PI', 485, 504); psi = np.nanmean(ann, axis=0)
    s_, b_ = strength_core(plat, z, psi), cell_base(plat, z, psi)
    if abs(s_ - GATE[k][0]) > 0.006 or abs(b_ - GATE[k][1]) > 1.0:
        sys.exit('STOP: %s PI psi %.3f Sv / %.1f m, expected %.2f / %d' % (k, s_, b_, GATE[k][0], GATE[k][1]))
    print('  %s PI psi %.2f Sv, cell base %.0f m [gate passed]' % (k, s_, b_), flush=True)
    psi_out['psi_' + k], psi_out['psilat_' + k], psi_out['psiz_' + k] = psi.astype('f4'), plat, z
np.savez_compressed(OUT + '/fig4_psi_pi.npz', **psi_out)
with open(OUT + '/_fig4_natl.json', 'w') as fh: json.dump(meta, fh)
print('DONE', flush=True)
