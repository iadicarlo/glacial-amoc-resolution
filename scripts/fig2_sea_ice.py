#!/usr/bin/env python
"""Figure 2: glacial winter sea ice, NH March (North Atlantic) and SH September, for the GLOMAP reconstruction (Paul
et al. 2021) and HR, LR and CESM1.2. Shading: LGM concentration on each dataset's native grid (no regridding); lines:
the LGM 15% edge, the same model's PI 15% edge and the GLOMAP 15% edge.
Input: monthly sea-ice climatologies of each window and glomap.nc. Output: fig2_sea_ice.png.
"""
import os, sys, warnings
import numpy as np, xarray as xr
import ultraplot as uplt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib import colors
from pop_native import cyclic, rows, grid, map_cells
warnings.filterwarnings('ignore')

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
V = DATA + '/casper_vvel'; LS = V + '/lr_suite'
GLO = DATA + '/glomap/glomap.nc'
SRC = {('ne120', 'LGM'): V + '/ne120_lgm_reduce/ne120_LGM_aice_hi_monthly_clim_0001-0015.nc',
       ('ne120', 'PI'): V + '/pi_control/ne120_PI_aice_monthly_clim_0041-0060.nc',
       ('f19', 'LGM'): LS + '/f19_LGM_aice_monthly_clim_0981-1000.nc',
       ('f19', 'PI'): LS + '/f19_PI_aice_monthly_clim_0485-0504.nc',
       ('c12', 'LGM'): LS + '/c12_LGM_aice_monthly_clim_0881-0900.nc',
       ('c12', 'PI'): LS + '/c12_PI_aice_monthly_clim_0881-0900.nc'}
# expected native-grid extents, 1e6 km2
GATE = {('ne120', 'LGM', 2): 23.17, ('ne120', 'PI', 2): 15.92,
        ('ne120', 'LGM', 8): 39.96, ('ne120', 'PI', 8): 20.04,
        ('f19', 'LGM', 2): 14.85, ('f19', 'PI', 2): 16.34, ('f19', 'LGM', 8): 45.08, ('f19', 'PI', 8): 20.05}
GATE_PCT = {('c12', 2): -11.3, ('c12', 8): 110.7}   # expected native-grid LGM - PI, percent of PI
COLS = [('glomap', 'GLOMAP reconstruction'), ('ne120', 'HR (0.1°)'), ('f19', 'LR (1°)'), ('c12', 'CESM1.2 (1°)')]
HEMI = [('NH March', 2, 'npstere', 35), ('SH September', 8, 'spstere', -40)]
NA_LON, NA_LAT = (-80, 25), (38, 82)     # North Atlantic sector of the NH row
C_GLO, C_PI = '#1f7a3d', '#d9480f'
OCEAN, LANDC = '#3d7fb8', '0.80'   # open water, land


G, bad, EXT = {}, [], {}
for key, p in SRC.items():
    ds = xr.open_dataset(p)
    a = ds['aice_monthly'].values.astype('f8'); a = np.where(np.abs(a) < 1e10, a, np.nan)
    assert a.shape[0] == 12 and np.nanmax(a) > 2.0, '%s: not 12 months of percent' % (key,)
    slon = (ds['TLON'] if 'TLON' in ds else ds['tlon']).values
    slat = (ds['TLAT'] if 'TLAT' in ds else ds['tlat']).values
    ta = ds['tarea'].values.astype('f8'); ta = ta * 1e-4 if np.nanmax(ta) > 1e13 else ta
    assert 1e14 < np.nansum(ta) < 1e15, '%s: tarea is not m2' % (key,)
    ds.close()
    for _, mi, _, _ in HEMI:
        hemi = (slat > 0) if mi == 2 else (slat < 0)
        ext = float(np.nansum(ta[hemi & np.isfinite(a[mi]) & (a[mi] >= 15.0)])) * 1e-12
        EXT[key + (mi,)] = ext
        if key + (mi,) in GATE:
            if abs(ext - GATE[key + (mi,)]) > 0.01: bad.append((key, mi, ext, GATE[key + (mi,)]))
            print('%-6s %-3s month %d: native extent %.2f (gate %.2f)' % (key + (mi, ext, GATE[key + (mi,)])), flush=True)
        else:
            print('%-6s %-3s month %d: native extent %.2f' % (key + (mi, ext)), flush=True)
    G[key] = (slon, slat, a)
for (k, mi), want in GATE_PCT.items():
    pct = 100.0 * (EXT[(k, 'LGM', mi)] - EXT[(k, 'PI', mi)]) / EXT[(k, 'PI', mi)]
    print('%-6s month %d: native change %+.2f%% (gate %+.1f%%)' % (k, mi, pct, want), flush=True)
    if abs(pct - want) > 0.05: bad.append((k, mi, pct, want))
if bad: sys.exit('STOP: extents do not reproduce the gate: %s' % bad)
print('[gate] native-grid extents reproduce the expected values: PASSED', flush=True)
NAT = {}
for k in ('ne120', 'f19', 'c12'):
    lo, la, aL = G[(k, 'LGM')]; lo2, la2, aP = G[(k, 'PI')]
    assert aL.shape == aP.shape, k
    grid(la2, lo2); lo, la, xv, yv = grid(la, lo, want_corners=True)   # full centres and corners, checked on BOTH files
    aP = np.where(np.isfinite(aL), aP, np.nan)            # PI edge only over the LGM ocean of the same grid
    lon, lat, fL, fP = cyclic(lo, la, aL, aP)
    for hn, mi, _, _ in HEMI:
        band = dict(south=25.0) if mi == 2 else dict(north=-30.0)
        sel = (la >= 25.0) if mi == 2 else (la <= -30.0)
        NAT[(k, mi)] = rows(lat, lon, fL[mi], fP[mi], **band) + ((xv, yv, aL[mi], sel),)
g = xr.open_dataset(GLO, decode_times=False)
gf = g['fice_lgm_extrap'].values.astype('f8'); gm = g['fice_lgm'].values.astype('f8')
glat, glon = g['latitude'].values, g['longitude'].values; g.close()
assert np.nanmin(gf) >= 0.0 and np.nanmax(gf) <= 1.0, 'fice_lgm_extrap is not 0-1'
# GLOMAP's own glacial ocean: where its non-extrapolated field is defined (fice_lgm, a 1/2 mask);
# the _extrap field is filled over land as well, so it is only used inside that ocean
o = np.argsort(glon % 360.0); glon = (glon % 360.0)[o]; gf = gf[..., o]; gm = gm[..., o]
GLON = np.r_[glon, glon[0] + 360.0]; GLAT = glat
dlo, dla = np.median(np.diff(glon)), np.median(np.abs(np.diff(glat)))
G2LO, G2LA = np.meshgrid(glon, glat)
GXV = np.stack([G2LO - dlo / 2, G2LO + dlo / 2, G2LO + dlo / 2, G2LO - dlo / 2], -1)
GYV = np.stack([G2LA - dla / 2, G2LA - dla / 2, G2LA + dla / 2, G2LA + dla / 2], -1)
for _, mi, _, _ in HEMI:
    v = np.where(np.isfinite(gm[mi]), 100.0 * gf[mi], np.nan)
    sel = (G2LA >= 25.0) if mi == 2 else (G2LA <= -30.0)
    NAT[('glomap', mi)] = (np.concatenate([v, v[:, :1]], axis=1), (GXV, GYV, v, sel))


def draw(ax, lon, lat, fL, cells, fP=None, glomap=None):
    """Land = gray background; every ocean cell drawn as its own polygon from its true corners (open water light,
    ice >= 15% on the blues); edges contoured on the native centres."""
    kw = dict(transform=ccrs.PlateCarree())
    xv, yv, val, sel = cells
    sel = sel & np.isfinite(val)
    q, iq, ex, ie = map_cells(xv, yv, sel, -30.0, ax.projection)       # corners pre-projected, split at the map's cut
    v = val[sel]

    def fcol(x):
        fc = np.tile(np.array(colors.to_rgba(OCEAN)), (x.size, 1)); ice = x >= 15.0
        fc[ice] = cmap(NORM(x[ice])); return fc
    ax.add_collection(PolyCollection(q, facecolors=fcol(v[iq]), edgecolors='none', antialiased=False,
                                     transform=ax.transData, rasterized=True, zorder=1))
    if ex:
        ax.add_collection(PolyCollection(ex, facecolors=fcol(v[ie]), edgecolors='none', antialiased=False,
                                         transform=ax.transData, rasterized=True, zorder=1))
    if fP is not None:
        ax.contour(lon, lat, fP, levels=[15.0], colors=C_PI, linewidths=0.9, linestyles='--', zorder=3, **kw)
    if glomap is not None:
        ax.contour(GLON, GLAT, glomap, levels=[15.0], colors=C_GLO, linewidths=1.3, zorder=4, **kw)
    ax.contour(lon, lat, fL, levels=[15.0], colors='k', linewidths=0.8, zorder=5, **kw)
    return int(len(q) + len(ex))


uplt.rc.update({'font.size': 7.5, 'title.size': 7.5, 'leftlabel.size': 8, 'toplabel.size': 8, 'abc.size': 8,
                'leftlabel.weight': 'bold', 'toplabel.weight': 'bold'})
cmap = uplt.Colormap('Blues_r', left=0.45, right=1.0)   # light blue at 15% to white at 100%
lev = [15, 30, 45, 60, 75, 90, 100]
NORM = BoundaryNorm(lev, cmap.N)
NA_PROJ = ccrs.LambertConformal(central_longitude=-30, central_latitude=55, standard_parallels=(45, 70))
fig, axs = uplt.subplots(nrows=2, ncols=4, proj=[NA_PROJ] * 4 + [ccrs.SouthPolarStereo(central_longitude=-30)] * 4, figwidth=7.0,
                         wspace=0.8, hspace=1.2, hratios=(1, 1.45))
pm = None
for ri, (hn, mi, _, blat) in enumerate(HEMI):
    for ci, (k, _) in enumerate(COLS):
        ax = axs[ri * 4 + ci]
        if mi == 2:
            ax.format(lonlim=NA_LON, latlim=NA_LAT, lonlines=30, latlines=15, labels=False,
                      gridlinewidth=0.4, gridcolor='0.55', gridalpha=0.5)
        else:
            ax.format(boundinglat=blat, lonlines=60, latlines=15, latmax=80, labels=False,
                      gridlinewidth=0.4, gridcolor='0.55', gridalpha=0.5)
        ax.set_facecolor(LANDC)
        if k == 'glomap':
            gv, gcells = NAT[('glomap', mi)]
            n = draw(ax, GLON, GLAT, gv, gcells)
        else:
            lat, lon, fL, fP, cells = NAT[(k, mi)]
            n = draw(ax, lon, lat, fL, cells, fP, glomap=NAT[('glomap', mi)][0])
        print('  %s %s: %d cells drawn' % (k, hn, n), flush=True)
axs.format(toplabels=[c for _, c in COLS], leftlabels=[h for h, _, _, _ in HEMI], abc='(a)', abcloc='ul')
fig.colorbar(ScalarMappable(norm=NORM, cmap=cmap), loc='b', cols=(1, 2), label='LGM sea-ice concentration (%)',
             length=0.9, width=0.09, ticks=lev)
h = [Line2D([0], [0], color='k', lw=0.8), Line2D([0], [0], color=C_PI, lw=0.9, ls='--'),
     Line2D([0], [0], color=C_GLO, lw=1.3)]
fig.legend(h, ['LGM 15% edge', 'pre-industrial 15% edge, same model', 'GLOMAP LGM 15% edge'],
           loc='b', cols=(3, 4), ncols=1, frame=False)
p = OUT + '/fig2_sea_ice.png'; fig.savefig(p, dpi=400, bbox_inches='tight'); print('wrote', p)
