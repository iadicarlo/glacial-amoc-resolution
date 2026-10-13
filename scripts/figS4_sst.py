#!/usr/bin/env python
"""Figure S4: glacial sea-surface temperature of HR, LR and CESM1.2 against reconstructions (GLOMAP, PALMOD sites,
Tierney et al. 2020 sites). (a-c) annual-mean LGM minus PI SST on each native grid, cells with 15% or more annual-mean
sea ice in either state left out; (d) zonal means; (e) Southern Ocean 60-40S and (f) Atlantic north of 25N, area means.
Inputs: SST and sea-ice climatologies, glomap.nc and the two proxy tables. Output: figS4_sst.png.
"""
import os, sys, warnings, csv
import numpy as np, xarray as xr
import ultraplot as uplt
import cartopy.crs as ccrs
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from pop_native import grid, area, map_cells
warnings.filterwarnings('ignore')

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
B = DATA + '/casper_vvel'; LS = B + '/lr_suite'
GLO = DATA + '/glomap/glomap.nc'; PAL = DATA + '/palmod/lgm_nsst.csv'
TIE = DATA + '/tierney2020/Tierney2020_ProxyDataPaired.csv'
SST = {('ne120', 'LGM'): B + '/ne120_lgm_reduce/ne120_LGM_SST_monthly_clim_0001-0015.nc',
       ('ne120', 'PI'): B + '/PI_ne120/clim/ne120_PI_SST_monthly_clim_0041-0060.nc',
       ('f19', 'LGM'): LS + '/f19_LGM_SST_monthly_clim_0981-1000.nc', ('f19', 'PI'): LS + '/f19_PI_SST_monthly_clim_0485-0504.nc',
       ('c12', 'LGM'): LS + '/c12_LGM_SST_monthly_clim_0881-0900.nc', ('c12', 'PI'): LS + '/c12_PI_SST_monthly_clim_0881-0900.nc'}
ICE = {('ne120', 'LGM'): B + '/ne120_lgm_reduce/ne120_LGM_aice_hi_monthly_clim_0001-0015.nc',
       ('ne120', 'PI'): B + '/pi_control/ne120_PI_aice_monthly_clim_0041-0060.nc',
       ('f19', 'LGM'): LS + '/f19_LGM_aice_monthly_clim_0981-1000.nc', ('f19', 'PI'): LS + '/f19_PI_aice_monthly_clim_0485-0504.nc',
       ('c12', 'LGM'): LS + '/c12_LGM_aice_monthly_clim_0881-0900.nc', ('c12', 'PI'): LS + '/c12_PI_aice_monthly_clim_0881-0900.nc'}
COORD = {'f19': LS + '/f19_LGM_REGION_MASK.nc', 'c12': LS + '/c12_LGM_REGION_MASK.nc'}
CONF = [('ne120', 'HR (0.1°)', '#1b3f6e'), ('f19', 'LR (1°)', '#2e7d4f'), ('c12', 'CESM1.2 (1°)', '#7b3294')]
ICE_THR = 15.0                       # percent, annual mean, either state
LAND, ICEC = '0.62', '#ffffff'


def clean(a): a = np.asarray(a, 'f8'); return np.where(np.abs(a) < 1e10, a, np.nan)


def load(conf, st):
    ds = xr.open_dataset(SST[(conf, st)])
    tv = [k for k in ('TEMP_ann', 'TEMP') if k in ds]
    if tv and 'SST' not in ''.join(ds.data_vars).upper():
        v = clean(ds[tv[0]].values); v = v[0] if v.ndim == 3 else v[0, 0]           # top level of the annual mean
    else:
        cand = [k for k in ds.data_vars if 'SST' in k.upper()]; assert len(cand) == 1, (conf, st, list(ds.data_vars))
        v = clean(ds[cand[0]].values); assert v.shape[0] == 12, (conf, st); v = np.nanmean(v, axis=0)
    if 'TLONG' in ds or 'TLON' in ds:
        lon = (ds['TLONG'] if 'TLONG' in ds else ds['TLON']).values; lat = ds['TLAT'].values
    else:
        g = xr.open_dataset(COORD[conf]); lon, lat = g['TLONG'].values, g['TLAT'].values; g.close()
    ds.close()
    assert -3.0 < np.nanmin(v) < 5.0 and 25.0 < np.nanmax(v) < 40.0, '%s %s: SST range %.1f..%.1f is not degC' % (conf, st, np.nanmin(v), np.nanmax(v))
    return clean(lon), clean(lat), v


def load_ice(conf, st):
    ds = xr.open_dataset(ICE[(conf, st)]); a = clean(ds['aice_monthly'].values)
    slat = (ds['TLAT'] if 'TLAT' in ds else ds['tlat']).values; ds.close()
    assert a.shape[0] == 12 and 2.0 < np.nanmax(a) <= 100.5, '%s %s: aice is not 12 months of percent' % (conf, st)
    return clean(slat), np.nanmean(a, axis=0)


def natl(lon, lat):
    """Atlantic north of 25N: 25-80N, 100W-20E, without the Mediterranean, the Baltic and Hudson Bay (one rule for
    every dataset, applied to cell centres)."""
    x = ((np.asarray(lon) + 180.0) % 360.0) - 180.0; y = np.asarray(lat)
    m = (y >= 25.0) & (y <= 80.0) & (x >= -100.0) & (x <= 20.0)
    m &= ~((x > -6.0) & (y < 47.0))                        # Mediterranean (and Black Sea) east of Gibraltar
    m &= ~((x > 9.0) & (y > 53.0) & (y < 66.0))             # Baltic
    m &= ~((x > -96.0) & (x < -75.0) & (y > 50.0) & (y < 70.0))   # Hudson Bay and Foxe Basin
    return m


REG = {'so': ('Southern Ocean, 60 to 40°S', lambda lo, la: (la >= -60.0) & (la <= -40.0)),
       'na': ('Atlantic north of 25°N', natl)}

M = {}
for conf, _, _ in CONF:
    lo, la, L = load(conf, 'LGM'); lo2, la2, P = load(conf, 'PI')
    assert L.shape == P.shape, conf
    ilaL, iL = load_ice(conf, 'LGM'); ilaP, iP = load_ice(conf, 'PI')
    assert iL.shape == L.shape and iP.shape == L.shape, conf
    xc, yc, xv, yv = grid(la, lo, want_corners=True); grid(la2, lo2); grid(ilaL, xc); grid(ilaP, xc)
    A = area(la)
    ocean = np.isfinite(L) & np.isfinite(P)
    icy = ocean & ((np.nan_to_num(iL) >= ICE_THR) | (np.nan_to_num(iP) >= ICE_THR))
    an = np.where(ocean & ~icy, L - P, np.nan)
    M[conf] = dict(anom=an, icy=icy, ocean=ocean, xc=xc, yc=yc, xv=xv, yv=yv, A=A)
    ok = np.isfinite(an)
    print('%-5s ice-free ocean mean LGM-PI SST %.2f K; ice-covered share of the ocean %.1f%%'
          % (conf, np.nansum(A[ok] * an[ok]) / np.sum(A[ok]), 100.0 * np.sum(A[icy]) / np.sum(A[ocean])), flush=True)

# ---------------- GLOMAP and PALMOD
g = xr.open_dataset(GLO, decode_times=False)
glat, glon = g['latitude'].values.astype('f8'), g['longitude'].values.astype('f8')
GA = np.nanmean(clean(g['nsst_anomaly'].values), axis=0); GU = np.nanmean(clean(g['nsst_anomaly_uncertainty'].values), axis=0)
GF = np.nanmean(clean(g['fice_lgm_extrap'].values), axis=0); GM = np.isfinite(np.nanmean(clean(g['fice_lgm'].values), axis=0))
g.close()
assert np.nanmax(GF) <= 1.0 + 1e-6, 'fice_lgm_extrap is not a fraction'
o = np.argsort(glon % 360.0); glon = (glon % 360.0)[o]; GA, GU, GF, GM = GA[:, o], GU[:, o], GF[:, o], GM[:, o]
if glat[0] > glat[-1]: glat = glat[::-1]; GA, GU, GF, GM = GA[::-1], GU[::-1], GF[::-1], GM[::-1]
G2LA, G2LO = np.meshgrid(glat, glon, indexing='ij')
GW = np.cos(np.radians(G2LA))
GICE = GM & (np.nan_to_num(GF) >= ICE_THR / 100.0)
gocean = GM & np.isfinite(GA)
GA = np.where(gocean & ~GICE, GA, np.nan)
print('GLOMAP: ice-covered share of its glacial ocean %.1f%%' % (100.0 * np.sum(GW[GICE & gocean]) / np.sum(GW[gocean])), flush=True)
pal = []
with open(PAL) as fh:
    for r in csv.DictReader(fh):
        try: a = float(r['sst_anom'])
        except ValueError: continue
        if np.isfinite(a): pal.append((float(r['lat']), float(r['lon']), a))
pal = np.array(pal); print('PALMOD sites with an LGM-LH anomaly: %d' % len(pal), flush=True)
rows_t = {}
with open(TIE) as fh:
    for r in csv.DictReader(fh):
        a = float(r['Median'])
        if np.isfinite(a):
            rows_t.setdefault((round(float(r['Latitude']), 1), round(float(r['Longitude']), 1)), []).append(a)
tie = np.array([(la_, lo_, np.mean(v_)) for (la_, lo_), v_ in rows_t.items()])
nrow_t = sum(len(v_) for v_ in rows_t.values())
print('Tierney et al. (2020) paired anomalies: %d rows at %d sites, median site anomaly %+.2f K' % (nrow_t, len(tie), np.median(tie[:, 2])), flush=True)
assert -6.0 < np.median(tie[:, 2]) < -1.0, 'Tierney median anomaly is not a glacial cooling in K'
assert np.all(np.abs(tie[:, 0]) <= 90) and np.all(np.abs(tie[:, 1]) <= 360), 'Tierney coordinates out of range'

# ---------------- (d) zonal means, (e, f) regional means, all over ice-free cells on own grids
edges = np.arange(-80, 91, 2.0); cen = 0.5 * (edges[1:] + edges[:-1])


def zonal(lat, v, w):
    ok = np.isfinite(v) & np.isfinite(lat); b = np.digitize(lat[ok], edges) - 1; m = (b >= 0) & (b < cen.size)
    num = np.bincount(b[m], weights=(w[ok] * v[ok])[m], minlength=cen.size); den = np.bincount(b[m], weights=w[ok][m], minlength=cen.size)
    return np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)


Z = {c: zonal(M[c]['yc'], M[c]['anom'], M[c]['A']) for c, _, _ in CONF}
ZG, ZGU = zonal(G2LA, GA, GW), zonal(G2LA, np.where(np.isfinite(GA), GU, np.nan), GW)
RM = {}
for rk, (rn, rf) in REG.items():
    print('\n%s, LGM - PI SST over ice-free cells (K):' % rn)
    for c, _, _ in CONF:
        m = M[c]; inr = rf(m['xc'], m['yc']) & m['ocean']; sel = inr & np.isfinite(m['anom'])
        RM[(rk, c)] = np.sum(m['A'][sel] * m['anom'][sel]) / np.sum(m['A'][sel])
        print('  %-5s %+.2f  (ice-free %.0f%% of the region\'s ocean)' % (c, RM[(rk, c)], 100.0 * np.sum(m['A'][sel]) / np.sum(m['A'][inr])))
    inr = rf(G2LO, G2LA) & gocean; sel = inr & np.isfinite(GA)
    RM[(rk, 'glomap')] = np.sum(GW[sel] * GA[sel]) / np.sum(GW[sel])
    uu = np.where(np.isfinite(GU[sel]), GU[sel], np.nanmedian(GU[sel]))
    RM[(rk, 'glomap_u')] = np.sum(GW[sel] * uu) / np.sum(GW[sel])
    RM[(rk, 'glomap_ui')] = np.sqrt(np.sum((GW[sel] * uu) ** 2)) / np.sum(GW[sel])
    ps = pal[rf(pal[:, 1], pal[:, 0])]; RM[(rk, 'pal')] = ps
    ts = tie[rf(tie[:, 1], tie[:, 0])]; RM[(rk, 'tie')] = ts
    print('  Tierney 2020: %d sites, mean %+.2f, median %+.2f' % (len(ts), ts[:, 2].mean() if len(ts) else np.nan,
                                                               np.median(ts[:, 2]) if len(ts) else np.nan), flush=True)
    print('  GLOMAP %+.2f (vs modern WOA98, ice-free %.0f%%), uncertainty %.2f (fully correlated) / %.2f (independent); '
          'PALMOD %d sites, mean %+.2f, median %+.2f' % (RM[(rk, 'glomap')], 100.0 * np.sum(GW[sel]) / np.sum(GW[inr]),
                                                         RM[(rk, 'glomap_u')], RM[(rk, 'glomap_ui')], len(ps),
                                                         ps[:, 2].mean() if len(ps) else np.nan, np.median(ps[:, 2]) if len(ps) else np.nan), flush=True)

# ---------------- figure
uplt.rc.update({'font.size': 7, 'title.size': 7.5, 'abc.size': 8, 'label.size': 7, 'tick.labelsize': 6.5, 'legend.fontsize': 6.8})
LON0 = -30.0
WS = [0.2] * 23; WS[9] = WS[16] = 4.4                    # room for the y labels of (e) and (f)
fig, axs = uplt.subplots([[1] * 8 + [2] * 8 + [3] * 8, [4] * 10 + [5] * 7 + [6] * 7], proj=['robin', 'robin', 'robin', None, None, None],
                         proj_kw={'lon0': LON0}, figwidth=7.0, hratios=(1, 1.3), wspace=WS, hspace=3.4, share=False)
lev = np.arange(-8, 2.1, 1.0)
cols = list(plt.get_cmap('Blues_r')(np.linspace(0.0, 0.8, 8))) + list(plt.get_cmap('Reds')(np.linspace(0.18, 0.55, 2)))
cmap = ListedColormap(cols); cmap.set_under(plt.get_cmap('Blues_r')(0.0)); cmap.set_over(plt.get_cmap('Reds')(0.7))
norm = BoundaryNorm(lev, cmap.N, extend='neither')
for i, (c, name, col) in enumerate(CONF[:3]):
    ax = axs[i]; m = M[c]
    for sel, fc in ((m['ocean'] & ~m['icy'], None), (m['icy'], ICEC)):
        q, iq, ex, ie = map_cells(m['xv'], m['yv'], sel, LON0, ax.projection)
        v = m['anom'][sel]
        face = (lambda k: cmap(norm(v[k]))) if fc is None else (lambda k: fc)
        ax.add_collection(PolyCollection(q, facecolors=face(iq), edgecolors='none', antialiased=False,
                                         transform=ax.transData, rasterized=True, zorder=1))
        if ex:
            ax.add_collection(PolyCollection(ex, facecolors=face(ie), edgecolors='none', antialiased=False,
                                             transform=ax.transData, rasterized=True, zorder=1))
    ax.set_facecolor(LAND)
    ax.scatter(pal[:, 1], pal[:, 0], c=pal[:, 2], cmap=cmap, norm=norm, s=7, lw=0.35, edgecolors='k', transform=ccrs.PlateCarree(), zorder=3)
    ax.scatter(tie[:, 1], tie[:, 0], c=tie[:, 2], cmap=cmap, norm=norm, s=3.5, marker='D', lw=0.25, edgecolors='k', transform=ccrs.PlateCarree(), zorder=2)
    ax.format(title=name, titleweight='bold', grid=False)
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs[:3], loc='r', label='LGM − PI SST (K)', ticks=2, length=0.9, width=0.09, extend='both')
ax = axs[3]
ax.fill_between(cen, ZG - ZGU, ZG + ZGU, color='0.85', lw=0, zorder=0)
ax.plot(cen, ZG, color='0.3', lw=1.0, ls='--', zorder=1)
ax.scatter(pal[:, 0], pal[:, 2], s=3, color='0.6', lw=0, zorder=1)
ax.scatter(tie[:, 0], tie[:, 2], s=4, marker='D', facecolor='none', edgecolor='0.35', lw=0.4, zorder=1)
for c, name, col in CONF:
    ax.plot(cen, Z[c], color=col, lw=1.2 if c == 'ne120' else 0.9, zorder=2)
ax.axhline(0, color='k', lw=0.5)
ax.format(xlim=(-78, 88), ylim=(-12, 3), xlabel='latitude (°N)', ylabel='LGM − PI SST (K)', title='zonal mean', xlocator=30)
XL = ['GLOMAP', 'Tierney 2020', 'HR', 'LR', 'CESM1.2']
for ax, rk in ((axs[4], 'so'), (axs[5], 'na')):
    ax.bar(0, RM[(rk, 'glomap')], width=0.62, color='0.8', edgecolor='k', lw=0.4)
    ax.errorbar(0, RM[(rk, 'glomap')], yerr=RM[(rk, 'glomap_u')], fmt='none', ecolor='k', elinewidth=0.8, capsize=2)
    ts = RM[(rk, 'tie')]
    if len(ts):
        ax.bar(1, ts[:, 2].mean(), width=0.62, color='white', edgecolor='k', lw=0.4, hatch='//////')
        ax.scatter(1 + np.linspace(-0.24, 0.24, len(ts)), ts[:, 2], s=5, marker='D', facecolor='white', edgecolor='0.2', lw=0.5, zorder=4)
    for i, (c, name, col) in enumerate(CONF, 2):
        ax.bar(i, RM[(rk, c)], width=0.62, color=col, edgecolor='k', lw=0.4)
    ps = RM[(rk, 'pal')]
    if len(ps): ax.scatter(np.linspace(-0.24, 0.24, len(ps)), ps[:, 2], s=5, facecolor='white', edgecolor='0.2', lw=0.5, zorder=4)
    if rk == 'so': ax.axhline(-3.0, color='#8b1a1a', lw=0.8, ls=':')
    lo_ = min([RM[(rk, c)] for c, _, _ in CONF] + [RM[(rk, 'glomap')] - RM[(rk, 'glomap_u')]] + ([ps[:, 2].min()] if len(ps) else [])
              + ([ts[:, 2].min()] if len(ts) else []))
    hi_ = max([0.0] + ([ps[:, 2].max()] if len(ps) else []) + ([ts[:, 2].max()] if len(ts) else []))
    ax.format(xticks=np.arange(5), xticklabels=XL, xrotation=45, ylabel='LGM − PI SST (K)', xlim=(-0.6, 4.6),
              ylim=(np.floor(lo_ - 0.5), hi_ + 0.6), xtickminor=False, xgrid=False, title=REG[rk][0], titleloc='l')
    for t_ in ax.get_xticklabels():
        t_.set_rotation(45); t_.set_ha('right'); t_.set_rotation_mode('anchor')
axs.format(abc='(a)', abcloc='l')
hd = [Line2D([0], [0], color='0.3', lw=1.0, ls='--'), Patch(fc='0.85', ec='none'),
      Line2D([0], [0], marker='o', ls='none', ms=3, mfc='0.6', mec='0.6'),
      Line2D([0], [0], marker='D', ls='none', ms=2.6, mfc='none', mec='0.35', mew=0.5)] + \
     [Line2D([0], [0], color=col, lw=1.2) for _, _, col in CONF]
fig.legend(hd, ['GLOMAP (vs modern)', 'GLOMAP uncertainty', 'PALMOD sites', 'Tierney 2020 sites'] + [n for _, n, _ in CONF], loc='b', ncols=4, frame=False)
p = OUT + '/figS4_sst.png'; fig.savefig(p, dpi=400, bbox_inches='tight'); print('wrote', p)
