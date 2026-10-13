#!/usr/bin/env python
"""Figure S3: overturning geometry of HR, LR and CESM1.2. (a-c) psi at 30S against depth, LGM and PI, with the psi = 0
interface between the northern cell and the abyssal cell; (d) the maximum of psi between 500 and 3000 m at every
latitude from 20 to 65N; (e) its glacial change with 5-95% moving-block bootstrap bands (2-year blocks, 400 draws).
Input: the overturning files of each run. Output: figS3_overturning.png.
"""
import os, sys, numpy as np, netCDF4 as nc, ultraplot as uplt
import f19_common as F19

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
ATL = 1; NB = 400; rng = np.random.default_rng(20260811)
OURS = {('0.1', 'LGM'): DATA + '/cesm/icesm13/archive/b.e13.Bi1850C5.ne120_t12.lgm.test.001/ocn/proc/mocz/mocz_1-15_all.nc',
        ('0.1', 'PI'): DATA + '/casper_vvel/PI_ne120/mocz_1-20_all.nc'}
EXPECT = {('0.1', 'LGM'): (11.12, 20.5), ('0.1', 'PI'): (18.55, 38.0)}
# The published bootstrap drew from one random stream that also served two other 240-month records right after the
# HR states. Their draws are consumed below without being used, so every band matches the published figure.
SKIP = (240, 240)
IFACE = {('0.1', 'LGM'): 1491, ('0.1', 'PI'): 2368, ('f19', 'LGM'): 2580, ('f19', 'PI'): 2870, ('c12', 'LGM'): 2540, ('c12', 'PI'): 2705}
LAT = np.arange(20.0, 65.01, 1.0)
col = {'0.1': '#1b3f6e', 'f19': '#2e7d4f', 'c12': '#7b3294'}
lab = {'0.1': 'HR (0.1°)', 'f19': 'LR (1°)', 'c12': 'CESM1.2 (1°)'}
lw = {'0.1': 2.0, 'f19': 1.3, 'c12': 1.3}


def load(p):
    d = nc.Dataset(p); moc = np.array(d.variables['MOC'][ATL]).astype('f8'); z = np.array(d.variables['z_t'][:]) / 100.0
    tl = np.array(d.variables['TLAT_1D'][:]); d.close()
    if tl.ndim == 3: tl = np.nanmean(tl, axis=0)
    lat = np.asarray(tl[ATL]).ravel()[:moc.shape[-1]]; ok = np.isfinite(lat); o = np.argsort(lat[ok])
    return lat[ok][o], z, moc[:, :, ok][:, :, o]


def curve(pm, lat, z, targets):   # maximum of psi, 500-3000 m, at each target latitude
    zm = (z >= 500.0) & (z <= 3000.0); out = []
    for t in targets:
        j = min(max(int(np.searchsorted(lat, t)) - 1, 0), len(lat) - 2); w = (t - lat[j]) / (lat[j + 1] - lat[j])
        a, b = pm[:, j], pm[:, j + 1]
        p = np.where(np.isfinite(a) & np.isfinite(b), (1 - w) * a + w * b, np.where(np.isfinite(a), a, b))
        out.append(np.nanmax(p[zm]))
    return np.array(out)


def interface(z, prof):   # depth where psi changes sign between its maximum and its minimum
    imax, imin = int(np.nanargmax(prof)), int(np.nanargmin(prof))
    if imin <= imax: return np.nan
    seg = np.arange(imax, imin + 1); s = np.sign(prof[seg])
    k = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    if not k.size: return np.nan
    i0, i1 = seg[k[0]], seg[k[0] + 1]
    return z[i0] + prof[i0] / (prof[i0] - prof[i1]) * (z[i1] - z[i0])


C, D, PROF = {}, {}, {}
for k, p in OURS.items():
    lat, z, moc = load(p); pm = np.nanmean(moc, axis=0)
    C[k] = curve(pm, lat, z, LAT)
    g = curve(pm, lat, z, [EXPECT[k][1]])[0]
    if abs(g - EXPECT[k][0]) > 0.05: sys.exit('STOP: gate failed for %s (%.2f vs %.2f)' % (k, g, EXPECT[k][0]))
    n = moc.shape[0]; Lb = 24; starts = np.arange(0, n - Lb + 1); kk = int(np.ceil(n / Lb))
    D[k] = np.array([curve(np.nanmean(moc[(starts[rng.integers(0, len(starts), size=kk)][:, None] + np.arange(Lb)[None, :]).ravel()[:n]], axis=0), lat, z, LAT) for _ in range(NB)])
    j = int(np.nanargmin(np.abs(lat + 30.0))); PROF[k] = (z, pm[:, j])
for n in SKIP:
    Lb = 24; starts = np.arange(0, n - Lb + 1); kk = int(np.ceil(n / Lb))
    for _ in range(NB): rng.integers(0, len(starts), size=kk)
for st in ('LGM', 'PI'):
    lat, z, moc, yrs = F19.load_annual(*F19.WIN[st]); k = ('f19', st)
    pm = np.nanmean(moc, axis=0); C[k] = curve(pm, lat, z, LAT)
    n = moc.shape[0]; Lb = 2; starts = np.arange(0, n - Lb + 1); kk = int(np.ceil(n / Lb))
    D[k] = np.array([curve(np.nanmean(moc[(starts[rng.integers(0, len(starts), size=kk)][:, None] + np.arange(Lb)[None, :]).ravel()[:n]], axis=0), lat, z, LAT) for _ in range(NB)])
    j = int(np.nanargmin(np.abs(lat + 30.0))); PROF[k] = (z, pm[:, j])
for st in ('LGM', 'PI'):
    lat, z, moc, yrs = F19.load_annual(*F19.CONF['c12'][st]); k = ('c12', st)
    pm = np.nanmean(moc, axis=0); C[k] = curve(pm, lat, z, LAT)
    n = moc.shape[0]; Lb = 2; starts = np.arange(0, n - Lb + 1); kk = int(np.ceil(n / Lb))
    D[k] = np.array([curve(np.nanmean(moc[(starts[rng.integers(0, len(starts), size=kk)][:, None] + np.arange(Lb)[None, :]).ravel()[:n]], axis=0), lat, z, LAT) for _ in range(NB)])
    j = int(np.nanargmin(np.abs(lat + 30.0))); PROF[k] = (z, pm[:, j])
for k, want in IFACE.items():
    zi = interface(*PROF[k])
    if abs(zi - want) > 1.0: sys.exit('STOP: 30S interface %s %.0f m, expected %d' % (k, zi, want))
print('[gate] strength curves and 30S interfaces reproduce the expected values', flush=True)

uplt.rc.update({'font.size': 7, 'title.size': 7.5, 'abc.size': 8, 'label.size': 7, 'tick.labelsize': 6.5, 'legend.fontsize': 6.3})
fig, axs = uplt.subplots([[1] * 4 + [2] * 4 + [3] * 4, [4] * 6 + [5] * 6], figwidth=7.0, hratios=(1.1, 1.0), hspace=5.2,
                         wspace=[0.3, 0.3, 0.3, 2.6, 0.3, 3.8, 0.3, 2.6, 0.3, 0.3, 0.3], share=False)
for i, r in enumerate(('0.1', 'f19', 'c12')):
    ax = axs[i]
    for st, ls in (('PI', '--'), ('LGM', '-')):
        z, prof = PROF[(r, st)]
        ax.plot(prof, z, color=col[r], lw=1.6, ls=ls, label=st)
        zi = interface(z, prof)
        ax.axhline(zi, color=col[r], lw=0.8, ls=':', alpha=0.9)
        ax.text(0.97, zi + (-60 if st == 'LGM' else 190), '%.0f m' % zi, color=col[r], fontsize=6.3, ha='right',
                va='bottom' if st == 'LGM' else 'top', transform=ax.get_yaxis_transform())
    ax.axvline(0, color='0.4', lw=0.7)
    ax.format(title=lab[r], xlabel='ψ at 30°S (Sv)', ylabel='depth (m)' if i == 0 else '', ylim=(5200, 0), xlim=(-12, 23),
              yticklabels=None if i == 0 else [])
    if i == 0: ax.legend(loc='lr', ncols=1, frame=False)
for r in ('0.1', 'f19', 'c12'):
    axs[3].plot(LAT, C[(r, 'LGM')], color=col[r], lw=lw[r], label='%s LGM' % lab[r])
    axs[3].plot(LAT, C[(r, 'PI')], color=col[r], lw=lw[r], ls='--', label='%s PI' % lab[r])
    d = D[(r, 'LGM')] - D[(r, 'PI')]
    axs[4].fill_between(LAT, np.percentile(d, 5, axis=0), np.percentile(d, 95, axis=0), color=col[r], alpha=0.25, lw=0)
    axs[4].plot(LAT, C[(r, 'LGM')] - C[(r, 'PI')], color=col[r], lw=lw[r], label=lab[r])
axs[4].axhline(0, color='k', lw=0.7)
for ax in axs[3:]:
    for x in (25, 30, 35): ax.axvline(x, color='0.6', lw=0.6, ls=':')
axs[3].format(ylabel='maximum of ψ, 500 to 3000 m (Sv)', title='strength by latitude', xlabel='latitude (°N)', xlim=(20, 65))
axs[4].format(ylabel='LGM − PI (Sv)', title='glacial change, 5 to 95% bootstrap', xlabel='latitude (°N)', xlim=(20, 65))
axs[3].legend(loc='b', ncols=3, frame=False)
axs[4].legend(loc='lr', ncols=2, frame=False)
axs.format(abc='(a)', abcloc='ul')
out = OUT + '/figS3_overturning.png'
fig.savefig(out, dpi=400, bbox_inches='tight'); print('wrote', out)
