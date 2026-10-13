#!/usr/bin/env python
"""Figure S1: full-run time series of annual AMOC strength, GMST, and subpolar North Atlantic (SPNA, 45-65N, 60-10W)
SST and March sea-ice extent for every glacial and pre-industrial run (HR, LR, CESM1.2), analysis windows shaded. The
0.1 degree glacial run is drawn as its chain (LGM.004 years 1-100, then test.001 years 1-15). Also prints the drift at
the end of each run and inside each analysis window. Output: figS1_timeseries.png.
"""
import glob, os, sys
import numpy as np, netCDF4 as nc, ultraplot as uplt

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
V = DATA + '/casper_vvel'
ATL = 1


def strength(pm, z, lat):
    return float(np.nanmax(pm[np.ix_((z >= 500) & (z <= 3000), (lat >= 20) & (lat <= 65))]))


def load(pattern):
    fs = sorted(glob.glob(pattern)); assert fs, pattern
    P, Y = [], []
    for f in fs:
        d = nc.Dataset(f); P.append(np.array(d.variables['MOC'][ATL], 'f8')); Y.append(np.array(d.variables['year'][:]))
        z = np.array(d.variables['z_t'][:]) / 100.0; tl = np.array(d.variables['TLAT_1D'][:]); d.close()
    psi, yrs = np.concatenate(P), np.concatenate(Y); o = np.argsort(yrs); psi, yrs = psi[o], yrs[o]
    lat = np.asarray(tl[ATL]).ravel()[:psi.shape[-1]]; ok = np.isfinite(lat); oo = np.argsort(lat[ok])
    return yrs, psi[:, :, ok][:, :, oo], z, lat[ok][oo]


def amoc_series(pattern):
    yrs, psi, z, lat = load(pattern)
    return yrs, np.array([strength(p, z, lat) for p in psi]), (psi, z, lat)


def monthly_to_annual(path):
    d = nc.Dataset(path); moc = np.array(d.variables['MOC'][ATL], 'f8'); z = np.array(d.variables['z_t'][:]) / 100.0
    tl = np.array(d.variables['TLAT_1D'][:]); d.close()
    if tl.ndim == 3: tl = np.nanmean(tl, axis=0)
    lat = np.asarray(tl[ATL]).ravel()[:moc.shape[-1]]; ok = np.isfinite(lat); oo = np.argsort(lat[ok])
    moc = moc[:, :, ok][:, :, oo]; lat = lat[ok][oo]; ny = moc.shape[0] // 12
    ann = moc[:12 * ny].reshape(ny, 12, *moc.shape[1:]).mean(axis=1)
    return np.arange(1, ny + 1), np.array([strength(p, z, lat) for p in ann])


def sfc(run):
    d = nc.Dataset(f'{V}/timeseries/ts_sfc_{run}.nc')
    r = {k: np.array(d[k][:]) for k in ('year', 'gmst', 'sst_spna', 'si_ext_mar_spna')}; d.close(); return r


# ---- AMOC
A = {}
y, s, (psi, z, lat) = amoc_series(f'{V}/hrpi_mocz/mocz_HRPI_*_annual.nc'); A[('0.1', 'PI')] = (y, s)
i = (y >= 41) & (y <= 60); g = strength(psi[i].mean(axis=0), z, lat)
print(f'[gate] 0.1 degree PI 41-60 from annual psi: {g:.2f} Sv vs paper 18.55', flush=True)
if abs(g - 18.55) > 0.05: sys.exit('STOP: annual PI pipeline does not reproduce the paper')
y1, s1, _ = amoc_series(f'{V}/lgm004_annual/mocz_LGM004_*_annual.nc')
# test.001 = the glacial analysis segment, years 1-15, one monthly file per year (mocz_00NN_all.nc)
MZ = DATA + '/cesm/icesm13/archive/b.e13.Bi1850C5.ne120_t12.lgm.test.001/ocn/proc/mocz'
y2, s2 = [], []
for yy_ in range(1, 16):
    f_ = f'{MZ}/mocz_{yy_:04d}_all.nc'
    try:
        _, v_ = monthly_to_annual(f_)
    except FileNotFoundError:
        sys.exit(f'STOP: {f_} missing')
    assert v_.size == 1, f'{f_}: {v_.size} years'
    y2.append(yy_); s2.append(float(v_[0]))
y2, s2 = np.array(y2), np.array(s2)
print('  test.001 annual AMOC years 1-15:', np.round(s2, 2), flush=True)


def cell_base(lat, z, psi):
    li = np.where((lat >= 30.0) & (lat <= 45.0))[0]
    p = np.nanmean(psi[:, li], axis=1)
    kmx = int(np.nanargmax(p))
    for k in range(kmx, len(z) - 1):
        if p[k] >= 0.0 and p[k + 1] < 0.0:
            f = p[k] / (p[k] - p[k + 1])
            return z[k] + f * (z[k + 1] - z[k])
    return np.nan


# strength and cell base of the window-mean psi over three windows of test.001
M = []
for yy_ in range(1, 16):
    d_ = nc.Dataset(f'{MZ}/mocz_{yy_:04d}_all.nc'); m_ = np.array(d_.variables['MOC'][ATL], 'f8').mean(axis=0)
    zz_ = np.array(d_.variables['z_t'][:]) / 100.0; tl_ = np.array(d_.variables['TLAT_1D'][:]); d_.close()
    if tl_.ndim == 3: tl_ = np.nanmean(tl_, axis=0)
    la_ = np.asarray(tl_[ATL]).ravel()[:m_.shape[-1]]; ok_ = np.isfinite(la_); oo_ = np.argsort(la_[ok_])
    M.append(m_[:, ok_][:, oo_]); LA_ = la_[ok_][oo_]
M = np.array(M)
for a_, b_ in ((1, 12), (1, 15), (4, 15)):
    pm_ = M[a_ - 1:b_].mean(axis=0)
    print(f'  test.001 window {a_}-{b_}: strength {strength(pm_, zz_, LA_):.2f} Sv, cell base {cell_base(LA_, zz_, pm_):.1f} m', flush=True)
A[('0.1', 'LGM')] = (np.concatenate([y1, y2 + 100]), np.concatenate([s1, s2]))
for key, pat in ((('f19', 'LGM'), 'f19_mocz/mocz_F19LGM_*'), (('f19', 'PI'), 'f19_mocz/mocz_F19PI_*'),
                 (('c12', 'LGM'), 'c12_mocz/mocz_C12LGM_*'), (('c12', 'PI'), 'c12_mocz/mocz_C12PI_*')):
    yy, ss, _ = amoc_series(f'{V}/{pat}_annual.nc'); A[key] = (yy, ss)
for k, (yy, ss) in A.items():
    print(f'  AMOC {k}: years {yy[0]}-{yy[-1]} ({yy.size}), {ss.min():.2f}..{ss.max():.2f} Sv', flush=True)

# ---- surface
S = {('0.1', 'PI'): sfc('HRPI'), ('f19', 'LGM'): sfc('F19LGM'), ('f19', 'PI'): sfc('F19PI'),
     ('c12', 'LGM'): sfc('C12LGM'), ('c12', 'PI'): sfc('C12PI')}
a, b = sfc('HRLGM004'), sfc('HRTEST001')
S[('0.1', 'LGM')] = {k: np.concatenate([a[k], b[k] + (100 if k == 'year' else 0)]) for k in a}

WIN = {('0.1', 'LGM'): (101, 115), ('0.1', 'PI'): (41, 60),
       ('f19', 'LGM'): (981, 1000), ('f19', 'PI'): (485, 504), ('c12', 'LGM'): (881, 900), ('c12', 'PI'): (881, 900)}
COLS = [('0.1', 'HR: iCESM1.3 0.1$^\\circ$ (ne120_t12)'), ('f19', 'LR: iCESM1.3 1$^\\circ$ (f19_g16)'),
        ('c12', 'CESM1.2 1$^\\circ$ (f19_g16)')]
ROWS = [('amoc', 'AMOC (Sv)'), ('gmst', 'GMST ($^\\circ$C)'), ('sst_spna', 'SPNA SST ($^\\circ$C)'),
        ('si_ext_mar_spna', 'SPNA March ice\nextent (10$^6$ km$^2$)')]
# x shared within each column (limits and tick labels, so only the bottom row is labelled and no tick row
# collides with the panel below it); y not shared and not spanned, so each row keeps its own label and range.
fig, axs = uplt.subplots(nrows=4, ncols=len(COLS), refwidth=2.0, refaspect=1.6, sharex=True, sharey=False, spany=False)
for c, (cfg, title) in enumerate(COLS):
    for r, (var, ylab) in enumerate(ROWS):
        ax = axs[r, c]
        for st, col in (('LGM', '#1b3f6e' if cfg == '0.1' else {'f19': '#2e7d4f', 'c12': '#7b3294'}[cfg]), ('PI', '0.5')):
            if var == 'amoc':
                yy, vv = A[(cfg, st)]
            else:
                yy, vv = S[(cfg, st)]['year'], S[(cfg, st)][var]
            lw = 0.9 if yy.size > 200 else 1.3
            ax.plot(yy, vv, color=col, lw=lw, label=st if r == 1 else None)
            w0, w1 = WIN[(cfg, st)]
            ax.axvspan(w0 - 0.5, w1 + 0.5, color=col, alpha=0.12, lw=0)
        if cfg == '0.1':
            ax.axvline(100.5, color='0.4', lw=0.6, ls=':')
        ax.format(ylabel=ylab if c == 0 else '', xlabel='model year' if r == 3 else '', title=title if r == 0 else '')
for c in range(3):   # LGM and PI keys in the empty band between the two GMST curves
    axs[1, c].legend(loc='cr', ncols=1, frame=False, prop={'size': 7})
axs.format(abc='(a)', abcloc='ul', abcsize=7, titlesize=8)
out = OUT + '/figS1_timeseries.png'
fig.savefig(out, dpi=250, bbox_inches='tight'); print('wrote', out)


def trend(yy, vv, n):
    # least-squares slope per century over the final n years of a series (None if the series is shorter than n)
    k = yy >= yy[-1] - (n - 1)
    return np.polyfit(yy[k], vv[k], 1)[0] * 100.0 if k.sum() >= n else None


def fmt(t, u):
    return '     n/a' if t is None else f'{t:+8.2f}'


# Drift at the END of each run, which is what an equilibration check asks. The last-100-year slope of the
# 0.1 degree chain spans the adjustment inside LGM.004, so the final 30 years are printed next to it.
print('\n  drift per century over the final 30 and 100 years of each run (n/a: run shorter than that)')
print(f"  {'cfg':4s} {'st':3s} {'AMOC 30':>8s} {'AMOC 100':>8s} {'GMST 30':>8s} {'GMST 100':>8s} {'SST 30':>8s} {'SST 100':>8s} {'ice 30':>8s} {'ice 100':>8s}   window means (AMOC, GMST, SST, ice)")
for key in sorted(A):
    yy, ss = A[key]; d = S[key]; w0, w1 = WIN[key]
    ia = (yy >= w0) & (yy <= w1); i = (d['year'] >= w0) & (d['year'] <= w1)
    cols = [trend(yy, ss, 30), trend(yy, ss, 100)]
    for v in ('gmst', 'sst_spna', 'si_ext_mar_spna'):
        cols += [trend(d['year'], d[v], 30), trend(d['year'], d[v], 100)]
    print(f'  {key[0]:4s} {key[1]:3s} ' + ' '.join(fmt(c, '') for c in cols)
          + f'   {ss[ia].mean():6.2f} Sv, {np.nanmean(d["gmst"][i]):5.2f} C, {np.nanmean(d["sst_spna"][i]):5.2f} C, {np.nanmean(d["si_ext_mar_spna"][i]):4.2f}', flush=True)


# Drift INSIDE each analysis window (for the 0.1 degree glacial state: chain years 101-115, the test.001 segment).
# Identical treatment for every run: the least-squares slope over its own window, with a 90% interval whose
# standard error uses the effective sample size of Santer et al. (2000, JGR 105:7337, doi:10.1029/1999JD901105),
# n_e = n (1 - r1) / (1 + r1) with r1 the lag-1 autocorrelation of the residuals (n_e = n when r1 <= 0), and a
# t distribution with n_e - 2 degrees of freedom.
from scipy import stats


def santer(t, y):
    t = np.asarray(t, 'f8'); y = np.asarray(y, 'f8'); ok = np.isfinite(y); t, y = t[ok], y[ok]
    n = t.size; b, a = np.polyfit(t, y, 1); e = y - (a + b * t)
    r1 = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if n > 3 else 0.0
    ne = n * (1 - r1) / (1 + r1) if r1 > 0 else float(n)
    if ne <= 3.0:
        return b * 100, np.nan, np.nan, ne, r1
    se = np.sqrt(np.sum(e ** 2) / (ne - 2)) / np.sqrt(np.sum((t - t.mean()) ** 2))
    q = stats.t.ppf(0.95, ne - 2)
    return b * 100, (b - q * se) * 100, (b + q * se) * 100, ne, r1


def show(label, t, y, unit):
    b, lo, hi, ne, r1 = santer(t, y)
    sig = 'n/a' if not np.isfinite(lo) else ('excludes 0' if (lo > 0 or hi < 0) else 'includes 0')
    print(f'    {label:14s} {b:+7.2f} [{lo:+7.2f}, {hi:+7.2f}] {unit} per century  (n {len(t)}, r1 {r1:+.2f}, n_e {ne:4.1f}; {sig})', flush=True)


print('\n  drift INSIDE each analysis window: slope per century [90% interval, Santer et al. 2000]')
VARS = (('AMOC', None, 'Sv'), ('GMST', 'gmst', 'K'), ('SPNA SST', 'sst_spna', 'K'), ('SPNA ice', 'si_ext_mar_spna', '1e6 km2'))
cases = [(k, WIN[k]) for k in sorted(A)] + [(('0.1', 'LGM'), (100, 115))]
for key, (w0, w1) in cases:
    tag = '%s %s years %d-%d' % (key[0], key[1], w0, w1) + ('  (test.001 plus the last LGM.004 year)' if (w0, w1) == (100, 115) else '')
    print('  ' + tag, flush=True)
    yy, ss = A[key]; i = (yy >= w0) & (yy <= w1)
    if i.sum() >= 10:
        show('AMOC', yy[i], ss[i], 'Sv')
    d = S[key]; j = (d['year'] >= w0) & (d['year'] <= w1)
    for lab, v, u in VARS[1:]:
        show(lab, d['year'][j], d[v][j], u)
