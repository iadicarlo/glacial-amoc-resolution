#!/usr/bin/env python
"""Figure S5: glacial minus pre-industrial d18O of precipitation at 17 ice-core sites for HR, LR and CESM1.2 against
the ice-core values compiled by Cauquoin et al. (2023, Clim. Past, Table 2), after bringing every run to one seawater
d18O reference (+1 permil glacial, 0 pre-industrial); 5-95% moving-block bootstrap intervals. Also prints the GMST
change. Input: site d18O and GMST reductions of each run. Output: figS5_isotopes.png and metrics_iso_icecores.csv.
"""
import csv, os, sys
import numpy as np, netCDF4 as nc, ultraplot as uplt

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
D = DATA + '/casper_vvel/atm_iso'
OUTF = OUT + '/figS5_isotopes.png'
OUTC = OUT + '/metrics_iso_icecores.csv'
# mean seawater d18O of each run over its window (permil) and the reference it is brought to; corrected value
# d_corr = (1000 + d) * f - 1000 with f = (1000 + d_target) / (1000 + d_run)
OCEAN = {'HRLGM': 1.085, 'HRPI': 0.000}
TARGET = {'HRLGM': 1.0, 'HRPI': 0.0, 'F19LGM': 1.0, 'F19PI': 0.0, 'C12LGM': 1.0, 'C12PI': 0.0}
import f19_common as F19
for _r in ('F19LGM', 'F19PI', 'C12LGM', 'C12PI'):   # measured from each run's own seawater d18O
    _d = nc.Dataset(F19.R18O_ANY[_r]); OCEAN[_r] = float(_d.d18O_sw_window_mean); _d.close()
print("seawater d18O offsets (permil):", {k: round(v, 3) for k, v in OCEAN.items()}, flush=True)
CONF = {'0.1': ('HRLGM', 'HRPI'), 'f19': ('F19LGM', 'F19PI'), 'c12': ('C12LGM', 'C12PI')}
# Cauquoin et al. 2023, Table 2: site, observed PI d18O, observed LGM-PI d18O (permil)
OBS = {'South Pole': (-51.0, -6.4), 'Vostok': (-56.8, -4.8), 'Dome F': (-54.6, -4.9), 'EDC': (-50.4, -5.6),
       'EDML': (-44.8, -6.3), 'Law Dome': (-22.4, -5.5), 'Taylor Dome': (-40.5, -3.5), 'Talos': (-36.1, -5.4),
       'Byrd': (-32.9, -7.3), 'Siple Dome': (-25.6, -7.8), 'WDC': (-34.0, -7.3), 'GRIP': (-35.3, -5.4),
       'NGRIP': (-35.2, -7.4), 'NEEM': (-33.0, -10.0), 'Camp Century': (-29.3, -12.9), 'Dye3': (-27.7, -7.3),
       'Renland': (-27.4, -3.8)}
GREEN = ['GRIP', 'NGRIP', 'NEEM', 'Camp Century', 'Dye3', 'Renland']
rng = np.random.default_rng(20260811); NB = 400
# The published intervals came from one random stream that also served two other 20-year records right after the
# 0.1 degree pair. Their draws are consumed without being used (before 'f19'), so every interval is unchanged.
SKIP = (20, 20)


def load(run):
    d = nc.Dataset(f'{D}/atm_iso_gmst_{run}.nc')
    names = d.site_names.split(';')
    r = dict(names=names, S18=np.array(d['site_S18'][:]), S16=np.array(d['site_S16'][:]), elev=np.array(d['site_elev'][:]),
             gmst=float(d.gmst_C), gmst_y=np.array(d['gmst_year'][:]), win=d.window)
    d.close(); return r


def corr(dv, run):
    f = (1000.0 + TARGET[run]) / (1000.0 + OCEAN[run])
    return (1000.0 + dv) * f - 1000.0


def boot_idx(n, L=2):
    starts = np.arange(0, n - L + 1); k = int(np.ceil(n / L))
    return [(starts[rng.integers(0, len(starts), size=k)][:, None] + np.arange(L)[None, :]).ravel()[:n] for _ in range(NB)]


R = {r: load(r) for r in OCEAN}
for r, v in R.items():
    if not (np.isfinite(v['gmst']) and 0 < v['gmst'] < 20): sys.exit(f"STOP: GMST of {r} is {v['gmst']}")
names = R['HRPI']['names']
rows, bad = [], []
res = {}
for res_, (L, P) in CONF.items():
    if res_ == 'f19':
        for n_ in SKIP: boot_idx(n_)
    iL, iP = boot_idx(R[L]['S18'].shape[0]), boot_idx(R[P]['S18'].shape[0])
    for s, nm in enumerate(names):
        d = {}
        for run, idxs in ((L, iL), (P, iP)):
            S18, S16 = R[run]['S18'][:, s], R[run]['S16'][:, s]
            full = (S18.sum() / S16.sum() - 1) * 1000
            draws = np.array([(S18[i].sum() / S16[i].sum() - 1) * 1000 for i in idxs])
            d[run] = (full, draws)
        if d[P][0] > -10: bad.append((res_, nm, d[P][0]))
        raw = d[L][0] - d[P][0]
        cL, cP = corr(d[L][0], L), corr(d[P][0], P)
        cd = corr(d[L][1], L) - corr(d[P][1], P)
        res[(res_, nm)] = (cL - cP, raw)
        rows.append([res_, nm, OBS[nm][0], OBS[nm][1], d[P][0], d[L][0], raw, cP, cL, cL - cP,
                     np.percentile(cd, 5), np.percentile(cd, 95), R[P]['elev'][s], R[L]['elev'][s]])
if bad: sys.exit(f"STOP: pre-industrial d18Op above -10 permil at {bad}: wrong column or units")
print("[gate] every site's pre-industrial d18Op is below -10 permil at both resolutions: PASSED")

hdr = ['res', 'site', 'obs_PI', 'obs_change', 'model_PI_raw', 'model_LGM_raw', 'change_raw', 'model_PI_corr',
       'model_LGM_corr', 'change_corr', 'change_corr_p5', 'change_corr_p95', 'elev_PI_m', 'elev_LGM_m']
with open(OUTC, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
print(f"\n{'site':13s} {'obs':>6s} | " + " | ".join(f"{c + ' corr [5,95]':>24s}" for c in CONF) + " | PI d18Op obs/" + "/".join(CONF))
for nm in names:
    a = {r[0]: r for r in rows if r[1] == nm}
    print(f"{nm:13s} {OBS[nm][1]:+6.1f} | " + " | ".join(f"{a[c][9]:+7.2f} [{a[c][10]:+6.2f},{a[c][11]:+6.2f}]      " for c in CONF)
          + f" | {OBS[nm][0]:.1f}/" + "/".join(f"{a[c][4]:.1f}" for c in CONF))
print()
for region, sel in (('Greenland', GREEN), ('Antarctica', [n for n in names if n not in GREEN])):
    obs = np.array([OBS[n][1] for n in sel])
    for res_ in CONF:
        for k, lab in ((0, 'corrected'), (1, 'raw')):
            m = np.array([res[(res_, n)][k] for n in sel])
            print(f"{region:10s} {res_:>3s} {lab:9s}: mean bias {np.mean(m - obs):+6.2f}, RMSE {np.sqrt(np.mean((m - obs) ** 2)):5.2f}, "
                  f"r {np.corrcoef(m, obs)[0, 1]:+.2f}", end='')
            if res_ == '0.1':
                for other in ('f19', 'c12'):
                    mo = np.array([res[(other, n)][k] for n in sel])
                    print(f" | 0.1 closer than {other} at {int(np.sum(np.abs(m - obs) < np.abs(mo - obs)))} of {len(sel)}", end='')
            print()

print("\n=== GMST (area-weighted TREFHT, month-length weighted) ===")
for res_, (L, P) in CONF.items():
    if res_ == 'f19':
        for n_ in SKIP: boot_idx(n_)
    gl, gp = R[L]['gmst_y'], R[P]['gmst_y']
    dr = np.array([gl[i].mean() - gp[j].mean() for i, j in zip(boot_idx(len(gl)), boot_idx(len(gp)))])
    print(f"  {res_:>3s} deg: LGM {R[L]['gmst']:.3f} C ({R[L]['win']}), PI {R[P]['gmst']:.3f} C ({R[P]['win']}), "
          f"change {R[L]['gmst'] - R[P]['gmst']:+.2f} K [{np.percentile(dr, 5):+.2f},{np.percentile(dr, 95):+.2f}]")

fig, axs = uplt.subplots(ncols=2, refwidth=2.9, refaspect=1, share=False, wspace=5)
C = {'0.1': '#1b3f6e', 'f19': '#2e7d4f', 'c12': '#7b3294'}
LBL = {'0.1': 'HR (0.1$^\\circ$)', 'f19': 'LR (1$^\\circ$)', 'c12': 'CESM1.2 (1$^\\circ$)'}
PLOT = ('0.1', 'f19', 'c12')
for ax, (region, sel) in zip(axs, (('Greenland', GREEN), ('Antarctica', [n for n in names if n not in GREEN]))):
    lo = min(min(OBS[n][1] for n in sel), min(min(res[(r_, n)]) for r_ in PLOT for n in sel)) - 1.5
    # upper limit from the data
    hi = max(1.0, max(r[11] for r in rows if r[0] in PLOT and r[1] in sel) + 0.8)
    ax.plot([lo, hi], [lo, hi], color='0.5', lw=0.8, ls='--', zorder=1)
    for res_, mk in (('0.1', 'o'), ('f19', '^'), ('c12', 'D')):
        x = [OBS[n][1] for n in sel]
        a = [r for r in rows if r[0] == res_ and r[1] in sel]
        y = [r[9] for r in a]; e = [[r[9] - r[10] for r in a], [r[11] - r[9] for r in a]]
        al = 1.0
        ax.errorbar(x, y, yerr=e, fmt=mk, color=C[res_], mec=C[res_], mfc=C[res_], ms=6, lw=0.9, capsize=2, alpha=al,
                    label=LBL[res_], zorder=3)
        # corrected values only; the correction moves each point by the amount printed here
        print(f"{region} {res_}: largest |corrected - uncorrected| {max(abs(r[9] - r[6]) for r in a):.3f} permil")
    if region == 'Greenland':
        for n in sel:
            ax.text(OBS[n][1] + 0.2, res[('0.1', n)][0] + 0.3, n, size=6.5, color='0.25')
    ax.format(title=region, xlabel='ice core LGM $-$ PI $\\delta^{18}$O (permil)', ylabel='model LGM $-$ PI $\\delta^{18}$O$_p$ (permil)',
              xlim=(lo, hi), ylim=(lo, hi))
# legend: plain markers, smaller than in the panels, no error-bar glyphs
from matplotlib.lines import Line2D
hd = [Line2D([0], [0], marker=mk, ls='none', ms=4.5, color=C[r_], mec=C[r_]) for r_, mk in (('0.1', 'o'), ('f19', '^'), ('c12', 'D'))]
fig.legend(hd, [LBL[r_] for r_ in ('0.1', 'f19', 'c12')], loc='b', ncols=3, frame=False, prop={'size': 7})
fig.format(abc='(a)', abcloc='l')
fig.savefig(OUTF, dpi=400, bbox_inches='tight'); print('wrote', OUTF, 'and', OUTC)
