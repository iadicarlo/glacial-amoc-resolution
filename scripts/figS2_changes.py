#!/usr/bin/env python
"""Figure S2: (a-c) annual AMOC strength in the long pre-industrial control of each configuration (HR, LR, CESM1.2)
with the glacial and pre-industrial window means; (d-j) LGM minus PI change of seven overturning metrics with 5-95%
moving-block bootstrap intervals, and gray bars for the largest difference between two independent control windows.
Inputs: metrics_lit_definitions.csv and the annual overturning of the controls. Output: figS2_changes.png.
"""
import csv, glob, os, sys
import numpy as np, netCDF4 as nc, ultraplot as uplt
from f19_common import load_annual

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
V = DATA + '/casper_vvel'; ATL = 1
CONF = [('0.1', 'HR (0.1°)', '#1b3f6e'), ('f19', 'LR (1°)', '#2e7d4f'), ('c12', 'CESM1.2 (1°)', '#7b3294')]
METRICS = [('paper_strength', 'maximum, 20 to 65°N (Sv)'), ('K21_max', 'maximum at 30°N (Sv)'), ('MS21_max', 'maximum at 25°N (Sv)'),
           ('cell_base', 'cell base (m)'), ('K21_extent', 'NADW extent (m)'), ('MS21_depth', 'cell boundary (m)'),
           ('interface_30S', '30°S interface (m)')]
# HR strength and cell-base change with the 5-95% interval quoted in the paper (15-year window bootstrap)
HR_PAPER = {('0.1', 'paper_strength'): (-7.43, -8.06, -7.11), ('0.1', 'cell_base'): (-683.0, -768.0, -480.0)}
# largest difference between two independent windows of the same lengths in each control (gray bars)
CTRLMAX = {('0.1', 'paper_strength'): 4.38, ('0.1', 'cell_base'): 333.0, ('0.1', 'interface_30S'): 385.0,
           ('f19', 'paper_strength'): 2.05, ('f19', 'cell_base'): 460.0, ('f19', 'interface_30S'): 517.0,
           ('c12', 'paper_strength'): 3.11, ('c12', 'cell_base'): 255.0, ('c12', 'interface_30S'): 150.0}
WIN = {'0.1': (11.12, 18.55), 'f19': (25.34, 22.33), 'c12': (24.41, 23.73)}   # LGM, PI window-mean strength (Sv)

C = {}
with open(DATA + '/derived/metrics_lit_definitions.csv') as fh:
    for r in csv.DictReader(fh):
        if r['config'] in WIN:
            C[(r['config'], r['metric'])] = (float(r['change']), float(r['change_p5']), float(r['change_p95']))
for k, (v, lo, hi) in HR_PAPER.items():
    tol = 0.01 if 'strength' in k[1] else 1.0
    if abs(C[k][0] - v) > tol: sys.exit('STOP: %s central %.3f vs expected %.2f' % (k, C[k][0], v))
    C[k] = (C[k][0], lo, hi)
for c in WIN:
    s = C[(c, 'paper_strength')][0]; want = WIN[c][0] - WIN[c][1]
    if abs(s - want) > 0.011: sys.exit('STOP: %s strength change %.3f vs window means %.3f' % (c, s, want))
print('[check] CSV central values equal the paper values; HR strength and cell-base intervals as quoted in the paper', flush=True)


def strength(pm, z, lat):
    return float(np.nanmax(pm[np.ix_((z >= 500) & (z <= 3000), (lat >= 20) & (lat <= 65))]))


def load_glob(pattern):   # same operations as figS1_timeseries.load
    fs = sorted(glob.glob(pattern)); assert fs, pattern
    P, Y = [], []
    for f in fs:
        d = nc.Dataset(f); P.append(np.array(d.variables['MOC'][ATL], 'f8')); Y.append(np.array(d.variables['year'][:]))
        z = np.array(d.variables['z_t'][:]) / 100.0; tl = np.array(d.variables['TLAT_1D'][:]); d.close()
    psi, yrs = np.concatenate(P), np.concatenate(Y); o = np.argsort(yrs); psi, yrs = psi[o], yrs[o]
    lat = np.asarray(tl[ATL]).ravel()[:psi.shape[-1]]; ok = np.isfinite(lat); oo = np.argsort(lat[ok])
    return yrs, psi[:, :, ok][:, :, oo], z, lat[ok][oo]


T = {}
y, psi, z, lat = load_glob(V + '/ihesp_hr_ctrl_mocz/mocz_CTRL_*_annual.nc')
k = (y >= 150) & (y <= 519); T['0.1'] = (y[k], np.array([strength(p, z, lat) for p in psi[k]]))
for c, run, y0, y1 in (('f19', 'F19PI', 105, 504), ('c12', 'C12PI', 101, 900)):
    lat_, z_, ann, yrs = load_annual(run, y0, y1); T[c] = (yrs, np.array([strength(p, z_, lat_) for p in ann]))
for c, (yy, ss) in T.items():
    assert np.all(np.diff(yy) == 1) and 5 < ss.min() and ss.max() < 40, c
    print('control %-4s years %d-%d (%d): mean %.2f Sv, annual sd %.2f, range %.2f..%.2f' % (c, yy[0], yy[-1], yy.size, ss.mean(), ss.std(ddof=1), ss.min(), ss.max()), flush=True)
assert T['0.1'][0].size == 370 and T['f19'][0].size == 400 and T['c12'][0].size == 800, 'control spans differ from the paper'

uplt.rc.update({'font.size': 7, 'title.size': 7, 'abc.size': 8, 'label.size': 7, 'tick.labelsize': 6.5})
layout = [[1] * 4 + [2] * 4 + [3] * 4, [4] * 4 + [5] * 4 + [6] * 4, [7] * 3 + [8] * 3 + [9] * 3 + [10] * 3]
fig, axs = uplt.subplots(layout, figwidth=7.0, hratios=(1.15, 0.8, 0.8), wspace=1.4, hspace=(6.0, 6.0), share=False)
for i, (c, name, col) in enumerate(CONF):
    ax = axs[i]; yy, ss = T[c]
    rm = np.convolve(ss, np.ones(20) / 20.0, mode='valid'); ry = yy[9:9 + rm.size] + 0.5
    ax.plot(yy, ss, color=col, lw=0.4, alpha=0.55); ax.plot(ry, rm, color=col, lw=1.4)
    ax.axhline(WIN[c][0], color=col, ls='--', lw=1.0); ax.axhline(WIN[c][1], color='0.35', ls=':', lw=1.0)
    ax.format(title=name + ' control', xlabel='model year', ylabel='AMOC strength (Sv)' if i == 0 else '', ylim=(9, 28), yticklabels=None if i == 0 else [])
for j, (m, label) in enumerate(METRICS):
    ax = axs[3 + j]
    for i, (c, name, col) in enumerate(CONF):
        v, lo, hi = C[(c, m)]; yrow = 2 - i
        if (c, m) in CTRLMAX:
            w = CTRLMAX[(c, m)]
            ax.fill_betweenx([yrow - 0.28, yrow + 0.28], -w, w, color='0.86', lw=0, zorder=0)
        ax.errorbar(v, yrow, xerr=[[v - lo], [hi - v]], fmt='o', color=col, ms=3.6, elinewidth=1.0, capsize=1.8, zorder=3)
    ax.axvline(0, color='k', lw=0.6, zorder=1)
    ext = [C[(c, m)][1] for c, _, _ in CONF] + [C[(c, m)][2] for c, _, _ in CONF] + [0.0] + \
          [s_ * CTRLMAX[(c, m)] for c, _, _ in CONF if (c, m) in CTRLMAX for s_ in (-1, 1)]
    pad = 0.08 * (max(ext) - min(ext)); ax.format(xlim=(min(ext) - pad, max(ext) + pad))
    ax.format(title=label, yticks=[2, 1, 0], yticklabels=[n for _, n, _ in CONF] if j in (0, 3) else [], ylim=(-0.6, 2.6),
              ytickminor=False, ygrid=False, xlabel='LGM − PI')
axs.format(abc='(a)', abcloc='ul')
p = OUT + '/figS2_changes.png'; fig.savefig(p, dpi=400, bbox_inches='tight'); print('wrote', p)
