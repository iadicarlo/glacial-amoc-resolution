#!/usr/bin/env python
"""Figure 3: (a-f) the globes of fig3a_globes.py (run it first) with the convective area deeper than 1000 m
under each globe; (g) a cartoon of the freshwater decomposition (fw_schematic.py); (h) glacial change of each term of
the freshwater import across 34S and (i) the eddy term itself, with 5-95% moving-block bootstrap intervals.
Inputs: the globes, convective-area and freshwater tables, the HR freshwater sections. Output: fig3_convection_freshwater.png.
"""
import csv, json, os, sys
import numpy as np
import matplotlib.image as mpimg
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from netCDF4 import Dataset
from PIL import Image
import ultraplot as uplt
from fw_schematic import draw_fw_schematic

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
HR = DATA + '/fw_budget/{s}/Ocean/Freshwater_transport_section_34S.nc'
CONF = [('ne120', 'HR (0.1°)'), ('f19', 'LR (1°)'), ('c12', 'CESM1.2 (1°)')]
COL = {'ne120': '#1b3f6e', 'f19': '#2e7d4f', 'c12': '#7b3294'}
SHORT = {'ne120': 'HR', 'f19': 'LR', 'c12': 'CESM1.2'}
ORDER = ['FW_OV', 'FW_AZ', 'FW_NONMEAN', 'FW_transport']
LBL = {'FW_OV': 'overturning', 'FW_AZ': 'gyre', 'FW_NONMEAN': 'eddy', 'FW_transport': 'total'}
NB, BLOCK, SEED = 400, 2, 20260811
rng = np.random.default_rng(SEED)
with open(OUT + '/_fig3_mld_colors.json') as fh:
    MC = json.load(fh)

# ---------------- convective areas, gated
AREA = {}
for f in ('metrics_convective_area_v3.csv', 'metrics_convective_area_cesm.csv'):
    with open('%s/derived/%s' % (DATA, f)) as fh:
        for r in csv.DictReader(fh):
            AREA[(r['domain'], r['state'], float(r['thr_m']))] = float(r['area_own_1e12m2'])
for key, want, tol in ((('north', 'ne120 PI', 1000.0), 1.123, 0.002), (('south', 'ne120 LGM', 1000.0), 5.244, 0.002),
                       (('north', 'f19 PI', 1000.0), 0.626, 0.002), (('north', 'f19 LGM', 1000.0), 0.319, 0.002),
                       (('south', 'f19 LGM', 1000.0), 2.176, 0.002),
                       (('north', 'c12 PI', 1000.0), 0.96, 0.005), (('north', 'c12 LGM', 1000.0), 0.06, 0.005)):
    if abs(AREA[key] - want) >= tol: sys.exit('STOP: convective area %s %.4f, expected %.3f' % (key, AREA[key], want))
print('[gate] convective areas reproduce the expected values', flush=True)


# ---------------- freshwater at 34S: HR from the sections, bootstrapped here; LR and CESM1.2 from the table
def load(path):
    fh = Dataset(path, 'r')
    d = {v: np.asarray(fh.variables[v][:], dtype=float) for v in fh.variables if v.startswith('FW_')}
    fh.close()
    d['FW_NONMEAN'] = d['FW_EDDY'] + d.get('FW_BOLUS', 0.0)
    return d


def boot(x):      # moving-block bootstrap of the mean
    n = len(x); Lb = min(max(BLOCK, 1), n)
    starts = np.arange(0, n - Lb + 1); k = int(np.ceil(n / Lb))
    out = np.empty(NB)
    for j in range(NB):
        pick = rng.integers(0, len(starts), size=k)
        idx = (starts[pick][:, None] + np.arange(Lb)[None, :]).ravel()[:n]
        out[j] = float(np.mean(x[idx]))
    return float(np.mean(x)), out


D = {('ne120', s): load(HR.format(s=s)) for s in ('LGM', 'PI')}
CH, ED = {}, {}
for res in ('ne120',):
    for t in ORDER:
        ml, bl = boot(D[(res, 'LGM')][t]); mp, bp = boot(D[(res, 'PI')][t]); d = bl - bp
        CH[(res, t)] = (ml - mp, np.percentile(d, 5), np.percentile(d, 95))
c, lo, hi = CH[('ne120', 'FW_NONMEAN')]
if not (abs(c - 0.075) < 0.002 and abs(lo - 0.065) < 0.002 and abs(hi - 0.085) < 0.002):
    sys.exit('STOP: 0.1 degree eddy change %+.3f [%+.3f,%+.3f] does not reproduce the expected value' % (c, lo, hi))
for res in ('ne120',):
    for s in ('PI', 'LGM'):
        mm, bb = boot(D[(res, s)]['FW_NONMEAN']); ED[(res, s)] = (mm, np.percentile(bb, 5), np.percentile(bb, 95))
TERM = {'overturning': 'FW_OV', 'gyre': 'FW_AZ', 'eddy': 'FW_NONMEAN', 'total': 'FW_transport'}
with open(DATA + '/derived/metrics_fw34S_cesm.csv') as fh:
    for r in csv.DictReader(fh):
        if r['config'] not in ('f19', 'c12'): continue
        v = (float(r['value']), float(r['p5']), float(r['p95']))
        if r['state'] == 'change': CH[(r['config'], TERM[r['term']])] = v
        else: ED[(r['config'], r['state'])] = v
assert abs(CH[('f19', 'FW_transport')][0] - 0.055) < 0.0005 and abs(CH[('f19', 'FW_NONMEAN')][0] + 0.004) < 0.0005
assert abs(CH[('c12', 'FW_transport')][0] - 0.031) < 0.0005 and abs(CH[('c12', 'FW_NONMEAN')][0] + 0.007) < 0.0005
print('[gate] 0.1 degree eddy change and f19 / CESM1.2 total and eddy reproduce the expected values', flush=True)
for k in CONF:
    print('  %-5s change: ' % k[0] + ', '.join('%s %+.3f [%+.3f,%+.3f]' % ((LBL[t],) + tuple(CH[(k[0], t)])) for t in ORDER)
          + ' | eddy PI %+.3f LGM %+.3f' % (ED[(k[0], 'PI')][0], ED[(k[0], 'LGM')][0]), flush=True)

uplt.rc.update({'font.size': 7, 'title.size': 7.5, 'abc.size': 8, 'label.size': 7, 'tick.labelsize': 6.5,
                'legend.fontsize': 7})

# ---------------- part A: the globes
figA, axs = uplt.subplots([[1, 2, 3], [4, 5, 6], [0, 7, 0]], figwidth=7.0, refaspect=1.0, hratios=(1, 1, 0.1),
                          wspace=0.8, hspace=(3.2, 2.2), share=False)     # 7: a blank slot under (e) for the line legend
fmt = lambda v: '0' if v == 0 else ('%.3f' % v if v < 0.01 else '%.2f' % v)
for ri, (view, dom, lab) in enumerate((('north', 'north', 'North Atlantic\nMarch'), ('south', 'south', 'Southern Ocean\nSeptember'))):
    for ci, (k, name) in enumerate(CONF):
        ax = axs[ri * 3 + ci]
        ax.imshow(mpimg.imread('%s/_fig3_globe_%s_%s.png' % (OUT, k, view)), interpolation='lanczos')
        ax.format(xticks=[], yticks=[], xloc='neither', yloc='neither', grid=False)
        for sp in ax.spines.values(): sp.set_visible(False)
        pi, lg = AREA[(dom, k + ' PI', 1000.0)], AREA[(dom, k + ' LGM', 1000.0)]
        ax.text(0.5, -0.02, 'PI %s  →  LGM %s' % (fmt(pi), fmt(lg)), transform=ax.transAxes,
                ha='center', va='top', fontsize=7, color='0.1')
        if ri == 0: ax.format(title=name, titleweight='bold', titlesize=8)
        if ci == 0: ax.text(-0.05, 0.5, lab, transform=ax.transAxes, rotation=90, ha='right', va='center',
                            multialignment='center', fontsize=8, weight='bold')
axs[:6].format(abc='(a)', abcloc='l')
lev = MC['levels']; cmap = ListedColormap(MC['colors']); cmap.set_over(MC['colors'][-1])
sm = ScalarMappable(norm=BoundaryNorm(lev, cmap.N), cmap=cmap)
figA.colorbar(sm, loc='r', rows=(1, 2), label='LGM mixed-layer depth (m)', length=0.8, width=0.1, extend='max',
              ticks=lev)
hl = [Line2D([0], [0], color=MC['ice'], lw=1.6), Line2D([0], [0], color=MC['pi'], lw=1.6)]
axs[6].axis('off')
axs[6].legend(hl, ['LGM 15% sea-ice edge', 'pre-industrial 1000 m mixed-layer contour'], loc='center', ncols=2, frame=False,
              handlelength=2.0, columnspacing=1.6)
pA = OUT + '/_fig3_partA.png'; figA.savefig(pA, dpi=400, bbox_inches='tight', pad_inches=0.03)

# ---------------- part B: cartoon, change by term, the eddy term
figB, bx = uplt.subplots(ncols=3, figsize=(7.0, 3.05), wratios=(2.75, 1.95, 1.3), wspace=(4.6, 5.4), share=False)
draw_fw_schematic(bx[0])
ax = bx[1]; w = 0.26; x = np.arange(len(ORDER))
for i, (k, name) in enumerate(CONF):
    v = np.array([CH[(k, t)] for t in ORDER])
    Axes.bar(ax, x + (i - 1) * w, v[:, 0], width=w, color=COL[k], edgecolor='k', linewidth=0.4, zorder=3, label=SHORT[k])
    ax.errorbar(x + (i - 1) * w, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt='none', ecolor='k',
                elinewidth=0.7, capsize=1.6, zorder=4)
ax.axhline(0, color='k', lw=0.6, zorder=2)
ax.format(xticks=x, xticklabels=[LBL[t] for t in ORDER], xrotation=35, xlim=(-0.55, 3.55), ylabel='glacial change (Sv)',
          ylim=(-0.14, 0.29), xtickminor=False, xgrid=False, title='import at 34°S, change by term')
ax.legend(loc='ur', ncols=1, frame=False)
ax = bx[2]; w = 0.38; x = np.arange(3)
for j, (s, hatch, a) in enumerate((('PI', '////', 0.35), ('LGM', None, 1.0))):
    for i, (k, _) in enumerate(CONF):
        m, lo, hi = ED[(k, s)]
        xx = x[i] + (j - 0.5) * w          # the two bars of one configuration share an edge
        Axes.bar(ax, xx, m, width=w, color=COL[k], alpha=a, hatch=hatch, edgecolor='k', linewidth=0.4, zorder=3)
        ax.errorbar(xx, m, yerr=[[m - lo], [hi - m]], fmt='none', ecolor='k', elinewidth=0.7, capsize=1.6, zorder=4)
ax.format(xticks=x, xticklabels=['HR', 'LR', 'CESM1.2'], xrotation=35, xlim=(-0.6, 2.6),
          ylabel='eddy import (Sv)', xtickminor=False, xgrid=False, ylim=(0, 0.165), title='the eddy term itself')
ax.legend([Patch(fc='0.85', ec='k', lw=0.4, hatch='////'), Patch(fc='0.35', ec='k', lw=0.4)],
          ['PI', 'LGM'], loc='ur', ncols=1, frame=False)
for ax, t in zip(bx, ('(g)', '(h)', '(i)')):     # letters continue those of part A
    ax.text(0.02, 0.985, t, transform=ax.transAxes, ha='left', va='top', fontsize=8, weight='bold', zorder=40)
pB = OUT + '/_fig3_partB.png'; figB.savefig(pB, dpi=400, bbox_inches='tight', pad_inches=0.03)

# ---------------- stack A over B, left-aligned, one white strip between
A, Bm = Image.open(pA).convert('RGB'), Image.open(pB).convert('RGB')
W = max(A.width, Bm.width); gap = int(0.10 * 400)
out = Image.new('RGB', (W, A.height + gap + Bm.height), 'white')
out.paste(A, (0, 0)); out.paste(Bm, (0, A.height + gap))
p = OUT + '/fig3_convection_freshwater.png'; out.save(p, dpi=(400, 400)); print('wrote', p, out.size)
