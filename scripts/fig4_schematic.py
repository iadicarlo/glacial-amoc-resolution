#!/usr/bin/env python
"""Figure 4: (a, b) glacial change of the Kageyama et al. (2021) overturning metrics at 30N for PMIP4 as published,
PMIP4 recomputed from each model's own velocities, CESM1.2 and this study, with the proxy-constrained weakening of
Poeppelmeier et al. (2023); (c-f) for LR and HR, the North Atlantic in March (map underlays from fig4a_map_underlays.py,
run it first) and the glacial Atlantic overturning, with schematic arrows and a key. Output: fig4_schematic.png.
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ultraplot as uplt
from contourpy import contour_generator
from matplotlib import patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch, Polygon
from matplotlib.ticker import NullLocator
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
FIGS = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(FIGS, exist_ok=True)
BLOCKS = os.path.join(DATA, 'derived', 'fig4_blocks_data.npz')
PSIPI = os.path.join(FIGS, 'fig4_psi_pi.npz')
NAJ = os.path.join(FIGS, '_fig4_natl.json')
NAPNG = os.path.join(FIGS, '_fig4_natl_%s.png')
CSV = os.path.join(DATA, 'derived', 'metrics_lit_definitions.csv')
OUT = os.path.join(FIGS, 'fig4_schematic.png')

# ----------------------------------------------------------------------------- data for (a, b)
PUB = [('AWIESM1', 2.7, 250), ('AWIESM2', 6.4, 250), ('CESM1.2', -1.4, -240), ('iLOVECLIM ICE-6G', 8.5, 2600),
       ('iLOVECLIM GLAC-1D', 7.9, 2600), ('INM-CM4-8', -2.4, None), ('IPSL-CM5A2', 11.0, 2450),
       ('MIROC-ES2L', 5.9, None), ('MPI-ESM1.2', 4.9, 120), ('UoT-CCSM4', -0.9, -100),
       ('HadCM3 ICE-6G', 4.3, 0), ('HadCM3 GLAC-1D', 4.4, 200), ('HadCM3 PMIP3', 1.5, 100)]   # Kageyama et al. (2021) Fig. 4, read from the figure
ARCH = [('AWI-ESM-1-1-LR', 'AWI-ESM-1-1-LR', 'AWIESM1'), ('CESM2-FV2', 'CESM2-PaleoCalibr', None),
        ('MIROC-ES2L', 'MIROC-ES2L', 'MIROC-ES2L'), ('MPI-ESM1-2-LR', 'MPI-ESM1-2-LR', 'MPI-ESM1.2'),
        ('c12', 'CESM1.2', 'CESM1.2')]
OURS = [('f19', 'LR, 1° f19_g16', '#2e7d4f', '^', 4.2), ('0.1', 'HR, 0.1° ne120_t12', '#1b3f6e', 'o', 3.8)]   # small enough that the whiskers show
PROXY = (-6.3, 1.4, '−6.3 ± 1.4 Sv')
# (b): shoaling of northern-sourced water, 950 +- 180 m (Poeppelmeier et al., 2023), from a dye tracer in Bern3D;
# a different quantity from the NADW extent at 30N, shown as a qualitative guide
PROXY_NSW = (-950.0, 180.0, '−950 ± 180 m')
R = {}
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        R[(r['config'], r['metric'])] = {k: float(r[k]) for k in r if k not in ('config', 'metric')}
for (c, m), want in ((('0.1', 'K21_max'), -7.72), (('0.1', 'K21_extent'), -798.0), (('f19', 'K21_max'), -1.26),
                     (('f19', 'K21_extent'), -298.0), (('c12', 'K21_max'), -0.82)):
    if abs(R[(c, m)]['change'] - want) > (0.006 if m == 'K21_max' else 0.6):
        sys.exit('STOP: %s %s change %.3f, expected %.2f' % (c, m, R[(c, m)]['change'], want))


def nd(c):   # no AABW cell at 30N in either state: extent change not defined
    m = R[(c, 'K21_min_deep')]; return m['LGM'] >= 0.0 or m['PI'] >= 0.0


# ----------------------------------------------------------------------------- style
PAGE_W, PAGE_H = 178.0, 196.0                       # mm
INK = '#2b2b2b'
C1, C01 = '#2e7d4f', '#1b3f6e'
C_WARM, C_PI = '#d6452a', '#e8710a'
PSI_LEVELS = np.arange(-12, 26.1, 2.0)
PSI_NORM = TwoSlopeNorm(vcenter=0.0, vmin=-12, vmax=26)
PSI_CMAP = uplt.Colormap('Balance')
FS_TICK, FS_LAB, FS_TITLE = 6.5, 7.0, 8.0
GATE_CB = {('f19', 'LGM'): 2701, ('f19', 'PI'): 3210, ('ne120', 'LGM'): 2110, ('ne120', 'PI'): 2794}
D, P = np.load(BLOCKS), np.load(PSIPI)
J = json.load(open(NAJ)); XL, YL, LON0, LAT0, MC = J['xlim'], J['ylim'], J['lon0'], J['lat0'], J['colors']
VX, VY = (-0.60, 0.60), (-0.48, 0.72)                 # the window shown, inside the rendered extent


def clean_axes(ax):
    ax.grid(False)
    ax.yaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_locator(NullLocator())
    for s in ('top', 'right'): ax.spines[s].set_visible(False)


def badge(ax, x, y, n, col, fs=FS_LAB, z=40, **kw):
    ax.text(x, y, str(n), ha='center', va='center', fontsize=fs, fontweight='bold', color='white', zorder=z,
            bbox=dict(boxstyle='circle,pad=0.3', facecolor=col, edgecolor='white', lw=0.8), **kw)


def direction(ax, left, right):
    for t, ha, dx in ((left, 'right', -4), (right, 'left', 4)):
        ax.annotate(t, xy=(0, 0), xycoords=('data', 'axes fraction'), xytext=(dx, -15), textcoords='offset points',
                    ha=ha, va='top', fontsize=FS_TICK, color=INK)


def forest(ax, metric, xlim, xlabel, band=None, labels=True):
    ys = {'hdr1': -1.0, 'ens': 0.0, 'hdr2': 1.6, 'hdr3': 8.0}
    for i, (c, _, _) in enumerate(ARCH): ys[c] = 2.5 + i
    for i, (c, _, _, _, _) in enumerate(OURS): ys[c] = 8.9 + i
    if band is not None:
        ax.axvspan(band[0] - band[1], band[0] + band[1], color='#2e8b57', alpha=0.18, lw=0, zorder=0)
        ax.axvline(band[0], color='#2e8b57', lw=0.8, zorder=1)
        off = 0.054 * (xlim[1] - xlim[0])
        ax.text(band[0] - band[1] - off, 4.7, 'proxy-constrained  ' + band[2], ha='center', va='center', rotation=90,
                fontsize=6.3, color='#1e6b42')
    ax.axvline(0, color='0.3', lw=0.7, zorder=1)
    vals = np.array([t[1] if metric == 'K21_max' else t[2] for t in PUB if (t[1] if metric == 'K21_max' else t[2]) is not None], float)
    m, sd = vals.mean(), vals.std(ddof=1)
    ax.scatter(vals, ys['ens'] + np.linspace(-0.22, 0.22, vals.size), s=6, color='0.55', zorder=3, lw=0)
    ax.plot([vals.min(), vals.max()], [ys['ens']] * 2, color='0.35', lw=0.8, zorder=2)
    ax.plot([m - sd, m + sd], [ys['ens']] * 2, color='0.35', lw=3.2, solid_capstyle='butt', zorder=2, alpha=0.6)
    ax.scatter([m], [ys['ens']], s=34, marker='D', color='0.25', zorder=4)
    for c, name, pub in ARCH:
        y = ys[c]
        if metric == 'K21_extent' and nd(c):
            ax.text(0, y, 'n.d.', ha='center', va='center', fontsize=FS_TICK, color='0.35', zorder=3,
                    bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none')); continue
        v = R[(c, metric)]
        ax.errorbar(v['change'], y, xerr=[[v['change'] - v['change_p5']], [v['change_p95'] - v['change']]], fmt='o',
                    color='k', ms=3.8, lw=0.9, capsize=1.8, zorder=4)
        if pub:
            p = [t for t in PUB if t[0] == pub][0][1 if metric == 'K21_max' else 2]
            if p is not None: ax.scatter([p], [y], marker='D', s=16, facecolor='none', edgecolor='0.35', lw=0.7, zorder=3)
    for c, name, col, mk, ms in OURS:
        v = R[(c, metric)]; y = ys[c]
        ax.errorbar(v['change'], y, xerr=[[v['change'] - v['change_p5']], [v['change_p95'] - v['change']]], fmt=mk,
                    color=col, mec=col, ms=ms, lw=1.1, capsize=2.0, zorder=5)
    rows = [('hdr1', 'PMIP4, published'), ('ens', 'ensemble of 13 models'), ('hdr2', 'PMIP4, recomputed here')] + \
           [(c, n) for c, n, _ in ARCH] + [('hdr3', 'iCESM1.3, this study')] + [(c, n) for c, n, _, _, _ in OURS]
    ax.set_yticks([ys[k] for k, _ in rows]); ax.set_yticklabels([t for _, t in rows] if labels else [''] * len(rows))
    if labels:
        for t, (k, _) in zip(ax.get_yticklabels(), rows):
            if k.startswith('hdr'): t.set_fontweight('bold'); t.set_fontsize(FS_LAB); t.set_color('0.1')
            if k == '0.1': t.set_fontweight('bold'); t.set_color(C01)
    ax.tick_params(axis='y', length=0, labelsize=FS_TICK); ax.tick_params(axis='x', labelsize=FS_TICK, length=2)
    ax.set_ylim(10.5, -1.5); ax.set_xlim(xlim); ax.set_xlabel(xlabel, fontsize=FS_LAB, labelpad=13)
    clean_axes(ax)
    return m, sd, vals


def ortho(lon, lat):
    lo, la, p0 = np.radians(np.asarray(lon, float) - LON0), np.radians(np.asarray(lat, float)), np.radians(LAT0)
    return np.cos(la) * np.sin(lo), np.cos(p0) * np.sin(la) - np.sin(p0) * np.cos(la) * np.cos(lo)


def draw_map(ax, k, path, width, sink, bll, col):
    ax.imshow(plt.imread(NAPNG % k), extent=(XL[0], XL[1], YL[0], YL[1]), origin='upper', interpolation='lanczos', zorder=1)
    t = np.linspace(0, 2 * np.pi, 721); ax.plot(np.cos(t), np.sin(t), color='0.55', lw=0.5, zorder=2)
    ax.set_xlim(VX); ax.set_ylim(VY); ax.set_aspect('equal'); ax.axis('off')
    if path is not None:
        x, y = ortho(*np.array(path).T)
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]; ss = np.linspace(0, s[-1], 240)
        cx, cy = CubicSpline(s, x)(ss), CubicSpline(s, y)(ss)
        per_pt = (VX[1] - VX[0]) / (ax.get_position().width * PAGE_W / 25.4 * 72.0)     # data units per point
        hl, hw = max(2.6 * width, 6.0) * per_pt, max(1.5 * width, 3.2) * per_pt            # head length, half width
        s2 = np.r_[0.0, np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))]
        kb = int(np.searchsorted(s2, s2[-1] - hl))                                          # base of the head
        u = np.array([cx[-1] - cx[kb], cy[-1] - cy[kb]]); u /= np.hypot(*u); nrm = np.array([-u[1], u[0]])
        base = np.array([cx[-1], cy[-1]]) - hl * u
        ln = ax.plot(np.r_[cx[:kb], base[0]], np.r_[cy[:kb], base[1]], color=C_WARM, lw=width, solid_capstyle='butt', zorder=6)[0]
        ln.set_path_effects([pe.Stroke(linewidth=width + 1.6, foreground='white'), pe.Normal()])
        ax.add_patch(Polygon([base + hw * nrm, (cx[-1], cy[-1]), base - hw * nrm], closed=True, fc=C_WARM, ec='white',
                             lw=0.8, zorder=7, joinstyle='miter'))
    if sink is not None:
        sx, sy = ortho(*sink)
        ax.plot(sx, sy, marker='o', ms=9, mfc='white', mec='0.1', mew=0.9, zorder=8)
        ax.plot(sx, sy, marker='x', ms=4.8, mec='0.1', mew=1.0, zorder=9)
    bx, by = ortho(*bll); badge(ax, bx, by, 1, col)


def nan_smooth(a, sigma):
    bad = np.isnan(a); num = gaussian_filter(np.where(bad, 0.0, a), sigma); den = gaussian_filter((~bad).astype(float), sigma)
    out = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan); out[bad] = np.nan; return out


def cell_base(lat, z, psi):
    li = np.where((lat >= 30) & (lat <= 45))[0]; p = np.nanmean(psi[:, li], axis=1); k0 = int(np.nanargmax(p))
    for k in range(k0, len(z) - 1):
        if p[k] >= 0 and p[k + 1] < 0:
            f = p[k] / (p[k] - p[k + 1]); return z[k] + f * (z[k + 1] - z[k])
    return np.nan


def streamline(ax, lat, z, psi, level, n_heads, sig, lw=0.7):
    sm = gaussian_filter(np.nan_to_num(psi, nan=0.0), sig)
    lines = [l for l in contour_generator(lat, z, sm).lines(level) if len(l) > 20]
    if not lines: return
    poly = max(lines, key=lambda p: abs(np.trapezoid(p[:, 1], p[:, 0])))
    i_top = int(np.argmin(poly[:, 1])); j = min(i_top + 3, len(poly) - 1)
    if ((poly[j, 0] - poly[i_top, 0]) > 0) != (level > 0): poly = poly[::-1]
    line = ax.plot(poly[:, 0], poly[:, 1], color=INK, lw=lw, zorder=7, solid_capstyle='round')[0]
    line.set_path_effects([pe.Stroke(linewidth=lw + 1.0, foreground='white', alpha=0.7), pe.Normal()])
    X = (poly[:, 0] - poly[:, 0].min()) / 104.0; Y = poly[:, 1] / 5000.0            # arc length in panel fractions
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    for frac in np.linspace(0.12, 0.88, n_heads):
        i = min(max(int(np.searchsorted(s, frac * s[-1])), 2), len(poly) - 3)
        ax.annotate('', xy=(poly[i + 2, 0], poly[i + 2, 1]), xytext=(poly[i - 2, 0], poly[i - 2, 1]), zorder=8,
                    arrowprops=dict(arrowstyle='-|>', mutation_scale=6.5, lw=0.0, color=INK, shrinkA=0, shrinkB=0))


def draw_amoc(ax, k, col, b2, b3, b4):
    plat, pz, psi = D['psilat_' + k], D['psiz_' + k].astype(float), D['psi_' + k].astype(float)
    ql, qz, qsi = P['psilat_' + k], P['psiz_' + k].astype(float), P['psi_' + k].astype(float)
    cbL, cbP = cell_base(plat, pz, psi), cell_base(ql, qz, qsi)
    for (st, cb) in (('LGM', cbL), ('PI', cbP)):
        if abs(cb - GATE_CB[(k, st)]) > 1.0: sys.exit('STOP: %s %s cell base %.1f m, expected %d' % (k, st, cb, GATE_CB[(k, st)]))
    zs = np.r_[0.0, pz]; psi = np.vstack([psi[:1], psi]); qzs = np.r_[0.0, qz]; qsi = np.vstack([qsi[:1], qsi])
    ax.contourf(plat, zs, psi, levels=PSI_LEVELS, cmap=PSI_CMAP, norm=PSI_NORM, extend='both', zorder=1)
    sig = (0.6, max(0.8, 0.5 / np.nanmedian(np.diff(plat)))); sq = (0.6, max(0.8, 0.5 / np.nanmedian(np.diff(ql))))
    zl = lambda la, z, f: np.where((z[:, None] >= 250.0) & (la[None, :] <= 58.0), f, np.nan)   # interface only
    ax.contour(plat, zs, zl(plat, zs, nan_smooth(psi, sig)), levels=[0.0], colors=[INK], linewidths=0.7, zorder=4)
    ax.contour(ql, qzs, zl(ql, qzs, nan_smooth(qsi, sq)), levels=[0.0], colors=[C_PI], linewidths=1.1,
               linestyles=[(0, (3, 1.6))], zorder=5)
    for level, n in ((8.0, 4), (-4.0, 3)): streamline(ax, plat, zs, psi, level, n, sig)
    a = FancyArrowPatch((43.0, cbP), (43.0, cbL), arrowstyle='-|>', mutation_scale=7, color='k', lw=1.0, zorder=9,
                        shrinkA=0, shrinkB=0)
    a.set_path_effects([pe.Stroke(linewidth=2.6, foreground='white'), pe.Normal()]); ax.add_patch(a)
    for n, (x, y) in ((2, b2), (3, b3), (4, b4)): badge(ax, x, y, n, col)
    ax.set_xlim(-34, 70); ax.set_ylim(5000, 0)
    ax.set_xticks([-30, 0, 30, 60]); ax.set_xticklabels(['30°S', '0°', '30°N', '60°N'])
    ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000]); ax.set_yticklabels(['0', '1', '2', '3', '4', '5'])
    ax.set_ylabel('depth (km)', fontsize=FS_LAB, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, length=2, pad=1.5)
    clean_axes(ax)
    for s in ('top', 'right'): ax.spines[s].set_visible(True)
    print('%s: cell base PI %.0f m -> LGM %.0f m [gate passed]' % (k, cbP, cbL))


def key(ax, col, items):
    ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = 0.96
    for n, tx in enumerate(items, 1):
        badge(ax, 0.045, y - 0.035, n, col)
        ax.text(0.13, y, tx, ha='left', va='top', fontsize=FS_LAB, color=INK, linespacing=1.25)
        y -= 0.13 + 0.105 * tx.count('\n')


def main():
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    fx = lambda mm: mm / PAGE_W; fy = lambda mm: mm / PAGE_H
    ax_at = lambda x0, top, w, h: fig.add_axes([fx(x0), fy(PAGE_H - top - h), fx(w), fy(h)])
    # (a, b)
    axa, axb = ax_at(44.0, 9.0, 57.0, 44.0), ax_at(118.0, 9.0, 57.0, 44.0)
    ma = forest(axa, 'K21_max', (-10.5, 12.5), 'LGM − PI, maximum of ψ at 30°N (Sv)', band=PROXY)
    mb = forest(axb, 'K21_extent', (-1500, 2850), 'LGM − PI, NADW extent at 30°N (m)', band=PROXY_NSW, labels=False)
    direction(axa, '← weaker', 'stronger →'); direction(axb, '← shallower', 'deeper →')
    # move (a, b) left so the row labels of (a) start at the left edge of the maps (c, e), 4 mm
    fig.canvas.draw(); rr = fig.canvas.get_renderer()
    lab0 = min(t.get_window_extent(rr).x0 for t in axa.get_yticklabels() if t.get_text()) / fig.dpi * 25.4
    pa = axa.get_position(); left_a = pa.x0 * PAGE_W - (lab0 - 4.0); wa = 60.0
    axa.set_position([left_a / PAGE_W, pa.y0, wa / PAGE_W, pa.height])
    left_b = left_a + wa + 16.0
    axb.set_position([left_b / PAGE_W, pa.y0, (177.0 - left_b) / PAGE_W, pa.height])
    print('(a) row labels moved from %.1f to 4.0 mm; (a) %.1f-%.1f mm, (b) %.1f-177.0 mm' % (lab0, left_a, left_a + wa, left_b))
    for ax, l, t in ((axa, '(a)', 'overturning strength'), (axb, '(b)', 'depth of the northern cell')):
        ax.text(0.0, 1.035, l + '  ' + t, transform=ax.transAxes, fontsize=FS_TITLE, fontweight='bold', va='bottom', ha='left')
    print('PMIP4 published ensemble: strength mean %.2f sd %.2f (n=%d, range %.1f to %.1f); extent mean %.0f sd %.0f (n=%d)'
          % (ma[0], ma[1], ma[2].size, ma[2].min(), ma[2].max(), mb[0], mb[1], mb[2].size))
    # (c-f): per configuration, the North Atlantic face on, the Atlantic overturning next to it, and the key
    ROWS = [('f19', C1, 'Low resolution, 1°', ('(c)', '(d)'), 25.34,
             [(-78, 28), (-73, 34.5), (-62, 39.0), (-48, 41.8), (-36, 43.2), (-27.0, 46.0), (-24.0, 51.0), (-26.8, 56.0), (-31.5, 59.0)], (-34.2, 60.4), (-19.0, 62.5),
             (47.0, 1300.0), (-18.0, 3900.0), (-27.5, 700.0),
             ['Convection shrinks toward\nthe ice edge', 'Northern cell keeps its strength\n(+3 to +13%)',
              'Abyssal cell barely rises\n(160 to 290 m at 30°S)', 'Overturning, not eddies, carries\nthe extra freshwater import']),
            ('ne120', C01, 'High resolution, 0.1°', ('(e)', '(f)'), 11.12,
             [(-78, 28), (-73, 34.5), (-62, 38.6), (-46, 40.3), (-30, 41.2), (-17, 42.3)], None, (-26.0, 61.0),
             (47.0, 1100.0), (-18.0, 3300.0), (-27.5, 700.0),
             ['Sea ice caps the convection sites', 'Northern cell weakens by 40%\nand shoals by 680 m',
              'Abyssal cell rises by 880 m at 30°S', 'Eddies nearly triple their\nfreshwater import at 34°S'])]
    top0 = 74.0; RH = 52.0
    for r, (k, col, title, (lm, la), smax, path, sink, bll, b2, b3, b4, items) in enumerate(ROWS):
        top = top0 + r * RH
        fig.text(fx(4.0), fy(PAGE_H - top), title, fontsize=FS_TITLE + 0.5, fontweight='bold', color=col, va='bottom', ha='left')
        hm = 44.0 * (VY[1] - VY[0]) / (VX[1] - VX[0])
        axm = ax_at(4.0, top + 2.5, 44.0, hm)
        draw_map(axm, k, None, 0.13 * smax, sink, bll, col)   # no surface-flow arrow: its route was schematic, not model output
        axm.text(0.0, 1.0, lm, transform=axm.transAxes, fontsize=FS_TITLE, fontweight='bold', va='top', ha='left', zorder=50,
                 path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
        axs_ = ax_at(62.0, top + 4.5, 55.0, 35.0)
        draw_amoc(axs_, k, col, b2, b3, b4)
        axs_.text(-0.02, 1.03, la, transform=axs_.transAxes, fontsize=FS_TITLE, fontweight='bold', va='bottom', ha='right')
        key(ax_at(122.0, top + 4.5, 55.0, 40.0), col, items)
    # legends and colour bar along the bottom
    yb = PAGE_H - (top0 + 2 * RH) - 1.0
    lax = ax_at(2.0, PAGE_H - yb, 58.0, 16.0); lax.axis('off')
    hd = [Patch(fc=MC['mld5'], ec='none'), Patch(fc=MC['mld10'], ec='none'), Patch(fc='white', ec='0.5', lw=0.5),
          (Line2D([0], [0], marker='o', ls='none', ms=7.5, mfc='white', mec='0.1', mew=0.9),
           Line2D([0], [0], marker='x', ls='none', ms=4.0, mec='0.1', mew=1.0)),
          Line2D([0], [0], color='k', lw=0.8), Line2D([0], [0], color=C_PI, lw=1.1, ls=(0, (3, 1.6)))]
    lax.legend(hd, ['MLD 500 to 1000 m', 'MLD over 1000 m', 'sea ice', 'deep convection', 'LGM ice edge', 'PI 1000 m MLD'],
               loc='upper left', ncol=2, frameon=False, fontsize=FS_TICK, handlelength=1.8, columnspacing=1.2,
               borderaxespad=0.0, labelspacing=0.45)
    cax = ax_at(64.0, PAGE_H - yb + 1.5, 51.0, 2.4)
    cb = fig.colorbar(ScalarMappable(cmap=PSI_CMAP, norm=PSI_NORM), cax=cax, orientation='horizontal', extend='both',
                      ticks=[-12, -8, -4, 0, 4, 8, 12, 16, 20, 24])
    cb.set_label('glacial overturning streamfunction ψ (Sv)', fontsize=FS_LAB, labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK, length=2, width=0.5, pad=1.5); cb.outline.set_linewidth(0.5)
    cb.ax.xaxis.set_minor_locator(NullLocator())
    lx = ax_at(122.0, PAGE_H - yb, 55.0, 12.0); lx.axis('off')
    lx.legend([Line2D([0], [0], color=INK, lw=0.7), Line2D([0], [0], color=C_PI, lw=1.1, ls=(0, (3, 1.6)))],
              ['glacial ψ = 0', 'pre-industrial ψ = 0'], loc='upper left', ncol=1, frameon=False, fontsize=FS_TICK,
              handlelength=1.8, borderaxespad=0.0, labelspacing=0.45)
    fig.savefig(OUT, dpi=400); print('wrote', OUT)


if __name__ == '__main__':
    main()
