#!/usr/bin/env python
"""Figure S7: profiles of the squared buoyancy frequency N2 (TEOS-10) for HR, LR and CESM1.2, LGM and PI: Southern
Ocean (65-50S) and Weddell Sea (72-58S) in September, Atlantic 45-65N and 20-40N annual means. The profiles are
precomputed (strat_*_profiles.npz for HR, curves_*_n2_lr.npz for the 1 degree runs). Output: figS7_stratification.png.
"""
import csv, os, sys
import numpy as np, ultraplot as uplt
DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
F = DATA + '/derived'
so = np.load(F + '/strat_so_profiles.npz'); at = np.load(F + '/strat_atl_profiles.npz')
lso = np.load(F + '/curves_so_n2_lr.npz'); lat_ = np.load(F + '/curves_atl_n2_lr.npz')
CONF = [('ne120', 'HR (0.1°)', '#1b3f6e'), ('f19', 'LR (1°)', '#2e7d4f'), ('c12', 'CESM1.2 (1°)', '#7b3294')]


def so_prof(c, st, key):
    if c == 'ne120':
        return so['ne120_%s_%s' % (st, key)] * 1e5, so['ne120_%s_z' % st]
    return lso['%s_%s_%s' % (c, st, key)], lso['%s_%s_z' % (c, st)]


def atl_prof(c, st, band):
    if c == 'ne120':
        return at['0.1_%s_%s' % (st, band)], at['0.1_%s_%s_z' % (st, band)]
    return lat_['%s_%s_%s' % (c, st, band)], lat_['%s_%s_%s_z' % (c, st, band)]


# physics checks on what is drawn
for c, _, _ in CONF:
    for st in ('LGM', 'PI'):
        for key in ('SO', 'Weddell'):
            v, z = so_prof(c, st, key); m = (z > 50) & (z < 2000) & np.isfinite(v)
            assert m.any() and 0.0 < np.nanmedian(v[m]) < 20.0, (c, st, key, 'SO N2 not in 1e-5 s-2')
        for band in ('45_65N', '20_40N'):
            v, z = atl_prof(c, st, band); m = (z > 300) & np.isfinite(v)
            assert m.any() and -5.0 < np.nanmedian(v[m]) < 60.0, (c, st, band, 'Atlantic N2 not in 1e-6 s-2')
print('[check] units and ranges of every profile: PASSED', flush=True)
step = {}
with open(F + '/metrics_si_tests_cesm.csv') as fh:
    for r in csv.DictReader(fh):
        if r['section'] == 'Atlantic' and r['quantity'].endswith(' deep'):
            step[(r['config'], r['state'], r['quantity'].split()[0])] = float(r['value'])
for c in ('f19', 'c12'):
    for b in ('45-65N', '20-40N'):
        print('deep sigma2 step %s %s: LGM %.3f PI %.3f change %+.3f' % (c, b, step[(c, 'LGM', b)], step[(c, 'PI', b)],
                                                                         step[(c, 'LGM', b)] - step[(c, 'PI', b)]), flush=True)

uplt.rc.update({'font.size': 7, 'title.size': 7.5, 'abc.size': 8, 'label.size': 7, 'tick.labelsize': 6.5, 'legend.fontsize': 6.3})
fig, axs = uplt.subplots(nrows=2, ncols=2, figwidth=5.6, refaspect=0.85, share=False, wspace=3.6, hspace=7.5)
for i, (key, title) in enumerate((('SO', 'Southern Ocean 65 to 50°S, September'), ('Weddell', 'Weddell Sea 72 to 58°S, September'))):
    ax = axs[i]
    for c, name, col in CONF:
        for st, ls, lw in (('LGM', '-', 1.5), ('PI', '--', 1.1)):
            v, z = so_prof(c, st, key)
            ax.plot(v, z, color=col, ls=ls, lw=lw, label='%s %s' % (name, st))
    ax.format(title=title, xlabel=r'$N^2$ ($10^{-5}$ s$^{-2}$)', ylabel='depth (m)', ylim=(2000, 0), xscale='log', xlim=(0.04, 6.0))
    if i == 0: ax.legend(loc='lr', ncols=1, frame=False)
for i, (band, title) in enumerate((('45_65N', 'Atlantic 45 to 65°N, annual'), ('20_40N', 'Atlantic 20 to 40°N, annual'))):
    ax = axs[2 + i]
    ax.axhspan(1500, 3500, color='0.93', zorder=0)
    for c, name, col in CONF:
        for st, ls, lw in (('LGM', '-', 1.5), ('PI', '--', 1.1)):
            v, z = atl_prof(c, st, band)
            ax.plot(v, z, color=col, ls=ls, lw=lw)
    ax.format(title=title, xlabel=r'$N^2$ ($10^{-6}$ s$^{-2}$)', ylabel='depth (m)', ylim=(4200, 50), xlim=(-2, 30))
axs.format(abc='(a)', abcloc='ul')
p = OUT + '/figS7_stratification.png'; fig.savefig(p, dpi=400, bbox_inches='tight'); print('wrote', p)
