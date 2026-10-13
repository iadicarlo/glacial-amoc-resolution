#!/usr/bin/env python
"""Figure S6: surface water-mass transformation F against surface density class sigma0 for HR, LR and CESM1.2,
Southern Ocean south of 50S and North Atlantic north of 45N, LGM and PI, and LGM minus PI (Walin 1982; Large & Nurser
2001; buoyancy flux after Cronin & Sprintall 2001). The curves are precomputed (curves_wmt_all.npz) and checked
against the peaks in metrics_si_tests_cesm.csv. Output: figS6_water_mass_transformation.png.
"""
import csv, os, sys
import numpy as np, ultraplot as uplt

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
d = np.load(DATA + '/derived/curves_wmt_all.npz'); CEN = d['CEN']
CONF = [('ne120', 'HR (0.1°)', '#1b3f6e'), ('f19', 'LR (1°)', '#2e7d4f'), ('c12', 'CESM1.2 (1°)', '#7b3294')]
REG = [('SO', 'Southern Ocean (south of 50°S)', 'SO (<50S)'), ('NA', 'North Atlantic (north of 45°N)', 'NA (>45N)')]
want = {}
with open(DATA + '/derived/metrics_si_tests_cesm.csv') as fh:
    for r in csv.DictReader(fh):
        if r['section'] == 'WMT': want[(r['config'], r['state'], r['quantity'])] = float(r['value'])
bad = []
for rk, _, rname in REG:
    for c, _, _ in CONF:
        for st in ('LGM', 'PI'):
            F = d['%s_%s_%s' % (rk, c, st)]
            pk, sg = float(F.max()), float(CEN[int(np.argmax(F))])
            wf, ws = want[(c, st, rname + ' peak F')], want[(c, st, rname + ' peak sigma0')]
            if abs(pk - wf) > 0.01 or abs(sg - ws) > 0.001: bad.append((rk, c, st, pk, wf, sg, ws))
            print('%s %-5s %-3s peak %.2f Sv at %.2f (CSV %.2f at %.2f)' % (rk, c, st, pk, sg, wf, ws), flush=True)
if bad: sys.exit('STOP: plotted peaks do not reproduce the CSV: %s' % bad)
print('[check] peaks reproduce metrics_si_tests_cesm.csv: PASSED', flush=True)

uplt.rc.update({'font.size': 7, 'title.size': 7.5, 'abc.size': 8, 'label.size': 7, 'tick.labelsize': 6.5, 'legend.fontsize': 6.5})
fig, axs = uplt.subplots(nrows=2, ncols=2, figwidth=6.2, refaspect=1.35, share=False, wspace=4.6, hspace=7.0)
for i, (rk, title, _) in enumerate(REG):
    ax, axd = axs[i], axs[2 + i]
    for c, name, col in CONF:
        L, P = d['%s_%s_LGM' % (rk, c)], d['%s_%s_PI' % (rk, c)]
        ax.plot(CEN, L, color=col, lw=1.3, label=name + ' LGM')
        ax.plot(CEN, P, color=col, lw=1.0, ls='--', label=name + ' PI')
        axd.plot(CEN, L - P, color=col, lw=1.3, label=name)
    for a_ in (ax, axd): a_.axhline(0, color='0.5', lw=0.5)
    ax.format(title=title, xlabel=r'surface $\sigma_0$ (kg m$^{-3}$)', ylabel='transformation F (Sv)', xlim=(24, 30.5))
    axd.format(title=title + ': LGM − PI', xlabel=r'surface $\sigma_0$ (kg m$^{-3}$)', ylabel=r'$\Delta$F (Sv)', xlim=(24, 30.5))
axs[0].legend(loc='lr', ncols=1, frame=False, handlelength=1.3, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
axs[2].legend(loc='ll', ncols=1, frame=False)
axs.format(abc='(a)', abcloc='ul')
p = OUT + '/figS6_water_mass_transformation.png'; fig.savefig(p, dpi=400, bbox_inches='tight'); print('wrote', p)
