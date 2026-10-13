#!/usr/bin/env python
"""Figure 3 (a-f), first step: one globe per configuration (HR, LR, CESM1.2) and region, North Atlantic in March and
Southern Ocean in September. Shading: LGM mixed-layer depth in bins, every native cell drawn as its own quadrilateral;
orange: the PI 1000 m mixed-layer contour; black: the LGM 15% sea-ice edge. Rendered offscreen with geovista.
Output: _fig3_globe_<conf>_<region>.png and _fig3_mld_colors.json, read by fig3_convection_freshwater.py.
"""
import json, os, sys
import numpy as np, xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from contourpy import contour_generator
import pyvista as pv
import geovista as gv
from geovista import Transform
from pop_native import cyclic, grid

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
OUT = os.environ.get('GLACIAL_AMOC_FIGS', 'figures')
os.makedirs(OUT, exist_ok=True)
B = DATA + '/casper_vvel'; LS = B + '/lr_suite'
MLD = {('ne120', 'PI'): B + '/MESACLIP_mld/PI_ne120_HMXL_monthly_clim_0041-0060.nc',
       ('ne120', 'LGM'): B + '/MESACLIP_mld/LGM_HMXL_monthly_clim_0001-0011.nc',
       ('f19', 'PI'): LS + '/f19_PI_HMXL_monthly_clim_0485-0504.nc',
       ('f19', 'LGM'): LS + '/f19_LGM_HMXL_monthly_clim_0981-1000.nc',
       ('c12', 'PI'): LS + '/c12_PI_HMXL_monthly_clim_0881-0900.nc',
       ('c12', 'LGM'): LS + '/c12_LGM_HMXL_monthly_clim_0881-0900.nc'}
GRID = {('f19', 'PI'): LS + '/f19_PI_REGION_MASK.nc', ('f19', 'LGM'): LS + '/f19_LGM_REGION_MASK.nc',
        ('c12', 'PI'): LS + '/c12_PI_REGION_MASK.nc', ('c12', 'LGM'): LS + '/c12_LGM_REGION_MASK.nc'}
ICE = {'ne120': B + '/ne120_lgm_reduce/ne120_LGM_aice_hi_monthly_clim_0001-0015.nc',
       'f19': LS + '/f19_LGM_aice_monthly_clim_0981-1000.nc',
       'c12': LS + '/c12_LGM_aice_monthly_clim_0881-0900.nc'}
# month index, view centre (lon, lat), cells kept, parallel scale (half height of the square frame, sphere radius 1),
# camera up vector. North: 36-84N at the central meridian. South: the pole in the centre, 0E up and 90E to the right
# (the orientation of the south polar panels of Figure 2), 47S at mid-edge.
VIEW = {'north': (2, -30.0, 60.0, lambda la: la > -5.0, 0.40, (0.0, 0.0, 1.0)),
        'south': (8, 0.0, -90.0, lambda la: la < 5.0, 0.68, (1.0, 0.0, 0.0))}
LEV = [0, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000]          # m; the last bin also holds > 4000 m
COLORS = (['#ffffff'] + [to_hex(c) for c in plt.get_cmap('Purples')(np.linspace(0.12, 0.42, 5))]      # 100 m to 1000 m
          + [to_hex(c) for c in plt.get_cmap('Purples')(np.linspace(0.72, 1.0, 4))])                    # 1000 m and deeper
assert len(COLORS) == len(LEV) - 1
LAND = '#b9bdc3'
C_ICE, C_PI = '#000000', '#e8710a'
with open(OUT + '/_fig3_mld_colors.json', 'w') as fh:
    json.dump({'levels': LEV, 'colors': COLORS, 'land': LAND, 'ice': C_ICE, 'pi': C_PI}, fh)


def clean(a): a = np.asarray(a, 'f8'); return np.where(np.abs(a) < 1e10, a, np.nan)


def load(key, mi):
    ds = xr.open_dataset(MLD[key]); m = clean(ds['HMXL_monthly'].values[mi])
    if 'TLONG' in ds: lon, lat = ds['TLONG'].values, ds['TLAT'].values
    else:
        g = xr.open_dataset(GRID[key]); lon, lat = g['TLONG'].values, g['TLAT'].values; g.close()
    ds.close(); assert 300.0 < np.nanmax(m) < 7000.0, '%s: MLD is not metres' % (key,)
    return clean(lon), clean(lat), m


def load_ice(conf, mi):
    ds = xr.open_dataset(ICE[conf]); a = clean(ds['aice_monthly'].values[mi])
    lon = (ds['TLON'] if 'TLON' in ds else ds['tlon']).values; lat = (ds['TLAT'] if 'TLAT' in ds else ds['tlat']).values
    ds.close(); assert np.nanmax(a) > 2.0, conf
    return clean(lon), clean(lat), a


def lines_on_sphere(lon, lat, f, level, lift):
    cl, ca, cf = cyclic(lon, lat, f)
    segs = contour_generator(cl, ca, np.ma.masked_invalid(cf)).lines(level)
    pts, cells, n0 = [], [], 0
    for s in segs:
        s = s[np.all(np.isfinite(s), axis=1)]
        if len(s) < 2: continue
        lo, la = np.radians(s[:, 0]), np.radians(s[:, 1])
        pts.append(lift * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        cells.append(np.r_[len(s), np.arange(n0, n0 + len(s))]); n0 += len(s)
    if not pts: return None
    poly = pv.PolyData(np.vstack(pts)); poly.lines = np.hstack(cells).astype(np.int64); return poly


def dir_xyz(lon, lat, r):
    lo, la = np.radians(lon), np.radians(lat)
    return (r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la))


CMAP = ListedColormap(COLORS)
for conf in ('ne120', 'f19', 'c12'):
    for view, (mi, clon, clat, keep, scale, up) in VIEW.items():
        lon, lat, mL = load((conf, 'LGM'), mi); _, latP, mP = load((conf, 'PI'), mi)
        assert mP.shape == mL.shape, conf
        mP = np.where(np.isfinite(mL), mP, np.nan)
        ilon, ilat, ice = load_ice(conf, mi); assert ice.shape == mL.shape, conf
        ice = np.where(np.isfinite(mL), ice, np.nan)
        grid(latP, load((conf, 'PI'), mi)[0]); grid(ilat, ilon)      # every file checked against the grid
        lon, lat, xv, yv = grid(lat, lon, want_corners=True)          # full centres and corners
        sel = keep(lat)
        # bin index + 0.5 on a listed colour map with clim (0, nbins): each cell gets exactly its bin's colour
        idx = np.where(np.isfinite(mL), np.digitize(mL, LEV[1:-1]) + 0.5, np.nan)
        mesh = Transform.from_unstructured(xv[sel], yv[sel], data=idx[sel])
        print('%s %s: %d cells drawn, LGM MLD max %.0f m' % (conf, view, int(sel.sum()), np.nanmax(mL[sel])), flush=True)
        p = gv.GeoPlotter(off_screen=True, window_size=(1400, 1400))
        # a land-coloured sphere under the mesh: the grids have no cells over their poles (the gx1v6 pole inside
        # Greenland, and south of 78.5-79S under Antarctica), which would otherwise show as holes
        p.add_mesh(pv.Sphere(radius=0.997, theta_resolution=720, phi_resolution=360), color=LAND, lighting=False)
        p.add_mesh(mesh, cmap=CMAP, clim=(0, len(COLORS)), nan_color=LAND, show_scalar_bar=False, lighting=False)
        for fld, lev, col, lw, lift in ((ice, 15.0, C_ICE, 6, 1.004), (mP, 1000.0, C_PI, 6, 1.006)):
            ln = lines_on_sphere(lon, lat, fld, lev, lift)
            if ln is not None: p.add_mesh(ln, color=col, line_width=lw, lighting=False)
        p.set_background('white')
        p.enable_parallel_projection()
        p.camera.focal_point = (0.0, 0.0, 0.0); p.camera.position = dir_xyz(clon, clat, 5.0); p.camera.up = up
        p.camera.parallel_scale = scale; p.camera.clipping_range = (0.5, 10.0)
        p.camera_set = True                                            # no automatic reset on the first render
        out = '%s/_fig3_globe_%s_%s.png' % (OUT, conf, view)
        p.screenshot(out, transparent_background=True); p.close()
        print('  wrote', out, os.path.getsize(out), flush=True)
print('DONE', flush=True)
