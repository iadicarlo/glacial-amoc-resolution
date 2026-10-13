#!/usr/bin/env python
"""Helpers to draw POP/CICE fields on their native ocean grids (gx1v6 at 1 degree, tx0.1v2 at 0.1 degree) without
regridding: full cell centres, corners and areas from the CESM coupler mapping files, a cyclic longitude seam for
contouring, row selection by latitude band, and cell polygons for global map projections.
"""
import os
import sys
import numpy as np
import netCDF4 as nc

DATA = os.environ.get('GLACIAL_AMOC_DATA', 'data')
GRIDMAP = {'tx0.1v2': DATA + '/cesm/inputdata/cpl/gridmaps/tx0.1v2/map_tx0.1v2_to_ne120np4_aave_110331.nc',
           'gx1v6': DATA + '/cesm/inputdata/cpl/gridmaps/gx1v6/map_gx1v6_TO_fv0.9x1.25_aave.130322.nc'}
SHAPE = {(2400, 3600): 'tx0.1v2', (384, 320): 'gx1v6'}
GRID_AREA = {}


def _clean(a):
    a = np.asarray(a, dtype='f8'); return np.where(np.abs(a) < 1e10, a, np.nan)


def grid(tlat, tlon, want_corners=False):
    """Full (unmasked) T-cell centres, and optionally corners, of the ocean grid a field lives on, from the CESM
    coupler mapping file (SCRIP source grid). Reduced files can mask TLAT/TLONG over land, and glacial land differs
    from modern land, so the file's own coordinates cannot be used as they are; they are only checked: wherever
    the file defines them, they must equal the grid centres to 0.01 degree, or the script stops."""
    g = SHAPE.get(tuple(np.shape(tlat)))
    if g is None: sys.exit('STOP: no known ocean grid has shape %s' % (np.shape(tlat),))
    d = nc.Dataset(GRIDMAP[g]); ni, nj = [int(v) for v in d.variables['src_grid_dims'][:]]
    # units from the file: these SCRIP grids store their coordinates in radians
    cv = (lambda a: np.degrees(a)) if 'rad' in getattr(d.variables['yc_a'], 'units', 'degrees').lower() else (lambda a: a)
    yc = cv(np.array(d.variables['yc_a'][:])).reshape(nj, ni); xc = cv(np.array(d.variables['xc_a'][:])).reshape(nj, ni) % 360.0
    if want_corners:
        yv = cv(np.array(d.variables['yv_a'][:])).reshape(nj, ni, -1); xv = cv(np.array(d.variables['xv_a'][:])).reshape(nj, ni, -1) % 360.0
    area = np.array(d.variables['area_a'][:]).reshape(nj, ni) * 6.371e6 ** 2      # steradians -> m2
    d.close()
    la, lo = _clean(tlat), _clean(tlon) % 360.0; ok = np.isfinite(la) & np.isfinite(lo)
    # row 0 is excluded: on tx0.1v2 the coupler grid puts it at 84.2S and the model files at 78.5S;
    # it is the southern edge row under the Antarctic ice shelves, land in every file, so it never carries ocean.
    ok[0, :] = False
    dl = np.max(np.abs(la[ok] - yc[ok])); dn = np.max(np.abs((lo[ok] - xc[ok] + 180.0) % 360.0 - 180.0))
    if not (dl < 0.01 and dn < 0.01):
        sys.exit('STOP: %s centres differ from the file coordinates by %.3g / %.3g degree' % (g, dl, dn))
    GRID_AREA[g] = area
    return (xc, yc, xv, yv) if want_corners else (xc, yc)


def area(tlat):
    """Cell areas (m2) of the grid of shape tlat.shape, from the coupler file's area_a (call grid() first)."""
    return GRID_AREA[SHAPE[tuple(np.shape(tlat))]]


def cyclic(tlon, tlat, *fields, lon_start=180.0):
    lon, lat = _clean(tlon) % 360.0, _clean(tlat)
    j0 = int(np.nanargmin(np.abs(np.nanmean(lat, axis=1))))              # an equatorial row: monotone longitudes
    i0 = int(np.nanargmin(np.abs(((lon[j0] - lon_start + 180.0) % 360.0) - 180.0)))
    lon, lat = np.roll(lon, -i0, axis=1), np.roll(lat, -i0, axis=1)
    out = [np.roll(_clean(f), -i0, axis=-1) for f in fields]
    ok = np.isfinite(lon)
    lon = np.where(ok, lon, 0.0)
    lon = np.rad2deg(np.unwrap(np.deg2rad(lon), axis=1))
    lon -= 360.0 * np.floor((lon[:, :1] - (lon_start - 180.0)) / 360.0)  # first column of every row in [start-180, start+180)
    lon = np.where(ok, lon, np.nan)
    lon = np.hstack([lon, lon[:, :1] + 360.0]); lat = np.hstack([lat, lat[:, :1]])
    lon[:, 0] -= 1e-8
    out = [np.concatenate([f, f[..., :1]], axis=-1) for f in out]
    return (lon, lat) + tuple(out)


def rows(lat, *arrays, south=None, north=None):
    lo, hi = np.nanmin(lat, axis=1), np.nanmax(lat, axis=1)
    keep = np.ones(lat.shape[0], bool)
    if south is not None: keep &= hi >= south
    if north is not None: keep &= lo <= north
    return (lat[keep],) + tuple(a[..., keep, :] for a in arrays)


def map_cells(xv, yv, sel, lon0, proj):
    """Cell polygons for a global map centred on lon0, every cell drawn from its own corners (no regridding).
    Each cell's corners are made continuous, the cell is moved to the map's frame [lon0-180, lon0+180), and the few
    cells that straddle the map's edge meridian are split there (shapely), so no polygon spans the map. Corners are
    projected here in one vectorized call (proj.transform_points), so the collections are added in map coordinates.
    Returns (quads (N,4,2), their index into the selected cells, extra polygons (list), their indices)."""
    import cartopy.crs as ccrs
    from shapely.geometry import Polygon as SPoly, box
    from shapely.affinity import translate
    X, Y = xv[sel].astype('f8'), yv[sel].astype('f8')
    ref = X[:, :1]; X = np.where(X - ref > 180.0, X - 360.0, np.where(X - ref < -180.0, X + 360.0, X))
    X = X - 360.0 * np.floor((X[:, :1] - (lon0 - 180.0)) / 360.0)
    lo, hi = lon0 - 180.0, lon0 + 180.0
    eps = 1e-6                                   # a corner on the edge meridian also counts
    bad = (X.max(axis=1) >= hi - eps) | (X.min(axis=1) < lo + eps)
    keep = np.where(~bad)[0]
    pc = ccrs.PlateCarree()
    q = proj.transform_points(pc, X[keep].ravel(), Y[keep].ravel())[:, :2].reshape(keep.size, -1, 2)
    extra, eidx = [], []
    frame = box(lo + 1e-6, -90.0, hi - 1e-6, 90.0)
    for i in np.where(bad)[0]:
        P = SPoly(np.c_[X[i], Y[i]])
        if not P.is_valid: P = P.buffer(0)
        for off in (0.0, -360.0, 360.0):
            R = translate(P, xoff=off).intersection(frame)
            for g in (getattr(R, 'geoms', [R]) if not R.is_empty else []):
                if g.geom_type == 'Polygon' and g.area > 0:
                    c = np.asarray(g.exterior.coords)[:-1]
                    extra.append(proj.transform_points(pc, c[:, 0], c[:, 1])[:, :2]); eidx.append(i)
    ok = np.all(np.isfinite(q), axis=(1, 2))
    return q[ok], keep[ok], extra, np.asarray(eidx, int)
