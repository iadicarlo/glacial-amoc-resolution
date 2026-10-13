"""Check the Python version and the packages listed in requirements.txt. Prints OK or what is missing."""
import sys
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError

REQUIRED = {'numpy': '2.4.3', 'scipy': '1.17.1', 'matplotlib': '3.10.8', 'ultraplot': '2.2.0', 'cartopy': '0.25.0',
            'netCDF4': '1.7.4', 'xarray': '2026.2.0', 'shapely': '2.1.2', 'contourpy': '1.3.3', 'pillow': '12.1.1'}
GLOBES = {'geovista': '0.5.3', 'pyvista': '0.48.4'}   # Figure 3 globes only
MODULE = {'pillow': 'PIL'}


def num(v):
    parts = []
    for p in v.split('.')[:3]:
        d = ''
        for c in p:
            if not c.isdigit():
                break
            d += c
        parts.append(int(d or 0))
    return tuple(parts)


def check(pkgs):
    bad = []
    for pkg, need in pkgs.items():
        try:
            mod = import_module(MODULE.get(pkg, pkg))
        except ImportError:
            bad.append('%s: not installed, %s or newer needed' % (pkg, need))
            continue
        try:
            have = version(pkg)
        except PackageNotFoundError:
            have = None
        have = have or getattr(mod, '__version__', '0')
        if num(have) < num(need):
            bad.append('%s: %s found, %s or newer needed' % (pkg, have, need))
    return bad


bad = [] if sys.version_info >= (3, 12) else ['python: %d.%d found, 3.12 or newer needed' % sys.version_info[:2]]
bad += check(REQUIRED)
for line in check(GLOBES):
    print('note, Figure 3 globes only: ' + line)
if bad:
    print('\n'.join(bad))
    sys.exit(1)
print('OK')
