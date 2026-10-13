"""Cartoon of the freshwater decomposition at 34S used in Figure 3 (panel g). Pure drawing, no data: an oblique block
of the South Atlantic cut along 34S, with (1) the overturning term (zonal-mean flow times zonal-mean salinity), (2) the
gyre term (east-west contrasts of the time-mean flow and salinity) and (3) the eddy term (the rest).
"""
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
import matplotlib.patheffects as pe

WATER_TOP, WATER_BOT = np.array([0.80, 0.91, 0.96]), np.array([0.40, 0.58, 0.74])
SAND, SAND_EDGE = '#e8d8b0', '#b59b62'
C_OV, C_GY, C_ED = '#e07b39', '#3a8f6b', '#8e44ad'
FS = 7.0
HALO = [pe.Stroke(linewidth=4.2, foreground='white'), pe.Normal()]


def badge(ax, x, y, n, col, r=0.44, fs=FS):
    ax.add_patch(Circle((x, y), r, fc=col, ec='white', lw=0.9, zorder=30))
    ax.text(x, y, str(n), ha='center', va='center_baseline', color='white', fontsize=fs, weight='bold', zorder=31)


def arrow(ax, p0, p1, col, lw=2.4, ms=10, rad=0.0, z=20, halo=False):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=col, lw=lw, zorder=z,
                        connectionstyle='arc3,rad=%.2f' % rad, shrinkA=0, shrinkB=0, capstyle='round')
    if halo: a.set_path_effects([pe.Stroke(linewidth=lw + 2.2, foreground='white'), pe.Normal()])
    ax.add_patch(a); return a


def draw_fw_schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(-0.7, 10); ax.set_aspect('equal'); ax.axis('off')
    x0, x1, yb, yt = 0.3, 6.9, 3.3, 6.0          # front face: the 34S section
    dx, dy = 2.7, 2.1                            # the 'north' direction of the oblique view
    face = Polygon([(x0, yb), (x1, yb), (x1, yt), (x0, yt)], closed=True, fc='none', ec='none'); ax.add_patch(face)
    grad = np.linspace(0, 1, 64)[:, None, None] * (WATER_BOT - WATER_TOP) + WATER_TOP
    im = ax.imshow(np.repeat(grad, 2, axis=1), extent=(x0, x1, yb, yt), origin='upper', zorder=1, interpolation='bicubic')
    im.set_clip_path(face)
    xs = np.linspace(x0, x1, 200)
    floor = (yb + 0.18 + 0.45 * np.exp(-((xs - 3.9) / 0.55) ** 2) + 1.6 * np.exp(-((xs - x0) / 0.34) ** 2)
             + 1.5 * np.exp(-((xs - x1) / 0.30) ** 2))
    ax.add_patch(Polygon(np.c_[np.r_[xs, x1, x0], np.r_[floor, yb, yb]], fc=SAND, ec=SAND_EDGE, lw=0.6, zorder=3))
    ax.add_patch(Polygon([(x0, yt), (x1, yt), (x1 + dx, yt + dy), (x0 + dx, yt + dy)], fc='#e3f2f8', ec='none', zorder=2))
    ax.add_patch(Polygon([(x0, yt), (x0 + 0.5, yt), (x0 + dx + 1.1, yt + dy), (x0 + dx, yt + dy)], fc=SAND,
                         ec=SAND_EDGE, lw=0.6, zorder=4))                     # South America
    ax.add_patch(Polygon([(x1 - 0.35, yt), (x1, yt), (x1 + dx, yt + dy), (x1 + dx - 1.0, yt + dy)], fc=SAND,
                         ec=SAND_EDGE, lw=0.6, zorder=4))                     # Africa
    ax.add_patch(Polygon([(x1, yb), (x1 + dx, yb + dy), (x1 + dx, yt + dy), (x1, yt)], fc='#d9c9a0', ec='none', zorder=2))
    for xx, yy in (([x0, x1, x1, x0, x0], [yb, yb, yt, yt, yb]), ([x0, x0 + dx, x1 + dx, x1], [yt, yt + dy, yt + dy, yt]),
                   ([x1 + dx, x1 + dx, x1], [yt + dy, yb + dy, yb])):
        ax.plot(xx, yy, color='0.25', lw=0.8, zorder=5)
    # north arrow (N), in the free space under the right face
    u = np.array([dx, dy]) / np.hypot(dx, dy)
    n0 = np.array([8.2, 2.75])
    arrow(ax, tuple(n0), tuple(n0 + 0.95 * u), '0.2', lw=1.0, ms=6)
    ax.text(*(n0 + 0.95 * u + np.array([0.12, 0.0])), 'N', fontsize=FS, weight='bold', color='0.15', ha='left', va='bottom')
    # (1) overturning: surface water flows north into the block (top face), deep water leaves it southward through
    # the section (out of the front face, toward the viewer): one colour, one badge each
    arrow(ax, (1.75, yt + 0.12), (1.75 + 0.52 * dx, yt + 0.12 + 0.52 * dy), C_OV, lw=2.8, ms=11, halo=True)
    badge(ax, 1.75 + 0.52 * dx + 0.62, yt + 0.12 + 0.52 * dy + 0.12, 1, C_OV)
    s0 = np.array([4.4, yb + 1.0]); s1 = s0 - 0.78 * np.array([dx, dy])
    ax.add_patch(Circle(tuple(s0), 0.14, fc=C_OV, ec='white', lw=0.8, zorder=22))
    arrow(ax, tuple(s0), tuple(s1), C_OV, lw=2.8, ms=11, z=21, halo=True)
    badge(ax, s1[0] - 0.55, s1[1] - 0.05, 1, C_OV)
    # (2) gyre: anticlockwise on the surface (south along the west, north along the east)
    cx, cy = 5.45, yt + 1.3
    t = np.linspace(0.2 * np.pi, 1.9 * np.pi, 140)
    gx = cx + 0.95 * np.cos(t) + 0.32 * np.sin(t); gy = cy + 0.48 * np.sin(t)
    ax.plot(gx[:-3], gy[:-3], color=C_GY, lw=2.0, zorder=18, solid_capstyle='round')
    arrow(ax, (gx[-6], gy[-6]), (gx[-1], gy[-1]), C_GY, lw=2.0, ms=8, z=19)
    badge(ax, cx, cy, 2, C_GY)
    # (3) eddies: rings leaving the tip of Africa toward the northwest
    for k, (ex, ey, rr) in enumerate(((x1 - 0.8, yt + 0.4, 0.40), (x1 - 1.6, yt + 0.36, 0.28))):
        th = np.linspace(0, 3.2 * np.pi, 110); r = rr * (1 - th / (4.0 * np.pi))
        ax.plot(ex + r * np.cos(th), ey + 0.55 * r * np.sin(th), color=C_ED, lw=1.6, zorder=18 + k)
    badge(ax, x1 + 0.72, yt + 0.72, 3, C_ED)
    # the sum on top, under the panel letter: 'freshwater import = (1) + (2) + (3)'. Each badge is placed relative to
    # the text before it (offset in points), so it cannot slide over the '=' when the layout changes the axes size
    t0 = ax.text(0.3, 8.85, 'freshwater import =', fontsize=FS, weight='bold', ha='left', va='center', color='0.1')
    prev = t0
    for n, col in ((1, C_OV), (2, C_GY), (3, C_ED)):
        prev = ax.annotate(str(n), xy=(1, 0.5), xycoords=prev, xytext=(7.5, 0), textcoords='offset points', ha='left',
                           va='center', fontsize=FS, weight='bold', color='white', zorder=31,
                           bbox=dict(boxstyle='circle,pad=0.42', fc=col, ec='white', lw=0.9))
        if n < 3:
            prev = ax.annotate('+', xy=(1, 0.5), xycoords=prev, xytext=(6.5, 0), textcoords='offset points', ha='left',
                               va='center', fontsize=FS, weight='bold', color='0.1')
    # key under the block
    rows = ((1, C_OV, 'overturning: zonal-mean flow and salinity'), (2, C_GY, 'gyre: their east-west contrasts'),
            (3, C_ED, 'eddies: the rest, resolved at 0.1°'), (None, None, 'and parameterized at 1°'))
    y = 1.55
    for n, col, tx in rows:
        if n is not None: badge(ax, 0.55, y, n, col, r=0.38)
        ax.text(1.2, y, tx, fontsize=FS, va='center', ha='left', color='0.1')
        y -= 0.78 if n is not None else 0.0
        if n == 3: y += 0.33


if __name__ == '__main__':
    import matplotlib.pyplot as plt, sys
    fig = plt.figure(figsize=(2.6, 2.8)); ax = fig.add_axes([0, 0, 1, 1])
    draw_fw_schematic(ax); fig.savefig(sys.argv[1], dpi=400)
