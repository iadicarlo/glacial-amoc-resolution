# glacial-amoc-resolution

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23361570.svg)](https://doi.org/10.5281/zenodo.23361570)

Scripts that make the figures of the paper "Resolution may hold the key to a weaker and shallower glacial Atlantic
overturning" (Abdelkader Di Carlo et al., submitted to Geophysical Research Letters): Figures 1 to 4 of the main text
and Figures S1 to S7 of the Supporting Information.

## 1. Set up Python

You need Python 3.12 or newer.

```bash
git clone https://github.com/iadicarlo/glacial-amoc-resolution.git
cd glacial-amoc-resolution
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_env.py
```

`check_env.py` prints `OK` when Python and all packages are recent enough. Otherwise it lists what to install or
update.

## 2. Get the data

The scripts read model output reduced to the analysis windows and a few small precomputed tables. These will be
published in a data repository after the paper is submitted, and the link will be added here. The original
simulations are listed in the Open Research section of the paper.

Put the data in a folder called `data` inside this repository, or tell the scripts where it is:

```bash
export GLACIAL_AMOC_DATA=/path/to/data
```

The figures are written to a folder called `figures`, or to the folder you set:

```bash
export GLACIAL_AMOC_FIGS=/path/to/output
```

## 3. Make the figures

Run the script of the figure you want from the repository folder, for example:

```bash
python scripts/fig1_overturning.py
```

| Figure | Script (in `scripts/`) | Output |
|---|---|---|
| 1 | `fig1_overturning.py` | `fig1_overturning.png` |
| 2 | `fig2_sea_ice.py` | `fig2_sea_ice.png` |
| 3 | `fig3a_globes.py`, then `fig3_convection_freshwater.py` | `fig3_convection_freshwater.png` |
| 4 | `fig4a_map_underlays.py`, then `fig4_schematic.py` | `fig4_schematic.png` |
| S1 | `figS1_timeseries.py` | `figS1_timeseries.png` |
| S2 | `figS2_changes.py` | `figS2_changes.png` |
| S3 | `figS3_overturning.py` | `figS3_overturning.png` |
| S4 | `figS4_sst.py` | `figS4_sst.png` |
| S5 | `figS5_isotopes.py` | `figS5_isotopes.png` |
| S6 | `figS6_water_mass_transformation.py` | `figS6_water_mass_transformation.png` |
| S7 | `figS7_stratification.py` | `figS7_stratification.png` |

Notes:

- Figures 3 and 4 take two steps. The first script writes intermediate images to the output folder, and the second
  script reads them.
- The Figure 3 globes are drawn offscreen with geovista and pyvista. On a computer without a screen this needs an
  offscreen OpenGL library (OSMesa or EGL).
- The layout of Figure 1 can shift by one pixel depending on Python's hash seed. Run it with `PYTHONHASHSEED=2` to get
  the published file exactly.
- `f19_common.py`, `pop_native.py` and `fw_schematic.py` are helper modules used by the scripts.

## License

MIT, see `LICENSE`. If you use these scripts, please cite the paper and this repository: Abdelkader Di Carlo, I.
(2026). Figure scripts for "Resolution may hold the key to a weaker and shallower glacial Atlantic overturning"
(v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23361571
