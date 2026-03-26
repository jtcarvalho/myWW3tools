#!/usr/bin/env python3
"""
WW3 Grid Interpolator

Reads an existing WW3 grid (dep, mask, obs) and interpolates all three files
to a new resolution. The grid geometry is read from the corresponding .meta
file (or specified via command-line flags / YAML config).

Usage:
    python interpWW3.py <prefix.meta>           # reads geometry from .meta
    python interpWW3.py config_interp.yaml      # reads from YAML config file
    python interpWW3.py --help

Output files (next to the input directory):
    {out_prefix}.dep   — interpolated bathymetry
    {out_prefix}.mask  — interpolated mask  (nearest-neighbour)
    {out_prefix}.obs   — interpolated obstacles (nearest-neighbour, 2*NY rows)

Notes:
    - Bathymetry (.dep) is interpolated with nearest-neighbour to preserve
      integer millimetre values and avoid artificial smoothing.
    - Mask (.mask) always uses nearest-neighbour.
    - Obstacles (.obs) always uses nearest-neighbour; NY is doubled internally.
    - Interpolation is performed with scipy.interpolate.RegularGridInterpolator
      (fast, memory-efficient) instead of griddata.
"""

import numpy as np
import os
import sys
import re

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from scipy.interpolate import RegularGridInterpolator


# ============================================================================
# Meta-file parser
# ============================================================================

def parse_meta(meta_path):
    """
    Extract NX, NY, dx, dy (in degrees), lon1, lat1 from a WW3 .meta file.

    The meta file stores increments in arc-minutes with a divisor of 60,
    so true degrees = value / 60.

    Returns dict with keys: nx, ny, dx, dy, lon1, lat1
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path) as f:
        lines = [l for l in f if not l.strip().startswith('$') and l.strip()]

    # Line 0: grid type  →  'RECT'  T  'NONE'
    # Line 1: NX  NY
    # Line 2: SX   SY   SCALE
    # Line 3: LON1  LAT1  1.00

    try:
        nums1 = list(map(int,   lines[1].split()))
        nx, ny = nums1[0], nums1[1]

        nums2 = list(map(float, lines[2].split()))
        sx, sy, scale = nums2[0], nums2[1], nums2[2]
        dx = sx / scale
        dy = sy / scale

        nums3 = list(map(float, lines[3].split()))
        lon1, lat1 = nums3[0], nums3[1]
    except Exception as e:
        raise ValueError(f"Could not parse meta file '{meta_path}': {e}")

    return dict(nx=nx, ny=ny, dx=dx, dy=dy, lon1=lon1, lat1=lat1)


# ============================================================================
# Config helpers
# ============================================================================

def load_yaml(path):
    if not HAS_YAML:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def get(cfg, key_path, default=None):
    keys = key_path.split('.')
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


# ============================================================================
# Core interpolation helpers
# ============================================================================

def _make_coords(lon1, lat1, dx, dy, nx, ny):
    lons = lon1 + np.arange(nx) * dx
    lats = lat1 + np.arange(ny) * dy
    return lons, lats


def _nearest_interp_2d(src_lons, src_lats, src_data, dst_lons, dst_lats):
    """Nearest-neighbour 2D interpolation via RegularGridInterpolator."""
    rgi = RegularGridInterpolator(
        (src_lats, src_lons), src_data.astype(float),
        method='nearest', bounds_error=False, fill_value=None
    )
    lon_g, lat_g = np.meshgrid(dst_lons, dst_lats)
    pts = np.column_stack([lat_g.ravel(), lon_g.ravel()])
    return rgi(pts).reshape(len(dst_lats), len(dst_lons))


# ============================================================================
# File I/O  (reuse write_* from genWW3grid_fromBounds convention)
# ============================================================================

LAND_MM = 999999000

def read_dep(path, nx, ny):
    data = np.loadtxt(path, dtype=np.int64)
    if data.shape != (ny, nx):
        raise ValueError(
            f"dep file shape {data.shape} does not match expected ({ny},{nx})")
    return data


def read_mask(path, nx, ny):
    data = np.loadtxt(path, dtype=int)
    if data.shape != (ny, nx):
        raise ValueError(
            f"mask file shape {data.shape} does not match expected ({ny},{nx})")
    return data


def read_obs(path, nx, ny):
    """obs file has 2*ny rows."""
    data = np.loadtxt(path, dtype=int)
    if data.shape != (2 * ny, nx):
        raise ValueError(
            f"obs file shape {data.shape} does not match expected ({2*ny},{nx})")
    return data


def write_dep(path, data_mm, nx, ny):
    data_mm = np.round(data_mm).astype(np.int64)
    data_mm[data_mm > 900_000_000] = LAND_MM
    with open(path, 'w') as f:
        for j in range(ny):
            f.write(''.join(f'{data_mm[j, i]:12d}' for i in range(nx)) + '\n')


def write_mask(path, mask, nx, ny):
    mask = np.round(mask).astype(int)
    with open(path, 'w') as f:
        for j in range(ny):
            f.write(''.join(f'{mask[j, i]:3d}' for i in range(nx)) + '\n')


def write_obs(path, obs, nx, ny):
    obs = np.round(obs).astype(int)
    obs = np.clip(obs, 0, 100)
    with open(path, 'w') as f:
        for j in range(2 * ny):
            f.write(''.join(f'{obs[j, i]:3d}' for i in range(nx)) + '\n')


def write_meta(filepath, nx, ny, dx, dy, lon1, lat1, prefix,
               zdep=-0.10, zmin=2.50, closure='NONE'):
    sx = dx * 60.0
    sy = dy * 60.0
    scale_xy = 60.0
    lon_span = (nx - 1) * dx
    if abs(lon_span - 360.0) < dx:
        closure = 'SMPL'

    META_HEADER = """\
$ Define grid -------------------------------------- $
$ Five records containing :
$  1 Type of grid, coordinate system and type of closure: GSTRG, FLAGLL,
$    CSTRG. Grid closure can only be applied in spherical coordinates.
$      GSTRG  : String indicating type of grid :
$               'RECT'  : rectilinear
$               'CURV'  : curvilinear
$      FLAGLL : Flag to indicate coordinate system :
$               T  : Spherical (lon/lat in degrees)
$               F  : Cartesian (meters)
$      CSTRG  : String indicating the type of grid index space closure :
$               'NONE'  : No closure is applied
$               'SMPL'  : Simple grid closure : Grid is periodic in the
$                         : i-index and wraps at i=NX+1. In other words,
$                         : (NX+1,J) => (1,J). A grid with simple closure
$                         : may be rectilinear or curvilinear.
$               'TRPL'  : Tripole grid closure : Grid is periodic in the
$                         : i-index and wraps at i=NX+1 and has closure at
$                         : j=NY+1. In other words, (NX+1,J<=NY) => (1,J)
$                         : and (I,NY+1) => (MOD(NX-I+1,NX)+1,NY). Tripole
$                         : grid closure requires that NX be even. A grid
$                         : with tripole closure must be curvilinear.
$  2 NX, NY. As the outer grid lines are always defined as land
$    points, the minimum size is 3x3.
$  3 Grid increments SX, SY (degr.or m) and scaling (division) factor.
$    If NX*SX = 360., latitudinal closure is applied.
$  4 Coordinates of (1,1) (degr.) and scaling (division) factor.
$  5 Limiting bottom depth (m) to discriminate between land and sea
$    points, minimum water depth (m) as allowed in model, unit number
$    of file with bottom depths, scale factor for bottom depths (mult.),
$    IDLA, IDFM, format for formatted read, FROM and filename.
$      IDLA : Layout indicator :
$                  1   : Read line-by-line bottom to top.
$                  2   : Like 1, single read statement.
$                  3   : Read line-by-line top to bottom.
$                  4   : Like 3, single read statement.
$      IDFM : format indicator :
$                  1   : Free format.
$                  2   : Fixed format with above format descriptor.
$                  3   : Unformatted.
$      FROM : file type parameter
$             'UNIT' : open file by unit number only.
$             'NAME' : open file by name and assign to unit.
$  If the Unit Numbers in above files is 10 then data is read from this file
$
"""
    with open(filepath, 'w') as f:
        f.write(META_HEADER)
        f.write(f"   'RECT'  T  '{closure}'\n")
        f.write(f"{nx:7d}  {ny:7d}\n")
        f.write(f"{sx:8.2f}   {sy:8.2f}   {scale_xy:.2f}\n")
        f.write(f"{lon1:.4f}         {lat1:.4f}         1.00\n")
        f.write(f"$ Bottom Bathymetry\n")
        f.write(f"{zdep:.2f}   {zmin:.2f}  40  0.001000  1  1 "
                f"'(....)'  NAME  './{prefix}.dep'\n")
        f.write(f"$ Sub-grid information\n")
        f.write(f"50  0.010000  1  1  '(....)'  NAME  './{prefix}.obs'\n")
        f.write(f"$ Mask Information\n")
        f.write(f"60  1  1  '(....)'  NAME  './{prefix}.mask'\n")


# ============================================================================
# Main logic
# ============================================================================

def run_interp(src_meta, src_dir, src_prefix,
               dst_nx, dst_ny, dst_dir, dst_prefix,
               zdep=-0.10, zmin=2.50):
    """
    Perform the actual interpolation from source grid to destination grid.
    """
    geo = parse_meta(src_meta)
    nx1, ny1 = geo['nx'], geo['ny']
    dx1, dy1 = geo['dx'], geo['dy']
    lon1, lat1 = geo['lon1'], geo['lat1']

    # Compute destination dx/dy from same domain extent
    lon_max = lon1 + (nx1 - 1) * dx1
    lat_max = lat1 + (ny1 - 1) * dy1

    dx2 = (lon_max - lon1) / (dst_nx - 1)
    dy2 = (lat_max - lat1) / (dst_ny - 1)

    lons1, lats1 = _make_coords(lon1, lat1, dx1, dy1, nx1, ny1)
    lons2, lats2 = _make_coords(lon1, lat1, dx2, dy2, dst_nx, dst_ny)

    print(f"\n  Source grid  : {nx1} x {ny1}  "
          f"(dx={dx1:.6f}°, dy={dy1:.6f}°)")
    print(f"  Target grid  : {dst_nx} x {dst_ny}  "
          f"(dx={dx2:.6f}°, dy={dy2:.6f}°)")
    print(f"  Domain       : lon [{lon1:.4f}, {lon_max:.4f}]°  "
          f"lat [{lat1:.4f}, {lat_max:.4f}]°")

    os.makedirs(dst_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Bathymetry
    # ------------------------------------------------------------------
    dep_src = os.path.join(src_dir, f'{src_prefix}.dep')
    if os.path.exists(dep_src):
        print(f"\n  Reading  {dep_src} …", end=' ', flush=True)
        dep1 = read_dep(dep_src, nx1, ny1).astype(float)
        dep2 = _nearest_interp_2d(lons1, lats1, dep1, lons2, lats2)
        dst_dep = os.path.join(dst_dir, f'{dst_prefix}.dep')
        print(f"writing {dst_dep} …", end=' ', flush=True)
        write_dep(dst_dep, dep2, dst_nx, dst_ny)
        print("done")
    else:
        print(f"  WARNING: {dep_src} not found — skipping .dep")

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------
    mask_src = os.path.join(src_dir, f'{src_prefix}.mask')
    if os.path.exists(mask_src):
        print(f"  Reading  {mask_src} …", end=' ', flush=True)
        mask1 = read_mask(mask_src, nx1, ny1).astype(float)
        mask2 = _nearest_interp_2d(lons1, lats1, mask1, lons2, lats2)
        mask2 = np.round(mask2).astype(int)
        dst_mask = os.path.join(dst_dir, f'{dst_prefix}.mask')
        print(f"writing {dst_mask} …", end=' ', flush=True)
        write_mask(dst_mask, mask2, dst_nx, dst_ny)
        print("done")
    else:
        print(f"  WARNING: {mask_src} not found — skipping .mask")

    # ------------------------------------------------------------------
    # Obstacles  (2*NY rows — treat as two separate NY-row arrays)
    # ------------------------------------------------------------------
    obs_src = os.path.join(src_dir, f'{src_prefix}.obs')
    if os.path.exists(obs_src):
        print(f"  Reading  {obs_src} …", end=' ', flush=True)
        obs1 = read_obs(obs_src, nx1, ny1).astype(float)
        # Split into E-W and N-S halves
        ew1 = obs1[:ny1, :]
        ns1 = obs1[ny1:, :]
        ew2 = _nearest_interp_2d(lons1, lats1, ew1, lons2, lats2)
        ns2 = _nearest_interp_2d(lons1, lats1, ns1, lons2, lats2)
        obs2 = np.vstack([ew2, ns2])
        dst_obs = os.path.join(dst_dir, f'{dst_prefix}.obs')
        print(f"writing {dst_obs} …", end=' ', flush=True)
        write_obs(dst_obs, obs2, dst_nx, dst_ny)
        print("done")
    else:
        print(f"  WARNING: {obs_src} not found — skipping .obs")

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    dst_meta = os.path.join(dst_dir, f'{dst_prefix}.meta')
    print(f"  Writing  {dst_meta} …", end=' ', flush=True)
    write_meta(dst_meta, dst_nx, dst_ny, dx2, dy2, lon1, lat1,
               dst_prefix, zdep=zdep, zmin=zmin)
    print("done")

    return dst_nx, dst_ny, dx2, dy2


# ============================================================================
# Entry point
# ============================================================================

def _usage():
    print(__doc__)
    sys.exit(0)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        _usage()

    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # ── Determine configuration source ────────────────────────────────
    if arg is None:
        print("ERROR: provide a .meta file or a YAML config file.")
        print("Usage:")
        print("  python interpWW3.py <prefix.meta>")
        print("  python interpWW3.py config_interp.yaml")
        sys.exit(1)

    if arg.endswith('.yaml') or arg.endswith('.yml'):
        if not HAS_YAML:
            print("ERROR: PyYAML is required for YAML configs.")
            print("  pip install pyyaml")
            sys.exit(1)
        cfg = load_yaml(arg)

        src_dir    = get(cfg, 'source.directory', '.')
        src_prefix = get(cfg, 'source.prefix')
        dst_dir    = get(cfg, 'output.directory', 'interp_output')
        dst_prefix = get(cfg, 'output.prefix', src_prefix + '_interp')
        dst_nx     = get(cfg, 'target.nx')
        dst_ny     = get(cfg, 'target.ny')
        zdep       = get(cfg, 'grid.zdep', -0.10)
        zmin       = get(cfg, 'grid.zmin',  2.50)

        if src_prefix is None:
            print("ERROR: source.prefix is required in the YAML config.")
            sys.exit(1)

        src_meta = os.path.join(src_dir, f'{src_prefix}.meta')

    elif arg.endswith('.meta'):
        # Derive everything from the meta filename
        src_meta   = arg
        src_dir    = os.path.dirname(os.path.abspath(arg))
        src_prefix = os.path.basename(arg)[:-5]  # strip .meta
        dst_nx     = None
        dst_ny     = None
        dst_dir    = None
        dst_prefix = None
        zdep       = -0.10
        zmin       =  2.50
    else:
        print(f"ERROR: unrecognised argument '{arg}'.")
        print("Provide a .meta file or a YAML config.")
        sys.exit(1)

    # ── Parse meta for geometry ────────────────────────────────────────
    geo = parse_meta(src_meta)
    nx1, ny1 = geo['nx'], geo['ny']

    # ── Interactive prompt if target size not given ────────────────────
    if dst_nx is None or dst_ny is None:
        print(f"\n  Source grid: {nx1} x {ny1}  "
              f"(dx={geo['dx']:.4f}°, dy={geo['dy']:.4f}°)")
        print("  Enter target grid size (or a resolution multiplier):")
        raw = input("  NX NY  (or just a multiplier, e.g. 2 for 2×): ").split()
        if len(raw) == 1:
            m = float(raw[0])
            dst_nx = int(round((nx1 - 1) * m)) + 1
            dst_ny = int(round((ny1 - 1) * m)) + 1
        elif len(raw) == 2:
            dst_nx, dst_ny = int(raw[0]), int(raw[1])
        else:
            print("ERROR: expected 1 or 2 numbers.")
            sys.exit(1)

    if dst_dir is None:
        dst_dir = os.path.join(src_dir, f'{src_prefix}_x{dst_nx}x{dst_ny}')
    if dst_prefix is None:
        dst_prefix = src_prefix

    # ── Run ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" WW3 GRID INTERPOLATOR")
    print("=" * 60)
    nx2, ny2, dx2, dy2 = run_interp(
        src_meta, src_dir, src_prefix,
        dst_nx, dst_ny, dst_dir, dst_prefix,
        zdep=zdep, zmin=zmin
    )

    print("\n" + "=" * 60)
    print(" INTERPOLATION COMPLETE")
    print("=" * 60)
    print(f"\n  Output directory : {os.path.abspath(dst_dir)}/")
    print(f"  Target grid size : {nx2} x {ny2}  (NX x NY)")
    print(f"  Target spacing   : {dx2:.6f}° x {dy2:.6f}°")
    print()


if __name__ == '__main__':
    main()
