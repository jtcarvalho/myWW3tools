#!/usr/bin/env python3
"""
WW3 Sub-grid Obstruction Calculator

Computes realistic sub-grid obstruction grids (sx, sy) for WW3 from
a land/sea mask (and optionally a bathymetry file for fractional blocking).

Background
----------
WW3 uses two 2D arrays sx and sy (values 0–100 %) to represent the fraction
of each cell face that is blocked by land.  A value of 0 means fully open,
100 means fully closed.  These control the propagation of wave energy across
narrow passages (straits, islands, etc.).

The MATLAB gridgen computes these by intersecting coastal polygon segments
with cell-face lines.  This script uses a simpler but physically consistent
approach:

  Method A — mask-based (default, fast)
  ──────────────────────────────────────
  For each pair of adjacent ocean cells, the shared face obstruction is
  estimated from the fraction of the *surrounding* cells that are land.
  Specifically, for the E-W face between cell (j,k) and (j+1,k):

      sx(k,j) = fraction of the 4 cells in the 2×2 block centred on
                the shared face that are classified as land.

  This captures partial blockage at the sub-grid scale and matches the
  "shadow zone" concept in the MATLAB algorithm.  If both neighbouring
  cells are ocean and all surrounding cells are ocean, sx=0.
  If one of the two sharing cells is land, sx=1 (fully blocked — this
  face would normally not be written, but WW3 handles it correctly).

  Method B — bathymetry-based (requires .dep file, more accurate)
  ───────────────────────────────────────────────────────────────
  Along each shared cell face, the fraction of sub-grid depth values
  that are below the water level (zdep) is used as the openness factor,
  so:
      sx(k,j) = 1 − (fraction of sub-grid points along face that are ocean)

  This requires the high-resolution source bathymetry file and is best
  for grids that were generated at a coarser resolution than the underlying
  bathymetry data.

In both cases:
  - Obstruction is set to 0 next to dry cells (to prevent spurious swell
    attenuation near the coast), matching the MATLAB gridgen behaviour.
  - Output: rows 0..NY-1 = sx (E-W face), rows NY..2NY-1 = sy (N-S face).
  - Values are rounded to integers in 0–100.

Usage:
    python computeObstrWW3.py <prefix.meta>
    python computeObstrWW3.py config_obstr.yaml

Config keys (YAML) — all optional except source.prefix:
    source:
      directory: .             # where .mask (and optionally .dep) live
      prefix: mygrid           # file prefix
      dep_file: null           # path to source bathymetry NetCDF (method B)
      zdep: -0.10              # depth threshold (metres)
    output:
      directory: .             # where to write the new .obs file
      prefix: mygrid           # output prefix (default = source prefix)
    method: mask               # 'mask' (default) or 'bathy'
"""

import numpy as np
import os
import sys

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ============================================================================
# Meta-file parser  (same as interpWW3.py)
# ============================================================================

def parse_meta(meta_path):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path) as f:
        lines = [l for l in f if not l.strip().startswith('$') and l.strip()]
    try:
        nums1 = list(map(int,   lines[1].split()))
        nx, ny = nums1[0], nums1[1]
        nums2 = list(map(float, lines[2].split()))
        sx_arc, sy_arc, scale = nums2[0], nums2[1], nums2[2]
        dx = sx_arc / scale
        dy = sy_arc / scale
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
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def cfg_get(cfg, key_path, default=None):
    keys = key_path.split('.')
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


# ============================================================================
# File readers
# ============================================================================

def read_mask(path, nx, ny):
    data = np.loadtxt(path, dtype=int)
    if data.shape != (ny, nx):
        raise ValueError(
            f"mask shape {data.shape} ≠ ({ny},{nx})")
    return data


def read_dep(path, nx, ny):
    data = np.loadtxt(path, dtype=np.int64)
    if data.shape != (ny, nx):
        raise ValueError(
            f"dep shape {data.shape} ≠ ({ny},{nx})")
    return data


# ============================================================================
# Method A — mask-based obstruction
# ============================================================================

def compute_obstr_from_mask(mask):
    """
    Estimate sx, sy (0.0–1.0) from the binary land/sea mask.

    For each shared E-W face between columns j and j+1 at row k:
      Consider the 2×2 block of cells: (k-1,j),(k,j),(k-1,j+1),(k,j+1)
      sx(k,j) = fraction of those cells that are land (mask==0).

    Similarly for N-S faces (sy).

    After computing the raw fractions, we apply two corrections:
      1. If either sharing cell is already fully land → sx/sy = 1 (irrelevant
         but kept consistent).
      2. If either sharing cell has a dry neighbour on the other side →
         sx/sy = 0  (prevent spurious swell attenuation near coast).
    """
    ny, nx = mask.shape
    ocean = (mask != 0).astype(float)   # 1=ocean, 0=land
    sx = np.zeros((ny, nx), dtype=float)
    sy = np.zeros((ny, nx), dtype=float)

    # -- sx: E-W face between (k, j) and (k, j+1) -------------------------
    # For interior faces j = 0 .. nx-2:
    for j in range(nx - 1):
        jj = j + 1
        # 2×2 block rows: k-1, k  — clamp at 0 and ny-1
        k_lo = np.maximum(np.arange(ny) - 1, 0)
        k_hi = np.minimum(np.arange(ny),     ny - 1)
        # land fractions from the 4 neighbours of the shared face
        land_frac = (
              (1 - ocean[k_lo, j])
            + (1 - ocean[k_hi, j])
            + (1 - ocean[k_lo, jj])
            + (1 - ocean[k_hi, jj])
        ) / 4.0
        sx[:, j] = land_frac

    # Zero-out next to dry cells (no spurious attenuation) — x direction
    dry = (mask == 0)
    sx[:, :-1][dry[:, 1:]] = 0.0   # right neighbour is land
    sx[:, 1:][dry[:, :-1]] = 0.0   # left  neighbour is land
    # Only relevant for ocean cells — land cells get 0 anyway
    sx[mask == 0] = 0.0

    # -- sy: N-S face between (k, j) and (k+1, j) -------------------------
    for k in range(ny - 1):
        kk = k + 1
        j_lo = np.maximum(np.arange(nx) - 1, 0)
        j_hi = np.minimum(np.arange(nx),     nx - 1)
        land_frac = (
              (1 - ocean[k,  j_lo])
            + (1 - ocean[k,  j_hi])
            + (1 - ocean[kk, j_lo])
            + (1 - ocean[kk, j_hi])
        ) / 4.0
        sy[k, :] = land_frac

    # Zero-out next to dry cells — y direction
    sy[:-1, :][dry[1:, :]] = 0.0
    sy[1:,  :][dry[:-1,:]] = 0.0
    sy[mask == 0] = 0.0

    return sx, sy


# ============================================================================
# Method B — bathymetry-based obstruction (higher accuracy)
# ============================================================================

def compute_obstr_from_bathy(dep_mm, mask, zdep_mm=-100):
    """
    Estimate sx, sy using sub-grid depth information.

    dep_mm : (ny, nx) integer array in millimetres (from .dep file).
    zdep_mm: depth threshold in mm (default -100 mm = -0.10 m).

    For each interior E-W face between (k,j) and (k,j+1):
      Take the two depth values and the two depth values at (k,j-1) and
      (k,j+2) as a 4-point cross-section along the face direction.
      The fraction of those that are LAND (depth >= zdep_mm) is sx(k,j).

    This is a simplified approximation; for true sub-grid accuracy a
    high-resolution bathymetry file should be passed instead of .dep values.
    """
    ny, nx = dep_mm.shape
    sx = np.zeros((ny, nx), dtype=float)
    sy = np.zeros((ny, nx), dtype=float)

    # -- sx faces --
    for j in range(nx - 1):
        jj = j + 1
        j0 = max(j - 1, 0)
        j2 = min(jj + 1, nx - 1)
        land_frac = (
              (dep_mm[:, j]  >= zdep_mm).astype(float)
            + (dep_mm[:, jj] >= zdep_mm).astype(float)
            + (dep_mm[:, j0] >= zdep_mm).astype(float)
            + (dep_mm[:, j2] >= zdep_mm).astype(float)
        ) / 4.0
        sx[:, j] = land_frac

    # -- sy faces --
    for k in range(ny - 1):
        kk = k + 1
        k0 = max(k - 1, 0)
        k2 = min(kk + 1, ny - 1)
        land_frac = (
              (dep_mm[k,  :] >= zdep_mm).astype(float)
            + (dep_mm[kk, :] >= zdep_mm).astype(float)
            + (dep_mm[k0, :] >= zdep_mm).astype(float)
            + (dep_mm[k2, :] >= zdep_mm).astype(float)
        ) / 4.0
        sy[k, :] = land_frac

    # Zero-out next to dry cells
    dry = (mask == 0)
    sx[:, :-1][dry[:, 1:]] = 0.0
    sx[:, 1:][dry[:, :-1]] = 0.0
    sx[mask == 0] = 0.0
    sy[:-1, :][dry[1:, :]] = 0.0
    sy[1:,  :][dry[:-1,:]] = 0.0
    sy[mask == 0] = 0.0

    return sx, sy


# ============================================================================
# File writer
# ============================================================================

def write_obs(path, sx, sy, nx, ny):
    """
    Write .obs file: rows 0..NY-1 = sx (E-W), rows NY..2NY-1 = sy (N-S).
    Values are clipped to [0, 100] and written as 3-character integers.
    """
    obs = np.vstack([sx, sy])
    obs_int = np.clip(np.round(obs * 100).astype(int), 0, 100)
    with open(path, 'w') as f:
        for j in range(2 * ny):
            f.write(''.join(f'{obs_int[j, i]:3d}' for i in range(nx)) + '\n')
    return obs_int


# ============================================================================
# Main
# ============================================================================

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg is None or arg in ('--help', '-h'):
        print(__doc__)
        sys.exit(0 if arg else 1)

    # ── Config ────────────────────────────────────────────────────────
    if arg.endswith(('.yaml', '.yml')):
        cfg = load_yaml(arg)
        src_dir    = cfg_get(cfg, 'source.directory', '.')
        src_prefix = cfg_get(cfg, 'source.prefix')
        dep_file   = cfg_get(cfg, 'source.dep_file', None)
        zdep       = float(cfg_get(cfg, 'source.zdep', -0.10))
        dst_dir    = cfg_get(cfg, 'output.directory', src_dir)
        dst_prefix = cfg_get(cfg, 'output.prefix', src_prefix)
        method     = cfg_get(cfg, 'method', 'mask').lower()
        if src_prefix is None:
            print("ERROR: source.prefix required in config.")
            sys.exit(1)
        src_meta = os.path.join(src_dir, f'{src_prefix}.meta')
    elif arg.endswith('.meta'):
        src_meta   = arg
        src_dir    = os.path.dirname(os.path.abspath(arg))
        src_prefix = os.path.basename(arg)[:-5]
        dep_file   = None
        zdep       = -0.10
        dst_dir    = src_dir
        dst_prefix = src_prefix
        method     = 'mask'
    else:
        print(f"ERROR: unrecognised argument '{arg}'.")
        sys.exit(1)

    # ── Parse meta ─────────────────────────────────────────────────────
    geo = parse_meta(src_meta)
    nx, ny = geo['nx'], geo['ny']
    dx, dy = geo['dx'], geo['dy']
    lon1, lat1 = geo['lon1'], geo['lat1']

    print("\n" + "=" * 60)
    print(" WW3 OBSTRUCTION CALCULATOR")
    print("=" * 60)
    print(f"\n  Grid : {nx} x {ny}  (dx={dx:.4f}°, dy={dy:.4f}°)")
    print(f"  Method : {method}")

    # ── Read mask ──────────────────────────────────────────────────────
    mask_path = os.path.join(src_dir, f'{src_prefix}.mask')
    if not os.path.exists(mask_path):
        print(f"ERROR: mask file not found: {mask_path}")
        sys.exit(1)
    print(f"\n  Reading mask  : {mask_path}")
    mask = read_mask(mask_path, nx, ny)
    n_ocean = int((mask != 0).sum())
    print(f"  Ocean cells   : {n_ocean} / {nx*ny} "
          f"({100*n_ocean/(nx*ny):.1f}%)")

    # ── Compute obstructions ───────────────────────────────────────────
    if method == 'bathy':
        dep_path = os.path.join(src_dir, f'{src_prefix}.dep')
        if not os.path.exists(dep_path):
            print(f"WARNING: dep file not found at '{dep_path}'. "
                  "Falling back to mask method.")
            method = 'mask'
        else:
            print(f"  Reading dep   : {dep_path}")
            dep_mm = read_dep(dep_path, nx, ny)
            zdep_mm = int(zdep * 1000)
            print(f"  Computing obstruction (bathy method, zdep={zdep} m) …",
                  end=' ', flush=True)
            sx, sy = compute_obstr_from_bathy(dep_mm, mask, zdep_mm)
            print("done")

    if method == 'mask':
        print(f"  Computing obstruction (mask method) …",
              end=' ', flush=True)
        sx, sy = compute_obstr_from_mask(mask)
        print("done")

    # ── Statistics ────────────────────────────────────────────────────
    sx_nonzero = (sx > 0).sum()
    sy_nonzero = (sy > 0).sum()
    if sx_nonzero + sy_nonzero > 0:
        sx_mean = sx[sx > 0].mean() * 100 if sx_nonzero else 0.0
        sy_mean = sy[sy > 0].mean() * 100 if sy_nonzero else 0.0
        print(f"  sx: {sx_nonzero} non-zero faces  "
              f"(mean={sx_mean:.1f}% where blocked)")
        print(f"  sy: {sy_nonzero} non-zero faces  "
              f"(mean={sy_mean:.1f}% where blocked)")
    else:
        print("  No obstructions computed (all-ocean or all-land grid?)")

    # ── Write output ───────────────────────────────────────────────────
    os.makedirs(dst_dir, exist_ok=True)
    obs_path = os.path.join(dst_dir, f'{dst_prefix}.obs')
    print(f"\n  Writing : {obs_path} …", end=' ', flush=True)
    obs_int = write_obs(obs_path, sx, sy, nx, ny)
    print("done")

    pct_blocked = 100 * (obs_int > 0).sum() / obs_int.size
    print(f"  Blocked faces: {(obs_int > 0).sum()} / {obs_int.size} "
          f"({pct_blocked:.1f}%)")

    # ── Plot obstructions ──────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Sub-grid obstructions — {dst_prefix}  ({method} method)',
                 fontsize=11)

    sx_pct = obs_int[:ny,  :].astype(float)
    sy_pct = obs_int[ny:,  :].astype(float)

    # mask zero values so land/open faces are white
    sx_plot = np.ma.masked_equal(sx_pct, 0)
    sy_plot = np.ma.masked_equal(sy_pct, 0)

    cmap = plt.cm.hot_r
    cmap.set_bad('white')
    norm = mcolors.Normalize(vmin=1, vmax=100)

    im0 = axes[0].pcolormesh(sx_plot, cmap=cmap, norm=norm)
    axes[0].set_title('sx  (E-W face blockage %)')
    axes[0].set_xlabel('i (lon index)')
    axes[0].set_ylabel('j (lat index)')
    plt.colorbar(im0, ax=axes[0], label='blockage (%)')

    im1 = axes[1].pcolormesh(sy_plot, cmap=cmap, norm=norm)
    axes[1].set_title('sy  (N-S face blockage %)')
    axes[1].set_xlabel('i (lon index)')
    axes[1].set_ylabel('j (lat index)')
    plt.colorbar(im1, ax=axes[1], label='blockage (%)')

    plt.tight_layout()
    fig_path = os.path.join(dst_dir, f'{dst_prefix}_obstr.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved  : {fig_path}")

    print("\n" + "=" * 60)
    print(" DONE")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
