#!/usr/bin/env python3
"""
Extract boundary-condition points from a WW3 .mask2 file.

Reads {area}.mask2 and {area}.meta (same directory) to obtain the
lon/lat of every grid cell flagged as a boundary point (value == 2).
Writes a text file listing those coordinates in WW3 point-input format.

Usage:
    python prepBC2WW3.py <mask2_file> [--id PREFIX]

Example:
    python prepBC2WW3.py ../output/rio_grid/rio.mask2
    python prepBC2WW3.py ../output/rio_grid/rio.mask2 --id bnd
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def parse_meta(meta_path):
    """Parse a WW3 .meta file and return grid parameters."""
    with open(meta_path, 'r') as f:
        lines = [l.strip() for l in f if not l.strip().startswith('$') and l.strip()]

    # Line 0: 'RECT' T 'NONE'
    # Line 1: NX NY
    # Line 2: SX SY scale
    # Line 3: x0 y0 coord_scale
    parts1 = lines[1].split()
    nx, ny = int(parts1[0]), int(parts1[1])

    parts2 = lines[2].split()
    sx, sy, scale_xy = float(parts2[0]), float(parts2[1]), float(parts2[2])

    parts3 = lines[3].split()
    x0, y0 = float(parts3[0]), float(parts3[1])
    coord_scale = float(parts3[2])

    dx = sx / scale_xy
    dy = sy / scale_xy
    x0 = x0 / coord_scale
    y0 = y0 / coord_scale

    return nx, ny, dx, dy, x0, y0


def main():
    parser = argparse.ArgumentParser(description='Extract WW3 boundary-condition points.')
    parser.add_argument('mask2_file', help='Path to the .mask2 file')
    parser.add_argument('--id', default='bnd', dest='prefix',
                        help="Label prefix for each point (default: 'bnd')")
    args = parser.parse_args()

    mask2_path = args.mask2_file
    prefix = args.prefix

    if not os.path.isfile(mask2_path):
        print(f"Error: file not found: {mask2_path}")
        sys.exit(1)

    # Derive meta file path (same directory, same base name)
    base_dir = os.path.dirname(mask2_path)
    area = os.path.basename(mask2_path).replace('.mask2', '')
    meta_path = os.path.join(base_dir, f'{area}.meta')

    if not os.path.isfile(meta_path):
        print(f"Error: meta file not found: {meta_path}")
        sys.exit(1)

    # Parse grid metadata
    nx, ny, dx, dy, x0, y0 = parse_meta(meta_path)
    print(f"Grid: NX={nx}, NY={ny}, dx={dx}, dy={dy}")
    print(f"Origin: x0={x0}, y0={y0}")

    # Build coordinate arrays
    lon = x0 + np.arange(nx) * dx
    lat = y0 + np.arange(ny) * dy

    # Read mask2 (IDLA=1: first line = bottom row → row 0 = lat_min)
    mask = np.loadtxt(mask2_path, dtype=int)
    print(f"Mask shape: {mask.shape}")

    if mask.shape != (ny, nx):
        print(f"Warning: mask shape {mask.shape} does not match meta NX={nx}, NY={ny}")

    # Find boundary points (value == 2)
    rows, cols = np.where(mask == 2)
    n_bc = len(rows)
    print(f"Found {n_bc} boundary points")

    if n_bc == 0:
        print("No boundary points found. Nothing to write.")
        sys.exit(0)

    # Write output
    out_path = os.path.join(base_dir, f'{area}.list')
    bc_lons = lon[cols]
    bc_lats = lat[rows]
    with open(out_path, 'w') as f:
        for k in range(n_bc):
            label = f'{prefix}_{k+1:05d}'
            f.write(f"  {bc_lons[k]:10.3f}{bc_lats[k]:10.3f} '{label}'\n")

    print(f"Output saved: {out_path}")

    # ---- Plot boundary points with cartopy coastline ----
    pad = 1.0  # extra degrees around the domain
    lon_min, lon_max = lon[0] - pad, lon[-1] + pad
    lat_min, lat_max = lat[0] - pad, lat[-1] + pad

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='bisque')
    ax.add_feature(cfeature.OCEAN, facecolor='lightcyan')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='k')
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle='--')

    ax.plot(bc_lons, bc_lats, 'r.', markersize=3, transform=ccrs.PlateCarree(),
            label=f'Boundary points ({n_bc})')

    # Draw grid domain rectangle
    dom_lons = [lon[0], lon[-1], lon[-1], lon[0], lon[0]]
    dom_lats = [lat[0], lat[0], lat[-1], lat[-1], lat[0]]
    ax.plot(dom_lons, dom_lats, 'b-', linewidth=1.2, transform=ccrs.PlateCarree(),
            label='Grid domain')

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.legend(loc='upper right')
    ax.set_title(f'{area} — WW3 boundary points')
    plt.tight_layout()

    fig_path = os.path.join(base_dir, f'{area}_bc_points.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


if __name__ == '__main__':
    main()

