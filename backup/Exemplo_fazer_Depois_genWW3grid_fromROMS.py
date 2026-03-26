#!/usr/bin/env python3
"""
ROMS Grid (NetCDF) → WW3 (ASCII) Converter

Converts ROMS ocean modeling grid data in NetCDF format to ASCII format
expected by the Wave Watch III (WW3) wave model.

Usage:
    python genWW3grid_v2.py [config_file.yaml]

If no config file is specified, defaults to 'config.yaml'.

Features:
  - Uses xarray for efficient NetCDF handling
  - Automatic grid dimension calculation based on spacing
  - Bathymetry and mask interpolation
  - Full configuration via YAML file
  - Comprehensive error handling
"""

import numpy as np
import xarray as xr
import os
import sys
from scipy.interpolate import griddata

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("⚠ PyYAML not installed. Install with: pip install pyyaml")

# ============================================================================
# Helper Functions
# ============================================================================

def load_config(config_file):
    """Load configuration from YAML file"""
    if not HAS_YAML:
        return None
    
    if not os.path.exists(config_file):
        print(f"⚠ Configuration file '{config_file}' not found.")
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from '{config_file}'")
        return config
    except Exception as e:
        print(f"✗ Error reading configuration file: {e}")
        return None

def get_config_value(config, key_path, default=None):
    """Get nested dictionary value using dot notation (e.g., 'grid.dx')"""
    if config is None:
        return default
    
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
    
    return value if value is not None else default

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def print_section(title):
    """Print section title"""
    print(f"\n→ {title}")

def calculate_grid_dimensions(lon_range, lat_range, dx, dy):
    """Calculate grid dimensions based on domain and spacing"""
    nx = int(np.ceil(lon_range / dx)) + 1
    ny = int(np.ceil(lat_range / dy)) + 1
    return nx, ny

# ============================================================================
# Main Conversion Process
# ============================================================================

def main():
    # Determine configuration file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'config.yaml'
    
    # Load configuration
    config = load_config(config_file) if HAS_YAML else None
    
    # Get configuration parameters with defaults
    roms_grid = get_config_value(config, 'files.roms_grid', 'grade-roms.nc')
    output_dir = get_config_value(config, 'files.output_dir', 'regional')
    output_prefix = get_config_value(config, 'files.output_prefix', 'repsol_regional')
    
    dx_ww3 = get_config_value(config, 'grid.dx', 0.05)
    dy_ww3 = get_config_value(config, 'grid.dy', 0.05)
    
    min_depth = get_config_value(config, 'bathymetry.min_depth', 0.5)
    bath_output_unit = get_config_value(config, 'bathymetry.output_unit', 'cm')
    bath_interp = get_config_value(config, 'bathymetry.interpolation_method', 'linear')
    bath_fill = get_config_value(config, 'bathymetry.fill_value', 0.1)
    
    mark_boundaries = get_config_value(config, 'mask.mark_boundaries_as_boundary', True)
    boundary_value = get_config_value(config, 'mask.boundary_value', 2)
    
    verbosity = get_config_value(config, 'processing.verbosity', 'normal')
    
    # ========================================================================
    # Read ROMS File (using xarray)
    # ========================================================================
    
    print_header("ROMS → WW3 CONVERTER")
    print_section("Reading ROMS file")
    
    if not os.path.exists(roms_grid):
        print(f"✗ ERROR: File '{roms_grid}' not found!")
        sys.exit(1)
    
    try:
        # Open with xarray - automatically handles NetCDF
        ds = xr.open_dataset(roms_grid)
        
        # Extract variables
        lon_rho = ds['lon_rho'].values
        lat_rho = ds['lat_rho'].values
        h = ds['h'].values
        mask_rho = ds['mask_rho'].values
        
        ds.close()
        
        print(f"✓ ROMS file read: {roms_grid}")
        print(f"  Dimensions: {h.shape[1]} × {h.shape[0]} points")
        print(f"  Lon: [{lon_rho.min():.2f}, {lon_rho.max():.2f}]")
        print(f"  Lat: [{lat_rho.min():.2f}, {lat_rho.max():.2f}]")
        print(f"  Depth: [{h.min():.2f}, {h.max():.2f}] m")
    
    except Exception as e:
        print(f"✗ Error reading ROMS file: {e}")
        sys.exit(1)
    
    # ========================================================================
    # Configure WW3 Grid
    # ========================================================================
    
    print_section("Configuring WW3 grid")
    
    lon_min = lon_rho.min()
    lon_max = lon_rho.max()
    lat_min = lat_rho.min()
    lat_max = lat_rho.max()
    
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    
    # Calculate grid dimensions automatically
    nx_ww3, ny_ww3 = calculate_grid_dimensions(lon_range, lat_range, dx_ww3, dy_ww3)
    
    # Create regular grid
    lon_ww3 = np.linspace(lon_min, lon_max, nx_ww3)
    lat_ww3 = np.linspace(lat_min, lat_max, ny_ww3)
    
    print(f"Grid dimensions calculated:")
    print(f"  Domain: Lon [{lon_min:.2f}, {lon_max:.2f}]° ({lon_range:.2f}°)")
    print(f"          Lat [{lat_min:.2f}, {lat_max:.2f}]° ({lat_range:.2f}°)")
    print(f"  Spacing: {dx_ww3:.6f}° × {dy_ww3:.6f}°")
    print(f"  Grid: {nx_ww3} × {ny_ww3} points")
    
    # Create meshgrid
    lon_grid_ww3, lat_grid_ww3 = np.meshgrid(lon_ww3, lat_ww3)
    
    # Flatten ROMS coordinates for interpolation
    lons_flat = lon_rho.flatten()
    lats_flat = lat_rho.flatten()
    h_flat = h.flatten()
    mask_flat = mask_rho.flatten()
    
    points = np.column_stack([lons_flat, lats_flat])
    points_grid = np.column_stack([lon_grid_ww3.flatten(), lat_grid_ww3.flatten()])
    
    # ========================================================================
    # Interpolation
    # ========================================================================
    
    print_section("Interpolating data")
    
    # Bathymetry interpolation
    print("Interpolating bathymetry...", end=' ')
    h_interp = griddata(points, h_flat, points_grid, method=bath_interp, fill_value=bath_fill)
    h_interp = h_interp.reshape((ny_ww3, nx_ww3))
    h_interp = np.nan_to_num(h_interp, nan=bath_fill)
    h_interp[h_interp < min_depth] = min_depth
    print("✓")
    
    if verbosity == 'verbose':
        print(f"  Range: [{h_interp.min():.2f}, {h_interp.max():.2f}] m")
    
    # Mask interpolation
    print("Interpolating mask...", end=' ')
    mask_interp = griddata(points, mask_flat, points_grid, method='nearest', fill_value=0)
    mask_interp = mask_interp.reshape((ny_ww3, nx_ww3)).astype(int)

    # Normalize to binary land/sea and mark boundaries only over ocean cells.
    ocean = mask_interp > 0
    ww3_mask = np.where(ocean, 1, 0).astype(int)
    if mark_boundaries:
        border = np.zeros_like(ww3_mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        ww3_mask[border & ocean] = boundary_value
    
    print("✓")
    
    if verbosity == 'verbose':
        unique_mask = np.unique(ww3_mask)
        print(f"  Unique values: {unique_mask}")
    
    # ========================================================================
    # Write Output Files
    # ========================================================================
    
    print_section("Writing output files")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get output format from config
    bath_fmt = get_config_value(config, 'output.bathymetry_format', '{:8.0f}')
    mask_fmt = get_config_value(config, 'output.mask_format', '{:6d}')
    
    # Convert units if needed
    if bath_output_unit.lower() == 'cm':
        h_output = h_interp * 100
    else:
        h_output = h_interp
    
    # Write bathymetry
    depth_file = os.path.join(output_dir, f'{output_prefix}.depth_ascii')
    print(f"Bathymetry: {depth_file}...", end=' ')
    with open(depth_file, 'w') as f:
        for j in range(ny_ww3):
            line_data = ''.join([bath_fmt.format(h_output[j, i]) for i in range(nx_ww3)])
            f.write(line_data + '\n')
    print("✓")
    
    # Write mask
    mask_file = os.path.join(output_dir, f'{output_prefix}.maskorig_ascii')
    print(f"Mask: {mask_file}...", end=' ')
    with open(mask_file, 'w') as f:
        for j in range(ny_ww3):
            line_data = ''.join([mask_fmt.format(ww3_mask[j, i]) for i in range(nx_ww3)])
            f.write(line_data + '\n')
    print("✓")
    
    # Write obstacles
    obstr_file = os.path.join(output_dir, f'{output_prefix}.obstr_lev1')
    print(f"Obstacles: {obstr_file}...", end=' ')
    obstr = np.zeros((ny_ww3, nx_ww3), dtype=int)
    with open(obstr_file, 'w') as f:
        for j in range(ny_ww3):
            line_data = ''.join([mask_fmt.format(obstr[j, i]) for i in range(nx_ww3)])
            f.write(line_data + '\n')
    print("✓")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print_header("CONVERSION COMPLETED SUCCESSFULLY!")
    
    print(f"\n📊 ROMS Grid (input):")
    print(f"   Dimensions: {h.shape[1]} × {h.shape[0]} points")
    print(f"   Domain: Lon [{lon_min:.2f}, {lon_max:.2f}]°")
    print(f"           Lat [{lat_min:.2f}, {lat_max:.2f}]°")
    
    print(f"\n📈 WW3 Grid (output):")
    print(f"   Dimensions: {nx_ww3} × {ny_ww3} points")
    print(f"   Spacing: {dx_ww3:.6f}° × {dy_ww3:.6f}°")
    print(f"   Domain: Lon [{lon_ww3[0]:.2f}, {lon_ww3[-1]:.2f}]°")
    print(f"           Lat [{lat_ww3[0]:.2f}, {lat_ww3[-1]:.2f}]°")
    
    print(f"\n📝 Interpolated data:")
    print(f"   Bathymetry: [{h_output.min():.0f}, {h_output.max():.0f}] {bath_output_unit}")
    print(f"   Mask values: {np.unique(ww3_mask)}")
    
    print(f"\n📁 Files generated in '{output_dir}/':")
    print(f"   ✓ {os.path.basename(depth_file)}")
    print(f"   ✓ {os.path.basename(mask_file)}")
    print(f"   ✓ {os.path.basename(obstr_file)}")
    
    print("="*70)

if __name__ == '__main__':
    main()
