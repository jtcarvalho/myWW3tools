#!/usr/bin/env python3
"""
WW3 Grid Generator

Creates WW3 (WAVE WATCH III) ASCII grid files from:
  - Geographic bounds (rectilinear mode) or a 2-D lon/lat array (curvilinear mode)
  - A global bathymetry dataset (GEBCO, ETOPO, or any NetCDF with elevation)

Outputs (same format as WW3 gridgen):
  {prefix}.dep   — bathymetry (mm, scale 0.001 → m)
  {prefix}.mask  — land/sea mask (0=land, 1=ocean)
  {prefix}.obs   — sub-grid obstacles (all zeros; use computeObstrWW3.py for real data)
  {prefix}.meta  — WW3 grid definition (can be embedded in ww3_grid.inp)
  {prefix}.lon   — longitude array  [CURV only, IDLA=1, degrees]
  {prefix}.lat   — latitude array   [CURV only, IDLA=1, degrees]

Usage:
    python genWW3grid_fromBounds.py [config_file.yaml]

If no config file is given, defaults to 'config_bounds.yaml'.

Grid types
----------
grid.type: rect  (default)
    Regular rectilinear grid defined by bounds + resolution.
    Set grid.lon_min/max, grid.lat_min/max, grid.dx, grid.dy.

grid.type: curv
    Curvilinear grid.  Two sub-modes:

    a) From a NetCDF file containing 2-D lon/lat arrays:
       grid.curv_file: path/to/grid.nc
       grid.lon_var: lon   # name of the longitude variable (2-D)
       grid.lat_var: lat   # name of the latitude variable  (2-D)

    b) Polar stereographic projection (built-in, no extra file needed):
       grid.projection: stereo
       grid.center_lon: -45.0    # longitude of projection centre
       grid.center_lat: -90.0    # latitude  of projection centre (pole)
       grid.nx: 200              # number of columns
       grid.ny: 200              # number of rows
       grid.dx_km: 9.0           # grid spacing in km (same in x and y)

Bathymetry sources (in order of priority):
  1. Local NetCDF file (GEBCO, ETOPO, etc.) — set bathymetry.file in config
  2. pygmt — set bathymetry.source: pygmt  (requires pygmt + GMT installed)

Tip: GEBCO NetCDF files are freely available at https://www.gebco.net/
"""

import numpy as np
import xarray as xr
import os
import sys
from scipy.interpolate import RegularGridInterpolator, griddata

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("WARNING: PyYAML not installed. Install with: pip install pyyaml")

# ============================================================================
# Configuration helpers
# ============================================================================

def load_config(config_file):
    if not HAS_YAML:
        return None
    if not os.path.exists(config_file):
        print(f"Config file '{config_file}' not found.")
        return None
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  Config loaded from '{config_file}'")
        return config
    except Exception as e:
        print(f"  Error reading config: {e}")
        return None


def get(config, key_path, default=None):
    """Get a nested config value using dot-notation (e.g., 'grid.dx')."""
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


# ============================================================================
# Bathymetry loading
# ============================================================================

def _normalize_coords(da):
    """
    Rename dimension/coordinate names so that the DataArray always
    has dims named 'lat' and 'lon' (1-D, monotonically increasing).
    """
    rename = {}
    for dim in da.dims:
        d = dim.lower()
        if d in ('latitude', 'y'):
            rename[dim] = 'lat'
        elif d in ('longitude', 'x'):
            rename[dim] = 'lon'
    if rename:
        da = da.rename(rename)
    # Also rename coordinates if they differ from dims
    coord_rename = {}
    for coord in da.coords:
        c = coord.lower()
        if c in ('latitude', 'y') and coord not in da.dims:
            coord_rename[coord] = 'lat'
        elif c in ('longitude', 'x') and coord not in da.dims:
            coord_rename[coord] = 'lon'
    if coord_rename:
        da = da.rename(coord_rename)
    return da


def _sort_coords(da):
    """Ensure lat and lon are monotonically increasing."""
    if 'lat' in da.dims and da['lat'].values[0] > da['lat'].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    if 'lon' in da.dims and da['lon'].values[0] > da['lon'].values[-1]:
        da = da.isel(lon=slice(None, None, -1))
    return da


def load_bathy_from_file(filepath, elev_var=None, lon_min=None, lon_max=None,
                         lat_min=None, lat_max=None, pad=1.0):
    """
    Load bathymetry from a local NetCDF file (GEBCO, ETOPO, etc.).
    Subsets to the requested region (+ pad degrees) for efficiency.

    Returns a DataArray with dims ('lat', 'lon') and elevation in metres
    (negative = ocean, positive = land).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bathymetry file not found: {filepath}")

    print(f"  Opening bathymetry file: {filepath}")
    ds = xr.open_dataset(filepath)

    # Auto-detect elevation variable
    if elev_var is None:
        for candidate in ['elevation', 'z', 'altitude', 'depth',
                          'topo', 'height', 'Band1']:
            if candidate in ds.data_vars:
                elev_var = candidate
                break
    if elev_var is None:
        raise ValueError(
            f"Cannot detect elevation variable. Available: {list(ds.data_vars)}\n"
            "Set bathymetry.variable in the config file."
        )

    da = ds[elev_var].squeeze()   # drop any size-1 dims (e.g., time)
    da = _normalize_coords(da)
    da = _sort_coords(da)

    # Subset to region
    if lat_min is not None:
        lat_lo = max(float(da['lat'].min()), lat_min - pad)
        lat_hi = min(float(da['lat'].max()), lat_max + pad)
        da = da.sel(lat=slice(lat_lo, lat_hi))

    if lon_min is not None:
        # Handle 0-360 vs -180-180 mismatch
        src_lon_min = float(da['lon'].min())
        src_lon_max = float(da['lon'].max())

        req_lon_min = lon_min - pad
        req_lon_max = lon_max + pad

        if src_lon_min >= 0 and req_lon_min < 0:
            # Source is 0-360, requested is negative → convert request
            req_lon_min += 360
            req_lon_max += 360
        elif src_lon_min < 0 and req_lon_min > 180:
            # Source is -180-180, requested >180 → convert request
            req_lon_min -= 360
            req_lon_max -= 360

        req_lon_min = max(src_lon_min, req_lon_min)
        req_lon_max = min(src_lon_max, req_lon_max)
        da = da.sel(lon=slice(req_lon_min, req_lon_max))

    print(f"  Variable '{elev_var}' | Region subset: "
          f"lat [{float(da.lat.min()):.2f}, {float(da.lat.max()):.2f}], "
          f"lon [{float(da.lon.min()):.2f}, {float(da.lon.max()):.2f}]")
    return da


def load_bathy_pygmt(lon_min, lon_max, lat_min, lat_max, resolution='01m'):
    """
    Load bathymetry via pygmt (downloads GEBCO/SRTM15+ tiles as needed).
    resolution: GMT grid resolution string, e.g. '01m' (1 arc-min), '30s', '15s'.
    """
    try:
        import pygmt
    except ImportError:
        raise ImportError(
            "pygmt is not installed. Install with: pip install pygmt\n"
            "Also requires GMT (https://www.generic-mapping-tools.org/)"
        )
    pad = 0.5
    region = [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad]
    print(f"  Downloading bathymetry via pygmt (resolution={resolution}, region={region})")
    grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region)
    # pygmt returns an xarray DataArray with dims 'lat'/'lon'
    da = _normalize_coords(grid)
    da = _sort_coords(da)
    return da


def load_bathymetry(config, lon_min, lon_max, lat_min, lat_max):
    """
    Load bathymetry from the configured source.
    Returns DataArray with dims ('lat', 'lon') and elevation in metres.
    """
    source = get(config, 'bathymetry.source', 'file')
    bathy_file = get(config, 'bathymetry.file', None)

    if source == 'pygmt' or (source == 'file' and bathy_file is None):
        if bathy_file is None:
            # No file → try pygmt
            resolution = get(config, 'bathymetry.pygmt_resolution', '01m')
            return load_bathy_pygmt(lon_min, lon_max, lat_min, lat_max, resolution)
        else:
            elev_var = get(config, 'bathymetry.variable', None)
            return load_bathy_from_file(
                bathy_file, elev_var, lon_min, lon_max, lat_min, lat_max)
    elif source == 'file':
        elev_var = get(config, 'bathymetry.variable', None)
        return load_bathy_from_file(
            bathy_file, elev_var, lon_min, lon_max, lat_min, lat_max)
    else:
        print(f"Unknown bathymetry source '{source}'. Trying pygmt …")
        resolution = get(config, 'bathymetry.pygmt_resolution', '01m')
        return load_bathy_pygmt(lon_min, lon_max, lat_min, lat_max, resolution)


# ============================================================================
# Interpolation
# ============================================================================

def interpolate_to_ww3_grid(bathy_da, lon_ww3, lat_ww3):
    """
    Bi-linear interpolation of source bathymetry onto a WW3 REGULAR grid.

    lon_ww3, lat_ww3 : 1-D arrays (rectilinear case).
    Points outside the source domain are filled with 999999 (land sentinel).
    Returns a 2-D array of shape (ny, nx) in metres.
    """
    src_lat = bathy_da['lat'].values.astype(float)
    src_lon = bathy_da['lon'].values.astype(float)
    src_values = bathy_da.values.astype(float)

    # Handle source lon convention vs request lon convention
    src_lon_0360 = src_lon[0] >= 0
    ww3_lon_neg  = lon_ww3[0] < 0

    interp_lon_ww3 = lon_ww3.copy()
    if src_lon_0360 and ww3_lon_neg:
        interp_lon_ww3 = lon_ww3 + 360.0
    elif not src_lon_0360 and not ww3_lon_neg and lon_ww3[0] > 180:
        interp_lon_ww3 = lon_ww3 - 360.0

    interp_lon_ww3 = np.clip(interp_lon_ww3, src_lon.min(), src_lon.max())
    interp_lat_ww3 = np.clip(lat_ww3,        src_lat.min(), src_lat.max())

    interpolator = RegularGridInterpolator(
        (src_lat, src_lon), src_values,
        method='linear', bounds_error=False, fill_value=999999.0
    )

    lon_grid, lat_grid = np.meshgrid(interp_lon_ww3, interp_lat_ww3)
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    elev = interpolator(pts).reshape(len(lat_ww3), len(lon_ww3))
    return elev


def interpolate_to_curv_grid(bathy_da, lon2d, lat2d):
    """
    Interpolate source bathymetry onto a WW3 CURVILINEAR grid.

    lon2d, lat2d : 2-D arrays of shape (ny, nx) — target grid node positions.
    Uses nearest-neighbour via scipy.griddata for robustness with irregular
    target geometries (polar stereographic, rotated, etc.).
    Points outside the source domain are filled with 999999 (land sentinel).
    Returns a 2-D array of shape (ny, nx) in metres.
    """
    src_lat = bathy_da['lat'].values.astype(float)
    src_lon = bathy_da['lon'].values.astype(float)
    src_values = bathy_da.values.astype(float)

    # Build flat source point cloud
    src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
    src_pts = np.column_stack([src_lon2d.ravel(), src_lat2d.ravel()])
    src_vals = src_values.ravel()

    # Harmonise lon conventions
    query_lon = lon2d.copy()
    if src_lon.min() >= 0 and query_lon.min() < 0:
        query_lon = query_lon + 360.0
    elif src_lon.min() < 0 and query_lon.max() > 180:
        query_lon = query_lon - 360.0

    dst_pts = np.column_stack([query_lon.ravel(), lat2d.ravel()])

    print("  Using linear griddata interpolation for curvilinear target …",
          end=' ', flush=True)
    elev = griddata(src_pts, src_vals, dst_pts, method='linear',
                    fill_value=999999.0)
    print("done")
    return elev.reshape(lon2d.shape)


# ============================================================================
# Mask creation
# ============================================================================

def make_mask(elevation, zdep=-0.10):
    """
    Build ocean/land mask from elevation (metres, negative = ocean).
      0 = land  (elevation >= zdep)
      1 = ocean (elevation <  zdep)

    Boundary-condition marking (value 2) is intentionally left to the
    separate prepMaskWW3.py script, matching the convention in the examples.
    """
    mask = np.where(elevation < zdep, 1, 0).astype(int)
    return mask


# ============================================================================
# File writers
# ============================================================================

LAND_VALUE_MM = 999999000   # sentinel for land points in dep file

def write_dep(filepath, elevation_m, nx, ny):
    """
    Write WW3 bathymetry file.
    Values are stored as integers in millimetres (scale factor 0.001 → m).
    Layout: IDLA=1, first row = southernmost latitude (bottom-to-top order).
    Land sentinel: 999999000 mm (= 999999 m).
    """
    data_mm = np.round(elevation_m * 1000).astype(np.int64)
    # Replace unrealistic values (fill from interpolation) with land sentinel
    data_mm[data_mm > 900000000] = LAND_VALUE_MM

    with open(filepath, 'w') as f:
        for j in range(ny):
            line = ''.join(f'{data_mm[j, i]:12d}' for i in range(nx))
            f.write(line + '\n')


def write_mask(filepath, mask, nx, ny):
    """
    Write WW3 mask file.
    0 = land, 1 = ocean.  Layout: IDLA=1 (south to north).
    """
    with open(filepath, 'w') as f:
        for j in range(ny):
            line = ''.join(f'{mask[j, i]:3d}' for i in range(nx))
            f.write(line + '\n')


def write_obs(filepath, nx, ny):
    """
    Write WW3 sub-grid obstacles file (all zeros).
    Shape: (2*NY, NX) — rows 0..NY-1 for E-W faces, NY..2NY-1 for N-S faces.
    Values 0-100 represent % blockage; all zeros = no obstacles.
    """
    with open(filepath, 'w') as f:
        for j in range(2 * ny):
            line = ''.join(f'{0:3d}' for _ in range(nx))
            f.write(line + '\n')


def write_curv_coord(filepath, data2d, nx, ny):
    """
    Write a WW3 curvilinear coordinate file (.lon or .lat).
    Format: IDLA=1, one row per latitude index (south to north),
    free-format floating-point (12.6f per value).
    """
    with open(filepath, 'w') as f:
        for j in range(ny):
            line = ''.join(f'{data2d[j, i]:12.6f}' for i in range(nx))
            f.write(line + '\n')


def write_meta_curv(filepath, nx, ny, prefix, zdep=-0.10, zmin=2.50,
                    coord_scale=1.0):
    """
    Write the WW3 grid meta file for a CURVILINEAR grid.

    The .lon and .lat files are read via unit numbers 20 and 30 respectively.
    coord_scale is the multiplier applied to the stored coordinate values
    (1.0 for degrees — WW3 reads them and multiplies by this factor).
    """
    with open(filepath, 'w') as f:
        f.write(_META_HEADER)
        f.write(f"   'CURV'  T  'NONE'\n")
        f.write(f"{nx:7d}  {ny:7d}\n")
        # X coordinate file
        f.write(f"20  {coord_scale:.6f}  0.00  1  1 "
                f"'(....)'  NAME  './{prefix}.lon'\n")
        # Y coordinate file
        f.write(f"30  {coord_scale:.6f}  0.00  1  1 "
                f"'(....)'  NAME  './{prefix}.lat'\n")
        f.write(f"$ Bottom Bathymetry\n")
        f.write(f"{zdep:.2f}   {zmin:.2f}  40  0.001000  1  1 "
                f"'(....)'  NAME  './{prefix}.dep'\n")
        f.write(f"$ Sub-grid information\n")
        f.write(f"50  0.010000  1  1  '(....)'  NAME  './{prefix}.obs'\n")
        f.write(f"$ Mask Information\n")
        f.write(f"60  1  1  '(....)'  NAME  './{prefix}.mask'\n")


# ============================================================================
# Curvilinear grid builders
# ============================================================================

def build_curv_from_file(config):
    """
    Load a curvilinear grid from a NetCDF file containing 2-D lon/lat arrays.

    Config keys used:
        grid.curv_file  : path to the NetCDF file
        grid.lon_var    : name of the longitude variable (default 'lon')
        grid.lat_var    : name of the latitude variable  (default 'lat')

    Returns lon2d, lat2d  both of shape (ny, nx)
    """
    curv_file = get(config, 'grid.curv_file')
    if curv_file is None:
        raise ValueError(
            "grid.curv_file must be set when grid.type = 'curv' "
            "and grid.projection is not specified.")
    if not os.path.exists(curv_file):
        raise FileNotFoundError(f"Curvilinear grid file not found: {curv_file}")

    lon_var = get(config, 'grid.lon_var', 'lon')
    lat_var = get(config, 'grid.lat_var', 'lat')

    print(f"  Loading curvilinear grid from: {curv_file}")
    ds = xr.open_dataset(curv_file)

    if lon_var not in ds and lat_var not in ds:
        # Try common alternatives
        for lv in ['longitude', 'LONGITUDE', 'x', 'X']:
            if lv in ds:
                lon_var = lv
                break
        for lv in ['latitude', 'LATITUDE', 'y', 'Y']:
            if lv in ds:
                lat_var = lv
                break

    lon2d = ds[lon_var].values.astype(float)
    lat2d = ds[lat_var].values.astype(float)

    if lon2d.ndim == 1:
        # 1-D arrays → make 2-D meshgrid
        lon2d, lat2d = np.meshgrid(lon2d, lat2d)

    ny, nx = lon2d.shape
    print(f"  Curvilinear grid size: {nx} x {ny}")
    print(f"  lon range: [{lon2d.min():.3f}, {lon2d.max():.3f}]°")
    print(f"  lat range: [{lat2d.min():.3f}, {lat2d.max():.3f}]°")
    return lon2d, lat2d


def build_curv_stereo(config):
    """
    Build a polar stereographic curvilinear grid.

    Config keys used:
        grid.center_lon  : central longitude (degrees, default 0)
        grid.center_lat  : central latitude  (degrees, default -90 = south pole)
        grid.nx          : number of columns
        grid.ny          : number of rows
        grid.dx_km       : grid spacing in km (uniform, default 25)

    Returns lon2d, lat2d  both of shape (ny, nx)
    """
    clon = float(get(config, 'grid.center_lon', 0.0))
    clat = float(get(config, 'grid.center_lat', -90.0))
    nx   = int(get(config, 'grid.nx', 100))
    ny   = int(get(config, 'grid.ny', 100))
    dkm  = float(get(config, 'grid.dx_km', 25.0))

    print(f"  Building polar stereographic grid:")
    print(f"    centre = ({clon}°, {clat}°)  |  "
          f"{nx} x {ny} cells  |  spacing = {dkm} km")

    # Grid coordinates in km, centred at origin
    x_km = (np.arange(nx) - (nx - 1) / 2.0) * dkm
    y_km = (np.arange(ny) - (ny - 1) / 2.0) * dkm
    X, Y = np.meshgrid(x_km, y_km)

    # Inverse polar stereographic projection (true at pole)
    # Reference: Snyder (1987) Map Projections — A Working Manual, p. 161
    R_earth = 6371.0  # km
    sign = np.sign(clat) if clat != 0 else 1.0  # hemisphere
    clat_r = np.radians(abs(clat))
    clon_r = np.radians(clon)

    rho = np.sqrt(X**2 + Y**2)
    # Avoid division by zero at the pole
    rho = np.where(rho == 0, 1e-12, rho)

    # For stereographic projection true at the pole:
    # c = 2 * arctan(rho / (2*R)), but since we use true-at-pole:
    # c = 2 * arctan2(rho, 2*R)
    c = 2.0 * np.arctan2(rho, 2.0 * R_earth)

    lat_r = np.arcsin(
        np.cos(c) * np.sin(np.radians(clat_r * sign))
        + sign * Y * np.sin(c) * np.cos(np.radians(clat_r)) / rho
    )
    lon_r = clon_r + np.arctan2(
        X * np.sin(c),
        (rho * np.cos(np.radians(clat_r)) * np.cos(c)
         - sign * Y * np.sin(np.radians(clat_r)) * np.sin(c))
    )

    lat2d = np.degrees(lat_r)
    lon2d = np.degrees(lon_r)
    # Wrap longitude to -180..180
    lon2d = ((lon2d + 180) % 360) - 180

    print(f"  lon range: [{lon2d.min():.3f}, {lon2d.max():.3f}]°")
    print(f"  lat range: [{lat2d.min():.3f}, {lat2d.max():.3f}]°")
    return lon2d, lat2d


_META_HEADER = """\
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


def write_meta(filepath, nx, ny, dx, dy, lon1, lat1, prefix,
               zdep=-0.10, zmin=2.50, closure='NONE'):
    """
    Write the WW3 grid meta file (can be embedded in ww3_grid.inp).

    Grid increments are stored in arc-minutes (SX = dx * 60) with
    SCALE=60, matching the convention used in the provided examples.

    Global grids (lon span ≈ 360°) automatically use 'SMPL' closure.
    """
    sx = dx * 60.0
    sy = dy * 60.0
    scale_xy = 60.0

    # Auto-detect simple closure for global grids
    lon_span = (nx - 1) * dx
    if abs(lon_span - 360.0) < dx:
        closure = 'SMPL'

    with open(filepath, 'w') as f:
        f.write(_META_HEADER)
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
# Main
# ============================================================================

def main():
    # ── Configuration ────────────────────────────────────────────────────────
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config_bounds.yaml'
    config = load_config(config_file)

    print("\n" + "=" * 70)
    print(" WW3 GRID GENERATOR")
    print("=" * 70)

    # Output
    output_dir = get(config, 'output.directory', 'ww3_grid_output')
    prefix     = get(config, 'output.prefix', 'grid')
    zdep       = float(get(config, 'grid.zdep', -0.10))
    zmin       = float(get(config, 'grid.zmin',  2.50))

    grid_type  = str(get(config, 'grid.type', 'rect')).lower()

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # CURVILINEAR path
    # =========================================================================
    if grid_type == 'curv':
        print("\n→ Grid type: CURVILINEAR")

        projection = get(config, 'grid.projection', None)

        if projection is not None and str(projection).lower() == 'stereo':
            print("\n→ Building polar stereographic grid")
            lon2d, lat2d = build_curv_stereo(config)
        else:
            print("\n→ Loading curvilinear grid from file")
            lon2d, lat2d = build_curv_from_file(config)

        ny, nx = lon2d.shape

        # Bounding box for bathymetry subset
        lon_min = float(np.nanmin(lon2d))
        lon_max = float(np.nanmax(lon2d))
        lat_min = float(np.nanmin(lat2d))
        lat_max = float(np.nanmax(lat2d))

        print(f"\n  Grid size : {nx} × {ny}  (NX × NY)")

        # ── Load and interpolate bathymetry ───────────────────────────────
        print("\n→ Loading bathymetry")
        bathy_da = load_bathymetry(config, lon_min, lon_max, lat_min, lat_max)

        print("\n→ Interpolating to curvilinear WW3 grid")
        elevation = interpolate_to_curv_grid(bathy_da, lon2d, lat2d)
        print(f"  Elevation range: [{elevation.min():.1f}, {elevation.max():.1f}] m")

        # ── Mask ──────────────────────────────────────────────────────────
        print("\n→ Computing land/sea mask")
        mask = make_mask(elevation, zdep=zdep)
        n_ocean = int((mask == 1).sum())
        n_land  = int((mask == 0).sum())
        pct     = 100.0 * n_ocean / (nx * ny)
        print(f"  Ocean cells : {n_ocean}  ({pct:.1f}%)")
        print(f"  Land cells  : {n_land}")

        if n_ocean == 0:
            print("\n  WARNING: All grid cells are classified as land!")

        # ── Write files ───────────────────────────────────────────────────
        print("\n→ Writing output files")

        dep_file  = os.path.join(output_dir, f'{prefix}.dep')
        mask_file = os.path.join(output_dir, f'{prefix}.mask')
        obs_file  = os.path.join(output_dir, f'{prefix}.obs')
        lon_file  = os.path.join(output_dir, f'{prefix}.lon')
        lat_file  = os.path.join(output_dir, f'{prefix}.lat')
        meta_file = os.path.join(output_dir, f'{prefix}.meta')

        print(f"  {dep_file} …",  end=' ', flush=True)
        write_dep(dep_file, elevation, nx, ny)
        print("done")

        print(f"  {mask_file} …", end=' ', flush=True)
        write_mask(mask_file, mask, nx, ny)
        print("done")

        print(f"  {obs_file} …",  end=' ', flush=True)
        write_obs(obs_file, nx, ny)
        print("done")

        print(f"  {lon_file} …",  end=' ', flush=True)
        write_curv_coord(lon_file, lon2d, nx, ny)
        print("done")

        print(f"  {lat_file} …",  end=' ', flush=True)
        write_curv_coord(lat_file, lat2d, nx, ny)
        print("done")

        print(f"  {meta_file} …", end=' ', flush=True)
        write_meta_curv(meta_file, nx, ny, prefix, zdep=zdep, zmin=zmin)
        print("done")

        out_files = [dep_file, mask_file, obs_file, lon_file, lat_file,
                     meta_file]

        print("\n" + "=" * 70)
        print(" GRID GENERATION COMPLETE  (CURVILINEAR)")
        print("=" * 70)
        print(f"\n  Output directory : {os.path.abspath(output_dir)}/")
        print(f"  Grid size        : {nx} × {ny}  (NX × NY)")
        print(f"  lon range        : [{lon2d.min():.4f}, {lon2d.max():.4f}]°")
        print(f"  lat range        : [{lat2d.min():.4f}, {lat2d.max():.4f}]°")
        print(f"  Bathymetry range : [{elevation.min():.1f}, {elevation.max():.1f}] m")
        print(f"  Ocean/Land       : {n_ocean}/{n_land} cells")

    # =========================================================================
    # RECTILINEAR path  (default)
    # =========================================================================
    else:
        print("\n→ Grid type: RECTILINEAR")

        lon_min = get(config, 'grid.lon_min')
        lon_max = get(config, 'grid.lon_max')
        lat_min = get(config, 'grid.lat_min')
        lat_max = get(config, 'grid.lat_max')
        dx      = get(config, 'grid.dx')
        dy      = get(config, 'grid.dy')

        if any(v is None for v in [lon_min, lon_max, lat_min, lat_max, dx, dy]):
            print("\nERROR: grid.lon_min, lon_max, lat_min, lat_max, dx, dy "
                  "are required.")
            print("Please provide a config YAML file.  Example:")
            print("  python genWW3grid_fromBounds.py config_bounds.yaml")
            sys.exit(1)

        # ── Build WW3 regular grid ─────────────────────────────────────────
        print("\n→ Building WW3 grid")
        nx = int((lon_max - lon_min) / dx) + 1
        ny = int((lat_max - lat_min) / dy) + 1

        lon_ww3 = lon_min + np.arange(nx) * dx
        lat_ww3 = lat_min + np.arange(ny) * dy

        print(f"  Domain : lon [{lon_ww3[0]:.4f}, {lon_ww3[-1]:.4f}]°  "
              f"lat [{lat_ww3[0]:.4f}, {lat_ww3[-1]:.4f}]°")
        print(f"  (Requested max: lon {lon_max:.4f}°, lat {lat_max:.4f}°)")
        print(f"  Size   : {nx} × {ny}  (lon × lat)")
        print(f"  Spacing: {dx:.6f}° × {dy:.6f}°")

        # ── Load and interpolate bathymetry ───────────────────────────────
        print("\n→ Loading bathymetry")
        bathy_da = load_bathymetry(config, lon_min, lon_max, lat_min, lat_max)

        print("\n→ Interpolating to WW3 grid")
        elevation = interpolate_to_ww3_grid(bathy_da, lon_ww3, lat_ww3)
        print(f"  Elevation range: [{elevation.min():.1f}, {elevation.max():.1f}] m")

        # ── Mask ──────────────────────────────────────────────────────────
        print("\n→ Computing land/sea mask")
        mask = make_mask(elevation, zdep=zdep)
        n_ocean = int((mask == 1).sum())
        n_land  = int((mask == 0).sum())
        pct     = 100.0 * n_ocean / (nx * ny)
        print(f"  Ocean cells : {n_ocean}  ({pct:.1f}%)")
        print(f"  Land cells  : {n_land}")

        if n_ocean == 0:
            print("\n  WARNING: All grid cells are classified as land!")
            print("  Check your domain bounds and bathymetry source.")

        # ── Write files ───────────────────────────────────────────────────
        print("\n→ Writing output files")

        dep_file  = os.path.join(output_dir, f'{prefix}.dep')
        mask_file = os.path.join(output_dir, f'{prefix}.mask')
        obs_file  = os.path.join(output_dir, f'{prefix}.obs')
        meta_file = os.path.join(output_dir, f'{prefix}.meta')

        print(f"  {dep_file} …",  end=' ', flush=True)
        write_dep(dep_file, elevation, nx, ny)
        print("done")

        print(f"  {mask_file} …", end=' ', flush=True)
        write_mask(mask_file, mask, nx, ny)
        print("done")

        print(f"  {obs_file} …",  end=' ', flush=True)
        write_obs(obs_file, nx, ny)
        print("done")

        print(f"  {meta_file} …", end=' ', flush=True)
        write_meta(meta_file, nx, ny, dx, dy,
                   lon_ww3[0], lat_ww3[0], prefix, zdep=zdep, zmin=zmin)
        print("done")

        out_files = [dep_file, mask_file, obs_file, meta_file]

        print("\n" + "=" * 70)
        print(" GRID GENERATION COMPLETE  (RECTILINEAR)")
        print("=" * 70)
        print(f"\n  Output directory : {os.path.abspath(output_dir)}/")
        print(f"  Grid size        : {nx} × {ny}  (NX × NY)")
        print(f"  Spacing          : {dx:.6f}° × {dy:.6f}°")
        print(f"  Domain           : lon [{lon_ww3[0]:.4f}, {lon_ww3[-1]:.4f}]°")
        print(f"                     lat [{lat_ww3[0]:.4f}, {lat_ww3[-1]:.4f}]°")
        print(f"  Bathymetry range : [{elevation.min():.1f}, {elevation.max():.1f}] m")
        print(f"  Ocean/Land       : {n_ocean}/{n_land} cells")

    # ── Common footer ─────────────────────────────────────────────────────────
    print()
    print(f"  Files generated:")
    for f in out_files:
        print(f"    {f}")
    print()
    print("  NOTE: Run prepMaskWW3.py to add open-boundary markers (value 2)")
    print("        to the mask file if needed.")
    print("  NOTE: Run computeObstrWW3.py to replace the zero .obs file with")
    print("        realistic sub-grid obstruction values.")
    print("=" * 70)


if __name__ == '__main__':
    main()
