from __future__ import unicode_literals 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import savgol_filter
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import xarray as xr
import netCDF4 as nc 
from datetime import datetime 
from NDBC.NDBC import DataBuoy

DB = DataBuoy()
DB.set_station_id(idbuoy)  
DB.get_station_metadata()

#choose the period (year, month, day, hour,min)
start_date = datetime(2019, 11, 1, 0, 0)
end_date = datetime(2019, 12, 31, 23, 0)

start_year = int(start_date.strftime('%Y'))
end_year = int(end_date.strftime('%Y'))+1
year_range = range(start_year, end_year)
start_date_n = (start_date.strftime('%Y%m%d'))

DB.get_data(years=year_range , datetime_index=True, data_type='stdmet')

#checking data availability
df = DB.data.get('stdmet').get('data')
wvht = df[~df['WVHT'].isna()]
print("The first time available is: ", wvht.index[0].to_pydatetime())
print("The last time available is: ", wvht.index[-1].to_pydatetime())

units_dict = DB.data.get('stdmet').get('meta').get('units')

#getting the data during the choosen time period
wvht.index = wvht.index.to_pydatetime()
mask = (wvht.index > start_date) & (wvht.index <= end_date)
wvht = wvht.loc[mask]
xr_ds = wvht.to_xarray()

xr_ds.to_netcdf(path='./'+str(idbuoy)+'.nc', mode='w', format=None, group=None,
engine=None, encoding=None, unlimited_dims=None, compute=True,\
invalid_netcdf=False)

#saving to netcdf files
keys_list = list(units_dict.keys())
ds = nc.Dataset('./'+str(idbuoy)+'.nc',mode='a')
var_names = ds.variables.keys()

for name in var_names:
    if name in keys_list:
     
       ds.variables[name].units = units_dict[name]

ds.close()
attr_dict = DB.station_info

f = nc.Dataset('./'+str(idbuoy)+'.nc','a')

for name, value in attr_dict.items():
    setattr(f, name.replace(" ", "_"), value)

f.renameDimension(u'index',u'time')
time = f.createVariable('time', np.int_, ('time',))
                                          
time.units = f.variables['index'].units
time.calendar = f.variables['index'].calendar
time[:] = f.variables['index'][:]

f.close()
