#%%
# import os,sys
import json
import glob
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import xarray as xr
import numpy as np
from glob import glob
from natsort import natsorted

def nearest(ds, x,y):
    return np.argmin(np.abs(ds.longitude.values-x)+np.abs(ds.latitude.values-y))

def buoy_extraction(base,x,y,outname, variables):

    buffer=[]
    for f in natsorted(glob(base)):
        print (f)
        ds=xr.open_dataset(f)
        ds=ds.isel(node=nearest(ds, x, y))[variables]
        #ds=ds.interp(node(ds,x,y),method='linear')[variables]        
        print(x,y)
        print('#-----#')
        buffer.append(ds)
    out=xr.concat(buffer,dim='time')
    out.to_netcdf(f'{outname}.nc')

#No Results for exp0,exp3,exp5
#ds=xr.open_dataset(path)
#timestr = ds['time'].dt.strftime('%Y-%m-%d %H')
#variable=['uwnd','vwnd'] 
variable=['hs'] 
#exp='exp21'

#----para as boias NDBC
with open('./pointsNDBC.info', 'r') as file:
     points = json.load(file)
#%%
# #----para Azores
# with open('./pointsAzores.info', 'r') as file:
#     points = json.load(file)

#----para as boias EMODNET
# with open('./pointsEMODNET.info', 'r') as file:
#      points = json.load(file)


#points = {
#    "6201012": {"x": -0.83, "y": 50.72, "area": 'no info'},
#}

# points = {
#     "51003": {"x": -160.639, "y": 19.196, "area": 'West Hawaii'},
#     "42001": {"x": -89.662, "y": 25.926, "area": 'Gulf of Mexico'},
#     "41047": {"x": -71.452, "y": 27.465, "area": 'US EastCoast' },
#     "46042": {"x": -122.396, "y": 36.785, "area": 'US EastCoast' },
#     "6200001": {"x": -5.0, "y": 45.2, "area": 'North Spain'}}


# Loop para imprimir nome e características de cada ponto
#for name, info in points.items():
#    pto = f"{name}"; x = points[f"{name}"]['x']; y = points[f"{name}"]['y']
#    print(pto, x ,y)



arqin='/Users/jtakeo/googleDrive/myProjects/cmcc/ww3GlobalUnst/buoys/ndbc/ww3/ww3.*.nc'
arqout='/Users/jtakeo/googleDrive/myProjects/cmcc/ww3GlobalUnst/buoys/ndbc/ww3/points'

for name, info in points.items():
    pto = f"{name}"; x = points[f"{name}"]['x']; y = points[f"{name}"]['y']
    print("Doing buoy id: ",pto)
    fileout=f'ww3_{pto}'   
    buoy_extraction(arqin,x,y,os.path.join(arqout, fileout),variable)

# %%
