#----fhis program extracts the points from WW3 corresponding 
#----to the NDBC buoy stations 
import os,sys
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

variable=['hs'] 

#----for NDBC buoys
with open('./pointsNDBC.info', 'r') as file:
     points = json.load(file)

#----fpath for ww3 files
arqin='/Users/jtakeo/googleDrive/myProjects/cmcc/ww3GlobalUnst/buoys/ndbc/ww3/ww3.*.nc'
#----fpath for the ww3 extracted points
arqout='/Users/jtakeo/googleDrive/myProjects/cmcc/ww3GlobalUnst/buoys/ndbc/ww3/points'

for name, info in points.items():
    pto = f"{name}"; x = points[f"{name}"]['x']; y = points[f"{name}"]['y']
    print("Doing buoy id: ",pto)
    fileout=f'ww3_{pto}'   
    buoy_extraction(arqin,x,y,os.path.join(arqout, fileout),variable)
