# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 01:40:22 2015

@author: bdyer
"""

from osgeo import gdal, ogr
from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.pyplot as plt
from matplotlib import cm

# read 2.5 minute U.S. DEM file using gdal.
# (http://www.prism.oregonstate.edu/docs/meta/dem_25m.htm)
gd = gdal.Open('test.tif')
band=gd.GetRasterBand(1)
array = band.ReadAsArray()
arrayMasked = ma.masked_greater(array,-30)
#array[~arrayMasked.mask]=float('nan')
arrayMasked = ma.masked_greater(array,0)
#array[arrayMasked.mask]=float('nan')
# get lat/lon coordinates from DEM file.
coords = gd.GetGeoTransform()
nlons = array.shape[1]; nlats = array.shape[0]
delon = coords[1]
delat = coords[5]
lons = coords[0] + delon*np.arange(nlons)
lats = coords[3] + delat*np.arange(nlats)[::-1] # reverse lats

#make coords arrays
iterator = range(0,array[:,1].size,50);
coordlons=[]; coordlats=[]; coordarray=[];
for x in range(0,array[1,:].size,50):
    coordlons.extend(lons[x]*np.ones(array[iterator,1].size))
    coordlats.extend(lats[iterator])
    for y in range(0,len(iterator)):
        coordarray.extend([array[iterator[y],x]])

coordlons=ma.asarray(coordlons).data
coordlats=ma.asarray(coordlats).data
coordarray=ma.asarray(coordarray).data

#fig = plt.figure(figsize=(19,14))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter3D(coordlons,coordlats,coordarray,c=coordarray,cmap=plt.cm.jet)  
#plt.show()

import numpy as np
from matplotlib.mlab import griddata
import scipy as sp
import scipy.interpolate

x=coordlons[~np.isnan(coordarray)]
y=coordlats[~np.isnan(coordarray)]
z=coordarray[~np.isnan(coordarray)]



spline = sp.interpolate.Rbf(x,y,z,function='thin-plate')
xi = np.linspace(min(x), max(x),num=175)
yi = np.linspace(min(y), max(y),num=175)
X, Y = np.meshgrid(xi, yi)

# interpolation
Z = spline(X,Y)

fig = plt.figure(figsize=(19,14))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=0.1, antialiased=True, vmin=-6000, vmax=200)
plt.show()

# setup figure.
#fig = plt.figure(figsize=(19,14))
## setup basemap instance.
#m = Basemap(llcrnrlon=-79,llcrnrlat=22,urcrnrlon=-75,urcrnrlat=26.5,
#            projection='lcc',lat_1=21,lat_2=26,lon_0=-75)
## create masked array, reversing data in latitude direction
## (so that data is oriented in increasing latitude, as transform_scalar requires).
#topoin = ma.masked_values(array[::-1,:],-999.)
## transform DEM data to a 4 km native projection grid
#nx = int((m.xmax-m.xmin)/400.)+1; ny = int((m.ymax-m.ymin)/400.)+1
#topodat = m.transform_scalar(topoin,lons,lats,nx,ny,masked=True)
## plot DEM image on map.
#im = m.imshow(topodat,cmap=plt.cm.jet)


m=None
topodat=None
im=None
gd=None
spline=None;
Z=None;

