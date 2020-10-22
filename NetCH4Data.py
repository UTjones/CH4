import netCDF4 
import numpy as np
import math
#https://iescoders.com/reading-netcdf4-data-in-python/


f = netCDF4.Dataset("xch4_nasa_run.nc4")
'''
printing f will give you the variables, the dimension of the variables, and what variables are required for getting another variable
e.g. XCH4 requires time, lat, lon to specify
 '''
print(f)
#print(f.variables.keys())
xch4_obj = f.variables['XCH4']
xch4 = xch4_obj[:]


lat_object = f.variables['lat']
lat = lat_object[:]
lon_object = f.variables['lon']
lon = lon_object[:]
time_object = f.variables['time']
time = time_object[:]
#print(xch4_obj)
#print(lon[0])
#print(lat[0])
#print(time[0])

#data = xch4[:, 0, 0]
#print(xch4[0, 0, 0])
#print(data)
#print(len(data))

import math

def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    #radius = 6371 # km
    radius = 3959 #mi

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

print(distance((lat[0], lon[20]), (lat[0], lon[21])))

print('lat 0', lat[0])
print('lat 1', lat[1])
print('lon 0', lon[0])
print('lon 1', lon[1])