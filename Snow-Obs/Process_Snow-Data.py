#!/usr/bin/env python
'''
    File name: Process_Snow-Data.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 21.05.2017
    Date last modified: 21.05.2017

    ##############################################################
    Purpos:

    Reads in hourly snow observations from txt files and write the data in to a NetCDF

    inputs:

    returns:

'''

########################################
#                            Load Modules
from dateutil import rrule
import datetime
import glob
from netCDF4 import Dataset
import netCDF4
import sys, traceback
import dateutil.parser as dparser
import string
from pdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
# from mpl_toolkits import basemap
import ESMF
import pickle
import subprocess
import pandas as pd
import copy
import linecache
from scipy.interpolate import interp1d
import pylab as plt
from HelperFunctions import fnSolidPrecip
from scipy import stats

########################################

sSnowData='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Snow/Raw/'
sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Snow/'

dStartDay=datetime.datetime(1960, 1, 1,0)
dStopDay=datetime.datetime(2017, 12, 31,23)
rgdTimeAll = pd.date_range(dStartDay, end=dStopDay, freq='h')
########################################

#--------------------------------------------------------------------------------------------------
#                           Read the Snow obs.

iFiles=glob.glob(sSnowData+"*")

for ff in range(len(iFiles)):
    sFile=iFiles[ff]
    print '    Process '+sFile

    # read in Data
    ii=0
    with open(sFile, "r") as ins:
        array = []
        for line in ins:
            if ii > 0:
                array.append(line)
            else:
                Station=line
            ii=ii+1
    
    # # split string at spaces <-- does not work because of poor data format
    # array=np.array([str.split(array[ii]) for ii in range(len(array))])
    
    # Get data from array
    Year=np.array([array[ii][0:3] for ii in range(len(array))]).astype('int')
    Month=np.array([array[ii][3:6] for ii in range(len(array))]).astype('int')
    Day=np.array([array[ii][6:9] for ii in range(len(array))]).astype('int')
    Hour=np.array([array[ii][9:12] for ii in range(len(array))]).astype('int')
    Extraterrestrial_Horizontal_Radiation=np.array([array[ii][12:17] for ii in range(len(array))]).astype('float')
    Extraterrestrial_Direct_Normal_Radiation=np.array([array[ii][17:22] for ii in range(len(array))]).astype('float')
    Global_Horizontal_Radiation=np.array([array[ii][22:27] for ii in range(len(array))]).astype('float')
    Direct_Normal_Radiation=np.array([array[ii][30:35] for ii in range(len(array))]).astype('float')
    Diffuse_Horizontal_Radiation=np.array([array[ii][38:43] for ii in range(len(array))]).astype('float')
    Total_Sky_Cover=np.array([array[ii][47:49] for ii in range(len(array))]).astype('float')
    Opaque_Sky_Cover=np.array([array[ii][49:52] for ii in range(len(array))]).astype('float')
    Dry_Bulb=np.array([array[ii][52:58] for ii in range(len(array))]).astype('float')
    Dew_Point=np.array([array[ii][58:64] for ii in range(len(array))]).astype('float')
    rRelHum=np.array([array[ii][64:68] for ii in range(len(array))]).astype('float')
    rPressure=np.array([array[ii][68:73] for ii in range(len(array))]).astype('float')
    Wind_Direction=np.array([array[ii][73:77] for ii in range(len(array))])
    Wind_Speed=np.array([array[ii][77:82] for ii in range(len(array))]).astype('float')
    rVisibility=np.array([array[ii][82:88] for ii in range(len(array))]).astype('float')
    Ceiling_Height=np.array([array[ii][88:94] for ii in range(len(array))]).astype('float')
    Present_weather=np.array([array[ii][95:105] for ii in range(len(array))]).astype('str'); Present_weather=netCDF4.stringtochar(np.array(Present_weather, 'S10'))
    Precipitable_Water=np.array([array[ii][105:109] for ii in range(len(array))]).astype('float')
    Broadband_Aerosol_Optical_Depth=np.array([array[ii][109:115] for ii in range(len(array))]).astype('float')
    Snow_Depth=np.array([array[ii][115:119] for ii in range(len(array))]).astype('float')
    Days_Since_Last_Snowfall=np.array([array[ii][119:122] for ii in range(len(array))]).astype('float')
    rPrecip=np.array([array[ii][122:129] for ii in range(len(array))]); rPrecip[(rPrecip == '       ')]=0; rPrecip=rPrecip.astype('float')
    sPRflag=np.array([array[ii][129:130] for ii in range(len(array))]); sPRflag=sPRflag.astype('str'); sPRflag=netCDF4.stringtochar(np.array(sPRflag, 'S1'))

    # # remove invali data
    # Wind_Direction[(Wind_Direction == '9999999.99999.999999')]='999999.'
    
    # Create time vector
    if Year[0] > 20:
        Yadd=1900
    else:
        Yadd=2000
    dStartDay=datetime.datetime(Year[0]+Yadd, Month[0], Day[0],Hour[0]-1)
    dStopDay=datetime.datetime(Year[-1]+Yadd, Month[-1], Day[-1],Hour[-1]-1)
    rgdTimeH = pd.date_range(dStartDay, end=dStopDay, freq='h')
    
    iStartDay=np.where((dStartDay.year == rgdTimeAll.year) & (dStartDay.month == rgdTimeAll.month) & (dStartDay.day == rgdTimeAll.day) & (dStartDay.hour == rgdTimeAll.hour))[0][0]
    rgiTime=iStartDay+np.array(range(len(rgdTimeH)))
    
    
    # ===============================================================
    # Save the data as NetCDF
    
    rgsStationID=np.array(str.split(Station))
    StationID='-'.join([str(x) for x in rgsStationID])
    
    sFileFin=sSaveDataDir+str(dStartDay.year)+'_'+StationID+'.nc'
    sFile=sFileFin+"_COPY"
    
    # ________________________________________________________________________
    # write the netcdf
    print '    ----------------------'
    print '    Save data from '+sFileFin
    root_grp = Dataset(sFile, 'w', format='NETCDF4')
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('rlon', 1)
    root_grp.createDimension('rlat', 1)
    # variables
    lat = root_grp.createVariable('lat', 'f4', ('rlat',))
    lon = root_grp.createVariable('lon', 'f4', ('rlon',))
    time = root_grp.createVariable('time', 'f8', ('time',))
    nchar = root_grp.createDimension('nchar', 10)
    ncharPR = root_grp.createDimension('ncharPR', 1)
    
    ExtraterrestrialHorizontalRadiation = root_grp.createVariable('ExtraterrestrialHorizontalRadiation', 'f4', ('time','rlat',),fill_value=-99999)
    ExtraterrestrialDirectNormalRadiation = root_grp.createVariable('ExtraterrestrialDirectNormalRadiation', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    GlobalHorizontalRadiation = root_grp.createVariable('GlobalHorizontalRadiation', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    DirectNormalRadiation = root_grp.createVariable('DirectNormalRadiation', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    DiffuseHorizontalRadiation = root_grp.createVariable('DiffuseHorizontalRadiation', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    TotalSkyCover = root_grp.createVariable('TotalSkyCover', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    OpaqueSkyCover = root_grp.createVariable('OpaqueSkyCover', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    DryBulb = root_grp.createVariable('DryBulb', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    DewPoint = root_grp.createVariable('DewPoint', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    RelHum = root_grp.createVariable('RelHum', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    Pressure = root_grp.createVariable('Pressure', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    WindDirection = root_grp.createVariable('WindDirection', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    WindSpeed = root_grp.createVariable('WindSpeed', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    Visibility = root_grp.createVariable('Visibility', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    CeilingHeight = root_grp.createVariable('CeilingHeight', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    SnowDepth = root_grp.createVariable('SnowDepth', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    DaysSinceLastSnowfall = root_grp.createVariable('DaysSinceLastSnowfall', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    Precip= root_grp.createVariable('Precip', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    Presentweather = root_grp.createVariable('Presentweather', 'S1', ('time','nchar'))
    # Presentweather= root_grp.createVariable('Presentweather', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    PrecipitableWater= root_grp.createVariable('PrecipitableWater', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    BroadbandAerosolOpticalDepth= root_grp.createVariable('BroadbandAerosolOpticalDepth', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    PRflag = root_grp.createVariable('PRflag', 'S1', ('time','ncharPR'))

    time.calendar = "gregorian"
    time.units = "hours since 1960-1-1 00:00:00"
    time.standard_name = "time"
    time.long_name = "time"
    time.axis = "T"
    
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"
    
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    
    # write data to netcdf
    lat[:]=float(rgsStationID[4][1:])+0.01*(float(rgsStationID[5])/60.)*100.
    lon[:]=float(rgsStationID[6][1:])+0.01*(float(rgsStationID[7])/60.)*100.
    
    ExtraterrestrialHorizontalRadiation[:]=Extraterrestrial_Horizontal_Radiation
    ExtraterrestrialDirectNormalRadiation[:]=Extraterrestrial_Direct_Normal_Radiation
    GlobalHorizontalRadiation[:]=Global_Horizontal_Radiation
    DirectNormalRadiation[:]=Direct_Normal_Radiation
    DiffuseHorizontalRadiation[:]=Diffuse_Horizontal_Radiation
    TotalSkyCover[:]=Total_Sky_Cover
    OpaqueSkyCover[:]=Opaque_Sky_Cover
    DryBulb[:]=Dry_Bulb
    DewPoint[:]=Dew_Point
    RelHum[:]=rRelHum
    Pressure[:]=rPressure
    WindDirection[:]=Wind_Direction
    WindSpeed[:]=Wind_Speed
    Visibility[:]=rVisibility
    CeilingHeight[:]=Ceiling_Height
    SnowDepth[:]=Snow_Depth
    DaysSinceLastSnowfall[:]=Days_Since_Last_Snowfall
    Precip[:]=rPrecip
    Presentweather[:]=Present_weather
    PrecipitableWater[:]=Precipitable_Water
    BroadbandAerosolOpticalDepth[:]=Broadband_Aerosol_Optical_Depth
    PRflag[:]=sPRflag

    time[:]=rgiTime
    root_grp.close()
    
    # compress the netcdf file
    subprocess.Popen("nccopy -k 4 -d 1 -s "+sFile+' '+sFileFin, shell=True)
    import time
    time.sleep(10)
    subprocess.Popen("rm  "+sFile, shell=True)

stop()
