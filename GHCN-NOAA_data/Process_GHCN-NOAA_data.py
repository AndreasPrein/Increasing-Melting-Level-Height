#!/usr/bin/env python
'''
    File name: Process_GHCN-NOAA_data.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 21.05.2017
    Date last modified: 21.05.2017

    ##############################################################
    Purpos:

    Reads in daily GHCN NOAA data from csv files and write the data in to a NetCDF

    The original data is derived from:
    ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/

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
import time
import sys


########################################
sGHCNData='/glade/scratch/prein/Papers/Trends_RadSoundings/data/GHCN-NOAA/ghcnd_all/raw/'
sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/GHCN-NOAA/ghcnd_all/npz/'
sStationFile=sGHCNData+'ghcnd-stations.txt'
variables=['PRCP','SNOW','SNWD','WESF','TMAX','TMIN']
year=int(>>YYYY<<)
# dStartDay=datetime.datetime(1950, 1, 1,0)
# dStopDay=datetime.datetime(2018, 12, 31,23)

dStartDay=datetime.datetime(year, 1, 1,0)
dStopDay=datetime.datetime(year, 12, 31,23)
rgdTimeAll = pd.date_range(dStartDay, end=dStopDay, freq='d')
Years=np.unique(rgdTimeAll.year)
########################################

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#                           Read the Station file
dtype = {'STATION_ID': str,
         'LATITUDE': str,
         'LONGITUDE': str,
         'ELEVATION': str,
         'STATE': str,
         'STATION_NAME': str,
         'GSN_FLAG': str,
         'HCN_CRN_FLAG': str,
         'WMO_ID': str}
names = ['STATION_ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'STATION_NAME', 'GSN_FLAG', 'HCN_CRN_FLAG', 'WMO_ID']
widths = [11,  # Station ID
          9,   # Latitude (decimal degrees)
          10,  # Longitude (decimal degrees)
          7,   # Elevation (meters)
          3,   # State (USA stations only)
          31,  # Station Name
          4,   # GSN Flag
          4,   # HCN/CRN Flag
          6]   # WMO ID
df = pd.read_fwf(sStationFile, widths=widths, names=names, dtype=dtype, header=None)
StationIDs=df['STATION_ID'].values

'''
Replace missing values (nan, -999.9)
'''
df['STATE'] = df['STATE'].replace('nan', '--')
df['GSN_FLAG'] = df['GSN_FLAG'].replace('nan', '---')
df['HCN_CRN_FLAG'] = df['GSN_FLAG'].replace('nan', '---')
df = df.replace(-999.9, float('nan'))

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#                           Read the Raw Data Year by Year
for yy in range(len(Years)):
    savefile=sSaveDataDir+str(Years[yy])+'_GHCN.npz'
    if os.path.isfile(savefile) == 0:
        print '    process year '+str(Years[yy])
        iYearAct=np.sum(rgdTimeAll.year == Years[yy])
        yeardd=rgdTimeAll[(rgdTimeAll.year == Years[yy])]
        days_yyyy=np.array([int(str(yeardd[dd].year)+str("%02d" % yeardd[dd].month)+str("%02d" % yeardd[dd].day)) for dd in range(len(yeardd))])
        rgrGHCNdata=np.zeros((iYearAct,len(StationIDs),len(variables))); rgrGHCNdata[:]=np.nan
        file_yy=sGHCNData+str(Years[yy])+'.csv'
        # read the csv data
        df_act = pd.read_csv(file_yy, names=['StID', 'YYYYMMMDD', 'VAR', 'VALUE', '1', '2','Source','ObsTime'])        
        stations_yyyy=np.unique(df_act['StID'].values)

        # ii_day=np.array([np.where(df_act['YYYYMMMDD'].values[ii] == days_yyyy)[0][0]  for ii in range(len(df_act['YYYYMMMDD'].values))])
        # for st in range(len(stations_yyyy)):
        #     print '         process station '+str(stations_yyyy[st])
        #     ST=(stations_yyyy[st] == StationIDs)
        #     for va in range(len(variables)):
        #         station_var=np.where((df_act['StID'].values == stations_yyyy[st]) & (df_act['VAR'].values == variables[va]))[0]
        #         if len(station_var) > 0:
        #             rgrGHCNdata[ii_day[station_var],ST,va]=df_act['VALUE'].values[station_var]

        # stop()
        ii_day=np.array([np.where(df_act['YYYYMMMDD'].values[ii] == days_yyyy)[0][0]  for ii in range(len(df_act['YYYYMMMDD'].values))])

        toolbar_width = int(len(stations_yyyy)/1000.)
        # # setup toolbar
        # sys.stdout.write("[%s]" % (" " * toolbar_width))
        # sys.stdout.flush()
        # sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        for st in range(len(stations_yyyy)):
            # sys.stdout.write("-")
            # sys.stdout.flush()
            print '         process station '+str(stations_yyyy[st])
            station_act=(df_act['StID'].values == stations_yyyy[st])
            ST=(stations_yyyy[st] == StationIDs)
            if np.sum(station_act) == 0:
                continue
            else:
                data_st_act=df_act['VALUE'][station_act].values
                var_st_act=df_act['VAR'][station_act].values
                date_st_act=df_act['YYYYMMMDD'][station_act].values
            for va in range(len(variables)):
                var_sel=(var_st_act == variables[va])
                if np.sum(var_sel) != 0:
                    rgrGHCNdata[ii_day[station_act][var_sel],ST,va]=data_st_act[var_sel]
                    # stop()
                    # ii_day[station_act][var_sel]
                    # for ii in range(np.sum(var_sel)):
                    #     iDD=(yeardd.month == int(str(date_st_act[var_sel][ii])[4:6])) & (yeardd.day == int(str(date_st_act[var_sel][ii])[6:8]))
                    #     rgrGHCNdata[iDD,ST,va]=data_st_act[var_sel][ii]
        sys.stdout.write("\n")
        np.savez(savefile, 
                 rgrGHCNdata=rgrGHCNdata,
                 yeardd=yeardd,
                 variables=variables,
                 StationIDs=StationIDs,
                 lon=df['LONGITUDE'].values,
                 lat=df['LATITUDE'].values,
                 elevation=df['ELEVATION'].values)
    else:
        print '    load data for year '+str(Years[yy])
