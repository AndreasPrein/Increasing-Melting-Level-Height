#!/usr/bin/env python
'''
    File name: GHCN-dataprocessing.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 21.05.2017
    Date last modified: 21.05.2017

    ##############################################################
    Purpos:

    Reads in annual proprocessed GHNC data from:
    /gpfs/u/home/prein/papers/Trends_RadSoundings/programs/GHCN-NOAA_data/Process_GHCN-NOAA_data.py

    Filters that data according to temporal coverage

    Calculates montly average snowdays, raindays, and sonw/rain day ratio

    Saves the data for future analysis by other programs.

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
sDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/GHCN-NOAA/ghcnd_all/npz/'
SaveDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/GHCN-NOAA/ghcnd_all/'
sStationFile='/glade/scratch/prein/Papers/Trends_RadSoundings/data/GHCN-NOAA/ghcnd_all/raw/ghcnd-stations.txt'
variables=['PRCP','SNOW','SNWD','WESF','TMAX','TMIN']
dStartDay=datetime.datetime(1979, 1, 1,0)
dStopDay=datetime.datetime(2018, 12, 31,23)
rgdTimeAll = pd.date_range(dStartDay, end=dStopDay, freq='d')
Years=np.unique(rgdTimeAll.year)

MinObsRatio=0.7 # minimum ration of observation per month and for the full record
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
#                           Read the Preprocessed Data Year by Year


TargetVariables=['SnowDays','RaynDays','SnowToPrecipDayRatio']
GHCN_monthly_data=np.zeros((len(Years),12,len(StationIDs),len(TargetVariables))); GHCN_monthly_data[:]=np.nan
for yy in range(len(Years)):
    print 'process year '+str(Years[yy])
    savefile=sDataDir+str(Years[yy])+'_GHCN.npz'
    DATA=np.load(savefile)

    rgrGHCNdata=np.array(DATA['rgrGHCNdata'])
    yeardd=rgdTimeAll[rgdTimeAll.year == Years[yy]]

    for mo in range(12):
        monthAct=(yeardd.month == (mo+1))
        SufficientData=(np.sum(~np.isnan(rgrGHCNdata[monthAct,:,:]), axis=0)/float(sum(monthAct)) >= MinObsRatio)
        SufficientData=SufficientData[:,variables.index('SNOW')]*SufficientData[:,variables.index('PRCP')]
        # calculate snow days
        SnowDays=rgrGHCNdata[monthAct,:,variables.index('SNOW')]; SnowDays[SnowDays < 1]=0; SnowDays[SnowDays > 1]=1
        SnowDays[rgrGHCNdata[monthAct,:,variables.index('SNOW')]/(rgrGHCNdata[monthAct,:,variables.index('PRCP')]/10.) <0.5]=0
        GHCN_monthly_data[yy,mo,SufficientData,TargetVariables.index('SnowDays')]=np.nansum(SnowDays[:,SufficientData], axis=0)
        # calculate rain days
        RainDays=rgrGHCNdata[monthAct,:,variables.index('PRCP')]/10.; RainDays[RainDays < 1]=0; RainDays[(RainDays > 1)]=1
        RainDays[rgrGHCNdata[monthAct,:,variables.index('SNOW')]/(rgrGHCNdata[monthAct,:,variables.index('PRCP')]/10.) >=0.5]=0
        GHCN_monthly_data[yy,mo,SufficientData,TargetVariables.index('RaynDays')]=np.nansum(RainDays[:,SufficientData], axis=0)
        # snow to precip day ratio
        PrecipDays=np.nansum(SnowDays, axis=0)+np.nansum(RainDays, axis=0)
        SPDR=(np.nansum(SnowDays, axis=0)/PrecipDays)[SufficientData]
        GHCN_monthly_data[yy,mo,SufficientData,TargetVariables.index('SnowToPrecipDayRatio')]=SPDR

# check if we have the minimum monthly coverage
MonMin=(np.sum(~np.isnan(GHCN_monthly_data), axis=0) <= len(Years)*MinObsRatio)
GHCN_monthly_data[:,MonMin]=np.nan
FinStations=(np.sum(~np.isnan(GHCN_monthly_data), axis=(0,1,3)) != 0)
GHCN_monthly_data=GHCN_monthly_data[:,:,FinStations,:]

np.savez(SaveDir+'GHCN_monthy-snow-data.npz', GHCN_monthly_data=GHCN_monthly_data, lon=df['LONGITUDE'].values[FinStations], lat=df['LATITUDE'].values[FinStations], elevation=df['ELEVATION'].values[FinStations],  StationID=df['STATION_ID'].values[FinStations], StationName=df['STATION_NAME'].values[FinStations], Years=Years, TargetVariables=TargetVariables)
stop()

