#!/usr/bin/env python
'''
    File name: Hail-Risk-Changes.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    # Read in hail observations (location, time, size) from U.S., Australia, and Europe

    Read in ERA-20C and ERA-Interim Data

    Calculate probability of large hail dependent on WBZH

    Calculate annual days with significant severe parameter (CAPExVS0-6) > 20000
    that have too high WBZH levels to support large hail

    Plot trend map and time series plots

'''

from dateutil import rrule
import datetime
from datetime import timedelta
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
from pdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
from mpl_toolkits import basemap
import ESMF
import pickle
import subprocess
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pylab as plt
import random
import scipy.ndimage as ndimage
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
import shapely.geometry
import shapefile
import math
from scipy.stats.kde import gaussian_kde
from math import radians, cos, sin, asin, sqrt
from shapely.geometry import Polygon, Point
from scipy.interpolate import interp1d
import csv
import os.path
import matplotlib.gridspec as gridspec
import matplotlib.path as mplPath
from scipy import stats
from matplotlib.mlab import griddata
from cartopy import config
import cartopy.crs as ccrs
import cartopy
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader
from shutil import copyfile
from matplotlib import path
from mpl_toolkits.basemap import Basemap; bm = Basemap()
from scipy.interpolate import interp1d
import scipy
from mpl_toolkits.basemap import Basemap; bm = Basemap()


from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
def grid(x, y, z, resX=20, resY=20):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


def running_mean(l, N):
    sum = 0
    result = list( 0 for x in l)
    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)
    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N
    return result

########################################
#                            Settings
rgsVars=['WBZheight', 'SSP'] #['WBZheight','ThreeCheight','ProbSnow','BflLR']
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
DataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Monthly_Data/'
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
rgsSeasons=['DJF','MAM','JJA','SON','Annual']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30,365.25]
sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/'

rgsData=['ERA-Interim','ERA-20C']

dStartDay_MSWEP=datetime.datetime(1979, 1, 1,0)
dStopDay_MSWEP=datetime.datetime(2017, 12, 31,0)
rgdTimeDD_MSWEP = pd.date_range(dStartDay_MSWEP, end=dStopDay_MSWEP, freq='d')
rgdTimeFin=rgdTimeDD_MSWEP
iYears=np.unique(rgdTimeFin.year)

########################################
# read in the subregions
rgsSubregions=['Africa','Asia','North-America','South-America','Australia','Europe']
sContinents='/glade/u/home/prein/ShapeFiles/Continents/'
grShapeFiles={}
for sr in range(len(rgsSubregions)):
    ctr = shapefile.Reader(sContinents+rgsSubregions[sr]+'_poly')
    geomet = ctr.shapeRecords() # will store the geometry separately
    for sh in range(len(geomet)):
        first = geomet[sh]
        grShapeFiles[rgsSubregions[sr]]=first.shape.points


########################################
#      READ IN RADIO SOUNDING DATA
DATA=np.load(DataDir+'RS-data-montly_19002017.npz')
VARIABLES=DATA['VARS']; VARIABLES=[VARIABLES[va] for va in range(len(VARIABLES))]
RS_data=DATA['Monthly_RO']
RS_lat=DATA['RS_Lat']
RS_lon=DATA['RS_Lon']
RS_time=DATA['RSTimeMM']
RS_Breaks=DATA['BreakFlag']
rgrLonS=RS_lon; rgrLatS=RS_lat

#      READ IN ERA-I DATA
DATA=np.load(DataDir+'EI-data-montly_19792017.npz_SSP.npz')
EI_data=DATA['rgrMonthlyData_EI']
EI_lat=DATA['rgrLat75_EI']
EI_lon=DATA['rgrLon75_EI']
EI_time=DATA['rgdTimeMM_EI']
EI_Breaks=DATA['BreakFlag']
rgiYY_EI=np.unique(pd.DatetimeIndex(EI_time).year)
rgrLonGrid2D_EI=np.asarray(([EI_lon,]*EI_lat.shape[0]))
rgrLatGrid2D_EI=np.asarray(([EI_lat,]*EI_lon.shape[0])).transpose()

#      READ IN ERA-20 DATA
DATA=np.load(DataDir+'E20-data-montly_19002010.npz')
E20_data=DATA['rgrMonthlyData_E20']
E20_lat=DATA['rgrLat75_E20']
E20_lon=DATA['rgrLon75_E20']
E20_time=DATA['rgdTimeMM_E20']
E20_Breaks=DATA['BreakFlag']
rgiYY=np.unique(pd.DatetimeIndex(E20_time).year)
rgrLonGrid2D=np.asarray(([E20_lon,]*E20_lat.shape[0]))    
rgrLatGrid2D=np.asarray(([E20_lat,]*E20_lon.shape[0])).transpose()


########################################
#      EXTRACT THE NECESSARY VARIABLES
RS_SSP=RS_data[:,:,:,VARIABLES.index('SSP')]
RS_SSP[:,:,(np.max(RS_Breaks[:,:,VARIABLES.index('SSP')], axis=0) > 0.5)]=np.nan; RS_SSP[:,:,np.sum(~np.isnan(RS_SSP), axis=(0,1)) < (RS_SSP.shape[0]*RS_SSP.shape[1]*0.8)]=np.nan
RS_SSP_ML=RS_data[:,:,:,VARIABLES.index('SSP_ML')]
RS_SSP_ML[:,:,(np.max(RS_Breaks[:,:,VARIABLES.index('SSP')], axis=0) > 0.5)]=np.nan; RS_SSP_ML[:,:,np.sum(~np.isnan(RS_SSP_ML), axis=(0,1)) < (RS_SSP_ML.shape[0]*RS_SSP_ML.shape[1]*0.8)]=np.nan
# RS_WCLD=RS_data[:,:,:,VARIABLES.index('SSP')]
# RS_WCLD[:,:,(np.max(RS_Breaks[:,:,VARIABLES.index('SSP')], axis=0) > 0.5)]=np.nan; RS_WCLD[:,:,np.sum(~np.isnan(RS_WCLD), axis=(0,1)) < (RS_WCLD.shape[0]*RS_WCLD.shape[1]*0.8)]=np.nan

EI_SSP=EI_data[:,:,:,:,VARIABLES.index('SSP')]
EI_SSP[:,:,(np.max(EI_Breaks[:,:,:,VARIABLES.index('SSP')], axis=0) > 0.5)]=np.nan; EI_SSP[:,:,np.sum(~np.isnan(EI_SSP), axis=(0,1)) < (EI_SSP.shape[0]*EI_SSP.shape[1]*0.8)]=np.nan
EI_SSP_ML=EI_data[:,:,:,:,VARIABLES.index('SSP_ML')]
EI_SSP_ML[:,:,(np.max(EI_Breaks[:,:,:,VARIABLES.index('SSP_ML')], axis=0) > 0.5)]=np.nan; EI_SSP_ML[:,:,np.sum(~np.isnan(EI_SSP_ML), axis=(0,1)) < (EI_SSP_ML.shape[0]*EI_SSP_ML.shape[1]*0.8)]=np.nan
# EI_WCLD=EI_data[:,:,:,:,VARIABLES.index('SSP')]
# EI_WCLD[:,:,(np.max(EI_Breaks[:,:,:,VARIABLES.index('SSP')], axis=0) > 0.5)]=np.nan; EI_WCLD[:,:,np.sum(~np.isnan(EI_WCLD), axis=(0,1)) < (EI_WCLD.shape[0]*EI_WCLD.shape[1]*0.8)]=np.nan

E20_SSP=E20_data[:,:,:,:,VARIABLES.index('SSP')]
E20_SSP[:,:,(np.max(E20_Breaks[:,:,:,VARIABLES.index('SSP')], axis=0) > 0.5)]=np.nan; E20_SSP[:,:,np.sum(~np.isnan(E20_SSP), axis=(0,1)) < (E20_SSP.shape[0]*E20_SSP.shape[1]*0.8)]=np.nan
E20_SSP_ML=E20_data[:,:,:,:,VARIABLES.index('SSP_ML')]
E20_SSP_ML[:,:,(np.max(E20_Breaks[:,:,:,VARIABLES.index('SSP_ML')], axis=0) > 0.5)]=np.nan; E20_SSP_ML[:,:,np.sum(~np.isnan(E20_SSP_ML), axis=(0,1)) < (E20_SSP_ML.shape[0]*E20_SSP_ML.shape[1]*0.8)]=np.nan
# E20_WCLD=E20_data[:,:,:,:,VARIABLES.index('SSP')]
# E20_WCLD[:,:,(np.max(E20_Breaks[:,:,:,VARIABLES.index('SSP')], axis=0) > 0.5)]=np.nan; E20_WCLD[:,:,np.sum(~np.isnan(E20_WCLD), axis=(0,1)) < (E20_WCLD.shape[0]*E20_WCLD.shape[1]*0.8)]=np.nan


# stop()
# rgiLandOcean=np.load('LandOcean_E20.npy'); rgiLandOcean=np.reshape(rgiLandOcean, (160,320))
# rgrSeasonalData=np.nansum(E20_SSP_ML[:,:,:,:][(np.array(range(1900,2011)) >=1979),:],axis=1)/np.nansum(E20_SSP[:,:,:,:][(np.array(range(1900,2011)) >=1979),:],axis=1)
# stats.linregress(range(rgrSeasonalData.shape[0]),np.nanmean(rgrSeasonalData[:,rgiLandOcean], axis=1))



########################################
# read in the hail observations

# NOAA
sHailstorms='/glade/u/home/prein/projects/Hail/data/NOAA-HailStorms-Gridded-1979-2015_GE1.0in.npz'
print '    Load: '+sHailstorms
data = np.load(sHailstorms)
rgrHailJulTime_NOAA=data['rgrHailJulTimeHS']
rgdTime_NOAA=data['rgrHailTimeHS']
rgrHailLat_NOAA=data['rgrHailLatHS']
rgrHailLon_NOAA=data['rgrHailLonHS']
rgrHailSize_NOAA=data['rgrHailSizeHS']

rgiYears=np.unique([rgdTime_NOAA[ii].year for ii in range(len(rgdTime_NOAA))])
rgiYearsHail=np.array([rgdTime_NOAA[ii].year for ii in range(len(rgdTime_NOAA))])
rgiMonthsHail=np.array([rgdTime_NOAA[ii].month for ii in range(len(rgdTime_NOAA))])
rgiDaysHail=np.array([rgdTime_NOAA[ii].day for ii in range(len(rgdTime_NOAA))])

# BOM
sHailBOM='/glade/u/home/prein/papers/HailModel/data/BOM-Hail/BOM-HailStorms-Gridded-1979-2016_GE2.5-mm.npz'
data = np.load(sHailBOM)
rgrHailJulTime_BOM=data['rgrHailJulTimeHS']
rgdTime_BOM=data['rgrHailTimeHS']
rgrHailLat_BOM=data['rgrHailLatHS']
rgrHailLon_BOM=data['rgrHailLonHS']
rgrHailSize_BOM=data['rgrHailSizeHS']

# ESSL
sHailESSL="/glade/work/prein/observations/ESSL-Hail/ESSL-HailStorms-Gridded-1979-2016_GE2.5-mm.npz"
data = np.load(sHailESSL)
rgrHailJulTime_ESSL=data['rgrHailJulTimeHS']
rgdTime_ESSL=data['rgrHailTimeHS']
rgrHailLat_ESSL=data['rgrHailLatHS']
rgrHailLon_ESSL=data['rgrHailLonHS']
rgrHailSize_ESSL=data['rgrHailSizeHS']

# Append the records
rgrHailJulTime=np.append(np.append(rgrHailJulTime_NOAA,rgrHailJulTime_BOM),rgrHailJulTime_ESSL)
rgdTime=np.append(np.append(rgdTime_NOAA,rgdTime_BOM),rgdTime_ESSL)
rgrHailLat=np.append(np.append(rgrHailLat_NOAA,rgrHailLat_BOM),rgrHailLat_ESSL)
rgrHailLon=np.append(np.append(rgrHailLon_NOAA,rgrHailLon_BOM),rgrHailLon_ESSL); rgrHailLon[rgrHailLon < 0]=rgrHailLon[rgrHailLon < 0]+360
rgrHailSize=np.append(np.append(rgrHailSize_NOAA,rgrHailSize_BOM),rgrHailSize_ESSL)


# ================================================================
# GET WBZH AT LOCATIONS WITH HAIL RECORDS ERA-20C

dStartDay=datetime.datetime(1979, 1, 1,0)
dStopDay=datetime.datetime(2010, 12, 31,23)
# dStopDay=datetime.datetime(2010, 12, 31,23)
rgdTime6H = pd.date_range(dStartDay, end=dStopDay, freq='3h')
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgiYY=np.unique(rgdTime6H.year)
rgiERA_hours=np.array([0,6,12,18])
rgdFullTime=pd.date_range(datetime.datetime(1900, 1, 1,00),
                        end=datetime.datetime(2010, 12, 31,23), freq='3h')
sERAconstantFields='/glade/work/prein/reanalyses/ERA-20c/e20c.oper.invariant.128_129_z.regn80sc.1900010100_2010123121.nc'
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat75=np.squeeze(ncid.variables['g4_lat_0'][:])
rgrLon75=np.squeeze(ncid.variables['g4_lon_1'][:])
rgrHeight=(np.squeeze(ncid.variables['Z_GDS4_SFC'][:]))/9.81
ncid.close()
rgrLonGrid2D=np.asarray(([rgrLon75,]*rgrLat75.shape[0]))
rgrLatGrid2D=np.asarray(([rgrLat75,]*rgrLon75.shape[0])).transpose()

# rgrDailyData=np.zeros((len(rgdTimeDD),len(rgrLat75), len(rgrLon75))); rgrDailyData[:]=np.nan
# tt=0
# for yy in range(len(rgiYY)): #range(len(TC_Name)): # loop over hurricanes
#     print ' '
#     print '    Workin on year '+str(rgiYY[yy])
#     # check if we already processed this file
#     for mm in range(12): #  loop over time
#         rgrTempData=np.zeros((31*8,len(rgrLat75), len(rgrLon75))); rgrTempData[:]=np.nan
#         for dd in range(2):
#             YYYY=str(rgiYY[yy])
#             MM=str("%02d" % (mm+1))
#             if dd == 0:
#                 DD1='0100'
#                 DD2='1521'
#             else:
#                 DD1='1600'
#                 iDaysInMon=rgdFullTime[((rgdFullTime.year == int(YYYY)) & (rgdFullTime.month == int(MM)))][-1].day
#                 DD2=str(iDaysInMon)+'21'
#             if YYYY != 2010:
#                 sDecade=str(YYYY)[:3]+'0_'+str(YYYY)[:3]+'9'
#             else:
#                 sDecade=str(YYYY)[:3]+'0_'+str(YYYY)[:3]+'0'
#             sDate=YYYY+MM+DD1+'_'+YYYY+MM+DD2
#             sFileFin=sSaveDataDir+'ERA-20c/'+sDate+'_ERA-20c_Hail-Env_RDA.nc'
#             ncid=Dataset(sFileFin, mode='r')
#             rgrDataWBZ=np.squeeze(ncid.variables['WBZheight'][:])
#             ncid.close()
#             rgrDataWBZ=np.mean(np.reshape(rgrDataWBZ, (rgrDataWBZ.shape[0]/8,8,rgrDataWBZ.shape[1],rgrDataWBZ.shape[2])),axis=1)
#             try:
#                 rgrDailyData[tt:tt+rgrDataWBZ.shape[0],:,:]=rgrDataWBZ
#             except:
#                 stop()
#             tt=int(tt+rgrDataWBZ.shape[0])

# # find WBZH at hail locations
# rgiLonMin=np.array([np.argmin(np.abs(rgrHailLon[ii] - rgrLon75)) for ii in range(len(rgrHailLon))])
# rgiLatMin=np.array([np.argmin(np.abs(rgrHailLat[ii] - rgrLat75)) for ii in range(len(rgrHailLat))])
# rgiTimeMin=np.zeros((len(rgdTime))); rgiTimeMin[:]=np.nan
# for dd in range(len(rgdTime)):
#     try:
#         rgiTimeMin[dd]=int(np.where((rgdTime[dd].year == rgdTimeDD.year) & (rgdTime[dd].month == rgdTimeDD.month) & (rgdTime[dd].day == rgdTimeDD.day))[0][0])
#     except:
#         continue

# rgiHail_WBZ_E20=np.zeros((len(rgrHailLon))); rgiHail_WBZ_E20[:]=np.nan
# for st in range(len(rgrHailLon)):
#     try:
#         rgiHail_WBZ_E20[st]=rgrDailyData[int(rgiTimeMin[st]),rgiLatMin[st],rgiLonMin[st]]
#     except:
#         continue

# np.save(sSaveDataDir+'ERA20_WBZH_1979-2009.npy',rgiHail_WBZ_E20)
rgiHail_WBZ_E20=np.load(sSaveDataDir+'ERA20_WBZH_1979-2009.npy')

# ================================================================
# GET WBZH AT LOCATIONS WITH HAIL RECORDS ERA-Interim

dStartDay_EI=datetime.datetime(1979, 1, 1,0)
dStopDay_EI=datetime.datetime(2017, 12, 31,23)
# dStopDay=datetime.datetime(2010, 12, 31,23)
rgdTime6H_EI = pd.date_range(dStartDay_EI, end=dStopDay_EI, freq='3h')
rgdTimeDD_EI = pd.date_range(dStartDay_EI, end=dStopDay_EI, freq='d')
rgdTimeMM_EI = pd.date_range(dStartDay_EI, end=dStopDay_EI, freq='m')
rgiYY_EI=np.unique(rgdTime6H_EI.year)
rgiERA_hours=np.array([0,6,12,18])
rgdFullTime_EI=pd.date_range(datetime.datetime(1900, 1, 1,00),
                        end=datetime.datetime(2017, 12, 31,23), freq='3h')
# # first read the coordinates
sERA_IconstantFields='/glade/work/prein/reanalyses/ERA-Interim/ERA-Interim-FullRes_Invariat.nc'
# read the ERA-Interim elevation
ncid=Dataset(sERA_IconstantFields, mode='r')
rgrLat75_EI=np.squeeze(ncid.variables['g4_lat_0'][:])
rgrLon75_EI=np.squeeze(ncid.variables['g4_lon_1'][:])
rgrHeight_EI=(np.squeeze(ncid.variables['Z_GDS4_SFC'][:]))/9.81
ncid.close()
rgrLonGrid2D_EI=np.asarray(([rgrLon75_EI,]*rgrLat75_EI.shape[0]))
rgrLatGrid2D_EI=np.asarray(([rgrLat75_EI,]*rgrLon75_EI.shape[0])).transpose()

# rgrdailydata_EI=np.zeros((len(rgdTimeDD_EI),len(rgrLat75_EI), len(rgrLon75_EI))); rgrDailyData_EI[:]=np.nan
# tt=0
# for yy in range(len(rgiYY_EI)): #range(len(TC_Name)): # loop over hurricanes
#     print ' '
#     print '    Workin on year '+str(rgiYY_EI[yy])
#     # check if we already processed this file
#     for mm in range(12): #  loop over time
#         rgiDD_EI=np.sum(((rgdTimeDD_EI.year == rgiYY_EI[yy]) & (rgdTimeDD_EI.month == (mm+1))))
#         YYYY=str(rgiYY_EI[yy])
#         MM=str("%02d" % (mm+1))
#         for dd in range(rgiDD_EI):
#             DD=str("%02d" % (dd+1))
#             sDate=YYYY+MM+DD
#             ncid=Dataset(sSaveDataDir+'ERA-Int/'+sDate+'_ERA-Int_Hail-Env_RDA.nc', mode='r')
#             rgrWBZ_EI=np.mean(np.squeeze(ncid.variables['WBZheight'][:]),axis=0)
#             ncid.close()
#             rgrDailyData_EI[tt,:,:]=rgrWBZ_EI
#             tt=tt+1

# # find WBZH at hail locations
# rgiLonMin=np.array([np.argmin(np.abs(rgrHailLon[ii] - rgrLon75_EI)) for ii in range(len(rgrHailLon))])
# rgiLatMin=np.array([np.argmin(np.abs(rgrHailLat[ii] - rgrLat75_EI)) for ii in range(len(rgrHailLat))])
# rgiTimeMin=np.zeros((len(rgdTime))); rgiTimeMin[:]=np.nan
# for dd in range(len(rgdTime)):
#     try:
#         rgiTimeMin[dd]=int(np.where((rgdTime[dd].year == rgdTimeDD_EI.year) & (rgdTime[dd].month == rgdTimeDD_EI.month) & (rgdTime[dd].day == rgdTimeDD_EI.day))[0][0])
#     except:
#         continue

# rgiHail_WBZ_EI=np.zeros((len(rgrHailLon))); rgiHail_WBZ_EI[:]=np.nan
# for st in range(len(rgrHailLon)):
#     try:
#         rgiHail_WBZ_EI[st]=rgrDailyData_EI[int(rgiTimeMin[st]),rgiLatMin[st],rgiLonMin[st]]
#     except:
#         continue

# np.save(sSaveDataDir+'ERAI_WBZH_1979-2016.npy',rgiHail_WBZ_EI)
rgiHail_WBZ_EI=np.load(sSaveDataDir+'ERAI_WBZH_1979-2016.npy')


########################################
#                      start plotting
fig = plt.figure(figsize=(7, 15))
FontSize=15
plt.rcParams.update({'font.size': FontSize})
# ==============================================================================
# ==============================================================================
# Add hail sizen vs. ssp plot
gs1 = gridspec.GridSpec(1,1)
gs1.update(left=0.12, right=0.99,
           bottom=0.705, top=0.98,
           wspace=0.3, hspace=0.3)

rgiDataReal_E20=((~np.isnan(rgrHailSize)) & (~np.isnan(rgiHail_WBZ_E20)))
rgiDataReal_EI=((~np.isnan(rgrHailSize)) & (~np.isnan(rgiHail_WBZ_EI)))

ax = plt.subplot(gs1[0,0])
iSmooth=300.
rgrXaxis=np.arange(0,6100,100)
rgrHailSize_E20=np.zeros((len(rgrXaxis))); rgrHailSize_E20[:]=np.nan
rgrHailSize_EI=np.zeros((len(rgrXaxis))); rgrHailSize_EI[:]=np.nan
for jj in range(len(rgrXaxis)):
    rgiBinE20=((rgiHail_WBZ_E20 > rgrXaxis[jj]-iSmooth/2) & (rgiHail_WBZ_E20 <= rgrXaxis[jj]+iSmooth/2))
    rgiBinEI=((rgiHail_WBZ_EI > rgrXaxis[jj]-iSmooth/2) & (rgiHail_WBZ_EI <= rgrXaxis[jj]+iSmooth/2))
    if sum(rgiBinE20) >= 200:
        rgrDataAct=np.sort(rgrHailSize[rgiBinE20])[-int(sum(rgiBinE20)*0.01):]
        rgrHailSize_E20[jj]=np.nanmean(rgrDataAct)
    if sum(rgiBinEI) >= 200:
        rgrDataAct=np.sort(rgrHailSize[rgiBinEI])[-int(sum(rgiBinEI)*0.01):]
        rgrHailSize_EI[jj]=np.nanmean(rgrDataAct)

ax.scatter(rgiHail_WBZ_E20/1000., rgrHailSize*25.4, s=50, color='#636363', alpha=0.3, edgecolors='none')
ax.set_ylabel('hail diameter [mm]')
ax.set_xlabel('melting level height [km]')
ax.set(xlim=(0.5,5.5),ylim=(50,250))

ax.plot(rgrXaxis/1000.,rgrHailSize_E20[:]*25.4, c='#1f78b4', label='ERA-20C', lw=2)
ax.plot(rgrXaxis/1000.,rgrHailSize_EI[:]*25.4, c='#e31a1c', label='ERA-Interim', lw=2)
ax.legend(loc="upper right",\
          ncol=1, prop={'size':15})
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.title(rgsLableABC[0]+') ')



###############################################################
print '    Add ERA-20C analysis'
gs1 = gridspec.GridSpec(1,1)
gs1.update(left=0.02, right=0.98,
           bottom=0.40, top=0.65,
           wspace=0.2, hspace=0.1)
ax = plt.subplot(gs1[0,0], projection=cartopy.crs.Robinson())

biasContDist=20

rgrColorTable=np.array(['#4457c9','#4f67d4','#6988ea','#84a6f9','#9ebdff','#b7d0fd','#cfdcf3','#f0f0f0','#f1d7c8','#f9c5ab','#f7aa8c','#f28d72','#df6553','#c93137','#bc052b'])
iContNr=len(rgrColorTable)
iMinMax=12.*biasContDist/2.+biasContDist/2.
clevsTD=np.arange(-iMinMax,iMinMax+0.0001,biasContDist)/10.
iScale=1

# rgrMonthAct=E20_WCLD
rgrLonAct=rgrLonGrid2D
rgrLatAct=rgrLatGrid2D
rgiYYE20=np.array(range(1900,2011))
rgrTrendsERA20=np.zeros((E20_SSP.shape[2],E20_SSP.shape[3],5)); rgrTrendsERA20[:]=np.nan
rgrSeasonalData=np.nansum(E20_SSP_ML[:,:,:,:][(rgiYYE20 >=1979),:],axis=1)/np.nansum(E20_SSP[:,:,:,:][(rgiYYE20 >=1979),:],axis=1)
for la in range(rgrSeasonalData.shape[1]):
    for lo in range(rgrSeasonalData.shape[2]):
        rgrTrendsERA20[la,lo,:]=stats.linregress(rgiYYE20[rgiYYE20 >=1979],rgrSeasonalData[:,la,lo])
rgrTrendsERA20[((np.nansum(rgrSeasonalData[:,:,:], axis=0)) ==0),0]=np.nan
# ERAI
# rgrMonthAct=EI_WCLD
rgrLonAct=rgrLonGrid2D_EI
rgrLatAct=rgrLatGrid2D_EI
rgrTrendsERAI=np.zeros((EI_SSP.shape[2],EI_SSP.shape[3],5)); rgrTrendsERAI[:]=np.nan
rgrSeasonalDataEI=np.nansum(EI_SSP_ML[:,:,:,:][((rgiYY_EI >=1979) & (rgiYY_EI <= 2010)),:],axis=1)/np.nansum(EI_SSP[:,:,:,:][((rgiYY_EI >=1979) & (rgiYY_EI <= 2010)),:],axis=1)
for la in range(rgrSeasonalDataEI.shape[1]):
    for lo in range(rgrSeasonalDataEI.shape[2]):
        rgrTrendsERAI[la,lo,:]=stats.linregress(rgiYY_EI[(rgiYY_EI >=1979) & (rgiYY_EI <= 2010)],rgrSeasonalDataEI[:,la,lo])
ERAi_on_ERA20_T=scipy.interpolate.griddata((rgrLonGrid2D_EI.flatten(),rgrLatGrid2D_EI.flatten()),rgrTrendsERAI[:,:,0].flatten() , (rgrLonGrid2D,rgrLatGrid2D),method='linear')
ERAi_on_ERA20_P=scipy.interpolate.griddata((rgrLonGrid2D_EI.flatten(),rgrLatGrid2D_EI.flatten()),rgrTrendsERAI[:,:,3].flatten() , (rgrLonGrid2D,rgrLatGrid2D),method='linear')

# stop()
# RelTrend_E20=(rgrTrendsERA20[:,:,0]*10.)/np.nanmean(rgrSeasonalData[:,:,:], axis=0)
# RelTrend_E20[np.nanmean(rgrSeasonalData[:,:,:], axis=0) < 0.1]=np.nan
# RelTrend_EI=(rgrTrendsERAI[:,:,0]*10.)/np.nanmean(rgrSeasonalDataEI[:,:,:], axis=0)
# RelTrend_EI[np.nanmean(rgrSeasonalDataEI[:,:,:], axis=0) < 0.1]=np.nan

AVERAGE_Trend=np.nanmean([ERAi_on_ERA20_T,rgrTrendsERA20[:,:,0]], axis=0)
# plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,colors=rgrColorTable, extend='both',levels=clevsTD) #, alpha=0.6)
# cs=plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,cmap=plt.cm.coolwarm, extend='both',levels=clevsTD, zorder=0) #, alpha=0.6)
cs=plt.contourf(rgrLonGrid2D,rgrLatGrid2D,AVERAGE_Trend*10*100*iScale,colors=rgrColorTable, extend='both',levels=clevsTD, zorder=0,transform=cartopy.crs.PlateCarree()) #, alpha=0.6)
# # grey out areas with broken or missing records
NAN=np.array(np.isnan(AVERAGE_Trend).astype('float')); NAN[NAN != 1]=np.nan
plt.contourf(rgrLonGrid2D,rgrLatGrid2D,NAN,levels=[0,1,2],colors=['#bdbdbd','#bdbdbd','#bdbdbd'], zorder=1,transform=cartopy.crs.PlateCarree())

# plot significant layer
# for la in range(rgrSeasonalData.shape[1])[::2]:
#     for lo in range(rgrSeasonalData.shape[2])[::2]:
#         if (rgrTrendsERA20[la,lo,3] <= 0.05) & (ERAi_on_ERA20_P[la,lo] <=0.05) & (rgrTrendsERA20[la,lo,0]*ERAi_on_ERA20_T[la,lo] > 0):
#             ax.plot(rgrLonGrid2D[la,lo], rgrLatGrid2D[la,lo],'o',color='k', ms=2, mec='k', alpha=0.5, markeredgewidth=0.0,transform=cartopy.crs.PlateCarree(), zorder=15)

SIG=((rgrTrendsERA20[:,:,3] <= 0.05) & (rgrTrendsERA20[:,:,0]*ERAi_on_ERA20_T[:,:] > 0)).astype('int')
csH=plt.contourf(rgrLonGrid2D[:,:], rgrLatGrid2D[:,:],SIG, hatches=['','///'],levels=[0,0.9,1], zorder=1, alpha=0 ,transform=cartopy.crs.PlateCarree())
SIG=((ERAi_on_ERA20_P[:,:] <=0.05) & (rgrTrendsERA20[:,:,0]*ERAi_on_ERA20_T[:,:] > 0)).astype('int')
csH=plt.contourf(rgrLonGrid2D[:,:], rgrLatGrid2D[:,:],SIG, hatches=['',"\\\ "],levels=[0,0.9,1], zorder=1, alpha=0 ,transform=cartopy.crs.PlateCarree())


ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5, zorder=11)
ax.add_feature(cartopy.feature.COASTLINE, zorder=11)
ax.add_feature(cartopy.feature.STATES, zorder=11, alpha=.5)
# ax.gridlines(zorder=11)
plt.axis('off')
    
print '    add colorbar'
CbarAx = axes([0.05, 0.37, 0.9, 0.015])
cb = colorbar(cs, cax = CbarAx, orientation='horizontal', extend='max', ticks=clevsTD, extendfrac='auto')
cb.ax.set_title('trend of severe convection days with high MLs [% per decade]',size=14)

# # add 5 days per year contour
# rgrPsnowClim=np.mean(rgrMonthAct[(rgiYY >=1979),:,:,:,va], axis=(0,1))
# ax.contour(rgrLonAct,rgrLatAct,rgrPsnowClim*iScale,levels=[10], zorder=20,transform=cartopy.crs.PlateCarree())
# rgrPsnowClim=np.mean(rgrMonthAct[(rgiYY <=1930),:,:,:,va], axis=(0,1))
# ax.contour(rgrLonAct,rgrLatAct,rgrPsnowClim*iScale,levels=[10], zorder=20,transform=cartopy.crs.PlateCarree())

# lable the maps
tt = ax.text(0.03,0.99, rgsLableABC[1]+') ' , ha='left',va='top', \
             transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=15)
tt.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='w'))



###################################
# Radiosoundings
rgrGCtrend=np.zeros((len(rgrLonS))); rgrGCtrend[:]=np.nan
rgrPvalue=np.copy(rgrGCtrend)
rgrR2=np.copy(rgrGCtrend)
# get the data
# rgrDataAct=rgrSSP_RSmonth[:,:,:]
rgrDataAnnual=np.nansum(RS_SSP_ML[:,:], axis=1)/np.nansum(RS_SSP[:,:], axis=1)# np.zeros((len(iYears),rgrDataAct.shape[1])); rgrDataAnnual[:]=np.nan
# for yy in range(len(iYears)):
#     rgiYYact=(rgdTimeFin[rgiTime].year == iYears[yy])
#     rgrDataAnnual[yy,:]=np.nanmean(rgrDataAct[rgiYYact,:], axis=0)
#     # remove data with too little coverage
#     iCoverage=np.sum(np.isnan(rgrDataAct[rgiYYact,:]),axis=0)/float(sum(rgiYYact))
#     rgrDataAnnual[yy,(iCoverage> 1.-iMinCov)]=np.nan

for st in range(len(rgrLonS)):
    # remove nans if there are some
    rgiReal=~np.isnan(rgrDataAnnual[(iYears <= 2010),st])
    if (np.mean(rgrDataAnnual[(iYears <= 2010),st][rgiReal]*iScale) < 10) | (sum(rgiReal) < (2010-1979)*0.8):
        continue
    try:
        print st
        slope, intercept, r_value, p_value, std_err = stats.linregress(iYears[(iYears <= 2010)][rgiReal],rgrDataAnnual[(iYears <= 2010),st][rgiReal])
    except:
        continue
    rgrGCtrend[st]=slope*10*100*iScale #(slope/np.nanmean(rgrDataAnnual[:,st]))*1000.
    rgrPvalue[st]=p_value
    rgrR2[st]=r_value**2

# plot circles that show the trend in area
iMS=4
if np.min(rgrLonS) < 0:
    rgrLonS[rgrLonS < 0]=rgrLonS[rgrLonS < 0]+360
for st in range(len(rgrLonS)):
    if np.isnan(rgrGCtrend[st]) != 1:
        try:
            iColor=np.where((clevsTD-rgrGCtrend[st]) > 0)[0][0]-1
        except:
            if rgrGCtrend[st] > 0:
                iColor=len(clevsTD)-1
        if iColor == -1:
            iColor=0
        if (rgrPvalue[st] <= 0.05):
            ax.plot(rgrLonS[st], rgrLatS[st],'o',color=rgrColorTable[iColor], ms=iMS, mec='k', alpha=1, markeredgewidth=1, transform=cartopy.crs.PlateCarree(), zorder=20)
        else:
            # not significant trend
            try:
                ax.plot(rgrLonS[st], rgrLatS[st],'o',color=rgrColorTable[iColor], ms=iMS, mec=rgrColorTable[iColor], alpha=1, markeredgewidth=0.0,transform=cartopy.crs.PlateCarree(), zorder=20)
            except:
                stop()




# ==============================================================================
# ==============================================================================
# Add time lines for subregions

rgrT2Mmonth=np.load('ERA-20C_T2M.npy')
rgsVars=rgsVars+['T2M']
Yminmax=[-12,7]
# rgrMonthlyData=np.append(rgrMonthlyData[:,:,:,:,None],rgrT2Mmonth[:,:,:,:,None], axis=4)


gs1 = gridspec.GridSpec(1,1)
gs1.update(left=0.19, right=0.86,
           bottom=0.05, top=0.3,
           wspace=0.3, hspace=0.3)

rgsSubregions=['Africa','Asia','North-America','South-America','Australia','Europe']
sContinents='/glade/u/home/prein/ShapeFiles/Continents/'
grShapeFiles={}
for sr in range(len(rgsSubregions)):
    ctr = shapefile.Reader(sContinents+rgsSubregions[sr]+'_poly')
    geomet = ctr.shapeRecords() # will store the geometry separately
    for sh in range(len(geomet)):
        first = geomet[sh]
        grShapeFiles[rgsSubregions[sr]]=first.shape.points

biasContDist=20
sLabel='$\Delta$ ratio of severe convection days\nwith high MLs'
sUnit='[% per year]'
# rgrDataVar=E20_WCLD
# rgrDataVarEI=EI_WCLD
# rgrROdata=RS_WCLD

# rgiFin=np.sum(~np.isnan(rgrROdata[:,:,:]), axis=(0,1))
# rgiFin=(rgiFin >= 3*26)
# rgrROdataAct=rgrROdata[:,:,:]# [:,:,rgiFin]
rgrLonAct=rgrLonS# [rgiFin]
rgrLatAct=rgrLatS# [rgiFin]
# rgrDataVarEIAct=rgrDataVarEI[:,:,:]
# rgrDataVarAct=rgrDataVar[:,:,:]
rgrDataVarActT2M=rgrT2Mmonth #rgrMonthlyData[:,:,:,:,1]

# if (rgsVars[va] == 'ProbSnow') | (rgsVars[va] == 'Snow_PRdays') | (rgsVars[va] == 'Rain_PRdays'):
#     # if se == 0:
#     rgrDataVarAct=rgrDataVarAct*365.25
#     rgrDataVarEIAct=rgrDataVarEIAct*365.25
#     rgrROdataAct=rgrROdataAct*365.25


ii=0
# iYrange=[7,7,7]
rgsSubregions=['Global','Global Land','Global Ocean']
for sr in [1]: #range(len(rgsSubregions)):
    ax = plt.subplot(gs1[0,0])
    # rgdShapeAct=grShapeFiles[rgsSubregions[sr]]
    # PATH = path.Path(rgdShapeAct)
    for da in range(len(rgsData)):
        if rgsData[da] == 'Rad-Soundings':
            LON=np.copy(rgrLonAct)
            LON[LON>180]=LON[LON>180]-360
            rgiLandOcean=[bm.is_land(LON[ll], rgrLatAct[ll]) for ll in range(len(rgrLatAct))]
            # flags = PATH.contains_points(np.hstack((TEST[:,np.newaxis],rgrLatAct[:,np.newaxis])))
            rgiYearsAct=np.array(range(1979,2017,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            DATA_SSP=RS_SSP
            DATA_SSP_ML=RS_SSP_ML
            sColor='k'
        if rgsData[da] == 'ERA-20C':
            TEST=np.copy(rgrLonGrid2D)
            TEST[TEST>180]=TEST[TEST>180]-360
            rgiYearsAct=np.array(range(1900,2011,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            DATA_SSP=np.reshape(E20_SSP, (E20_SSP.shape[0],E20_SSP.shape[1],E20_SSP.shape[2]*E20_SSP.shape[3]))
            DATA_SSP_ML=np.reshape(E20_SSP_ML, (E20_SSP_ML.shape[0],E20_SSP_ML.shape[1],E20_SSP_ML.shape[2]*E20_SSP_ML.shape[3]))
            # T2M
            DATAT2M=np.reshape(rgrDataVarActT2M, (rgrDataVarActT2M.shape[0],rgrDataVarActT2M.shape[1],rgrDataVarActT2M.shape[2]*rgrDataVarActT2M.shape[3]))
            sColor='#1f78b4'
            rgiLandOcean=np.load('LandOcean_E20.npy')
            Weights=np.transpose(np.tile(np.cos(np.linspace(-90,90,E20_SSP.shape[2])/(360/(np.pi*2))),(E20_SSP.shape[3],1))).flatten()
        if rgsData[da] == 'ERA-Interim':
            TEST=np.copy(rgrLonGrid2D_EI)
            TEST[TEST>180]=TEST[TEST>180]-360
            rgiYearsAct=np.array(range(1979,2018,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            DATA_SSP=np.reshape(EI_SSP, (EI_SSP.shape[0],EI_SSP.shape[1],EI_SSP.shape[2]*EI_SSP.shape[3]))*12
            DATA_SSP_ML=np.reshape(EI_SSP_ML, (EI_SSP_ML.shape[0],EI_SSP_ML.shape[1],EI_SSP_ML.shape[2]*EI_SSP_ML.shape[3]))*12
            sColor='#e31a1c'
            rgiLandOcean=np.load('LandOcean_EI.npy')
            Weights=np.transpose(np.tile(np.cos(np.linspace(-90,90,EI_SSP.shape[2])/(360/(np.pi*2))),(EI_SSP.shape[3],1))).flatten()

        if rgsSubregions[sr] == 'Global':
            flags=np.array([True]*DATA_SSP.shape[2])
        elif rgsSubregions[sr] == 'Global Land':
            flags=rgiLandOcean
        elif rgsSubregions[sr] == 'Global Ocean':
            flags=(rgiLandOcean == False)

        rgrTMPData=np.nanmean((np.nansum(DATA_SSP_ML[:,:,flags],axis=(1))/np.nansum(DATA_SSP[:,:,flags],axis=(1)))-(np.nanmean(np.nansum(DATA_SSP_ML[rgiSelYY,:,:][:,:,flags], axis=(1)), axis=0)/np.nanmean(np.nansum(DATA_SSP[rgiSelYY,:,:][:,:,flags], axis=(1)), axis=0))[None,:], axis=1)
        rgrTMPData=rgrTMPData*100.
        if rgsSubregions[sr] == 'Global Land':
            print ' '
            print '    '+rgsData[da]
            print '     slope '+str(stats.linregress(range(rgrSeasonalData.shape[0]),rgrTMPData[rgiSelYY])[0]*10)
            print '     p-value '+str(stats.linregress(range(rgrSeasonalData.shape[0]),rgrTMPData[rgiSelYY])[3])


        # if (rgsData[da] == 'Rad-Soundings'):
        #     rgrTMPData=np.nanmean(np.nanmean(DATA[:,:,flags],axis=(1))-np.nanmean(DATA[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
        # else:
        #     indices = ~np.isnan(np.mean(DATA[:,:,flags], axis=(0,1)))
        #     Full=np.array([np.average(np.median(DATA[yy,:,flags],axis=1)[indices], weights=Weights[flags][indices]) for yy in range(DATA.shape[0])])
        #     rgrTMPData=Full-np.median(Full[rgiSelYY])

        plt.plot(rgiYearsAct, rgrTMPData, c=sColor, lw=0.5, zorder=3, alpha=0.5)
        plt.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPData,10), c=sColor, lw=3, zorder=3, label=rgsData[da])
        # plt.plot(iYears_I[iComYears], rgrModData, c='#1f78b4')
        # plt.plot(iYears_I[iComYears], rgrModDataEI, c='#e31a1c')

        plt.title(rgsLableABC[ii+2]+') '+rgsSubregions[sr]) #+' | '+str(rgrROdataAct.shape[2]))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ii == 0:
            ax.set_ylabel(sLabel+' '+sUnit)
        else:
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelleft=False) # labels along the bottom edge are off
        ax.set(ylim=(Yminmax[0],Yminmax[1]))
        ax.set_xlabel('Year')

        if rgsData[da] == 'ERA-20C':
            rgiYYE20=np.array(range(1900,2010))
            rgrTMPDataT2M=np.nanmean(DATAT2M[:,:,flags],axis=(1,2))-np.nanmean(DATAT2M[(rgiYYE20 >=1979),:,:][:,:,flags])
            Y2col='#636363'
            # Plot T2M on a second y-axis
            ax2 = ax.twinx()
            ax2.set_ylabel('2m temperature (T2M) [$^{\circ}$C]', color=Y2col)
            ax2.plot(rgiYYE20, scipy.ndimage.uniform_filter(rgrTMPDataT2M,10), c=Y2col, lw=3, zorder=1, alpha=0.5)
            ax2.plot(rgiYYE20, rgrTMPDataT2M, c=Y2col, lw=0.5, zorder=1, alpha=0.5)
            ax2.set(ylim=(-2,2))
            # ax2.set_yticks(np.arange(30, 50, 5))
            ax2.spines['right'].set_color(Y2col)
            ax2.yaxis.label.set_color(Y2col)
            ax2.tick_params(axis='y', colors=Y2col)
            ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)

    ax.plot([2000,2000],[0,0],lw=3,alpha=0.5, c='#636363', label='ERA-20C T2M')
    if ii == 0:
        # lns = lns1+lns2
        # labs = [l.get_label() for l in lns]
        ax.legend(loc="upper left",\
                  ncol=1, prop={'size':15})
    ii=ii+1


sPlotFile=PlotDir
sPlotName= 'Hail-Risk-Changes.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
