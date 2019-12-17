#!/usr/bin/env python
'''
    File name: Plot_RadSoundEnv_ERA20.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    Read in the preprocessed data from:
    ~/projects/Hail/programs/RadioSoundings/Preprocessor/HailParametersFromRadioSounding.py

    Plot the frequency of hail environments and the trend in the annual
    time series

    Also plot the trend in individual parameters such as FLH

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
rgsVars=['WBZheight','ThreeCheight','ProbSnow','BflLR']
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
rgsSeasons=['DJF','MAM','JJA','SON']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30]

rgsData=['ERA-20C', 'ERA-Interim']

########################################
#                      read in the radio sounding data
sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/'
sRadSoundData=sSaveDataDir+'RadSoundHailEnvDat-1979-2017.pkl'
grStatisitcs = pickle.load( open(sRadSoundData, "rb" ) )

# rgrCAPED=grStatisitcs['rgrCAPE']
# rgrRHbfD=grStatisitcs['rgrRHbf']*100.
# rgrRHmlD=grStatisitcs['rgrRHml']*100.
# rgrVS0_1D=grStatisitcs['rgrVS0_1']
# rgrVS0_3D=grStatisitcs['rgrVS0_3']
# rgrVS0_6D=grStatisitcs['rgrVS0_6']
# rgrVS6_12D=grStatisitcs['rgrVS6_12']
# rgrPWD=grStatisitcs['PW']
# rgrFLHD=grStatisitcs['rgrFLH']
rgrWBZD=grStatisitcs['rgrWBZ']
rgrThreeCheight=grStatisitcs['rgrThreeCheight']
rgrLRbF=grStatisitcs['rgrLRbF']
rgrProbSnow=grStatisitcs['rgrProbSnow']
rgsVariables=grStatisitcs['rgsVariables']
rgdTimeFin=grStatisitcs['rgdTime']
rgrLon=grStatisitcs['rgrLon']
rgrLat=grStatisitcs['rgrLat']
rgrElev=grStatisitcs['rgrElev']
rgsStatID=np.array(grStatisitcs['rgsStatID'])
rgsStatNr=np.array(grStatisitcs['rgsStatNr'])

iYears=np.unique(rgdTimeFin.year)

########################################
#                      check if we have enough data
iMinCov=0.7
rgiStations=np.zeros((len(rgrLon))); rgiStations[:]=0
for st in range(len(rgrLon)):
    rgiFin=np.sum(~np.isnan(rgrWBZD[:,st]))# *~np.isnan(rgrThreeCheight[:,st])*\
        # ~np.isnan(rgrProbSnow[:,st]))
    if float(rgiFin)/float(rgrWBZD.shape[0]) > iMinCov:
        rgiStations[st]=1
rgiStations=(rgiStations == 1)
# select the stations which have sufficient coverage
rgrWBZD=rgrWBZD[:,rgiStations]
rgrThreeCheight=rgrThreeCheight[:,rgiStations]
rgrLRbF=rgrLRbF[:,rgiStations]
rgrProbSnow=rgrProbSnow[:,rgiStations]
rgrLon=rgrLon[rgiStations]
rgrLat=rgrLat[rgiStations]
rgrElev=rgrElev[rgiStations]
rgsStatID=rgsStatID[rgiStations]
rgsStatNr=rgsStatNr[rgiStations]

rgrProbSnow[rgrProbSnow < 0.5]=0
rgrProbSnow[rgrProbSnow > 0.5]=1



# ########################################
# # Write the data to netcdf for Andy Hymsfield

# root_grp = Dataset('Radiosounding_derived-variables.nc', 'w', format='NETCDF3_CLASSIC')
# # dimensions
# root_grp.createDimension('location',len(rgrLat))
# root_grp.createDimension('time',None)
# # variables
# CAPE = root_grp.createVariable('CAPE', 'f8', ('time','location',),fill_value=1.e+20)
# RHbf = root_grp.createVariable('RHbf', 'f8', ('time','location',),fill_value=1.e+20)
# RHml = root_grp.createVariable('RHml', 'f8', ('time','location',),fill_value=1.e+20)
# VS01 = root_grp.createVariable('VS01', 'f8', ('time','location',),fill_value=1.e+20)
# VS03 = root_grp.createVariable('VS03', 'f8', ('time','location',),fill_value=1.e+20)
# VS06 = root_grp.createVariable('VS06', 'f8', ('time','location',),fill_value=1.e+20)
# PW = root_grp.createVariable('PW', 'f8', ('time','location',),fill_value=1.e+20)
# FLH = root_grp.createVariable('FLH', 'f8', ('time','location',),fill_value=1.e+20)
# WBZH = root_grp.createVariable('WBZH', 'f8', ('time','location',),fill_value=1.e+20)
# lat= root_grp.createVariable('lat', 'f8', ('location',))
# lon= root_grp.createVariable('lon', 'f8', ('location',))
# times= root_grp.createVariable('times', 'f8', ('time',))
# # Variable Attributes

# times.units = 'days since 1979-01-01 00:00:00'
# times.calendar = 'gregorian'

# lat.units = 'degrees_north'
# lat.long_name = 'latitude'
# lat.standard_name = "latitude"

# lon.units = 'degrees_east'
# lon.long_name = 'longitude'
# lon.standard_name = "longitude"

# CAPE.long_name = "CAPE"
# CAPE.level = "integral"
# CAPE.units = "J m-2"

# RHbf.long_name = "relative humidity below freezing level"
# RHbf.level = "average"
# RHbf.units = "%"

# RHml.long_name = "relative humidity in 700-500 hPa level"
# RHml.level = "average"
# RHml.units = "%"

# VS01.long_name = "vector windspeed shear"
# VS01.level = "0-1 km"
# VS01.units = "m s-1"

# VS03.long_name = "vector windspeed shear"
# VS03.level = "0-3 km"
# VS03.units = "m s-1"

# VS06.long_name = "vector windspeed shear"
# VS06.level = "0-6 km"
# VS06.units = "m s-1"

# PW.long_name = "precipitable water"
# PW.level = "integral"
# PW.units = "mm"

# FLH.long_name = "height of freezing level above surface"
# FLH.level = ""
# FLH.units = "m"

# WBZH.long_name = "height of wet-bulb zero level"
# WBZH.level = ""
# WBZH.units = "m"

# root_grp.Author = "Andreas Prein"
# root_grp.email = "prein@ucar.edu"
# root_grp.timestamp = "Sept. 27, 2018"

# # write data to netcdf
# lat[:]=rgrLat
# lon[:]=rgrLon
# times[:]=range(rgrCAPED.shape[0])
# CAPE[:]=rgrCAPED
# RHbf[:]=rgrRHbfD
# RHml[:]=rgrRHmlD
# VS01[:]=rgrVS0_1D
# VS03[:]=rgrVS0_3D
# VS06[:]=rgrVS0_6D
# PW[:]=rgrPWD
# FLH[:]=rgrFLHD
# WBZH[:]=rgrWBZD
# root_grp.close()
# stop()


# ################################################################################
# ################################################################################
# ################################################################################
# #          Process ERA-20
# ################################################################################

dStartDay=datetime.datetime(1979, 1, 1,0)
dStopDay=datetime.datetime(2009, 12, 31,23)
# dStopDay=datetime.datetime(2010, 12, 31,23)
rgdTime6H = pd.date_range(dStartDay, end=dStopDay, freq='3h')
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgiYY=np.unique(rgdTime6H.year)
rgiERA_hours=np.array([0,6,12,18])
rgdFullTime=pd.date_range(datetime.datetime(1900, 1, 1,00),
                        end=datetime.datetime(2009, 12, 31,23), freq='3h')
# # first read the coordinates
sERAconstantFields='/glade/work/prein/reanalyses/ERA-20c/e20c.oper.invariant.128_129_z.regn80sc.1900010100_2010123121.nc'
# read the ERA-Interim elevation
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat75=np.squeeze(ncid.variables['g4_lat_0'][:])
rgrLon75=np.squeeze(ncid.variables['g4_lon_1'][:])
rgrHeight=(np.squeeze(ncid.variables['Z_GDS4_SFC'][:]))/9.81
ncid.close()
rgrLonGrid2D=np.asarray(([rgrLon75,]*rgrLat75.shape[0]))
rgrLatGrid2D=np.asarray(([rgrLat75,]*rgrLon75.shape[0])).transpose()

# rgrMonthlyData=np.zeros((len(rgiYY),12,len(rgrLat75), len(rgrLon75),len(rgsVars))); rgrMonthlyData[:]=np.nan
# for yy in range(len(rgiYY)): #range(len(TC_Name)): # loop over hurricanes
#     print ' '
#     print '    Workin on year '+str(rgiYY[yy])
#     # check if we already processed this file
#     for mm in range(12): #  loop over time
#         tt=0
#         rgrTempData=np.zeros((31*8,len(rgrLat75), len(rgrLon75),len(rgsVars))); rgrTempData[:]=np.nan
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
#             for va in range(len(rgsVars)):
#                 rgrDataTMP=np.squeeze(ncid.variables[rgsVars[va]][:])
#                 rgrDataTMP=np.mean(np.reshape(rgrDataTMP, (rgrDataTMP.shape[0]/8,8,rgrDataTMP.shape[1],rgrDataTMP.shape[2])),axis=1)
#                 try:
#                     rgrTempData[tt:tt+rgrDataTMP.shape[0],:,:,va]=rgrDataTMP
#                 except:
#                     stop()
#             ncid.close()
#             tt=int(tt+rgrDataTMP.shape[0])
#         rgrMonthlyData[yy,mm,:,:,:]=np.nanmean(rgrTempData[:,:,:,:], axis=0)
# np.save(sSaveDataDir+'ERA20_monthlydata.npy',rgrMonthlyData)

rgrMonthlyData=np.load(sSaveDataDir+'ERA20_monthlydata.npy')
# convert probability of snow to yes or no
Psnow=rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]
Psnow[Psnow>=0.5]=1; Psnow[Psnow<0.5]=0
rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]=Psnow




# ################################################################################
# ################################################################################
# ################################################################################
# #          Process ERA-Interim
# ################################################################################

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

# rgrMonthlyData_EI=np.zeros((len(rgiYY_EI),12,len(rgrLat75_EI), len(rgrLon75_EI),len(rgsVars))); rgrMonthlyData_EI[:]=np.nan
# for yy in range(len(rgiYY_EI)): #range(len(TC_Name)): # loop over hurricanes
#     print ' '
#     print '    Workin on year '+str(rgiYY_EI[yy])
#     # check if we already processed this file
#     for mm in range(12): #  loop over time
#         rgiDD_EI=np.sum(((rgdTimeDD_EI.year == rgiYY_EI[yy]) & (rgdTimeDD_EI.month == (mm+1))))
#         rgrTempData_EI=np.zeros((rgiDD_EI,len(rgrLat75_EI), len(rgrLon75_EI),len(rgsVars))); rgrTempData_EI[:]=np.nan
#         YYYY=str(rgiYY_EI[yy])
#         MM=str("%02d" % (mm+1))
#         for dd in range(rgiDD_EI):
#             DD=str("%02d" % (dd+1))
#             sDate=YYYY+MM+DD
#             sFileFin=sSaveDataDir+'ERA-Int/'+sDate+'_ERA-Int_Hail-Env_RDA.nc'
#             ncid=Dataset(sFileFin, mode='r')
#             for va in range(len(rgsVars)):
#                 rgrDataTMP=np.squeeze(ncid.variables[rgsVars[va]][:])
#                 if rgsVars[va] != 'ProbSnow':
#                     rgrDataTMP_EI=np.mean(rgrDataTMP,axis=0)
#                 else:
#                     rgrDataTMP_EI=np.max(rgrDataTMP,axis=0)
#                 try:
#                     rgrTempData_EI[dd,:,:,va]=rgrDataTMP_EI
#                 except:
#                     stop()
#             ncid.close()
#         rgrMonthlyData_EI[yy,mm,:,:,:]=np.nanmean(rgrTempData_EI[:,:,:,:], axis=0)
#         rgrTMP=rgrMonthlyData_EI[yy,mm,:,:,rgsVars.index('ProbSnow')]; rgrTMP[rgrTMP>=0.5]=1; rgrTMP[rgrTMP<=0.5]=0
#         rgrMonthlyData_EI[yy,mm,:,:,rgsVars.index('ProbSnow')]=rgrTMP
# np.save(sSaveDataDir+'ERA-Int_monthlydata.npy',rgrMonthlyData_EI)

rgrMonthlyData_EI=np.load(sSaveDataDir+'ERA-Int_monthlydata.npy')[:-8]
# convert probability of snow to yes or no
Psnow=rgrMonthlyData_EI[:,:,:,:,rgsVars.index('ProbSnow')]
Psnow[Psnow>=0.5]=1; Psnow[Psnow<0.5]=0
rgrMonthlyData_EI[:,:,:,:,rgsVars.index('ProbSnow')]=Psnow


# # =======================
# # READ T2M SEPERATELY
# sDir='/gpfs/fs1/collections/rda/data/ds626.0/e20c.oper.fc.sfc.3hr/'
# # rgdTimeDD
# rgrT2Mmonth=np.zeros((len(rgiYY),12,len(rgrLat75),len(rgrLon75))); rgrT2Mmonth[:]=np.nan
# for yy in range(len(rgiYY)):
#     rgdTimeAct=rgdTimeDD[rgdTimeDD.year == rgiYY[yy]]
#     iFile=sDir+str(rgiYY[yy])[:3]+'0_'+str(rgiYY[yy])[:3]+'9/e20c.oper.fc.sfc.3hr.128_167_2t.regn80sc.'+str(rgiYY[yy])+'010109_'+str(rgiYY[yy]+1)+'010106.grb'
#     iFileGRIB='/glade/scratch/prein/tmp/e20c.oper.fc.sfc.3hr.128_167_2t.regn80sc.'+str(rgiYY[yy])+'010109_'+str(rgiYY[yy]+1)+'010106.grb'
#     iFileNC='/glade/scratch/prein/tmp/e20c.oper.fc.sfc.3hr.128_167_2t.regn80sc.'+str(rgiYY[yy])+'010109_'+str(rgiYY[yy]+1)+'010106.nc'
#     copyfile(iFile, iFileGRIB)
#     subprocess.call("ncl_convert2nc "+iFileGRIB+' -o '+'/glade/scratch/prein/tmp/'+' -L', shell=True)
#     print '    Load '+iFileNC
#     # read in the variables
#     ncid=Dataset(iFileNC, mode='r')
#     rgrT2Act=np.mean(np.squeeze(ncid.variables['2T_GDS4_SFC']), axis=1)
#     for mo in range(12):
#         rgiAct=(rgdTimeAct.month == (mo+1))
#         rgrT2Mmonth[yy,mo,:,:]=np.mean(rgrT2Act[rgiAct,:,:], axis=0)
#     ncid.close()
#     # clean up
#     os.system("rm "+iFileGRIB); os.system("rm "+iFileNC)
# np.save('ERA-20C_T2M.npy',rgrT2Mmonth)
# stop()
rgrT2Mmonth=np.load('ERA-20C_T2M.npy')
rgsVars=rgsVars+['T2M']
rgrMonthlyData=np.append(rgrMonthlyData,rgrT2Mmonth[:,:,:,:,None], axis=4)

########################################
#                      start plotting



# loop over the different variables
# rgsVars=['BflLR']

for va in range(len(rgsVars)):
    print '    Plotting '+rgsVars[va]
    if rgsVars[va] == 'CAPE':
        biasContDist=10
        sLabel='trend in CAPE [J kg$^{-1}$ decade$^{-1}$]'
        rgrDataVar=rgrCAPED
    elif rgsVars[va] == 'WBZheight':
        biasContDist=20
        sLabel='trend in melting level height [m per decade]'
        rgrDataVar=rgrWBZD
    elif rgsVars[va] == 'RHbf':
        biasContDist=1
        sLabel='trend in mean RH below freezing level [% per decade]'
        rgrDataVar=rgrRHbfD
    elif rgsVars[va] == 'RHml':
        biasContDist=1
        sLabel='trend in men RH 700-500 hPa [% per decade]'
        rgrDataVar=rgrRHmlD
    elif rgsVars[va] == 'VS0_1':
        biasContDist=0.2
        sLabel='trend in 0-1 km shear [m s$^{-1}$ per decade]'
        rgrDataVar=rgrVS0_1D
    elif rgsVars[va] == 'VS0_3':
        biasContDist=0.2
        sLabel='trend in 0-3 km shear [m s$^{-1}$ per decade]'
        rgrDataVar=rgrVS0_3D
    elif rgsVars[va] == 'VS0_6':
        biasContDist=0.2
        sLabel='trend in 0-6 km shear [m s$^{-1}$ per decade]'
        rgrDataVar=rgrVS0_6D
    elif rgsVars[va] == 'PW':
        biasContDist=2
        sLabel='trend in precipitable water [mm per decade]'
        rgrDataVar=rgrPWD
    elif rgsVars[va] == 'FLH':
        biasContDist=20
        sLabel='trend in freezing level height [m per decade]'
        rgrDataVar=rgrFLHD
    elif rgsVars[va] == 'ThreeCheight':
        biasContDist=20
        sLabel='trend in 3 $^{\circ}$C height [m per decade]'
        rgrDataVar=rgrThreeCheight
    elif rgsVars[va] == 'ProbSnow':
        biasContDist=1
        sLabel='trend in probability of snow [days per decade]'
        rgrDataVar=rgrProbSnow
    elif rgsVars[va] == 'T2M':
        biasContDist=0.16
        sLabel='trend in 2m temperature [$^{\circ}$C per decade]'
    elif rgsVars[va] == 'BflLR':
        biasContDist=0.05
        sLabel='trend in Surf-ML Lapse Rate [$^{\circ}$C km$^{-1}$ per decade]'
        rgrDataVar=rgrLRbF

    for da in range(len(rgsData)):

        fig = plt.figure(figsize=(15, 7))
        gs1 = gridspec.GridSpec(2,2)
        gs1.update(left=0.05, right=0.96,
                   bottom=0.12, top=0.94,
                   wspace=0.2, hspace=0.1)
        XX=[0,1,0,1]
        YY=[0,0,1,1]
        for se in range(4):
            # scalings
            if rgsVars[va] == 'ProbSnow':
                iScale=iDaysInSeason[se]
            else:
                iScale=1.
            print '    Work on Season '+str(se)
            if se == 0:
                rgiTime=(rgdTimeFin.month == 12) | (rgdTimeFin.month <= 2)
            elif se == 1:
                rgiTime=(rgdTimeFin.month >= 3) & (rgdTimeFin.month <= 5)
            elif se == 2:
                rgiTime=(rgdTimeFin.month >= 6) & (rgdTimeFin.month <= 8)
            elif se == 3:
                rgiTime=(rgdTimeFin.month >= 9) & (rgdTimeFin.month <= 11)
        
            rgrColorTable=np.array(['#4457c9','#4f67d4','#6988ea','#84a6f9','#9ebdff','#b7d0fd','#cfdcf3','#e2e1e0','#f1d7c8','#f9c5ab','#f7aa8c','#f28d72','#df6553','#c93137','#bc052b'])
            iContNr=len(rgrColorTable)
            # clevsTD=np.arange(0, iContNr*biasContDist,biasContDist)-(iContNr*biasContDist)/2.+biasContDist/2.
            iMinMax=12.*biasContDist/2.+biasContDist/2.
            clevsTD=np.arange(-iMinMax,iMinMax+0.0001,biasContDist)
    
    
            ###################################
            # calculate and plot trends
            # ax = plt.subplot(gs1[YY[se],XX[se]], projection=ccrs.Robinson(central_longitude=0))
            ax = plt.subplot(gs1[YY[se],XX[se]], projection=cartopy.crs.PlateCarree())
            ax.set_extent([-175, 160, -60, 80], crs=ccrs.PlateCarree())
            # ax.set_global()
    
        
            if rgsVars[va] != 'T2M':
                rgrGCtrend=np.zeros((len(rgrLon))); rgrGCtrend[:]=np.nan
                rgrPvalue=np.copy(rgrGCtrend)
                rgrR2=np.copy(rgrGCtrend)
                # get the data
                rgrDataAct=rgrDataVar[rgiTime,:]
                rgrDataAnnual=np.zeros((len(iYears),rgrDataAct.shape[1])); rgrDataAnnual[:]=np.nan
                for yy in range(len(iYears)):
                    rgiYYact=(rgdTimeFin[rgiTime].year == iYears[yy])
                    rgrDataAnnual[yy,:]=np.nanmean(rgrDataAct[rgiYYact,:], axis=0)
                    # remove data with too little coverage
                    iCoverage=np.sum(np.isnan(rgrDataAct[rgiYYact,:]),axis=0)/float(sum(rgiYYact))
                    rgrDataAnnual[yy,(iCoverage> 1.-iMinCov)]=np.nan
            
                from StatisticFunktions import fnPerturbTrend
                for st in range(len(rgrLon)):
                    # remove nans if there are some
                    rgiReal=~np.isnan(rgrDataAnnual[:,st])
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(iYears[rgiReal],rgrDataAnnual[:,st][rgiReal])
                    except:
                        continue
                    rgrGCtrend[st]=slope*10*iScale #(slope/np.nanmean(rgrDataAnnual[:,st]))*1000.
                    rgrPvalue[st]=p_value
                    rgrR2[st]=r_value**2
                
                
                # # save the data so we can include it in the ERA-Interim trend analysis
                # sSaveData='./RadSoundingHailEnvTrends.npz'
                # np.savez(sSaveData, \
                #              rgrGCtrend=rgrGCtrend, \
                #              rgrPvalue=rgrPvalue, \
                #              rgrLon=rgrLon, \
                #              rgrLat=rgrLat)
            
            
                # plot circles that show the trend in area
                if np.min(rgrLon) < 0:
                    rgrLon[rgrLon < 0]=rgrLon[rgrLon < 0]+360
                for st in range(len(rgrLon)):
                    if np.isnan(rgrGCtrend[st]) != 1:
                        try:
                            iColor=np.where((clevsTD-rgrGCtrend[st]) > 0)[0][0]-1
                        except:
                            if rgrGCtrend[st] > 0:
                                iColor=len(clevsTD)-1
                        if iColor == -1:
                            iColor=0
                        if (rgrPvalue[st] <= 0.05):
                            ax.plot(rgrLon[st], rgrLat[st],'o',color=rgrColorTable[iColor], ms=5, mec='k', alpha=1, markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=20)
                        else:
                            # not significant trend
                            try:
                                ax.plot(rgrLon[st], rgrLat[st],'o',color=rgrColorTable[iColor], ms=5, mec=rgrColorTable[iColor], alpha=1, markeredgewidth=0.0,transform=ccrs.Geodetic(), zorder=20)
                            except:
                                stop()
            
            
            
            ###############################################################
            if rgsData[da] == 'ERA-20C':
                print '    Add ERA-20C analysis'
                rgrMonthAct=rgrMonthlyData
                rgrLonAct=rgrLonGrid2D
                rgrLatAct=rgrLatGrid2D
            elif rgsData[da] == 'ERA-Interim':
                print '    Add ERA-Interim analysis'
                rgrMonthAct=rgrMonthlyData_EI
                rgrLonAct=rgrLonGrid2D_EI
                rgrLatAct=rgrLatGrid2D_EI

            rgrTrendsERA20=np.zeros((rgrMonthAct.shape[2],rgrMonthAct.shape[3],5)); rgrTrendsERA20[:]=np.nan
            if se == 0:
                rgrSeasonalData=np.mean(rgrMonthAct[:,(0,1,11),:,:,va],axis=0)
            elif se == 1:
                rgrSeasonalData=np.mean(rgrMonthAct[:,(2,3,4),:,:,va],axis=0)
            elif se == 2:
                rgrSeasonalData=np.mean(rgrMonthAct[:,(5,6,7),:,:,va],axis=0)
            elif se == 3:
                rgrSeasonalData=np.mean(rgrMonthAct[:,(8,9,10),:,:,va],axis=0)
            for la in range(rgrSeasonalData.shape[1]):
                for lo in range(rgrSeasonalData.shape[2]):
                    rgrTrendsERA20[la,lo,:]=stats.linregress(rgiYY,rgrSeasonalData[:,la,lo])
            if rgsVars[va] == 'ProbSnow':
                iScale=iDaysInSeason[se]
            else:
                iScale=1
            # plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,colors=rgrColorTable, extend='both',levels=clevsTD) #, alpha=0.6)
            cs=plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,cmap=plt.cm.coolwarm, extend='both',levels=clevsTD, zorder=0) #, alpha=0.6)
    
            
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5, zorder=11)
            ax.add_feature(cartopy.feature.COASTLINE, zorder=11)
            ax.add_feature(cartopy.feature.STATES, zorder=11, alpha=.5)
            ax.gridlines(zorder=11)
            plt.axis('off')
    
            print '    add colorbar'
            CbarAx = axes([0.1, 0.05, 0.8, 0.02])
            cb = colorbar(cs, cax = CbarAx, orientation='horizontal', extend='max', ticks=clevsTD, extendfrac='auto')
            cb.ax.set_title(sLabel)
            # ax3 = fig.add_axes([0.1, 0.08, 0.8, 0.02])
            # cmap = mpl.colors.ListedColormap(rgrColorTable[:-1])
            # cmap.set_under('#313695')
            # cmap.set_over('#a50026')
            # bounds = clevsTD
            # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            # cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
            #                                 norm=norm,
            #                                 boundaries=[-10] + bounds + [10],
            #                                 extend='both',
            #                                 # Make the length of each extension
            #                                 # the same as the length of the
            #                                 # interior colors:
            #                                 extendfrac='auto',
            #                                 ticks=bounds,
            #                                 spacing='uniform',
            #                                 orientation='horizontal')
            
            # cb3.set_label(sLabel)
    
    
            # lable the maps
            ax.text(0.03,1.04, rgsLableABC[se]+') '+rgsSeasons[se] , ha='left',va='center', \
                        transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=20)
    

        sPlotFile=PlotDir
        sPlotName= 'RadioSoundingHailEnv-1979-2015_'+rgsVars[va]+'_'+rgsData[da]+'.pdf'
        if os.path.isdir(sPlotFile) != 1:
            subprocess.call(["mkdir","-p",sPlotFile])
        print '        Plot map to: '+sPlotFile+sPlotName
        fig.savefig(sPlotFile+sPlotName)






stop()
