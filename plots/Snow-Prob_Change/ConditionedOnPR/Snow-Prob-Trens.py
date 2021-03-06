#!/usr/bin/env python
'''
    File name: Snow-Prob-Trens.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    Read in the preprocessed data from:
    ~/projects/Hail/programs/RadioSoundings/Preprocessor/HailParametersFromRadioSounding.py

    Read in ERA-20C and ERA-Interim Data

    Plot a map that shows snow probability trends in the period 1979-2010

    Add contour lines that show 5 days/year lines in 30-year climatologies

    Add inlays that show time series for focus regions


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
rgsVars=['ProbSnow','Snow_PRdays','Rain_PRdays'] #['WBZheight','ThreeCheight','ProbSnow','BflLR']
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
rgsSeasons=['DJF','MAM','JJA','SON','Annual']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30,365.25]

rgsData=['Rad-Soundings', 'ERA-Interim','ERA-20C']

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
#             READ IN THE U-WYOMING RADIO SOUNDING DATA
sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/'
sRadSoundData=sSaveDataDir+'RadSoundHailEnvDat-1979-2017.pkl'
grStatisitcs = pickle.load( open(sRadSoundData, "rb" ) )
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

rgsVarsUW=['Snow','WBZ','ThreeC','FLH','TS','LRbF','Snow_PRdays', 'Rain_PRdays']

# # # load grid from MSWAP
# dStartDay_MSWEP=datetime.datetime(1979, 1, 1,0)
# dStopDay_MSWEP=datetime.datetime(2016, 12, 31,0)
# rgdTimeDD_MSWEP = pd.date_range(dStartDay_MSWEP, end=dStopDay_MSWEP, freq='d')
# # sMSWEP_Grid='/glade/scratch/prein/MSWEP_V2.1/data/197901.nc'
# # ncid=Dataset(sMSWEP_Grid, mode='r')
# # rgrLat_MSWEP=np.squeeze(ncid.variables['lat'][:])
# # rgrLon_MSWEP=np.squeeze(ncid.variables['lon'][:]); rgrLon_MSWEP[rgrLon_MSWEP<0]=rgrLon_MSWEP[rgrLon_MSWEP<0]+360
# # ncid.close()
# # rgiLonMin_MSWEP=np.array([np.argmin(np.abs(rgrLonS[ii] - rgrLon_MSWEP)) for ii in range(len(rgrLonS))])
# # rgiLatMin_MSWEP=np.array([np.argmin(np.abs(rgrLatS[ii] - rgrLat_MSWEP)) for ii in range(len(rgrLonS))])
# # YYYY_MSWEP=np.unique(rgdTimeDD_MSWEP.year)
# # rgrMSWEP_PR=np.zeros((len(rgdTimeDD_MSWEP), len(rgrLonS)))
# # for yy in range(len(YYYY_MSWEP)):
# #     print '    Process Year '+str(YYYY_MSWEP[yy])
# #     for mm in range(12):
# #         sFileName='/glade/scratch/prein/MSWEP_V2.1/data/'+str(YYYY_MSWEP[yy])+str("%02d" % (mm+1))+'.nc'
# #         rgiTimeAct=((rgdTimeDD_MSWEP.year == YYYY_MSWEP[yy]) & (rgdTimeDD_MSWEP.month == (mm+1)))
# #         ncid=Dataset(sFileName, mode='r')
# #         for st in range(len(rgrLonS)):
# #             rgrDATAtmp=ncid.variables['precipitation'][:,rgiLatMin_MSWEP[st],rgiLonMin_MSWEP[st]]
# #             rgrMSWEP_PR[rgiTimeAct,st]=np.sum(np.reshape(rgrDATAtmp,(len(rgrDATAtmp)/8,8)), axis=1)
# #         ncid.close()
# # np.savez('RS_MSWEP_PR.npz', rgrMSWEP_PR=rgrMSWEP_PR)
# rgrMSWEP_PR=np.load('RS_MSWEP_PR.npz')['rgrMSWEP_PR']
# rgrMSWEP_PR=rgrMSWEP_PR[:,:len(rgrLon)]
# rgiMSWEP_PR=np.copy(rgrMSWEP_PR)
# rgiMSWEP_PR[(rgrMSWEP_PR < 1)]=0; rgiMSWEP_PR[(rgrMSWEP_PR >= 1)]=1
# # rgiMSWEP_PR=np.transpose(rgiMSWEP_PR)

# # convert to monthly data
# # CALCULATE MONTHY AVERAGES WITH THE REQUIREMENT THAT WE HAVE AT LEAST 70 % COVERAGE
# rgrMonUW=np.zeros((len(iYears),12,rgrWBZD.shape[1],len(rgsVarsUW))); rgrMonUW[:]=np.nan
# for yy in range(len(iYears)):
#     print '    Process Year '+str(iYears[yy])
#     for mm in range(12):
#         rgiTimes=((rgdTimeFin.year == iYears[yy]) & (rgdTimeFin.month == (mm+1)))
#         for va in range(len(rgsVarsUW)):
#             if rgsVarsUW[va] == 'Snow':
#                 rgrData=rgrProbSnow[rgiTimes,:]
#                 rgrData[rgrData >= 0.5]=1; rgrData[rgrData < 0.5]= 0
#             if rgsVarsUW[va] == 'Snow_PRdays':
#                 rgiTimePR=((rgdTimeDD_MSWEP.year == iYears[yy]) & (rgdTimeDD_MSWEP.month == (mm+1)))
#                 rgrData=rgrProbSnow[rgiTimes,:]
#                 try:
#                     rgrDataPR=np.array(rgiMSWEP_PR[rgiTimePR,:]); rgrDataPR[np.isnan(rgrData)]=np.nan
#                 except:
#                     continue
#                 rgrTMP=np.copy(rgrData); rgrTMP[:]=0
#                 rgrTMP[(rgrData >= 0.5) & (rgrDataPR >= 1)]=1
#                 rgrTMP[np.isnan(rgrData) ==1]=np.nan
#                 rgrData=rgrTMP
#             if rgsVarsUW[va] == 'Rain_PRdays':
#                 rgiTimePR=((rgdTimeDD_MSWEP.year == iYears[yy]) & (rgdTimeDD_MSWEP.month == (mm+1)))
#                 rgrData=rgrProbSnow[rgiTimes,:]
#                 try:
#                     rgrDataPR=np.array(rgiMSWEP_PR[rgiTimePR,:]); rgrDataPR[np.isnan(rgrData)]=np.nan
#                 except:
#                     continue
#                 rgrTMP=np.copy(rgrData); rgrTMP[:]=0
#                 rgrTMP[(rgrData < 0.5) & (rgrDataPR >= 1)]=1
#                 rgrTMP[np.isnan(rgrData) ==1]=np.nan
#                 rgrData=rgrTMP
#             if rgsVarsUW[va] == 'WBZ':
#                 rgrData=rgrWBZD[rgiTimes,:]
#             if rgsVarsUW[va] == 'ThreeC':
#                 rgrData=rgrThreeCheight[rgiTimes,:]
#             if rgsVarsUW[va] == 'FLH':
#                 continue
#             if rgsVarsUW[va] == 'TS':
#                 continue
#             if rgsVarsUW[va] == 'LRbF':
#                 rgrData=rgrLRbF[rgiTimes,:]
#             rgiNaNs=np.sum(np.isnan(rgrData), axis=0)/float(sum(rgiTimes))
#             rgrMonUW[yy,mm,rgiNaNs < 0.3,va]=np.nanmean(rgrData[:,rgiNaNs < 0.3], axis=0)

# np.save('UW-Monthly-RS.npy',rgrMonUW)
rgrMonUW=np.load('UW-Monthly-RS.npy')

########################################
#      READ IN THE IGRA MON-AVERAGE RADIO SOUNDING DATA
sRadSoundData='/glade/work/prein/papers/Trends_RadSoundings/data/IGRA/IGRA-Sounding-Data-1979-2017.pkl'
grStatisitcs = pickle.load( open(sRadSoundData, "rb" ))
rgrMonthDat_I=grStatisitcs['rgrMonIGRA']
rgsVariables_I=grStatisitcs['rgsVars']
rgdTimeFin_I=grStatisitcs['rgdTime']
rgrLon_I=grStatisitcs['rgiStatLon']
rgrLat_I=grStatisitcs['rgiStatLat']
rgrElev_I=grStatisitcs['rgiStatAlt']
rgsStatID_I=np.array(grStatisitcs['rgsStatID'])
rgsStatNr_I=np.array(grStatisitcs['rgsStatName'])

iYears_I=np.unique(rgdTimeFin_I.year)


# combine the RO datasets
rgiIgramVars=[rgsVariables_I.index('Snow'),rgsVariables_I.index('WBZ'),rgsVariables_I.index('ThreeC'),rgsVariables_I.index('FLH'),rgsVariables_I.index('TS'),rgsVariables_I.index('LRbF'),rgsVariables_I.index('Snow_PRdays'),rgsVariables_I.index('Rain_PRdays')]
rgrMonDat=np.append(rgrMonUW,rgrMonthDat_I[:,:,:,rgiIgramVars], axis=2)
rgrLonS=np.append(rgrLon,rgrLon_I, axis=0); rgrLonS[rgrLonS<0]=rgrLonS[rgrLonS<0]+360
rgrLatS=np.append(rgrLat,rgrLat_I, axis=0)

iMinCov=0.7
iMinCovMon=int(rgrMonDat.shape[0]*rgrMonDat.shape[1]*iMinCov)
rgiFin=np.array([np.sum(~np.isnan(rgrMonDat[:,:,st,:]), axis=(0,1)) for st in range(rgrMonDat.shape[2])])
rgrMonDat[:,:,(rgiFin < iMinCov)]=np.nan


# ################################################################################
# ################################################################################
# ################################################################################
# #          Process ERA-20
# ################################################################################

dStartDay=datetime.datetime(1900, 1, 1,0)
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

# # read in the daily precipiation from 
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
#             for va in range(len(rgsVars)):
#                 if (rgsVars[va] != 'Snow_PRdays') & (rgsVars[va] != 'Rain_PRdays'):
#                     ncid=Dataset(sFileFin, mode='r')
#                     rgrDataTMP=np.squeeze(ncid.variables[rgsVars[va]][:])
#                     ncid.close()
#                 else:
#                     ncid=Dataset(sFileFin, mode='r')
#                     rgrDataTMP=np.squeeze(ncid.variables['ProbSnow'][:])
#                     ncid.close()
#                 if rgsVars[va] == 'ProbSnow':
#                     DATA=np.copy(rgrDataTMP); DATA[:]=0
#                     DATA[rgrDataTMP >=0.5]=1; rgrDataTMP=DATA
#                     rgrDataTMP=np.max(np.reshape(rgrDataTMP, (rgrDataTMP.shape[0]/8,8,rgrDataTMP.shape[1],rgrDataTMP.shape[2])),axis=1)
#                 elif rgsVars[va] == 'Snow_PRdays':
#                     rgrDataTMP=np.max(np.reshape(rgrDataTMP, (rgrDataTMP.shape[0]/8,8,rgrDataTMP.shape[1],rgrDataTMP.shape[2])),axis=1)
#                     iE20timeSTART=np.where(rgdTimeDD == datetime.datetime(int(YYYY), int(MM), int(DD1[:2])))[0][0]
#                     iE20timeSTOP=np.where(rgdTimeDD == datetime.datetime(int(YYYY), int(MM), int(DD2[:2])))[0][0]+1
#                     ncid=Dataset('/glade/scratch/prein/ERA-20C/PR_1900-2010.nc', mode='r')
#                     rgrDataPR=(np.squeeze(ncid.variables['tp'][iE20timeSTART:iE20timeSTOP,:,:]))*1000.
#                     ncid.close()
#                     rgrTMP=np.copy(rgrDataTMP); rgrTMP[:]=0
#                     rgrTMP[:,:-1,:][(rgrDataTMP[:,:-1,:] > 0.5) & (rgrDataPR >= 1)]=1
#                     rgrTMP[np.isnan(rgrDataTMP) ==1]=np.nan
#                     rgrDataTMP=rgrTMP
#                 elif rgsVars[va] == 'Rain_PRdays':
#                     rgrDataTMP=np.max(np.reshape(rgrDataTMP, (rgrDataTMP.shape[0]/8,8,rgrDataTMP.shape[1],rgrDataTMP.shape[2])),axis=1)
#                     iE20timeSTART=np.where(rgdTimeDD == datetime.datetime(int(YYYY), int(MM), int(DD1[:2])))[0][0]
#                     iE20timeSTOP=np.where(rgdTimeDD == datetime.datetime(int(YYYY), int(MM), int(DD2[:2])))[0][0]+1
#                     ncid=Dataset('/glade/scratch/prein/ERA-20C/PR_1900-2010.nc', mode='r')
#                     rgrDataPR=(np.squeeze(ncid.variables['tp'][iE20timeSTART:iE20timeSTOP,:,:]))*1000.
#                     ncid.close()
#                     rgrTMP=np.copy(rgrDataTMP); rgrTMP[:]=0
#                     rgrTMP[:,:-1,:][(rgrDataTMP[:,:-1,:] < 0.5) & (rgrDataPR >= 1)]=1
#                     rgrTMP[np.isnan(rgrDataTMP) ==1]=np.nan
#                     rgrDataTMP=rgrTMP
#                 else:
#                     rgrDataTMP=np.mean(np.reshape(rgrDataTMP, (rgrDataTMP.shape[0]/8,8,rgrDataTMP.shape[1],rgrDataTMP.shape[2])),axis=1)
#                 try:
#                     rgrTempData[tt:tt+rgrDataTMP.shape[0],:,:,va]=rgrDataTMP
#                 except:
#                     stop()
#             tt=int(tt+rgrDataTMP.shape[0])
#         rgrMonthlyData[yy,mm,:,:,:]=np.nanmean(rgrTempData[:,:,:,:], axis=0)
# np.save(sSaveDataDir+'ERA20_monthlydata_1900-2009.npy',rgrMonthlyData)

rgrMonthlyData=np.load(sSaveDataDir+'ERA20_monthlydata_1900-2009.npy')
# # convert probability of snow to yes or no
# Psnow=rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]
# Psnow[Psnow>=0.5]=1; Psnow[Psnow<0.5]=0
# rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]=Psnow



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

# # read ERA-Interim daily precipitation sum
# sERA_Int_PR='/gpfs/fs1/scratch/prein/ERA-Interim/PR_1979-2018_Day.nc'
# rgdERAInt_time=pd.date_range(datetime.datetime(1979, 01, 1,12), end=datetime.datetime(2019, 1, 1,12), freq='d')
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
#             for va in range(len(rgsVars)):
#                 if (rgsVars[va] != 'Snow_PRdays') & (rgsVars[va] != 'Rain_PRdays'):
#                     ncid=Dataset(sFileFin, mode='r')
#                     rgrDataTMP=np.squeeze(ncid.variables[rgsVars[va]][:])
#                     ncid.close()
#                 else:
#                     ncid=Dataset(sFileFin, mode='r')
#                     rgrDataTMP=np.squeeze(ncid.variables['ProbSnow'][:])
#                     ncid.close()
#                 if rgsVars[va] == 'ProbSnow':
#                     DATA=np.copy(rgrDataTMP); DATA[:]=0
#                     DATA[rgrDataTMP >=0.5]=1; rgrDataTMP=DATA
#                     rgrDataTMP_EI=np.max(rgrDataTMP,axis=0)
#                 elif rgsVars[va] == 'Snow_PRdays':
#                     rgrDataTMP=np.max(rgrDataTMP,axis=0)
#                     iERAtime=np.where(rgdERAInt_time == datetime.datetime(int(YYYY), int(MM), int(DD),12))[0][0]
#                     ncid=Dataset(sERA_Int_PR, mode='r')
#                     rgrDataPR=(np.squeeze(ncid.variables['tp'][iERAtime,:,:]))*1000.
#                     ncid.close()
#                     rgrTMP=np.copy(rgrDataTMP); rgrTMP[:]=0
#                     rgrTMP[:-1,:][(rgrDataTMP[:-1,:] > 0.5) & (rgrDataPR >= 1)]=1
#                     rgrTMP[np.isnan(rgrDataTMP) ==1]=np.nan
#                     rgrDataTMP_EI=rgrTMP
#                 elif rgsVars[va] == 'Rain_PRdays':
#                     rgrDataTMP=np.max(rgrDataTMP,axis=0)
#                     iERAtime=np.where(rgdERAInt_time == datetime.datetime(int(YYYY), int(MM), int(DD),12))[0][0]
#                     ncid=Dataset(sERA_Int_PR, mode='r')
#                     rgrDataPR=(np.squeeze(ncid.variables['tp'][iERAtime,:,:]))*1000.
#                     ncid.close()
#                     rgrTMP=np.copy(rgrDataTMP); rgrTMP[:]=0
#                     rgrTMP[:-1,:][(rgrDataTMP[:-1,:] <= 0.5) & (rgrDataPR >= 1)]=1
#                     rgrTMP[np.isnan(rgrDataTMP) ==1]=np.nan
#                     rgrDataTMP_EI=rgrTMP
#                 else:
#                     rgrDataTMP_EI=np.mean(rgrDataTMP,axis=0)
#                 rgrTempData_EI[dd,:,:,va]=rgrDataTMP_EI
#         rgrMonthlyData_EI[yy,mm,:,:,:]=np.nanmean(rgrTempData_EI[:,:,:,:], axis=0)
#         rgrTMP=rgrMonthlyData_EI[yy,mm,:,:,rgsVars.index('ProbSnow')]; rgrTMP[rgrTMP>=0.5]=1; rgrTMP[rgrTMP<=0.5]=0
#         rgrMonthlyData_EI[yy,mm,:,:,rgsVars.index('ProbSnow')]=rgrTMP
# np.save(sSaveDataDir+'ERA-Int_monthlydata.npy',rgrMonthlyData_EI)

rgrMonthlyData_EI=np.load(sSaveDataDir+'ERA-Int_monthlydata.npy')
# convert probability of snow to yes or no
# Psnow=rgrMonthlyData_EI[:,:,:,:,rgsVars.index('ProbSnow')]
# Psnow[Psnow>=0.5]=1; Psnow[Psnow<0.5]=0
# rgrMonthlyData_EI[:,:,:,:,rgsVars.index('ProbSnow')]=Psnow


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

for va in [rgsVars.index('Rain_PRdays')]: # 'ProbSnow | Snow_PRdays | Rain_PRdays
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
        biasContDist=2
        sLabel='trend in probability of snow [days per decade]'
        rgrDataVar=rgrMonDat[:,:,:,rgsVarsUW.index('Snow')]
        Yminmax=[-23,23]
    elif rgsVars[va] == 'Snow_PRdays':
        biasContDist=2
        sLabel='trend in snow days [days per decade]'
        rgrDataVar=rgrMonDat[:,:,:,rgsVarsUW.index('Snow_PRdays')]
        Yminmax=[-7,7]
    elif rgsVars[va] == 'Rain_PRdays':
        biasContDist=2
        sLabel='trend in rain days [days per decade]'
        rgrDataVar=rgrMonDat[:,:,:,rgsVarsUW.index('Rain_PRdays')]
        Yminmax=[-12,12]
    elif rgsVars[va] == 'T2M':
        biasContDist=0.16
        sLabel='trend in 2m temperature [$^{\circ}$C per decade]'
    elif rgsVars[va] == 'BflLR':
        biasContDist=0.05
        sLabel='trend in Surf-ML Lapse Rate [$^{\circ}$C km$^{-1}$ per decade]'
        rgrDataVar=rgrLRbF

    fig = plt.figure(figsize=(15, 11))
    rgsSeasons=['annual','DJF','MAM','JJA','SON']
    rgsDaysSeas=[365.25,90.25,92,92,91]
    for se in range(5):
        print '    work on '+rgsSeasons[se]
        if se == 0:
            gs1 = gridspec.GridSpec(1,1)
            gs1.update(left=0.01, right=0.64,
                       bottom=0.5, top=0.95,
                       wspace=0.2, hspace=0.1)
            ax = plt.subplot(gs1[0,0], projection=cartopy.crs.Robinson())
            rgiTime=(rgdTimeFin.month <= 12)
            rgiMonths=[0,1,2,3,4,5,6,7,8,9,10,11]
            iMS=7 # marker size for RO stations
        else:
            iXX=[0,1,0,1]
            iYY=[0,0,1,1]
            gs1 = gridspec.GridSpec(2,2)
            gs1.update(left=0.01, right=0.64,
                       bottom=0.1, top=0.47,
                       wspace=0.05, hspace=0.05)
            ax = plt.subplot(gs1[iYY[se-1],iXX[se-1]], projection=cartopy.crs.Mercator())
            ax.set_extent([-175, 160, 15, 80], crs=ccrs.PlateCarree())
            # ax.set_adjustable('datalim')
            if rgsSeasons[se] == 'SON': 
                rgiTime=((rgdTimeFin.month <= 11) & (rgdTimeFin.month >= 9))
                rgiMonths=[8,9,10]
            if rgsSeasons[se] == 'MAM': 
                rgiTime=((rgdTimeFin.month <= 5) & (rgdTimeFin.month >= 3))
                rgiMonths=[2,3,4]
            if rgsSeasons[se] == 'DJF': 
                rgiTime=((rgdTimeFin.month == 12) | (rgdTimeFin.month <= 2))
                rgiMonths=[0,1,11]
            if rgsSeasons[se] == 'JJA': 
                rgiTime=((rgdTimeFin.month >= 6) & (rgdTimeFin.month <= 8))
                rgiMonths=[5,6,7]
            iMS=4

        iScale=rgsDaysSeas[se]
        
       
        rgrColorTable=np.array(['#4457c9','#4f67d4','#6988ea','#84a6f9','#9ebdff','#b7d0fd','#cfdcf3','#ffffff','#f1d7c8','#f9c5ab','#f7aa8c','#f28d72','#df6553','#c93137','#bc052b'])
        iContNr=len(rgrColorTable)
        iMinMax=12.*biasContDist/2.+biasContDist/2.
        clevsTD=np.arange(-iMinMax,iMinMax+0.0001,biasContDist)
    
        ###################################
        # calculate and plot trends
        # ax = plt.subplot(gs1[YY[se],XX[se]], projection=ccrs.Robinson(central_longitude=0))
         #, projection=cartopy.crs.PlateCarree())
        # ax.outline_patch.set_linewidth(0)
        plt.axis('off')
        # ax.set_global()
    
        rgrGCtrend=np.zeros((len(rgrLon))); rgrGCtrend[:]=np.nan
        rgrPvalue=np.copy(rgrGCtrend)
        rgrR2=np.copy(rgrGCtrend)
        # get the data
        rgrDataAct=rgrDataVar[:,rgiMonths,:]
        rgrDataAnnual=np.nanmean(rgrDataAct, axis=1)# np.zeros((len(iYears),rgrDataAct.shape[1])); rgrDataAnnual[:]=np.nan
        # for yy in range(len(iYears)):
        #     rgiYYact=(rgdTimeFin[rgiTime].year == iYears[yy])
        #     rgrDataAnnual[yy,:]=np.nanmean(rgrDataAct[rgiYYact,:], axis=0)
        #     # remove data with too little coverage
        #     iCoverage=np.sum(np.isnan(rgrDataAct[rgiYYact,:]),axis=0)/float(sum(rgiYYact))
        #     rgrDataAnnual[yy,(iCoverage> 1.-iMinCov)]=np.nan

        for st in range(len(rgrLon)):
            # remove nans if there are some
            rgiReal=~np.isnan(rgrDataAnnual[(iYears <= 2009),st])
            if (np.mean(rgrDataAnnual[(iYears <= 2009),st][rgiReal]*iScale) < 10) | (len(rgiReal) < (2009-1979)*0.7):
                continue
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(iYears[(iYears <= 2009)][rgiReal],rgrDataAnnual[(iYears <= 2009),st][rgiReal])
            except:
                continue
            rgrGCtrend[st]=slope*10*iScale #(slope/np.nanmean(rgrDataAnnual[:,st]))*1000.
            rgrPvalue[st]=p_value
            rgrR2[st]=r_value**2
    
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
                    ax.plot(rgrLon[st], rgrLat[st],'o',color=rgrColorTable[iColor], ms=iMS, mec='k', alpha=1, markeredgewidth=1, transform=cartopy.crs.PlateCarree(), zorder=20)
                else:
                    # not significant trend
                    try:
                        ax.plot(rgrLon[st], rgrLat[st],'o',color=rgrColorTable[iColor], ms=iMS, mec=rgrColorTable[iColor], alpha=1, markeredgewidth=0.0,transform=cartopy.crs.PlateCarree(), zorder=20)
                    except:
                        stop()

        
        ###############################################################
        print '    Add ERA-20C analysis'
        rgrMonthAct=rgrMonthlyData
        rgrLonAct=rgrLonGrid2D
        rgrLatAct=rgrLatGrid2D
    
        rgrTrendsERA20=np.zeros((rgrMonthAct.shape[2],rgrMonthAct.shape[3],5)); rgrTrendsERA20[:]=np.nan
        rgrSeasonalData=np.mean(rgrMonthAct[:,rgiMonths,:,:,va][:,(rgiYY >=1979)],axis=0)
        for la in range(rgrSeasonalData.shape[1]):
            for lo in range(rgrSeasonalData.shape[2]):
                rgrTrendsERA20[la,lo,:]=stats.linregress(rgiYY[(rgiYY >=1979)],rgrSeasonalData[:,la,lo])

        # plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,colors=rgrColorTable, extend='both',levels=clevsTD) #, alpha=0.6)
        # cs=plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,cmap=plt.cm.coolwarm, extend='both',levels=clevsTD, zorder=0) #, alpha=0.6)
        cs=plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,colors=rgrColorTable, extend='both',levels=clevsTD, zorder=0,transform=cartopy.crs.PlateCarree()) #, alpha=0.6)

        # plot significant layer
        for la in range(rgrSeasonalData.shape[1])[::2]:
            for lo in range(rgrSeasonalData.shape[2])[::2]:
                if rgrTrendsERA20[la,lo,3] < 0.05:
                    ax.plot(rgrLonAct[la,lo], rgrLatAct[la,lo],'o',color='k', ms=2, mec=rgrColorTable[iColor], alpha=0.5, markeredgewidth=0.0,transform=cartopy.crs.PlateCarree(), zorder=15)
    
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5, zorder=11)
        ax.add_feature(cartopy.feature.COASTLINE, zorder=11)
        ax.add_feature(cartopy.feature.STATES, zorder=11, alpha=.5)
        # ax.gridlines(zorder=11)
        plt.axis('off')
    
        print '    add colorbar'
        CbarAx = axes([0.01, 0.05, 0.63, 0.02])
        cb = colorbar(cs, cax = CbarAx, orientation='horizontal', extend='max', ticks=clevsTD, extendfrac='auto')
        cb.ax.set_title(sLabel)
    
        # # add 5 days per year contour
        # rgrPsnowClim=np.mean(rgrMonthAct[(rgiYY >=1979),:,:,:,va], axis=(0,1))
        # ax.contour(rgrLonAct,rgrLatAct,rgrPsnowClim*iScale,levels=[10], zorder=20,transform=cartopy.crs.PlateCarree())
        # rgrPsnowClim=np.mean(rgrMonthAct[(rgiYY <=1930),:,:,:,va], axis=(0,1))
        # ax.contour(rgrLonAct,rgrLatAct,rgrPsnowClim*iScale,levels=[10], zorder=20,transform=cartopy.crs.PlateCarree())

        # lable the maps
        tt = ax.text(0.03,0.99, rgsLableABC[se]+') '+rgsSeasons[se] , ha='left',va='top', \
                    transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=20)
        tt.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='w'))


    # ==============================================================================
    # ==============================================================================
    # Add time lines for subregions

    rgrT2Mmonth=np.load('ERA-20C_T2M.npy')
    rgsVars=rgsVars+['T2M']
    rgrMonthlyData=np.append(rgrMonthlyData,rgrT2Mmonth[:,:,:,:,None], axis=4)

    # read in the subregions
    gs1 = gridspec.GridSpec(3,1)
    gs1.update(left=0.7, right=0.95,
               bottom=0.05, top=0.95,
               wspace=0.2, hspace=0.2)
    
    rgsSubregions=['Africa','Asia','North-America','South-America','Australia','Europe']
    sContinents='/glade/u/home/prein/ShapeFiles/Continents/'
    grShapeFiles={}
    for sr in range(len(rgsSubregions)):
        ctr = shapefile.Reader(sContinents+rgsSubregions[sr]+'_poly')
        geomet = ctr.shapeRecords() # will store the geometry separately
        for sh in range(len(geomet)):
            first = geomet[sh]
            grShapeFiles[rgsSubregions[sr]]=first.shape.points

    if rgsVars[va] == 'ProbSnow':
        biasContDist=20
        sLabel='prob. of snow'
        sUnit='[days]'
        rgrDataVar=rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]
        rgrDataVarEI=rgrMonthlyData_EI[:,:,:,:,rgsVars.index('ProbSnow')]
        rgrROdata=rgrMonDat[:,:,:,rgsVarsUW.index('Snow')]
    elif rgsVars[va] == 'Snow_PRdays':
        biasContDist=20
        sLabel='snow days'
        sUnit='[days]'
        rgrDataVar=rgrMonthlyData[:,:,:,:,rgsVars.index('Snow_PRdays')]
        rgrDataVarEI=rgrMonthlyData_EI[:,:,:,:,rgsVars.index('Snow_PRdays')]
        rgrROdata=rgrMonDat[:,:,:,rgsVarsUW.index('Snow_PRdays')]
    elif rgsVars[va] == 'Rain_PRdays':
        biasContDist=20
        sLabel='rain days'
        sUnit='[days]'
        rgrDataVar=rgrMonthlyData[:,:,:,:,rgsVars.index('Rain_PRdays')]
        rgrDataVarEI=rgrMonthlyData_EI[:,:,:,:,rgsVars.index('Rain_PRdays')]
        rgrROdata=rgrMonDat[:,:,:,rgsVarsUW.index('Rain_PRdays')]
    
    rgiFin=np.sum(~np.isnan(rgrROdata[:,:,:]), axis=(0,1))
    rgiFin=(rgiFin >= 3*26)
    rgrROdataAct=rgrROdata[:,:,:][:,:,rgiFin]
    rgrLonAct=rgrLonS[rgiFin]
    rgrLatAct=rgrLatS[rgiFin]
    rgrDataVarEIAct=rgrDataVarEI[:,:,:]
    rgrDataVarAct=rgrDataVar[:,:,:]
    rgrDataVarActT2M=rgrMonthlyData[:,:,:,:,rgsVars.index('T2M')]

    if (rgsVars[va] == 'ProbSnow') | (rgsVars[va] == 'Snow_PRdays') | (rgsVars[va] == 'Rain_PRdays'):
        # if se == 0:
        rgrDataVarAct=rgrDataVarAct*365.25
        rgrDataVarEIAct=rgrDataVarEIAct*365.25
        rgrROdataAct=rgrROdataAct*365.25


    ii=0
    # iYrange=[7,7,7]
    for sr in [2,5,1]:
        ax = plt.subplot(gs1[ii,0])
        rgdShapeAct=grShapeFiles[rgsSubregions[sr]]
        PATH = path.Path(rgdShapeAct)
        for da in range(len(rgsData)):
            if rgsData[da] == 'Rad-Soundings':
                TEST=np.copy(rgrLonAct)
                TEST[TEST>180]=TEST[TEST>180]-360
                flags = PATH.contains_points(np.hstack((TEST[:,np.newaxis],rgrLatAct[:,np.newaxis])))
                rgiROstations=np.sum(flags)
                rgiYearsAct=np.array(range(1979,2018,1))
                rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2009))
                rgrTMPData=np.nanmean(np.nanmean(rgrROdataAct[:,:,flags],axis=(1))-np.nanmean(rgrROdataAct[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                sColor='k'
            if rgsData[da] == 'ERA-20C':
                TEST=np.copy(rgrLonGrid2D)
                TEST[TEST>180]=TEST[TEST>180]-360
                flags = PATH.contains_points(np.hstack((TEST.flatten()[:,np.newaxis],rgrLatGrid2D.flatten()[:,np.newaxis])))
                rgiYearsAct=np.array(range(1900,2010,1))
                rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2009))
                rgrDataVarTMP=np.reshape(rgrDataVarAct, (rgrDataVarAct.shape[0],rgrDataVarAct.shape[1],rgrDataVarAct.shape[2]*rgrDataVarAct.shape[3]))
                rgrTMPData=np.nanmean(np.nanmean(rgrDataVarTMP[:,:,flags],axis=(1))-np.nanmean(rgrDataVarTMP[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                # T2M
                rgrDataVarTMPT2M=np.reshape(rgrDataVarActT2M, (rgrDataVarActT2M.shape[0],rgrDataVarActT2M.shape[1],rgrDataVarActT2M.shape[2]*rgrDataVarActT2M.shape[3]))
                rgrTMPDataT2M=np.nanmean(np.nanmean(rgrDataVarTMPT2M[:,:,flags],axis=1)-np.nanmean(rgrDataVarTMPT2M[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                sColor='#1f78b4'
            if rgsData[da] == 'ERA-Interim':
                TEST=np.copy(rgrLonGrid2D_EI)
                TEST[TEST>180]=TEST[TEST>180]-360
                flags = PATH.contains_points(np.hstack((TEST.flatten()[:,np.newaxis],rgrLatGrid2D_EI.flatten()[:,np.newaxis])))
                rgiYearsAct=np.array(range(1979,2018,1))
                rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2009))
                rgrDataVarTMP=np.reshape(rgrDataVarEIAct, (rgrDataVarEIAct.shape[0],rgrDataVarEIAct.shape[1],rgrDataVarEIAct.shape[2]*rgrDataVarEIAct.shape[3]))
                try:
                    rgrTMPData=np.nanmean(np.nanmean(rgrDataVarTMP[:,:,flags],axis=(1))-np.nanmean(rgrDataVarTMP[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                except:
                    stop()
                sColor='#e31a1c'


            plt.plot(rgiYearsAct, rgrTMPData, c=sColor, lw=0.5, zorder=3, alpha=0.5)
            plt.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPData,10), c=sColor, lw=3, zorder=3, label=rgsData[da])
            # plt.plot(iYears_I[iComYears], rgrModData, c='#1f78b4')
            # plt.plot(iYears_I[iComYears], rgrModDataEI, c='#e31a1c')

            plt.title(rgsLableABC[ii+5]+') '+rgsSubregions[sr]+' | '+str(rgiROstations))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if ii == 2:
                ax.set_xlabel('Year')
            else:
                plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
            ax.set_ylabel(sLabel+' '+sUnit)
            ax.set(ylim=(Yminmax[0],Yminmax[1]))

            if rgsData[da] == 'ERA-20C':
                Y2col='#636363'
                # Plot T2M on a second y-axis
                ax2 = ax.twinx()
                ax2.set_ylabel('2m temperature (T2M) [$^{\circ}$C]', color=Y2col)
                ax2.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPDataT2M,10), c=Y2col, lw=3, zorder=1, alpha=0.5)
                ax2.plot(rgiYearsAct, rgrTMPDataT2M, c=Y2col, lw=0.5, zorder=1, alpha=0.5)
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
                      ncol=1, prop={'size':12})
        ii=ii+1
        

    sPlotFile=PlotDir
    sPlotName= rgsVars[va]+'-Changes.pdf'
    if os.path.isdir(sPlotFile) != 1:
        subprocess.call(["mkdir","-p",sPlotFile])
    print '        Plot map to: '+sPlotFile+sPlotName
    fig.savefig(sPlotFile+sPlotName)

    stop()
