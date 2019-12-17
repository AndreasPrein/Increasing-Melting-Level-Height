#!/usr/bin/env python
'''
    File name: ERA-2C_evaluation.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    Read in the preprocessed data from:
    ~/papers/Trends_RadSoundings/programs/RadioSoundings/UWY_Preprocessor/HailParametersFromRadioSounding.py
    and
    ~/papers/Trends_RadSoundings/programs/RadioSoundings/IGRA_Preprocessor/Indices_from_IGRA-soundings.py

    Calculate monthly mean indices and save the data in python format

    Derive and save the indices at the same location/time from ERA-20C preprocessed by:
    ~/papers/Trends_RadSoundings/programs/ERA-20C-Data/ERA-20C-Data-processing.py


    Plot statistical analysis of seasonal absolut biases, interannual variability, annual cycle RMSE...

'''

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
# import ESMF
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
rgsVars=['WBZheight','ThreeCheight','ProbSnow', 'BflLR','LCL', 'CAPE', 'SH06']
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/ERA-20C_Evaluation/'
rgsSeasons=['DJF','MAM','JJA','SON', 'Annual']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30,365.25]



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
#             READ IN THE U-WYOMING  RADIO SOUNDING DATA
sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/'
sRadSoundData=sSaveDataDir+'RadSoundHailEnvDat-1979-2017_incl-CAPE.pkl'
grStatisitcs = pickle.load(open(sRadSoundData, "rb"))
rgrWBZD=grStatisitcs['rgrWBZ']
rgrThreeCheight=grStatisitcs['rgrThreeCheight']
rgrLRbF=grStatisitcs['rgrLRbF']
rgrProbSnow=grStatisitcs['rgrProbSnow']
rgrCAPE=grStatisitcs['CAPE']
rgrCIN=grStatisitcs['CIN']
rgrLCL=grStatisitcs['LCL']
rgrLFC=grStatisitcs['LFC']
rgrVS03=grStatisitcs['VS0_3']
rgrVS06=grStatisitcs['VS0_6']
rgsVariables=grStatisitcs['rgsVariables']
rgdTimeFin=grStatisitcs['rgdTime']
rgrLon=grStatisitcs['rgrLon']
rgrLat=grStatisitcs['rgrLat']
rgrElev=grStatisitcs['rgrElev']
rgsStatID=np.array(grStatisitcs['rgsStatID'])
rgsStatNr=np.array(grStatisitcs['rgsStatNr'])
iYears=np.unique(rgdTimeFin.year)

# convert to monthly data
# CALCULATE MONTHY AVERAGES WITH THE REQUIREMENT THAT WE HAVE AT LEAST 80 % COVERAGE
rgsVarsUW=['Snow','WBZ','ThreeC','FLH','TS','LRbF', 'CAPE','CIN','LCL','LFC','SH03','SH06']
rgrMonUW=np.zeros((len(iYears),12,rgrWBZD.shape[1],len(rgsVarsUW))); rgrMonUW[:]=np.nan
for yy in range(len(iYears)):
    print('    Process Year '+str(iYears[yy]))
    for mm in range(12):
        rgiTimes=((rgdTimeFin.year == iYears[yy]) & (rgdTimeFin.month == (mm+1)))
        for va in range(len(rgsVarsUW)):
            if rgsVarsUW[va] == 'Snow':
                rgrData=rgrProbSnow[rgiTimes,:]
                rgrData[rgrData >= 0.5]=1; rgrData[rgrData < 0.5]= 0
            if rgsVarsUW[va] == 'WBZ':
                rgrData=rgrWBZD[rgiTimes,:]
            if rgsVarsUW[va] == 'ThreeC':
                rgrData=rgrThreeCheight[rgiTimes,:]
            if rgsVarsUW[va] == 'FLH':
                continue
            if rgsVarsUW[va] == 'TS':
                continue
            if rgsVarsUW[va] == 'LRbF':
                rgrData=rgrLRbF[rgiTimes,:]
            if rgsVarsUW[va] == 'CAPE':
                rgrData=rgrCAPE[rgiTimes,:]
            if rgsVarsUW[va] == 'CIN':
                rgrData=rgrCIN[rgiTimes,:]
            if rgsVarsUW[va] == 'LCL':
                rgrData=rgrLCL[rgiTimes,:]
            if rgsVarsUW[va] == 'LFC':
                rgrData=rgrLFC[rgiTimes,:]
            if rgsVarsUW[va] == 'SH03':
                rgrData=rgrVS03[rgiTimes,:]
            if rgsVarsUW[va] == 'SH06':
                rgrData=rgrVS06[rgiTimes,:]
            rgiNaNs=np.sum(np.isnan(rgrData), axis=0)/float(sum(rgiTimes))
            rgrMonUW[yy,mm,rgiNaNs < 0.2,va]=np.nanmean(rgrData[:,rgiNaNs < 0.2], axis=0)

np.save('UW-Monthly-RS.npy',rgrMonUW)
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


# ################################################################################
# ################################################################################
# ################################################################################
# #          Process ERA-20
# ################################################################################

# dStartDay=datetime.datetime(1979, 1, 1,0)
dStartDay=datetime.datetime(2000, 1, 1,0)
dStopDay=datetime.datetime(2009, 12, 31,23)
# dStopDay=datetime.datetime(2010, 12, 31,23)
rgdTime6H = pd.date_range(dStartDay, end=dStopDay, freq='3h')
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTimeMM = pd.date_range(dStartDay, end=dStopDay, freq='m')
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
#             for va in range(len(rgsVars)):
#                 if rgsVars[va] in ['CAPE','CIN','LCL','LFC','SH03','SH06']:
#                     sFileFin=sSaveDataDir+'ERA-20c/Full/'+sDate+'_ERA-20c_Hail-Env_RDA.nc'
#                 else:
#                     sFileFin=sSaveDataDir+'ERA-20c/'+sDate+'_ERA-20c_Hail-Env_RDA.nc'
#                 ncid=Dataset(sFileFin, mode='r')
#                 rgrDataTMP=np.squeeze(ncid.variables[rgsVars[va]][:])
#                 if rgsVars[va] != 'ProbSnow':
#                     rgrDataTMP=np.mean(np.reshape(rgrDataTMP, (rgrDataTMP.shape[0]/8,8,rgrDataTMP.shape[1],rgrDataTMP.shape[2]))[:,[0,4],:,:],axis=1)
#                 else:
#                     rgrDataTMP=np.max(np.reshape(rgrDataTMP, (rgrDataTMP.shape[0]/8,8,rgrDataTMP.shape[1],rgrDataTMP.shape[2]))[:,[0,4],:,:],axis=1)
#                 try:
#                     rgrTempData[tt:tt+rgrDataTMP.shape[0],:,:,va]=rgrDataTMP
#                 except:
#                     stop()
#             ncid.close()
#             tt=int(tt+rgrDataTMP.shape[0])
#         rgrMonthlyData[yy,mm,:,:,:]=np.nanmean(rgrTempData[:,:,:,:], axis=0)
#         rgrTMP=rgrMonthlyData[yy,mm,:,:,rgsVars.index('ProbSnow')]; rgrTMP[rgrTMP>=0.5]=1; rgrTMP[rgrTMP<=0.5]=0
#         rgrMonthlyData[yy,mm,:,:,rgsVars.index('ProbSnow')]=rgrTMP
# np.save(sSaveDataDir+'ERA20_monthlydata.npy',rgrMonthlyData)

rgrMonthlyData=np.load(sSaveDataDir+'ERA20_monthlydata.npy')
# convert probability of snow to yes or no
# Psnow=rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]
# Psnow[Psnow>=0.5]=1; Psnow[Psnow<0.5]=0
# rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]=Psnow

rgrMonthlyData[:,:,:,:,rgsVars.index('BflLR')]=rgrMonthlyData[:,:,:,:,rgsVars.index('BflLR')]





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

rgrMonthlyData_EI=np.zeros((len(rgiYY_EI),12,len(rgrLat75_EI), len(rgrLon75_EI),len(rgsVars))); rgrMonthlyData_EI[:]=np.nan
for yy in range(len(rgiYY_EI)): #range(len(TC_Name)): # loop over hurricanes
    print ' '
    print '    Workin on year '+str(rgiYY_EI[yy])
    # check if we already processed this file
    for mm in range(12): #  loop over time
        rgiDD_EI=np.sum(((rgdTimeDD_EI.year == rgiYY_EI[yy]) & (rgdTimeDD_EI.month == (mm+1))))
        rgrTempData_EI=np.zeros((rgiDD_EI,len(rgrLat75_EI), len(rgrLon75_EI),len(rgsVars))); rgrTempData_EI[:]=np.nan
        YYYY=str(rgiYY_EI[yy])
        MM=str("%02d" % (mm+1))
        for dd in range(rgiDD_EI):
            DD=str("%02d" % (dd+1))
            sDate=YYYY+MM+DD
            for va in range(len(rgsVars)):
                if rgsVars[va] in ['CAPE','CIN','LCL','LFC','SH03','SH06']:
                    sFileFin=sSaveDataDir+'ERA-20c/Full/'+sDate+'_ERA-20c_Hail-Env_RDA.nc'
                else:
                    sFileFin=sSaveDataDir+'ERA-Int/'+sDate+'_ERA-Int_Hail-Env_RDA.nc'
                ncid=Dataset(sFileFin, mode='r')
                rgrDataTMP=np.squeeze(ncid.variables[rgsVars[va]][:])
                if rgsVars[va] != 'ProbSnow':
                    rgrDataTMP_EI=np.mean(rgrDataTMP,axis=0)
                else:
                    rgrDataTMP_EI=np.max(rgrDataTMP,axis=0)
                try:
                    rgrTempData_EI[dd,:,:,va]=rgrDataTMP_EI
                except:
                    stop()
            ncid.close()
        rgrMonthlyData_EI[yy,mm,:,:,:]=np.nanmean(rgrTempData_EI[:,:,:,:], axis=0)
        rgrTMP=rgrMonthlyData_EI[yy,mm,:,:,rgsVars.index('ProbSnow')]; rgrTMP[rgrTMP>=0.5]=1; rgrTMP[rgrTMP<=0.5]=0
        rgrMonthlyData_EI[yy,mm,:,:,rgsVars.index('ProbSnow')]=rgrTMP
np.save(sSaveDataDir+'ERA-Int_monthlydata.npy',rgrMonthlyData_EI)

rgrMonthlyData_EI=np.load(sSaveDataDir+'ERA-Int_monthlydata.npy')[:-8]
# convert probability of snow to yes or no
# Psnow=rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]
# Psnow[Psnow>=0.5]=1; Psnow[Psnow<0.5]=0
# rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]=Psnow




# =======================================================
# =======================================================
# CALCULTE THE EVALUATION STATISTICS AT STATION LOCATIONS
# =======================================================

# # combine the RO datasets
rgrMonDat=np.append(rgrMonUW,rgrMonthDat_I, axis=2)
rgrLonS=np.append(rgrLon,rgrLon_I, axis=0); rgrLonS[rgrLonS<0]=rgrLonS[rgrLonS<0]+360
rgrLatS=np.append(rgrLat,rgrLat_I, axis=0)
# rgrMonDat=rgrMonUW
# rgrLonS=np.array(rgrLon); rgrLonS[rgrLonS<0]=rgrLonS[rgrLonS<0]+360
# rgrLatS=rgrLat
rgrERA20C_stations=np.zeros((rgrMonthlyData.shape[0],rgrMonthlyData.shape[1],len(rgrLonS),rgrMonthlyData.shape[4])); rgrERA20C_stations[:]=np.nan
for st in range(len(rgrLonS)):
    iLonMin=np.argmin(np.abs(rgrLon75-rgrLonS[st]))
    iLatMin=np.argmin(np.abs(rgrLat75-rgrLatS[st]))
    try:
        rgrERA20C_stations[:,:,st,:]=rgrMonthlyData[:,:,iLatMin,iLonMin,:]
    except:
        stop()
np.save('ERA-20C-at-RSloc.npy',rgrERA20C_stations)
rgrERA20C_stations=np.load('ERA-20C-at-RSloc.npy')


# rgrERA_EI_stations=np.zeros((rgrMonthlyData_EI.shape[0],rgrMonthlyData_EI.shape[1],len(rgrLonS),rgrMonthlyData_EI.shape[4])); rgrERA_EI_stations[:]=np.nan
# for st in range(len(rgrLonS)):
#     iLonMin=np.argmin(np.abs(rgrLon75_EI-rgrLonS[st]))
#     iLatMin=np.argmin(np.abs(rgrLat75_EI-rgrLatS[st]))
#     try:
#         rgrERA_EI_stations[:,:,st,:]=rgrMonthlyData_EI[:,:,iLatMin,iLonMin,:]
#     except:
#         stop()
# np.save('ERA-EI-at-RSloc.npy',rgrERA_EI_stations)
rgrERA_EI_stations=np.load('ERA-EI-at-RSloc.npy')



# =======================================================
# =======================================================
# PLOT THE EVALUATION FOR SINGLE STATIONS
# =======================================================

iComYears=((rgiYY[0] <= iYears_I) & (rgiYY[-1] >= iYears_I))
rgsStatistics=['bias','stddev']

# Do this for ERA-20C and ERA-Interim
rgsDataset=['ERA-20C','ERA-Interim']

for da in [0]: #range(len(rgsDataset)):
    if da == 0:
        rgrDATA=rgrERA20C_stations
    else:
        rgrDATA=rgrERA_EI_stations
    for sta in range(len(rgsStatistics)):
        for va in [4]: #range(len(rgsVars)):
            print('    Plotting '+rgsVars[va])
            if rgsVars[va] == 'WBZheight':
                biasContDist=20
                sLabel='melting level height'
                sUnit='[m]'
                rgrDataVar=rgrDATA[:,:,:,rgsVars.index('WBZheight')]
                rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('WBZ')]
                rgrFullData=rgrMonthlyData[:,:,:,:,rgsVars.index('WBZheight')]
                levelsBG=np.array([   0.,  100., 250., 500., 1000., 2000., 3000., 5000., 7000.])
            elif rgsVars[va] == 'ThreeCheight':
                biasContDist=20
                sLabel='3 $^{\circ}$C height'
                sUnit='[m]'
                rgrDataVar=rgrDATA[:,:,:,rgsVars.index('ThreeCheight')]
                rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('ThreeC')]
                rgrFullData=rgrMonthlyData[:,:,:,:,rgsVars.index('ThreeCheight')]
                levelsBG=np.array([   0.,  100., 250., 500., 1000., 2000., 4000., 6000., 9000.])
            elif rgsVars[va] == 'ProbSnow':
                biasContDist=20
                sLabel='probability of snow'
                sUnit='[days]'
                rgrDataVar=rgrDATA[:,:,:,rgsVars.index('ProbSnow')]
                rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('Snow')]
                rgrFullData=rgrMonthlyData[:,:,:,:,rgsVars.index('ProbSnow')]
                levelsBG=np.array([   0.,  1., 5., 10., 20., 40., 80., 120., 160.])
            elif rgsVars[va] == 'BflLR':
                biasContDist=20
                sLabel='Lapse rate below freezing level'
                sUnit='[m]'
                rgrDataVar=rgrDATA[:,:,:,rgsVars.index('BflLR')]
                rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('LRbF')]
                rgrFullData=rgrMonthlyData[:,:,:,:,rgsVars.index('BflLR')]
                levelsBG=np.arange(-8,2,1)
            elif rgsVars[va] == 'LCL':
                biasContDist=20
                sLabel='Lifting Condensation Level'
                sUnit='[m]'
                rgrDataVar=rgrDATA[:,:,:,rgsVars.index('LCL')]
                rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('LCL')]
                rgrFullData=rgrMonthlyData[:,:,:,:,rgsVars.index('LCL')]
                levelsBG=np.array([   0.,  100., 250., 500., 1000., 2000., 3000., 5000., 7000.])
            elif rgsVars[va] == 'CAPE':
                biasContDist=20
                sLabel='CAPE'
                sUnit='[J kg$^{-1}$]'
                rgrDataVar=rgrDATA[:,:,:,rgsVars.index('CAPE')]
                rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('CAPE')]
                rgrFullData=rgrMonthlyData[:,:,:,:,rgsVars.index('CAPE')]
                levelsBG=np.array([   0.,  100., 250., 500., 1000., 2000., 3000., 5000., 7000.])/4.
            elif rgsVars[va] == 'SH06':
                biasContDist=20
                sLabel='Wind Shear 0-6 km'
                sUnit='[m s$^{-2}$]'
                rgrDataVar=rgrDATA[:,:,:,rgsVars.index('SH06')]
                rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('SH06')]
                rgrFullData=rgrMonthlyData[:,:,:,:,rgsVars.index('SH06')]
                levelsBG=np.array([   0.,  1., 2., 4., 7., 10., 14., 18., 25.])*2.

            fig = plt.figure(figsize=(15, 7))
            gs1 = gridspec.GridSpec(2,2)
            gs1.update(left=0.05, right=0.9,
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
                print('    Work on Season '+str(se))
                if se == 0:
                    rgiTime=np.array([0,1,11])
                elif se == 1:
                    rgiTime=np.array([2,3,4])
                elif se == 2:
                    rgiTime=np.array([5,6,7])
                elif se == 3:
                    rgiTime=np.array([8,9,10])
        
                # remove stations with too little coverage --> min are 16 months
                try:
                    rgiFin=np.sum(~np.isnan(rgrROdata[:,rgiTime,:]), axis=(0,1))
                except:
                    stop()
                rgiFin=(rgiFin >= 3*10)
                try:
                    rgrDataVarAct=rgrDataVar[:,rgiTime,:][:,:,rgiFin]
                except:
                    stop()
                rgrROdataAct=rgrROdata[:,rgiTime,:][:,:,rgiFin]
                rgrLonAct=rgrLonS[rgiFin]
                rgrLatAct=rgrLatS[rgiFin]
                rgrFullAct=np.mean(rgrFullData[:,rgiTime,:,:], axis=(0,1))
                if rgsVars[va] == 'ProbSnow':
                    rgrDataVarAct=rgrDataVarAct*iDaysInSeason[se]
                    rgrROdataAct=rgrROdataAct*iDaysInSeason[se]
                    rgrFullAct=rgrFullAct*iDaysInSeason[se]
        
                rgrColorTable=np.array(['#4f67d4','#6988ea','#84a6f9','#9ebdff','#b7d0fd','#cfdcf3','#e2e1e0','#f1d7c8','#f9c5ab','#f7aa8c','#f28d72','#df6553','#c93137','#bc052b'])
                iContNr=len(rgrColorTable)+1
                # clevsTD=np.arange(0, iContNr*biasContDist,biasContDist)-(iContNr*biasContDist)/2.+biasContDist/2.
                iMinMax=12.*biasContDist/2.+biasContDist/2.
                clevsTD=np.arange(-iMinMax,iMinMax+0.0001,biasContDist)
        
        
                ###################################
                # calculate the statistics
                iNaN=np.isnan(rgrROdataAct)
                rgrDataVarAct[iNaN]=np.nan
                if rgsStatistics[sta] == 'bias':
                    # rgrStat=np.nanmean(((rgrDataVarAct-rgrROdataAct)/rgrROdataAct)*100., axis=(0,1))
                    rgrStat=((np.nanmean(rgrDataVarAct, axis=(0,1)) - np.nanmean(rgrROdataAct, axis=(0,1)))/np.nanmean(rgrROdataAct, axis=(0,1)))*100.
                    # if rgsVars[va] != 'ProbSnow'):
                    #     rgrStat[np.nanmean(rgrROdataAct, axis=(0,1)) < 10]=np.nan
                if rgsStatistics[sta] == 'stddev':
                    STDobs=np.nanstd(np.nanmean(rgrROdataAct,axis=1), axis=0)
                    STDmod=np.nanstd(np.nanmean(rgrDataVarAct,axis=1), axis=0)
                    rgiVar=((STDobs < 0.1))
                    rgrStat=((STDmod/STDobs)-1)*100.
                    rgrStat[rgiVar]=np.nan
                    rgrFullAct=np.nanstd(np.mean(rgrFullData[:,rgiTime,:,:], axis=(1)), axis=0)
                    if se == 0:
                        levelsBG=levelsBG/20.
                        if rgsVars[va] == 'ProbSnow':
                            levelsBG=levelsBG/10.
                        if rgsVars[va] == 'BflLR':
                            levelsBG=np.arange(0,2.2,0.2)
                    # rgrStat[rgiVar]=0
                    # rgrStat=((STDmod[rgiVar]/STDobs[rgiVar])-1)*100.
                    # rgrLonAct=rgrLonAct[rgiVar]
                    # rgrLatAct=rgrLatAct[rgiVar]
                    
        
                # ax = plt.subplot(gs1[YY[se],XX[se]], projection=ccrs.Robinson(central_longitude=0))
                ax = plt.subplot(gs1[YY[se],XX[se]], projection=cartopy.crs.PlateCarree())
                ax.set_extent([-175, 160, -60, 80], crs=ccrs.PlateCarree())
                # ax.set_global()
        
                # plot the ERA20C data in the packground
                rgrColBackg=['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000']
                if rgsVars[va] == 'BflLR':
                    rgrColBackg=rgrColBackg[::-1]
                ct2=plt.contourf(rgrLonGrid2D,rgrLatGrid2D,rgrFullAct, cmap=plt.cm.binary, extend='max',levels=levelsBG, zorder=1)
                CbarAx = axes([0.93, 0.1, 0.01, 0.8])
                cb2 = colorbar(ct2, cax = CbarAx, orientation='vertical', extend='max', extendfrac='auto', ticks=levelsBG)
                cb2.ax.set_ylabel(sLabel+' '+sUnit)
        
                # plot circles that show the trend in area
                for st in range(len(rgrLonAct)):
                    try:
                        iColor=np.where((clevsTD-rgrStat[st]) > 0)[0][0]-1
                    except:
                        try:
                            if rgrStat[st] > 0:
                                iColor=len(clevsTD)-1
                        except:
                            stop()
                    try:
                        if iColor == -1:
                            iColor=0
                    except:
                        stop()
                    ax.plot(rgrLonAct[st], rgrLatAct[st],'o',color=rgrColorTable[iColor], ms=5, mec=rgrColorTable[iColor], alpha=1, markeredgewidth=0.0,transform=ccrs.Geodetic(), zorder=20)
        
                        # if (rgrPvalue[st] <= 0.05):
                        #     ax.plot(rgrLon[st], rgrLat[st],'o',color=rgrColorTable[iColor], ms=5, mec='k', alpha=1, markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=20)
                        # else:
                        #     # not significant trend
                        #     try:
                        #         ax.plot(rgrLon[st], rgrLat[st],'o',color=rgrColorTable[iColor], ms=5, mec=rgrColorTable[iColor], alpha=1, markeredgewidth=0.0,transform=ccrs.Geodetic(), zorder=20)
                        #     except:
                        #         stop()
        
                ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5, zorder=11)
                ax.add_feature(cartopy.feature.COASTLINE, zorder=11)
                ax.add_feature(cartopy.feature.STATES, zorder=11, alpha=.5)
                ax.gridlines(zorder=11)
                # plt.axis('off')        
                
                # for sr in range(len(rgsSubregions)):
                #     # plot the subregions
                #     rgrShapeAct=grShapeFiles[rgsSubregions[sr]]
                #     rgrLonShape=np.array([rgrShapeAct[ii][0] for ii in range(len(rgrShapeAct))])
                #     rgrLatShape=np.array([rgrShapeAct[ii][1] for ii in range(len(rgrShapeAct))])
                #     x, y = m(rgrLonShape, rgrLatShape)
                #     m.plot(x, y, color='k', lw=0.5)
                
        
                ax3 = fig.add_axes([0.1, 0.08, 0.8, 0.02])
                cmap = mpl.colors.ListedColormap(rgrColorTable[:-1])
                cmap.set_under('#313695')
                cmap.set_over('#a50026')
                bounds = clevsTD
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                                norm=norm,
                                                boundaries=[-10] + bounds + [10],
                                                extend='both',
                                                # Make the length of each extension
                                                # the same as the length of the
                                                # interior colors:
                                                extendfrac='auto',
                                                ticks=bounds,
                                                spacing='uniform',
                                                orientation='horizontal')
                
                cb3.set_label(rgsStatistics[sta]+' in '+sLabel+' [%]')
        
        
                # lable the maps
                ax.text(0.03,1.04, rgsLableABC[se]+') '+rgsSeasons[se] , ha='left',va='bottom', \
                            transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=20)
        
        
            
            sPlotFile=PlotDir
            sPlotName= rgsDataset[da]+'-Eval_AbsBias_'+rgsVars[va]+'-'+rgsStatistics[sta]+'.pdf'
            if os.path.isdir(sPlotFile) != 1:
                subprocess.call(["mkdir","-p",sPlotFile])
            print('        Plot map to: '+sPlotFile+sPlotName)
            fig.savefig(sPlotFile+sPlotName)




stop()

# =======================================================
# =======================================================
# PLOT THE EVALUATION FOR CONTINENTS
# =======================================================


iComYears=((rgiYY[0] <= iYears_I) & (rgiYY[-1] >= iYears_I))

for va in range(len(rgsVars)):
    print('    Plotting '+rgsVars[va])
    if rgsVars[va] == 'WBZheight':
        biasContDist=20
        sLabel='ML height'
        sUnit='[km]'
        rgrDataVar=rgrERA20C_stations[:,:,:,rgsVars.index('WBZheight')]/1000.
        rgrDataVarEI=rgrERA_EI_stations[:,:,:,rgsVars.index('WBZheight')]/1000.
        rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('WBZ')]/1000.
    elif rgsVars[va] == 'ThreeCheight':
        biasContDist=20
        sLabel='3 $^{\circ}$C height'
        sUnit='[km]'
        rgrDataVar=rgrERA20C_stations[:,:,:,rgsVars.index('ThreeCheight')]/1000.
        rgrDataVarEI=rgrERA_EI_stations[:,:,:,rgsVars.index('ThreeCheight')]/1000.
        rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('ThreeC')]/1000.
    elif rgsVars[va] == 'ProbSnow':
        biasContDist=20
        sLabel='prob. of snow'
        sUnit='[days]'
        rgrDataVar=rgrERA20C_stations[:,:,:,rgsVars.index('ProbSnow')]
        rgrDataVarEI=rgrERA_EI_stations[:,:,:,rgsVars.index('ProbSnow')]
        rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('Snow')]
    elif rgsVars[va] == 'BflLR':
        biasContDist=20
        sLabel='lapse rate 0-FLH'
        sUnit='$^{\circ}$C km$^{-1}$'
        rgrDataVar=rgrERA20C_stations[:,:,:,rgsVars.index('BflLR')]
        rgrDataVarEI=rgrERA_EI_stations[:,:,:,rgsVars.index('BflLR')]
        rgrROdata=rgrMonDat[iComYears,:,:,rgsVariables_I.index('LRbF')]


    for se in range(5):
        fig = plt.figure(figsize=(13, 7))
        plt.rcParams.update({'font.size': 10})
        gs1 = gridspec.GridSpec(1,1)
        gs1.update(left=0.05, right=0.98,
                   bottom=0.05, top=0.95,
                   wspace=0.2, hspace=0.1)
        # scalings
        if rgsVars[va] == 'ProbSnow':
            iScale=iDaysInSeason[se]
        else:
            iScale=1.
        print('    Work on Season '+str(se))
        if se == 0:
            rgiTime=np.array([0,1,11])
        elif se == 1:
            rgiTime=np.array([2,3,4])
        elif se == 2:
            rgiTime=np.array([5,6,7])
        elif se == 3:
            rgiTime=np.array([8,9,10])
        elif se == 4:
            rgiTime=np.array([0,1,2,3,4,5,6,7,8,9,10,11])

        # remove stations with too little coverage --> min are 16 months
        rgiFin=np.sum(~np.isnan(rgrROdata[:,rgiTime,:]), axis=(0,1))
        rgiFin=(rgiFin >= 3*10)
        rgrROdataAct=rgrROdata[:,rgiTime,:][:,:,rgiFin]
        rgrDataVarAct=rgrDataVar[:,rgiTime,:][:,:,rgiFin]; rgrDataVarAct[np.isnan(rgrROdataAct)]=np.nan
        rgrDataVarActEI=rgrDataVarEI[:,rgiTime,:][:,:,rgiFin]; rgrDataVarActEI[np.isnan(rgrROdataAct)]=np.nan
        rgrLonAct=rgrLonS[rgiFin]; rgrLonAct[rgrLonAct > 180]=rgrLonAct[rgrLonAct > 180]-360
        rgrLatAct=rgrLatS[rgiFin]
        if rgsVars[va] == 'ProbSnow':
            rgrDataVarAct=rgrDataVarAct*iDaysInSeason[se]
            rgrDataVarActEI=rgrDataVarActEI*iDaysInSeason[se]
            rgrROdataAct=rgrROdataAct*iDaysInSeason[se]

        # ax = plt.subplot(gs1[0,0], projection=cartopy.crs.PlateCarree())
        # ax.set_extent([-175, 170, -60, 80], crs=ccrs.PlateCarree())
        ax = plt.subplot(gs1[0,0], projection=cartopy.crs.Robinson())
        ax.outline_patch.set_linewidth(0)

        ax.add_feature(cartopy.feature.OCEAN, zorder=0, facecolor='#c6dbef')
        ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='#ffffe5', facecolor='#ffffe5')        
        plt.title(sLabel+' '+rgsSeasons[se], fontsize=17)
        plt.axis('off')      

        # ========================
        # plot inlays for each continent
        for sr in range(len(rgsSubregions)):
            rgdShapeAct=grShapeFiles[rgsSubregions[sr]]
            PATH = path.Path(rgdShapeAct)
            flags = PATH.contains_points(np.hstack((rgrLonAct[:,np.newaxis],rgrLatAct[:,np.newaxis])))
            rgrObsData=np.nanmean(rgrROdataAct[:,:,flags]-np.nanmean(rgrROdataAct[:,:,flags], axis=(0,1)), axis=(1,2))+np.nanmean(rgrROdataAct[:,:,flags])
            rgrModData=np.nanmean(rgrDataVarAct[:,:,flags]-np.nanmean(rgrDataVarAct[:,:,flags], axis=(0,1)), axis=(1,2))+np.nanmean(rgrDataVarAct[:,:,flags])
            rgrModDataEI=np.nanmean(rgrDataVarActEI[:,:,flags]-np.nanmean(rgrDataVarActEI[:,:,flags], axis=(0,1)), axis=(1,2))+np.nanmean(rgrDataVarActEI[:,:,flags])
            # this is an inset axes over the main axes
            if rgsSubregions[sr] == 'Africa':
                a = plt.axes([.50, .34, .15, .18], facecolor=None)
            if rgsSubregions[sr] == 'Asia':
                a = plt.axes([.7, .65, .15, .18], facecolor=None)
            if rgsSubregions[sr] == 'North-America':
                a = plt.axes([.22, .63, .15, .18], facecolor=None)
            if rgsSubregions[sr] == 'South-America':
                a = plt.axes([.28, .30, .15, .18], facecolor=None)
            if rgsSubregions[sr] == 'Australia':
                a = plt.axes([.78, .28, .15, .18], facecolor=None)
            if rgsSubregions[sr] == 'Europe':
                a = plt.axes([.48, .70, .15, .18], facecolor=None)
            a.patch.set_alpha(0)
            plt.plot(iYears_I[iComYears], rgrObsData, c='k', lw=2)
            plt.plot(iYears_I[iComYears], rgrModData, c='#1f78b4')
            plt.plot(iYears_I[iComYears], rgrModDataEI, c='#e31a1c')
            plt.title(rgsSubregions[sr])
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            plt.xlabel('Year')
            plt.ylabel(sLabel+' '+sUnit)


        ###################################
        # get the time series per continent
        # iNaN=np.isnan(rgrROdataAct)
        # rgrDataVarAct[iNaN]=np.nan
        # if rgsStatistics[sta] == 'bias':
        #     # rgrStat=np.nanmean(((rgrDataVarAct-rgrROdataAct)/rgrROdataAct)*100., axis=(0,1))
        #     rgrStat=((np.nanmean(rgrDataVarAct, axis=(0,1)) - np.nanmean(rgrROdataAct, axis=(0,1)))/np.nanmean(rgrROdataAct, axis=(0,1)))*100.
        #     if rgsVars[va] != 'ProbSnow':
        #         rgrStat[np.nanmean(rgrROdataAct, axis=(0,1)) < 10]=np.nan
        # if rgsStatistics[sta] == 'stddev':
        #     STDobs=np.nanstd(np.nanmean(rgrROdataAct,axis=1), axis=0)
        #     STDmod=np.nanstd(np.nanmean(rgrDataVarAct,axis=1), axis=0)
        #     rgiVar=((STDobs < 0.1))
        #     rgrStat=((STDmod/STDobs)-1)*100.
        #     rgrStat[rgiVar]=np.nan
        #     rgrFullAct=np.nanstd(np.mean(rgrFullData[:,rgiTime,:,:], axis=(1)), axis=0)
        #     if se == 0:
        #         levelsBG=levelsBG/20.
        #         if rgsVars[va] == 'ProbSnow':
        #             levelsBG=levelsBG/10.
        #     # rgrStat[rgiVar]=0
        #     # rgrStat=((STDmod[rgiVar]/STDobs[rgiVar])-1)*100.
        #     # rgrLonAct=rgrLonAct[rgiVar]
        #     # rgrLatAct=rgrLatAct[rgiVar]
            

        

        sPlotFile=PlotDir+'TimeSeries-Continents/'
        sPlotName= 'ERA20C-Eval_TimeSeries_'+rgsVars[va]+'-'+rgsSeasons[se]+'.pdf'
        if os.path.isdir(sPlotFile) != 1:
            subprocess.call(["mkdir","-p",sPlotFile])
        print('        Plot map to: '+sPlotFile+sPlotName)
        fig.savefig(sPlotFile+sPlotName)




stop()
