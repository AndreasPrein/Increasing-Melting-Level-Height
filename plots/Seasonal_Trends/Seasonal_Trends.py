#!/usr/bin/env python
'''
    File name: LapseRate_Change.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    Read in the preprocessed data from:
    ~/projects/Hail/programs/RadioSoundings/Preprocessor/HailParametersFromRadioSounding.py

    Read in ERA-20C and ERA-Interim Data

    Plot a map that shows lapse reate trends in the period 1979-2010

    Add inlays that show time series for global, land, and ocean


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
from scipy.signal import medfilt

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
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
DataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Monthly_Data/'
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
rgsSeasons=['DJF','MAM','JJA','SON']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30]

rgsData=['Rad-Soundings', 'ERA-Interim','ERA-20C']

dStartDay_MSWEP=datetime.datetime(1979, 1, 1,0)
dStopDay_MSWEP=datetime.datetime(2017, 12, 31,0)
rgdTimeDD_MSWEP = pd.date_range(dStartDay_MSWEP, end=dStopDay_MSWEP, freq='d')
rgdTimeFin=rgdTimeDD_MSWEP
iYears=np.unique(rgdTimeFin.year)


rgsVars=['WBZheight','ThreeCheight','ProbSnow','LRbF','VS0_3','VS0_6','CAPE','CIN','LCL','LFC','CBT','PR','WCLD','SSP','CBVP','CPMR','LR_CB_ML','PR_days','Snow_days']

PRconditioning=['LCL','CBT','WCLD','CBVP','CPMR']

VAR='WCLD'
if VAR == 'WCLD_G3.8km':
    iVAR=rgsVars.index('WCLD')
else:
    iVAR=rgsVars.index(VAR)
########################################
#      READ IN RADIO SOUNDING DATA
if VAR in PRconditioning:
    DATA=np.load(DataDir+'RS-data-montly_19002017.npz_PRcond.npz')
else:
    DATA=np.load(DataDir+'RS-data-montly_19002017.npz')
VARIABLES=DATA['VARS']; VARIABLES=[VARIABLES[va] for va in range(len(VARIABLES))]
RS_data=DATA['Monthly_RO']
rgrLat=DATA['RS_Lat']
rgrLon=DATA['RS_Lon']
RS_time=DATA['RSTimeMM']
RS_Breaks=DATA['BreakFlag']

#      READ IN ERA-I DATA
if VAR in PRconditioning:
    DATA=np.load(DataDir+'EI-data-montly_19792017.npz_PRcond.npz')
elif VAR == 'WCLD_G3.8km':
    DATA=np.load(DataDir+'EI-data-montly_19792017.npz_deep_WCLD_35.npz')
else:
    DATA=np.load(DataDir+'EI-data-montly_19792017.npz_backup')
VARIABLES_EI=DATA['Variables']; VARIABLES_EI=[VARIABLES_EI[va] for va in range(len(VARIABLES_EI))]
EI_data=DATA['rgrMonthlyData_EI']
EI_lat=DATA['rgrLat75_EI']
EI_lon=DATA['rgrLon75_EI']
EI_time=DATA['rgdTimeMM_EI']
EI_Breaks=DATA['BreakFlag']
rgiYY_EI=np.unique(pd.DatetimeIndex(EI_time).year)
rgrLonGrid2D_EI=np.asarray(([EI_lon,]*EI_lat.shape[0]))
rgrLatGrid2D_EI=np.asarray(([EI_lat,]*EI_lon.shape[0])).transpose()

#      READ IN ERA-20 DATA
if VAR in PRconditioning:
    DATA=np.load(DataDir+'E20-data-montly_19002010.npz_PRcond.npz')
elif VAR == 'WCLD_G3.8km':
    DATA=np.load(DataDir+'E20-data-montly_19002010.npz_deep_WCLD.npz')
else:
    DATA=np.load(DataDir+'E20-data-montly_19002010.npz')
VARIABLES_E20=DATA['Variables']; VARIABLES_E20=[VARIABLES_E20[va] for va in range(len(VARIABLES_E20))]
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
RS_WCLD=RS_data[:,:,:,VARIABLES.index(VAR)]
RS_WCLD[:,:,(np.max(RS_Breaks[:,:,VARIABLES.index(VAR)], axis=0) > 0.5)]=np.nan; RS_WCLD[:,:,np.sum(~np.isnan(RS_WCLD), axis=(0,1)) < (RS_WCLD.shape[0]*RS_WCLD.shape[1]*0.8)]=np.nan
if VAR == 'CPMR':
    RS_WCLD[RS_WCLD == 0]=np.nan
if VAR in PRconditioning:
    RS_WCLD[-1:,:,:]=np.nan


EI_WCLD=EI_data[:,:,:,:,VARIABLES_EI.index(VAR)]
EI_WCLD[:,:,(np.max(EI_Breaks[:,:,:,VARIABLES_EI.index(VAR)], axis=0) > 0.5)]=np.nan# ; EI_WCLD[:,:,np.sum(~np.isnan(EI_WCLD), axis=(0,1)) < (EI_WCLD.shape[0]*EI_WCLD.shape[1]*0.8)]=np.nan
if VAR == 'CPMR':
    EI_WCLD[EI_WCLD == 0]=np.nan

E20_WCLD=E20_data[:,:,:,:,VARIABLES_E20.index(VAR)]
E20_WCLD[:,:,(np.max(E20_Breaks[:,:,:,VARIABLES_E20.index(VAR)], axis=0) > 0.5)]=np.nan# ; E20_WCLD[:,:,np.sum(~np.isnan(E20_WCLD), axis=(0,1)) < (E20_WCLD.shape[0]*E20_WCLD.shape[1]*0.8)]=np.nan
if VAR == 'CPMR':
    E20_WCLD[E20_WCLD == 0]=np.nan


########################################
#                      start plotting

if VAR == 'LRbF':
    biasContDist=0.02
    sLabel='trend in Surf-ML Lapse\n Rate [$^{\circ}$C km$^{-1}$ 10y$^{-1}$]'
    YRange=biasContDist*8
    Scale=1.
if VAR == 'CAPE':
    biasContDist=10
    sLabel='trend in CAPE [J kg$^{-1}$ 10y$^{-1}$]'
    YRange=biasContDist*8
    Scale=1.
if VAR == 'VS0_6':
    biasContDist=0.1
    sLabel='trend in 0-6 km shear [m s$^{-1}$ 10y$^{-1}$]'
    YRange=biasContDist*8
    Scale=1.
if VAR == 'LR_CB_ML':
    biasContDist=0.02
    sLabel='trend in cloud base to ML\n lapse rate [$^{\circ}$C km$^{-1}$ 10y$^{-1}$]'
    YRange=biasContDist*8
    Scale=1.
if VAR == 'ProbSnow':
    biasContDist=0.6
    sLabel='trend in Probability of snow [% 10y$^{-1}$]'
    YRange=biasContDist*4
    Scale=100.
if VAR == 'WBZheight':
    biasContDist=10
    sLabel='trend in melting level height [m 10y$^{-1}$]'
    YRange=biasContDist*16
    Scale=1.
if VAR == 'LCL':
    biasContDist=10
    sLabel='trend in cloud base height [m 10y$^{-1}$]'
    YRange=biasContDist*16
    Scale=1.
if VAR == 'CBT':
    biasContDist=0.14
    sLabel='trend in cloud base\n temperature [$^{\circ}$C 10y$^{-1}$]'
    YRange=biasContDist*10
    Scale=1.
if VAR == 'CPMR':
    biasContDist=0.04
    sLabel='trend in cloud base saturation\n mixing ratio [g kg$^{-1}$ 10y$^{-1}$]'
    YRange=biasContDist*10
    Scale=1000.
if VAR == 'CBVP':
    biasContDist=0.4
    sLabel='trend in cloud base saturation\n vapor pressure [Pa 10y$^{-1}$]'
    YRange=biasContDist*10
    Scale=1.
if VAR == 'WCLD':
    biasContDist=10
    sLabel='trend in warm cloud level depth\n on precipitation days [m 10y$^{-1}$]'
    YRange=biasContDist*10
    Scale=1.
if VAR == 'WCLD_G3.8km':
    biasContDist=10
    sLabel='trend in frequency of warm cloud levels\n deeper than 3.5 km [% 10y$^{-1}$]'
    YRange=biasContDist*0.01
    Scale=0.001

RS_WCLD=RS_WCLD*Scale
EI_WCLD=EI_WCLD*Scale
E20_WCLD=E20_WCLD*Scale

rgrDataVar=RS_WCLD

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
        iMS=10 # marker size for RO stations
    else:
        iXX=[0,1,0,1]
        iYY=[0,0,1,1]
        gs1 = gridspec.GridSpec(2,2)
        gs1.update(left=0.01, right=0.64,
                   bottom=0.1, top=0.47,
                   wspace=0.05, hspace=0.05)
        ax = plt.subplot(gs1[iYY[se-1],iXX[se-1]], projection=cartopy.crs.Robinson())
        # ax.set_extent([-175, 160, 15, 80], crs=ccrs.PlateCarree())
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
    rgrDataAct=rgrDataVar[:,rgiMonths,:]; rgrDataAct[:,:,np.sum(~np.isnan(rgrDataAct), axis=(0,1)) < (rgrDataAct.shape[0]*rgrDataAct.shape[1]*0.8)]=np.nan
    rgrDataAnnual=np.nanmean(rgrDataAct, axis=1)

    for st in range(len(rgrLon)):
        # remove nans if there are some
        rgiReal=~np.isnan(rgrDataAnnual[(iYears <= 2010),st])
        if (len(rgiReal) < (2010-1979)*0.7):
            continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(iYears[(iYears <= 2010)][rgiReal],rgrDataAnnual[(iYears <= 2010),st][rgiReal])
        except:
            continue
        rgrGCtrend[st]=slope*10 #(slope/np.nanmean(rgrDataAnnual[:,st]))*1000.
        rgrPvalue[st]=p_value
        rgrR2[st]=r_value**2
        if VAR == 'WCLD_G3.8km':
            rgrGCtrend[st]=(rgrGCtrend[st]/np.nanmean(rgrDataAnnual[(iYears <= 2010),st][rgiReal]))*100.

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
    rgrMonthAct=E20_WCLD
    rgrLonAct=rgrLonGrid2D
    rgrLatAct=rgrLatGrid2D

    rgrTrendsERA20=np.zeros((rgrMonthAct.shape[2],rgrMonthAct.shape[3],5)); rgrTrendsERA20[:]=np.nan
    rgrSeasonalData=rgrMonthAct[:,rgiMonths,:,:][(rgiYY >=1979),:]# ; rgrSeasonalData[:,:,np.sum(~np.isnan(rgrSeasonalData), axis=(0,1)) < (rgrSeasonalData.shape[0]*rgrSeasonalData.shape[1]*0.8)]=np.nan
    rgrSeasonalData=np.nanmean(rgrSeasonalData,axis=1)
    for la in range(rgrSeasonalData.shape[1]):
        for lo in range(rgrSeasonalData.shape[2]):
            rgrTrendsERA20[la,lo,:]=stats.linregress(rgiYY[(rgiYY >=1979)],rgrSeasonalData[:,la,lo])

    rgrMonthAct_EI=EI_WCLD
    rgrLonAct_EI=rgrLonGrid2D_EI
    rgrLatAct_EI=rgrLatGrid2D_EI
    if VAR == 'WCLD_G3.8km':
        rgrTrendsERA20[:,:,0]=(rgrTrendsERA20[:,:,0]/np.nanmean(rgrSeasonalData[:,:,:],axis=0))*100.

    rgrTrendsERAI=np.zeros((rgrMonthAct_EI.shape[2],rgrMonthAct_EI.shape[3],5)); rgrTrendsERAI[:]=np.nan
    rgrSeasonalData_EI=rgrMonthAct_EI[:,rgiMonths,:,:][((rgiYY_EI >=1979) & (rgiYY_EI <=2010)),:]# ; rgrSeasonalData_EI[:,:,np.sum(~np.isnan(rgrSeasonalData_EI), axis=(0,1)) < (rgrSeasonalData_EI.shape[0]*rgrSeasonalData_EI.shape[1]*0.8)]=np.nan
    rgrSeasonalData_EI=np.nanmean(rgrSeasonalData_EI,axis=1)
    for la in range(rgrSeasonalData_EI.shape[1]):
        for lo in range(rgrSeasonalData_EI.shape[2]):
            rgrTrendsERAI[la,lo,:]=stats.linregress(rgiYY_EI[((rgiYY_EI >=1979) & (rgiYY_EI <=2010))],rgrSeasonalData_EI[:,la,lo])
    if VAR == 'WCLD_G3.8km':
        rgrTrendsERAI[:,:,0]=(rgrTrendsERAI[:,:,0]/np.nanmean(rgrSeasonalData_EI[:,:,:],axis=0))*100.
    ERAi_on_ERA20_T=scipy.interpolate.griddata((rgrLonGrid2D_EI.flatten(),rgrLatGrid2D_EI.flatten()),rgrTrendsERAI[:,:,0].flatten() , (rgrLonAct,rgrLatAct),method='linear')
    ERAi_on_ERA20_P=scipy.interpolate.griddata((rgrLonGrid2D_EI.flatten(),rgrLatGrid2D_EI.flatten()),rgrTrendsERAI[:,:,3].flatten() , (rgrLonAct,rgrLatAct),method='linear')
    AVERAGE_Trend=np.nanmean([ERAi_on_ERA20_T,rgrTrendsERA20[:,:,0]], axis=0)

    # plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,colors=rgrColorTable, extend='both',levels=clevsTD) #, alpha=0.6)
    # cs=plt.contourf(rgrLonAct,rgrLatAct,rgrTrendsERA20[:,:,0]*10*iScale,cmap=plt.cm.coolwarm, extend='both',levels=clevsTD, zorder=0) #, alpha=0.6)
    try:
        cs=plt.contourf(rgrLonAct,rgrLatAct,AVERAGE_Trend*10,colors=rgrColorTable, extend='both',levels=clevsTD, zorder=0,transform=cartopy.crs.PlateCarree()) #, alpha=0.6)
    except:
        stop()
    # grey out areas with broken or missing records
    NAN=np.array(np.isnan(AVERAGE_Trend).astype('float')); NAN[NAN != 1]=np.nan
    plt.contourf(rgrLonGrid2D,rgrLatGrid2D,NAN,levels=[0,1,2],colors=['#bdbdbd','#bdbdbd','#bdbdbd'], zorder=1,transform=cartopy.crs.PlateCarree())

    # # plot significant layer
    # for la in range(rgrSeasonalData.shape[1])[::2]:
    #     for lo in range(rgrSeasonalData.shape[2])[::2]:
    #         if (rgrTrendsERA20[la,lo,3] <= 0.05) & (ERAi_on_ERA20_P[la,lo] <= 0.05) & (rgrTrendsERA20[la,lo,0] * ERAi_on_ERA20_T[la,lo] > 0):
    #             ax.plot(rgrLonAct[la,lo], rgrLatAct[la,lo],'o',color='k', ms=2, alpha=0.5, markeredgewidth=0.0,transform=cartopy.crs.PlateCarree(), zorder=15) # , mec=rgrColorTable[iColor]
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
    CbarAx = axes([0.01, 0.035, 0.63, 0.02])
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

# rgrT2Mmonth=np.load('ERA-20C_T2M.npy')
# rgsVars=rgsVars+['T2M']
# rgrMonthlyData=np.append(rgrMonthlyData,rgrT2Mmonth[:,:,:,:,None], axis=4)

# read in the subregions
gs1 = gridspec.GridSpec(3,1)
gs1.update(left=0.7, right=0.99,
           bottom=0.05, top=0.95,
           wspace=0.2, hspace=0.2)


sLabel='$\Delta$ '+sLabel
rgrDataVar=E20_WCLD
rgrDataVarEI=EI_WCLD
rgrROdata=RS_WCLD

rgrROdataAct=rgrROdata # [:,:,:][:,:,rgiFin]
rgrLonAct=rgrLon# [rgiFin]
rgrLatAct=rgrLat# [rgiFin]
rgrDataVarEIAct=rgrDataVarEI[:,:,:]
rgrDataVarAct=rgrDataVar[:,:,:]

ii=0
rgsGlobal=['Global','Global Land','Global Ocean']
for sr in range(len(rgsGlobal)):
    ax = plt.subplot(gs1[ii,0])
    for da in range(len(rgsData)):
        if rgsData[da] == 'Rad-Soundings':
            LON=rgrLonAct
            LAT=rgrLatAct
            DATA=rgrROdataAct
            rgiYearsAct=np.array(range(1979,2018,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            sColor='k'
            rgiLandOcean=[bm.is_land(LON[ll], LAT[ll]) for ll in range(len(LAT))]
        if rgsData[da] == 'ERA-20C':
            LON=rgrLonGrid2D.flatten()
            LAT=rgrLatGrid2D.flatten()
            DATA=np.reshape(rgrDataVarAct, (rgrDataVarAct.shape[0],rgrDataVarAct.shape[1],rgrDataVarAct.shape[2]*rgrDataVarAct.shape[3]))
            rgiYearsAct=np.array(range(1900,2011,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            sColor='#1f78b4'
            # # T2M
            # DATAT2M=np.reshape(rgrDataVarActT2M, (rgrDataVarActT2M.shape[0],rgrDataVarActT2M.shape[1],rgrDataVarActT2M.shape[2]*rgrDataVarActT2M.shape[3]))
            # # rgiLandOceanE20=[bm.is_land(LON[ll], LAT[ll]) for ll in range(len(LAT))]
            # # np.save('LandOcean_E20.npy',rgiLandOceanE20)
            rgiLandOcean=np.load('LandOcean_E20.npy')
            Weights=np.transpose(np.tile(np.cos(np.linspace(-90,90,rgrDataVarAct.shape[2])/(360/(np.pi*2))),(rgrDataVarAct.shape[3],1))).flatten()
        if rgsData[da] == 'ERA-Interim':
            LON=rgrLonGrid2D_EI.flatten()
            LAT=rgrLatGrid2D_EI.flatten()
            DATA=np.reshape(rgrDataVarEIAct, (rgrDataVarEIAct.shape[0],rgrDataVarEIAct.shape[1],rgrDataVarEIAct.shape[2]*rgrDataVarEIAct.shape[3]))
            rgiYearsAct=np.array(range(1979,2018,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            sColor='#e31a1c'
            # rgiLandOceanEI=[bm.is_land(LON[ll], LAT[ll]) for ll in range(len(LAT))]
            # np.save('LandOcean_EI.npy',rgiLandOceanEI)
            rgiLandOcean=np.load('LandOcean_EI.npy')
            Weights=np.transpose(np.tile(np.cos(np.linspace(-90,90,rgrDataVarEIAct.shape[2])/(360/(np.pi*2))),(rgrDataVarEIAct.shape[3],1))).flatten()

        if rgsGlobal[sr] == 'Global':
            flags=np.array([True]*DATA.shape[2])
        if rgsGlobal[sr] == 'Global Land':
            flags=np.array([True]*DATA.shape[2])
            # flags=rgiLandOcean
        if rgsGlobal[sr] == 'Global Ocean':
            flags=(rgiLandOcean == False)

        if (rgsData[da] == 'Rad-Soundings'):
            rgrTMPData=np.nanmedian(np.nanmean(DATA[:,:,flags],axis=(1))-np.nanmean(DATA[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
            # rgrTMPData=np.nanmean(np.nanmean(DATA[:,:,flags],axis=(1))-np.nanmean(DATA[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
        else:
            indices = ~np.isnan(np.mean(DATA[:,:,flags], axis=(0,1)))
            try:
                Full=np.array([np.average(np.median(DATA[yy,:,flags],axis=1)[indices], weights=Weights[flags][indices]) for yy in range(DATA.shape[0])])
                rgrTMPData=Full-np.median(Full[rgiSelYY])
            except:
                continue

        if rgsGlobal[sr] == 'Global Land':
            print ' '
            print '    '+rgsData[da]
            print '     slope '+str(stats.linregress(range(np.sum(rgiSelYY)),rgrTMPData[rgiSelYY])[0]*10)
            print '     p-value '+str(stats.linregress(range(np.sum(rgiSelYY)),rgrTMPData[rgiSelYY])[3])

        # CCS_bs=[np.mean(np.random.choice(rgrTMPData[-30:],len(rgrTMPData[-30:]))) - np.mean(np.random.choice(rgrTMPData[:30],len(rgrTMPData[:30]))) for ii in range(1000)]
        # TRENDS=stats.linregress(rgiYearsAct[(rgiYearsAct >= 1979) & (rgiYearsAct <=2010)],rgrTMPData[(rgiYearsAct >= 1979) & (rgiYearsAct <=2010)])

        plt.plot(rgiYearsAct, rgrTMPData, c=sColor, lw=0.5, zorder=3, alpha=0.5)
        # plt.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPData,10), c=sColor, lw=3, zorder=3, label=rgsData[da])
        try:
            plt.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPData,10), c=sColor, lw=3, zorder=3, label=rgsData[da])
        except:
            stop()
        # plt.plot(iYears_I[iComYears], rgrModData, c='#1f78b4')
        # plt.plot(iYears_I[iComYears], rgrModDataEI, c='#e31a1c')

        plt.title(rgsLableABC[ii+5]+') '+rgsGlobal[sr])
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
        ax.set_ylabel(sLabel)
        ax.set(ylim=(-YRange,YRange))

    #     if rgsData[da] == 'ERA-20C':
    #         rgrTMPDataT2M=np.nanmean(np.nanmean(DATAT2M[:,:,flags],axis=(1))-np.nanmean(DATAT2M[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
    #         Y2col='#636363'
    #         # Plot T2M on a second y-axis
    #         ax2 = ax.twinx()
    #         ax2.set_ylabel('2m temperature (T2M) [$^{\circ}$C]', color=Y2col)
    #         ax2.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPDataT2M,10), c=Y2col, lw=3, zorder=1, alpha=0.5)
    #         ax2.plot(rgiYearsAct, rgrTMPDataT2M, c=Y2col, lw=0.5, zorder=1, alpha=0.5)
    #         ax2.set(ylim=(-2,2))
    #         # ax2.set_yticks(np.arange(30, 50, 5))
    #         ax2.spines['right'].set_color(Y2col)
    #         ax2.yaxis.label.set_color(Y2col)
    #         ax2.tick_params(axis='y', colors=Y2col)
    #         ax2.spines['left'].set_visible(False)
    #         ax2.spines['top'].set_visible(False)

    # ax.plot([2000,2000],[0,0],lw=3,alpha=0, c='#636363', label='ERA-20C T2M')
    if ii == 0:
        # lns = lns1+lns2
        # labs = [l.get_label() for l in lns]
        ax.legend(loc="upper left",\
                  ncol=1, prop={'size':12})

    ii=ii+1


sPlotFile=PlotDir
sPlotName= 'Seasonal_changes_'+VAR+'.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
