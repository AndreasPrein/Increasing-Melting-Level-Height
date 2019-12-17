#!/usr/bin/env python
'''
    File name: Snow-VS-PR-Day_changes.py
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
import seaborn as sns
from scipy import signal
from matplotlib.widgets import RectangleSelector
from scipy import optimize

from numpy import linspace, meshgrid
from matplotlib.mlab import griddata



from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

# https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


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

def RemoveSpouriousData(yy, sigma):
    # function detrends a time series and removes all points that are
    # further than sigma x standard-deviations from the mean
    yy_dt=signal.detrend(yy)
    STDDEV=np.std(yy_dt)
    iOutlier=((yy_dt > STDDEV*sigma) | (yy_dt < -STDDEV*sigma))
    # if np.sum(iOutlier) > 0:
    #     plt.plot(xx,yy); plt.plot(xx,yy_dt); plt.plot(xx[iOutlier],yy_dt[iOutlier], 'ro');plt.show()
    yy[iOutlier]=np.nan
    return yy

########################################
#                            Settings
rgsVars=['ProbSnow','Snow_PRdays','Rain_PRdays'] #['WBZheight','ThreeCheight','ProbSnow','BflLR']
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
DataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Monthly_Data/'
rgsSeasons=['DJF','MAM','JJA','SON','Annual']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30,365.25]
iMinCov=0.8
iScale=1 #rgsDaysSeas[se]

rgsData=['Rad-Soundings', 'ERA-Interim','ERA-20C']

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
VARIABLES_RS=DATA['VARS']; VARIABLES_RS=[VARIABLES_RS[va] for va in range(len(VARIABLES_RS))]
RS_data=DATA['Monthly_RO']
rgrLat=DATA['RS_Lat']
rgrLon=DATA['RS_Lon']
RS_time=DATA['RSTimeMM']
RS_Breaks=DATA['BreakFlag']

#      READ IN ERA-I DATA
DATA=np.load(DataDir+'EI-data-montly_19792017.npz_backup')
VARIABLES=DATA['Variables']; VARIABLES=[VARIABLES[va] for va in range(len(VARIABLES))]
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
RS_SNOW=RS_data[:,:,:,VARIABLES_RS.index('Snow_days')] ; RS_SNOW[:,:,(np.max(RS_Breaks[:,:,VARIABLES_RS.index('Snow_days')], axis=0) > 0.5)]=np.nan
RS_PR=RS_data[:,:,:,VARIABLES_RS.index('PR_days')] ; RS_PR[:,:,(np.max(RS_Breaks[:,:,VARIABLES_RS.index('PR_days')], axis=0) > 0.5)]=np.nan
RS_WCLD=np.sum(RS_SNOW,axis=1)/np.sum(RS_PR,axis=1)*100.; RS_WCLD[:,np.sum(~np.isnan(RS_WCLD), axis=(0)) < (RS_WCLD.shape[0]*0.66)]=np.nan
RS_WCLD[:,np.nanmean(RS_WCLD, axis=(0)) < 0.05]=np.nan

E20_SNOW=E20_data[:,:,:,:,VARIABLES_E20.index('Snow_days')]; E20_SNOW[:,:,(np.max(E20_Breaks[:,:,:,VARIABLES_E20.index('Snow_days')], axis=0) > 0.5)]=np.nan
E20_PR=E20_data[:,:,:,:,VARIABLES_E20.index('PR_days')]; E20_PR[:,:,(np.max(E20_Breaks[:,:,:,VARIABLES_E20.index('PR_days')], axis=0) > 0.5)]=np.nan
E20_WCLD=np.sum(E20_SNOW,axis=1)/np.sum(E20_PR,axis=1)*100.; E20_WCLD[:,np.nanmean(E20_WCLD, axis=(0)) < 0.05]=np.nan

EI_SNOW=EI_data[:,:,:,:,VARIABLES.index('Snow_days')]; EI_SNOW[:,:,(np.max(EI_Breaks[:,:,:,VARIABLES.index('Snow_days')], axis=0) > 0.5)]=np.nan
EI_PR=EI_data[:,:,:,:,VARIABLES.index('PR_days')]; EI_PR[:,:,(np.max(EI_Breaks[:,:,:,VARIABLES.index('PR_days')], axis=0) > 0.5)]=np.nan
EI_WCLD=np.sum(EI_SNOW,axis=1)/np.sum(EI_PR,axis=1)*100.; EI_WCLD[:,np.nanmean(EI_WCLD, axis=(0)) < 0.05]=np.nan

# ################################################################################
# ################################################################################
# ################################################################################
# #          Read the GHCN data
# ################################################################################
# this was preprocessed by /gpfs/u/home/prein/papers/Trends_RadSoundings/programs/GHCN-NOAA_data/GHCN-dataprocessing.py
GHCN=np.load('/glade/scratch/prein/Papers/Trends_RadSoundings/data/GHCN-NOAA/ghcnd_all/GHCN_monthy-snow-data.npz')
GHCN_monthly_data=GHCN['GHCN_monthly_data'] .astype('float')
lonGHCN=GHCN['lon'].astype('float')
latGHCN=GHCN['lat'].astype('float')
elevation=GHCN['elevation'].astype('float')
StationID=GHCN['StationID']
StationName=GHCN['StationName']
Years=GHCN['Years']
TargetVariables=np.ndarray.tolist(GHCN['TargetVariables'])


# =============================================================================================
# =============================================================================================
# =============================================================================================
#                                      START PLOTTING

biasContDist=1
sLabel='Snow vs. precipitation day ratio [% per decade]'
Yminmax=[-6,5]

fig = plt.figure(figsize=(15, 12))
rgsSeasons=['annual','DJF','MAM','JJA','SON']
rgsDaysSeas=[365.25,90.25,92,92,91]
for se in [0]: #range(5):
    print '    work on '+rgsSeasons[se]
    if se == 0:
        gs1 = gridspec.GridSpec(1,1)
        gs1.update(left=0.01, right=0.47,
                   bottom=0.68, top=0.98,
                   wspace=0.2, hspace=0.1)
        ax = plt.subplot(gs1[0,0], projection=cartopy.crs.Robinson())
        rgiTime=(rgdTimeFin.month <= 12)
        rgiMonths=[0,1,2,3,4,5,6,7,8,9,10,11]
        iMS=7 # marker size for RO stations
    else:
        iXX=[0,1,0,1]
        iYY=[0,0,1,1]
        gs1 = gridspec.GridSpec(2,2)
        gs1.update(left=0.48, right=0.99,
                   bottom=0.68, top=0.98,
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

    iScale=1 #rgsDaysSeas[se]
    
    rgrColorTable=np.array(['#4457c9','#4f67d4','#6988ea','#84a6f9','#9ebdff','#b7d0fd','#cfdcf3','#ffffff','#f1d7c8','#f9c5ab','#f7aa8c','#f28d72','#df6553','#c93137','#bc052b'])
    iContNr=len(rgrColorTable)
    iMinMax=12.*biasContDist/2.+biasContDist/2.
    clevsTD=np.arange(-iMinMax,iMinMax+0.0001,biasContDist)

    ###################################
    # calculate and plot trends
    plt.axis('off')
    # ax.set_global()

    rgrGCtrend=np.zeros((len(rgrLon))); rgrGCtrend[:]=np.nan
    rgrPvalue=np.copy(rgrGCtrend)
    rgrR2=np.copy(rgrGCtrend)
    # get the data
    SnowRS=RS_SNOW[:,rgiMonths,:]; SnowRS[:,:,np.sum(~np.isnan(SnowRS), axis=(0,1)) < (SnowRS.shape[0]*SnowRS.shape[1]*0.66)]=np.nan; SnowRS[:,:,np.nanmedian(SnowRS, axis=(0,1)) == 0]=np.nan
    PrecipSR=RS_PR[:,rgiMonths,:]; PrecipSR[:,:,np.sum(~np.isnan(PrecipSR), axis=(0,1)) < (PrecipSR.shape[0]*PrecipSR.shape[1]*0.66)]=np.nan
    rgrDataAnnual=(np.nansum(SnowRS,axis=1)/np.nansum(PrecipSR,axis=1))*100

    for st in range(len(rgrLon)):
        # remove nans if there are some
        rgiReal=~np.isnan(rgrDataAnnual[(iYears <= 2010),st])
        # if (np.nanmean(np.sum(SnowRS[:,:,st], axis=0)[(iYears <= 2010)][rgiReal]*iScale) < 3) | (len(rgiReal) < (2010-1979)*0.7):
        #     continue
        try:
            test=np.max(rgrDataAnnual[(iYears <= 2010),st][rgiReal])
        except:
            continue
        if np.max(rgrDataAnnual[(iYears <= 2010),st][rgiReal]) != 0:
            try:
                # XX=iYears[(iYears <= 2010)][rgiReal]
                # YY=rgrDataAnnual[:,st][rgiReal]
                # YYc=RemoveSpouriousData(YY,sigma=2)
                # rgiReal=~np.isnan(YYc)
                # slope, intercept, r_value, p_value, std_err = stats.linregress(XX[rgiReal],YYc[rgiReal])
                slope, intercept, r_value, p_value, std_err = stats.linregress(iYears[(iYears <= 2010)][rgiReal],rgrDataAnnual[(iYears <= 2010),st][rgiReal])
                # if slope*10*iScale > 6:
                #     stop()
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
                iColor=np.where((clevsTD-rgrGCtrend[st]) > 0)[0][0]
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
    print '    Add ERA-20C & ERA-Interim analysis'
    # ERA20C
    rgrLonAct=rgrLonGrid2D
    rgrLatAct=rgrLatGrid2D
    rgrTrendsERA20=np.zeros((E20_SNOW.shape[2],E20_SNOW.shape[3],5)); rgrTrendsERA20[:]=np.nan
    SnowE20=E20_SNOW[:,rgiMonths,:,:]; SnowE20[:,:,np.sum(~np.isnan(SnowE20), axis=(0,1)) < (SnowE20.shape[0]*SnowE20.shape[1]*0.8)]=np.nan
    PrE20=E20_PR[:,rgiMonths,:,:]; PrE20[:,:,np.sum(~np.isnan(PrE20), axis=(0,1)) < (PrE20.shape[0]*PrE20.shape[1]*0.8)]=np.nan
    SnowPRratio_E20=np.sum(SnowE20, axis=1)[(rgiYY >=1979),:]/np.sum(PrE20, axis=1)[(rgiYY >=1979),:]*100.
    rgrSeasonalData=SnowPRratio_E20
    for la in range(rgrSeasonalData.shape[1]):
        for lo in range(rgrSeasonalData.shape[2]):
            FinData=~np.isnan(rgrSeasonalData[:,la,lo])
            if sum(FinData)/float(len(FinData)) >= iMinCov:
                rgrTrendsERA20[la,lo,:]=stats.linregress(rgiYY[(rgiYY >=1979)][FinData],rgrSeasonalData[FinData,la,lo])
    # ERAI
    rgrLonEI=rgrLonGrid2D_EI
    rgrLatEI=rgrLatGrid2D_EI
    rgrTrendsERAI=np.zeros((EI_SNOW.shape[2],EI_SNOW.shape[3],5)); rgrTrendsERAI[:]=np.nan
    SnowEI=EI_SNOW[:,rgiMonths,:,:]; SnowEI[:,:,np.sum(~np.isnan(SnowEI), axis=(0,1)) < (SnowEI.shape[0]*SnowEI.shape[1]*0.8)]=np.nan
    PrEI=EI_PR[:,rgiMonths,:,:]; PrEI[:,:,np.sum(~np.isnan(PrEI), axis=(0,1)) < (PrEI.shape[0]*PrEI.shape[1]*0.8)]=np.nan
    SnowPRratio_EI=np.sum(SnowEI, axis=1)[(rgiYY_EI >=1979) & (rgiYY_EI <= 2010),:]/np.sum(PrEI, axis=1)[(rgiYY_EI >=1979) & (rgiYY_EI <= 2010),:]*100.
    rgrSeasonalDataEI=SnowPRratio_EI
    for la in range(rgrSeasonalDataEI.shape[1]):
        for lo in range(rgrSeasonalDataEI.shape[2]):
            FinData=~np.isnan(rgrSeasonalDataEI[:,la,lo])
            if sum(FinData)/float(len(FinData)) >= iMinCov:
                rgrTrendsERAI[la,lo,:]=stats.linregress(rgiYY[(rgiYY >=1979)][FinData],rgrSeasonalDataEI[FinData,la,lo])
    ERAi_on_ERA20_T=scipy.interpolate.griddata((rgrLonEI.flatten(),rgrLatEI.flatten()),rgrTrendsERAI[:,:,0].flatten() , (rgrLonAct,rgrLatAct),method='linear')
    ERAi_on_ERA20_P=scipy.interpolate.griddata((rgrLonEI.flatten(),rgrLatEI.flatten()),rgrTrendsERAI[:,:,3].flatten() , (rgrLonAct,rgrLatAct),method='linear')
    # average ERAI and ERA-20C data
    AVERAGE_Trend=np.nanmean([ERAi_on_ERA20_T,rgrTrendsERA20[:,:,0]], axis=0)*10*iScale
    cs=plt.contourf(rgrLonAct,rgrLatAct,AVERAGE_Trend,colors=rgrColorTable, extend='both',levels=clevsTD*1.0001, zorder=0,transform=cartopy.crs.PlateCarree()) #, alpha=0.6)
    # grey out areas with broken or missing records
    NAN=np.array(np.isnan(AVERAGE_Trend).astype('float')); NAN[NAN != 1]=np.nan
    plt.contourf(rgrLonGrid2D,rgrLatGrid2D,NAN,levels=[0,1,2],colors=['#bdbdbd','#bdbdbd','#bdbdbd'], zorder=1,transform=cartopy.crs.PlateCarree())

    # plot significant layer
    for la in range(rgrSeasonalData.shape[1])[::2]:
        for lo in range(rgrSeasonalData.shape[2])[::2]:
            if (rgrTrendsERA20[la,lo,3] <= 0.05) & (ERAi_on_ERA20_P[la,lo] <= 0.05) & (ERAi_on_ERA20_T[la,lo]*rgrTrendsERA20[la,lo,0] > 0):
                ax.plot(rgrLonAct[la,lo], rgrLatAct[la,lo],'o',color='k', ms=2, mec=rgrColorTable[iColor], alpha=0.5, markeredgewidth=0.0,transform=cartopy.crs.PlateCarree(), zorder=15)
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
    CbarAx = axes([0.1, 0.29, 0.8, 0.02])
    cb = colorbar(cs, cax = CbarAx, orientation='horizontal', extend='max', ticks=clevsTD, extendfrac='auto')
    cb.ax.set_title(sLabel)

    # lable the maps
    tt = ax.text(0.03,0.99, rgsLableABC[se]+') '+rgsSeasons[se] , ha='left',va='top', \
                transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=20)
    tt.set_bbox(dict(facecolor='w', alpha=1, edgecolor='w'))


# # ================================================================================
# # ================================================================================
# #                  ADD THE GHCN STATIONS
# for se in range(5):
#     YearsSelectGHNC=((Years >= 1979) & (Years <= 2010))
#     print '    work on '+rgsSeasons[se]
#     if se == 0:
#         gs1 = gridspec.GridSpec(1,1)
#         gs1.update(left=0.01, right=0.35,
#                    bottom=0.35, top=0.65,
#                    wspace=0.2, hspace=0.1)
#         ax = plt.subplot(gs1[0,0], projection=cartopy.crs.NearsidePerspective(central_longitude=-100, central_latitude=45, satellite_height=35785831 ))
#         rgiTime=(rgdTimeFin.month <= 12)
#         rgiMonths=[0,1,2,3,4,5,6,7,8,9,10,11]
#         iMS=3 # marker size for RO stations
#         ax.set_extent([-125, -50, 28, 70], crs=ccrs.PlateCarree())
#     else:
#         iXX=[0,1,0,1]
#         iYY=[0,0,1,1]
#         gs1 = gridspec.GridSpec(2,2)
#         gs1.update(left=0.42, right=0.89,
#                    bottom=0.35, top=0.65,
#                    wspace=0.6, hspace=0.05)
#         # ax.set_adjustable('datalim')
#         if rgsSeasons[se] == 'SON': 
#             rgiTime=((rgdTimeFin.month <= 11) & (rgdTimeFin.month >= 9))
#             rgiMonths=[8,9,10]
#         if rgsSeasons[se] == 'MAM': 
#             rgiTime=((rgdTimeFin.month <= 5) & (rgdTimeFin.month >= 3))
#             rgiMonths=[2,3,4]
#         if rgsSeasons[se] == 'DJF': 
#             rgiTime=((rgdTimeFin.month == 12) | (rgdTimeFin.month <= 2))
#             rgiMonths=[0,1,11]
#         if rgsSeasons[se] == 'JJA': 
#             rgiTime=((rgdTimeFin.month >= 6) & (rgdTimeFin.month <= 8))
#             rgiMonths=[5,6,7]
#         if se != 3:
#             ax = plt.subplot(gs1[iYY[se-1],iXX[se-1]], projection=cartopy.crs.NearsidePerspective(central_longitude=-100, central_latitude=45, satellite_height=35785831))
#             ax.set_extent([-125, -70, 30, 52], crs=ccrs.PlateCarree())
#             iMS=2
#         else:
#             ax = plt.subplot(gs1[iYY[se-1],iXX[se-1]], projection=cartopy.crs.NearsidePerspective(central_longitude=-100, central_latitude=65, satellite_height=35785831))
#             ax.set_extent([-150, -55, 45, 80], crs=ccrs.PlateCarree())
#             iMS=6

#     iScale=1 #rgsDaysSeas[se]

#     ###################################
#     # calculate and plot trends
#     plt.axis('off')
#     # ax.set_global()
#     YearsSelectGHNC=((Years >= 1979) & (Years <= 2010))
#     rgrGCtrend=np.zeros((len(lonGHCN))); rgrGCtrend[:]=np.nan
#     rgrPvalue=np.copy(rgrGCtrend)
#     rgrR2=np.copy(rgrGCtrend)
#     # get the data
#     rgrDataAnnual=np.nanmean(GHCN_monthly_data[YearsSelectGHNC,:,:,TargetVariables.index('SnowToPrecipDayRatio')][:,rgiMonths,:]*100., axis=1)# np.zeros((len(iYears),rgrDataAct.shape[1])); rgrDataAnnual[:]=np.nan
#     for st in range(len(lonGHCN)):
#         # remove nans if there are some
#         rgiReal=~np.isnan(rgrDataAnnual[:,st])
#         if (np.median(rgrDataAnnual[:,st][rgiReal]*iScale) < 2) | (len(rgiReal) < (2010-1979)*0.7):
#             continue
#         try:
#             # remove spurious data
#             XX=Years[YearsSelectGHNC][rgiReal]
#             YY=rgrDataAnnual[:,st][rgiReal]
#             YYc=RemoveSpouriousData(YY,sigma=2)
#             rgiReal=~np.isnan(YYc)
#             slope, intercept, r_value, p_value, std_err = stats.linregress(XX[rgiReal],YYc[rgiReal])
#             # scipy.stats.mstats.theilslopes(Years[YearsSelectGHNC][rgiReal],rgrDataAnnual[:,st][rgiReal])
#             # if slope*10 > 3:
#             #     stop()
#         except:
#             continue
#         rgrGCtrend[st]=slope*10*iScale #(slope/np.nanmean(rgrDataAnnual[:,st]))*1000.
#         rgrPvalue[st]=p_value
#         rgrR2[st]=r_value**2

#     # plot circles that show the trend in area
#     if np.min(lonGHCN) < 0:
#         lonGHCN[lonGHCN < 0]=lonGHCN[lonGHCN < 0]+360
#     for st in range(len(lonGHCN)):
#         if np.isnan(rgrGCtrend[st]) != 1:
#             try:
#                 iColor=np.where((clevsTD-rgrGCtrend[st]) > 0)[0][0]-1
#             except:
#                 if rgrGCtrend[st] > 0:
#                     iColor=len(clevsTD)-1
#             if iColor == -1:
#                 iColor=0
#             if (rgrPvalue[st] <= 0.05):
#                 ax.plot(lonGHCN[st], latGHCN[st],'o',color=rgrColorTable[iColor], ms=iMS, mec='k', alpha=1, markeredgewidth=0.5, transform=cartopy.crs.PlateCarree(), zorder=20)
#             else:
#                 # not significant trend
#                 try:
#                     ax.plot(lonGHCN[st], latGHCN[st],'o',color=rgrColorTable[iColor], ms=iMS, mec=rgrColorTable[iColor], alpha=1, markeredgewidth=0.0, transform=cartopy.crs.PlateCarree(), zorder=20)
#                 except:
#                     stop()

#     ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='#bdbdbd', 
#                    facecolor='#bdbdbd')
#     ax.add_feature(cartopy.feature.OCEAN, zorder=0, facecolor='white', edgecolor='white')
#     ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5, zorder=11, edgecolor='white')
#     ax.add_feature(cartopy.feature.COASTLINE, zorder=11, edgecolor='white')
#     ax.add_feature(cartopy.feature.STATES, zorder=11, alpha=.5, edgecolor='white')

#     ax.outline_patch.set_edgecolor('white')
#     # ax.gridlines(zorder=11)
#     plt.axis('off')

#     # lable the maps
#     tt = ax.text(0.03,0.99, rgsLableABC[se+5]+') '+rgsSeasons[se] , ha='left',va='top', \
#                  transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=20, zorder=21)
#     tt.set_bbox(dict(facecolor='w', alpha=1, edgecolor='w'))
    
#     # ==========================================
#     # Add inlay that shows histogram of trends
#     if se == 0:
#         sub_axes = plt.axes([0.3, .38, .12, .13])
#     if se == 1:
#         sub_axes = plt.axes([0.61, .53, .08, .08])
#     if se == 2:
#         sub_axes = plt.axes([0.9, .53, .08, .08])
#     if se == 3:
#         sub_axes = plt.axes([0.61, .38, .08, .08])
#     if se == 4:
#         sub_axes = plt.axes([0.9, .38, .08, .08])
#     # Narrower bandwidth
#     if se != 3:
#         Trends=rgrGCtrend[(~np.isnan(rgrGCtrend) == True) & ((lonGHCN < 360-60) & (lonGHCN > 360-160) & (latGHCN > 25))]
#     else:
#         Trends=rgrGCtrend[(~np.isnan(rgrGCtrend) == True) & ((lonGHCN < 360-60) & (lonGHCN > 360-160) & (latGHCN > 49))]
#     sns.kdeplot(Trends, shade=True, bw=.05, color="k")
#     Pos=(np.sum(Trends > 0)/float(len(Trends)))*100.
#     plt.title(str(int(np.round(100-Pos)))+'% | '+str(int(np.round(Pos)))+' %', fontsize=12)

#     plt.xlim(-10, 10)
#     if se == 0:
#         plt.ylabel('Probability []')
#     plt.xlabel('Trend [% decade$^{-1}$]')
#     sub_axes.axvline(x=0, c='k', lw=0.5)
#     sub_axes.spines['right'].set_visible(False)
#     sub_axes.spines['top'].set_visible(False)
#     sub_axes.yaxis.set_ticks_position('left')
#     sub_axes.xaxis.set_ticks_position('bottom')



# ==============================================================================
# ==============================================================================
# Add time lines for subregions

# rgrT2Mmonth=np.load('ERA-20C_T2M.npy')
# rgsVars=rgsVars+['T2M']
# rgrMonthlyData=np.append(rgrMonthlyData,rgrT2Mmonth[:,:,:,:,None], axis=4)

# read in the subregions
gs1 = gridspec.GridSpec(1,3)
gs1.update(left=0.06, right=0.95,
           bottom=0.05, top=0.23,
           wspace=0.35, hspace=0.2)

rgsSubregions=['Africa','Asia','North-America','South-America','Australia','Europe']
sContinents='/glade/u/home/prein/ShapeFiles/Continents/'
grShapeFiles={}
for sr in range(len(rgsSubregions)):
    ctr = shapefile.Reader(sContinents+rgsSubregions[sr]+'_poly')
    geomet = ctr.shapeRecords() # will store the geometry separately
    for sh in range(len(geomet)):
        first = geomet[sh]
        grShapeFiles[rgsSubregions[sr]]=first.shape.points

sLabel='$\Delta$ snow vs. precipitation\nday ratio'
sUnit='[%]'
# rgrDataVar=(rgrMonthlyData[:,:,:,:,rgsVars.index('Snow_PRdays')]/(rgrMonthlyData[:,:,:,:,rgsVars.index('Snow_PRdays')]+rgrMonthlyData[:,:,:,:,rgsVars.index('Rain_PRdays')]))*100.
# rgrDataVarEI=(rgrMonthlyData_EI[:,:,:,:,rgsVars.index('Snow_PRdays')]/(rgrMonthlyData_EI[:,:,:,:,rgsVars.index('Snow_PRdays')]+rgrMonthlyData_EI[:,:,:,:,rgsVars.index('Rain_PRdays')]))*100.
# rgrROdata=(rgrMonDat[:,:,:,rgsVarsUW.index('Snow_PRdays')]/(rgrMonDat[:,:,:,rgsVarsUW.index('Snow_PRdays')]+rgrMonDat[:,:,:,rgsVarsUW.index('Rain_PRdays')]))*100.


# rgiFin=np.sum(~np.isnan(rgrROdata[:,:,:]), axis=(0,1))
# rgiFin=(rgiFin >= 3*26)
# rgrROdataAct=rgrROdata[:,:,:][:,:,rgiFin]
rgrLonAct=rgrLon
rgrLatAct=rgrLat
# rgrDataVarEIAct=rgrDataVarEI[:,:,:]
# rgrDataVarAct=rgrDataVar[:,:,:]
rgrDataVar=E20_WCLD
rgrDataVarEI=EI_WCLD

SnowRS=RS_SNOW[:,:,:]; SnowRS[:,:,np.sum(~np.isnan(SnowRS), axis=(0,1)) < (SnowRS.shape[0]*SnowRS.shape[1]*0.66)]=np.nan; SnowRS[:,:,np.nanmedian(SnowRS, axis=(0,1)) == 0]=np.nan
PrecipSR=RS_PR[:,:,:]; PrecipSR[:,:,np.sum(~np.isnan(PrecipSR), axis=(0,1)) < (PrecipSR.shape[0]*PrecipSR.shape[1]*0.66)]=np.nan
rgrDataAnnual=(np.nansum(SnowRS,axis=1)/np.nansum(PrecipSR,axis=1))*100

rgrROdata=rgrDataAnnual #RS_WCLD
rgrROdataAct=rgrROdata
rgrDataVarEIAct=rgrDataVarEI
rgrDataVarAct=rgrDataVar

# loead monthly ERA-20C T2M data | comes from ~/ERA-Batch-Download/T2M_1900_2010_ERA-20C.py
E20_YYYY_Full=pd.date_range(datetime.datetime(1900, 1, 1,00),
                        end=datetime.datetime(2010, 12, 31,23), freq='y')
ncid=Dataset('/glade/scratch/prein/ERA-20C/T2M_1900-2010.nc', mode='r')
rgrDataVarActT2M=np.squeeze(ncid.variables['t2m'][:])
NaNs=np.zeros((rgrDataVarActT2M.shape[0],1,rgrDataVarActT2M.shape[2])); NaNs[:]=np.nan
rgrDataVarActT2M=np.append(rgrDataVarActT2M,NaNs, axis=1)
rgrDataVarActT2M=np.reshape(rgrDataVarActT2M, (int(rgrDataVarActT2M.shape[0]/12), 12,rgrDataVarActT2M.shape[1], rgrDataVarActT2M.shape[2]))
ncid.close()

ii=0
# iYrange=[7,7,7]
for sr in [2,5,1]:
    ax = plt.subplot(gs1[0,ii])
    rgdShapeAct=np.array(grShapeFiles[rgsSubregions[sr]])
    rgdShapeAct[rgdShapeAct < 0]=rgdShapeAct[rgdShapeAct < 0]+360
    PATH = path.Path(rgdShapeAct)
    for da in range(len(rgsData)):
        if rgsData[da] == 'Rad-Soundings':
            TEST=np.copy(rgrLonAct)
            # TEST[TEST>180]=TEST[TEST>180]-360
            flags = PATH.contains_points(np.hstack((TEST[:,np.newaxis],rgrLatAct[:,np.newaxis])))
            rgiYearsAct=np.array(range(1979,2018,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            rgiROstations=np.nansum((np.nansum(rgrROdataAct[:,flags], axis=0) != 0))
            iROstations=(np.nansum(rgrROdataAct[:,flags], axis=0) != 0)
            rgrTMPData=np.nanmedian(rgrROdataAct[:,flags][:,iROstations]-np.nanmean(rgrROdataAct[rgiSelYY,:][:,flags][:,iROstations], axis=(0)), axis=1)
            
            sColor='k'
            # plt.plot(np.array(rgdShapeAct)[:,0],np.array(rgdShapeAct)[:,1]); plt.scatter(TEST[flags],rgrLatAct[flags]); plt.show()
        if rgsData[da] == 'ERA-20C':
            TEST=np.copy(rgrLonGrid2D)
            # TEST[TEST>180]=TEST[TEST>180]-360
            flags = PATH.contains_points(np.hstack((TEST.flatten()[:,np.newaxis],rgrLatGrid2D.flatten()[:,np.newaxis])))
            rgiYearsAct=np.array(range(1900,2011,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            rgrDataVarTMP=np.reshape(rgrDataVarAct, (rgrDataVarAct.shape[0],rgrDataVarAct.shape[1]*rgrDataVarAct.shape[2]))
            rgrTMPData=np.nanmean(rgrDataVarTMP[:,flags]-np.nanmean(rgrDataVarTMP[rgiSelYY,:][:,flags], axis=(0))[None,:], axis=1)
            # T2M
            rgrDataVarTMPT2M=np.reshape(rgrDataVarActT2M, (rgrDataVarActT2M.shape[0],rgrDataVarActT2M.shape[1],rgrDataVarActT2M.shape[2]*rgrDataVarActT2M.shape[3]))
            rgiYearsAll=np.array(range(1900,2011,1))
            rgiSelYYAll=((rgiYearsAll >= 1979) & (rgiYearsAll <= 2010))
            rgrTMPDataT2M=np.nanmean(np.nanmean(rgrDataVarTMPT2M[:,:,flags],axis=1)-np.nanmean(rgrDataVarTMPT2M[rgiSelYYAll,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
            sColor='#1f78b4'
        if rgsData[da] == 'ERA-Interim':
            TEST=np.copy(rgrLonGrid2D_EI)
            # TEST[TEST>180]=TEST[TEST>180]-360
            flags = PATH.contains_points(np.hstack((TEST.flatten()[:,np.newaxis],rgrLatGrid2D_EI.flatten()[:,np.newaxis])))
            rgiYearsAct=np.array(range(1979,2018,1))
            rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
            rgrDataVarTMP=np.reshape(rgrDataVarEIAct, (rgrDataVarEIAct.shape[0],rgrDataVarEIAct.shape[1]*rgrDataVarEIAct.shape[2]))
            try:
                rgrTMPData=np.nanmean(rgrDataVarTMP[:,flags]-np.nanmean(rgrDataVarTMP[rgiSelYY,:][:,flags], axis=(0))[None,:], axis=1)
            except:
                stop()
            sColor='#e31a1c'

        if (sr == 2):
            if rgsData[da] == 'Rad-Soundings':
                LON=rgrLonAct
                LAT=rgrLatAct
                rgiLandOcean=[bm.is_land(LON[ll], LAT[ll]) for ll in range(len(LAT))]
                rgrDataVarTMP=rgrROdataAct
            if rgsData[da] == 'ERA-20C':
                rgiLandOcean=np.load('LandOcean_E20.npy')
            if rgsData[da] == 'ERA-Interim':
                rgiLandOcean=np.load('LandOcean_EI.npy')
            flags=rgiLandOcean

            # rgrTMPDataGL=np.nanmean(rgrDataVarTMP[:,flags]-np.nanmean(rgrDataVarTMP[rgiSelYY,:][:,flags], axis=(0))[None,:], axis=1)
            # print ' '
            # print '    '+rgsData[da]
            # print '     slope '+str(stats.linregress(range(np.sum(rgiSelYY)),rgrTMPDataGL[rgiSelYY])[0]*10)
            # print '     p-value '+str(stats.linregress(range(np.sum(rgiSelYY)),rgrTMPDataGL[rgiSelYY])[3])

        print ' '
        print '    '+rgsData[da]+' - '+rgsSubregions[sr]

        try:
            print '     slope '+str(stats.linregress(range(rgrSeasonalData.shape[0]),rgrTMPData[rgiSelYY])[0]*10)
            print '     p-value '+str(stats.linregress(range(rgrSeasonalData.shape[0]),rgrTMPData[rgiSelYY])[3])

            x=np.array(range(np.sum(rgiSelYY)), dtype=float)
            y=np.array(rgrTMPData[rgiSelYY], dtype=float)

            pp , ee = optimize.curve_fit(piecewise_linear, x,y)
            xd = np.array(range(32), dtype=float)
            YFitedData=piecewise_linear(xd, *pp)
            start=0
            TRENDS=np.zeros((len(x))); TRENDS[:]=np.nan
            for yy in range(1,len(x)-1,1):
                Tr0=stats.linregress(x[start:yy+1],YFitedData[start:yy+1])[0]
                Tr1=stats.linregress(x[start:yy+2],YFitedData[start:yy+2])[0]
                if Tr0 != Tr1:
                    start=yy
                TRENDS[yy]=Tr0
            TRENDS=np.round(TRENDS, 5); TRENDS[0]=TRENDS[1]; TRENDS[-1]=TRENDS[-2]
            UniqueTR=np.unique(TRENDS); UniqueTR=UniqueTR[~np.isnan(UniqueTR)]
            ComYYYY=np.array(range(1979,2011,1))
            for bp in range(len(UniqueTR)):
                print('       '+str(ComYYYY[np.where(TRENDS == UniqueTR[bp])[0][0]])+'-'+str(ComYYYY[np.where(TRENDS == UniqueTR[bp])[0][-1]])+' Trend = '+str(UniqueTR[bp]*10))
                print('      ')

        except:
            continue


        if (rgiROstations < 10) & (rgsData[da] == 'Rad-Soundings'):
            continue
        else:
            plt.plot(rgiYearsAct, rgrTMPData, c=sColor, lw=0.5, zorder=3, alpha=0.5)
            plt.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPData,10), c=sColor, lw=3, zorder=3, label=rgsData[da])
            # plt.plot(iYears_I[iComYears], rgrModData, c='#1f78b4')
            # plt.plot(iYears_I[iComYears], rgrModDataEI, c='#e31a1c')

        if (sr == 2) & (da == 0):
            # add GHCN stations
            YearsSelectGHNC=((Years >= 1979) & (Years <= 2018))
            rgiSelYY=((Years >= 1979) & (Years <= 2010))
            PrecipDaysY=np.nansum(GHCN_monthly_data[YearsSelectGHNC,:,:,TargetVariables.index('RaynDays')]+GHCN_monthly_data[YearsSelectGHNC,:,:,TargetVariables.index('SnowDays')], axis=1)
            SnowDaysY=np.nansum(GHCN_monthly_data[YearsSelectGHNC,:,:,TargetVariables.index('SnowDays')], axis=1)
            SnowRatioY=(SnowDaysY/PrecipDaysY)-np.nanmean((SnowDaysY/PrecipDaysY)[rgiSelYY,:], axis=0)[None,:]
            for st in range(SnowRatioY.shape[1]):
                rgiReal=~np.isnan(SnowRatioY[:,st])
                if (np.median(SnowDaysY[:,st][rgiReal]*iScale) < 2) | (sum(rgiReal) < SnowRatioY.shape[0]*0.7):
                    SnowRatioY[:,st]=np.nan
                else:
                    # remove spurious data
                    YY=SnowRatioY[:,st][rgiReal]
                    YYc=RemoveSpouriousData(YY,sigma=2)
                    if np.sum(rgiReal == False) >0:
                        for kk in range(np.sum(rgiReal == False)):
                            try:
                                YYc=np.insert(YYc,np.where(rgiReal == False)[0][kk],np.nan)
                            except:
                                stop()
                    SnowRatioY[:,st]=YYc
            FIN=((lonGHCN < -60) & (lonGHCN > -160) & (latGHCN > 25) & (np.nanmean(SnowDaysY, axis=(0)) >= 2) & (np.sum(~np.isnan(SnowRatioY), axis=0) > float(len(Years))*iMinCov ))
            GHCN_YYYY=np.nanmean(SnowRatioY[:,FIN]*100., axis=1)
            ax.plot(Years[YearsSelectGHNC],GHCN_YYYY, c='#33a02c', lw=0.5,zorder=3,alpha=0.5)
            plt.plot(Years[YearsSelectGHNC], scipy.ndimage.uniform_filter(GHCN_YYYY,10), c='#33a02c', lw=3, zorder=3, label='observations')

        plt.title(rgsLableABC[ii+10]+') '+rgsSubregions[sr]+' | '+str(rgiROstations))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('Year')
        if ii == 0:
            ax.set_ylabel(sLabel+' '+sUnit)
        else:
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        ax.set(ylim=(Yminmax[0],Yminmax[1]))

        if rgsData[da] == 'ERA-20C':
            Y2col='#636363'
            # Plot T2M on a second y-axis
            ax2 = ax.twinx()
            ax2.set_ylabel('2m temperature (T2M) [$^{\circ}$C]', color=Y2col)
            ax2.plot(rgiYearsAll, scipy.ndimage.uniform_filter(rgrTMPDataT2M,10), c=Y2col, lw=3, zorder=1, alpha=0.5)
            ax2.plot(rgiYearsAll, rgrTMPDataT2M, c=Y2col, lw=0.5, zorder=1, alpha=0.5)
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
sPlotName= 'Snow-VS-PR-Day-Changes.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
