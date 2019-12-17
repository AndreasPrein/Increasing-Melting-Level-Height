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
rgsVars=['ProbSnow']
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
DataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Monthly_Data/'
rgsSeasons=['DJF','MAM','JJA','SON']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30]

rgsData=['Rad-Soundings', 'ERA-Interim','ERA-20C']

dStartDay_MSWEP=datetime.datetime(1979, 1, 1,0)
dStopDay_MSWEP=datetime.datetime(2017, 12, 31,0)
rgdTimeDD_MSWEP = pd.date_range(dStartDay_MSWEP, end=dStopDay_MSWEP, freq='d')
rgdTimeFin=rgdTimeDD_MSWEP
iYears=np.unique(rgdTimeFin.year)

########################################
#      READ IN RADIO SOUNDING DATA
DATA=np.load(DataDir+'RS-data-montly_19002017.npz')
VARIABLES=DATA['VARS']; VARIABLES=[VARIABLES[va] for va in range(len(VARIABLES))]
RS_data=DATA['Monthly_RO']
RS_lat=DATA['RS_Lat']
RS_lon=DATA['RS_Lon']
RS_time=DATA['RSTimeMM']
RS_Breaks=DATA['BreakFlag']

#      READ IN ERA-I DATA
DATA=np.load(DataDir+'EI-data-montly_19792017.npz')
EI_data=DATA['rgrMonthlyData_EI']
EI_lat=DATA['rgrLat75_EI']
EI_lon=DATA['rgrLon75_EI']
EI_time=DATA['rgdTimeMM_EI']
EI_Breaks=DATA['BreakFlag']
rgiYY_EI=np.unique(pd.DatetimeIndex(EI_time).year)
rgrLonGrid2D_EI=np.asarray(([EI_lon,]*EI_lat.shape[0]))
rgrLatGrid2D_EI=np.asarray(([EI_lat,]*EI_lon.shape[0])).transpose()

#      READ IN ERA-20 DATA
DATA=np.load(DataDir+'E20-data-montly_19792010.npz')
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
stop()
RS_WCLD=RS_data[:,:,:,VARIABLES.index(rgsVars[0])]
RS_WCLD[:,:,(np.max(RS_Breaks[:,:,VARIABLES.index(rgsVars[0])], axis=0) > 0.5)]=np.nan; RS_WCLD[:,:,np.sum(~np.isnan(RS_WCLD), axis=(0,1)) < (RS_WCLD.shape[0]*RS_WCLD.shape[1]*0.8)]=np.nan
FIN=~np.isnan(np.nanmean(RS_WCLD, axis=(0,1)))
RS_WCLD=RS_WCLD[:,:,FIN]; RS_lat=RS_lat[FIN] ;RS_lon=RS_lon[FIN]

EI_WCLD=EI_data[:,:,:,:,VARIABLES.index(rgsVars[0])]
EI_WCLD[:,:,(np.max(EI_Breaks[:,:,:,VARIABLES.index(rgsVars[0])], axis=0) > 0.5)]=np.nan; EI_WCLD[:,:,np.sum(~np.isnan(EI_WCLD), axis=(0,1)) < (EI_WCLD.shape[0]*EI_WCLD.shape[1]*0.8)]=np.nan

E20_WCLD=E20_data[:,:,:,:,VARIABLES.index(rgsVars[0])]
E20_WCLD[:,:,(np.max(E20_Breaks[:,:,:,VARIABLES.index(rgsVars[0])], axis=0) > 0.5)]=np.nan; E20_WCLD[:,:,np.sum(~np.isnan(E20_WCLD), axis=(0,1)) < (E20_WCLD.shape[0]*E20_WCLD.shape[1]*0.8)]=np.nan

########################################
# loead monthly ERA-20C T2M data | comes from ~/ERA-Batch-Download/T2M_1900_2010_ERA-20C.py
E20_YYYY_Full=pd.date_range(datetime.datetime(1900, 1, 1,00),
                        end=datetime.datetime(2010, 12, 31,23), freq='y')
ncid=Dataset('/glade/scratch/prein/ERA-20C/T2M_1900-2010.nc', mode='r')
rgrDataVarActT2M=np.squeeze(ncid.variables['t2m'][:])
NaNs=np.zeros((rgrDataVarActT2M.shape[0],1,rgrDataVarActT2M.shape[2])); NaNs[:]=np.nan
rgrDataVarActT2M=np.append(rgrDataVarActT2M,NaNs, axis=1)
rgrDataVarActT2M=np.reshape(rgrDataVarActT2M, (int(rgrDataVarActT2M.shape[0]/12), 12,rgrDataVarActT2M.shape[1], rgrDataVarActT2M.shape[2]))
ncid.close()
rgiYearsAct=np.array(range(1900,2011,1))
rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
rgrDataVarActT2M=rgrDataVarActT2M[rgiSelYY,:,:,:]

########################################
#                      start plotting
# loop over the different variables
# rgsVars=['BflLR']

for va in [rgsVars.index('ProbSnow')]: #range(len(rgsVars)):
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
        rgrDataVar=RS_WCLD
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
    
        rgrGCtrend=np.zeros((len(RS_lon))); rgrGCtrend[:]=np.nan
        rgrPvalue=np.copy(rgrGCtrend)
        rgrR2=np.copy(rgrGCtrend)
        # get the data
        rgrDataAnnual=np.mean(rgrDataVar,axis=1)

        for st in range(len(RS_lon)):
            # remove nans if there are some
            rgiReal=~np.isnan(rgrDataAnnual[(iYears <= 2010),st])
            if (np.mean(rgrDataAnnual[(iYears <= 2010),st][rgiReal]*iScale) < 10) | (len(rgiReal) < (2010-1979)*0.7):
                continue
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(iYears[(iYears <= 2010)][rgiReal],rgrDataAnnual[(iYears <= 2010),st][rgiReal])
            except:
                continue
            rgrGCtrend[st]=slope*10*iScale #(slope/np.nanmean(rgrDataAnnual[:,st]))*1000.
            rgrPvalue[st]=p_value
            rgrR2[st]=r_value**2
    
        # plot circles that show the trend in area
        if np.min(RS_lon) < 0:
            RS_lon[RS_lon < 0]=RS_lon[RS_lon < 0]+360
        for st in range(len(RS_lon)):
            if np.isnan(rgrGCtrend[st]) != 1:
                try:
                    iColor=np.where((clevsTD-rgrGCtrend[st]) > 0)[0][0]-1
                except:
                    if rgrGCtrend[st] > 0:
                        iColor=len(clevsTD)-1
                if iColor == -1:
                    iColor=0
                if (rgrPvalue[st] <= 0.05):
                    ax.plot(RS_lon[st], RS_lat[st],'o',color=rgrColorTable[iColor], ms=iMS, mec='k', alpha=1, markeredgewidth=1, transform=cartopy.crs.PlateCarree(), zorder=20)
                else:
                    # not significant trend
                    try:
                        ax.plot(RS_lon[st], RS_lat[st],'o',color=rgrColorTable[iColor], ms=iMS, mec=rgrColorTable[iColor], alpha=1, markeredgewidth=0.0,transform=cartopy.crs.PlateCarree(), zorder=20)
                    except:
                        stop()

        
        ###############################################################
        print '    Add ERA-20C analysis'
        rgrMonthAct=E20_WCLD
        rgrLonAct=rgrLonGrid2D
        rgrLatAct=rgrLatGrid2D
    
        rgrTrendsERA20=np.zeros((rgrMonthAct.shape[2],rgrMonthAct.shape[3],5)); rgrTrendsERA20[:]=np.nan
        rgrSeasonalData=np.mean(rgrMonthAct[:,rgiMonths,:,:][(rgiYY >=1979),:],axis=1)
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

    biasContDist=20
    sLabel='prob. of snow'
    sUnit='[days]'
    rgrDataVar=E20_WCLD
    rgrDataVarEI=EI_WCLD
    rgrROdata=RS_WCLD
    
    rgiFin=np.sum(~np.isnan(rgrROdata[:,:,:]), axis=(0,1))
    rgiFin=(rgiFin >= 3*26)
    rgrROdataAct=rgrROdata[:,:,:][:,:,rgiFin]
    rgrLonAct=rgrLonS[rgiFin]
    rgrLatAct=rgrLatS[rgiFin]
    rgrDataVarEIAct=rgrDataVarEI[:,:,:]
    rgrDataVarAct=rgrDataVar[:,:,:]
    rgrDataVarActT2M=rgrDataVarActT2M

    if (rgsVars[va] == 'ProbSnow'):
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
                rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
                rgrTMPData=np.nanmean(np.nanmean(rgrROdataAct[:,:,flags],axis=(1))-np.nanmean(rgrROdataAct[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                sColor='k'
            if rgsData[da] == 'ERA-20C':
                TEST=np.copy(rgrLonGrid2D)
                TEST[TEST>180]=TEST[TEST>180]-360
                flags = PATH.contains_points(np.hstack((TEST.flatten()[:,np.newaxis],rgrLatGrid2D.flatten()[:,np.newaxis])))
                rgiYearsAct=np.array(range(1900,2010,1))
                rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
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
                rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
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
            ax.set(ylim=(-23,23))

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
    sPlotName= 'Snow-Probability-Changes.pdf'
    if os.path.isdir(sPlotFile) != 1:
        subprocess.call(["mkdir","-p",sPlotFile])
    print '        Plot map to: '+sPlotFile+sPlotName
    fig.savefig(sPlotFile+sPlotName)

    stop()
