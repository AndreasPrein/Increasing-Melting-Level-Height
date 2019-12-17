#!/usr/bin/env python
'''
    File name: Trends-WBZH_final-plot.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    Read in the preprocessed data from:
    ~/projects/Hail/programs/RadioSoundings/Preprocessor/HailParametersFromRadioSounding.py

    Calculate the annual time series for each contintne from the RS, ERA-20, and ERA-Interim dataset

    Plot the annual and seasonal time series

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
import scipy
from scipy import signal
from matplotlib.widgets import RectangleSelector
from scipy import optimize

from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
def grid(x, y, z, resX=20, resY=20):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z

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

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

########################################
#                            Settings
rgsVars=['WBZheight']
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
DataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Monthly_Data/'
rgsSeasons=['DJF','MAM','JJA','SON','Annual']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30,365.25]

rgsData=['Rad-Soundings','ERA-20C', 'ERA-Interim']

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
RS_lat=DATA['RS_Lat']
RS_lon=DATA['RS_Lon']
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
RS_WCLD=RS_data[:,:,:,VARIABLES_RS.index(rgsVars[0])]
RS_WCLD[:,:,(np.max(RS_Breaks[:,:,VARIABLES_RS.index(rgsVars[0])], axis=0) > 0.5)]=np.nan; RS_WCLD[:,:,np.sum(~np.isnan(RS_WCLD), axis=(0,1)) < (RS_WCLD.shape[0]*RS_WCLD.shape[1]*0.8)]=np.nan
FIN=~np.isnan(np.nanmean(RS_WCLD, axis=(0,1)))
RS_WCLD=RS_WCLD[:,:,FIN]; RS_lat=RS_lat[FIN] ;RS_lon=RS_lon[FIN]

EI_WCLD=EI_data[:,:,:,:,VARIABLES.index(rgsVars[0])]
EI_WCLD[:,:,(np.max(EI_Breaks[:,:,:,VARIABLES.index(rgsVars[0])], axis=0) > 0.5)]=np.nan; EI_WCLD[:,:,np.sum(~np.isnan(EI_WCLD), axis=(0,1)) < (EI_WCLD.shape[0]*EI_WCLD.shape[1]*0.8)]=np.nan

E20_WCLD=E20_data[:,:,:,:,VARIABLES_E20.index(rgsVars[0])]
E20_WCLD[:,:,(np.max(E20_Breaks[:,:,:,VARIABLES_E20.index(rgsVars[0])], axis=0) > 0.5)]=np.nan; E20_WCLD[:,:,np.sum(~np.isnan(E20_WCLD), axis=(0,1)) < (E20_WCLD.shape[0]*E20_WCLD.shape[1]*0.8)]=np.nan

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
rgiSelYY=((rgiYearsAct >= 1900) & (rgiYearsAct <= 2010))
rgrDataVarActT2M=rgrDataVarActT2M[rgiSelYY,:,:,:]

# =======================================================
# =======================================================
# PLOT THE EVALUATION FOR CONTINENTS
# =======================================================

for va in [0]:
    print '    Plotting '+rgsVars[va]
    if rgsVars[va] == 'WBZheight':
        biasContDist=20
        sLabel='$\Delta$ML height'
        sUnit='[km]'

        rgrDataVar=E20_WCLD[:,:,:]/1000.
        rgrDataVarEI=EI_WCLD/1000.
        rgrROdata=RS_WCLD/1000.
        Yrange=[-0.2,0.2]

    for se in [4]: #range(5):
        fig = plt.figure(figsize=(13, 8))
        plt.rcParams.update({'font.size': 8})
        gs1 = gridspec.GridSpec(1,1)
        gs1.update(left=0.15, right=1,
                   bottom=0.33, top=0.94,
                   wspace=0.2, hspace=0.1)

        # scalings
        if rgsVars[va] == 'ProbSnow':
            iScale=iDaysInSeason[se]
        else:
            iScale=1.
        print '    Work on Season '+str(se)
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

        # remove stations with too little coverage --> min are 26 years
        rgiFin=np.sum(~np.isnan(rgrROdata[:,rgiTime,:]), axis=(0,1))
        rgiFin=(rgiFin >= 3*26)
        rgrROdataAct=rgrROdata[:,rgiTime,:][:,:,rgiFin]
        rgrLonAct=RS_lon[:][rgiFin]
        rgrLatAct=RS_lat[:][rgiFin]
        rgrDataVarEIAct=EI_WCLD[:,rgiTime,:]
        rgrDataVarAct=E20_WCLD[:,rgiTime,:]
        rgrDataVarActT2M=rgrDataVarActT2M[:,rgiTime,:,:]

        # ax = plt.subplot(gs1[0,0], projection=cartopy.crs.PlateCarree())
        # ax.set_extent([-175, 170, -60, 80], crs=ccrs.PlateCarree())
        ax = plt.subplot(gs1[0,0], projection=cartopy.crs.Robinson())
        ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
        ax.outline_patch.set_linewidth(0)
        ax.set_facecolor('#c6dbef')
        fig.patch.set_facecolor('#c6dbef')

        ax.add_feature(cartopy.feature.OCEAN, zorder=0, facecolor='#c6dbef')
        ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='#ffffe5', facecolor='#ffffe5')        
        # plt.title(sLabel+' '+rgsSeasons[se], fontsize=17)
        plt.axis('off')  

        # ========================
        # plot inlays for each continent
        for sr in range(len(rgsSubregions)):
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
                    rgiYearsAct=np.array(range(1900,2011,1))
                    rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
                    rgrDataVarTMP=np.reshape(rgrDataVarAct, (rgrDataVarAct.shape[0],rgrDataVarAct.shape[1],rgrDataVarAct.shape[2]*rgrDataVarAct.shape[3]))
                    Weights=np.transpose(np.tile(np.cos(np.linspace(-90,90,rgrDataVarAct.shape[2])/(360/(np.pi*2))),(rgrDataVarAct.shape[3],1))).flatten()
                    indices = ~np.isnan(np.mean(rgrDataVarTMP[:,:,flags], axis=(0,1)))
                    Full=np.array([np.average(np.median(rgrDataVarTMP[yy,:,flags],axis=1)[indices], weights=Weights[flags][indices]) for yy in range(rgrDataVarTMP.shape[0])])
                    rgrTMPData=(Full-np.median(Full[rgiSelYY]))/1000.
                    # rgrTMPData=np.nanmean(np.nanmean(rgrDataVarTMP[:,:,flags],axis=(1))-np.nanmean(rgrDataVarTMP[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)/1000.
                    # T2M
                    rgrDataVarTMPT2M=np.reshape(rgrDataVarActT2M, (rgrDataVarActT2M.shape[0],rgrDataVarActT2M.shape[1],rgrDataVarActT2M.shape[2]*rgrDataVarActT2M.shape[3]))
                    indices = ~np.isnan(np.mean(rgrDataVarTMPT2M[:,:,flags], axis=(0,1)))
                    Full=np.array([np.average(np.median(rgrDataVarTMPT2M[yy,:,flags],axis=1)[indices], weights=Weights[flags][indices]) for yy in range(rgrDataVarTMPT2M.shape[0])])
                    rgrTMPDataT2M=Full-np.median(Full[rgiSelYY])
                    # rgrTMPDataT2M=np.nanmean(np.nanmean(rgrDataVarTMPT2M[:,:,flags],axis=1)-np.nanmean(rgrDataVarTMPT2M[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                    sColor='#1f78b4'
                if rgsData[da] == 'ERA-Interim':
                    TEST=np.copy(rgrLonGrid2D_EI)
                    TEST[TEST>180]=TEST[TEST>180]-360
                    flags = PATH.contains_points(np.hstack((TEST.flatten()[:,np.newaxis],rgrLatGrid2D_EI.flatten()[:,np.newaxis])))
                    rgiYearsAct=np.array(range(1979,2018,1))
                    rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
                    Weights=np.transpose(np.tile(np.cos(np.linspace(-90,90,rgrDataVarEIAct.shape[2])/(360/(np.pi*2))),(rgrDataVarEIAct.shape[3],1))).flatten()
                    rgrDataVarTMP=np.reshape(rgrDataVarEIAct, (rgrDataVarEIAct.shape[0],rgrDataVarEIAct.shape[1],rgrDataVarEIAct.shape[2]*rgrDataVarEIAct.shape[3]))
                    try:
                        indices = ~np.isnan(np.mean(rgrDataVarTMP[:,:,flags], axis=(0,1)))
                        Full=np.array([np.average(np.median(rgrDataVarTMP[yy,:,flags],axis=1)[indices], weights=Weights[flags][indices]) for yy in range(rgrDataVarTMP.shape[0])])
                        rgrTMPData=(Full-np.median(Full[rgiSelYY]))/1000.
                        # rgrTMPData=np.nanmean(np.nanmean(rgrDataVarTMP[:,:,flags],axis=(1))-np.nanmean(rgrDataVarTMP[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)/1000.
                    except:
                        stop()
                    sColor='#e31a1c'
    
                if rgsData[da] == 'ERA-20C':
                    Y2col='#636363'
                    # Plot T2M on a second y-axis
                    ax2 = a.twinx()
                    ax2.set_ylabel('$\Delta$2m temperature [$^{\circ}$C]', color=Y2col)
                    ax2.plot(rgiYearsAct, rgrTMPDataT2M, c=Y2col, lw=0.5, label='hail diameter', zorder=1, alpha=0.5)
                    ax2.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPDataT2M,10), c=Y2col, lw=2, zorder=1, alpha=0.5)
                    # ax2.set(ylim=(30,45))
                    # ax2.set_yticks(np.arange(30, 50, 5))
                    ax2.spines['right'].set_color(Y2col)
                    ax2.yaxis.label.set_color(Y2col)
                    ax2.tick_params(axis='y', colors=Y2col)
                    ax2.spines['left'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    ax2.set_ylim(-1.1,1.1)


                # rgrObsData=np.nanmean(rgrROdataAct[:,:,flags]-np.nanmean(rgrROdataAct[:,:,flags], axis=(0,1)), axis=(1,2))+np.nanmean(rgrROdataAct[:,:,flags])
                # rgrModData=np.nanmean(rgrDataVarAct[:,:,flags]-np.nanmean(rgrDataVarAct[:,:,flags], axis=(0,1)), axis=(1,2))+np.nanmean(rgrDataVarAct[:,:,flags])
                # rgrModDataEI=np.nanmean(rgrDataVarActEI[:,:,flags]-np.nanmean(rgrDataVarActEI[:,:,flags], axis=(0,1)), axis=(1,2))+np.nanmean(rgrDataVarActEI[:,:,flags])
                # this is an inset axes over the main axes
                Low=0.2
                Hight=0.8
                if rgsSubregions[sr] == 'Africa':
                    a = plt.axes([.57, .48, .15, .22*Hight], facecolor=None)
                    lab='g'
                if rgsSubregions[sr] == 'Asia':
                    a = plt.axes([.77, .63+0.07, .15, .22*Hight], facecolor=None)
                    lab='h'
                if rgsSubregions[sr] == 'North-America':
                    a = plt.axes([.29, .61+0.07, .15, .22*Hight], facecolor=None)
                    lab='d'
                if rgsSubregions[sr] == 'South-America':
                    a = plt.axes([.33, .39, .15, .22*Hight], facecolor=None)
                    lab='e'
                if rgsSubregions[sr] == 'Australia':
                    a = plt.axes([.805, .42, .15, .22*Hight], facecolor=None)
                    lab='i'
                if rgsSubregions[sr] == 'Europe':
                    a = plt.axes([.53, .73, .15, .22*Hight], facecolor=None)
                    lab='f'
                a.patch.set_alpha(0)
                if (rgiROstations < 10) & (rgsData[da] == 'Rad-Soundings'):
                    continue
                else:
                    plt.plot(rgiYearsAct, rgrTMPData, c=sColor, lw=0.5, zorder=3, alpha=0.5)
                    plt.plot(rgiYearsAct,  scipy.ndimage.uniform_filter(rgrTMPData,10), c=sColor, lw=2, zorder=3)
                    # plt.plot(iYears_I[iComYears], rgrModData, c='#1f78b4')
                    # plt.plot(iYears_I[iComYears], rgrModDataEI, c='#e31a1c')

                plt.title(lab+') '+rgsSubregions[sr]+' | '+str(rgiROstations))
                a.spines['right'].set_visible(False)
                a.spines['top'].set_visible(False)
                plt.xlabel('Year')
                plt.ylabel(sLabel+' '+sUnit)
                plt.ylim(Yrange[0],Yrange[1])

        # ========================
        # plot inlays for land, ocean, and global
        rgsGlobal=['Global','Global Land','Global Ocean']
        for sr in range(len(rgsGlobal)):
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
                    DATA=np.reshape(rgrDataVarAct, (rgrDataVarAct.shape[0],rgrDataVarAct.shape[1],rgrDataVarAct.shape[2]*rgrDataVarAct.shape[3]))/1000.
                    rgiYearsAct=np.array(range(1900,2011,1))
                    rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
                    sColor='#1f78b4'
                    # T2M
                    DATAT2M=np.reshape(rgrDataVarActT2M, (rgrDataVarActT2M.shape[0],rgrDataVarActT2M.shape[1],rgrDataVarActT2M.shape[2]*rgrDataVarActT2M.shape[3]))
                    # TEST=np.copy(LON)
                    # TEST[TEST>180]=TEST[TEST>180]-360
                    # rgiLandOceanE20=[bm.is_land(TEST[ll], LAT[ll]) for ll in range(len(LAT))]
                    # np.save('LandOcean_E20.npy',rgiLandOceanE20)
                    rgiLandOcean=np.load('LandOcean_E20.npy')
                    Weights=np.transpose(np.tile(np.cos(np.linspace(-90,90,rgrDataVarAct.shape[2])/(360/(np.pi*2))),(rgrDataVarAct.shape[3],1))).flatten()
                if rgsData[da] == 'ERA-Interim':
                    LON=rgrLonGrid2D_EI.flatten()
                    LAT=rgrLatGrid2D_EI.flatten()
                    DATA=np.reshape(rgrDataVarEIAct, (rgrDataVarEIAct.shape[0],rgrDataVarEIAct.shape[1],rgrDataVarEIAct.shape[2]*rgrDataVarEIAct.shape[3]))/1000.
                    rgiYearsAct=np.array(range(1979,2018,1))
                    rgiSelYY=((rgiYearsAct >= 1979) & (rgiYearsAct <= 2010))
                    sColor='#e31a1c'
                    # TEST=np.copy(LON)
                    # TEST[TEST>180]=TEST[TEST>180]-360
                    # rgiLandOceanEI=[bm.is_land(TEST[ll], LAT[ll]) for ll in range(len(LAT))]
                    # np.save('LandOcean_EI.npy',rgiLandOceanEI)
                    rgiLandOcean=np.load('LandOcean_EI.npy')
                    Weights=np.transpose(np.tile(np.cos(np.linspace(-90,90,rgrDataVarEIAct.shape[2])/(360/(np.pi*2))),(rgrDataVarEIAct.shape[3],1))).flatten()


                if rgsGlobal[sr] == 'Global':
                    flags=np.array([True]*DATA.shape[2])
                    a = plt.axes([0.05, .80, .15, .14]) #, facecolor='y')
                if rgsGlobal[sr] == 'Global Land':
                    a = plt.axes([0.05, .59, .15, .14]) #, facecolor='y')
                    flags=rgiLandOcean
                if rgsGlobal[sr] == 'Global Ocean':
                    a = plt.axes([0.05, .38, .15, .14]) #, facecolor='y')
                    flags=(rgiLandOcean == False)

                
                if (rgsData[da] == 'Rad-Soundings'):
                    rgrTMPData=np.nanmedian(np.nanmean(DATA[:,:,flags],axis=(1))-np.nanmean(DATA[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                else:
                    indices = ~np.isnan(np.mean(DATA[:,:,flags], axis=(0,1)))
                    Full=np.array([np.average(np.median(DATA[yy,:,flags],axis=1)[indices], weights=Weights[flags][indices]) for yy in range(DATA.shape[0])])
                    rgrTMPData=Full-np.median(Full[rgiSelYY])

                if rgsGlobal[sr] == 'Global Land':
                    print ' '
                    print '    '+rgsData[da]
                    print '     slope '+str(stats.linregress(range(np.sum(rgiSelYY)),rgrTMPData[rgiSelYY])[0]*10)
                    print '     p-value '+str(stats.linregress(range(np.sum(rgiSelYY)),rgrTMPData[rgiSelYY])[3])

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

                    # plt.plot(ComYYYY,rgrTMPData[rgiSelYY])[0]; plt.plot(xd+1979,YFitedData);  plt.show()
                    
                # rgrTMPData=np.nanmean(np.nanmean(DATA[:,:,flags],axis=(1))-np.nanmean(DATA[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
    
                # # calculate cc signals and trends for the paper
                # CCS_bs=[np.mean(np.random.choice(rgrTMPData[-30:],len(rgrTMPData[-30:]))) - np.mean(np.random.choice(rgrTMPData[:30],len(rgrTMPData[:30]))) for ii in range(1000)]
                # rgrTMPDataT2M=np.nanmean(np.nanmean(DATAT2M[:,:,flags],axis=(0))-np.nanmean(DATAT2M[:,rgiSelYY,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                # SCALING=[(np.mean(np.random.choice(rgrTMPData[-30:],len(rgrTMPData[-30:]))) - np.mean(np.random.choice(rgrTMPData[:30],len(rgrTMPData[:30])))) / (np.mean(np.random.choice(rgrTMPDataT2M[-30:],len(rgrTMPDataT2M[-30:]))) - np.mean(np.random.choice(rgrTMPDataT2M[:30],len(rgrTMPDataT2M[:30])))) for ii in range(1000)]
                # TRENDS=stats.linregress(rgiYearsAct[(rgiYearsAct >= 1979) & (rgiYearsAct <=2010)],rgrTMPData[(rgiYearsAct >= 1979) & (rgiYearsAct <=2010)])
                # stop()
                # rgdShapeAct=grShapeFiles['North-America']
                # PATH = path.Path(rgdShapeAct)
                # TEST=np.copy(rgrLonGrid2D)
                # TEST[TEST>180]=TEST[TEST>180]-360
                # flags = PATH.contains_points(np.hstack((TEST.flatten()[:,np.newaxis],rgrLatGrid2D.flatten()[:,np.newaxis])))

                a.patch.set_alpha(0)
                a.plot(rgiYearsAct, rgrTMPData, c=sColor, lw=1)
                a.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPData,10), c=sColor, lw=2, label=rgsData[da])
                plt.title(rgsLableABC[sr]+') '+rgsGlobal[sr])
                a.spines['right'].set_visible(False)
                a.spines['top'].set_visible(False)
                if sr ==2:
                    plt.xlabel('Year')
                plt.ylabel(sLabel+' '+sUnit)
                a.set_facecolor('#ffffff')
                plt.ylim(Yrange[0],Yrange[1])

                if rgsData[da] == 'ERA-20C':
                    indices = ~np.isnan(np.mean(DATAT2M[:,:,flags], axis=(0,1)))
                    Full=np.array([np.average(np.mean(DATAT2M[yy,:,flags],axis=1)[indices], weights=Weights[flags][indices]) for yy in range(DATAT2M.shape[0])])
                    rgrTMPDataT2M=Full-np.mean(Full[rgiSelYY])
                    # rgrTMPDataT2M=np.nanmean(np.nanmean(DATAT2M[:,:,flags],axis=(1))-np.nanmean(DATAT2M[rgiSelYY,:,:][:,:,flags], axis=(0,1))[None,:], axis=1)
                    Y2col='#636363'
                    # Plot T2M on a second y-axis
                    ax2 = a.twinx()
                    ax2.set_ylabel('$\Delta$2m temperature [$^{\circ}$C]', color=Y2col)
                    ax2.plot(rgiYearsAct, rgrTMPDataT2M, c=Y2col, lw=0.5, zorder=1, alpha=0.5)
                    ax2.plot(rgiYearsAct, scipy.ndimage.uniform_filter(rgrTMPDataT2M), c=Y2col, lw=2, zorder=1, alpha=0.5)
                    # ax2.set(ylim=(30,45))
                    # ax2.set_yticks(np.arange(30, 50, 5))
                    ax2.spines['right'].set_color(Y2col)
                    ax2.yaxis.label.set_color(Y2col)
                    ax2.tick_params(axis='y', colors=Y2col)
                    ax2.spines['left'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    ax2.set_ylim(-1.1,1.1)
                    
            a.plot([2000,2000],[0,0],lw=2,alpha=0.5, c='#636363', label='ERA-20C T2M')
            if sr == 0:
                # lns = lns1+lns2
                # labs = [l.get_label() for l in lns]
                a.legend(ncol=1, prop={'size':8},\
                         bbox_to_anchor=(4.2, -2.6), framealpha=0.3,fancybox=True)
                
        from matplotlib.widgets import RectangleSelector
        overax = fig.add_axes([0,0,1,1])
        overax.patch.set_alpha(0)
        overax.axis("off")
        rs = RectangleSelector(overax, select,
                       drawtype='box', useblit=True,
                       button=[1, 3],
                       minspanx=5, minspany=5,
                       spancoords='pixels',
                       interactive=True)

        # -----------------------------------------------------
        # -----------------------------------------------------
        #            ADD GLOBAL MAPS OF T2M, WBZH, & SCALING
        gs1 = gridspec.GridSpec(1,3)
        gs1.update(left=0.01, right=0.93,
                   bottom=0.02, top=0.32,
                   wspace=0.2, hspace=0.1)
        # T2M
        ax = plt.subplot(gs1[0,0], projection=cartopy.crs.Robinson())
        biasContDist=0.6
        iMinMax=8.*biasContDist/2.+biasContDist/2.
        clevsTD=np.arange(-iMinMax,iMinMax+0.0001,biasContDist)
        DATA=np.mean(rgrDataVarActT2M, axis=1)
        TRENDS_T2M=np.zeros((2,DATA.shape[1],DATA.shape[2])); TRENDS_T2M[:]=np.nan
        for la in range(DATA.shape[1]):
            for lo in range(DATA.shape[2]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(rgiYY[-30:],DATA[-30:,la,lo])
                TRENDS_T2M[:,la,lo]=[slope,p_value]
        # get T2M data from ERA-Interim
        rgdTimeDD_EI = pd.date_range(datetime.datetime(1979, 1, 1,0), end=datetime.datetime(2017, 12, 31,23), freq='d')
        ncid=Dataset('/glade/scratch/prein/ERA-Interim/T2M_1979-2018.nc', mode='r')
        T2M_EI=np.squeeze(ncid.variables['t2m'][:np.sum(rgdTimeDD_EI.year <=2010),:,:])
        LON_EI=np.squeeze(ncid.variables['longitude'][:])
        LAT_EI=np.squeeze(ncid.variables['latitude'][:])
        ncid.close()
        LON_EI2D=np.asarray(([LON_EI,]*LAT_EI.shape[0]))
        LAT_EI2D=np.asarray(([LAT_EI,]*LON_EI.shape[0])).transpose()
        DATA=np.array([np.mean(T2M_EI[(rgdTimeDD_EI[rgdTimeDD_EI.year <= 2010].year == yy),:,:], axis=0) for yy in range(1979,2010,1)])
        TRENDS_T2M_EI=np.zeros((2,DATA.shape[1],DATA.shape[2])); TRENDS_T2M_EI[:]=np.nan
        for la in range(DATA.shape[1]):
            for lo in range(DATA.shape[2]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(rgiYY[-30:],DATA[-30:,la,lo])
                TRENDS_T2M_EI[:,la,lo]=[slope,p_value]
        ERAi_on_ERA20_T=scipy.interpolate.griddata((LON_EI2D.flatten(),LAT_EI2D.flatten()),TRENDS_T2M_EI[0,:,:].flatten() , (rgrLonGrid2D,rgrLatGrid2D),method='linear')
        ERAi_on_ERA20_P=scipy.interpolate.griddata((LON_EI2D.flatten(),LAT_EI2D.flatten()),TRENDS_T2M_EI[1,:,:].flatten() , (rgrLonGrid2D,rgrLatGrid2D),method='linear')
        # average ERAI and ERA-20C data
        AVERAGE_Trend=np.mean([ERAi_on_ERA20_T,TRENDS_T2M[0,:,:]], axis=0)
        plt.axis('off')
        cs=plt.contourf(E20_lon,E20_lat,AVERAGE_Trend*30, cmap="coolwarm", extend='both',levels=clevsTD, zorder=0,transform=cartopy.crs.PlateCarree())
        # SIG=((ERAi_on_ERA20_P <=0.05) & (TRENDS_T2M[1,:,:] <= 0.05) & (ERAi_on_ERA20_T*TRENDS_T2M[0,:,:] > 0)).astype('int')
        SIG=((TRENDS_T2M[1,:,:] <= 0.05) & (ERAi_on_ERA20_T*TRENDS_T2M[0,:,:] > 0)).astype('int')
        csH=plt.contourf(E20_lon,E20_lat,SIG, hatches=['','//'],levels=[0,0.9,1], zorder=1, alpha=0 ,transform=cartopy.crs.PlateCarree())
        SIG=((ERAi_on_ERA20_P <=0.05) & (ERAi_on_ERA20_T*TRENDS_T2M[0,:,:] > 0)).astype('int')
        csH=plt.contourf(E20_lon,E20_lat,SIG, hatches=['',"\\ "],levels=[0,0.9,1], zorder=1, alpha=0 ,transform=cartopy.crs.PlateCarree())
        
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5, zorder=11)
        ax.add_feature(cartopy.feature.COASTLINE, zorder=11)
        ax.add_feature(cartopy.feature.STATES, zorder=11, alpha=.5)
        plt.axis('off')
        CbarAx = axes([0.285,0.05,0.01,0.23])
        cb = colorbar(cs, cax = CbarAx, orientation='vertical', extend='both', ticks=clevsTD, extendfrac='auto')
        cb.ax.set_title('T2M\n [$^{\circ}$C 10y$^{-1}$]')
        cb.ax.tick_params(labelsize=10)
        lon_bounds=np.copy(E20_lon)
        lon_bounds[E20_lon > 180] -= 360
        ax.set_extent([np.min(lon_bounds), np.max(lon_bounds), np.min(E20_lat), np.max(E20_lat)])
        ax.set_global()
        tt = ax.text(0.03,1.01, rgsLableABC[9]+') T2M trend' , ha='left',va='bottom', \
                     transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=10)
        # WBZH
        ax = plt.subplot(gs1[0,1], projection=cartopy.crs.Robinson())
        biasContDist=40
        iMinMax=8.*biasContDist/2.+biasContDist/2.
        clevsTD=np.arange(-iMinMax,iMinMax+0.0001,biasContDist)
        DATA=np.mean(E20_WCLD, axis=1)
        TRENDS_WBZH=np.zeros((2,DATA.shape[1],DATA.shape[2])); TRENDS_WBZH[:]=np.nan
        for la in range(DATA.shape[1]):
            for lo in range(DATA.shape[2]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(rgiYY[-30:],DATA[-30:,la,lo])
                TRENDS_WBZH[:,la,lo]=[slope,p_value]
        DATA=np.mean(EI_WCLD, axis=1)
        TRENDS_WBZH_EI=np.zeros((2,DATA.shape[1],DATA.shape[2])); TRENDS_WBZH_EI[:]=np.nan
        for la in range(DATA.shape[1]):
            for lo in range(DATA.shape[2]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(rgiYY[-30:],DATA[-30:,la,lo])
                TRENDS_WBZH_EI[:,la,lo]=[slope,p_value]
        ERAi_on_ERA20_T=scipy.interpolate.griddata((rgrLonGrid2D_EI.flatten(),rgrLatGrid2D_EI.flatten()),TRENDS_WBZH_EI[0,:,:].flatten() , (rgrLonGrid2D,rgrLatGrid2D),method='linear')
        ERAi_on_ERA20_P=scipy.interpolate.griddata((rgrLonGrid2D_EI.flatten(),rgrLatGrid2D_EI.flatten()),TRENDS_WBZH_EI[1,:,:].flatten() , (rgrLonGrid2D,rgrLatGrid2D),method='linear')
        AVERAGE_Trend_WBZH=np.nanmean([ERAi_on_ERA20_T,TRENDS_WBZH[0,:,:]], axis=0)
        plt.axis('off')
        cs=plt.contourf(E20_lon,E20_lat,AVERAGE_Trend_WBZH*30, cmap="coolwarm", extend='both',levels=clevsTD, zorder=0,transform=cartopy.crs.PlateCarree())
        # SIG_WBZH=((ERAi_on_ERA20_P <=0.05) & (TRENDS_WBZH[1,:,:] <= 0.05) & (ERAi_on_ERA20_T*TRENDS_WBZH[0,:,:] > 0)).astype('int')
        SIG_WBZH=((TRENDS_WBZH[1,:,:] <= 0.05) & (ERAi_on_ERA20_T*TRENDS_WBZH[0,:,:] > 0)).astype('int')
        csH=plt.contourf(E20_lon,E20_lat,SIG_WBZH, hatches=['','//'],levels=[0,0.9,1.1], zorder=1, alpha=0 ,transform=cartopy.crs.PlateCarree())
        SIG_WBZH=((ERAi_on_ERA20_P <=0.05) & (ERAi_on_ERA20_T*TRENDS_WBZH[0,:,:] > 0)).astype('int')
        csH=plt.contourf(E20_lon,E20_lat,SIG_WBZH, hatches=['','\\'],levels=[0,0.9,1.1], zorder=1, alpha=0 ,transform=cartopy.crs.PlateCarree())
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5, zorder=11)
        ax.add_feature(cartopy.feature.COASTLINE, zorder=11)
        ax.add_feature(cartopy.feature.STATES, zorder=11, alpha=.5)
        plt.axis('off')
        CbarAx = axes([0.61,0.05,0.01,0.23])
        cb = colorbar(cs, cax = CbarAx, orientation='vertical', extend='both', ticks=clevsTD, extendfrac='auto')
        cb.ax.set_title('ML heigth\n [m 10y$^{-1}$]')
        cb.ax.tick_params(labelsize=10)
        tt = ax.text(0.03,1.01, rgsLableABC[10]+') ML height trend' , ha='left',va='bottom', \
                     transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=10)
        # WBZH per degree warming
        ax = plt.subplot(gs1[0,2], projection=cartopy.crs.Robinson())
        biasContDist=100
        iMinMax=8.*biasContDist/2.+biasContDist/2.
        clevsTD=np.arange(-iMinMax,iMinMax+0.0001,biasContDist)
        SCALING=AVERAGE_Trend_WBZH/AVERAGE_Trend
        SIG=((SIG == 1) & (SIG_WBZH == 1)).astype('float')
        plt.axis('off')
        cs=plt.contourf(E20_lon,E20_lat,SCALING[:,:], cmap="coolwarm", extend='both',levels=clevsTD, zorder=0,transform=cartopy.crs.PlateCarree())
        # csH=plt.contourf(E20_lon,E20_lat, SIG, hatches=['','///'],levels=[0,0.9,1], zorder=1, alpha=0 ,transform=cartopy.crs.PlateCarree())
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5, zorder=11)
        ax.add_feature(cartopy.feature.COASTLINE, zorder=11)
        ax.add_feature(cartopy.feature.STATES, zorder=11, alpha=.5)
        plt.axis('off')
        ax.set_global()
        CbarAx = axes([0.935,0.05,0.01,0.23])
        cb = colorbar(cs, cax = CbarAx, orientation='vertical', extend='both', ticks=clevsTD, extendfrac='auto')
        cb.ax.set_title('ML height scaling \n[m $^{\circ}$C$^{-1}$ 10y$^{-1}$]')
        cb.ax.tick_params(labelsize=10)
        tt = ax.text(0.03,1.01, rgsLableABC[11]+') ML height change per\ndegree warming' , ha='left',va='bottom', \
                     transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=10)

        sPlotFile=PlotDir+'TimeSeries-Continents/'
        sPlotName= 'Continental_TimeSeries_'+rgsVars[va]+'-'+rgsSeasons[se]+'_1900-2017.pdf'
        if os.path.isdir(sPlotFile) != 1:
            subprocess.call(["mkdir","-p",sPlotFile])
        print '        Plot map to: '+sPlotFile+sPlotName
        fig.savefig(sPlotFile+sPlotName, facecolor=fig.get_facecolor(), edgecolor='none')

stop()
