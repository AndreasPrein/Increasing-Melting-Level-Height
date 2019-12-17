#!/usr/bin/env python
'''
    File name: PR_scaling_regions.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    Read in the monthly preprocessed data from:
    python3 ~/papers/Trends_RadSoundings/programs/MontataProcessing/MonthlyDataProcessing.py

    We plot cloud properties vs. precipitation propperties for tropics, subtropics, and continents


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
from mpl_toolkits.basemap import Basemap; bm = Basemap()
from scipy.interpolate import interp1d
import scipy
from mpl_toolkits.basemap import Basemap; bm = Basemap()
from scipy import signal

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
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
DataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Monthly_Data/'
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
rgsSeasons=['DJF','MAM','JJA','SON','Annual']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30,365.25]

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



# # =======================

# # # READ DAILY DATA FOR ESTABLISHING RELATIONSHIPS BETWEEN CLOUD VARIABLES AND RAINFALL
# this data is processed in /glade/u/home/prein/papers/Trends_RadSoundings/programs/MonthlyDataProcessing/MonthlyDataProcessing.py
DATA=np.load('/glade/scratch/prein/Papers/Trends_RadSoundings/data/DailyCloudData/RS_DailyCloudData.npz')
RS_WCLD_D=DATA['WCLD']
rgrMSWEP_PR=DATA['MSWEP_PR']
RS_CBT_D=DATA['CBT']
RS_MRCB_D=DATA['MRCB']
RS_LCL_D=DATA['LCL']
RS_CAPE_D=DATA['CAPE']
RS_WBZH_D=DATA['WBZH']
rgrLat=DATA['Lat']
rgrLon=DATA['Lon']

rgiMSWEP_PR=np.copy(rgrMSWEP_PR)
rgiMSWEP_PR[(rgrMSWEP_PR < 1)]=0; rgiMSWEP_PR[(rgrMSWEP_PR >= 1)]=1


########################################
#                      start plotting

biasContDist=10
sLabel='warm cloud layer depth [m per decade]'
# rgrDataVar=RS_WCLD[:,:,:]
Yminmax=[-100,100]

fig = plt.figure(figsize=(12, 18))
rgsSeasons=['annual','DJF','MAM','JJA','SON']
rgsDaysSeas=[365.25,90.25,92,92,91]


# ==============================================================================
# ==============================================================================
# Add precipitation vs. WCLD plot
gs1 = gridspec.GridSpec(4,2)
gs1.update(left=0.06, right=0.98,
           bottom=0.06, top=0.98,
           wspace=0.3, hspace=0.3)

for yy in range(4):
    if yy == 0:
        Label="Tropics"
        REGION=(np.abs(rgrLat) <= 30)
    if yy == 1:
        Label="N-America"
        REGION=((rgrLat > 20) & (rgrLat < 49) &(rgrLon > 230) & (rgrLon < 310))
    if yy == 2:
        Label="Europe"
        REGION=((rgrLat > 35) & (rgrLon >= 0) & (rgrLon < 50))
    if yy == 3:
        Label="Asia"
        REGION=((rgrLat > 30) & (rgrLon >= 50) & (rgrLon < 140))


    rgiDataReal=(rgrMSWEP_PR[:,REGION] > 1) & (~np.isnan(RS_WCLD_D[:,REGION]))
    WCLD=RS_WCLD_D[:,REGION][rgiDataReal]
    PR=rgrMSWEP_PR[:,REGION][rgiDataReal]
    MR=RS_MRCB_D[:,REGION][rgiDataReal]
    
    ax = plt.subplot(gs1[yy,0])
    ax.set(xlim=(0,5.5),ylim=(0.1,300))
    iSmooth=500.
    rgrXaxis=np.linspace(0,6100,50)
    rgrPRrate=np.zeros((len(rgrXaxis),5)); rgrPRrate[:]=np.nan
    rgrMR=np.copy(rgrPRrate)
    for jj in range(len(rgrXaxis)):
        rgiBin=((WCLD > rgrXaxis[jj]-iSmooth/2) & (WCLD <= rgrXaxis[jj]+iSmooth/2))
        if sum(rgiBin) >= 200:
            rgrPRrate[jj,:]=np.nanpercentile(PR[rgiBin], (95,99,99.9,99.99,99.999))
            rgrMR[jj,:]=np.nanpercentile(MR[rgiBin], (95,99,99.9,99.99,99.999))
    #ax.scatter(WCLD/1000., PR, s=0.1, alpha=0.4,c='#b2df8a')
    ax.set_yscale('log')
    
    Low=[25,50,90]
    for ll in range(len(Low)):
        CC=[Low[ll]*(1.17)**ii for ii in range(30)]
        ax.plot(range(30),CC, c='k',ls='--', zorder=-1)
    
    Low=[10,20,35]
    for ll in range(len(Low)):
        CC=np.array([Low[ll]*(1.7)**ii for ii in range(-10,10,1)])
        ax.plot(np.array(range(-10,10,1))+1,CC, c='k',ls='--', zorder=-1)
    # C-C estimate
    Low=[20,39,70]
    for ll in range(len(Low)):
        CC=np.array([Low[ll]*(1.325)**ii for ii in range(-10,10,1)])
        ax.plot(np.array(range(-10,10,1))+1,CC, c='r',ls=':', zorder=-1)
    
    # Label the helper lines
    ax.annotate('17 % km$^{-1}$', xy=(1.3,135), ha='center', va='center', rotation=14, weight='bold')
    ax.annotate('70 % km$^{-1}$', xy=(4.6,275), ha='center', va='center', rotation=40, weight='bold')
    ax.annotate('C-C rate 35 % km$^{-1}$', xy=(1.3,82), ha='center', va='center', rotation=26, weight='bold',color='r')
    
    
    sc=plt.scatter(rgrXaxis/1000.,rgrPRrate[:,0], marker='+', s=150, linewidths=4, c=rgrMR[:,0], cmap=plt.cm.coolwarm, vmin=8, vmax=28, zorder=2, label='P95')
    sc=plt.scatter(rgrXaxis/1000.,rgrPRrate[:,1], marker='x', s=150, linewidths=4, c=rgrMR[:,1], cmap=plt.cm.coolwarm, vmin=8, vmax=28, zorder=2, label='P99')
    sc=plt.scatter(rgrXaxis/1000.,rgrPRrate[:,2], marker='.', s=150, linewidths=4, c=rgrMR[:,2], cmap=plt.cm.coolwarm, vmin=8, vmax=28, zorder=2, label='P99.9')
    
    #sc=plt.scatter(rgrXaxis/1000.,rgrPRrate[:,3], marker='.', s=150, linewidths=4, c=rgrMR[:,3], cmap=plt.cm.coolwarm, vmin=0, vmax=28)
    # sc=plt.scatter(rgrXaxis/1000.,rgrPRrate[:,4], marker='.', s=150, linewidths=4, c=rgrMR[:,4], cmap=plt.cm.coolwarm, vmin=0, vmax=28)
    ax.axvline(x=3.8, color='#525252', linestyle='--', lw=3, alpha=0.4)
    cbar=plt.colorbar(sc)
    cbar.set_label('cloud base mixing ratio [g kg$^{-1}$]', rotation=90)
    
    ax.set_ylabel('precipitation [mm d$^{-1}$]')
    ax.set_xlabel('warm cloud layer depth [km]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc="upper left",\
              ncol=1, prop={'size':12})
    ax.set(xlim=(0,6.5),ylim=(20,350))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([25,50,100,200,300])
    plt.title(rgsLableABC[yy*2]+') '+Label)
    
    
    
    
    
    
    # ==============================================================================
    # ==============================================================================
    # Add precipitation vs. Cloud Base Mixing Ratio
    rgiDataReal=(rgrMSWEP_PR[:,REGION] > 1) & (~np.isnan(RS_WCLD_D[:,REGION]))
    WCLD=RS_WCLD_D[:,REGION][rgiDataReal]
    PR=rgrMSWEP_PR[:,REGION][rgiDataReal]
    MR=RS_MRCB_D[:,REGION][rgiDataReal]
    
    ax = plt.subplot(gs1[yy,1])
    iSmooth=3.
    rgrXaxis=np.linspace(0,35,50)
    rgrPRrate=np.zeros((len(rgrXaxis),5)); rgrPRrate[:]=np.nan
    rgrWCLD=np.copy(rgrPRrate)
    for jj in range(len(rgrXaxis)):
        rgiBin=((MR > rgrXaxis[jj]-iSmooth/2) & (MR <= rgrXaxis[jj]+iSmooth/2))
        if sum(rgiBin) >= 200:
            rgrPRrate[jj,:]=np.nanpercentile(PR[rgiBin], (95,99,99.9,99.99,99.999))
            rgrWCLD[jj,:]=np.nanpercentile(WCLD[rgiBin], (95,99,99.9,99.99,99.999))
    #ax.scatter(WCLD/1000., PR, s=0.1, alpha=0.4,c='#b2df8a')
    ax.set_yscale('log')
    
    Low=[20,35,70]
    for ll in range(len(Low)):
        CC=[Low[ll]*(1.065)**ii for ii in range(35)]
        ax.plot(range(35),CC, c='r',ls=':', zorder=-1)
    
    ax.annotate('C-C rate 6.5 % $^{\circ}$C$^{-1}$', xy=(20,275), ha='center', va='center', rotation=22, weight='bold',color='r')
    
    sc=plt.scatter(rgrXaxis,rgrPRrate[:,0], marker='+', s=150, linewidths=4, c=rgrWCLD[:,0]/1000., cmap=plt.cm.coolwarm, vmin=0, vmax=6, label='P95')
    sc=plt.scatter(rgrXaxis,rgrPRrate[:,1], marker='x', s=150, linewidths=4, c=rgrWCLD[:,1]/1000., cmap=plt.cm.coolwarm, vmin=0, vmax=6, label='P99')
    sc=plt.scatter(rgrXaxis,rgrPRrate[:,2], marker='.', s=150, linewidths=4, c=rgrWCLD[:,2]/1000., cmap=plt.cm.coolwarm, vmin=0, vmax=6, label='P99.9')
    
    #sc=plt.scatter(rgrXaxis/1000.,rgrPRrate[:,3], marker='.', s=150, linewidths=4, c=rgrMR[:,3], cmap=plt.cm.coolwarm, vmin=0, vmax=28)
    # sc=plt.scatter(rgrXaxis/1000.,rgrPRrate[:,4], marker='.', s=150, linewidths=4, c=rgrMR[:,4], cmap=plt.cm.coolwarm, vmin=0, vmax=28)
    # ax.axvline(x=3.8, color='#525252', linestyle='--', lw=3)
    cbar=plt.colorbar(sc)
    cbar.set_label('warm cloud layer depth [km]', rotation=90)
    
    ax.set_ylabel('precipitation [mm d$^{-1}$]')
    ax.set_xlabel('cloud base mixing ratio [g kg$^{-1}$]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc="upper left",\
              ncol=1, prop={'size':12})
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    ax.set(xlim=(0,25),ylim=(20,350))
    ax.set_yticks([25,50,100,200,300])
    plt.title(rgsLableABC[yy*2+1]+') '+ Label)
    #     ii=ii+1
    

sPlotFile=PlotDir
sPlotName= 'PR_Scling_Regions.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
