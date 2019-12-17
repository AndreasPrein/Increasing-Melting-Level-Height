#!/usr/bin/env python
'''
    File name: BreakPointExample.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    Read in the preprocessed data from:
    ~/projects/Hail/programs/RadioSoundings/Preprocessor/HailParametersFromRadioSounding.py

    Load the preprocessed RS data - find a representative record with a breaking point
    and show the break point detection on a time-variable diagram


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

########################################
#                            Settings
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
DataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Monthly_Data/'
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/BreakPointExcample/'
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

VAR='WBZheight'
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


########################################
#      EXTRACT THE NECESSARY VARIABLES
RS_WCLD=RS_data[:,:,:,VARIABLES.index(VAR)]
WCLD_Annual=np.nanmean(RS_WCLD, axis=1)

fig = plt.figure(figsize=(7, 11))
gs1 = gridspec.GridSpec(3,1)
gs1.update(left=0.15, right=0.87,bottom=0.08, top=0.95, wspace=0.2, hspace=0.45)

STall=[16,311,211]
STnameAll=['03953 Valentia Observatory, IR','72572 SLC Salt Lake City, US','57972 Chenzhou, CH']
ABC=['a','b','c']

for st in range(len(STall)):
    ST=STall[st]
    STname=STnameAll[st]
    YYYY_DT=signal.detrend(WCLD_Annual[:,ST])+np.mean(WCLD_Annual[:,ST])
    ax = plt.subplot(gs1[st,0])
    ax2 = ax.twinx()
    ax2.plot(range(1979,2018), YYYY_DT/1000., marker='o', alpha=0.8, c='k', zorder=1)
    plt.title(ABC[st]+') '+STname+' ['+str(rgrLon[ST])+','+str(rgrLat[ST])+']')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.set_xlabel('Years []')
    ax2.set_ylabel('ML height [km]')
    # ax2.set(ylim=(3.15,3.7))

    # Plot preak point likelyhood
    # ax2 = ax.twinx()
    BARS=RS_Breaks[:,ST,VARIABLES.index(VAR)]*100
    ax.bar(range(1979,2018),BARS, color='#a6cee3' ,width=1, zorder=0)
    BARS[BARS<=50]=0
    ax.bar(range(1979,2018),BARS, color='#fb9a99' ,width=1, zorder=0)
    ax.set_ylabel('Break Point Probability [%]', color='#1f78b4')
    ax.set(ylim=(0,100))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelcolor='#1f78b4')

sPlotFile=PlotDir
sPlotName= 'Breakpoint-Detection-Excample.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
