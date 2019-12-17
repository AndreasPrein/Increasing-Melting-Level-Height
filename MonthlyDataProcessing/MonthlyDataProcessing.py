#!/usr/bin/env python
'''
    File name: MonthlyDataProcessing.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 05.04.2017
    Date last modified: 05.04.2017

    ##############################################################
    Purpos:

    Reads in the preprocessed data from RO and ranalysis datasets

    Calculate Monthly Averages of the targed Variables

    Do quallity and homogenity testing on annual averages

    Save the data for processing in ploting programs


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
from StatisticFunktions_P3 import fnPhysicalBounds
from StatisticFunktions_P3 import fnHomogenTesting

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
rgsVars=['WBZheight','ThreeCheight','ProbSnow','LRbF','VS0_3','VS0_6','CAPE','CIN','LCL','LFC','CBT']
DerivedVars=['PR','WCLD','SSP','SSP_ML','CBVP','CPMR','LR_CB_ML','PR_days','Snow_days', 'WCLD_G3.8km']
rgsLableABC=list(string.ascii_lowercase)
plt.rcParams.update({'font.size': 12})
SaveDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/Monthly_Data/'
PlotDir='/glade/u/home/prein/papers/Trends_RadSoundings/plots/Trends_Map/'
rgsSeasons=['DJF','MAM','JJA','SON','Annual']
iDaysInSeason=[31+31+28.25,31+30+31,30+31+31,30+31+30,365.25]
MinCov=0.7 # minimum data coverage

rgsData=['Rad-Soundings', 'ERA-Interim','ERA-20C']

dStartDay=datetime.datetime(1900, 1, 1,0)
dStopDay=datetime.datetime(2017, 12, 31,0)
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTimeMM = pd.date_range(dStartDay, end=dStopDay, freq='m')
iYears=np.unique(rgdTimeMM.year)

# index of cloud properties that should be conditioned on precipitation
AllVars=np.append(rgsVars,DerivedVars)
PRconditioning=np.isin(AllVars, ['LCL','CBT','WCLD','CBVP','CPMR'])
iPRconditioning=np.where(PRconditioning == 1)[0]

########################################
########################################
########################################
########################################
#   COMBINE THE RS DATASTES
dStartDayRS=datetime.datetime(1979, 1, 1,0)
dStopDayRS=datetime.datetime(2017, 12, 31,0)
RSTimeDD = pd.date_range(dStartDayRS, end=dStopDayRS, freq='d')
RSTimeMM = pd.date_range(dStartDayRS, end=dStopDayRS, freq='m')
iYearsRS=np.unique(RSTimeDD.year)

SaveFile=SaveDir+'RS-data-montly_'+str(iYears[0])+''+str(iYears[-1])+'.npz_MRupdate'
if os.path.isfile(SaveFile) == False:
    #      READ IN THE U-WYOMING RADIO SOUNDING DATA
    # Read preprocessed data from /gpfs/u/home/prein/papers/Trends_RadSoundings/programs/RadioSoundings/UWY_Preprocessor/HailParametersFromRadioSounding.py 
    UW_Vars=['rgrWBZ','rgrThreeCheight','rgrLRbF','rgrProbSnow','CAPE','CIN','LCL','LFC','CBT','VS0_3','VS0_6']
    UW_Vars_sort=[0,1,3,2,6,7,8,9,10,4,5]
    file=open('/glade/scratch/prein/Papers/Trends_RadSoundings/data/RadSoundHailEnvDat-1979-2017_incl-CAPE.pkl', 'rb')
    DATA = pickle.load(file)
    rgrWBZ_UW=DATA['rgrWBZ']
    UW_RS_ALL=np.zeros((rgrWBZ_UW.shape[0],rgrWBZ_UW.shape[1],len(UW_Vars))); UW_RS_ALL[:]=np.nan
    for va in range(len(UW_Vars)):
        UW_RS_ALL[:,:,UW_Vars_sort[va]]=DATA[UW_Vars[va]]
    
    rgrLon_UW=DATA['rgrLon']
    rgrLat_UW=DATA['rgrLat']
    rgrTime_UW=DATA['rgdTime']
    # rgrWCLD_UW=rgrWBZ_UW-rgrLCL_UW; rgrWCLD_UW[rgrWCLD_UW < 0]=0
    
    #      READ IN THE IGRA DAILY RADIO SOUNDING DATA
    # This data is preprocessed in: /gpfs/u/home/prein/papers/Trends_RadSoundings/programs/RadioSoundings/IGRA_Preprocessor/Indices_from_IGRA-soundings.py
    I_Vars=['rgrWBZ','rgr3CLevel','rgrSnowProb','rgrLRbF','rgrSH03','rgrSH06','rgrCAPE','rgrCIN','rgrLCL','rgrLFC','rgrCBT']
    DATA=np.load('/glade/work/prein/papers/Trends_RadSoundings/data/IGRA/All_IGRA-Data_daily_1979-2017.npz')
    rgrWCLD_I=DATA['rgrWBZ']
    I_RS_ALL=np.zeros((rgrWCLD_I.shape[0],rgrWCLD_I.shape[1],len(I_Vars))); I_RS_ALL[:]=np.nan
    for va in range(len(UW_Vars)):
        I_RS_ALL[:,:,va]=DATA[I_Vars[va]]
    
    rgrLon_I=DATA['rgiStatLon']
    rgrLat_I=DATA['rgiStatLat']
    rgdTimeFin_I=DATA['rgdTimeFin']
    
    # COMBINE THE RS DATASETS
    RS_All=np.array(np.append(UW_RS_ALL,np.moveaxis(I_RS_ALL, 0,1), axis=1))
    RS_Lon=np.array(np.append(rgrLon_UW,rgrLon_I, axis=0))
    RS_Lat=np.array(np.append(rgrLat_UW,rgrLat_I, axis=0))
    
    # PERFORM QUALLITY CONTROL AND REMOVE SUSBITIOUS DATA
    RS_All=fnPhysicalBounds(RS_All, rgsVars)
    
    # # READ THE DAILY PRECIPITAITON AT RS LOCATIONS FROM MSWEP
    # # # load grid from MSWAP
    # sMSWEP_Grid='/glade/scratch/prein/MSWEP_V2.1/data/197901.nc'
    # ncid=Dataset(sMSWEP_Grid, mode='r')
    # rgrLat_MSWEP=np.squeeze(ncid.variables['lat'][:])
    # rgrLon_MSWEP=np.squeeze(ncid.variables['lon'][:]); rgrLon_MSWEP[rgrLon_MSWEP<0]=rgrLon_MSWEP[rgrLon_MSWEP<0]+360
    # ncid.close()
    # rgiLonMin_MSWEP=np.array([np.argmin(np.abs(rgrLonS[ii] - rgrLon_MSWEP)) for ii in range(len(rgrLonS))])
    # rgiLatMin_MSWEP=np.array([np.argmin(np.abs(rgrLatS[ii] - rgrLat_MSWEP)) for ii in range(len(rgrLonS))])
    # YYYY_MSWEP=np.unique(rgdTimeDD_MSWEP.year)
    # rgrMSWEP_PR=np.zeros((len(rgdTimeDD_MSWEP), len(rgrLonS)))
    # for yy in range(len(YYYY_MSWEP)):
    #     print('    Process Year '+str(YYYY_MSWEP[yy]))
    #     for mm in range(12):
    #         sFileName='/glade/scratch/prein/MSWEP_V2.1/data/'+str(YYYY_MSWEP[yy])+str("%02d" % (mm+1))+'.nc'
    #         rgiTimeAct=((rgdTimeDD_MSWEP.year == YYYY_MSWEP[yy]) & (rgdTimeDD_MSWEP.month == (mm+1)))
    #         ncid=Dataset(sFileName, mode='r')
    #         for st in range(len(rgrLonS)):
    #             rgrDATAtmp=ncid.variables['precipitation'][:,rgiLatMin_MSWEP[st],rgiLonMin_MSWEP[st]]
    #             rgrMSWEP_PR[rgiTimeAct,st]=np.sum(np.reshape(rgrDATAtmp,(int(len(rgrDATAtmp)/8),8)), axis=1)
    #         ncid.close()
    # np.savez('RS_MSWEP_PR.npz', rgrMSWEP_PR=rgrMSWEP_PR)
    # rgrMSWEP_PR=np.load('RS_MSWEP_PR.npz')['rgrMSWEP_PR']
    # rgiMSWEP_PR=np.copy(rgrMSWEP_PR)
    # rgiMSWEP_PR[(rgrMSWEP_PR < 1)]=0; rgiMSWEP_PR[(rgrMSWEP_PR >= 1)]=1
    # # rgiMSWEP_PR=np.transpose(rgiMSWEP_PR)
    # np.savez('Sounding_WCLD-PR.npz', rgrWCLD_RS=rgrWCLD_RS[:rgiMSWEP_PR.shape[0],:], rgrMSWEP_PR=rgrMSWEP_PR, rgrLon=rgrLonS, rgrLat=rgrLatS)
    # stop()
    DATA=np.load('/glade/u/home/prein/papers/Trends_RadSoundings/programs/plots/WCLD-Changes/Sounding_WCLD-PR.npz')
    rgrWCLD_RS=DATA['rgrWCLD_RS']
    rgrMSWEP_PR=DATA['rgrMSWEP_PR']
    rgiMSWEP_PR=np.copy(rgrMSWEP_PR)
    rgiMSWEP_PR[(rgrMSWEP_PR < 1)]=0; rgiMSWEP_PR[(rgrMSWEP_PR >= 1)]=1
    rgrLon=DATA['rgrLon']
    rgrLat=DATA['rgrLat']
    rgiDataValues=((rgiMSWEP_PR == 1) & (~np.isnan(rgrWCLD_RS)))
    # add pr data to RS_ALLdStartDay=datetime.datetime(1900, 1, 1,0)
    rgdTimeMSWEP = pd.date_range(datetime.datetime(1979, 1, 1,0), end=datetime.datetime(2016, 12, 31,0), freq='d')
    PR_RS=np.zeros((RS_All.shape[0],RS_All.shape[1],1)); PR_RS[:]=np.nan
    PR_RS[:np.sum(rgdTimeMSWEP.year < 2017),:,:]=rgrMSWEP_PR[:,:,None]
    RS_All=np.array(np.append(RS_All,PR_RS, axis=2))
    
    # CALCULATE WARM CLOUD LAYER DEPTH
    WCLD_RS=RS_All[:,:,rgsVars.index('WBZheight')]-RS_All[:,:,rgsVars.index('LCL')]; WCLD_RS[WCLD_RS < 0]=0
    WCLD_RS[WCLD_RS < 0]
    RS_All=np.array(np.append(RS_All,WCLD_RS[:,:,None], axis=2))
    
    # CALCULATE SEVERE CONVECTION PARAMETER
    SSP=RS_All[:,:,rgsVars.index('CAPE')]*RS_All[:,:,rgsVars.index('VS0_6')]; SSP[SSP < 0]=0
    SSP_Hail=np.copy(SSP); SSP_Hail[:]=0
    SSP_Hail[SSP > 10000]=1
    RS_All=np.array(np.append(RS_All,SSP_Hail[:,:,None], axis=2))

    SSP_FL=((SSP > 10000) & (RS_All[:,:,rgsVars.index('WBZheight')] > 4500))
    RS_All=np.array(np.append(RS_All,SSP_FL[:,:,None], axis=2))

    # CALCULATE SATURATION VAPOR PRESSURE AT AND MIXING RATIO CLOUD BASE
    from thermodynamics_p3 import VaporPressure
    VPCB_RS=VaporPressure(RS_All[:,:,rgsVars.index('CBT')])
    RS_All=np.array(np.append(RS_All,VPCB_RS[:,:,None], axis=2))

    # ESTIMATE SATURATION MIXING RATION USING US STANDARD ATM. | https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere
    PP=101325.-((101325.-22632.1)/(288.15-216.65))*(288.15-(RS_All[:,:,rgsVars.index('CBT')]+273.15))
    from thermodynamics_p3 import MixRatio

    # stop()
    # TESTS OF PR REAKTION TO CLOUD PROPERTIES
    # MR=((0.622*VPCB_RS)/(PP-VPCB_RS))*1000.
    # MR_PR=MR[:rgiMSWEP_PR.shape[0],:]
    # NAN=((MR_PR.flatten() > 0) & ~np.isnan(MR_PR.flatten()))
    # # plt.scatter(MR_PR.flatten()[NAN],rgrMSWEP_PR.flatten()[NAN]); plt.show()


    # CBT=rgrWCLD_RS #RS_All[:,:,rgsVars.index('WBZheight')]
    # MR_PR=CBT[:rgrMSWEP_PR.shape[0],:]
    # MR_PR=MR_PR[:,rgrLat > 0]; rgrMSWEP_PR=rgrMSWEP_PR[:,rgrLat > 0]
    # NAN=((MR_PR.flatten() > 0) & ~np.isnan(MR_PR.flatten()))
    # Temp=MR_PR.flatten()[NAN]
    # PR=rgrMSWEP_PR.flatten()[NAN]
    # # plt.scatter(MR_PR.flatten()[NAN],rgrMSWEP_PR.flatten()[NAN]); plt.show()

    # iSmooth=500.
    # rgrXaxis=np.linspace(0,6000,100)
    # rgrPRrate=np.zeros((len(rgrXaxis),3)); rgrPRrate[:]=np.nan
    # for jj in range(len(rgrXaxis)):
    #     rgiBin=((Temp > rgrXaxis[jj]-iSmooth/2) & (Temp <= rgrXaxis[jj]+iSmooth/2))
    #     if sum(rgiBin) >= 1000:
    #         rgrPRrate[jj,:]=np.nanpercentile(PR[rgiBin], (95,99,99.9)) #,99.99,99.999))
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(rgrXaxis,rgrPRrate[:,:])
    # CC=[30*(1.07)**ii for ii in range(30)]
    # ax.plot(range(30),CC, c='k',ls='--')
    # CC=[50*(1.07)**ii for ii in range(30)]
    # ax.plot(range(30),CC, c='k',ls='--')
    # CC=[80*(1.07)**ii for ii in range(30)]
    # ax.plot(range(30),CC, c='k',ls='--')
    # ax.set_yscale('log')
    # plt.show()


    # stop()

    RS_SMR=((0.622*VPCB_RS)/(PP-VPCB_RS))*1000.
    RS_All=np.array(np.append(RS_All,RS_SMR[:,:,None], axis=2))
    
    # CALCULATE LAPSE RATE BETWEEN CLOUD BASE AND FREEZING LEVEL
    LR_CB2FL_RS=((0-RS_All[:,:,rgsVars.index('CBT')])/(RS_All[:,:,rgsVars.index('LCL')]-RS_All[:,:,rgsVars.index('WBZheight')]))*1000.
    LR_CB2FL_RS[(RS_All[:,:,rgsVars.index('CBT')] < 0) | ((RS_All[:,:,rgsVars.index('LCL')]-RS_All[:,:,rgsVars.index('WBZheight')]) < 100)]=np.nan
    LR_CB2FL_RS[LR_CB2FL_RS < -25]=np.nan
    RS_All=np.array(np.append(RS_All,LR_CB2FL_RS[:,:,None], axis=2))

    # CALCULATE PRECIPITATION DAYS AND SNOW DAYS
    PRday_RS=(PR_RS[:,:,0] > 1); PRday_RS=PRday_RS.astype('float')
    PRday_RS[np.isnan(RS_All[:,:,rgsVars.index('ProbSnow')])]=np.nan
    RS_All=np.array(np.append(RS_All,PRday_RS[:,:,None], axis=2))
    SnowDays_RS=((RS_All[:,:,rgsVars.index('ProbSnow')] >= 0.5) & (PRday_RS == 1)); SnowDays_RS=SnowDays_RS.astype('float')
    SnowDays_RS[np.isnan(RS_All[:,:,rgsVars.index('ProbSnow')])]=np.nan
    RS_All=np.array(np.append(RS_All,SnowDays_RS[:,:,None], axis=2))

    # CALCULATE NUMBER OF DAYS WITH WCLD OF LARGER THAN 3.8 KM
    NrDeepWCs=np.copy(WCLD_RS); NrDeepWCs[:]=0
    NrDeepWCs[(WCLD_RS > 3500)]=1; NrDeepWCs[np.isnan(WCLD_RS) == 1]=np.nan
    RS_All=np.array(np.append(RS_All,NrDeepWCs[:,:,None], axis=2))
    # TEST IF ANNUAL TIME SERIES ARE HOMOGENEOUS
    # calculate annual average
    TEST=np.array([np.nanmean(RS_All[(RSTimeDD.year == iYearsRS[yy]),20,2]) for yy in range(len(iYearsRS))])


    #  =========  SAVE DATA FOR HIGH-RES PROCESSING ===========
    np.savez('/glade/scratch/prein/Papers/Trends_RadSoundings/data/DailyCloudData/RS_DailyCloudData.npz',
             WCLD=rgrWCLD_RS[:rgiMSWEP_PR.shape[0],:],
             MSWEP_PR=rgrMSWEP_PR,
             CBT=RS_All[:,:,rgsVars.index('CBT')][:rgiMSWEP_PR.shape[0],:],
             MRCB=RS_SMR[:rgiMSWEP_PR.shape[0],:],
             LCL=RS_All[:,:,rgsVars.index('LCL')][:rgiMSWEP_PR.shape[0],:],
             CAPE=RS_All[:,:,rgsVars.index('CAPE')][:rgiMSWEP_PR.shape[0],:],
             WBZH=RS_All[:,:,rgsVars.index('WBZheight')][:rgiMSWEP_PR.shape[0],:],
             Lat=rgrLat,
             Lon=rgrLon)

    YYYY=np.zeros((len(iYearsRS),RS_All.shape[1],RS_All.shape[2])); YYYY[:]=np.nan
    for yy in range(len(iYearsRS)):
        DD=RSTimeDD.year == iYearsRS[yy]
        for va in range(RS_All.shape[2]):
            YYYY[yy,:,va]=np.nanmean(RS_All[DD,:,va],axis=0)
            NaNact=np.sum(~np.isnan(RS_All[DD,:,va]), axis=0)/np.sum(DD)
            YYYY[yy,(NaNact< MinCov),va]=np.nan

    BreakFlag=np.zeros((len(iYearsRS),RS_All.shape[1],RS_All.shape[2])); BreakFlag[:]=np.nan
    for va in range(RS_All.shape[2]):
        for st in range(RS_All.shape[1]):
            # Take care of NaNs
            if np.sum(~np.isnan(YYYY[:,st,va])/YYYY.shape[0]) >= MinCov:
                if np.sum(~np.isnan(YYYY[:,st,va])/YYYY.shape[0]) != 1:
                    ss = pd.Series(YYYY[:,st,va])
                    TMP_DATA=ss.interpolate().values
                    ss = pd.Series(TMP_DATA[::-1])
                    TMP_DATA=ss.interpolate().values
                    TMP_DATA=TMP_DATA[::-1]
                    YYYY[:,st,va]=TMP_DATA
                if np.std(YYYY[:,st,va])!=0:
                    Breaks = fnHomogenTesting(YYYY[:,st,va])
                    BreakFlag[:,st,va]=Breaks
    
    # CALCULATE MONTHLY AVERAGES
    Monthly_RO=np.zeros((len(iYearsRS),12,RS_All.shape[1],RS_All.shape[2])); Monthly_RO[:]=np.nan
    for yy in range(len(iYearsRS)):
        for mo in range(12):
            MM=((RSTimeDD.year == iYearsRS[yy]) & ((mo+1) == RSTimeDD.month))
            Monthly_RO[yy,mo,:,:]=np.nanmean(RS_All[MM,:,:],axis=0)
            NaNact=np.sum(~np.isnan(RS_All[MM,:,:]), axis=0)/np.sum(MM)
            try:
                Monthly_RO[yy,mo,(NaNact< MinCov)]=np.nan
            except:
                stop()
            # condition cloud properties on rainy days
            PRcondit=RS_All[MM,:,:][:,:,(PRconditioning==1)]
            PRcondit[(np.squeeze(PR_RS[MM,:]) < 1),:]=np.nan
            # NotNAN=~np.isnan(Monthly_RO[yy,mo,:,(PRconditioning==1)])
            Monthly_RO[yy,mo,:,(PRconditioning==1)]=np.nan
            Monthly_RO[yy,mo,:,(PRconditioning==1)]=np.transpose(np.nanmean(PRcondit[:,:,:],axis=0))

    # OK=(np.nanmax(BreakFlag[:,:,20], axis=0) < 0.5)
    # plt.plot(np.nanmean(Monthly_RO[:,:,OK,20], axis=(1,2))); plt.show()
    # SAVE THE DATA FOR FURTHER PROCESSING
    np.savez(SaveFile,
             RSTimeMM=RSTimeMM,
             RS_Lon=RS_Lon,
             RS_Lat=RS_Lat,
             VARS=rgsVars+DerivedVars,
             Monthly_RO=Monthly_RO,
             BreakFlag=BreakFlag)


# ################################################################################
# ################################################################################
# ################################################################################
# ################################################################################
# #          PROCESS ERA-INTERIM

# dStartDay_EI=datetime.datetime(1979, 1, 1,0)
# dStopDay_EI=datetime.datetime(2017, 12, 31,23)
# # dStopDay=datetime.datetime(2010, 12, 31,23)
# rgdTime6H_EI = pd.date_range(dStartDay_EI, end=dStopDay_EI, freq='3h')
# rgdTimeDD_EI = pd.date_range(dStartDay_EI, end=dStopDay_EI, freq='d')
# rgdTimeMM_EI = pd.date_range(dStartDay_EI, end=dStopDay_EI, freq='m')
# rgiYY_EI=np.unique(rgdTime6H_EI.year)
# rgiERA_hours=np.array([0,6,12,18])
# rgdFullTime_EI=pd.date_range(datetime.datetime(1900, 1, 1,00),
#                         end=datetime.datetime(2017, 12, 31,23), freq='3h')
# EI_Vars=['WBZheight','ThreeCheight','ProbSnow','BflLR','VS03','VS06','CAPE','CIN','LCL','LFC','CBT']
# rgsHours=['00','06','12','18']

# # # first read the coordinates
# sERA_IconstantFields='/glade/work/prein/reanalyses/ERA-Interim/ERA-Interim-FullRes_Invariat.nc'
# # read the ERA-Interim elevation
# ncid=Dataset(sERA_IconstantFields, mode='r')
# rgrLat75_EI=np.squeeze(ncid.variables['g4_lat_0'][:])
# rgrLon75_EI=np.squeeze(ncid.variables['g4_lon_1'][:])
# rgrHeight_EI=(np.squeeze(ncid.variables['Z_GDS4_SFC'][:]))/9.81
# ncid.close()
# rgrLonGrid2D_EI=np.asarray(([rgrLon75_EI,]*rgrLat75_EI.shape[0]))
# rgrLatGrid2D_EI=np.asarray(([rgrLat75_EI,]*rgrLon75_EI.shape[0])).transpose()

# SaveFile=SaveDir+'EI-data-montly_'+str(rgiYY_EI[0])+''+str(rgiYY_EI[-1])+'.npz_PRcond'
# if os.path.isfile(SaveFile) == False:
#     #      READ IN ERA-INTERIM DATA THAT WAS PREVIOUSLY CREATED BY
#     # ~/papers/Trends_RadSoundings/programs/ERA-Interim-Data/ERA-Interim-Data-processing.py

#     rgrMonthlyData_EI=np.zeros((len(rgiYY_EI),12,len(rgrLat75_EI), len(rgrLon75_EI),len(rgsVars)+len(DerivedVars))); rgrMonthlyData_EI[:]=np.nan
#     for yy in range(len(rgiYY_EI)): #range(len(TC_Name)): # loop over hurricanes
#         print(' ')
#         print('    Workin on year '+str(rgiYY_EI[yy]))
#         # check if we already processed this file
#         for mm in range(12): #  loop over time
#             rgiDD_EI=np.sum(((rgdTimeDD_EI.year == rgiYY_EI[yy]) & (rgdTimeDD_EI.month == (mm+1))))
#             rgrTempData_EI=np.zeros((rgiDD_EI,len(rgrLat75_EI), len(rgrLon75_EI),len(rgsVars))); rgrTempData_EI[:]=np.nan
#             YYYY=str(rgiYY_EI[yy])
#             MM=str("%02d" % (mm+1))
#             for dd in range(rgiDD_EI):
#                 DD=str("%02d" % (dd+1))
#                 sDate=YYYY+MM+DD
#                 for va in range(len(EI_Vars)):
#                     # print('        read ERA-I '+EI_Vars[va])
#                     if (EI_Vars[va] == 'VS03') | (EI_Vars[va] == 'VS06'):
#                         rgrSH06_EI=np.zeros((len(rgsHours),len(rgrLat75_EI), len(rgrLon75_EI))); rgrSH06_EI[:]=np.nan
#                         for hh in range(len(rgsHours)):
#                             ncid=Dataset('/glade/scratch/prein/ERA-Interim/ERA-Interim_RDA/6hourly/'+YYYY+'-'+MM+'-'+DD+'_'+rgsHours[hh]+':00:00_ERA-Interim_HailPredictors_RDA.nc', mode='r')
#                             rgrSH06_EI[hh,:,:]=np.squeeze(ncid.variables[EI_Vars[va]][:])
#                             ncid.close()
#                         rgrTempData_EI[dd,:,:,va]=np.mean(rgrSH06_EI, axis=0)
#                     else:
#                         EIDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/ERA-Int/Sounding/'
#                         ncid=Dataset(EIDir+sDate+'_ERA-Int_Hail-Env_RDA_Sounding.nc', mode='r')
#                         if (EI_Vars[va] != 'CAPE') | (EI_Vars[va] != 'CIN'):
#                             rgrTempData_EI[dd,:,:,va]=np.mean(np.squeeze(ncid.variables[EI_Vars[va]][:]), axis=0)
#                         else:
#                             rgrTempData_EI[dd,:,:,va]=np.nanmax(np.squeeze(ncid.variables[EI_Vars[va]][:]), axis=0)
#                         ncid.close()

#             rgrTempData_EI_ADD=np.zeros((rgiDD_EI,len(rgrLat75_EI), len(rgrLon75_EI),len(DerivedVars))); rgrTempData_EI_ADD[:]=np.nan
#             # LOAD PRECIPITATION DATA
#             EI_PR='/glade/scratch/prein/ERA-Interim/PR_1979-2018.nc'
#             # tis file comes from here: ~/ERA-Batch-Download/PR_1979-2018.py
#             EI_PR_DD=pd.date_range(datetime.datetime(1979, 1, 1,0), end=datetime.datetime(2018, 12, 31,23), freq='12h')
#             DD=((EI_PR_DD.year == int(YYYY)) & (EI_PR_DD.month == int(MM)))
#             ncid=Dataset(EI_PR, mode='r')
#             EI_PR_ACT=np.squeeze(ncid.variables['tp'][DD,:,:])
#             rgrTempData_EI_ADD[:,:-1,:,0]=np.mean(np.reshape(EI_PR_ACT, (int(sum(DD)/2),2,EI_PR_ACT.shape[1],EI_PR_ACT.shape[2])),axis=1)
#             ncid.close()
#             # CALCULATE WCL DEPTH
#             EI_WCLD=rgrTempData_EI[:,:,:,rgsVars.index('WBZheight')]-rgrTempData_EI[:,:,:,rgsVars.index('LCL')]
#             EI_WCLD[EI_WCLD < 0]=0
#             rgrTempData_EI_ADD[:,:,:,1]=EI_WCLD
#             # CALCULATE SSP
#             EI_SSP=rgrTempData_EI[:,:,:,rgsVars.index('CAPE')]*rgrTempData_EI[:,:,:,rgsVars.index('VS0_6')]
#             EI_SSP[EI_SSP < 0]=0; EI_SSP_Hail=np.copy(EI_SSP); EI_SSP_Hail[:]=0
#             EI_SSP_Hail[EI_SSP > 10000]=1
#             rgrTempData_EI_ADD[:,:,:,2]=EI_SSP_Hail
#             EI_SSP_ML=((EI_SSP > 10000) & (rgrTempData_EI[:,:,:,rgsVars.index('WBZheight')] > 3800))
#             rgrTempData_EI_ADD[:,:,:,3]=EI_SSP_ML
#             # CALCULATE SATURATION VAPOR PRESSURE AT AND MIXING RATIO CLOUD BASE
#             from thermodynamics_p3 import VaporPressure
#             EI_VPCB=VaporPressure(rgrTempData_EI[:,:,:,rgsVars.index('CBT')]-273.15)
#             rgrTempData_EI_ADD[:,:,:,4]=EI_VPCB
#             # ESTIMATE SATURATION MIXING RATION USING US STANDARD ATM. | https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere
#             PP=101325-((101325-22632.1)/(288.15-216.65))*(288.15-rgrTempData_EI[:,:,:,rgsVars.index('CBT')])
#             from thermodynamics_p3 import MixRatio
#             EI_SMR=MixRatio(rgrTempData_EI_ADD[:,:,:,3],PP)
#             rgrTempData_EI_ADD[:,:,:,5]=EI_SMR    
#             # CALCULATE LAPSE RATE BETWEEN CLOUD BASE AND FREEZING LEVEL
#             LR_CB2FL_EI=((0-rgrTempData_EI[:,:,:,rgsVars.index('CBT')])/(rgrTempData_EI[:,:,:,rgsVars.index('LCL')]-rgrTempData_EI[:,:,:,rgsVars.index('WBZheight')]))*1000.
#             LR_CB2FL_EI[(rgrTempData_EI[:,:,:,rgsVars.index('CBT')] < 0) | ((rgrTempData_EI[:,:,:,rgsVars.index('LCL')]-rgrTempData_EI[:,:,:,rgsVars.index('WBZheight')]) < 100)]=np.nan
#             LR_CB2FL_EI[LR_CB2FL_EI < -25]=np.nan
#             rgrTempData_EI_ADD[:,:,:,6]=LR_CB2FL_EI
#             # CALCULATE PRECIPITATION DAYS AND SNOW DAYS
#             PRday_EI=(rgrTempData_EI_ADD[:,:,:,0]*3600. >= 1)
#             rgrTempData_EI_ADD[:,:,:,7]=PRday_EI
#             SnowDays_EI=((rgrTempData_EI[:,:,:,rgsVars.index('ProbSnow')] >= 0.5) & (PRday_EI == 1))
#             rgrTempData_EI_ADD[:,:,:,8]=SnowDays_EI
#             # CALCULATE FREQUENCY OF WCLD > 3800 M
#             NrDeepWCs=np.copy(EI_WCLD); NrDeepWCs[:]=0
#             NrDeepWCs[(EI_WCLD > 3500)]=1; NrDeepWCs[np.isnan(EI_WCLD) == 1]=np.nan
#             rgrTempData_EI_ADD[:,:,:,9]=NrDeepWCs

#             rgrTempData_EI=np.append(rgrTempData_EI,rgrTempData_EI_ADD,3)

#             # CALCULATE MONTHLY AVERAGES
#             NaNact=np.sum(~np.isnan(rgrTempData_EI[:,:,:,:]), axis=0)/rgrTempData_EI.shape[0]
#             rgrMonthlyData_EI[yy,mm,:,:,:]=np.nanmean(rgrTempData_EI, axis=0)
#             # rgrMonthlyData_EI[yy,mm,(NaNact< MinCov)]=np.nan

#             # CONDITION CLOUD PROPERTIES ON RAINY DAYS
#             PRcondit=rgrTempData_EI[:,:,:,(PRconditioning==1)]

#             PRcondit[(np.squeeze((rgrTempData_EI_ADD[:,:,:,0]*3600.)) < 1),:]=np.nan
#             rgrMonthlyData_EI[yy,mo,:,:,(PRconditioning==1)]=np.nan
#             rgrMonthlyData_EI[yy,mo,:,:,(PRconditioning==1)]=np.moveaxis(np.nanmean(PRcondit[:,:,:],axis=0),2,0)

#     np.savez(SaveDir+'TMP_ERA-I_monthly.npz', rgrMonthlyData_EI=rgrMonthlyData_EI)
#     DATA=np.load(SaveDir+'TMP_ERA-I_monthly.npz')
#     rgrMonthlyData_EI=DATA['rgrMonthlyData_EI']

#     # TEST IF ANNUAL TIME SERIES ARE HOMOGENEOUS
#     YYYY=np.nanmean(rgrMonthlyData_EI, axis=1)
#     BreakFlag=np.zeros((YYYY.shape[0],YYYY.shape[1],YYYY.shape[2],YYYY.shape[3])); BreakFlag[:]=np.nan
#     for va in iPRconditioning: #range(YYYY.shape[3]):
#         print('    working on '+(rgsVars+DerivedVars)[va])
#         for la in range(YYYY.shape[1]):
#             for lo in range(YYYY.shape[2]):
#                 # Take care of NaNs
#                 if np.sum(~np.isnan(YYYY[:,la,lo,va])/YYYY.shape[0]) >= MinCov:
#                     if (np.sum(~np.isnan(YYYY[:,la,lo,va]))/YYYY.shape[0]) != 1:
#                         ss = pd.Series(YYYY[:,la,lo,va])
#                         TMP_DATA=ss.interpolate().values
#                         ss = pd.Series(TMP_DATA[::-1])
#                         TMP_DATA=ss.interpolate().values
#                         TMP_DATA=TMP_DATA[::-1]
#                         YYYY[:,la,lo,va]=TMP_DATA
#                     if np.std(YYYY[:,la,lo,va])!=0:
#                         Breaks = fnHomogenTesting(YYYY[:,la,lo,va])
#                         BreakFlag[:,la,lo,va]=Breaks


#     np.savez(SaveFile,
#              rgdTimeMM_EI=rgdTimeMM_EI,
#              rgrLat75_EI=rgrLat75_EI,
#              rgrLon75_EI=rgrLon75_EI,
#              Variables=(rgsVars+DerivedVars),
#              rgrMonthlyData_EI=rgrMonthlyData_EI,
#              BreakFlag=BreakFlag)

# stop()

################################################################################
################################################################################
################################################################################
################################################################################
#          PROCESS ERA-20C

dStartDay_E20=datetime.datetime(1900, 1, 1,0)
dStopDay_E20=datetime.datetime(2010, 12, 31,23)
# dStopDay=datetime.datetime(2010, 12, 31,23)
rgdTime6H_E20 = pd.date_range(dStartDay_E20, end=dStopDay_E20, freq='3h')
rgdTimeDD_E20 = pd.date_range(dStartDay_E20, end=dStopDay_E20, freq='d')
rgdTimeMM_E20 = pd.date_range(dStartDay_E20, end=dStopDay_E20, freq='m')
rgiYY_E20=np.unique(rgdTime6H_E20.year)
rgiERA_hours=np.array([0,6,12,18])
rgdFullTime=pd.date_range(datetime.datetime(1900, 1, 1,00),
                        end=datetime.datetime(2010, 12, 31,23), freq='3h')
E20_Vars=['WBZheight','ThreeCheight','ProbSnow','BflLR','SH03','SH06','CAPE','CIN','LCL','LFC','CBT']
rgsHours=['00','06','12','18']
sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/ERA-20c/Full/'

# # first read the coordinates
sERAconstantFields='/glade/work/prein/reanalyses/ERA-20c/e20c.oper.invariant.128_129_z.regn80sc.1900010100_2010123121.nc'
# read the ERA-Interim elevation
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat75_E20=np.squeeze(ncid.variables['g4_lat_0'][:])
rgrLon75_E20=np.squeeze(ncid.variables['g4_lon_1'][:])
rgrHeight=(np.squeeze(ncid.variables['Z_GDS4_SFC'][:]))/9.81
ncid.close()
rgrLonGrid2D=np.asarray(([rgrLon75_E20,]*rgrLat75_E20.shape[0]))
rgrLatGrid2D=np.asarray(([rgrLat75_E20,]*rgrLon75_E20.shape[0])).transpose()

SaveFile=SaveDir+'E20-data-montly_'+str(rgiYY_E20[0])+''+str(rgiYY_E20[-1])+'.npz_PRcond'
if os.path.isfile(SaveFile) == False:
    #      READ IN ERA-INTERIM DATA THAT WAS PREVIOUSLY CREATED BY
    # ~/papers/Trends_RadSoundings/programs/ERA-20C-Data/ERA-20C-Data-processing.py

    # rgrMonthlyData_E20=np.zeros((len(rgiYY_E20),12,len(rgrLat75_E20), len(rgrLon75_E20),len(rgsVars)+len(DerivedVars))); rgrMonthlyData_E20[:]=np.nan
    # for yy in range(len(rgiYY_E20)): #range(len(TC_Name)): # loop over hurricanes
    #     print(' ')
    #     print('    Workin on year '+str(rgiYY_E20[yy]))
    #     # check if we already processed this file
    #     for mm in range(12): #  loop over time
    #         tt=0
    #         DD_actmon=np.sum((rgdTime6H_E20.year == rgiYY_E20[yy]) & (rgdTime6H_E20.month == (mm+1)))
    #         rgrTempData=np.zeros((DD_actmon,len(rgrLat75_E20), len(rgrLon75_E20),len(rgsVars))); rgrTempData[:]=np.nan
    #         for dd in range(2):
    #             YYYY=str(rgiYY_E20[yy])
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
    #                 ncid=Dataset(sSaveDataDir+sDate+'_ERA-20c_Hail-Env_RDA.nc', mode='r')
    #                 rgrDataE20=np.squeeze(ncid.variables[E20_Vars[va]][:])
    #                 ncid.close()
    #                 rgrTempData[tt:tt+rgrDataE20.shape[0],:,:,va]=rgrDataE20
    #             tt=int(tt+rgrDataE20.shape[0])
    #         # calculate daily averages
    #         rgrTempData=np.reshape(rgrTempData, (int(rgrTempData.shape[0]/8),8, rgrTempData.shape[1],rgrTempData.shape[2],rgrTempData.shape[3]))
    #         DayMean_E20=np.mean(rgrTempData, axis=1)
    #         DayMean_E20[:,:,:,rgsVars.index('CAPE')]=np.nanmax(rgrTempData[:,:,:,:,rgsVars.index('CAPE')], axis=1)
    #         DayMean_E20[:,:,:,rgsVars.index('CIN')]=np.nanmax(rgrTempData[:,:,:,:,rgsVars.index('CIN')], axis=1)          

    #         rgrTempData_E20_ADD=np.zeros((int(DayMean_E20.shape[0]),len(rgrLat75_E20), len(rgrLon75_E20),len(DerivedVars))); rgrTempData_E20_ADD[:]=np.nan
    #         # LOAD PRECIPITATION DATA
    #         E20_PR='/glade/scratch/prein/ERA-20C/PR_1900-2010.nc'
    #         # tis file comes from here: ~/ERA-Batch-Download/PR_1900-2010_ERA-20C.py
    #         E20_PR_DD=pd.date_range(datetime.datetime(1900, 1, 1,0), end=datetime.datetime(2010, 12, 31,23), freq='1d')
    #         DD=((E20_PR_DD.year == int(YYYY)) & (E20_PR_DD.month == int(MM)))
    #         ncid=Dataset(E20_PR, mode='r')
    #         E20_PR_ACT=np.squeeze(ncid.variables['tp'][DD,:,:])
    #         rgrTempData_E20_ADD[:,:-1,:,0]=E20_PR_ACT
    #         ncid.close()
    #         # CALCULATE WCL DEPTH
    #         E20_WCLD=DayMean_E20[:,:,:,rgsVars.index('WBZheight')]-DayMean_E20[:,:,:,rgsVars.index('LCL')]
    #         E20_WCLD[E20_WCLD < 0]=0
    #         rgrTempData_E20_ADD[:,:,:,1]=E20_WCLD
    #         # CALCULATE SSP
    #         E20_SSP=DayMean_E20[:,:,:,rgsVars.index('CAPE')]*DayMean_E20[:,:,:,rgsVars.index('VS0_6')]
    #         E20_SSP[E20_SSP < 0]=0; E20_SSP_Hail=np.copy(E20_SSP); E20_SSP_Hail[:]=0
    #         E20_SSP_Hail[E20_SSP > 10000]=1
    #         rgrTempData_E20_ADD[:,:,:,2]=E20_SSP_Hail
    #         E20_SSP_ML=((E20_SSP > 10000) & (DayMean_E20[:,:,:,rgsVars.index('WBZheight')] > 4500))
    #         rgrTempData_E20_ADD[:,:,:,3]=E20_SSP_ML
    #         # CALCULATE SATURATION VAPOR PRESSURE AT AND MIXING RATIO CLOUD BASE
    #         from thermodynamics_p3 import VaporPressure
    #         E20_VPCB=VaporPressure(DayMean_E20[:,:,:,rgsVars.index('CBT')]-273.15)
    #         rgrTempData_E20_ADD[:,:,:,4]=E20_VPCB
    #         # ESTIMATE SATURATION MIXING RATION USING US STANDARD ATM. | https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere
    #         PP=101325-((101325-22632.1)/(288.15-216.65))*(288.15-DayMean_E20[:,:,:,rgsVars.index('CBT')])
    #         from thermodynamics_p3 import MixRatio
    #         E20_SMR=MixRatio(rgrTempData_E20_ADD[:,:,:,3],PP)
    #         rgrTempData_E20_ADD[:,:,:,5]=E20_SMR
    #         # CALCULATE LAPSE RATE BETWEEN CLOUD BASE AND FREEZING LEVEL
    #         LR_CB2FL_E20=((0-DayMean_E20[:,:,:,rgsVars.index('CBT')])/(DayMean_E20[:,:,:,rgsVars.index('LCL')]-DayMean_E20[:,:,:,rgsVars.index('WBZheight')]))*1000.
    #         LR_CB2FL_E20[(DayMean_E20[:,:,:,rgsVars.index('CBT')] < 0) | ((DayMean_E20[:,:,:,rgsVars.index('LCL')]-DayMean_E20[:,:,:,rgsVars.index('WBZheight')]) < 100)]=np.nan
    #         LR_CB2FL_E20[LR_CB2FL_E20 < -25]=np.nan
    #         rgrTempData_E20_ADD[:,:,:,6]=LR_CB2FL_E20
    #         # CALCULATE PRECIPITATION DAYS AND SNOW DAYS
    #         PRday_E20=(rgrTempData_E20_ADD[:,:,:,0]*3600. >= 1)
    #         rgrTempData_E20_ADD[:,:,:,7]=PRday_E20
    #         SnowDays_E20=((DayMean_E20[:,:,:,rgsVars.index('ProbSnow')] >= 0.5) & (PRday_E20 == 1))
    #         rgrTempData_E20_ADD[:,:,:,8]=SnowDays_E20
    #         # CALCULATE FREQUENCY OF WCLD > 3800 M
    #         NrDeepWCs=np.copy(E20_WCLD); NrDeepWCs[:]=0
    #         NrDeepWCs[(E20_WCLD > 3800)]=1; NrDeepWCs[np.isnan(E20_WCLD) == 1]=np.nan
    #         rgrTempData_E20_ADD[:,:,:,9]=NrDeepWCs

    #         rgrTempData=np.append(DayMean_E20,rgrTempData_E20_ADD,3)

    #         # CALCULATE MONTHLY AVERAGES
    #         NaNact=np.sum(~np.isnan(rgrTempData[:,:,:,:]), axis=0)/rgrTempData.shape[0]
    #         rgrMonthlyData_E20[yy,mm,:,:,:]=np.nanmean(rgrTempData, axis=0)
    #         # rgrMonthlyData_E20[yy,mm,(NaNact< MinCov)]=np.nan

    #         # CONDITION CLOUD PROPERTIES ON RAINY DAYS
    #         PRcondit=rgrTempData[:,:,:,(PRconditioning==1)]
    #         PRcondit[(np.squeeze((rgrTempData_E20_ADD[:,:,:,0]*3600.)) < 1),:]=np.nan
    #         rgrMonthlyData_E20[yy,mo,:,:,(PRconditioning==1)]=np.nan
    #         rgrMonthlyData_E20[yy,mo,:,:,(PRconditioning==1)]=np.moveaxis(np.nanmean(PRcondit[:,:,:],axis=0),2,0)

    # np.savez(SaveDir+'TMP_ERA-20_monthly.npz', rgrMonthlyData_E20=rgrMonthlyData_E20)

    DATA=np.load(SaveDir+'TMP_ERA-20_monthly.npz')
    rgrMonthlyData_E20=DATA['rgrMonthlyData_E20']

    # TEST IF ANNUAL TIME SERIES ARE HOMOGENEOUS
    YYYY=np.nanmean(rgrMonthlyData_E20[-32:,:], axis=1)
    BreakFlag=np.zeros((YYYY.shape[0],YYYY.shape[1],YYYY.shape[2],YYYY.shape[3])); BreakFlag[:]=np.nan
    for va in iPRconditioning: #range(YYYY.shape[3]):
        print('    working on '+(rgsVars+DerivedVars)[va])
        for la in range(YYYY.shape[1]):
            for lo in range(YYYY.shape[2]):
                # Take care of NaNs
                if np.sum(~np.isnan(YYYY[:,la,lo,va])/YYYY.shape[0]) >= MinCov:
                    if (np.sum(~np.isnan(YYYY[:,la,lo,va]))/YYYY.shape[0]) != 1:
                        ss = pd.Series(YYYY[:,la,lo,va])
                        TMP_DATA=ss.interpolate().values
                        ss = pd.Series(TMP_DATA[::-1])
                        TMP_DATA=ss.interpolate().values
                        TMP_DATA=TMP_DATA[::-1]
                        YYYY[:,la,lo,va]=TMP_DATA
                    if np.std(YYYY[:,la,lo,va])!=0:
                        Breaks = fnHomogenTesting(YYYY[:,la,lo,va])
                        BreakFlag[:,la,lo,va]=Breaks

    np.savez(SaveFile,
             rgdTimeMM_E20=rgdTimeMM_E20,
             rgrLat75_E20=rgrLat75_E20,
             rgrLon75_E20=rgrLon75_E20,
             Variables=(rgsVars+DerivedVars),
             rgrMonthlyData_E20=rgrMonthlyData_E20,
             BreakFlag=BreakFlag)
