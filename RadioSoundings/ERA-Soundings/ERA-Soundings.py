#!/usr/bin/env python
'''File name: ERA-Soundings.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 29.03.2017
    Date last modified: 29.03.2017

    ##############################################################
    Purpos:

    1) Reads  in location of  preprocessed hail probabilities  for specific
    rectengular region

    2) gets sounding data from ERA-Interim for the modeled hail days

    3) derives statistics for these soundings (mean, median, quantile
    range, 5-95 percentiles)

    4) calculates indices for the soundings (CAPE, CIN, LCL, FLH, LFC, ...)

    5) save the output

'''


from dateutil import rrule
import datetime
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
from ipdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
# import shapely.geometry
# import descartes
import shapefile
import math
from scipy.stats.kde import gaussian_kde
from math import radians, cos, sin, asin, sqrt
# from shapely.geometry import Polygon, Point
from scipy.interpolate import interp1d
import os.path
import pygrib
from scipy import interpolate
from shutil import copyfile
import time
import wrf
import operator

#Calculation of geopotential and height
def calculategeoh(z, lnsp, ts, qs, levels):
    heighttoreturn=np.full([ts.shape[0],ts.shape[1],ts.shape[2]], -999, np.double)
    geotoreturn=np.copy(heighttoreturn)
    Rd = 287.06
    z_h = 0
    #surface pressure
    sp = np.exp(lnsp)
    # A and B parameters to calculate pressures for model levels,
    #  extracted from an ECMWF ERA-Interim GRIB file and then hardcoded here
    pv =  [
      0.0000000000e+000, 2.0000000000e+001, 3.8425338745e+001, 6.3647796631e+001, 9.5636962891e+001,
      1.3448330688e+002, 1.8058435059e+002, 2.3477905273e+002, 2.9849584961e+002, 3.7397192383e+002,
      4.6461816406e+002, 5.7565112305e+002, 7.1321801758e+002, 8.8366040039e+002, 1.0948347168e+003,
      1.3564746094e+003, 1.6806403809e+003, 2.0822739258e+003, 2.5798886719e+003, 3.1964216309e+003,
      3.9602915039e+003, 4.9067070313e+003, 6.0180195313e+003, 7.3066328125e+003, 8.7650546875e+003,
      1.0376125000e+004, 1.2077445313e+004, 1.3775324219e+004, 1.5379804688e+004, 1.6819472656e+004,
      1.8045183594e+004, 1.9027695313e+004, 1.9755109375e+004, 2.0222203125e+004, 2.0429863281e+004,
      2.0384480469e+004, 2.0097402344e+004, 1.9584328125e+004, 1.8864750000e+004, 1.7961359375e+004,
      1.6899468750e+004, 1.5706449219e+004, 1.4411125000e+004, 1.3043218750e+004, 1.1632757813e+004,
      1.0209500000e+004, 8.8023554688e+003, 7.4388046875e+003, 6.1443164063e+003, 4.9417773438e+003,
      3.8509133301e+003, 2.8876965332e+003, 2.0637797852e+003, 1.3859125977e+003, 8.5536181641e+002,
      4.6733349609e+002, 2.1039389038e+002, 6.5889236450e+001, 7.3677425385e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      7.5823496445e-005, 4.6139489859e-004, 1.8151560798e-003, 5.0811171532e-003, 1.1142909527e-002,
      2.0677875727e-002, 3.4121163189e-002, 5.1690407097e-002, 7.3533833027e-002, 9.9674701691e-002,
      1.3002252579e-001, 1.6438430548e-001, 2.0247590542e-001, 2.4393314123e-001, 2.8832298517e-001,
      3.3515489101e-001, 3.8389211893e-001, 4.3396294117e-001, 4.8477154970e-001, 5.3570991755e-001,
      5.8616840839e-001, 6.3554745913e-001, 6.8326860666e-001, 7.2878581285e-001, 7.7159661055e-001,
      8.1125342846e-001, 8.4737491608e-001, 8.7965691090e-001, 9.0788388252e-001, 9.3194031715e-001,
      9.5182150602e-001, 9.6764522791e-001, 9.7966271639e-001, 9.8827010393e-001, 9.9401944876e-001,
      9.9763011932e-001, 1.0000000000e+000 ]
    levelSize=len(levels) #60
    A = pv[0:levelSize+1]
    B = pv[levelSize+1:]
    Ph_levplusone = A[levelSize] + (B[levelSize]*sp)
    #Get a list of level numbers in reverse order
    reversedlevels=np.full(levels.shape[0], -999, np.int32)
    for iLev in list(reversed(range(levels.shape[0]))):
        reversedlevels[levels.shape[0] - 1 - iLev] = levels[iLev]
    #Integrate up into the atmosphere from lowest level
    for lev in reversedlevels:
        #lev is the level number 1-60, we need a corresponding index into ts and qs
        ilevel=np.where(levels==lev)[0]
        t_level=ts[ilevel]
        q_level=qs[ilevel]
        #compute moist temperature
        t_level = t_level * (1.+0.609133*q_level)
        #compute the pressures (on half-levels)
        Ph_lev = A[lev-1] + (B[lev-1] * sp)
        if lev == 1:
            dlogP = np.log(Ph_levplusone/0.1)
            alpha = np.log(2)
        else:
            dlogP = np.log(Ph_levplusone/Ph_lev)
            dP    = Ph_levplusone-Ph_lev
            alpha = 1. - ((Ph_lev/dP)*dlogP)
        TRd = t_level*Rd
        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the full level
        z_f = z_h + (TRd*alpha)
        #Convert geopotential to height
        heighttoreturn[ilevel] = z_f / 9.80665
        #Geopotential (add in surface geopotential)
        geotoreturn[ilevel] = z_f + z
        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h=z_h+(TRd*dlogP)
        Ph_levplusone = Ph_lev
    return geotoreturn, heighttoreturn

########################################
#                 Settings

# rgrSubRegion=[-13.7,129,-16.7,124.9]; Region='N-Kimperley_AUS'
# rgrSubRegion=[40,360-94,37,360-98.9]; Region='Kansas_US'
rgrSubRegion=[44,360-91,41,360-95]; Region='Minnesota-Iowa'
rgrSubRegion=[35,360-82,32,360-86]; Region='Atlanta'
# rgrSubRegion=[41.5,360-103,38.5,360-107]; Region='Denver'
# rgrSubRegion=[53,12,50,8]; Region='CentralGER'

if Region == 'Denver':
    iSelMon=[6,6] # focus month for investigations
elif Region == 'Minnesota-Iowa':
    iSelMon=[9,9]
elif Region == 'Atlanta':
    # iSelMon=[8,8] # <-- no modeled hail!
    iSelMon=[6,6]
elif Region == 'CentralGER':
    # iSelMon=[8,8] # <-- no modeled hail!
    iSelMon=[5,5]

# dStartDay=datetime.datetime(1979, 1, 1,12)
# dStopDay=datetime.datetime(2016, 12, 31,12)
dStartDay=datetime.datetime(2000, 1, 1,12)
dStopDay=datetime.datetime(2015, 12, 31,23)
rgdTimeAll = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgiMonSel=((rgdTimeAll.month >= iSelMon[0]) & (rgdTimeAll.month <= iSelMon[1]))
rgdTimeFocus=rgdTimeAll[rgiMonSel]
iYears=np.unique(rgdTimeAll.year)

rgsData=['OBS','ERA']  # Options are ERA or OBS

rgrLevels=np.array(range(1,61,1))
rgrLevels3D=array([array([rgrLevels,]*256),]*512).transpose()
# rgrLevels=[600,550,500,450,400,350,300,250]
Variables=['T','Z','P','Q','U','V']


rgsERAdata='/glade/p/rda/data/ds627.0/ei.oper.an.pl/'
rgsERAdataSurf='/glade/p/rda/data/ds627.0/ei.oper.an.sfc/'
rgsOutputFolder='/glade/scratch/prein/ERA-Interim/ERA-Soundings/'

rgsHailPotentialData='/glade/scratch/prein/Papers/HailModel/data/V6_3-25_CAPE-FLH_CAPE_SRH03_VS03/HailProbabilitiesERA-Interim_3-25/'
sDataTMP='/glade/scratch/prein/tmp/ERA-Interim/Hail-Soundings/'

########################################
#            READ IN THE HAIL ESTIMATES
for da in range(len(rgsData)):
    sData=rgsData[da]
    if sData =='ERA':
        # read the ERA-Interim constant fields
        rgsERAdata='/glade/scratch/prein/ERA-Interim/'
        sERAconstantFields='/glade/u/home/prein/ERA-Batch-Download/ei.oper.an.sfc.regn128sc.2014123100.nc'
        ncid=Dataset(sERAconstantFields, mode='r')
        rgrLat=np.squeeze(ncid.variables['g4_lat_0'][:])[::-1]
        rgrLon=np.squeeze(ncid.variables['g4_lon_1'][:])
        rgrHeight=np.squeeze(ncid.variables['Z_GDS4_SFC'][:])/9.81
        ncid.close()
        rgiSize=rgrHeight.shape
        rgrLonGrid2D=np.asarray(([rgrLon,]*rgrLat.shape[0]))
        rgrLatGrid2D=np.asarray(([rgrLat,]*rgrLon.shape[0])).transpose()
    
        rgrERAdataAll=np.zeros((len(rgdTimeAll), rgiSize[0], rgiSize[1])); rgrERAdataAll[:]=np.nan
        tt=0L
        for yy in iYears: # we do this year by year
            print '    Loading hail potential '+str(yy)
            iDays=np.sum(rgdTimeAll.year == yy)
            ncid=Dataset(rgsHailPotentialData+str(yy)+'_HailProbabilities_ERA-Interim_daily.nc', mode='r')
            rgrERAdataAll[tt:tt+iDays,:,:]=np.squeeze(ncid.variables['HailProb'][:])
            ncid.close()
            tt=tt+iDays
        rgrERAdataAll[rgrERAdataAll <= 0.5]=0
        rgrERAdataAll[rgrERAdataAll > 0.5]=1
    if sData == 'OBS':
        # START WITH READING THE NOAA DATA
        sNOOAgrid='/glade/p_old/work/prein/observations/NOAA-Hail/gridded/NOAA-Hail-StormReports_gridded-75km_1979.nc'
        ncid=Dataset(sNOOAgrid, mode='r')
        rgrLat=np.squeeze(ncid.variables['lat'][:])
        rgrLon=np.squeeze(ncid.variables['lon'][:]); rgrLon[rgrLon < 0]=rgrLon[rgrLon < 0]+360
        ncid.close()
        rgrLonGrid2D=rgrLon   #np.asarray(([rgrLon,]*rgrLat.shape[0]))
        rgrLatGrid2D=rgrLat   #np.asarray(([rgrLat,]*rgrLon.shape[0])).transpose()
        rgrLon=rgrLon[0,:]
        rgrLat=rgrLat[:,0]
        # get the hight data
        sERAconstantFields='/glade/u/home/prein/ERA-Batch-Download/ei.oper.an.sfc.regn128sc.2014123100.nc'
        ncid=Dataset(sERAconstantFields, mode='r')
        rgrLatERA=np.squeeze(ncid.variables['g4_lat_0'][:])[::-1]
        rgrLonERA=np.squeeze(ncid.variables['g4_lon_1'][:])
        rgrHeight=np.squeeze(ncid.variables['Z_GDS4_SFC'][:])/9.81
        ncid.close()
        rgrHeight=rgrHeight[np.where(rgrLatERA == rgrLat[0])[0][0]:np.where(rgrLatERA == rgrLat[0])[0][0]+len(rgrLat),np.where(rgrLonERA == rgrLon[0])[0][0]:np.where(rgrLonERA == rgrLon[0])[0][0]+len(rgrLon)]
        # in case we want hail maps to only show the average form 2000-2015
        rgrERAdataAll=np.zeros((len(rgdTimeAll),rgrLatGrid2D.shape[0],rgrLatGrid2D.shape[1])); rgrERAdataAll[:]=np.nan
        for yy in range(len(iYears)):
            sFileFin='/glade/p_old/work/prein/observations/NOAA-Hail/gridded/NOAA-Hail-StormReports_gridded-75km_'+str(iYears[yy])+'.nc'
            rgiDDact=np.where(rgdTimeAll.year == iYears[yy])[0]
            ncid=Dataset(sFileFin, mode='r')
            rgrERAdataAll[rgiDDact,:,:]=np.squeeze(ncid.variables['Hail'][:])
            ncid.close()
    
    # CUT OUT THE REGION OF INTEREST
    from HelperFunctions import fnSubregion
    rgrDomainOffsets = fnSubregion(rgrLonGrid2D,rgrLatGrid2D,rgrSubRegion)
    rgrERAdataAll=rgrERAdataAll[:,rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]]
    # get the selected months
    rgrERAdataAll=rgrERAdataAll[rgiMonSel,:,:]
    print np.sum(rgrERAdataAll, axis=(1,2))
    
    
    ########################################
    #                Start reading ERA data
    
    #  define time in ERA
    rgdTimeERA=pd.date_range(datetime.datetime(1900, 1, 1,0), end=datetime.datetime(2020, 12, 31,23), freq='6h')
    
    # read the ERA-Interim elevation
    sERAconstantFields='/glade/u/home/prein/ERA-Batch-Download/ei.oper.an.sfc.regn128sc.2014123100.nc'
    ncid=Dataset(sERAconstantFields, mode='r')
    rgrLat=np.squeeze(ncid.variables['g4_lat_0'][:])[::-1]
    rgrLon=np.squeeze(ncid.variables['g4_lon_1'][:]) #; rgrLon[(rgrLon>180)]=rgrLon[(rgrLon>180)]-360
    rgrHeight=(np.squeeze(ncid.variables['Z_GDS4_SFC'][:])/9.81)
    ncid.close()
    rgiSize=rgrHeight.shape
    rgrLonGrid2D=np.asarray(([rgrLon,]*rgrLat.shape[0]))
    rgrLatGrid2D=np.asarray(([rgrLat,]*rgrLon.shape[0])).transpose()
    rgrDomainOffsets = fnSubregion(rgrLonGrid2D,rgrLatGrid2D[::-1,:],rgrSubRegion)
    if rgrDomainOffsets[1]-rgrDomainOffsets[3] > rgrERAdataAll.shape[2]:
        rgrDomainOffsets[1]=rgrDomainOffsets[3]+rgrERAdataAll.shape[2]
    
    ########################################
    #                            Start reading ERA data
    
    #  define time in ERA
    rgdTimeERA=pd.date_range(datetime.datetime(1900, 1, 1,0), end=datetime.datetime(2020, 12, 31,23), freq='6h')
    
    # read in ERA-Interim 0.75
    ncid=Dataset('/glade/scratch/prein/ERA-Interim/ERA-Interim_invariat-fields.nc', mode='r')
    rgrLat75=np.squeeze(ncid.variables['latitude'][:])
    rgrLon75=np.squeeze(ncid.variables['longitude'][:])
    ncid.close()
    rgrLonGrid2D75=np.asarray(([rgrLon75,]*rgrLat75.shape[0]))
    rgrLatGrid2D75=np.asarray(([rgrLat75,]*rgrLon75.shape[0])).transpose()
    rgiSize75=rgrLatGrid2D75.shape
    
    SubDailyDataAll=np.zeros((len(rgrLevels),100000,len(Variables))); SubDailyDataAll[:]=np.nan
    SubDailyParcelAll=np.zeros((100000,4)); SubDailyParcelAll[:]=np.nan
    kk=0
    for yy in iYears: # loop over the months
        sFile=rgsOutputFolder+'Parcel06/fin_Parcel_ERA-Interim_12-0_'+str(yy)+'.nc'
        if os.path.isfile(sFile) == 0:
            # loop over months
            iDaysYear=np.sum((yy == rgdTimeAll.year))
            rgr06WindShear=np.zeros((iDaysYear*4, rgiSize75[0],rgiSize75[1])); rgr06WindShear[:]=np.nan
            rgiTimeNetCDF=np.array(np.where((yy == rgdTimeERA.year))[0])*6
            iTT=0
            for mo in range(iSelMon[0],iSelMon[1]+1):
                sFolderYYMO=str(yy)+str("%02d" % (mo))
                # loop over days
                iDays=np.sum((yy == rgdTimeAll.year) & ((mo) == rgdTimeAll.month))
                for dd in range(iDays):
                    # see if there was hail estimated on this day
                    try:
                        iDay=np.where((rgdTimeFocus.year == yy) & (rgdTimeFocus.month == mo) & (rgdTimeFocus.day == dd+1))[0][0]
                    except:
                        stop()
                    if np.sum(rgrERAdataAll[iDay,:,:]) == 0:
                        # no hail on this day in the region
                        continue
                    # finaly loop over the 6 hour time steps
                    SubDailyData=np.zeros((4,len(rgrLevels),rgrERAdataAll.shape[1],rgrERAdataAll.shape[2],len(Variables))); SubDailyData[:]=np.nan
                    SubDailyParcel=np.zeros((4,rgrERAdataAll.shape[1],rgrERAdataAll.shape[2],4)); SubDailyParcel[:]=np.nan
                    ii =0
                    for hh in ['00','06','12','18']:
                        sFileName='/glade/p_old/rda/data_old/ds627.0/ei.oper.an.ml/'+sFolderYYMO+'/ei.oper.an.ml.regn128sc.'+str(yy)+str("%02d" % (mo))+str("%02d" % (dd+1))+hh
                        print '    Reading: '+sFileName
                        iFileGRIB=sDataTMP+sFileName.split('/')[-1]+'.grib'
                        iFileNC=sDataTMP+sFileName.split('/')[-1]+'.nc'
                        # convert the data to NetCDF format
                        copyfile(sFileName, iFileGRIB)
                        subprocess.Popen("ncl_convert2nc "+iFileGRIB+' -o '+sDataTMP+' -L', shell=True)
                        # read the data from the NetCDF
                        time.sleep(20)
                        print '    Load '+iFileNC
                        ncid=Dataset(iFileNC, mode='r')
                        Tact=np.squeeze(ncid.variables['T_GDS4_HYBL'][:,rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]])
                        Qact=np.squeeze(ncid.variables['Q_GDS4_HYBL'][:,rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]])
                        Zact=np.squeeze(ncid.variables['Z_GDS4_HYBL'][rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]])
                        LnPSFact=np.squeeze(ncid.variables['LNSP_GDS4_HYBL'][rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]])
                        ncid.close()
                        os.system("rm "+iFileGRIB); os.system("rm "+iFileNC)
    
                        # read in U and V
                        sFileName='/glade/p_old/rda/data_old/ds627.0/ei.oper.an.ml/'+sFolderYYMO+'/ei.oper.an.ml.regn128uv.'+str(yy)+str("%02d" % (mo))+str("%02d" % (dd+1))+hh
                        print '    Reading: '+sFileName
                        iFileGRIB=sDataTMP+sFileName.split('/')[-1]+'.grib'
                        iFileNC=sDataTMP+sFileName.split('/')[-1]+'.nc'
                        # convert the data to NetCDF format
                        copyfile(sFileName, iFileGRIB)
                        subprocess.Popen("ncl_convert2nc "+iFileGRIB+' -o '+sDataTMP+' -L', shell=True)
                        # read the data from the NetCDF
                        time.sleep(20)
                        print '    Load '+iFileNC
                        ncid=Dataset(iFileNC, mode='r')
                        Uact=np.squeeze(ncid.variables['U_GDS4_HYBL'][:,rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]])
                        Vact=np.squeeze(ncid.variables['V_GDS4_HYBL'][:,rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]])
                        ncid.close()
                        os.system("rm "+iFileGRIB); os.system("rm "+iFileNC)
    
    
                        # calculate height of model levels
                        try:
                            geotoreturn, heighttoreturn = calculategeoh(Zact, LnPSFact, Tact, Qact, rgrLevels)
                        except:
                            stop()
                        # calculate the pressure at model levels
                        # coefficients from: http://www.ecmwf.int/en/forecasts/documentation-and-support/60-model-levels
                        ak=np.array([0,20,38.425343,63.647804,95.636963,134.483307,180.584351,234.779053,298.495789,373.971924,464.618134,575.651001,713.218079,883.660522,1094.834717,1356.474609,1680.640259,2082.273926,2579.888672,3196.421631,3960.291504,4906.708496,6018.019531,7306.631348,8765.053711,10376.126953,12077.446289,13775.325195,15379.805664,16819.474609,18045.183594,19027.695313,19755.109375,20222.205078,20429.863281,20384.480469,20097.402344,19584.330078,18864.75,17961.357422,16899.46875,15706.447266,14411.124023,13043.21875,11632.758789,10209.500977,8802.356445,7438.803223,6144.314941,4941.77832,3850.91333,2887.696533,2063.779785,1385.912598,855.361755,467.333588,210.39389,65.889244,7.367743,0,0])
                        bk=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000076,0.000461,0.001815,0.005081,0.011143,0.020678,0.034121,0.05169,0.073534,0.099675,0.130023,0.164384,0.202476,0.243933,0.288323,0.335155,0.383892,0.433963,0.484772,0.53571,0.586168,0.635547,0.683269,0.728786,0.771597,0.811253,0.847375,0.879657,0.907884,0.93194,0.951822,0.967645,0.979663,0.98827,0.994019,0.99763,1])
                        # equation form: https://rda.ucar.edu/datasets/ds627.2/docs/Eta_coordinate/
                        rgrPonML=np.array([ak[lv]+bk[lv]*np.exp(LnPSFact) for lv in range(len(rgrLevels))])
                        # convert to m above surface
                        Tact=Tact-273.15
                        # calculate dewpoint temperature
                        from thermodynamics import ThetaE, TempToDewpTemp
                        rgrDT=TempToDewpTemp(Qact, rgrPonML)
    
                        SubDailyData[ii,:,:,:,Variables.index('T')]=Tact
                        SubDailyData[ii,:,:,:,Variables.index('Z')]=heighttoreturn
                        SubDailyData[ii,:,:,:,Variables.index('P')]=rgrPonML
                        SubDailyData[ii,:,:,:,Variables.index('Q')]=Qact
                        SubDailyData[ii,:,:,:,Variables.index('U')]=Uact
                        SubDailyData[ii,:,:,:,Variables.index('V')]=Vact
    
    
                        # xx=1
                        # yy=2
                        # rgrHeightA=heighttoreturn[10:]
                        # rgrPressure=rgrPonML[10:]
                        # rgrTempAjust=Tact[10:]
                        # from thermodynamics import TempToDewpTemp
                        # rgrTDadjust=(TempToDewpTemp(Qact[10:],rgrPressure)-273.15)
                        # wind_abs=((Uact[10:]**2+Vact[10:]**2)**0.5)
                        # sknt=wind_abs*1.94384
                        # drct=(np.arctan2(Uact[10:]/wind_abs, Vact[10:]/wind_abs))* 180/pi+180
                        # mydata=dict(zip(('hght','pres','temp','dwpt','sknt','drct'),(rgrHeightA[::-1,xx,yy],(rgrPressure/100.)[::-1,xx,yy],rgrTempAjust[::-1,xx,yy],rgrTDadjust[::-1,xx,yy],sknt[::-1,xx,yy], drct[::-1,xx,yy])))
                        # import SkewT
                        # S=SkewT.Sounding(soundingdata=mydata)
                        # parcel=S.most_unstable_parcel(depth=300)
                        # # parcel=S.mixed_layer_parcel(depth=50)
                        # # P_lcl,P_lfc,P_el,CAPE,CIN
                        # rgrSoundingIndices=S.get_cape(*parcel[0:3],totalcape=True)
                        # S.make_skewt_axes(tmin=-40.,tmax=30.,pmin=100.,pmax=1050.)
                        # S.add_profile(color='r',bloc=0)
                        # S.lift_parcel(*parcel)
                        # plt.show()
                        # # S=SkewT.Sounding("ExampleSounding.txt")
                        # stop()
    
    
    
    
                        # calculate sounding indicies
                        CAPE,CIN,LCL,LFC=wrf.cape_2d(rgrPonML/100., (Tact+273.15), Qact, heighttoreturn, (rgrHeight[rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]]), np.exp(LnPSFact)/100., ter_follow=False, missing=-99999., meta=False)
    
                        rgrParcel=np.zeros((rgrDT.shape[1],rgrDT.shape[2],7)); rgrParcel[:]=np.nan
                        CAPE,CIN=wrf.cape_3d(rgrPonML/100., (Tact+273.15), Qact, heighttoreturn, (rgrHeight[rgrDomainOffsets[2]:rgrDomainOffsets[0],rgrDomainOffsets[3]:rgrDomainOffsets[1]]/9.81), np.exp(LnPSFact)/100., ter_follow=False, missing=-99999., meta=False)
                        CAPE=np.max(CAPE[:,:,:], axis=0); CIN=np.max(CIN[:,:,:], axis=0)
                        SubDailyParcel[ii,:,:,0]=CAPE
                        SubDailyParcel[ii,:,:,1]=CIN
                        SubDailyParcel[ii,:,:,2]=LCL
                        SubDailyParcel[ii,:,:,3]=LFC
                        ii=ii+1
    
                        # if np.max(CAPE) > 700:
                        #     print '    HIGH CAPE SOUNDING: '+str(np.max(CAPE))
                        #     stop()
    
                    # # select locations with high hail risk
                    # HailAct=np.where(rgrERAdataAll[iDay,:,:] == 1)
                    # SubDailyData=SubDailyData[:,:,HailAct[0],HailAct[1],:]
                    # SubDailyParcel=SubDailyParcel[:,HailAct[0],HailAct[1],:]
                    # select locations with high CAPE
                    CAPEAct=SubDailyParcel[:,:,:,0]
                    CAPEAct[np.array(CAPEAct) >= 1.00000002e+19 ]=np.nan
                    HailAct=np.where(np.nanmax(np.array(CAPEAct), axis=0) >= 600)
                    if len(HailAct[0]) == 0:
                        continue
                    else:
                        SubDailyData=SubDailyData[:,:,HailAct[0],HailAct[1],:]
                        SubDailyParcel=SubDailyParcel[:,HailAct[0],HailAct[1],:]
        
                        # select the time slice with highest CAPE
                        MaxCAPEtime=np.array([max(enumerate(SubDailyParcel[:,jj,0]), key=operator.itemgetter(1))[0] for jj in range(SubDailyParcel.shape[1])])
        
                        # store data to common array
                        for oo in range(len(HailAct[0])):
                            try:
                                SubDailyDataAll[:,kk,:]=SubDailyData[MaxCAPEtime[oo],:,oo,:]
                                SubDailyParcelAll[kk,:]=SubDailyParcel[MaxCAPEtime[oo],oo,:]
                                # print '    CAPE: '+str(SubDailyParcelAll[kk,0])
                                kk=kk+1
                            except:
                                stop()
        
                            # test=SubDailyData[MaxCAPEtime[oo],:,oo,:]
                            # rgrHeight=test[:,Variables.index('Z')][10:]
                            # rgrPressure=test[:,Variables.index('P')][10:]
                            # rgrTempAjust=test[:,Variables.index('T')][10:]
                            # rgrU=test[:,Variables.index('U')][10:]
                            # rgrV=test[:,Variables.index('V')][10:]
                            # from thermodynamics import TempToDewpTemp
                            # rgrTDadjust=(TempToDewpTemp(test[:,Variables.index('Q')][10:],rgrPressure)-273.15)
                            # wind_abs=((rgrU**2+rgrV**2)**0.5)
                            # sknt=wind_abs*1.94384
                            # drct=(np.arctan2(rgrU/wind_abs, rgrV/wind_abs))* 180/pi+180
                            # mydata=dict(zip(('hght','pres','temp','dwpt','sknt','drct'),(rgrHeight[::-1],(rgrPressure/100.)[::-1],rgrTempAjust[::-1],rgrTDadjust[::-1],sknt[::-1], drct[::-1])))
                            # import SkewT
                            # S=SkewT.Sounding(soundingdata=mydata)
                            # # parcel=S.most_unstable_parcel(depth=200)
                            # parcel=S.mixed_layer_parcel(depth=50)
                            # # P_lcl,P_lfc,P_el,CAPE,CIN
                            # rgrSoundingIndices=S.get_cape(*parcel[0:3],totalcape=True)
                            # S.make_skewt_axes(tmin=-40.,tmax=30.,pmin=100.,pmax=1050.)
                            # S.add_profile(color='r',bloc=0)
                            # S.lift_parcel(*parcel)
                            # plt.show()
                            # # S=SkewT.Sounding("ExampleSounding.txt")
    
    # remove slices with NaN
    rgiFin=~np.isnan(SubDailyParcelAll[:,0])
    SubDailyDataAll=SubDailyDataAll[:,rgiFin,:]; SubDailyParcelAll=SubDailyParcelAll[rgiFin,:]
    # save the data in python format
    SaveFile=sDataTMP+'/'+Region+'-HailSoundings_'+str(rgdTimeAll[0].year)+'-'+str(rgdTimeAll[-1].year)+'_'+sData+'.np'
    np.savez(SaveFile, SubDailyDataAll=SubDailyDataAll, SubDailyParcelAll=SubDailyParcelAll)

stop()
