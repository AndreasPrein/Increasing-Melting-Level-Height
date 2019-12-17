#!/usr/bin/env python
'''
    File name: ERA-Interim-Data-processing.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 23.08.2017
    Date last modified: 23.08.2017

    ##############################################################
    Purpos:

    1) Reads in 6 hourly ERA-Interim data from RDA

    2) Calculates variables that are related to atmospherioc theromodynamics

    3) Saves the varaibles to yearly files in NCDF format

'''

from dateutil import rrule
import datetime
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
# import descartes
import shapefile
import math
from scipy.stats.kde import gaussian_kde
from math import radians, cos, sin, asin, sqrt
from scipy import spatial
import scipy.ndimage
import matplotlib.path as mplPath
from scipy.interpolate import interp1d
import time
from math import atan2, degrees, pi
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import SkewT
import csv
import pygrib
from scipy import interpolate
from shutil import copyfile
from HelperFunctions import fnSolidPrecip
import wrf

#Calculation of geopotential and height
def calculategeoh(z, lnsp, ts, qs, levels):
    heighttoreturn=np.full([ts.shape[0],ts.shape[1],ts.shape[2],ts.shape[3]], -999, np.double)
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
        ilevel=np.where(levels==lev)[0][0]
        t_level=np.squeeze(ts[:,ilevel,:,:])
        q_level=np.squeeze(qs[:,ilevel,:,:])
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
        heighttoreturn[:,ilevel] = z_f / 9.80665
        #Geopotential (add in surface geopotential)
        try:
            geotoreturn[:,ilevel] = z_f + z[:,0,:,:]
        except:
            stop()
        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h=z_h+(TRd*dlogP)
        Ph_levplusone = Ph_lev
    return geotoreturn, heighttoreturn

# coefficients from: http://www.ecmwf.int/en/forecasts/documentation-and-support/60-model-levels
ak=np.array([0,20,38.425343,63.647804,95.636963,134.483307,180.584351,234.779053,298.495789,373.971924,464.618134,575.651001,713.218079,883.660522,1094.834717,1356.474609,1680.640259,2082.273926,2579.888672,3196.421631,3960.291504,4906.708496,6018.019531,7306.631348,8765.053711,10376.126953,12077.446289,13775.325195,15379.805664,16819.474609,18045.183594,19027.695313,19755.109375,20222.205078,20429.863281,20384.480469,20097.402344,19584.330078,18864.75,17961.357422,16899.46875,15706.447266,14411.124023,13043.21875,11632.758789,10209.500977,8802.356445,7438.803223,6144.314941,4941.77832,3850.91333,2887.696533,2063.779785,1385.912598,855.361755,467.333588,210.39389,65.889244,7.367743,0,0])
bk=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000076,0.000461,0.001815,0.005081,0.011143,0.020678,0.034121,0.05169,0.073534,0.099675,0.130023,0.164384,0.202476,0.243933,0.288323,0.335155,0.383892,0.433963,0.484772,0.53571,0.586168,0.635547,0.683269,0.728786,0.771597,0.811253,0.847375,0.879657,0.907884,0.93194,0.951822,0.967645,0.979663,0.98827,0.994019,0.99763,1])

########################################
#                            USER INPUT SECTION

sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/ERA-Int/Sounding/'

# dStartDay=datetime.datetime(int(>>YYYY<<), 1, 1,0)
# dStopDay=datetime.datetime(int(>>YYYY<<), 12, 31,23)
YYYY=int(>>YYYY<<)
dStartDay=datetime.datetime(int(YYYY), 1, 1,0)
dStopDay=datetime.datetime(int(YYYY), 12, 31,23)

rgdTime6H = pd.date_range(dStartDay, end=dStopDay, freq='6h')
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgiYY=np.unique(rgdTime6H.year)
rgiERA_hours=np.array([0,6,12,18])

rgdFullTime=pd.date_range(datetime.datetime(1979, 1, 1,00),
                          end=datetime.datetime(2017, 12, 31,23), freq='6h')
rgiTimeFull=np.array(range(len(rgdFullTime)))*6

# define hight levels for vertical interpolation
rgrLevelsZZ=[1000,3000,6000,12000] # height levels in m

# ERA levels
rgrLevelsERA=np.array(range(1,61,1))

rgr3Dvariables=['00_00_t','01_00_q','03_25_lnsp','Z', 'P']
rgr3DERAvars=['T_GDS4_HYBL','Q_GDS4_HYBL','LNSP_GDS4_HYBL','Z_GDS4_HYBL']

rgr3Dstaggerd=['U','V']
rgr3DERAStag=['U_GDS4_HYBL','V_GDS4_HYBL']

rgr2Dvariables=['PS']
rgr2DERAvars=['LNSP_GDS4_HYBL']

grsHH=['00','06','12','18']
# ________________________________________________________________________
# ________________________________________________________________________
# Define konstance
cpa=1006.      # (J/kg C)  ||   heat capacity of air at constant pressure
Rd=287.058  # J kg-1 K-1 ||  gas constant of dry ari
Rv=461.5      # J/(kg K) ||  gas constant of water vapor
Lv= 2501000.  #  (J/kg) || latent heat of vaporization
cpw = 1840.  # (J/kg K) ||   heat capacity of water vapor at constant pressure
cw = 4190.  #   (J/kg K) || specific heat water
g = 9.81 # (m/s^2) || gravitatoinal acceleration

# ________________________________________________________________________
# ________________________________________________________________________
sERAconstantFields='/glade/work/prein/reanalyses/ERA-Interim/ERA-Interim-FullRes_Invariat.nc'
# read the ERA-Interim elevation
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat75=np.squeeze(ncid.variables['g4_lat_0'][:])
rgrLon75=np.squeeze(ncid.variables['g4_lon_1'][:])
rgrHeight=(np.squeeze(ncid.variables['Z_GDS4_SFC'][:]))/9.81
ncid.close()
rgiSize=rgrHeight.shape
rgrLon=np.asarray(([rgrLon75,]*rgrLat75.shape[0]))
rgrLat=np.asarray(([rgrLat75,]*rgrLon75.shape[0])).transpose()


# ________________________________________________________________________
# ________________________________________________________________________
# read in the 3D ERA-Int fields, do the statistics, and save the data
dfTCstatsAll={}
for yy in range(len(rgiYY)): #range(len(TC_Name)): # loop over hurricanes
    print ' '
    print '    Workin on year '+str(rgiYY[yy])
    s3Dfolder='/gpfs/fs1/collections/rda/data/ds627.0/ei.oper.an.ml/'    # 2000/wrf3d_d01_CTRL_U_20001001.nc
    s3Dextension='ei.oper.an.ml.regn128'
    s2Dfolder='/gpfs/fs1/collections/rda/data/ds627.0/ei.oper.fc.sfc/'    # 2000/wrf2d_d01_CTRL_LH_200101-200103.nc
    s2Dextension='ei.oper.fc.sfc.regn128'

    # check if we already processed this file
    for mm in range(12): #  loop over time
        MM=mm+1
        iDaysMM=np.sum((rgdTimeDD.year == YYYY) & (rgdTimeDD.month == MM))
        for dd in range(iDaysMM):
            DD=dd+1
            sDate=str(YYYY)+str("%02d" % MM)+str("%02d" % DD)
            sFileFin=sSaveDataDir+sDate+'_ERA-Int_Hail-Env_RDA_Sounding.nc'
            sFile=sFileFin+"_COPY"
            if os.path.isfile(sFileFin) != 1:
                iTimeSteps=4 #np.sum(((rgdFullTime >= datetime.datetime(int(YYYY), int(MM), 1)) & (rgdFullTime <= datetime.datetime(int(YYYY), int(MM), iDaysMM,21))))
                rgrData3DAct=np.zeros((len(rgr3Dvariables),iTimeSteps, 60, 256,512)); rgrData3DAct[:]=np.nan
                tt=0
                for hh in  range(len(grsHH)):
                    # _______________________________________________________________________
                    # start reading data
                    sFileName=s3Dfolder+str(YYYY)+str("%02d" % MM)+'/ei.oper.an.ml.regn128sc.'+str(YYYY)+str("%02d" % MM)+str("%02d" % (dd+1))+grsHH[hh]
                                                      # ei.oper.an.ml.regn128uv.2008012200
                    print '        Reading: '+sFileName
                    # convert data to netcdf for faster processing
                    iFileGRIB='/glade/scratch/prein/tmp/'+sFileName.split('/')[-1]+'.grib2'
                    iFileNC='/glade/scratch/prein/tmp/'+sFileName.split('/')[-1]+'.nc'
                    copyfile(sFileName, iFileGRIB)
                    subprocess.call("ncl_convert2nc "+iFileGRIB+' -o '+'/glade/scratch/prein/tmp/'+' -L', shell=True)
                    print '    Load '+iFileNC
                    # read in the variables
                    for v3 in range(len(rgr3Dvariables)-1):
                        ncid=Dataset(iFileNC, mode='r')
                        if rgr3Dvariables[v3] == '03_25_lnsp':
                            rgrLNSP=np.squeeze(ncid.variables[rgr3DERAvars[v3]][:])
                        else:
                            rgrData3DAct[v3,tt,:,:,:]=np.squeeze(ncid.variables[rgr3DERAvars[v3]][:])
                        ncid.close()
                    # clean up
                    os.system("rm "+iFileGRIB); os.system("rm "+iFileNC)
                    # calculate the pressure at model levels; equation form: https://rda.ucar.edu/datasets/ds627.2/docs/Eta_coordinate/
                    rgrData3DAct[rgr3Dvariables.index('03_25_lnsp'),tt,:,:,:]=np.array([ak[lv]+bk[lv]*np.exp(rgrLNSP) for lv in range(len(rgrLevelsERA))])
                    tt=tt+1

                print '    calculate height of model levels'
                geotoreturn, rgrData3DAct[rgr3Dvariables.index('Z'),:,:,:,:] = calculategeoh(rgrData3DAct[rgr3Dvariables.index('Z'),:,:,:,:], rgrLNSP, rgrData3DAct[rgr3Dvariables.index('00_00_t'),:,:,:], rgrData3DAct[rgr3Dvariables.index('01_00_q'),:,:,:], rgrLevelsERA)
    
    
                # # read in the staggered wind variables
                # sFileName=s3Dfolder+sFolderYYMO+'/ei.oper.an.ml.regn128uv.'+str(dDate.year)+str("%02d" % (dDate.month))+str("%02d" % (dDate.day))+str("%02d" % (dDate.hour))
                # iFileGRIB='/glade/scratch/prein/tmp/'+sFileName.split('/')[-1]+'.grib'
                # iFileNC='/glade/scratch/prein/tmp/'+sFileName.split('/')[-1]+'.nc'
                # copyfile(sFileName, iFileGRIB)
                # subprocess.call("ncl_convert2nc "+iFileGRIB+' -o '+'/glade/scratch/prein/tmp/'+' -L', shell=True)
                # print '    Load '+iFileNC
                # # read in the variables
                # ncid=Dataset(iFileNC, mode='r')
                # for v3 in range(len(rgr3Dstaggerd)):
                #     rgrData3Dstag[v3,:,:,:]=np.squeeze(ncid.variables[rgr3DERAStag[v3]][:])
                # ncid.close()
                # # clean up
                # os.system("rm "+iFileGRIB); os.system("rm "+iFileNC)
    
    
                ####################################################################################################
                ####################################################################################################
                #############                        Derive the target variables                        ############
                ####################################################################################################
                ####################################################################################################
    
                print '    Wet bulb zero high'
                from thermodynamics import VaporPressure, MixRatio, WetBulb, esat, MixR2VaporPress
                ESAT=esat(rgrData3DAct[rgr3Dvariables.index('00_00_t'),:,:,:])
                VapPres=MixR2VaporPress(rgrData3DAct[rgr3Dvariables.index('01_00_q'),:,:,:],rgrData3DAct[rgr3Dvariables.index('03_25_lnsp'),:,:,:])
                rgrRH=(VapPres/ESAT)*100.
                rgrWBT=WetBulb(rgrData3DAct[rgr3Dvariables.index('00_00_t'),:,:,:]-273.15,rgrRH)
                rgrWBZheight=np.zeros((iTimeSteps,rgrData3DAct.shape[3],rgrData3DAct.shape[4])); rgrWBZheight[:]=np.nan
                for tt in range(iTimeSteps):
                    print '        '+str(tt)+' of '+str(iTimeSteps)
                    for lo in range(rgrData3DAct.shape[3]):
                        for la in range(rgrData3DAct.shape[4]):
                            if rgrWBT[tt,-1,lo,la] <= 0:
                                rgrWBZheight[tt,lo,la]=0
                            else:
                                ff = interpolate.interp1d(rgrWBT[tt,-40:,lo,la], rgrData3DAct[rgr3Dvariables.index('Z'),tt,-40:,lo,la])
                                rgrWBZheight[tt,lo,la]=ff(0)
                rgrWBZheight[rgrWBZheight < 0]=0

                print '   3 Degree Temp. height'
                rgr3Cheight=np.zeros((iTimeSteps,rgrData3DAct.shape[3],rgrData3DAct.shape[4])); rgr3Cheight[:]=np.nan
                for tt in range(iTimeSteps):
                    print '        '+str(tt)+' of '+str(iTimeSteps)
                    for lo in range(rgrData3DAct.shape[3]):
                        for la in range(rgrData3DAct.shape[4]):
                            if rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-1,lo,la]-273.15 < 3:
                                rgr3Cheight[tt,lo,la]=0
                            else:
                                ff = interpolate.interp1d(rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-40:,lo,la]-273.15, rgrData3DAct[rgr3Dvariables.index('Z'),tt,-40:,lo,la])
                                rgr3Cheight[tt,lo,la]=ff(3)
                rgr3Cheight[rgr3Cheight < 0]=0
    
                print '   Probability for snow'
                rgrPsnow=np.zeros((iTimeSteps,rgrData3DAct.shape[3],rgrData3DAct.shape[4])); rgrPsnow[:]=np.nan
                for tt in range(iTimeSteps):
                    print '        '+str(tt)+' of '+str(iTimeSteps)
                    for lo in range(rgrData3DAct.shape[3]):
                        for la in range(rgrData3DAct.shape[4]):
                            rgi500=np.where(rgrData3DAct[rgr3Dvariables.index('Z'),tt,-40:,lo,la] <=500)[0]
                            try:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(rgrData3DAct[rgr3Dvariables.index('Z'),tt,-40:,lo,la][rgi500],rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-40:,lo,la][rgi500])
                                rgrPsnow[tt,lo,la]=fnSolidPrecip(rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-1:,lo,la]-273.15,rgrWBT[tt,-1:,lo,la],slope*1000.)
                            except:
                                continue
                            
                rgrPsnow[rgrPsnow < 0]=0
                rgrPsnow[np.isnan(rgrPsnow)]=0
    
                print '   Bellow Freezing Level Lapse Rate'
                rgrBflLR=np.zeros((iTimeSteps,rgrData3DAct.shape[3],rgrData3DAct.shape[4])); rgrBflLR[:]=np.nan
                for tt in range(iTimeSteps):
                    print '        '+str(tt)+' of '+str(iTimeSteps)
                    for lo in range(rgrData3DAct.shape[3]):
                        for la in range(rgrData3DAct.shape[4]):
                            if rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-1:,lo,la]-273.15 <= 0:
                                rgrBflLR[tt,lo,la]=np.nan
                            else:
                                rgiPosTemp=((rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-40:,lo,la]-273.15) >= 0)
                                if np.sum(rgiPosTemp) <= 3:
                                    rgrBflLR[tt,lo,la]=np.nan
                                else:
                                    try:
                                        f = interp1d(rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-40:,lo,la][-(1+sum(rgiPosTemp)):], rgrData3DAct[rgr3Dvariables.index('Z'),tt,-40:,lo,la][-(1+sum(rgiPosTemp)):])
                                        ZCheight=f(273.15)
                                        rgrBflLR[tt,lo,la]=-(rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-1:,lo,la][0]-273.15)/(ZCheight/1000.)
                                        # slope, intercept, r_value, p_value, std_err = stats.linregress(rgrData3DAct[rgr3Dvariables.index('Z'),tt,-40:,lo,la][rgiPosTemp],rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-40:,lo,la][rgiPosTemp])
                                        # rgrBflLR[tt,lo,la]=slope*1000.
                                    except:
                                        continue

                print '   CAPE, CIN, LCL, LFC, CBT'
                rgrSoundingProp=np.zeros((iTimeSteps,5,rgrData3DAct.shape[3],rgrData3DAct.shape[4])); rgrSoundingProp[:]=np.nan
                for tt in range(iTimeSteps):
                    print '        '+str(tt)+' of '+str(iTimeSteps)
                    CAPE, CIN, LCL, LFC=np.array(wrf.cape_2d(rgrData3DAct[rgr3Dvariables.index('03_25_lnsp'),tt,:,:,:]/100., rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,:,:,:], rgrData3DAct[rgr3Dvariables.index('01_00_q'),tt,:,:,:], rgrData3DAct[rgr3Dvariables.index('Z'),tt,:,:,:], rgrHeight, rgrData3DAct[rgr3Dvariables.index('03_25_lnsp'),tt,-1,:,:]/100., 1, missing=9.969209968386869e+36, meta=True))
                    rgiFin=np.isnan(CAPE)
                    LFC[rgiFin]=np.nan; LFC[LFC < 0]=np.nan
                    LCL[LCL<0]=np.nan
                    # calculate cloud base temperature
                    CBT=np.zeros((rgrData3DAct.shape[3],rgrData3DAct.shape[4])); CBT[:]=np.nan
                    for lo in range(rgrData3DAct.shape[3]):
                        for la in range(rgrData3DAct.shape[4]):
                            if np.isnan(LCL[lo,la]) != 1:
                                f = interp1d(rgrData3DAct[rgr3Dvariables.index('Z'),tt,:,lo,la],rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,:,lo,la])
                                if LCL[lo,la] <= rgrData3DAct[rgr3Dvariables.index('Z'),tt,-1,lo,la]:
                                    CBT[lo,la]=rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-1,lo,la]
                                else:
                                    try:
                                        CBT[lo,la]=f(LCL[lo,la])
                                    except:
                                        stop()
                    rgrSoundingProp[tt,:,:,:]=[CAPE,CIN,LCL,LFC,CBT]

                # ________________________________________________________________________
                # write the netcdf
                iTime=rgiTimeFull[(rgdFullTime >= datetime.datetime(int(YYYY), int(MM), int(DD),0)) & (rgdFullTime <= datetime.datetime(int(YYYY), int(MM), int(DD), 18))]
                print '    ----------------------'
                print '    Save data from '+sFileFin
                root_grp = Dataset(sFile, 'w', format='NETCDF4')
                # dimensions
                root_grp.createDimension('time', None)
                root_grp.createDimension('rlon', rgrLat.shape[1])
                root_grp.createDimension('rlat', rgrLon.shape[0])
                # variables
                lat = root_grp.createVariable('lat', 'f4', ('rlat','rlon',))
                lon = root_grp.createVariable('lon', 'f4', ('rlat','rlon',))
                time = root_grp.createVariable('time', 'f8', ('time',))
    
                WBZheight = root_grp.createVariable('WBZheight', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                ThreeCheight = root_grp.createVariable('ThreeCheight', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                ProbSnow = root_grp.createVariable('ProbSnow', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                BflLR = root_grp.createVariable('BflLR', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                CAPE = root_grp.createVariable('CAPE', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                CIN = root_grp.createVariable('CIN', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                LCL = root_grp.createVariable('LCL', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                LFC = root_grp.createVariable('LFC', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                CBT = root_grp.createVariable('CBT', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    
                time.calendar = "gregorian"
                time.units = "hours since 1979-1-1 00:00:00"
                time.standard_name = "time"
                time.long_name = "time"
                time.axis = "T"
    
                lon.standard_name = "longitude"
                lon.long_name = "longitude"
                lon.units = "degrees_east"
    
                lat.standard_name = "latitude"
                lat.long_name = "latitude"
                lat.units = "degrees_north"
    
                # write data to netcdf
                lat[:]=rgrLat
                lon[:]=rgrLon
                WBZheight[:]=rgrWBZheight
                ThreeCheight[:]=rgr3Cheight
                ProbSnow[:]=rgrPsnow
                BflLR[:]=rgrBflLR
                CAPE[:]=rgrSoundingProp[:,0,:,:]
                CIN[:]=rgrSoundingProp[:,1,:,:]
                LCL[:]=rgrSoundingProp[:,2,:,:]
                LFC[:]=rgrSoundingProp[:,3,:,:]
                CBT[:]=rgrSoundingProp[:,4,:,:]
    
                time[:]=iTime
                root_grp.close()
    
                # compress the netcdf file
                subprocess.Popen("nccopy -k 4 -d 1 -s "+sFile+' '+sFileFin, shell=True)
                import time
                time.sleep(10)
                subprocess.Popen("rm  "+sFile, shell=True)





