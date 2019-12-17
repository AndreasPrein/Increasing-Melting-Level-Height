#!/usr/bin/env python
'''
    File name: ERA-20C-Data-processing.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 23.08.2017
    Date last modified: 23.08.2017

    ##############################################################
    Purpos:

    1) Reads in 6 hourly ERA-20c data from RDA

    2) Calculates variables that are related to large hail
       development

    3) Saves the varaibles to yearly files in NCDF format

'''

from dateutil import rrule
import datetime
import glob
from netCDF4 import Dataset
import netCDF4
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
# import wrf

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
    # pv =  [
    #   0.0000000000e+000, 2.0000000000e+001, 3.8425338745e+001, 6.3647796631e+001, 9.5636962891e+001,
    #   1.3448330688e+002, 1.8058435059e+002, 2.3477905273e+002, 2.9849584961e+002, 3.7397192383e+002,
    #   4.6461816406e+002, 5.7565112305e+002, 7.1321801758e+002, 8.8366040039e+002, 1.0948347168e+003,
    #   1.3564746094e+003, 1.6806403809e+003, 2.0822739258e+003, 2.5798886719e+003, 3.1964216309e+003,
    #   3.9602915039e+003, 4.9067070313e+003, 6.0180195313e+003, 7.3066328125e+003, 8.7650546875e+003,
    #   1.0376125000e+004, 1.2077445313e+004, 1.3775324219e+004, 1.5379804688e+004, 1.6819472656e+004,
    #   1.8045183594e+004, 1.9027695313e+004, 1.9755109375e+004, 2.0222203125e+004, 2.0429863281e+004,
    #   2.0384480469e+004, 2.0097402344e+004, 1.9584328125e+004, 1.8864750000e+004, 1.7961359375e+004,
    #   1.6899468750e+004, 1.5706449219e+004, 1.4411125000e+004, 1.3043218750e+004, 1.1632757813e+004,
    #   1.0209500000e+004, 8.8023554688e+003, 7.4388046875e+003, 6.1443164063e+003, 4.9417773438e+003,
    #   3.8509133301e+003, 2.8876965332e+003, 2.0637797852e+003, 1.3859125977e+003, 8.5536181641e+002,
    #   4.6733349609e+002, 2.1039389038e+002, 6.5889236450e+001, 7.3677425385e+000, 0.0000000000e+000,
    #   0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
    #   0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
    #   0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
    #   0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
    #   0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
    #   7.5823496445e-005, 4.6139489859e-004, 1.8151560798e-003, 5.0811171532e-003, 1.1142909527e-002,
    #   2.0677875727e-002, 3.4121163189e-002, 5.1690407097e-002, 7.3533833027e-002, 9.9674701691e-002,
    #   1.3002252579e-001, 1.6438430548e-001, 2.0247590542e-001, 2.4393314123e-001, 2.8832298517e-001,
    #   3.3515489101e-001, 3.8389211893e-001, 4.3396294117e-001, 4.8477154970e-001, 5.3570991755e-001,
    #   5.8616840839e-001, 6.3554745913e-001, 6.8326860666e-001, 7.2878581285e-001, 7.7159661055e-001,
    #   8.1125342846e-001, 8.4737491608e-001, 8.7965691090e-001, 9.0788388252e-001, 9.3194031715e-001,
    #   9.5182150602e-001, 9.6764522791e-001, 9.7966271639e-001, 9.8827010393e-001, 9.9401944876e-001,
    #   9.9763011932e-001, 1.0000000000e+000 ]

    # These are simple the a and b parameter appended into one linst!
    pv = [0,2.00004,3.980832,7.387186,12.908319,21.413612,33.952858,51.746601,76.167656,108.715561,150.986023,204.637451,271.356506,352.824493,450.685791,566.519226,701.813354,857.945801,1036.166504,1237.585449,1463.16394,1713.709595,1989.87439,2292.155518,2620.898438,2976.302246,3358.425781,3767.196045,4202.416504,4663.776367,5150.859863,5663.15625,6199.839355,6759.727051,7341.469727,7942.92627,8564.624023,9208.305664,9873.560547,10558.88184,11262.48438,11982.66211,12713.89746,13453.22559,14192.00977,14922.68555,15638.05371,16329.56055,16990.62305,17613.28125,18191.0293,18716.96875,19184.54492,19587.51367,19919.79688,20175.39453,20348.91602,20434.1582,20426.21875,20319.01172,20107.03125,19785.35742,19348.77539,18798.82227,18141.29688,17385.5957,16544.58594,15633.56641,14665.64551,13653.21973,12608.38379,11543.16699,10471.31055,9405.222656,8356.25293,7335.164551,6353.920898,5422.802734,4550.21582,3743.464355,3010.146973,2356.202637,1784.854614,1297.656128,895.193542,576.314148,336.772369,162.043427,54.208336,6.575628,0.00316,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000014,0.000055,0.000131,0.000279,0.000548,0.001,0.001701,0.002765,0.004267,0.006322,0.009035,0.012508,0.01686,0.022189,0.02861,0.036227,0.045146,0.055474,0.067316,0.080777,0.095964,0.112979,0.131935,0.152934,0.176091,0.20152,0.229315,0.259554,0.291993,0.326329,0.362203,0.399205,0.436906,0.475016,0.51328,0.551458,0.589317,0.626559,0.662934,0.698224,0.732224,0.764679,0.795385,0.824185,0.85095,0.875518,0.897767,0.917651,0.935157,0.950274,0.963007,0.973466,0.982238,0.989153,0.994204,0.99763,1]


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
        t_level=np.squeeze(ts[ilevel,:,:])
        q_level=np.squeeze(qs[ilevel,:,:])
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
        try:
            z_f = z_h + (TRd*alpha)
        except:
            stop()
        #Convert geopotential to height
        heighttoreturn[ilevel,:,:] = z_f / 9.80665
        #Geopotential (add in surface geopotential)
        try:
            geotoreturn[:,:,:] = z_f + z
        except:
            stop()
        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h=z_h+(TRd*dlogP)
        Ph_levplusone = Ph_lev
    return geotoreturn, heighttoreturn

# coefficients from: https://www.ecmwf.int/en/forecasts/documentation-and-support/91-model-levels
ak=np.array([0,2.00004,3.980832,7.387186,12.908319,21.413612,33.952858,51.746601,76.167656,108.715561,150.986023,204.637451,271.356506,352.824493,450.685791,566.519226,701.813354,857.945801,1036.166504,1237.585449,1463.16394,1713.709595,1989.87439,2292.155518,2620.898438,2976.302246,3358.425781,3767.196045,4202.416504,4663.776367,5150.859863,5663.15625,6199.839355,6759.727051,7341.469727,7942.92627,8564.624023,9208.305664,9873.560547,10558.88184,11262.48438,11982.66211,12713.89746,13453.22559,14192.00977,14922.68555,15638.05371,16329.56055,16990.62305,17613.28125,18191.0293,18716.96875,19184.54492,19587.51367,19919.79688,20175.39453,20348.91602,20434.1582,20426.21875,20319.01172,20107.03125,19785.35742,19348.77539,18798.82227,18141.29688,17385.5957,16544.58594,15633.56641,14665.64551,13653.21973,12608.38379,11543.16699,10471.31055,9405.222656,8356.25293,7335.164551,6353.920898,5422.802734,4550.21582,3743.464355,3010.146973,2356.202637,1784.854614,1297.656128,895.193542,576.314148,336.772369,162.043427,54.208336,6.575628,0.00316,0])
bk=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000014,0.000055,0.000131,0.000279,0.000548,0.001,0.001701,0.002765,0.004267,0.006322,0.009035,0.012508,0.01686,0.022189,0.02861,0.036227,0.045146,0.055474,0.067316,0.080777,0.095964,0.112979,0.131935,0.152934,0.176091,0.20152,0.229315,0.259554,0.291993,0.326329,0.362203,0.399205,0.436906,0.475016,0.51328,0.551458,0.589317,0.626559,0.662934,0.698224,0.732224,0.764679,0.795385,0.824185,0.85095,0.875518,0.897767,0.917651,0.935157,0.950274,0.963007,0.973466,0.982238,0.989153,0.994204,0.99763,1])


########################################
#                            USER INPUT SECTION

sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/ERA-20c/'

# dStartDay=datetime.datetime(int(>>YYYY<<), 1, 1,0)
# dStopDay=datetime.datetime(int(>>YYYY<<), 12, 31,23)
YYYY=int(>>YYYY<<)
dStartDay=datetime.datetime(int(YYYY), 1, 1,0)
dStopDay=datetime.datetime(int(YYYY), 12, 31,23)

rgdTime6H = pd.date_range(dStartDay, end=dStopDay, freq='3h')
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgiYY=np.unique(rgdTime6H.year)
rgiERA_hours=np.array([0,6,12,18])

rgdFullTime=pd.date_range(datetime.datetime(1900, 1, 1,00),
                          end=datetime.datetime(2010, 12, 31,23), freq='3h')
rgiTimeFull=np.array(range(len(rgdFullTime)))*3

# define hight levels for vertical interpolation
rgrLevelsZZ=[1000,3000,6000,12000] # height levels in m

# ERA levels
rgrLevelsERA=np.array(range(1,91,1))

rgr3Dvariables=['00_00_t','01_00_q','03_25_lnsp','Z']
rgr3DERAvars=['TMP_P0_L105_GGA0','SPFH_P0_L105_GGA0','NLPRES_P0_L105_GGA0']

rgr3Dstaggerd=['02_02_u','02_03_v']
rgr3DERAStag=['UGRD_P0_L105_GGA0','VGRD_P0_L105_GGA0']

rgr2Dvariables=['PS']
rgr2DERAvars=['LNSP_GDS4_HYBL']
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
# # first read the coordinates
sERAconstantFields='/glade/work/prein/reanalyses/ERA-20c/e20c.oper.invariant.128_129_z.regn80sc.1900010100_2010123121.nc'
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
# read in the 3D ERA-20c fields, do the statistics, and save the data
dfTCstatsAll={}
for yy in range(len(rgiYY)): #range(len(TC_Name)): # loop over hurricanes
    print ' '
    print '    Workin on year '+str(rgiYY[yy])
    s3Dfolder='/gpfs/fs1/collections/rda/data/ds626.0/e20c.oper.an.ml.3hr/'    # 2000/wrf3d_d01_CTRL_U_20001001.nc
    s3Dextension='e20c.oper.an.ml.3hr.0_5_'
    s2Dfolder='/gpfs/fs1/collections/rda/data/ds626.0/e20c.oper.an.sfc.3hr/'    # 2000/wrf2d_d01_CTRL_LH_200101-200103.nc
    s2Dextension='e20c.oper.an.sfc.3hr.'

    # check if we already processed this file
    for mm in range(12): #  loop over time
        for dd in range(2):
            YYYY=str(rgiYY[yy])
            MM=str("%02d" % (mm+1))
            if dd == 0:
                DD1='0100'
                DD2='1521'
            else:
                DD1='1600'
                iDaysInMon=rgdFullTime[((rgdFullTime.year == int(YYYY)) & (rgdFullTime.month == int(MM)))][-1].day
                DD2=str(iDaysInMon)+'21'
            if YYYY != '2010':
                sDecade=str(YYYY)[:3]+'0_'+str(YYYY)[:3]+'9'
            else:
                sDecade=str(YYYY)[:3]+'0_'+str(YYYY)[:3]+'0'
            sDate=YYYY+MM+DD1+'_'+YYYY+MM+DD2
            sFileFin=sSaveDataDir+'Full/'+sDate+'_ERA-20c_Hail-Env_RDA.nc'
            sFile=sFileFin+"_COPY"
            # stop()
            # dset = netCDF4.Dataset(sFileFin)
            # Vars=dset.variables.keys()
            Vars=['NaN']

            if os.path.isfile(sFileFin) != 1:
            # if ('CAPE' in Vars) == False:
                iTimeSteps=np.sum(((rgdFullTime >= datetime.datetime(int(YYYY), int(MM), int(DD1[:2]),0)) & (rgdFullTime <= datetime.datetime(int(YYYY), int(MM), int(DD2[:2]),21))))
                rgrData3DAct=np.zeros((len(rgr3Dvariables),iTimeSteps, 91, 160,320)); rgrData3DAct[:]=np.nan
                rgrData3Dstag=np.zeros((len(rgr3Dstaggerd),iTimeSteps, 91, 160,320)); rgrData3Dstag[:]=np.nan
                # _______________________________________________________________________
                # start reading data
                for v3 in range(len(rgr3Dvariables)-1):
                    sFileName=s3Dfolder+sDecade+'/e20c.oper.an.ml.3hr.0_5_'+rgr3Dvariables[v3]+'.regn80sc.'+sDate
                    print '        Reading: '+sFileName
                    # convert data to netcdf for faster processing
                    iFileGRIB='/glade/scratch/prein/tmp/'+sFileName.split('/')[-1]+'.grib2'
                    iFileNC='/glade/scratch/prein/tmp/'+sFileName.split('/')[-1]+'.nc'
                    copyfile(sFileName+'.grib2', iFileGRIB)
                    subprocess.call("ncl_convert2nc "+iFileGRIB+' -o '+'/glade/scratch/prein/tmp/'+' -L', shell=True)
                    print '    Load '+iFileNC
                    # read in the variables
                    ncid=Dataset(iFileNC, mode='r')
                    if rgr3Dvariables[v3] == '03_25_lnsp':
                        rgrLNSP=np.squeeze(ncid.variables[rgr3DERAvars[v3]][:])
                    else:
                        rgrData3DAct[v3,:,:,:,:]=np.squeeze(ncid.variables[rgr3DERAvars[v3]][:])
                    ncid.close()
                    # clean up
                    os.system("rm "+iFileGRIB); os.system("rm "+iFileNC)
                #Calculate pressure on model levels
                rgrPRhalflev=ak[None,:,None,None]+bk[None,:,None,None]*np.exp(rgrLNSP[:,None,:,:])
                rgrData3DAct[v3,:,:,:,:]=0.5*(rgrPRhalflev[:,:-1,:,:]+rgrPRhalflev[:,1:,:,:])
          
                print '    calculate height of model levels'
                for tt in range(iTimeSteps):
                    geotoreturn, rgrData3DAct[-1,tt,:,:,:] = calculategeoh(rgrHeight[None,:,:],
                                                                           rgrLNSP[tt,:,:],
                                                                           rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,:,:],
                                                                           rgrData3DAct[rgr3Dvariables.index('01_00_q'),tt,:,:],
                                                                           rgrLevelsERA)

                # set surface hight to zero
                rgrData3DAct[rgr3Dvariables.index('Z'),:,:,:,:][(rgrData3DAct[rgr3Dvariables.index('Z'),:,:,:,:] == -999.)]=0

                # read in the staggered wind variables
                for v3 in range(len(rgr3Dstaggerd)):
                    sFileName=s3Dfolder+sDecade+'/e20c.oper.an.ml.3hr.0_5_'+rgr3Dstaggerd[v3]+'.regn80uv.'+sDate
                    iFileGRIB='/glade/scratch/prein/tmp/'+sFileName.split('/')[-1]+'.grib2'
                    iFileNC='/glade/scratch/prein/tmp/'+sFileName.split('/')[-1]+'.nc'
                    copyfile(sFileName+'.grib2', iFileGRIB)
                    subprocess.call("ncl_convert2nc "+iFileGRIB+' -o '+'/glade/scratch/prein/tmp/'+' -L', shell=True)
                    print '    Load '+iFileNC
                    # read in the variables
                    ncid=Dataset(iFileNC, mode='r')
                    rgrData3Dstag[v3,:,:,:]=np.squeeze(ncid.variables[rgr3DERAStag[v3]][:])
                    ncid.close()
                    # clean up
                    os.system("rm "+iFileGRIB); os.system("rm "+iFileNC)
    
    
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
                            if rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-2,lo,la]-273.15 < 3:
                                rgr3Cheight[tt,lo,la]=0
                            else:
                                ff = interpolate.interp1d(rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-60:-1,lo,la]-273.15, rgrData3DAct[rgr3Dvariables.index('Z'),tt,-60:-1,lo,la])
                                rgr3Cheight[tt,lo,la]=ff(3)
                rgr3Cheight[rgr3Cheight < 0]=0

                print '   Probability for snow'
                rgrPsnow=np.zeros((iTimeSteps,rgrData3DAct.shape[3],rgrData3DAct.shape[4])); rgrPsnow[:]=np.nan
                for tt in range(iTimeSteps):
                    print '        '+str(tt)+' of '+str(iTimeSteps)
                    for lo in range(rgrData3DAct.shape[3]):
                        for la in range(rgrData3DAct.shape[4]):
                            rgi500=np.where(rgrData3DAct[rgr3Dvariables.index('Z'),tt,-40:,lo,la] <=500)[0]
                            slope, intercept, r_value, p_value, std_err = stats.linregress(rgrData3DAct[rgr3Dvariables.index('Z'),tt,-40:,lo,la][rgi500],rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-40:,lo,la][rgi500])
                            rgrPsnow[tt,lo,la]=fnSolidPrecip(rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-1:,lo,la]-273.15,rgrWBT[tt,-1:,lo,la],slope*1000.)
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
                                rgiPosTemp=np.where((rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-60:,lo,la]-273.15) >= 0)[0]
                                if len(rgiPosTemp) <= 3:
                                    rgrBflLR[tt,lo,la]=np.nan
                                else:
                                    f = interp1d(rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-60:,lo,la][-(1+len(rgiPosTemp)):], rgrData3DAct[rgr3Dvariables.index('Z'),tt,-60:,lo,la][-(1+len(rgiPosTemp)):])
                                    ZCheight=f(273.15)
                                    rgrBflLR[tt,lo,la]=-(rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-1:,lo,la][0]-273.15)/(ZCheight/1000.)
                                    # slope, intercept, r_value, p_value, std_err = stats.linregress(rgrData3DAct[rgr3Dvariables.index('Z'),tt,-60:,lo,la][rgiPosTemp[0]:],rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt,-60:,lo,la][rgiPosTemp[0]:])
                                    # rgrBflLR[tt,lo,la]=slope*1000.

                print '   Sounding properties'
                rgrSoundingProp=np.zeros((iTimeSteps,5,rgrData3DAct.shape[3],rgrData3DAct.shape[4])); rgrSoundingProp[:]=np.nan
                for tt in range(iTimeSteps):
                    CAPE, CIN, LCL, LFC=np.array(wrf.cape_2d(rgrData3DAct[rgr3Dvariables.index('03_25_lnsp'),tt]/100., rgrData3DAct[rgr3Dvariables.index('00_00_t'),tt], rgrData3DAct[rgr3Dvariables.index('01_00_q'),tt], rgrData3DAct[rgr3Dvariables.index('Z'),tt,:,:,:], rgrHeight, np.exp(rgrLNSP[tt,:])/100., 1, missing=9.969209968386869e+36, meta=True))
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
                    rgrSoundingProp[tt,:,:]=[CAPE,CIN,LCL,LFC,CBT]


                print '    Calculate Windspeed Shear'
                rgrShear=np.zeros((iTimeSteps,2,rgrData3DAct.shape[3],rgrData3DAct.shape[4])); rgrShear[:]=np.nan
                UV=(rgrData3Dstag[0,:]**2+rgrData3Dstag[1,:]**2)**(0.5)
                for tt in range(iTimeSteps):
                    print '        '+str(tt)+' of '+str(iTimeSteps)
                    for lo in range(rgrData3DAct.shape[3]):
                        for la in range(rgrData3DAct.shape[4]):
                            ff = interpolate.interp1d(rgrData3DAct[rgr3Dvariables.index('Z'),tt,-60:-1,lo,la],UV[tt,-60:-1,lo,la])
                            rgrShear[tt,:,lo,la]=[ff(3000),ff(6000)]

                # ________________________________________________________________________
                # add data to existing NetCDF
                print '    ----------------------'
                print '    Add data to '+sFileFin
                iTime=rgiTimeFull[(rgdFullTime >= datetime.datetime(int(YYYY), int(MM), int(DD1[:2]),0)) & (rgdFullTime <= datetime.datetime(int(YYYY), int(MM), int(DD2[:2]),21))]
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
                SH03 = root_grp.createVariable('SH03', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                SH06 = root_grp.createVariable('SH06', 'f4', ('time','rlat','rlon',),fill_value=-99999)

                time.calendar = "gregorian"
                time.units = "hours since 1900-1-1 00:00:00"
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
                CAPE[:]=rgrSoundingProp[:,0,:]
                CIN[:]=rgrSoundingProp[:,1,:]
                LCL[:]=rgrSoundingProp[:,2,:]
                LFC[:]=rgrSoundingProp[:,3,:]
                CBT[:]=rgrSoundingProp[:,4,:]
                SH03[:]=rgrShear[:,0,:]
                SH06[:]=rgrShear[:,1,:]
                
                time[:]=iTime
                root_grp.close()

                # Append the file to the existing file
                subprocess.Popen("ncks -A "+sFileFin+' '+sFile, shell=True)
                # compress the netcdf file
                subprocess.Popen("nccopy -k 4 -d 1 -s "+sFile+' '+sSaveDataDir+sDate+'_ERA-20c_Hail-Env_RDA.nc', shell=True)
                import time
                time.sleep(10)
                subprocess.Popen("rm  "+sFile, shell=True)


                # # ________________________________________________________________________
                # # write the netcdf
                # iTime=rgiTimeFull[(rgdFullTime >= datetime.datetime(int(YYYY), int(MM), int(DD1[:2]),0)) & (rgdFullTime <= datetime.datetime(int(YYYY), int(MM), int(DD2[:2]),21))]
                # print '    ----------------------'
                # print '    Save data from '+sFileFin
                # root_grp = Dataset(sFile, 'w', format='NETCDF4')
                # # dimensions
                # root_grp.createDimension('time', None)
                # root_grp.createDimension('rlon', rgrLat.shape[1])
                # root_grp.createDimension('rlat', rgrLon.shape[0])
                # # variables
                # lat = root_grp.createVariable('lat', 'f4', ('rlat','rlon',))
                # lon = root_grp.createVariable('lon', 'f4', ('rlat','rlon',))
                # time = root_grp.createVariable('time', 'f8', ('time',))
    
                # WBZheight = root_grp.createVariable('WBZheight', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # ThreeCheight = root_grp.createVariable('ThreeCheight', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # ProbSnow = root_grp.createVariable('ProbSnow', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # BflLR = root_grp.createVariable('BflLR', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # CAPE = root_grp.createVariable('CAPE', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # CIN = root_grp.createVariable('CIN', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # LCL = root_grp.createVariable('LCL', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # LFC = root_grp.createVariable('LFC', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # SH03 = root_grp.createVariable('SH03', 'f4', ('time','rlat','rlon',),fill_value=-99999)
                # SH06 = root_grp.createVariable('SH06', 'f4', ('time','rlat','rlon',),fill_value=-99999)
    
                # time.calendar = "gregorian"
                # time.units = "hours since 1900-1-1 00:00:00"
                # time.standard_name = "time"
                # time.long_name = "time"
                # time.axis = "T"
    
                # lon.standard_name = "longitude"
                # lon.long_name = "longitude"
                # lon.units = "degrees_east"
    
                # lat.standard_name = "latitude"
                # lat.long_name = "latitude"
                # lat.units = "degrees_north"
    
                # # write data to netcdf
                # lat[:]=rgrLat
                # lon[:]=rgrLon
                # WBZheight[:]=rgrWBZheight
                # ThreeCheight[:]=rgr3Cheight
                # ProbSnow[:]=rgrPsnow
                # BflLR[:]=rgrBflLR
                # CAPE[:]=rgrSoundingProp[:,0,:]
                # CIN[:]=rgrSoundingProp[:,1,:]
                # LCL[:]=rgrSoundingProp[:,2,:]
                # LFC[:]=rgrSoundingProp[:,3,:]
                # SH03[:]=rgrShear[:,0,:]
                # SH06[:]=rgrShear[:,1,:]
    
                # time[:]=iTime
                # root_grp.close()
    
                # # compress the netcdf file
                # subprocess.Popen("nccopy -k 4 -d 1 -s "+sFile+' '+sFileFin, shell=True)
                # import time
                # time.sleep(10)
                # subprocess.Popen("rm  "+sFile, shell=True)







