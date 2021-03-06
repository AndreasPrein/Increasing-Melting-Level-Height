#!/usr/bin/env python
'''
    File name: Process_Snow-Data_post1990.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 21.05.2017
    Date last modified: 21.05.2017

    ##############################################################
    Purpos:

    Reads in hourly snow observations from txt files and write the data in to a NetCDF

    inputs:

    returns:

'''

########################################
#                            Load Modules
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
# from mpl_toolkits import basemap
import ESMF
import pickle
import subprocess
import pandas as pd
import copy
import linecache
from scipy.interpolate import interp1d
import pylab as plt
from HelperFunctions import fnSolidPrecip
from scipy import stats

########################################

sSnowData='/glade/work/prein/papers/Trends_RadSoundings/data/SNOW-Obs/'
sSaveDataDir=sSnowData+'ncdf/'

dStartDay=datetime.datetime(1960, 1, 1,0)
dStopDay=datetime.datetime(2017, 12, 31,23)
rgdTimeAll = pd.date_range(dStartDay, end=dStopDay, freq='h')
########################################

#--------------------------------------------------------------------------------------------------
#                           Read the Snow obs.

iFiles=glob.glob(sSnowData+"*.txt")

for ff in range(len(iFiles)):
    sFile=iFiles[ff]
    print '    Process '+sFile

    # read in Data
    ii=0
    with open(sFile, "r") as ins:
        array = []
        for line in ins:
            if ii > 0:
                array.append(line)
            else:
                Station=line
            ii=ii+1
    
    # # split string at spaces <-- does not work because of poor data format
    # array=np.array([str.split(array[ii]) for ii in range(len(array))])
    # Get data from array
    rUSAF=np.array([array[ii][0:6] for ii in range(len(array))]).astype('int')
    rWBAN=np.array([array[ii][7:12] for ii in range(len(array))]).astype('str'); rWBAN=netCDF4.stringtochar(np.array(rWBAN, 'S4'))
    rYR_MODAHRMN=np.array([array[ii][13:25] for ii in range(len(array))]).astype('int')
    rDIR=np.array([array[ii][26:29] for ii in range(len(array))]).astype('str'); rDIR[rDIR =='***']=-99; rDIR=netCDF4.stringtochar(np.array(rDIR, 'S3'))
    rSPD=np.array([array[ii][30:33] for ii in range(len(array))]).astype('str'); rSPD[rSPD =='***']=-99; rSPD=rSPD.astype('int')
    rGUS=np.array([array[ii][34:37] for ii in range(len(array))]).astype('str'); rGUS[rGUS =='***']=-99; rGUS=rGUS.astype('int')
    rCLG=np.array([array[ii][38:41] for ii in range(len(array))]).astype('str'); rCLG[rCLG =='***']=-99; rCLG=rCLG.astype('int')
    rSKC=np.array([array[ii][42:45] for ii in range(len(array))]).astype('str'); rSKC=netCDF4.stringtochar(np.array(rSKC, 'S3'))
    rL=np.array([array[ii][46:47] for ii in range(len(array))]).astype('str'); rL=netCDF4.stringtochar(np.array(rL, 'S1'))
    rM=np.array([array[ii][48:49] for ii in range(len(array))]).astype('str'); rM=netCDF4.stringtochar(np.array(rM, 'S1'))
    rH=np.array([array[ii][50:51] for ii in range(len(array))]).astype('str'); rH=netCDF4.stringtochar(np.array(rH, 'S1'))
    rVSBq=np.array([array[ii][52:56] for ii in range(len(array))]).astype('str'); rVSBq[rVSBq =='****']=-999; rVSBq=netCDF4.stringtochar(np.array(rVSBq, 'S4'))
    rMW1=np.array([array[ii][57:59] for ii in range(len(array))]).astype('str'); rMW1[rMW1 =='**']=-9; rMW1[rMW1 ==' *']=-9; rMW1=rMW1.astype('int')
    rMW2=np.array([array[ii][60:62] for ii in range(len(array))]).astype('str'); rMW2[rMW2 =='**']=-9; rMW2[rMW2 ==' *']=-9; rMW2=rMW2.astype('int')
    rMW3=np.array([array[ii][63:65] for ii in range(len(array))]).astype('str'); rMW3[rMW3 =='**']=-9; rMW3[rMW3 ==' *']=-9; rMW3=rMW3.astype('int')
    rMW4=np.array([array[ii][66:68] for ii in range(len(array))]).astype('str'); rMW4[rMW4 =='**']=-9; rMW4[rMW4 ==' *']=-9; rMW4=rMW4.astype('int')
    rAW1=np.array([array[ii][69:71] for ii in range(len(array))]).astype('str'); rAW1[rAW1 =='**']=-9; rAW1[rAW1 ==' *']=-9; rAW1=rAW1.astype('int')
    rAW2=np.array([array[ii][72:74] for ii in range(len(array))]).astype('str'); rAW2[rAW2 =='**']=-9; rAW2[rAW2 ==' *']=-9; rAW2=rAW2.astype('int')
    rAW3=np.array([array[ii][75:77] for ii in range(len(array))]).astype('str'); rAW3[rAW3 =='**']=-9; rAW3[rAW3 ==' *']=-9; rAW3=rAW3.astype('int')
    rAW4=np.array([array[ii][78:80] for ii in range(len(array))]).astype('str'); rAW4[rAW4 =='**']=-9; rAW4[rAW4 ==' *']=-9; rAW4=rAW4.astype('int')
    rW=np.array([array[ii][81:82] for ii in range(len(array))]).astype('str'); rW=netCDF4.stringtochar(np.array(rW, 'S1'))
    rTEMP=np.array([array[ii][83:87] for ii in range(len(array))]).astype('str'); rTEMP[rTEMP =='****']=-999; rTEMP=rTEMP.astype('int')
    rDEWP=np.array([array[ii][88:92] for ii in range(len(array))]).astype('str'); rDEWP[rDEWP =='****']=-999; rDEWP=rDEWP.astype('int')
    rSLP=np.array([array[ii][93:99] for ii in range(len(array))]).astype('str'); rSLP[rSLP =='******']=-99999; rSLP[rSLP ==' *****']=-99999; rSLP=rSLP.astype('float')
    rALT=np.array([array[ii][100:105] for ii in range(len(array))]).astype('str'); rALT[rALT =='*****']=-9999; rALT=rALT.astype('float')
    rSTP=np.array([array[ii][106:112] for ii in range(len(array))]).astype('str') ; rSTP[rSTP =='******']=-99999; rSTP[rSTP ==' *****']=-99999; rSTP=rSTP.astype('float')
    rMAX=np.array([array[ii][113:116] for ii in range(len(array))]).astype('str'); rMAX[rMAX =='***']=-99; rMAX[rMAX ==' **']=-99; rMAX=rMAX.astype('int')
    rMIN=np.array([array[ii][117:120] for ii in range(len(array))]).astype('str'); rMIN[rMIN =='***']=-99; rMIN[rMIN ==' **']=-99; rMIN=rMIN.astype('int')
    rPCP01=np.array([array[ii][121:126] for ii in range(len(array))]).astype('str'); rPCP01[rPCP01 =='*****']=-9999; rPCP01[rPCP01 ==' ****']=-9999; rPCP01=rPCP01.astype('float')
    rPCP06=np.array([array[ii][127:132] for ii in range(len(array))]).astype('str'); rPCP06[rPCP06 =='*****']=-9999; rPCP06[rPCP06 ==' ****']=-9999; rPCP06[rPCP06 =='T****']=-9999; rPCP06=rPCP06.astype('float')
    rPCP24=np.array([array[ii][133:138] for ii in range(len(array))]).astype('str'); rPCP24[rPCP24 =='*****']=-9999; rPCP24[rPCP24 ==' ****']=-9999; rPCP24=rPCP24.astype('float')
    rPCPXX=np.array([array[ii][139:144] for ii in range(len(array))]).astype('str'); rPCPXX[rPCPXX =='*****']=-9999; rPCPXX[rPCPXX ==' ****']=-9999; rPCPXX=rPCPXX.astype('float')
    rSD=np.array([array[ii][145:147] for ii in range(len(array))]).astype('str'); rSD[rSD =='**']=-9; rSD[rSD ==' *']=-9; rSD[rSD =='  ']=-9; rSD=rSD.astype('int')

    # # Create time vector
    # if Year[0] > 20:
    #     Yadd=1900
    # else:
    #     Yadd=2000
    # dStartDay=datetime.datetime(Year[0]+Yadd, Month[0], Day[0],Hour[0]-1)
    # dStopDay=datetime.datetime(Year[-1]+Yadd, Month[-1], Day[-1],Hour[-1]-1)
    # rgdTimeH = pd.date_range(dStartDay, end=dStopDay, freq='h')
    
    # iStartDay=np.where((dStartDay.year == rgdTimeAll.year) & (dStartDay.month == rgdTimeAll.month) & (dStartDay.day == rgdTimeAll.day) & (dStartDay.hour == rgdTimeAll.hour))[0][0]
    # rgiTime=iStartDay+np.array(range(len(rgdTimeH)))
    
    
    # ===============================================================
    # Save the data as NetCDF
    
    sFileFin=sSaveDataDir+str.split(str.split(sFile,'/')[-1],'.')[0]+'.nc'
    sFile=sFileFin+"_COPY"
    
    # ________________________________________________________________________
    # write the netcdf
    print '    ----------------------'
    print '    Save data from '+sFileFin
    root_grp = Dataset(sFileFin, 'w', format='NETCDF4')
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('rlon', 1)
    root_grp.createDimension('rlat', 1)
    # variables
    # lat = root_grp.createVariable('lat', 'f4', ('rlat',))
    # lon = root_grp.createVariable('lon', 'f4', ('rlon',))
    time = root_grp.createVariable('time', 'f8', ('time',))
    S4 = root_grp.createDimension('S4', 4)
    S3 = root_grp.createDimension('S3', 3)
    S1 = root_grp.createDimension('S1', 1)
    
    USAF = root_grp.createVariable('USAF', 'f4', ('time',),fill_value=-999)
    WBAN = root_grp.createVariable('WBAN', 'S1', ('time','S4',))
    YRMODAHRMN = root_grp.createVariable('YRMODAHRMN', np.int64, ('time',),fill_value=-99999)
    DIR = root_grp.createVariable('DIR', 'S1', ('time','S3',))
    SPD = root_grp.createVariable('SPD', 'f4', ('time',),fill_value=-99)
    GUS = root_grp.createVariable('GUS', 'f4', ('time',),fill_value=-99)
    CLG = root_grp.createVariable('CLG', 'f4', ('time',),fill_value=-99)
    SKC = root_grp.createVariable('SKC', 'S1', ('time','S3',))
    L = root_grp.createVariable('L', 'S1', ('time','S1',))
    M = root_grp.createVariable('M', 'S1', ('time','S1',))
    H = root_grp.createVariable('H', 'S1', ('time','S1',))
    VSBq = root_grp.createVariable('VSBq', 'S1', ('time','S4',))
    MW1 = root_grp.createVariable('MW1', 'f4', ('time',),fill_value=-9)
    MW2 = root_grp.createVariable('MW2', 'f4', ('time',),fill_value=-9)
    MW3 = root_grp.createVariable('MW3', 'f4', ('time',),fill_value=-9)
    MW4 = root_grp.createVariable('MW4', 'f4', ('time',),fill_value=-9)
    AW1 = root_grp.createVariable('AW1', 'f4', ('time',),fill_value=-9)
    AW2 = root_grp.createVariable('AW2', 'f4', ('time',),fill_value=-9)
    AW3 = root_grp.createVariable('AW3', 'f4', ('time',),fill_value=-9)
    AW4 = root_grp.createVariable('AW4', 'f4', ('time',),fill_value=-9)
    W = root_grp.createVariable('W', 'S1', ('time','S1',))
    TEMP = root_grp.createVariable('TEMP', 'f4', ('time',),fill_value=-999)
    DEWP = root_grp.createVariable('DEWP', 'f4', ('time',),fill_value=-999)
    SLP = root_grp.createVariable('SLP', 'f4', ('time',),fill_value=-99999)
    ALT = root_grp.createVariable('ALT', 'f4', ('time',),fill_value=-9999)
    STP = root_grp.createVariable('STP', 'f4', ('time',),fill_value=-99999)
    MAX = root_grp.createVariable('MAX', 'f4', ('time',),fill_value=-99)
    MIN = root_grp.createVariable('MIN', 'f4', ('time',),fill_value=-99)
    PCP01 = root_grp.createVariable('PCP01', 'f4', ('time',),fill_value=-9999)
    PCP06 = root_grp.createVariable('PCP06', 'f4', ('time',),fill_value=-9999)
    PCP24 = root_grp.createVariable('PCP24', 'f4', ('time',),fill_value=-9999)
    PCPXX = root_grp.createVariable('PCPXX', 'f4', ('time',),fill_value=-9999)
    SD = root_grp.createVariable('SD', 'f4', ('time',),fill_value=-9)
    

    time.calendar = "gregorian"
    # time.units = "hours since 1960-1-1 00:00:00"
    time.standard_name = "time"
    time.long_name = "time"
    time.axis = "T"
    
    # lon.standard_name = "longitude"
    # lon.long_name = "longitude"
    # lon.units = "degrees_east"
    
    # lat.standard_name = "latitude"
    # lat.long_name = "latitude"
    # lat.units = "degrees_north"

    USAF[:]=rUSAF
    WBAN[:]=rWBAN
    YRMODAHRMN[:]=rYR_MODAHRMN
    DIR[:]=rDIR
    SPD[:]=rSPD
    GUS[:]=rGUS
    CLG[:]=rCLG
    SKC[:]=rSKC
    L[:]=rL
    M[:]=rM
    H[:]=rH
    VSBq[:]=rVSBq
    MW1[:]=rMW1
    MW2[:]=rMW2
    MW3[:]=rMW3
    MW4[:]=rMW4
    AW1[:]=rAW1
    AW2[:]=rAW2
    AW3[:]=rAW3
    AW4[:]=rAW4
    W[:]=rW
    TEMP[:]=rTEMP
    DEWP[:]=rDEWP
    SLP[:]=rSLP
    ALT[:]=rALT
    STP[:]=rSTP
    MAX[:]=rMAX
    MIN[:]=rMIN
    PCP01[:]=rPCP01
    PCP06[:]=rPCP06
    PCP24[:]=rPCP24
    PCPXX[:]=rPCPXX
    SD[:]=rSD
    root_grp.close()
    
    # # compress the netcdf file
    # subprocess.Popen("nccopy -k 4 -d 1 -s "+sFile+' '+sFileFin, shell=True)
    # import time
    # time.sleep(10)
    # subprocess.Popen("rm  "+sFile, shell=True)

    # # read the data for testing the file
    # ncid=Dataset(sFileFin, mode='r')
    # rgiTimeTEST=np.squeeze(ncid.variables['YRMODAHRMN'][:])
    # ncid.close()

stop()
