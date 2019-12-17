#!/usr/bin/env python
'''
    File name: Indices_from_IGRA-soundings.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 21.05.2017
    Date last modified: 21.05.2017

    ##############################################################
    Purpos:

    Read in the 

    Then we calculate the parameters that are important for hail
    environments and store them in a common format.

    The goal is to assess if there are trends in hail environments,
    which will be evaluated and displayed in a seperate program.

    inputs:

    returns:

'''

########################################
#                            Load Modules
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
# from mpl_toolkits import basemap
# import ESMF
import pickle
import subprocess
import pandas as pd
import copy
import linecache
from scipy.interpolate import interp1d
from thermodynamics_p3 import VaporPressure, MixRatio, WetBulb, esat, MixR2VaporPress
import pylab as plt
from HelperFunctions import fnSolidPrecip
from scipy import stats
import wrf
from StatisticFunktions_P3 import fnPhysicalBounds
from StatisticFunktions_P3 import fnHomogenTesting
# import ruptures package form time series homogenity testing  !!! PYTHON3
import sys
import ruptures as rpt

########################################
#                            Define setup options here
StartTime=datetime.datetime(1979, 0o1, 0o1)  #Y-M-D --> days are smallest possible period
# StopTime=datetime.datetime(1980, 12, 31,12)
StopTime=datetime.datetime(2017, 12, 31,12)
rgdTime = pd.date_range(StartTime, end=StopTime, freq='12h')
rgdTimeFin = pd.date_range(StartTime, end=StopTime, freq='d')
rgdTimeMonth = pd.date_range(StartTime, end=StopTime, freq='m')
rgiYears=np.unique(rgdTimeFin.year)

rMinCov=0.8 # at least X % of the yearly files has to be available

sROdata='/glade/work/prein/observations/RadioSoundings_IGRA/'
sSaveDataDir='/glade/work/prein/papers/Trends_RadSoundings/data/IGRA/'
rgsSondVars=['P','Z',"T","RH","DPD",'WDIR','WSPD']
########################################

#--------------------------------------------------------------------------------------------------
#                           Get the Station Numbers

# read in IGRA station list
sStatFile='/glade/work/prein/papers/Trends_RadSoundings/data/IGRA/StationList/igra2-station-list.txt'
# rgrSationData=np.genfromtxt(sStatFile, delimiter=',')

datafile = open(sStatFile)
ii=0
for i, line in enumerate(datafile,1):
    line=linecache.getline(sStatFile, ii)
    if ii == 1:
        rgsIGRAststions=np.array(line.split(','))[:,None]
    if ii >1:
        try:
            rgsIGRAststions=np.append(rgsIGRAststions, np.array(line.split(','))[:,None], axis=1)
        except:
            continue
    ii=ii+1

rgsStatID=np.array([rgsIGRAststions[0,ii][:] for ii in range(rgsIGRAststions.shape[1])])
rgsStatName=np.array([rgsIGRAststions[4,ii][:] for ii in range(rgsIGRAststions.shape[1])])
rgiStatLat=np.array(rgsIGRAststions[1,:]).astype('float')
rgiStatLon=np.array(rgsIGRAststions[2,:]).astype('float')
rgiStatAlt=np.array(rgsIGRAststions[3,:]).astype('float')
rgiStatStartY=np.array(rgsIGRAststions[5,:]).astype('float')
rgiStatStopY=np.array(rgsIGRAststions[6,:]).astype('float')

# preselect stations according to data availability and time coverage
iMinYears=int(float(len(rgiYears))*rMinCov); diffYears=len(rgiYears)-iMinYears
rgiValidTimePer=np.array([((rgiStatStartY[ii]-diffYears <= rgiYears[0]) & (rgiStatStopY[ii]+diffYears >= rgiYears[-1]) & (rgiStatStopY[ii]-rgiStatStartY[ii] >= iMinYears)) for ii in range(len(rgiStatStartY))])

rgsStatID=rgsStatID[rgiValidTimePer]
rgsStatName=rgsStatName[rgiValidTimePer]
rgiStatLat=rgiStatLat[rgiValidTimePer]
rgiStatLon=rgiStatLon[rgiValidTimePer]
rgiStatAlt=rgiStatAlt[rgiValidTimePer]
rgiStatStartY=rgiStatStartY[rgiValidTimePer]
rgiStatStopY=rgiStatStopY[rgiValidTimePer]

# =========================================================
#           LOAD THE DATA WITH LONG RECORDS

rgrFreezingLevel=np.zeros((len(rgsStatID),len(rgdTime))); rgrFreezingLevel[:]=np.nan
rgr3CLevel=np.copy(rgrFreezingLevel)
rgrWBZ=np.copy(rgrFreezingLevel)
rgrSnowProb=np.copy(rgrFreezingLevel)
rgrLRbF=np.copy(rgrFreezingLevel)
rgrTS=np.copy(rgrFreezingLevel)
rgrCAPE=np.copy(rgrFreezingLevel)
rgrCIN=np.copy(rgrFreezingLevel)
rgrLCL=np.copy(rgrFreezingLevel)
rgrLFC=np.copy(rgrFreezingLevel)
rgrCBT=np.copy(rgrFreezingLevel)
rgrSH03=np.copy(rgrFreezingLevel)
rgrSH06=np.copy(rgrFreezingLevel)

# FLAGS
rgsVariablesHT=['WBZheight','ThreeCheight','ProbSnow','LRbF','VS0_3','VS0_6','CAPE','CIN','LCL','LFC','CBT']
MissingDataFlag=np.zeros((len(rgsStatID),len(rgsVariablesHT))); MissingDataFlag[:]=np.nan # is set to one if not enough years for trend calculation
BreakFlag=np.zeros((len(rgiYears),len(rgsStatID),len(rgsVariablesHT))); BreakFlag[:]=np.nan  # is set to one if more than 50% of test show same breakpoint

for st in range(len(rgsStatID)):
    print('Station: '+sROdata+rgsStatID[st]+'-data.txt')
    sFileName=sROdata+rgsStatID[st]+'-data.txt'
    SSaveName=sSaveDataDir+rgsStatID[st]+'-data'
    if os.path.isfile(SSaveName+'.npz') == 0:
        if os.path.isfile(sFileName) == 0:
            continue
        with open(sFileName) as f:
            content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        # find soundings within time period
        rgbSoundingsAct=np.array([rgsStatID[st] in content[ii] for ii in range(len(content))])
        rgiSoundingsAct=np.array(list(range(len(content))))[rgbSoundingsAct]
        for ss in range(len(rgiSoundingsAct)):
            rgsHeader=content[rgiSoundingsAct[ss]].split(' ')
            try:
                sTimeStamp=datetime.datetime(int(rgsHeader[1]),int(rgsHeader[2]),int(rgsHeader[3]),int(rgsHeader[4]))
            except:
                continue
            if sTimeStamp not in rgdTime:
                continue
            else:
                print('    processing '+str(sTimeStamp))
                iTT=np.where((sTimeStamp.year == rgdTime.year) & (sTimeStamp.month == rgdTime.month) &(sTimeStamp.day == rgdTime.day) &(sTimeStamp.hour == rgdTime.hour))[0][0]
                # read the sounding data
                if ss != len(rgiSoundingsAct)-1:
                    try:
                        rgrData_act=np.zeros((len(rgsSondVars),rgiSoundingsAct[ss+1]-(rgiSoundingsAct[ss]+1))); rgrData_act[:]=np.nan
                    except:
                        stop()
                else:
                    rgrData_act=np.zeros((len(rgsSondVars),len(rgbSoundingsAct)-1-(rgiSoundingsAct[ss]+1))); rgrData_act[:]=np.nan
                for ii in range(rgrData_act.shape[1]):
                    jj=rgiSoundingsAct[ss]+ii+1
                    if content[jj][15:16] != 'A':
                        try:
                            rgrData_act[rgsSondVars.index('P'),ii]=float(content[jj][9:15])
                        except:
                            stop()
                    if content[jj][21:22] != 'A':
                        rgrData_act[rgsSondVars.index('Z'),ii]=float(content[jj][16:21])
                    if content[jj][27:28] != 'A':
                        rgrData_act[rgsSondVars.index('T'),ii]=float(content[jj][22:27])/10.
                    rgrData_act[rgsSondVars.index('RH'),ii]=float(content[jj][28:33])
                    rgrData_act[rgsSondVars.index('DPD'),ii]=float(content[jj][34:39])/10.
                    rgrData_act[rgsSondVars.index('WDIR'),ii]=float(content[jj][40:45])
                    rgrData_act[rgsSondVars.index('WSPD'),ii]=float(content[jj][46:51])/10.
                rgrData_act[rgrData_act < -777] = np.nan
                rgiReal=~np.isnan(np.mean(rgrData_act[[0,1,2,4],:],axis=0))
                rgrData_act=np.array(rgrData_act[:,rgiReal])
                # we want at least 7 measurements bellow 300 hPa
                if np.sum(rgrData_act[rgsSondVars.index('P'),:] > 30000) <7:
                    continue
                else:
                    if np.nanmin(rgrData_act[rgsSondVars.index('T'),:]) > 0:
                        continue
                    # get surface temperature
                    if rgrData_act[rgsSondVars.index('Z'),0] == 0:
                        rgrTS[st,iTT]=rgrData_act[rgsSondVars.index('T'),0]
                    # rgrTS[st,iTT]=rgrData_act[rgsSondVars.index('T'),:]
                    # calculate freezing level height and 3C height
                    if rgrData_act[rgsSondVars.index('T'),0] <= 0:
                        rgrFreezingLevel[st,iTT]=0
                    else:
                        f = interp1d(rgrData_act[rgsSondVars.index('T'),:], rgrData_act[rgsSondVars.index('Z'),:])
                        rgrFreezingLevel[st,iTT]=f(0)
                    if rgrData_act[rgsSondVars.index('T'),0] <= 3:
                        rgr3CLevel[st,iTT]=0
                    else:
                        f = interp1d(rgrData_act[rgsSondVars.index('T'),:], rgrData_act[rgsSondVars.index('Z'),:])
                        rgr3CLevel[st,iTT]=f(3)
                    # calculate melting level height
                    ESAT=esat(rgrData_act[rgsSondVars.index('T'),:]+273.15)
                    ee=VaporPressure(rgrData_act[rgsSondVars.index('T'),:]-rgrData_act[rgsSondVars.index('DPD'),:],phase="liquid")
                    rgrMixRatio=MixRatio(ee,rgrData_act[rgsSondVars.index('P'),:])
                    VapPres=MixR2VaporPress(rgrMixRatio,rgrData_act[rgsSondVars.index('P'),:])
                    rgrRH=(VapPres/ESAT)*100.
                    rgrWBT=WetBulb(rgrData_act[rgsSondVars.index('T'),:],rgrRH)
                    if rgrWBT[0] <= 0:
                        rgrWBZ[st,iTT]=0
                    else:
                        f = interp1d(rgrWBT, rgrData_act[rgsSondVars.index('Z'),:])
                        rgrWBZ[st,iTT]=f(0)
                    # rSolPrecProb=fnSolidPrecip(1,-5,0)
                    try:
                        iZeroH=np.where(rgrData_act[rgsSondVars.index('Z'),:] == 0)[0][0]
                    except:
                        continue
                    rgi500=np.where(rgrData_act[rgsSondVars.index('Z'),:] <=500)[0]
                    if len(rgi500) >= 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(rgrData_act[rgsSondVars.index('Z'),:][rgi500],rgrData_act[rgsSondVars.index('T'),:][rgi500])
                        rgrSnowProb[st,iTT]=fnSolidPrecip(rgrData_act[rgsSondVars.index('T'),iZeroH],rgrWBT[iZeroH],slope*1000.)
                    else:
                        rgrSnowProb[st,iTT]=np.nan
                    # calculate average lapse rate below frezing level
                    if np.sum(~np.isnan(rgrData_act[rgsSondVars.index('T'),iZeroH])) > 0:
                        iPosTemp=np.where(rgrData_act[rgsSondVars.index('T'),iZeroH:] >= 0)[0]
                        if (len(iPosTemp) >3) & (np.nanmin(rgrData_act[rgsSondVars.index('T'),iZeroH:]) < 0) & (rgrData_act[rgsSondVars.index('T'),iZeroH] > 0):

                            # slope, intercept, r_value, p_value, std_err = stats.linregress(rgrData_act[rgsSondVars.index('Z'),iZeroH:][:iPosTemp[-1]+1],rgrData_act[rgsSondVars.index('T'),iZeroH:][:iPosTemp[-1]+1])
                            # rgrLRbF[st,iTT]=slope*1000.
                            f = interp1d(rgrData_act[rgsSondVars.index('T'),iZeroH:][iPosTemp[-1]-1:iPosTemp[-1]+2], rgrData_act[rgsSondVars.index('Z'),iZeroH:][iPosTemp[-1]-1:iPosTemp[-1]+2])
                            try:
                                ZCheight=f(0)
                            except:
                                continue
                            rgrLRbF[st,iTT]=-rgrData_act[rgsSondVars.index('T'),iZeroH]/(ZCheight/1000.)
                    else:
                        rgrLRbF[st,iTT]=0
                    # calculate CAPE, CIN, LCL, LFC
                    rgiFin=~np.isnan(rgrData_act[rgsSondVars.index('Z'),:])
                    try:
                        i0H=np.where(rgrData_act[rgsSondVars.index('Z'),iZeroH:][rgiFin] == 0)[0][0]
                    except:
                        continue
                    if np.min(rgrData_act[rgsSondVars.index('P'),:][rgiFin]) < 15000:
                        try:
                            rgiFin[np.min(rgrData_act[rgsSondVars.index('P'),:][rgiFin]) < 150] = False
                        except:
                            continue
                    if np.sum(rgiFin) < 10:
                        continue
                    if (np.sum((rgrData_act[rgsSondVars.index('Z'),:][rgiFin] ==0)) == 1) & (np.min(rgrData_act[rgsSondVars.index('P'),:][rgiFin]) <= 30000) & (np.sum(rgiFin[i0H:]) > 10) & (np.min(rgrData_act[rgsSondVars.index('Z'),i0H:][rgiFin][1:]-rgrData_act[rgsSondVars.index('Z'),i0H:][rgiFin][:-1]) > 0):
                        if np.nanmax(rgrData_act[rgsSondVars.index('T'),:][rgiFin][i0H:]) < 50:
                            if np.sum(rgiFin) > 100:
                                # try to avoide core dumps
                                kk=int(np.ceil(np.sum(rgiFin)/100.))
                                CAPE, CIN, LCL, LFC=np.array(wrf.cape_2d(rgrData_act[rgsSondVars.index('P'),:][rgiFin][rgiFin][i0H:][::kk]/100., rgrData_act[rgsSondVars.index('T'),:][rgiFin][i0H:][::kk]+273.15, rgrMixRatio[rgiFin][i0H:][::kk], rgrData_act[rgsSondVars.index('Z'),:][rgiFin][i0H:][::kk], 0., rgrData_act[rgsSondVars.index('P'),:][rgiFin][rgiFin][i0H]/100, 0, meta=True))
                            else:
                                CAPE, CIN, LCL, LFC=np.array(wrf.cape_2d(rgrData_act[rgsSondVars.index('P'),:][rgiFin][rgiFin][i0H:]/100., rgrData_act[rgsSondVars.index('T'),:][rgiFin][i0H:]+273.15, rgrMixRatio[rgiFin][i0H:], rgrData_act[rgsSondVars.index('Z'),:][rgiFin][i0H:], 0., rgrData_act[rgsSondVars.index('P'),:][rgiFin][rgiFin][i0H]/100., 0, meta=True))
                            if (np.isnan(CAPE) == 1) & (np.isnan(LCL) == 0):
                                CAPE=0
                            rgrCAPE[st,iTT]=CAPE
                            rgrCIN[st,iTT]=CIN
                            rgrLCL[st,iTT]=LCL
                            rgrLFC[st,iTT]=LFC
                            # calculate Cloud Base Temperature (CBT)
                            f = interp1d(rgrData_act[rgsSondVars.index('Z'),:][rgiFin][i0H:],rgrData_act[rgsSondVars.index('T'),:][rgiFin][i0H:])
                            try:
                                CBT=f(LCL)
                            except:
                                continue
                            rgrCBT[st,iTT]=CBT
                    # 3 and 6 km vector shear
                    Ugeo = -rgrData_act[rgsSondVars.index('WSPD'),:] * np.sin(rgrData_act[rgsSondVars.index('WDIR'),:] * np.pi/180.)
                    Vgeo = -rgrData_act[rgsSondVars.index('WSPD'),:] * np.cos(rgrData_act[rgsSondVars.index('WDIR'),:] * np.pi/180.)
                    rgiFin=~np.isnan(Ugeo)
                    if np.sum(rgiFin) > 2:
                        HEIGHT=rgrData_act[rgsSondVars.index('Z'),:][rgiFin]
                        fu = interp1d(HEIGHT, Ugeo[rgiFin])
                        fv = interp1d(HEIGHT, Vgeo[rgiFin])
                        if (np.sum(HEIGHT ==0) ==1) &  (np.min(np.abs(HEIGHT - 3000)) <= 500) & (np.max(HEIGHT) > 3000):
                            try:
                                rgrSH03[st,iTT]=((fu(0)-fv(0))**2+(fu(3000)-fv(3000))**2)
                            except:
                                stop()
                        if (np.sum(HEIGHT ==0) ==1) &  (np.min(np.abs(HEIGHT - 6000)) <= 500) & (np.max(HEIGHT) > 6000):
                            rgrSH06[st,iTT]=((fu(0)-fv(0))**2+(fu(6000)-fv(6000))**2)

        # ============================
        # PERFORM HOMOGENITY TEST
        rgrHailPara=np.zeros((len(rgdTime), len(rgsVariablesHT))); rgrHailPara[:]=np.nan
        rgrHailPara[:,rgsVariablesHT.index('WBZheight')]=rgrWBZ[st,:]
        rgrHailPara[:,rgsVariablesHT.index('ThreeCheight')]=rgr3CLevel[st,:]
        rgrHailPara[:,rgsVariablesHT.index('ProbSnow')]=rgrSnowProb[st,:]
        rgrHailPara[:,rgsVariablesHT.index('LRbF')]=rgrLRbF[st,:]
        rgrHailPara[:,rgsVariablesHT.index('VS0_3')]=rgrSH03[st,:]
        rgrHailPara[:,rgsVariablesHT.index('VS0_6')]=rgrSH06[st,:]
        rgrHailPara[:,rgsVariablesHT.index('CAPE')]=rgrCAPE[st,:]
        rgrHailPara[:,rgsVariablesHT.index('CIN')]=rgrCIN[st,:]
        rgrHailPara[:,rgsVariablesHT.index('LCL')]=rgrLCL[st,:]
        rgrHailPara[:,rgsVariablesHT.index('LFC')]=rgrLFC[st,:]
        rgrHailPara[:,rgsVariablesHT.index('CBT')]=rgrCBT[st,:]

        DayAverage=np.reshape(rgrHailPara, (int(rgrHailPara.shape[0]/2),2,rgrHailPara.shape[1]))
        DayAverage=fnPhysicalBounds(DayAverage, rgsVariablesHT)
        DayAverage=np.mean(DayAverage, axis=1)
        # calculate annual average
        YYYY=np.zeros((len(rgiYears),DayAverage.shape[1])); YYYY[:]=np.nan
        MinCov=0.8 # % of days per year
        for yy in range(len(rgiYears)):
            DD=rgdTimeFin.year == rgiYears[yy]
            for va in range(DayAverage.shape[1]):
                if np.sum(~np.isnan(DayAverage[DD,va]))/np.sum(DD) >= MinCov:
                    YYYY[yy,va]=np.nanmean(DayAverage[DD,va])

        for va in range(DayAverage.shape[1]):
            # Take care of NaNs
            if np.sum(~np.isnan(YYYY[:,va])/YYYY.shape[0]) >= MinCov:
                if np.sum(~np.isnan(YYYY[:,va])/YYYY.shape[0]) != 1:
                    ss = pd.Series(YYYY[:,va])
                    TMP_DATA=ss.interpolate().values
                    ss = pd.Series(TMP_DATA[::-1])
                    TMP_DATA=ss.interpolate().values
                    TMP_DATA=TMP_DATA[::-1]
                    YYYY[:,va]=TMP_DATA
                Breaks = fnHomogenTesting(YYYY[:,va])
                BreakFlag[:,st,va]=Breaks
                MissingDataFlag[st,va]=0
            else:
                MissingDataFlag[st,va]=1

        # save the data for this station
        np.savez(SSaveName+'.npz', 
                 Snow=rgrSnowProb[st,:], 
                 WBZ=rgrWBZ[st,:], 
                 ThreeC=rgr3CLevel[st,:], 
                 FLH=rgrFreezingLevel[st,:],
                 TS=rgrTS[st,:],
                 LRbF=rgrLRbF[st,:],
                 CAPE=rgrCAPE[st,:],
                 CIN=rgrCIN[st,:],
                 LCL=rgrLCL[st,:],
                 LFC=rgrLFC[st,:],
                 CBT=rgrCBT[st,:],
                 SH03=rgrSH03[st,:],
                 SH06=rgrSH06[st,:],
                 Breaks=BreakFlag[:,st,:],
                 MissingData=MissingDataFlag[st,:],
                 VariablesFlags=rgsVariablesHT)
    else:
        # load the station data
        npzfile = np.load(SSaveName+'.npz')
        rgrSnowProb[st,:]=npzfile['Snow']
        rgrWBZ[st,:]=npzfile['WBZ']
        rgr3CLevel[st,:]=npzfile['ThreeC']
        rgrFreezingLevel[st,:]=npzfile['FLH']
        rgrTS[st,:]=npzfile['TS']
        rgrLRbF[st,:]=npzfile['LRbF']
        rgrCAPE[st,:]=npzfile['CAPE']
        rgrCIN[st,:]=npzfile['CIN']
        rgrLCL[st,:]=npzfile['LCL']
        rgrLFC[st,:]=npzfile['LFC']
        rgrCBT[st,:]=npzfile['CBT']
        rgrSH03[st,:]=npzfile['SH03']
        rgrSH06[st,:]=npzfile['SH06']
        BreakFlag[:,st,:]=npzfile['Breaks']
        MissingDataFlag[st,:]=npzfile['MissingData']

# convert to daily data by averaging over the 12 and 0 UTC sounding
rgrSnowProb=np.nanmean(np.reshape(rgrSnowProb, (rgrSnowProb.shape[0],int(rgrSnowProb.shape[1]/2),2)), axis=2)
rgrWBZ=np.nanmean(np.reshape(rgrWBZ, (rgrWBZ.shape[0],int(rgrWBZ.shape[1]/2),2)), axis=2)
rgr3CLevel=np.nanmean(np.reshape(rgr3CLevel, (rgr3CLevel.shape[0],int(rgr3CLevel.shape[1]/2),2)), axis=2)
rgrFreezingLevel=np.nanmean(np.reshape(rgrFreezingLevel, (rgrFreezingLevel.shape[0],int(rgrFreezingLevel.shape[1]/2),2)), axis=2)
rgrTS=np.nanmean(np.reshape(rgrTS, (rgrTS.shape[0],int(rgrTS.shape[1]/2),2)), axis=2)
rgrLRbF=np.nanmean(np.reshape(rgrLRbF, (rgrLRbF.shape[0],int(rgrLRbF.shape[1]/2),2)), axis=2)
rgrCAPE=np.nanmax(np.reshape(rgrCAPE, (rgrCAPE.shape[0],int(rgrCAPE.shape[1]/2),2)), axis=2)
rgrCIN=np.nanmax(np.reshape(rgrCIN, (rgrCIN.shape[0],int(rgrCIN.shape[1]/2),2)), axis=2)
rgrLCL=np.nanmean(np.reshape(rgrLCL, (rgrLCL.shape[0],int(rgrLCL.shape[1]/2),2)), axis=2)
rgrLFC=np.nanmean(np.reshape(rgrLFC, (rgrLFC.shape[0],int(rgrLFC.shape[1]/2),2)), axis=2)
rgrCBT=np.nanmean(np.reshape(rgrCBT, (rgrCBT.shape[0],int(rgrCBT.shape[1]/2),2)), axis=2)
rgrSH03=np.nanmean(np.reshape(rgrSH03, (rgrSH03.shape[0],int(rgrSH03.shape[1]/2),2)), axis=2)
rgrSH06=np.nanmean(np.reshape(rgrSH06, (rgrSH06.shape[0],int(rgrSH06.shape[1]/2),2)), axis=2)


# save the 6-hourly data
np.savez(sSaveDataDir+'All_IGRA-Data_daily_1979-2017.npz',
         rgrSnowProb=rgrSnowProb,
         rgrWBZ=rgrWBZ,
         rgr3CLevel=rgr3CLevel,
         rgrFreezingLevel=rgrFreezingLevel,
         rgrTS=rgrTS,
         rgrLRbF=rgrLRbF,
         rgrCAPE=rgrCAPE,
         rgrCIN=rgrCIN,
         rgrLCL=rgrLCL,
         rgrLFC=rgrLFC,
         rgrCBT=rgrCBT,
         rgrSH03=rgrSH03,
         rgrSH06=rgrSH06,
         BreakFlag=BreakFlag,
         MissingDataFlag=MissingDataFlag,
         rgdTimeFin=rgdTimeFin,
         rgiStatLat=rgiStatLat,
         rgiStatLon=rgiStatLon)

stop()

# SAVE THE DAILY SSP > 20000 ON DAYS WITH wbzh > 45000
rgrSPP_I=np.array((rgrSH06*rgrCAPE >= 20000) & (rgrWBZ >= 4500)).astype('float')
rgrSPP_I[np.isnan(rgrSH06*rgrCAPE*rgrWBZ)]=np.nan
np.savez('/glade/u/home/prein/papers/Trends_RadSoundings/programs/RadioSoundings/IGRA_Preprocessor/SSP_daily.npz',rgrSPP_I=rgrSPP_I, rbgrLon=rgiStatLon, rbgrLat=rgiStatLat, rgdTimeFin=rgdTimeFin)

# SAVE THE DAILY WBZ DATA
rgrSPP_I=np.array((rgrSH06*rgrCAPE >= 20000) & (rgrWBZ >= 4500)).astype('float')
rgrSPP_I[np.isnan(rgrSH06*rgrCAPE*rgrWBZ)]=np.nan
np.savez('/glade/u/home/prein/papers/Trends_RadSoundings/programs/RadioSoundings/IGRA_Preprocessor/WBCD_daily.npz',rgrWBZ=rgrWBZ, rbgrLon=rgiStatLon, rbgrLat=rgiStatLat, rgdTimeFin=rgdTimeFin, BreakFlag=BreakFlag[:,:,rgsVariablesHT.index('WBZheight')])


# # ===============================================
# # ===============================================
# #       READ IN THE MSWAP PRECIPITATION DATA AND
# #       CONDITION THE SNOWFALL PROBABILITY AND
# #       CALCULATE RAINFALL FREQUENCY

dStartDay_MSWEP=datetime.datetime(1979, 1, 1,0)
dStopDay_MSWEP=datetime.datetime(2016, 12, 31,0)
rgdTimeDD_MSWEP = pd.date_range(dStartDay_MSWEP, end=dStopDay_MSWEP, freq='d')

# # load grid from MSWAP
# sMSWEP_Grid='/glade/scratch/prein/MSWEP_V2.1/data/197901.nc'
# ncid=Dataset(sMSWEP_Grid, mode='r')
# rgrLat_MSWEP=np.squeeze(ncid.variables['lat'][:])
# rgrLon_MSWEP=np.squeeze(ncid.variables['lon'][:]) #; rgrLon_MSWEP[rgrLon_MSWEP<0]=rgrLon_MSWEP[rgrLon_MSWEP<0]+360
# ncid.close()
# rgiLonMin_MSWEP=np.array([np.argmin(np.abs(rgiStatLon[ii] - rgrLon_MSWEP)) for ii in range(len(rgiStatLon))])
# rgiLatMin_MSWEP=np.array([np.argmin(np.abs(rgiStatLat[ii] - rgrLat_MSWEP)) for ii in range(len(rgiStatLat))])
# YYYY_MSWEP=np.unique(rgdTimeDD_MSWEP.year)
# rgrMSWEP_PR=np.zeros((len(rgdTimeDD_MSWEP), len(rgiStatLon)))
# for yy in range(len(YYYY_MSWEP)):
#     print '    Process Year '+str(YYYY_MSWEP[yy])
#     for mm in range(12):
#         sFileName='/glade/scratch/prein/MSWEP_V2.1/data/'+str(YYYY_MSWEP[yy])+str("%02d" % (mm+1))+'.nc'
#         rgiTimeAct=((rgdTimeDD_MSWEP.year == YYYY_MSWEP[yy]) & (rgdTimeDD_MSWEP.month == (mm+1)))
#         ncid=Dataset(sFileName, mode='r')
#         for st in range(len(rgiStatLon)):
#             rgrDATAtmp=ncid.variables['precipitation'][:,rgiLatMin_MSWEP[st],rgiLonMin_MSWEP[st]]
#             rgrMSWEP_PR[rgiTimeAct,st]=np.sum(np.reshape(rgrDATAtmp,(len(rgrDATAtmp)/8,8)), axis=1)
#         ncid.close()
# np.savez('RS_MSWEP_PR.npz', rgrMSWEP_PR=rgrMSWEP_PR)
rgrMSWEP_PR=np.load('/glade/u/home/prein/papers/Trends_RadSoundings/programs/RadioSoundings/IGRA_Preprocessor/RS_MSWEP_PR.npz')['rgrMSWEP_PR']
rgiMSWEP_PR=np.copy(rgrMSWEP_PR)
rgiMSWEP_PR[(rgrMSWEP_PR < 1)]=0; rgiMSWEP_PR[(rgrMSWEP_PR >= 1)]=1
rgiMSWEP_PR=np.transpose(rgiMSWEP_PR)

# CALCULATE MONTHY AVERAGES WITH THE REQUIREMENT THAT WE HAVE AT LEAST 70 % COVERAGE
rgsVars=['Snow','WBZ','ThreeC','FLH','TS','LRbF','CAPE','CIN','LCL','LFC', 'CBT', 'SH03','SH06','Snow_PRdays', 'Rain_PRdays']
rgrMonIGRA=np.zeros((len(rgiYears),12,rgrWBZ.shape[0],len(rgsVars))); rgrMonIGRA[:]=np.nan
for yy in range(len(rgiYears)):
    print('    Process Year '+str(rgiYears[yy]))
    for mm in range(12):
        rgiTimes=((rgdTimeFin.year == rgiYears[yy]) & (rgdTimeFin.month == (mm+1)))
        for va in range(len(rgsVars)):
            if rgsVars[va] == 'Snow':
                rgrData=rgrSnowProb[:,rgiTimes]
                rgrData[rgrData >= 0.5]=1; rgrData[rgrData < 0.5]= 0
            if rgsVars[va] == 'Snow_PRdays':
                rgiTimePR=((rgdTimeDD_MSWEP.year == rgiYears[yy]) & (rgdTimeDD_MSWEP.month == (mm+1)))
                rgrData=rgrSnowProb[:,rgiTimes]
                try:
                    rgrDataPR=np.array(rgiMSWEP_PR[:,rgiTimePR]); rgrDataPR[np.isnan(rgrData)]=np.nan
                except:
                    continue
                rgrTMP=np.copy(rgrData); rgrTMP[:]=0
                rgrTMP[(rgrData >= 0.5) & (rgrDataPR >= 1)]=1
                rgrTMP[np.isnan(rgrData) == 1]=np.nan
                rgrData=rgrTMP
                # if np.nansum(rgrTMP) > 1:
                #     stop()
                #     np.where(np.nansum(rgrData, axis=1) >0)
                #     plt.plot(rgrTMP[477,:], c='k'); plt.plot(np.array(rgrMSWEP_PR[rgiTimePR,:])[:,477], c='b'); plt.plot(rgrSnowProb[:,rgiTimes][477,:], c='r'); plt.show()
            if rgsVars[va] == 'Rain_PRdays':
                rgiTimePR=((rgdTimeDD_MSWEP.year == rgiYears[yy]) & (rgdTimeDD_MSWEP.month == (mm+1)))
                rgrData=rgrSnowProb[:,rgiTimes]
                try:
                    rgrDataPR=np.array(rgiMSWEP_PR[:,rgiTimePR]); rgrDataPR[np.isnan(rgrData)]=np.nan
                except:
                    continue
                rgrTMP=np.copy(rgrData); rgrTMP[:]=0
                rgrTMP[(rgrData < 0.5) & (rgrDataPR >= 1)]=1
                rgrTMP[np.isnan(rgrData) ==1]=np.nan
                rgrData=rgrTMP
            if rgsVars[va] == 'WBZ':
                rgrData=rgrWBZ[:,rgiTimes]
            if rgsVars[va] == 'ThreeC':
                rgrData=rgr3CLevel[:,rgiTimes]
            if rgsVars[va] == 'FLH':
                rgrData=rgrFreezingLevel[:,rgiTimes]
            if rgsVars[va] == 'TS':
                rgrData=rgrTS[:,rgiTimes]
            if rgsVars[va] == 'LRbF':
                rgrData=rgrLRbF[:,rgiTimes]
            if rgsVars[va] == 'CAPE':
                rgrData=rgrCAPE[:,rgiTimes]
            if rgsVars[va] == 'CIN':
                rgrData=rgrCIN[:,rgiTimes]
            if rgsVars[va] == 'LCL':
                rgrData=rgrLCL[:,rgiTimes]
            if rgsVars[va] == 'LFC':
                rgrData=rgrLFC[:,rgiTimes]
            if rgsVars[va] == 'CBT':
                rgrData=rgrCBT[:,rgiTimes]
            if rgsVars[va] == 'CBT':
                rgrData=rgrCBT[:,rgiTimes]
            if rgsVars[va] == 'SH03':
                rgrData=rgrSH03[:,rgiTimes]
            if rgsVars[va] == 'SH06':
                rgrData=rgrSH06[:,rgiTimes]
            rgiNaNs=np.sum(np.isnan(rgrData), axis=1)/float(sum(rgiTimes))
            rgrMonIGRA[yy,mm,rgiNaNs < 0.3,va]=np.nanmean(rgrData[rgiNaNs < 0.3,:], axis=1)

# save the data
grStatisitcs={'rgrMonIGRA':rgrMonIGRA,
              'rgsVars':rgsVars,
              'rgdTime':rgdTimeMonth,
              'rgiStatLat':rgiStatLat,
              'rgiStatLon':rgiStatLon,
              'rgiStatAlt':rgiStatAlt,
              'rgsStatID':rgsStatID,
              'rgsStatName':rgsStatName,
              'Beaks':BreakFlag,
              'MissinfData':MissingDataFlag}

sTempFileName=sSaveDataDir+'IGRA-Sounding-Data-'+str(StartTime.year)+'-'+str(StopTime.year)+'.pkl'
print('    Save: '+sTempFileName)
fh = open(sTempFileName,"wb")
pickle.dump(grStatisitcs,fh)
fh.close()

stop()
