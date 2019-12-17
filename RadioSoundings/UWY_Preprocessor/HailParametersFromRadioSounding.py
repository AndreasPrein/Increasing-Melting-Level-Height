#!/usr/bin/env python
'''
    File name: HailParametersFromRadioSounding.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 21.05.2017
    Date last modified: 21.05.2017

    ##############################################################
    Purpos:

    We download all radio sounding observations Worldwide from the
    University of Whyoming database:
    http://weather.uwyo.edu/upperair/sounding.html

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
import pickle
import subprocess
import pandas as pd
import copy
import linecache
from scipy.interpolate import interp1d
import pylab as plt
from HelperFunctions import fnSolidPrecip
from StatisticFunktions_P3 import fnPhysicalBounds
from StatisticFunktions_P3 import fnHomogenTesting
from scipy import stats
import wrf
import time
import resource
from scipy import signal

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
rgiYears=np.unique(rgdTimeFin.year)

rMinCov=0.8 # at least X % of the yearly files has to be available

sROdata='/glade/work/prein/observations/RadioSoundings_Uni-Wyoming/data/'
sSaveDataDir='/glade/scratch/prein/Papers/Trends_RadSoundings/data/'
########################################

#--------------------------------------------------------------------------------------------------
#                           Get the Station Numbers

# read in ROAB station list
sROABstatFile='/glade/work/prein/observations/RadioSoundings_Uni-Wyoming/raob.stn.txt'
datafile = open(sROABstatFile)
ii=0
for i, line in enumerate(datafile,1):
    line=linecache.getline(sROABstatFile, ii)
    if ii == 10:
        rgsROABststions=np.array(line.split(','))[:,None]
    if ii >10:
        try:
            rgsROABststions=np.append(rgsROABststions, np.array(line.split(','))[:,None], axis=1)
        except:
            continue
    ii=ii+1

rgsCountries=[rgsROABststions[3,ii][-2:] for ii in range(rgsROABststions.shape[1])]
# rgiUS=(np.array(rgsCountries) == 'US')
# rgiStatNr=rgsROABststions[0,rgiUS]
# rgsStationName=rgsROABststions[1,rgiUS]
rgiStatNr=rgsROABststions[0,:]
rgsStationName=rgsROABststions[1,:]
rgsStationName=[rgsStationName[ii].replace(" ", "") for ii in range(len(rgsStationName))]

# preselect stations according to data availability and time coverage
rgsNetCDFs=glob.glob(sROdata+"*.nc")
rgsStatNrsAll=np.array([rgsNetCDFs[ii].split('/')[-1][:5] for ii in range(len(rgsNetCDFs))])
rgsUniqueStats=np.unique(rgsStatNrsAll)
rgiYeasPerStation=np.array([np.sum(rgsStatNrsAll == rgsUniqueStats[ii]) for ii in range(len(rgsUniqueStats))])
rgiStatNr=rgsUniqueStats[rgiYeasPerStation >= len(rgiYears)*rMinCov]

#--------------------------------------------------------------------------------------------------
#       Derive Radio Sounding data from University of Wyoming
from IO_funktions import fnGetUWSounding

######################################
# derive the hail data for all stations
# rgsVariables=['CAPE','RHbf','RHml','VS0_1','VS0_3','VS0_6','VS6_12','PW','FLH','WBZ']
rgsVariables=['WBZheight','ThreeCheight','ProbSnow','LRbF','VS0_3','VS0_6','CAPE','CIN','LCL','LFC','CBT']
# rgsVariables=['VS0_3','VS0_6','CAPE','CIN','LCL','LFC']

rgrHailParaAll=np.zeros((len(rgdTime), len(rgiStatNr),len(rgsVariables))); rgrHailParaAll[:]=np.nan
rgrLonAll=np.zeros((len(rgiStatNr))); rgrLonAll[:]=np.nan
rgrLatAll=np.copy(rgrLonAll)
rgrElevAll=np.copy(rgrLonAll)
rgsStatIDAll=['']*len(rgiStatNr)
rgsStatNrAll=['']*len(rgiStatNr)
# FLAGS
MissingDataFlag=np.zeros((len(rgiStatNr),len(rgsVariables))); MissingDataFlag[:]=np.nan # is set to one if not enough years for trend calculation
BreakFlag=np.zeros((len(rgiYears),len(rgiStatNr),len(rgsVariables))); BreakFlag[:]=np.nan  # is set to one if more than 50% of test show same breakpoint

for st in range(len(rgiStatNr)):
    sStationSave=sSaveDataDir+'UW-RS-stations/'+rgiStatNr[st]+'.npz'
    if os.path.exists(sStationSave) == 0:
        start_time = time.time()
        print('    process station '+str(st)+' out of '+str(len(rgiStatNr)))
        dcRSobs=fnGetUWSounding(StartTime,
                                StopTime,
                                [rgiStatNr[st]],
                                sROdata)
        rgrHailPara=np.zeros((len(rgdTime),len(rgsVariables))); rgrHailPara[:]=np.nan
        
    
        # derive station methadata
        rgrLon=dcRSobs[list(dcRSobs.keys())[0]][4]['lon']
        rgrLat=dcRSobs[list(dcRSobs.keys())[0]][4]['lat']
        rgrElev=dcRSobs[list(dcRSobs.keys())[0]][4]['elev']
        rgsStatID=dcRSobs[list(dcRSobs.keys())[0]][4]['ID']
        rgsStatNr=list(dcRSobs.keys())[0]

        # derive heights
        rgrDataAct=dcRSobs[list(dcRSobs.keys())[0]][0]
        rgrHeight=np.array(rgrDataAct.major_xs('HGHT [m]'))-dcRSobs[list(dcRSobs.keys())[0]][4]['elev'] # to height above surface
        rgrPRES=np.array(rgrDataAct.major_xs('PRES [hPa]'))
        rgrTEMP=np.array(rgrDataAct.major_xs('TEMP [C]'))
        rgrMR=np.array(rgrDataAct.major_xs('MIXR [g/kg]'))/1000.  # convert to kg/kg
        rgrWD=np.array(rgrDataAct.major_xs('DRCT [deg]'))
        rgrWS=np.array(rgrDataAct.major_xs('SKNT [knot]'))*0.514444 # convert to m/s
        # calculate U and V component
        rgrU = -rgrWS * np.sin(rgrWD * np.pi/180.0)
        rgrV = -rgrWS * np.cos(rgrWD * np.pi/180.0)
    
        # calcualte satturation mixing ratio
        from thermodynamics_p3 import VaporPressure, MixRatio, WetBulb, esat, MixR2VaporPress
        rgrSVP=VaporPressure(rgrTEMP, phase="liquid")
        rgrMRs=MixRatio(rgrSVP,rgrPRES*100.)
        for va in range(len(rgsVariables)):
            print('        '+rgsVariables[va])
            if rgsVariables[va] == 'CAPE':
                for tt in range(len(rgdTime)):
                    # tt=24611
                    # print '            '+str(tt)+' of '+str(len(rgdTime))
                    rgiFin=~np.isnan(rgrHeight[:,tt])
                    try:
                        i0H=np.where((rgrHeight[:,tt][rgiFin]==0))[0][0]
                    except:
                        continue
                    if np.min(rgrPRES[:,tt][rgiFin]) < 150:
                        try:
                            rgiFin[rgrPRES[:,tt] < 150] = False
                        except:
                            continue
                    if np.sum(rgiFin) < 10:
                        continue
                    if (np.sum((rgrHeight[:,tt][rgiFin]==0)) == 1) & (np.min(rgrPRES[:,tt][rgiFin]) < 250) & (np.sum(rgiFin[i0H:]) > 10) & (np.min(rgrHeight[:,tt][rgiFin][1:]-rgrHeight[:,tt][rgiFin][:-1]) > 0):
                        if np.nanmax(rgrTEMP[:,tt][rgiFin][i0H:]) < 50:
                            # for kk in range(-50,0,1):
                            #     print kk
                            #     print rgrPRES[:,tt][rgiFin][i0H:kk]
                            #     print rgrTEMP[:,tt][rgiFin][i0H:kk]
                            #     print rgrMR[:,tt][rgiFin][i0H:kk]
                            #     print rgrHeight[:,tt][rgiFin][i0H:kk]
                            #     print ''
                            #     wrf.cape_2d(rgrPRES[:,tt][rgiFin][i0H:kk], rgrTEMP[:,tt][rgiFin][i0H:kk]+273.15, rgrMR[:,tt][rgiFin][i0H:kk], rgrHeight[:,tt][rgiFin][i0H:kk], rgrElev[st], rgrPRES[:,tt][rgiFin][i0H], 0, meta=False)
                            if np.sum(rgiFin) > 100:
                                # try to avoide core dumps
                                kk=int(np.ceil(np.sum(rgiFin)/100.))
                                CAPE, CIN, LCL, LFC=np.array(wrf.cape_2d(rgrPRES[:,tt][rgiFin][i0H:][::kk], rgrTEMP[:,tt][rgiFin][i0H:][::kk]+273.15, rgrMR[:,tt][rgiFin][i0H:][::kk], rgrHeight[:,tt][rgiFin][i0H:][::kk], rgrElev, rgrPRES[:,tt][rgiFin][i0H], 0, meta=True))
                            else:
                                CAPE, CIN, LCL, LFC=np.array(wrf.cape_2d(rgrPRES[:,tt][rgiFin][i0H:], rgrTEMP[:,tt][rgiFin][i0H:]+273.15, rgrMR[:,tt][rgiFin][i0H:], rgrHeight[:,tt][rgiFin][i0H:], rgrElev, rgrPRES[:,tt][rgiFin][i0H], 0, meta=True))
                            if (np.isnan(CAPE) == 1) & (dcRSobs[list(dcRSobs.keys())[0]][2][tt] == 0):
                                CAPE=0
                            # calculate Cloud Base Temperature (CBT)
                            f = interp1d(rgrHeight[:,tt][rgiFin][i0H:],rgrTEMP[:,tt][rgiFin][i0H:])
                            try:
                                CBT=f(LCL)
                            except:
                                continue
                            rgrHailPara[tt,rgsVariables.index('CAPE'):]=[CAPE, CIN, LCL, LFC, CBT]
            if rgsVariables[va] == 'RHbf':
                 for tt in range(len(rgdTime)):
                      if np.sum(~np.isnan(rgrTEMP[:,tt])) > 0:
                           iFL=np.where((np.nanmin(abs(rgrTEMP[:,tt])) == abs(rgrTEMP[:,tt])))[0][0]
                           rgrMRbf=rgrMR[0:iFL,tt]
                           rgrMRsBF=rgrMRs[0:iFL,tt]
                           # wheighted according to height
                           rgrHGTcopy=np.copy(rgrHeight[:,tt])
                           rgrHGTcopy=rgrHGTcopy[1:]-rgrHGTcopy[:-1]
                           rgrHGTcopy=rgrHGTcopy[0:iFL]
                           rgrHGTsum=np.nansum(rgrHGTcopy, axis=0)
                           rgrHailPara[tt,va]=((np.nansum(rgrMRbf*rgrHGTcopy,axis=0)/rgrHGTsum)/(np.nansum(rgrMRsBF*rgrHGTcopy, axis=0)/rgrHGTsum))
            if rgsVariables[va] == 'RHml':
                 for tt in range(len(rgdTime)):
                      if np.sum(~np.isnan(rgrPRES[:,tt])) > 0:
                           i700=np.where((np.nanmin(abs(rgrPRES[:,tt]-700)) == abs(rgrPRES[:,tt]-700)))[0][0]
                           i500=np.where((np.nanmin(abs(rgrPRES[:,tt]-500)) == abs(rgrPRES[:,tt]-500)))[0][0]
                           rgrMRbf=rgrMR[i700:i500,tt]
                           rgrMRsBF=rgrMRs[i700:i500,tt]
                           # wheighted according to height
                           rgrHGTcopy=np.copy(rgrHeight[:,tt])
                           rgrHGTcopy=rgrHGTcopy[1:]-rgrHGTcopy[:-1]
                           rgrHGTcopy=rgrHGTcopy[i700:i500]
                           rgrHGTsum=np.nansum(rgrHGTcopy, axis=0)
                           rgrHailPara[tt,va]=((np.nansum(rgrMRbf*rgrHGTcopy,axis=0)/rgrHGTsum)/(np.nansum(rgrMRsBF*rgrHGTcopy, axis=0)/rgrHGTsum))
            if rgsVariables[va] == 'VS0_1':
                 for tt in range(len(rgdTime)):
                      if np.sum(~np.isnan(rgrHeight[:,tt])) > 0:
                           try:
                                i1000=np.where((np.nanmin(abs(rgrHeight[:,tt]-1000)) == abs(rgrHeight[:,tt]-1000)))[0][0]
                                rgrHailPara[tt,va]=((rgrU[i1000,tt]-rgrU[:,tt][~np.isnan(rgrU[:,tt])][0])**2+\
                                                           (rgrV[i1000,tt]-rgrV[:,tt][~np.isnan(rgrV[:,tt])][0])**2)**0.5
                           except:
                                continue
            if rgsVariables[va] == 'VS0_3':
                 for tt in range(len(rgdTime)):
                      if np.sum(~np.isnan(rgrHeight[:,tt])) > 0:
                           try:
                                i3000=np.where((np.nanmin(abs(rgrHeight[:,tt]-3000)) == abs(rgrHeight[:,tt]-3000)))[0][0]
                                rgrHailPara[tt,va]=((rgrU[i3000,tt]-rgrU[:,tt][~np.isnan(rgrU[:,tt])][0])**2+\
                                                           (rgrV[i3000,tt]-rgrV[:,tt][~np.isnan(rgrV[:,tt])][0])**2)**0.5
                           except:
                                continue
            if rgsVariables[va] == 'VS0_6':
                 for tt in range(len(rgdTime)):
                      if np.sum(~np.isnan(rgrHeight[:,tt])) > 0:
                           try:
                                i6000=np.where((np.nanmin(abs(rgrHeight[:,tt]-6000)) == abs(rgrHeight[:,tt]-6000)))[0][0]
                                rgrHailPara[tt,va]=((rgrU[i6000,tt]-rgrU[:,tt][~np.isnan(rgrU[:,tt])][0])**2+\
                                                           (rgrV[i6000,tt]-rgrV[:,tt][~np.isnan(rgrV[:,tt])][0])**2)**0.5
                           except:
                                continue
            if rgsVariables[va] == 'VS6_12':
                 for tt in range(len(rgdTime)):
                      if np.sum(~np.isnan(rgrHeight[:,tt])) > 0:
                           try:
                                i6000=np.where((np.nanmin(abs(rgrHeight[:,tt]-6000)) == abs(rgrHeight[:,tt]-6000)))[0][0]
                                i12000=np.where((np.nanmin(abs(rgrHeight[:,tt]-12000)) == abs(rgrHeight[:,tt]-12000)))[0][0]
                                rgrHailPara[tt,va]=((rgrU[i12000,tt]-rgrU[i6000,tt])**2+\
                                                           (rgrV[i12000,tt]-rgrV[i6000,tt])**2)**0.5
                           except:
                                continue
            if rgsVariables[va] == 'PW':
                 rgrHailPara[:,st,va]=np.array(dcRSobs[list(dcRSobs.keys())[st]][3])
            if rgsVariables[va] == 'FLH':
                for tt in range(len(rgdTime)):
                    if np.sum(~np.isnan(rgrTEMP[:,tt])) > 0:
                        if (np.nanmax(rgrTEMP[:5,tt]) > 0) & (np.nanmin(rgrTEMP[5:,tt]) < 0):
                            f = interp1d(rgrTEMP[:,tt], rgrHeight[:,tt])
                            try:
                                rgrHailPara[tt,va]=f(0)
                            except:
                                stop()
                        else:
                            rgrHailPara[tt,st,va]=0
            if rgsVariables[va] == 'ThreeCheight':
                for tt in range(len(rgdTime)):
                    if ~np.isnan(rgrTEMP[0,tt]) ==1:
                        if (np.nanmax(rgrTEMP[0,tt]) > 3) & (np.nanmin(rgrTEMP[5:,tt]) < 0):
                            f = interp1d(rgrTEMP[:,tt], rgrHeight[:,tt])
                            try:
                                rgrHailPara[tt,va]=f(3)
                            except:
                                continue
                        else:
                            if (np.nanmax(rgrTEMP[0,tt]) < 3):
                                rgrHailPara[tt,va]=0
            if rgsVariables[va] == 'WBZheight':
                for tt in range(len(rgdTime)):
                    if np.sum(~np.isnan(rgrTEMP[:,tt])) > 0:
                        # calculate web bulb temperature
                        if np.nanmax(rgrTEMP[:,tt]) > 0:
                            ESAT=esat(rgrTEMP[:,tt]+273.15)
                            VapPres=MixR2VaporPress(rgrMR[:,tt],rgrPRES[:,tt]*100)
                            rgrRH=(VapPres/ESAT)*100.
                            rgrWBT=WetBulb(rgrTEMP[:,tt],rgrRH)
                            if np.sum(~np.isnan(rgrWBT)) == 0:
                                continue
                            if (np.nanmax(rgrWBT[0:5]) > 0) & (np.nanmin(rgrWBT[5:]) < 0):
                                f = interp1d(rgrWBT, rgrHeight[:,tt])
                                try:
                                    rgrHailPara[tt,va]=f(0)
                                except:
                                    stop()
                            else:
                                rgrHailPara[tt,va]=0
                        else:
                            rgrHailPara[tt,va]=0
            if rgsVariables[va] == 'ProbSnow':
                for tt in range(len(rgdTime)):
                    ESAT=esat(rgrTEMP[:,tt]+273.15)
                    VapPres=MixR2VaporPress(rgrMR[:,tt],rgrPRES[:,tt]*100)
                    rgrRH=(VapPres/ESAT)*100.
                    rgrWBT=WetBulb(rgrTEMP[:,tt],rgrRH)
                    try:
                        iZeroH=np.where(rgrHeight[:,tt] == 0)[0][0]
                    except:
                        continue
                    rgi500=np.where(rgrHeight[iZeroH:,tt] <=500)[0]
                    if len(np.unique(rgrHeight[iZeroH:,tt][rgi500])) < 2:
                        continue
                    slope, intercept, r_value, p_value, std_err = stats.linregress(rgrHeight[iZeroH:,tt][rgi500],rgrTEMP[iZeroH:,tt][rgi500])
                    if (~np.isnan(slope) == 0) | (~np.isnan(rgrWBT[iZeroH]) == 0):
                        continue
                    try:
                        rgrHailPara[tt,va]=fnSolidPrecip(rgrTEMP[iZeroH,tt],rgrWBT[iZeroH],slope*1000.)
                    except:
                        stop()
            if rgsVariables[va] == 'LRbF':
                for tt in range(len(rgdTime)):
                    try:
                        iZeroH=np.where(rgrHeight[:,tt] == 0)[0][0]
                    except:
                        continue
                    if np.sum(~np.isnan(rgrTEMP[iZeroH,tt])) > 0:
                        iPosTemp=np.where(rgrTEMP[iZeroH:,tt] > 0)[0]
                        if (len(iPosTemp) > 3) & (np.nanmin(rgrTEMP[iZeroH:,tt]) < 0) & (rgrTEMP[iZeroH,tt] > 0):
                            # slope, intercept, r_value, p_value, std_err = stats.linregress(rgrHeight[iZeroH:,tt][:iPosTemp[-1]+1],rgrTEMP[iZeroH:,tt][:iPosTemp[-1]+1])
                            # rgrHailPara[tt,va]=slope*1000.
                            # zero C height
                            f = interp1d(rgrTEMP[iZeroH:,tt][iPosTemp[-1]-1:iPosTemp[-1]+2], rgrHeight[iZeroH:,tt][iPosTemp[-1]-1:iPosTemp[-1]+2])
                            try:
                                ZCheight=f(0)
                            except:
                                continue
                            # TempAct=rgrTEMP[iZeroH:,tt][:iPosTemp[-1]+2]; TempAct[-1]=0
                            # HeightAct=rgrHeight[iZeroH:,tt][:iPosTemp[-1]+2]; HeightAct[-1]=ZCheight
                            # LRact=(TempAct[1:]-TempAct[:-1])/((HeightAct[1:]-HeightAct[:-1])/1000.)
                            # rgrHailPara[tt,va]=np.average(LRact, weights=(HeightAct[1:]-HeightAct[:-1]))
                            rgrHailPara[tt,va]=-rgrTEMP[iZeroH,tt]/(ZCheight/1000.)
                    else:
                        rgrHailPara[tt,va]=0

        # ============================
        # PERFORM HOMOGENITY TEST
        DayAverage=np.reshape(rgrHailPara, (int(rgrHailPara.shape[0]/2),2,rgrHailPara.shape[1]))
        DayAverage=fnPhysicalBounds(DayAverage, rgsVariables)
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

        # save the station data
        np.savez(sStationSave, 
                 rgrHailPara=rgrHailPara,
                 rgrLon=rgrLon,
                 rgrLat=rgrLat,
                 rgrElev=rgrElev,
                 rgsStatID=rgsStatID,
                 rgsStatNr=rgsStatNr,
                 MissingDataFlag=MissingDataFlag[st,:],
                 BreakFlag=BreakFlag[:,st,:])

        rgrHailParaAll[:,st,:]=rgrHailPara
        rgrLonAll[st]=rgrLon
        rgrLatAll[st]=rgrLat
        rgrElevAll[st]=rgrElev
        rgsStatIDAll[st]=rgsStatID
        rgsStatNrAll[st]=rgsStatNr
        MissingDataFlag[st,:]=MissingDataFlag[st,:]
        BreakFlag[:,st,:]=BreakFlag[:,st,:]

        print(MissingDataFlag[st,:])
        print(BreakFlag[:,st,:])

        print(("        --- %s seconds ---" % (time.time() - start_time)))
    else:
        print('    load station '+str(st)+' out of '+str(len(rgiStatNr)))
        grDATA=np.load(sStationSave)
        rgrHailParaAll[:,st,:]=grDATA['rgrHailPara']
        rgrLonAll[st]=grDATA['rgrLon']
        rgrLatAll[st]=grDATA['rgrLat']
        rgrElevAll[st]=grDATA['rgrElev']
        rgsStatIDAll[st]=grDATA['rgsStatID']
        rgsStatNrAll[st]=grDATA['rgsStatNr']
        MissingDataFlag[st,:]=grDATA['MissingDataFlag']
        BreakFlag[:,st,:]=grDATA['BreakFlag']


# reform for daily statisitcs calculation
rgrHailParaAll=np.reshape(rgrHailParaAll, (int(len(rgdTime)/2), 2, len(rgiStatNr),len(rgsVariables)))
# remove -99999 values
rgrHailParaAll[(rgrHailParaAll==-99999)]=np.nan

# rgrCAPE=np.max(rgrHailPara[:,:,:,rgsVariables.index('CAPE')], axis=1)
# rgrRHbf=np.mean(rgrHailPara[:,:,:,rgsVariables.index('RHbf')], axis=1)
# rgrRHml=np.mean(rgrHailPara[:,:,:,rgsVariables.index('RHml')], axis=1)
# rgrVS0_1=np.max(rgrHailPara[:,:,:,rgsVariables.index('VS0_1')], axis=1)
# rgrVS0_3=np.max(rgrHailPara[:,:,:,rgsVariables.index('VS0_3')], axis=1)
# rgrVS0_6=np.max(rgrHailPara[:,:,:,rgsVariables.index('VS0_6')], axis=1)
# rgrVS6_12=np.max(rgrHailPara[:,:,:,rgsVariables.index('VS6_12')], axis=1)
# PW=np.mean(rgrHailPara[:,:,:,rgsVariables.index('PW')], axis=1)
# rgrFLH=np.mean(rgrHailPara[:,:,:,rgsVariables.index('FLH')], axis=1)



rgrWBZ=np.mean(rgrHailParaAll[:,:,:,rgsVariables.index('WBZheight')], axis=1)
rgrThreeCheight=np.mean(rgrHailParaAll[:,:,:,rgsVariables.index('ThreeCheight')], axis=1)
rgrLRbF=np.mean(rgrHailParaAll[:,:,:,rgsVariables.index('LRbF')], axis=1)
rgrProbSnow=np.max(rgrHailParaAll[:,:,:,rgsVariables.index('ProbSnow')], axis=1)
rgrCAPE=np.max(rgrHailParaAll[:,:,:,rgsVariables.index('CAPE')], axis=1)
rgrCIN=np.max(rgrHailParaAll[:,:,:,rgsVariables.index('CIN')], axis=1)
rgrLCL=np.mean(rgrHailParaAll[:,:,:,rgsVariables.index('LCL')], axis=1)
rgrLFC=np.mean(rgrHailParaAll[:,:,:,rgsVariables.index('LFC')], axis=1)
rgrCBT=np.mean(rgrHailParaAll[:,:,:,rgsVariables.index('CBT')], axis=1)
rgVS03=np.mean(rgrHailParaAll[:,:,:,rgsVariables.index('VS0_3')], axis=1)
rgrVS06=np.mean(rgrHailParaAll[:,:,:,rgsVariables.index('VS0_6')], axis=1)


# # clean up obvious outliers
# rgrWBZ[(rgrWBZ>7500)]=np.nan
# rgrThreeCheight[(rgrThreeCheight>7500)]=np.nan


# save the data
grStatisitcs={'rgrWBZ':rgrWBZ,
              'rgrThreeCheight':rgrThreeCheight,
              'rgrLRbF':rgrLRbF,
              'rgrProbSnow':rgrProbSnow,
              'CAPE':rgrCAPE,
              'CIN':rgrCIN,
              'LCL':rgrLCL,
              'LFC':rgrLFC,
              'CBT':rgrCBT,
              'VS0_3':rgVS03,
              'VS0_6':rgrVS06,
              'rgsVariables':rgsVariables,
              'rgdTime':rgdTimeFin,
              'rgrLon':rgrLonAll,
              'rgrLat':rgrLatAll,
              'rgrElev':rgrElevAll,
              'rgsStatID':rgsStatIDAll,
              'rgsStatNr':rgsStatNrAll,
              'BreakFlag':BreakFlag}
sTempFileName=sSaveDataDir+'RadSoundHailEnvDat-'+str(StartTime.year)+'-'+str(StopTime.year)+'_incl-CAPE.pkl'
print('    Save: '+sTempFileName)

fh = open(sTempFileName,"wb")
pickle.dump(grStatisitcs,fh)
fh.close()

stop()
