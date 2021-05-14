#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:51:28 2020
@author: emilywilliams
relevant functions to run the AMLD algorithm
"""
import datetime


def dt_to_epoch(dt):
    from datetime import datetime
    epoch = datetime.utcfromtimestamp(0)
    return str((dt - epoch).total_seconds() * 1000.0)

def check_lst(opList):
    if isinstance(opList, str):
        opList = str_list(opList)
    return (opList)

def unique(my_list):
    """ Condense a list and return only its unique entries
    input:
        my_list -- a list
    output:
        a list of the unique items
    """
    return [x for x in my_list if x not in locals()['_[1]']]

def unIfInt(a, b):
    """ If two lists intersect, return their union
    input:
        a,b: two lists
    output:
        if the two lists intersect, returns a list of their union.
        if the two lists do not intersect, returns an empty list
    """
    if len(intersect(a, b)) != 0:
        return (list(set(a).union(b)))

def intersect(a, b):
    """ List of intersections of two lists
    input:
        a,b: lists
    output:
        a intersect b
    """
    return list(set(a) & set(b))

def weighted_loc2(df, lat, lon, by, val2avg):
    """ find the weighted centroid of a data frame
    input:
        df: data frame with gps locations, a grouping variable, and a value to weight with
        lat: name of the column with latitude
        lon: name of the column with longitude
        by: name of the column to group by (i.e. a observed peak name)
        val2avg: name of the column that is being used to weight the location
    output:
        dataframe with weighted location for each grouping variable
    """
    import pandas as pd
    import swifter
    df_use = df.loc[:, [(lat), (lon), (by), val2avg]]
    df_use.loc[:, 'lat_wt'] = df_use.swifter.apply(lambda y: y[lat] * y[val2avg], axis=1).copy()
    df_use.loc[:, 'lon_wt'] = df_use.swifter.apply(lambda y: y[lon] * y[val2avg], axis=1).copy()

    sumwts = pd.DataFrame(df_use.copy().groupby(str(by)).apply(lambda y: sum_values(y[str(val2avg)])), columns={'totwts'})
    sumwts.loc[:, 'min_reads'] = sumwts.copy().index
    sumwts = sumwts.reset_index(drop=True).rename(columns={"min_reads": str(by)})
    totlats = pd.DataFrame(df_use.groupby(str(by)).apply(lambda y: sum_values(y['lat_wt'])), columns=['totlats'])
    totlats['min_reads'] = totlats.index.copy()
    totlats = totlats.reset_index(drop=True)
    totlats = totlats.rename(columns={"min_reads": str(by)})
    totlons = pd.DataFrame(df_use.groupby(str(by)).apply(lambda y: sum_values(y['lon_wt'])), columns=['totlons'])
    totlons['min_reads'] = totlons.index.copy()
    totlons = totlons.reset_index(drop=True)
    totlons = totlons.rename(columns={"min_reads": str(by)})
    df_use = pd.merge(totlats, df_use, on=str(by))
    df_use = pd.merge(totlons, df_use, on=str(by))
    df_use = pd.merge(sumwts, df_use, on=str(by))
    df_use.loc[:, 'overall_LON'] = df_use.swifter.apply(lambda y: y['totlons'] / y['totwts'], axis=1)
    df_use.loc[:, 'overall_LAT'] = df_use.swifter.apply(lambda y: y['totlats'] / y['totwts'], axis=1)
    return (df_use.loc[:, [(str(by)), ('overall_LON'), ('overall_LAT')]].drop_duplicates().rename(
        columns={'overall_LON': str(lon), 'overall_LAT': str(lat)}))

def weighted_loc(df, lat, lon, by, val2avg):
    """ find the weighted centroid of a data frame
    input:
        df: data frame with gps locations, a grouping variable, and a value to weight with
        lat: name of the column with latitude
        lon: name of the column with longitude
        by: name of the column to group by (i.e. a observed peak name)
        val2avg: name of the column that is being used to weight the location
    output:
        dataframe with weighted location for each grouping variable
    """
    import pandas as pd

    import sys
    if sys.platform == 'win32':
        windows = True
    elif sys.platform != 'win32':
        windows = False

    if not windows:
        import swifter
    df_use = df.loc[:, [(lat), (lon), (by), val2avg]]

    if windows:
        df_use.loc[:, 'lat_wt'] = df_use.apply(lambda y: y[lat] * y[val2avg], axis=1).copy()
        df_use.loc[:, 'lon_wt'] = df_use.apply(lambda y: y[lon] * y[val2avg], axis=1).copy()
    elif not windows:
        df_use.loc[:, 'lat_wt'] = df_use.swifter.apply(lambda y: y[lat] * y[val2avg], axis=1).copy()
        df_use.loc[:, 'lon_wt'] = df_use.swifter.apply(lambda y: y[lon] * y[val2avg], axis=1).copy()

    sumwts = pd.DataFrame(df_use.copy().groupby(str(by)).apply(lambda y: sum_values(y[str(val2avg)])), columns={'totwts'})
    sumwts.loc[:, 'min_reads'] = sumwts.copy().index
    sumwts = sumwts.reset_index(drop=True).rename(columns={"min_reads": str(by)})
    totlats = pd.DataFrame(df_use.groupby(str(by)).apply(lambda y: sum_values(y['lat_wt'])), columns=['totlats'])
    totlats['min_reads'] = totlats.index.copy()
    totlats = totlats.reset_index(drop=True)
    totlats = totlats.rename(columns={"min_reads": str(by)})
    totlons = pd.DataFrame(df_use.groupby(str(by)).apply(lambda y: sum_values(y['lon_wt'])), columns=['totlons'])
    totlons['min_reads'] = totlons.index.copy()
    totlons = totlons.reset_index(drop=True)
    totlons = totlons.rename(columns={"min_reads": str(by)})
    df_use = pd.merge(totlats, df_use, on=str(by))
    df_use = pd.merge(totlons, df_use, on=str(by))
    df_use = pd.merge(sumwts, df_use, on=str(by))
    if not windows:
        df_use.loc[:, 'overall_LON'] = df_use.swifter.apply(lambda y: y['totlons'] / y['totwts'], axis=1)
        df_use.loc[:, 'overall_LAT'] = df_use.swifter.apply(lambda y: y['totlats'] / y['totwts'], axis=1)
    elif windows:
        df_use.loc[:, 'overall_LON'] = df_use.apply(lambda y: y['totlons'] / y['totwts'], axis=1)
        df_use.loc[:, 'overall_LAT'] = df_use.apply(lambda y: y['totlats'] / y['totwts'], axis=1)

    return (df_use.loc[:, [(str(by)), ('overall_LON'), ('overall_LAT')]].drop_duplicates().rename(
        columns={'overall_LON': str(lon), 'overall_LAT': str(lat)}))

def verPk(totalData):
    """ make a dataframe of the verified peaks
    input:
        df: data frame with all the data
    output:
        geodataframe of locations
    """
    import pandas as pd  #
    from numpy import log
    import geopandas as gpd
    from shapely.geometry import Point  # Shapely for converting latitude/longtitude to geometry

    import sys
    if sys.platform == 'win32':
        windows = True
    elif sys.platform != 'win32':
        windows = False
        import swifter

    totalData = totalData[totalData.numtimes != 1]
    pkRed = totalData[
        ['PEAK_NUM', 'pk_LON', 'pk_LAT', 'pk_maxCH4_AB', 'numtimes', 'min_read']].drop_duplicates().reset_index()
    verLoc = weighted_loc(pkRed, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB')
    if not windows:
        pkRed.loc[:, ('logCH4')] = pkRed.swifter.apply(lambda y: log(y.pk_maxCH4_AB), axis=1)
    elif windows:
        pkRed.loc[:, ('logCH4')] = pkRed.apply(lambda y: log(y.pk_maxCH4_AB), axis=1)

    mnVals = pkRed.groupby('min_read', as_index=False).logCH4.mean()
    together = pd.merge(verLoc, mnVals, on=['min_read'])
    geometry_temp = [Point(lon, lat) for lon, lat in zip(together['pk_LON'], together['pk_LAT'])]

    #crs = {'init': 'epsg:32610'}
    crs = 'EPSG:32610'

    tog_dat = gpd.GeoDataFrame(together, crs=crs, geometry=geometry_temp)
    tog_dat = tog_dat.to_crs(epsg=3857)

def estimate_emissions(excess_CH4):
    """ estimate emissions of the methane leak, using maximum values found at each OP
    input:
        excessCH4: amount of excess ch4
    output:
        estimated emission from that observed ch4 level
    """
    import math
    a = 0.4630664
    b = 0.7443749
    a1 = 1.2889
    b1 = 0.35232
    a2 = 1.755891
    b2 = 0.4438203

    m = math.exp((excess_CH4 - a) / b)
    # if m < math.exp(3.157):
    #    if m < math.exp(2):
    #       m = math.exp((np.log(m) - a1)/b1)
    #  if m > math.exp(2):
    #     m = math.exp((np.log(m) - a2)/b2)
    return (m)


def haversine(lat1, lon1, lat2, lon2, radius=6371):
    """ calculate the distance between two gps coordinates, using haversine function
    input:
        lat1,lon1: location 1
        lat2,lon2: location 2
        radius: earth's radius at the location used (km). Default is 6371km
    output:
        distance between the points (m)
    """
    from math import radians, sin, cos, sqrt, asin
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    c = 2 * asin(sqrt(sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2))
    return radius * c * 1000  # return in meters


def wt_time_Locs(wt, loc):
    """ takes wt * loc to
    """
    return (wt * loc)

def sum_values(values):
    """ sum everything in the values
    """
    return (sum(values))

def make_GEO(df, lat, lon):
    """ make a geodataframe
    input:
        df: dataframe
        lat: name of column with latitude
        lon: name of column with longitude
    output:
        geometry with points turned to geometries
    """
    from shapely.geometry import Point
    geo = [Point(lon, lat) for lon, lat in zip(df[(lon)], df[(lat)])]
    return (geo)

def make_GPD(df, lat, lon, cps='epsg:4326'):
    """ make geodataframe
    input:
        df: dataframe to turn into geodataframe
        lat: name of latitude column
        lon: name of longitude column
        cps: coordinate system to use
    output:
        geodataframe with the corresponding crs
    """
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(df, crs=cps, geometry=make_GEO(df, lat, lon))
    return (gdf)


def calc_velocity(timediff, distance):
    if timediff == 0:
        return (0)
    elif timediff != 0:
        return (distance / timediff)


# def minread_to_date(min_read,xCar):
#     import datetime
#     m_date = int(float(min_read[len(xCar)+1:]))
#     return(datetime.datetime.fromtimestamp(m_date).strftime('%Y-%m-%d %H:%M:%S'))

def minread_to_date(min_read,xCar):
    import datetime
    try:
        m_date = int(float(min_read[len(xCar)+1:]))
        return(datetime.datetime.fromtimestamp(m_date).strftime('%Y-%m-%d %H:%M:%S'))
    except:
        return(datetime.datetime.fromtimestamp(1).strftime('%Y-%m-%d %H:%M:%S'))


def summarize_dat(totalData):
    """ take all data from analyses, and output summary information
    input:
        df
    output:
        shorter df with summary
    """
    import pandas as pd
    from numpy import log
    pkRed = totalData.loc[:, ['PEAK_NUM', 'pk_LON', 'pk_LAT', 'pk_maxCH4_AB', 'numtimes',
                              'min_read']].drop_duplicates().reset_index().loc[:,
            ['PEAK_NUM', 'pk_LON', 'pk_LAT', 'pk_maxCH4_AB', 'numtimes', 'min_read']]
    verLoc = weighted_loc(pkRed, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB').rename(
        columns={'pk_LAT': 'overallLAT', 'pk_LON': 'overallLON'})
    pkRed['logCH4'] = pkRed.apply(lambda y: log(y.pk_maxCH4_AB), axis=1)
    mnVals = pkRed.groupby('min_read', as_index=False).logCH4.mean().rename(columns={'logCH4': 'mnlogCH4'}).loc[:,
             ['min_read', 'mnlogCH4']]
    together = pd.merge(verLoc, mnVals, on=['min_read'])
    final = pd.merge(together, totalData, on=['min_read'])
    return (final)

def get_quadrant(x, y):
    """ given an x,y coordinate, return which quadrant it is in
    input:
        x,y values
    output:
        quadrant
    exceptions:

    """
    try:
        x = int(x)
        y = int(y)
    except ValueError:
        return (0)

    if y >= 0 and x > 0:
        return (1)
    elif y >= 0 and x < 0:
        return (2)
    elif y < 0 and x < 0:
        return (3)
    else:
        return (4)

def calc_theta(U, V, quad, h_length, radians):
    """ given wind coordinates, quadrant, and the length of horizontal wind, return theta value
    input:
        U,V: wind directions
        quad: quadrant
        h_length: horizontal wind component
        radians: T/F value for radians or degrees
    output:
        theta value of the wind
    """
    import numpy as np
    theta = np.arcsin(U / h_length)
    import numpy as np
    if quad == 1:
        theta = theta
    elif quad == 2:
        theta = -theta + np.pi / 2
    elif quad - - 3:
        theta = np.pi / 2 + theta + np.pi
    elif quad == 4:
        theta = 3 * np.pi / 2
    theta = 2 * np.pi - theta
    if not radians:
        theta = theta * 180 / np.pi

    return (theta)

def calc_bearing(lat1, lat2, long1, long2, radians):
    """ calculating the direction (bearing) of driving
    input:
        lat1,lon1: location 1
        lat2,lon2: location 2
        radians: T/F
    output:
        direction (degrees/radians) of motion
    """
    from math import atan2
    from numpy import pi
    from math import radians, sin, cos

    lat1r = lat1 * (pi / 180)
    lat2r = lat2 * (pi / 180)
    long1r = long1 * (pi / 180)
    long2r = long2 * (pi / 180)
    X = cos(lat2r) * sin(long2r - long1r)
    Y = cos(lat1r) * sin(lat2r) - (sin(lat1r) * cos(lat2r) * cos(long2r - long1r))

    theta = atan2(X, Y)
    theta = theta % (2 * pi)

    if not radians:
        return (theta * 180 / pi)
    elif radians:
        return (theta)

def process_raw_data_eng(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack,
                      shift, maxSpeed='45', minSpeed='2'):
    """ input a raw .txt file with data (enginnering file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import pandas as pd
    from datetime import datetime
    import os
    import gzip
    import sys
    from math import floor
    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)

        ########################################################################
        #### WE DON'T HAVE AN RSSI INPUT
        ### (SO THIS IS A PLACEHOLDER FOR SOME SORT OF QA/QC VARIABLE)
        ##  xMinRSSI = 50  #if RSSI is below this we don't like it
        ##################################################################

        # reading in the (.txt) data with specific headers --> need to change this
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        # sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        # sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        # sHeader = "Time Stamp,Inlet Number,P (mbars),T0 (degC),T5 (degC), Laser PID Readout,Det PID Readout,win0Fit0,win0Fit1,win0Fit3,win1Fit4,win0Fit5,win0Fit6,win0Fit7,win0Fit8,win0Fit9,win1Fit0,win1Fit1,win1Fit2,win1Fit3,win1Fit4,win1Fit5,win1Fit6,Det Bkgd,Ramp Ampl,CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Battery T (degC),FET T (degC),GPS Time,Latitude,Longitude"
        sHeader = "Time Stamp,Inlet Number,P (mbars),T0 (degC),T5 (degC),Laser PID Readout,Det PID Readout,win0Fit0,win0Fit1,win0Fit2,win0Fit3,win0Fit4,win0Fit5,win0Fit6,win0Fit7,win0Fit8,win0Fit9,win1Fit0,win1Fit1,win1Fit2,win1Fit3,win1Fit4,win1Fit5,win1Fit6,Det Bkgd,Ramp Ampl,CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Battery T (degC),FET T (degC),GPS Time,Latitude,Longitude"
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"

        headerNames = sHeader.split(',')
        GPS_loc = 37  # Where the GPS data is located (in the row)

        infoHeader = "FILENAME\n"

        # gZIP is indicating if it is a ZIP file (I don't think I've written this in)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename,'r')
        else:
            #f = open(xDir + "/" + xFilename, 'r')
            f = open(xDir + xFilename, 'r')

        ### FIGURING OUT DATE FROM FILENAME (WILL NEED TO CHANGE THIS IF DIFFERENT FILENAME)
        xdat = str('20') + xFilename[11:17]

        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        fnLog = xOut + xCar + "_" + xdat + "_log.csv"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"

        # FINDING THE FIRST TIME NOTED
        firsttime = int(float(open(xDir + xFilename).readlines().pop(1).split(',')[37][:-4]))

        ## MAKING TEMPORARY FILE (FOR IF LATER YOU HAVE TO ADD A DATE)
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        if bFirst:
            # 3fOut = open(fnOutTemp, 'w')
            # fOut.write(sOutHeader)
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")
        if not bFirst:
            fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        fOut = open(fnOutTemp, 'w')
        fOut.write(sOutHeader)

        # READ IN THE LINES
        xCntObs = -1
        xCntGoodValues = 0
        for row in f:
            bGood = True
            if xCntObs < 0:
                bGood = False
                xCntObs += 1

            if bGood:
                lstS = row.split(',')
                gpstime = lstS[GPS_loc]
                dtime = lstS[0]
                dt = lstS[1]
                time_dt = lstS[2]
                epoch = lstS[3]
                # nano = lstS[4]

                gps_time = lstS[37]
                dateob = datetime.fromtimestamp(int(gps_time[:-4]))
                nano = gps_time[-4:]

                # dateob = datetime(int(dt[0:4]),int(dt[5:7]),int(dt[8:10]),int(time_dt[0:2]),int(time_dt[3:5]),int(time_dt[6:8]),int(float(nano)*1e-9))

                dtime = int(dateob.strftime('%Y%m%d%H%M%S'))
                # Date = dateob.strftime('%m%/%d/%Y')
                Date = dateob.strftime('%Y-%m-%d')

                GPS_Time = dateob.strftime('%H%:%M:%S')
                seconds = floor(float(gpstime))
                nano = dateob.strftime('%f')

                # dateob = datetime(int(dtime[6:10]),int(dtime[0:2]),int(dtime[3:5]),int(dtime[11:13]),int(dtime[14:16]),int(dtime[17:19]),int(float(dtime[19:23])*1000000))
                # epoch = dateob.strftime('%s.%f')

                # THIS IS USING THE CSU METHOD. IN OUR METHOD, WE DO THE SPEED LATER IN THE ALGORITHM.

                # # if RSSI of bottome sensor is below 50 if float(lstS[28]) < xMinRSSI: fLog.write("RSSI (Bottom)
                # value less than 50: "+ str(lstS[28]) + "\n") continue # Car Speed if float(lstS[12]) >
                # xMaxCarSpeed: fLog.write("Car speed of " + str(float(lstS[12])) + " exceeds max threshold of: " +
                # str(xMaxCarSpeed) + "\n") continue if float(lstS[12]) < xMinCarSpeed: fLog.write("Car speed of " +
                # str(float(lstS[12])) + " less than min threshold of: " + str(xMinCarSpeed) + "\n") continue

                # For some reason it is producing its longitude in positive number while USA is located at negative longitude
                # thats why we do -1 * float(lstS[7])

                # fix this when we have stuffs

                #                s1 = str(lstS[1])+","+str(lstS[2])+","+str(lstS[3])+","+str(lstS[4])+","+str(lstS[6])+","
                #                s1 += str(-1 * float(lstS[7]))+","+str(lstS[12])+","+str(lstS[14])+","+str(lstS[15])+","+str(lstS[16])+","+str(lstS[25])+","
                #                s1 += str(lstS[28])+","+str(lstS[38])+","+str(lstS[41])+"\n"

                ## choosing what to write in the .csv

                # if sys.platform.startswith('win'):
                #     ## DATE, TIME, SECONDS,NANOSECONDS
                #     csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(dateob.strftime('%H:%M:%S')) + ',' + str(
                #         float(pd.to_numeric(dateob.strftime('%S.%f')))) + ',' + str(
                #         pd.to_numeric(dateob.strftime('%f')) * 1000) + str(',')
                #     ## VELOCITY, U,V,W,BCH4,BRSSI,TCH4
                #     csvWrite += str('50') + ',' + str('0') + ',' + str('0') + ',' + str('0') + ',' + str(
                #         lstS[26]) + ',' + str('0') + ',' + str(lstS[26]) + ','
                #     ## TRSSI, PRESS_MBAR, INLET, TEMPC, CH4, H20,C2H6
                #     csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(lstS[3]) + ',' + str(
                #         lstS[26]) + ',' + str(lstS[27]) + ',' + str(lstS[28]) + ','
                #     # R, C2C1, BATTV, POWMV,CURRMA, SOCPER,LAT,LONG
                #     csvWrite += str(lstS[29]) + ',' + str(lstS[30]) + ',' + str(lstS[31]) + ',' + str(
                #         lstS[32]) + ',' + str(lstS[33]) + ',' + str(lstS[34]) + ',' + str(lstS[38]) + str(',') + str(
                #         lstS[39])

                # =============================================================================
                #                 if not sys.platform.startswith('win'):
                #                     ## DATE, TIME, SECONDS,NANOSECONDS
                #                     csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(dateob.strftime('%H:%M:%S'))  + ',' + str((int(floor(pd.to_numeric(dateob.strftime('%s.%f')))))) + ',' + str((pd.to_numeric(dateob.strftime('%f')) *1000)) + str(',')
                #                     ## VELOCITY, U,V,W,BCH4,BRSSI,TCH4
                #                     csvWrite += str('50') + ',' + str('0') + ',' + str('0') + ',' + str('0') + ',' + str(lstS[26]) + ',' + str('0') + ','+  str(lstS[26]) + ','
                #                     ## TRSSI, PRESS_MBAR, INLET, TEMPC, CH4, H20,C2H6
                #                     csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(lstS[3]) + ',' + str(lstS[26]) + ',' + str(lstS[27]) +',' +  str(lstS[28]) + ','
                #                     # R, C2C1, BATTV, POWMV,CURRMA, SOCPER,LAT,LONG
                #                     csvWrite += str(lstS[29]) + ',' + str(lstS[30]) + ',' + str(lstS[31]) + ',' + str(lstS[32]) + ','+ str(lstS[33]) + ',' + str(lstS[34]) + ',' + str(lstS[38]) + str(',') + str(lstS[39][:-1]) + str('\n')
                #                 #fOut.write('\n')
                #                 fOut.write(csvWrite)
                #                 #fOut.write('\n')
                #
                # =============================================================================
                # if not sys.platform.startswith('win'):
                if 1 == 1:
                    ## DATE, TIME, SECONDS,NANOSECONDS
                    csvWrite = str(Date) + ',' + str(GPS_Time) + ',' + str(seconds) + ',' + str(nano) + str(',')
                    ## VELOCITY, U,V,W,BCH4,BRSSI,TCH4
                    csvWrite += str('50') + ',' + str('0') + ',' + str('0') + ',' + str('0') + ',' + str(
                        lstS[26]) + ',' + str('0') + ',' + str(lstS[26]) + ','
                    ## TRSSI, PRESS_MBAR, INLET, TEMPC, CH4, H20,C2H6
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(lstS[3]) + ',' + str(
                        lstS[26]) + ',' + str(lstS[27]) + ',' + str(lstS[28]) + ','
                    # R, C2C1, BATTV, POWMV,CURRMA, SOCPER,LAT,LONG
                    csvWrite += str(lstS[29]) + ',' + str(lstS[30]) + ',' + str(lstS[31]) + ',' + str(
                        lstS[32]) + ',' + str(lstS[33]) + ',' + str(lstS[34]) + ',' + str(lstS[38]) + str(',') + str(
                        lstS[39])
                # fOut.write('\n')

                #### REMOVING THE FIRST BIT OF DATA (if you need to )
                if seconds >= (firsttime + (60 * float(initialTimeBack))):
                    fOut.write(csvWrite)

                del (csvWrite)
            #                xCntGoodValues += 1

            xCntObs += 1

        # sOut = str(gZIP) + "," + str(f) + "," + str(xCntObs) + "," + str(xCntGoodValues) + "\n"
        # fLog.write(sOut)

        infOut.write(str(xFilename) + '\n')

        fOut.close()
        fLog.close()
        infOut.close()

        # xDate = dateob.strftime("%Y%m%d")

        # newfnOut = xOutDir + xCar + "_" + xDate + "_dat.csv"       #set CSV output for raw data
        # newfnLog = xOutDir + xCar + "_" + xDate + "_log.csv"

        # print(xCar + "\t" + xdat + "\t" + fnOut[-22:] + "\t" + str(xCntObs) + "\t" + str(xCntGoodValues) + "\t" + str(
        #    gZIP))

        print(f"{xCar} \t {xdat} \t {fnOut[-(17 + len(xCar)):]} \t {xCntObs} \t {xCntGoodValues} \t {gZIP}")


        import numpy as np
        radians = False
        wind_df = pd.read_csv(fnOutTemp)
        wind_df['QUADRANT'] = wind_df.apply(lambda row: get_quadrant(row['U'], row['V']), axis=1)

        wind_df['secnan'] = wind_df.apply(lambda row: row['SECONDS'], axis=1)  # + row['NANOSECONDS']*1e-9,axis=1)
        wind_df['prev_LAT'] = wind_df.LAT.shift(periods=1)
        wind_df['next_LAT'] = wind_df.LAT.shift(periods=-1)
        wind_df['prev_LONG'] = wind_df.LONG.shift(periods=1)
        wind_df['next_LONG'] = wind_df.LONG.shift(periods=-1)
        wind_df['prev_TIME'] = wind_df.secnan.shift(periods=1)
        wind_df['next_TIME'] = wind_df.secnan.shift(periods=-1)
        wind_df['distance'] = wind_df.apply(
            lambda row: haversine(row['prev_LAT'], row['prev_LONG'], row['next_LAT'], row['next_LONG']), axis=1)
        wind_df['bearing'] = wind_df.apply(
            lambda row: calc_bearing(row['prev_LAT'], row['next_LAT'], row['prev_LONG'], row['next_LONG'], radians),
            axis=1)
        wind_df['timediff'] = wind_df.apply(lambda row: row['next_TIME'] - row['prev_TIME'], axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda row: calc_velocity(row['timediff'], row['distance']), axis=1)
        wind_df['U_cor'] = wind_df.apply(lambda row: row['U'] + row['VELOCITY'], axis=1)
        wind_df['horz_length'] = wind_df.apply(lambda row: np.sqrt(row['U_cor'] ** 2 + row['V'] ** 2), axis=1)
        wind_df['uncor_theta'] = wind_df.apply(
            lambda row: calc_bearing(row['U_cor'], row['V'], row['QUADRANT'], row['horz_length'], radians), axis=1)
        wind_df['adj_theta'] = wind_df.apply(lambda row: (row['uncor_theta'] + row['bearing']) % 360, axis=1)
        wind_df['totalWind'] = wind_df.apply(lambda row: np.sqrt(row['horz_length'] ** 2 + row['W'] ** 2), axis=1)
        wind_df['phi'] = wind_df.apply(lambda row: np.arctan(row['horz_length']), axis=1)
        wind_df['shift_CH4'] = wind_df.CH4.shift(periods=int(float(shift)))
        wind_df['raw_CH4'] = wind_df.apply(lambda row: row['BCH4'], axis=1)
        wind_df['BCH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['CH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['TCH4'] = wind_df.loc[:, ['shift_CH4']]

        wind_df2 = wind_df[wind_df.CH4.notnull()]
        wind_df3 = wind_df2.drop(
            ['QUADRANT', 'secnan', 'prev_LAT', 'next_LAT', 'prev_LONG', 'next_LONG', 'prev_TIME', 'next_TIME',
             'distance', 'timediff', 'uncor_theta', 'CH4'], axis=1)
        wind_df3['CH4'] = wind_df3.loc[:, 'shift_CH4']
        wind_df3 = wind_df3.drop(['shift_CH4'], axis=1).loc[:,
                   ['DATE', 'TIME', 'SECONDS', 'NANOSECONDS', 'VELOCITY', 'U', 'V', 'W', 'BCH4', 'BRSSI', 'TCH4',
                    'TRSSI',
                    'PRESS_MBAR', 'INLET', 'TEMPC', 'CH4', 'H20', 'C2H6', 'R', 'C2C1', 'BATTV', 'POWMV', 'CURRMA',
                    'SOCPER',
                    'LAT', 'LONG', 'bearing', 'U_cor', 'horz_length', 'adj_theta', 'totalWind', 'phi', 'raw_CH4']]
        wind_df4 = add_odometer(wind_df3.loc[wind_df3.totalWind.notnull(), :], 'LAT', 'LONG')
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > xMinCarSpeed, :]
        wind_df4 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :].copy().drop_duplicates()
        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
        os.remove(fnOutTemp)
        return True
    except ValueError:
        return False

def process_raw_data_what(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack, shift, maxSpeed='45',
                   minSpeed='2'):
    """ input a raw .txt file with data (not engineering file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import pandas as pd
    from datetime import datetime
    import os,gzip,csv,sys
    from numpy import pi
    import numpy as np
    radians = False

    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)

        # reading in the data with specific headers
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"
        infoHeader = "FILENAME\n"
        # somehow gZIP is indicating if  it is the first file name (I think if it is 0 then it is the first file)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename, 'r')
        else:
            f = open(xDir + "/" + xFilename, 'r')

        infoHeader = "FILENAME\n"

        # process
        # if first time on this car/date, then write header out
        headerNames = sHeader.split(',')
        xdat = str('20') + xFilename[11:17]
        # xdat = str(xFilename[len(xCar)+1:len(xCar) + 9])

        # fnOut = xOutDir + xCar + "_" + xDate.replace("-", "") + "_dat.csv"       #set CSV output for raw data
        # fnLog = xOutDir + xCar + "_" + xDate.replace("-", "") + "_log.csv"       #output for logfile

        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        fnLog = xOut + xCar + "_" + xdat + "_log.csv"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"
        #

        dtime = open(xDir + xFilename).readlines().pop(1).split(',')[0]
        firstdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]), int(dtime[14:16]),
                             int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
        #firsttime = firstdate.strftime('%s.%f')
        firsttime = dt_to_epoch(firstdate)
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        if bFirst:
            # fOut = open(fnOut, 'w')
            # fOut.write(sOutHeader)
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")
        if not bFirst:
            fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        fOut = open(fnOutTemp, 'w')
        fOut.write(sOutHeader)

        # read all lines
        xCntObs = -1
        xCntGoodValues = 0
        for row in f:
            bGood = True
            if xCntObs < 0:
                bGood = False
                xCntObs += 1
            if bGood:
                lstS = row.split(",")
                dtime = lstS[0]
                dateob = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                  int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                # epoch = dateob.strftime('%s.%f')
                # dtime = int(dateob.strftime('%Y%m%d%H%M%S'))

                fdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                 int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                #seconds = fdate.strftime('%s.%f')
                seconds = float(dt_to_epoch(fdate)) * 1e-3

                if 1 == 2: #sys.platform.startswith('win'):
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(dateob.strftime('%H:%M:%S')) + ',' + str(
                        int(pd.to_numeric(dateob.strftime('%S.%f')))) + ',' + str(
                        pd.to_numeric(dateob.strftime('%f')) * 1000) + str(',')
                    csvWrite += str('50') + ',' + str('0') + ',' + str('0') + ',' + str('0') + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(lstS[3]) + ',' + str(
                        lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(',') + str(
                        lstS[14])
                if 1==1:
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(dateob.strftime('%H:%M:%S')) + ',' + str(
                        str(seconds)[:10]) + ',' + str(int(pd.to_numeric(str(seconds)[11:]) * 1000)) + str(',')
                    csvWrite += str('50') + ',' + str('0') + ',' + str('0') + ',' + str('0') + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(lstS[3]) + ',' + str(
                        lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(',') + str(
                        lstS[14])
                if float(seconds) >= (float(firsttime) + (60 * float(initialTimeBack))):
                    fOut.write(csvWrite)
                    del (seconds)
                del (csvWrite)

            xCntObs += 1

        # sOut = str(gZIP) + "," + str(f) + "," + str(xCntObs) + "," + str(xCntGoodValues) + "\n"
        # fLog.write(sOut)
        infOut.write(str(xFilename) + '\n')

        fOut.close()
        fLog.close()
        infOut.close()

        print(f"{xCar} \t {xdat} \t {fnOut[-(17 + len(xCar)):]} \t {xCntObs} \t {xCntGoodValues} \t  {gZIP}")

        wind_df = pd.read_csv(fnOutTemp)
        wind_df['QUADRANT'] = wind_df.apply(lambda row: get_quadrant(row['U'], row['V']), axis=1)
        wind_df['secnan'] = wind_df.apply(lambda row: row['SECONDS'] + row['NANOSECONDS'] * 1e-9,
                                          axis=1)  # + row['NANOSECONDS']*1e-9,axis=1)
        wind_df['prev_LAT'] = wind_df.LAT.shift(periods=1)
        wind_df['next_LAT'] = wind_df.LAT.shift(periods=-1)
        wind_df['prev_LONG'] = wind_df.LONG.shift(periods=1)
        wind_df['next_LONG'] = wind_df.LONG.shift(periods=-1)
        wind_df['prev_TIME'] = wind_df.secnan.shift(periods=1)
        wind_df['next_TIME'] = wind_df.secnan.shift(periods=-1)
        wind_df['distance'] = wind_df.apply(
            lambda row: haversine(row['prev_LAT'], row['prev_LONG'], row['next_LAT'], row['next_LONG']), axis=1)
        wind_df['bearing'] = wind_df.apply(
            lambda row: calc_bearing(row['prev_LAT'], row['next_LAT'], row['prev_LONG'], row['next_LONG'], radians),
            axis=1)
        wind_df['timediff'] = wind_df.apply(lambda row: row['next_TIME'] - row['prev_TIME'], axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda row: calc_velocity(row['timediff'], row['distance']), axis=1)
        wind_df['U_cor'] = wind_df.apply(lambda row: row['U'] + row['VELOCITY'], axis=1)
        wind_df['horz_length'] = wind_df.apply(lambda row: np.sqrt(row['U_cor'] ** 2 + row['V'] ** 2), axis=1)
        wind_df['uncor_theta'] = wind_df.apply(
            lambda row: calc_bearing(row['U_cor'], row['V'], row['QUADRANT'], row['horz_length'], radians), axis=1)
        wind_df['adj_theta'] = wind_df.apply(lambda row: (row['uncor_theta'] + row['bearing']) % 360, axis=1)
        wind_df['totalWind'] = wind_df.apply(lambda row: np.sqrt(row['horz_length'] ** 2 + row['W'] ** 2), axis=1)
        wind_df['phi'] = wind_df.apply(lambda row: np.arctan(row['horz_length']), axis=1)
        wind_df['shift_CH4'] = wind_df.CH4.shift(periods=int(float(shift)))
        wind_df['raw_CH4'] = wind_df.apply(lambda row: row['BCH4'], axis=1)
        wind_df['BCH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['CH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['TCH4'] = wind_df.loc[:, ['shift_CH4']]

        wind_df2 = wind_df[wind_df.CH4.notnull()]
        wind_df2 = wind_df.copy()
        wind_df3 = wind_df2.drop(
            ['QUADRANT', 'secnan', 'prev_LAT', 'next_LAT', 'prev_LONG', 'next_LONG', 'prev_TIME', 'next_TIME',
             'timediff', 'uncor_theta', 'CH4'], axis=1)
        wind_df3['CH4'] = wind_df3.loc[:, 'shift_CH4']
        wind_df3 = wind_df3.drop(['shift_CH4'], axis=1).loc[:,
                   ['DATE', 'TIME', 'SECONDS', 'NANOSECONDS', 'VELOCITY', 'U', 'V', 'W', 'BCH4', 'BRSSI', 'TCH4',
                    'TRSSI',
                    'PRESS_MBAR', 'INLET', 'TEMPC', 'CH4', 'H20', 'C2H6', 'R', 'C2C1', 'BATTV', 'POWMV', 'CURRMA',
                    'SOCPER',
                    'LAT', 'LONG', 'bearing', 'U_cor', 'horz_length', 'adj_theta', 'totalWind', 'phi', 'raw_CH4',
                    'distance']]
        wind_df3['odometer'] = wind_df3.loc[:, 'distance'].cumsum()
        wind_df4 = wind_df3.copy()
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > xMinCarSpeed, :]
        wind_df4 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :].copy().drop_duplicates()
        wind_df5 = wind_df4.loc[wind_df4.CH4.notnull(), :]
        wind_df4 = wind_df5.copy()
        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
        os.remove(fnOutTemp)
        return True
    except ValueError:
        return False

def process_raw_data(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack, shift, maxSpeed='45',
                   minSpeed='2'):
    """ input a raw .txt file with data (not engineering file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import pandas as pd
    from datetime import datetime
    import os,gzip,csv,sys
    from numpy import pi
    import numpy as np
    radians = False

    import sys
    if sys.platform == 'win32':
        windows = True
    elif sys.platform != 'win32':
        windows = False

    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)

        # reading in the data with specific headers
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"
        infoHeader = "FILENAME\n"
        # somehow gZIP is indicating if  it is the first file name (I think if it is 0 then it is the first file)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename, 'r')
        else:
            f = open(xDir + "/" + xFilename, 'r')

        infoHeader = "FILENAME\n"

        # process
        # if first time on this car/date, then write header out
        headerNames = sHeader.split(',')
        xdat = str('20') + xFilename[11:17]
        # xdat = str(xFilename[len(xCar)+1:len(xCar) + 9])

        # fnOut = xOutDir + xCar + "_" + xDate.replace("-", "") + "_dat.csv"       #set CSV output for raw data
        # fnLog = xOutDir + xCar + "_" + xDate.replace("-", "") + "_log.csv"       #output for logfile

        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        fnLog = xOut + xCar + "_" + xdat + "_log.csv"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"
        #

        dtime = open(xDir + xFilename).readlines().pop(1).split(',')[0]
        firstdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]), int(dtime[14:16]),
                             int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
        #firsttime = firstdate.strftime('%s.%f')
        firsttime = dt_to_epoch(firstdate)
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        if bFirst:
            # fOut = open(fnOut, 'w')
            # fOut.write(sOutHeader)
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")
        if not bFirst:
            fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        fOut = open(fnOutTemp, 'w')
        fOut.write(sOutHeader)

        # read all lines
        xCntObs = -1
        xCntGoodValues = 0
        for row in f:
            bGood = True
            if xCntObs < 0:
                bGood = False
                xCntObs += 1
            if bGood:
                lstS = row.split(",")
                dtime = lstS[0]
                dateob = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                  int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                # epoch = dateob.strftime('%s.%f')
                # dtime = int(dateob.strftime('%Y%m%d%H%M%S'))

                fdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                 int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                #seconds = fdate.strftime('%s.%f')
                seconds = dt_to_epoch(fdate)
                def getNS(seconds):
                    ns = str(float(seconds) * 1e-3)[11:]
                    #str(pd.to_numeric(str(float(seconds) * 1e-3)[11:]) * 100000)[:9]
                    return (str(ns).ljust(15, '0'))[:9]

                csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(dateob.strftime('%H:%M:%S')) + ',' + str(
                    str(float(seconds)*1e-3)[:10]) + ',' + getNS(seconds) + str(',')
                csvWrite += str('50') + ',' + str('0') + ',' + str('0') + ',' + str('0') + ',' + str(
                    lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(lstS[3]) + ',' + str(
                    lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                    lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(',') + str(
                    lstS[14])

                if float(seconds) >= (float(firsttime) + (60 * float(initialTimeBack))):
                    fOut.write(csvWrite)
                    del (seconds)
                del (csvWrite)
                xCntObs += 1
        # sOut = str(gZIP) + "," + str(f) + "," + str(xCntObs) + "," + str(xCntGoodValues) + "\n"
        # fLog.write(sOut)
            infOut.write(str(xFilename) + '\n')
        fOut.close()
        fLog.close()
        infOut.close()

        print(f"{xCar} \t {xdat} \t {fnOut[-(17 + len(xCar)):]} \t {xCntObs} \t {xCntGoodValues} \t  {gZIP}")

        wind_df = pd.read_csv(fnOutTemp)
        wind_df['shift_CH4'] = wind_df.CH4.shift(periods=int(float(shift)))
        wind_df['raw_CH4'] = wind_df.apply(lambda row: row['BCH4'], axis=1)
        wind_df['BCH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['CH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['TCH4'] = wind_df.loc[:, ['shift_CH4']]

        wind_df['shift_R'] = wind_df.R.shift(periods=int(float(shift)))
        wind_df['raw_R'] = wind_df.apply(lambda row: row['R'], axis=1)
        wind_df_not_null = wind_df.loc[wind_df['LAT'].notnull(),].reset_index(drop=True)
        del (wind_df)
        wind_df = wind_df_not_null.copy()

        radians = False
        wind_df['QUADRANT'] = wind_df.apply(lambda row: get_quadrant(row['U'], row['V']), axis=1)
        wind_df['secnan'] = wind_df.apply(lambda row: row['SECONDS'] + row['NANOSECONDS'] * 1e-9,
                                          axis=1)  # + row['NANOSECONDS']*1e-9,axis=1)
        wind_df['prev_LAT'] = wind_df.LAT.shift(periods=1)
        wind_df['next_LAT'] = wind_df.LAT.shift(periods=-1)
        wind_df['prev_LONG'] = wind_df.LONG.shift(periods=1)
        wind_df['next_LONG'] = wind_df.LONG.shift(periods=-1)
        wind_df['prev_TIME'] = wind_df.secnan.shift(periods=1)
        wind_df['next_TIME'] = wind_df.secnan.shift(periods=-1)
        wind_df['distance'] = wind_df.apply(
            lambda row: haversine(row['prev_LAT'], row['prev_LONG'], row['next_LAT'], row['next_LONG']), axis=1)
        wind_df['bearing'] = wind_df.apply(
            lambda row: calc_bearing(row['prev_LAT'], row['next_LAT'], row['prev_LONG'], row['next_LONG'], radians),
            axis=1)
        wind_df['timediff'] = wind_df.apply(lambda row: row['next_TIME'] - row['prev_TIME'], axis=1)
        wind_df['VELOCITY_calc'] = wind_df.apply(lambda row:calc_velocity(row['timediff'],row['distance']),axis=1)
        wind_df = wind_df.drop(columns = ['VELOCITY'])
        wind_df = wind_df.rename(columns = {'VELOCITY_calc':'VELOCITY'})

        wind_df['VELOCITY'] = wind_df.apply(lambda x: (str(x.VELOCITY)), axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda x: 0 if x.VELOCITY == 'XX.X' else x.VELOCITY, axis=1)
        wind_df['fVel'] = wind_df["VELOCITY"].str.split(".", n=1, expand=True)[0]
        wind_df = wind_df.loc[wind_df['fVel'].notnull(),:].reset_index(drop=True)
        wind_df = wind_df.loc[wind_df['fVel'] != 'nan',:].reset_index(drop=True)
        wind_df['firstVel'] = wind_df.apply(lambda x: int(x['fVel']), axis=1)

        wind_df['sVel'] = wind_df["VELOCITY"].str.split(".", n=1, expand=True)[1]
        wind_df = wind_df.loc[wind_df['sVel'].notnull(),].reset_index(drop=True)
        wind_df['secVel'] = wind_df.apply(lambda x: int(x['sVel']), axis=1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstVel) + '.' + str(x.secVel)), axis=1)
        wind_df2 = wind_df.drop(columns=['VELOCITY', 'secVel', 'sVel', 'fVel', 'firstVel'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'vloc': 'VELOCITY'})
        wind_df = wind_df2.copy()
        del (wind_df2)
        ## CORRECT W WIND THING
        wind_df['W'] = wind_df.apply(lambda x: (str(x.W)), axis=1)
        wind_df['W'] = wind_df.apply(lambda x: 0 if x.W == 'XX.X' else x.W, axis=1)
        wind_df['W'] = wind_df.apply(lambda x: '0.0' if x.W == '0' else x.W, axis = 1)
        wind_df['fW'] = wind_df["W"].str.split(".", n=1, expand=True)[0]
        # wind_df = wind_df.loc[wind_df['fW'].notnull(),].reset_index(drop=True)
        wind_df['firstW'] = wind_df.apply(lambda x: int(x['fW']), axis=1)
        wind_df['sW'] = wind_df["W"].str.split(".", n=1, expand=True)[1]
        # wind_df = wind_df.loc[wind_df['sW'].notnull(),].reset_index(drop=True)
        wind_df['secW'] = wind_df.apply(lambda x: int(x['sW']), axis=1)
        wind_df['wloc'] = wind_df.apply(lambda x: float(str(x.firstW) + '.' + str(x.secW)), axis=1)
        wind_df2 = wind_df.drop(columns=['W', 'secW', 'sW', 'fW', 'firstW'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'wloc': 'W'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        ## CORRECT U WIND THING
        wind_df['U'] = wind_df.apply(lambda x: (str(x.U)), axis=1)
        wind_df['U'] = wind_df.apply(lambda x: 0 if x.U == 'XX.X' else x.U, axis=1)
        wind_df['U'] = wind_df.apply(lambda x: '0.0' if x.U == '0' else x.U, axis = 1)

        wind_df['fU'] = wind_df["U"].str.split(".", n=1, expand=True)[0]
        wind_df['firstU'] = wind_df.apply(lambda x: int(x['fU']), axis=1)
        wind_df['sU'] = wind_df["U"].str.split(".", n=1, expand=True)[1]
        wind_df['secU'] = wind_df.apply(lambda x: int(x['sU']), axis=1)
        wind_df['uloc'] = wind_df.apply(lambda x: float(str(x.firstU) + '.' + str(x.secU)), axis=1)
        wind_df2 = wind_df.drop(columns=['U', 'secU', 'sU', 'fU', 'firstU'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'uloc': 'U'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        ## CORRECT V WIND THING
        wind_df['V'] = wind_df.apply(lambda x: (str(x.V)), axis=1)
        wind_df['V'] = wind_df.apply(lambda x: 0 if x.V == 'XX.X' else x.V, axis=1)
        wind_df['V'] = wind_df.apply(lambda x: '0.0' if x.V == '0' else x.V, axis = 1)

        wind_df['fV'] = wind_df["V"].str.split(".", n=1, expand=True)[0]
        wind_df['firstV'] = wind_df.apply(lambda x: int(x['fV']), axis=1)
        wind_df['sV'] = wind_df["V"].str.split(".", n=1, expand=True)[1]
        wind_df['secV'] = wind_df.apply(lambda x: int(x['sV']), axis=1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstV) + '.' + str(x.secV)), axis=1)
        wind_df2 = wind_df.drop(columns=['V', 'secV', 'sV', 'fV', 'firstV'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'vloc': 'V'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        wind_df['U_cor'] = wind_df.apply(lambda row: float(row['U']) + float(row['VELOCITY']), axis=1)
        wind_df['horz_length'] = wind_df.apply(lambda row: np.sqrt(row['U_cor'] ** 2 + row['V'] ** 2), axis=1)
        wind_df['uncor_theta'] = wind_df.apply(
            lambda row: calc_bearing(row['U_cor'], row['V'], row['QUADRANT'], row['horz_length'], radians), axis=1)
        wind_df['adj_theta'] = wind_df.apply(lambda row: (row['uncor_theta'] + row['bearing']) % 360, axis=1)
        wind_df['totalWind'] = wind_df.apply(lambda row: np.sqrt(row['horz_length'] ** 2 + row['W'] ** 2), axis=1)
        wind_df['phi'] = wind_df.apply(lambda row: np.arctan(row['horz_length']), axis=1)

        wind_df['adj_v'] = wind_df.apply(lambda row: -row['horz_length'] * np.cos(row['adj_theta']), axis=1)
        wind_df['adj_u'] = wind_df.apply(lambda row: row['horz_length'] * np.sin(row['adj_theta']), axis=1)

        ## GO THROUGH WIND
        window_size = 30
        u_series = pd.Series(wind_df['adj_u'])
        u_windows = u_series.rolling(window_size)
        u_averages = pd.DataFrame(u_windows.mean())
        u_averages.columns = ['U_avg']
        u_averages['key'] = u_averages.index

        v_series = pd.Series(wind_df['adj_v'])
        v_windows = v_series.rolling(window_size)
        v_averages = pd.DataFrame(v_windows.mean())
        v_averages.columns = ['V_avg']
        v_averages['key'] = v_averages.index

        w_series = pd.Series(wind_df['W'])
        w_windows = w_series.rolling(window_size)
        w_averages = pd.DataFrame(w_windows.mean())
        w_averages.columns = ['W_avg']
        w_averages['key'] = w_averages.index

        vw_df = w_averages.set_index('key').join(v_averages.set_index('key'))
        vw_df['key'] = vw_df.index
        uvw_df = vw_df.set_index('key').join(u_averages.set_index('key'))
        uvw_df['key'] = uvw_df.index
        wind_df2 = wind_df.copy()
        wind_df2['key'] = wind_df2.index
        wind_df = uvw_df.set_index('key').join(wind_df2.set_index('key'))

        wind_df['r_avg'] = wind_df.apply(lambda row: np.sqrt(row['U_avg'] ** 2 + row['V_avg'] ** 2), axis=1)
        wind_df['theta_avg'] = wind_df.apply(lambda row: 0 if row.V_avg == 0 else np.arctan(-row['U_avg'] / row['V_avg']), axis=1)
        # wind_df2 = wind_df[wind_df.CH4.notnull()]
        wind_df3 = wind_df[wind_df.CH4.notnull()].drop(columns=
                                                       ['QUADRANT', 'secnan', 'prev_LAT', 'next_LAT', 'prev_LONG',
                                                        'next_LONG', 'prev_TIME', 'next_TIME',
                                                        'timediff', 'uncor_theta', 'CH4', 'R'], axis=1)
        wind_df3['CH4'] = wind_df3.loc[:, 'shift_CH4']
        wind_df3['R'] = wind_df3.loc[:, 'shift_R']
        wind_df3 = wind_df3.drop(['shift_CH4', 'shift_R'], axis=1)
        # wind_df4 = wind_df3.loc[wind_df3.totalWind.notnull(),:]
        wind_df3['odometer'] = wind_df3.loc[:, 'distance'].cumsum()
        wind_df4 = wind_df3.loc[:,
                   ['DATE', 'TIME', 'SECONDS', 'NANOSECONDS', 'VELOCITY', 'U', 'V', 'W', 'BCH4', 'BRSSI', 'TCH4',
                    'TRSSI',
                    'PRESS_MBAR', 'INLET', 'TEMPC', 'CH4', 'H20', 'C2H6', 'R', 'C2C1', 'BATTV', 'POWMV', 'CURRMA',
                    'SOCPER',
                    'LAT', 'LONG', 'bearing', 'U_cor', 'horz_length', 'adj_theta', 'totalWind',
                    'phi', 'raw_CH4', 'raw_R', 'U_avg', 'V_avg', 'W_avg', 'r_avg', 'theta_avg', 'distance', 'odometer']]

        # wind_df7 = add_odometer(wind_df4,'LAT','LONG')

        # wind_df4 = wind_df7.copy()
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > xMinCarSpeed, :]
        wind_df6 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :]

        # wind_df6 = wind_df6a.loc[wind_df6a.R > .6999, :]

        del (wind_df4)
        wind_df4 = wind_df6.copy().drop_duplicates()
        wind_df5 = wind_df4.loc[wind_df4.CH4.notnull(), :]
        wind_df4 = wind_df5.copy()
        wind_df4 = wind_df5.copy()
        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
        os.remove(fnOutTemp)
        return True
    except ValueError:
        return False

def nanthing(thing):
    import math
    if (math.isnan(thing) == True):
        return (0)
    else:
        return (thing)

def process_raw_data_aeris2(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack, shift, maxSpeed='45',
                        minSpeed='2'):
    """ input a raw .txt file with data (from aeris data file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import sys
    if sys.platform == 'win32':
        windows = True
    elif sys.platform != 'win32':
        windows = False

    import pandas as pd
    from datetime import datetime
    import os
    import gzip
    import numpy as np
    # import csv
    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)

        ########################################################################
        #### WE DON'T HAVE AN RSSI INPUT
        ### (SO THIS IS A PLACEHOLDER FOR SOME SORT OF QA/QC VARIABLE)
        ##  xMinRSSI = 50  #if RSSI is below this we don't like it
        ##################################################################

        # reading in the data with specific headers
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        sHeader = 'Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude,U (m/sec),V (m/sec),W (m/sec),T (degC),Dir (deg),Speed (m/sec),Compass (deg)'
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"
        infoHeader = "FILENAME\n"
        # somehow gZIP is indicating if  it is the first file name (I think if it is 0 then it is the first file)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename,
                          'r')  # if in python 3, change this to "r" or just "b" can't remember but something about a bit not a string
        else:
            f = open(xDir + xFilename, 'r')

        infoHeader = "FILENAME\n"

        # process - if first time on this car/date, then write header out
        headerNames = sHeader.split(',')
        xdat = str('20') + xFilename[11:17]
        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        fnLog = xOut + xCar + "_" + xdat + "_log.csv"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"
        #

        dtime = open(xDir + xFilename).readlines().pop(1).split(',')[0]
        firstdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                             int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
        #firsttime = firstdate.strftime('%s.%f')
        firsttime = dt_to_epoch(firstdate)
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        if bFirst:
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")
        if not bFirst:
            fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        fOut = open(fnOutTemp, 'w')
        fOut.write(sOutHeader)

        # read all lines
        xCntObs = -1
        xCntGoodValues = 0
        for row in f:
            woo = row
            bGood = True
            if xCntObs < 0:
                bGood = False
                xCntObs += 1
            if bGood:
                lstS = row.split(",")
                dtime = lstS[0]
                dateob = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                  int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                fdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                 int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                #seconds = fdate.strftime('%s.%f')
                seconds = dt_to_epoch(fdate)
                def getNS(seconds):
                    ns = str(float(seconds) * 1e-3)[11:]
                    # str(pd.to_numeric(str(float(seconds) * 1e-3)[11:]) * 100000)[:9]
                    return (str(ns).ljust(15, '0'))[:9]

                import sys
                if sys.platform.startswith('win'):
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(
                        dateob.strftime('%H:%M:%S')) + ',' + str(
                        int(pd.to_numeric(dateob.strftime('%S.%f')))) + ',' + str(
                        pd.to_numeric(dateob.strftime('%f')) * 1000) + str(',')
                    csvWrite += str('50') + ',' + str('0') + ',' + str('0') + ',' + str('0') + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(
                        lstS[3]) + ',' + str(lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(
                        ',') + str(lstS[14])
                if not sys.platform.startswith('win'):
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(
                        dateob.strftime('%H:%M:%S')) + ',' + str(str(float(seconds)*1e-3)[:10]) + ',' + getNS(seconds)+ str(',')
                    csvWrite += str(lstS[20]) + ',' + str(lstS[15]) + ',' + str(lstS[16]) + ',' + str(
                        lstS[17]) + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(
                        lstS[3]) + ',' + str(lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(
                        ',') + str(lstS[14]) + '\n'
                fOut.write(csvWrite)
                xCntObs += 1
            infOut.write(str(xFilename) + '\n')
        fOut.close()
        fLog.close()
        infOut.close()
        # print(xCar + "\t" + xdat + "\t" + fnOut[-22:] + "\t" + str(xCntObs) + "\t" + str(xCntGoodValues) + "\t" + str(
        #    gZIP))
        print(f"{xCar} \t {xdat} \t {fnOut[-(17 + len(xCar)):]} \t  {xCntObs} \t {xCntGoodValues} \t {gZIP}")

        wind_df = pd.read_csv(fnOutTemp)
        wind_df_not_null = wind_df.loc[wind_df['LAT'].notnull(),].reset_index(drop=True)
        del (wind_df)
        wind_df = wind_df_not_null.copy()

        radians = False
        wind_df['QUADRANT'] = wind_df.apply(lambda row: get_quadrant(row['U'], row['V']), axis=1)
        wind_df['secnan'] = wind_df.apply(lambda row: row['SECONDS'] + row['NANOSECONDS'] * 1e-9,
                                          axis=1)  # + row['NANOSECONDS']*1e-9,axis=1)
        wind_df['prev_LAT'] = wind_df.LAT.shift(periods=1)
        wind_df['next_LAT'] = wind_df.LAT.shift(periods=-1)
        wind_df['prev_LONG'] = wind_df.LONG.shift(periods=1)
        wind_df['next_LONG'] = wind_df.LONG.shift(periods=-1)
        wind_df['prev_TIME'] = wind_df.secnan.shift(periods=1)
        wind_df['next_TIME'] = wind_df.secnan.shift(periods=-1)
        wind_df['distance'] = wind_df.apply(
            lambda row: haversine(row['prev_LAT'], row['prev_LONG'], row['next_LAT'], row['next_LONG']), axis=1)

        wind_df['bearing'] = wind_df.apply(
            lambda row: calc_bearing(row['prev_LAT'], row['next_LAT'], row['prev_LONG'], row['next_LONG'], radians),
            axis=1)
        wind_df['timediff'] = wind_df.apply(lambda row: row['next_TIME'] - row['prev_TIME'], axis=1)
        # wind_df['VELOCITY_calc'] = wind_df.apply(lambda row:calc_velocity(row['timediff'],row['distance']),axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda x: (str(x.VELOCITY)),axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda x: 0 if x.VELOCITY == 'XX.X' else x.VELOCITY,axis = 1)
        wind_df['fVel'] = wind_df["VELOCITY"].str.split(".", n = 1, expand = True)[0]
        wind_df = wind_df.loc[wind_df['fVel'].notnull(),].reset_index(drop=True)
        wind_df['firstVel'] = wind_df.apply(lambda x: int(x['fVel']),axis = 1)

        wind_df['sVel'] = wind_df["VELOCITY"].str.split(".", n = 1, expand = True)[1]
        wind_df = wind_df.loc[wind_df['sVel'].notnull(),].reset_index(drop=True)
        wind_df['secVel'] = wind_df.apply(lambda x: int(x['sVel']),axis = 1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstVel) + '.' + str(x.secVel)),axis = 1)
        wind_df2 = wind_df.drop(columns = ['VELOCITY','secVel','sVel','fVel','firstVel'])
        del(wind_df)
        wind_df2 = wind_df2.rename(columns = {'vloc':'VELOCITY'})
        wind_df = wind_df2.copy()
        del(wind_df2)
        ## CORRECT W WIND THING
        wind_df['W'] = wind_df.apply(lambda x: (str(x.W)),axis=1)
        wind_df['W'] = wind_df.apply(lambda x: 0 if x.W == 'XX.X' else x.W,axis = 1)
        wind_df['fW'] = wind_df["W"].str.split(".", n = 1, expand = True)[0]
        #wind_df = wind_df.loc[wind_df['fW'].notnull(),].reset_index(drop=True)
        wind_df['firstW'] = wind_df.apply(lambda x: int(x['fW']),axis = 1)
        wind_df['sW'] = wind_df["W"].str.split(".", n = 1, expand = True)[1]
        #wind_df = wind_df.loc[wind_df['sW'].notnull(),].reset_index(drop=True)
        wind_df['secW'] = wind_df.apply(lambda x: int(x['sW']),axis = 1)
        wind_df['wloc'] = wind_df.apply(lambda x: float(str(x.firstW) + '.' + str(x.secW)),axis = 1)
        wind_df2 = wind_df.drop(columns = ['W','secW','sW','fW','firstW'])
        del(wind_df)
        wind_df2 = wind_df2.rename(columns = {'wloc':'W'})
        wind_df = wind_df2.copy()
        del(wind_df2)

        ## CORRECT U WIND THING
        wind_df['U'] = wind_df.apply(lambda x: (str(x.U)),axis=1)
        wind_df['U'] = wind_df.apply(lambda x: 0 if x.U == 'XX.X' else x.U,axis = 1)
        wind_df['fU'] = wind_df["U"].str.split(".", n = 1, expand = True)[0]
        wind_df['firstU'] = wind_df.apply(lambda x: int(x['fU']),axis = 1)
        wind_df['sU'] = wind_df["U"].str.split(".", n = 1, expand = True)[1]
        wind_df['secU'] = wind_df.apply(lambda x: int(x['sU']),axis = 1)
        wind_df['uloc'] = wind_df.apply(lambda x: float(str(x.firstU) + '.' + str(x.secU)),axis = 1)
        wind_df2 = wind_df.drop(columns = ['U','secU','sU','fU','firstU'])
        del(wind_df)
        wind_df2 = wind_df2.rename(columns = {'uloc':'U'})
        wind_df = wind_df2.copy()
        del(wind_df2)

        ## CORRECT V WIND THING
        wind_df['V'] = wind_df.apply(lambda x: (str(x.V)),axis=1)
        wind_df['V'] = wind_df.apply(lambda x: 0 if x.V == 'XX.X' else x.V,axis = 1)
        wind_df['fV'] = wind_df["V"].str.split(".", n = 1, expand = True)[0]
        wind_df['firstV'] = wind_df.apply(lambda x: int(x['fV']),axis = 1)
        wind_df['sV'] = wind_df["V"].str.split(".", n = 1, expand = True)[1]
        wind_df['secV'] = wind_df.apply(lambda x: int(x['sV']),axis = 1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstV) + '.' + str(x.secV)),axis = 1)
        wind_df2 = wind_df.drop(columns = ['V','secV','sV','fV','firstV'])
        del(wind_df)
        wind_df2 = wind_df2.rename(columns = {'vloc':'V'})
        wind_df = wind_df2.copy()
        del(wind_df2)

        wind_df['U_cor'] = wind_df.apply(lambda row: float(row['U']) + float(row['VELOCITY']), axis=1)
        wind_df['horz_length'] = wind_df.apply(lambda row: np.sqrt(row['U_cor'] ** 2 + row['V'] ** 2), axis=1)
        wind_df['uncor_theta'] = wind_df.apply(
            lambda row: calc_bearing(row['U_cor'], row['V'], row['QUADRANT'], row['horz_length'], radians), axis=1)
        wind_df['adj_theta'] = wind_df.apply(lambda row: (row['uncor_theta'] + row['bearing']) % 360, axis=1)
        wind_df['totalWind'] = wind_df.apply(lambda row: np.sqrt(row['horz_length'] ** 2 + row['W'] ** 2), axis=1)
        wind_df['phi'] = wind_df.apply(lambda row: np.arctan(row['horz_length']), axis=1)
        wind_df['shift_CH4'] = wind_df.CH4.shift(periods=int(float(shift)))
        wind_df['raw_CH4'] = wind_df.apply(lambda row: row['BCH4'], axis=1)
        wind_df['BCH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['CH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['TCH4'] = wind_df.loc[:, ['shift_CH4']]

        wind_df2 = wind_df[wind_df.CH4.notnull()]
        wind_df2 = wind_df.copy()
        wind_df3 = wind_df2.drop(
            ['QUADRANT', 'secnan', 'prev_LAT', 'next_LAT', 'prev_LONG', 'next_LONG', 'prev_TIME', 'next_TIME',
             'timediff', 'uncor_theta', 'CH4'], axis=1)

        wind_df3['CH4'] = wind_df3.loc[:, 'shift_CH4']
        wind_df3 = wind_df3.drop(['shift_CH4'], axis=1)
        # wind_df4 = wind_df3.loc[wind_df3.totalWind.notnull(),:]
        wind_df3 = wind_df3.loc[:,
                   ['DATE', 'TIME', 'SECONDS', 'NANOSECONDS', 'VELOCITY', 'U', 'V', 'W', 'BCH4', 'BRSSI', 'TCH4',
                    'TRSSI',
                    'PRESS_MBAR', 'INLET', 'TEMPC', 'CH4', 'H20', 'C2H6', 'R', 'C2C1', 'BATTV', 'POWMV', 'CURRMA',
                    'SOCPER',
                    'LAT', 'LONG', 'bearing', 'U_cor', 'horz_length', 'adj_theta', 'totalWind', 'phi', 'raw_CH4',
                    'distance']]
        wind_df3['odometer'] = wind_df3.loc[:, 'distance'].cumsum()
        wind_df4 = wind_df3.copy()

        # wind_df7 = add_odometer(wind_df4,'LAT','LONG')

        # wind_df4 = wind_df7.copy()
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > xMinCarSpeed, :]
        wind_df6 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :]

        del (wind_df4)
        wind_df4 = wind_df6.copy().drop_duplicates()
        wind_df5 = wind_df4.loc[wind_df4.CH4.notnull(), :]
        wind_df4 = wind_df5.copy()
        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
        os.remove(fnOutTemp)
        return True
    except ValueError:
        return False

def process_raw_data_aeris(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack, shift, maxSpeed='45',
                        minSpeed='2'):
    """ input a raw .txt file with data (from aeris data file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import sys
    if sys.platform == 'win32':
        windows = True
    elif sys.platform != 'win32':
        windows = False

    import pandas as pd
    from datetime import datetime
    import os
    import gzip
    import numpy as np
    # import csv
    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)

        ########################################################################
        #### WE DON'T HAVE AN RSSI INPUT
        ### (SO THIS IS A PLACEHOLDER FOR SOME SORT OF QA/QC VARIABLE)
        ##  xMinRSSI = 50  #if RSSI is below this we don't like it
        ##################################################################

        # reading in the data with specific headers
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        sHeader = 'Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude,U (m/sec),V (m/sec),W (m/sec),T (degC),Dir (deg),Speed (m/sec),Compass (deg)'
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"
        infoHeader = "FILENAME\n"
        # somehow gZIP is indicating if  it is the first file name (I think if it is 0 then it is the first file)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename,
                          'r')  # if in python 3, change this to "r" or just "b" can't remember but something about a bit not a string
        else:
            f = open(xDir + xFilename, 'r')

        infoHeader = "FILENAME\n"

        # process - if first time on this car/date, then write header out
        headerNames = sHeader.split(',')
        xdat = str('20') + xFilename[11:17]
        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        fnLog = xOut + xCar + "_" + xdat + "_log.csv"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"
        #

        dtime = open(xDir + xFilename).readlines().pop(2).split(',')[0]
        firstdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                             int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
        #firsttime = firstdate.strftime('%s.%f')
        firsttime = dt_to_epoch(firstdate)
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        if bFirst:
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")
        if not bFirst:
            fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        fOut = open(fnOutTemp, 'w')
        fOut.write(sOutHeader)

        # read all lines
        xCntObs = -1
        xCntGoodValues = 0
        for row in f:
            woo = row
            bGood = True
            if xCntObs !=-1:
                lstS = row.split(",")
                if float(lstS[2])<20:
                    bGood = False
                    xCntObs +=1
            if xCntObs < 0:
                bGood = False
                xCntObs += 1
            if bGood:
                lstS = row.split(",")
                dtime = lstS[0]
                dateob = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                  int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                fdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                 int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                #seconds = fdate.strftime('%s.%f')
                seconds = dt_to_epoch(fdate)
                def getNS(seconds):
                    ns = str(float(seconds) * 1e-3)[11:]
                    # str(pd.to_numeric(str(float(seconds) * 1e-3)[11:]) * 100000)[:9]
                    return (str(ns).ljust(15, '0'))[:9]

                if len(lstS)> 6 and float(lstS[2])>20:
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(
                        dateob.strftime('%H:%M:%S')) + ',' + str(str(float(seconds)*1e-3)[:10]) + ',' + getNS(seconds)+ str(',')
                    csvWrite += str(lstS[20]) + ',' + str(lstS[15]) + ',' + str(lstS[16]) + ',' + str(
                        lstS[17]) + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(
                        lstS[3]) + ',' + str(lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(
                        ',') + str(lstS[14]) + '\n'
                    fOut.write(csvWrite)
                    xCntObs += 1
        infOut.write(str(xFilename) + '\n')
        fOut.close()
        fLog.close()
        infOut.close()
        # print(xCar + "\t" + xdat + "\t" + fnOut[-22:] + "\t" + str(xCntObs) + "\t" + str(xCntGoodValues) + "\t" + str(
        #    gZIP))
        print(f"{xCar} \t {xdat} \t {fnOut[-(17 + len(xCar)):]} \t  {xCntObs} \t {xCntGoodValues} \t {gZIP}")

        wind_df = pd.read_csv(fnOutTemp)
        wind_df_not_null = wind_df.loc[wind_df['LAT'].notnull(),].reset_index(drop=True)
        del (wind_df)
        wind_df = wind_df_not_null.copy()

        radians = False
        wind_df['QUADRANT'] = wind_df.apply(lambda row: get_quadrant(row['U'], row['V']), axis=1)
        wind_df['secnan'] = wind_df.apply(lambda row: row['SECONDS'] + row['NANOSECONDS'] * 1e-9,
                                          axis=1)  # + row['NANOSECONDS']*1e-9,axis=1)
        wind_df['prev_LAT'] = wind_df.LAT.shift(periods=1)
        wind_df['next_LAT'] = wind_df.LAT.shift(periods=-1)
        wind_df['prev_LONG'] = wind_df.LONG.shift(periods=1)
        wind_df['next_LONG'] = wind_df.LONG.shift(periods=-1)
        wind_df['prev_TIME'] = wind_df.secnan.shift(periods=1)
        wind_df['next_TIME'] = wind_df.secnan.shift(periods=-1)
        wind_df['distance'] = wind_df.apply(
            lambda row: haversine(row['prev_LAT'], row['prev_LONG'], row['next_LAT'], row['next_LONG']), axis=1)
        wind_df['bearing'] = wind_df.apply(
            lambda row: calc_bearing(row['prev_LAT'], row['next_LAT'], row['prev_LONG'], row['next_LONG'], radians),
            axis=1)
        wind_df['timediff'] = wind_df.apply(lambda row: row['next_TIME'] - row['prev_TIME'], axis=1)
        # wind_df['VELOCITY_calc'] = wind_df.apply(lambda row:calc_velocity(row['timediff'],row['distance']),axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda x: (str(x.VELOCITY)),axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda x: 0 if x.VELOCITY == 'XX.X' else x.VELOCITY,axis = 1)
        wind_df['fVel'] = wind_df["VELOCITY"].str.split(".", n = 1, expand = True)[0]
        wind_df = wind_df.loc[wind_df['fVel'].notnull(),].reset_index(drop=True)
        wind_df['firstVel'] = wind_df.apply(lambda x: int(x['fVel']),axis = 1)

        wind_df['sVel'] = wind_df["VELOCITY"].str.split(".", n = 1, expand = True)[1]
        wind_df = wind_df.loc[wind_df['sVel'].notnull(),].reset_index(drop=True)
        wind_df['secVel'] = wind_df.apply(lambda x: int(x['sVel']),axis = 1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstVel) + '.' + str(x.secVel)),axis = 1)
        wind_df2 = wind_df.drop(columns = ['VELOCITY','secVel','sVel','fVel','firstVel'])
        del(wind_df)
        wind_df2 = wind_df2.rename(columns = {'vloc':'VELOCITY'})
        wind_df = wind_df2.copy()
        del(wind_df2)
        ## CORRECT W WIND THING
        wind_df['W'] = wind_df.apply(lambda x: (str(x.W)),axis=1)
        wind_df['W'] = wind_df.apply(lambda x: 0 if x.W == 'XX.X' else x.W,axis = 1)
        wind_df['fW'] = wind_df["W"].str.split(".", n = 1, expand = True)[0]
        #wind_df = wind_df.loc[wind_df['fW'].notnull(),].reset_index(drop=True)
        wind_df['firstW'] = wind_df.apply(lambda x: int(x['fW']),axis = 1)
        wind_df['sW'] = wind_df["W"].str.split(".", n = 1, expand = True)[1]
        #wind_df = wind_df.loc[wind_df['sW'].notnull(),].reset_index(drop=True)
        wind_df['secW'] = wind_df.apply(lambda x: int(x['sW']),axis = 1)
        wind_df['wloc'] = wind_df.apply(lambda x: float(str(x.firstW) + '.' + str(x.secW)),axis = 1)
        wind_df2 = wind_df.drop(columns = ['W','secW','sW','fW','firstW'])
        del(wind_df)
        wind_df2 = wind_df2.rename(columns = {'wloc':'W'})
        wind_df = wind_df2.copy()
        del(wind_df2)

        ## CORRECT U WIND THING
        wind_df['U'] = wind_df.apply(lambda x: (str(x.U)),axis=1)
        wind_df['U'] = wind_df.apply(lambda x: 0 if x.U == 'XX.X' else x.U,axis = 1)
        wind_df['fU'] = wind_df["U"].str.split(".", n = 1, expand = True)[0]
        wind_df['firstU'] = wind_df.apply(lambda x: int(x['fU']),axis = 1)
        wind_df['sU'] = wind_df["U"].str.split(".", n = 1, expand = True)[1]
        wind_df['secU'] = wind_df.apply(lambda x: int(x['sU']),axis = 1)
        wind_df['uloc'] = wind_df.apply(lambda x: float(str(x.firstU) + '.' + str(x.secU)),axis = 1)
        wind_df2 = wind_df.drop(columns = ['U','secU','sU','fU','firstU'])
        del(wind_df)
        wind_df2 = wind_df2.rename(columns = {'uloc':'U'})
        wind_df = wind_df2.copy()
        del(wind_df2)

        ## CORRECT V WIND THING
        wind_df['V'] = wind_df.apply(lambda x: (str(x.V)),axis=1)
        wind_df['V'] = wind_df.apply(lambda x: 0 if x.V == 'XX.X' else x.V,axis = 1)
        wind_df['fV'] = wind_df["V"].str.split(".", n = 1, expand = True)[0]
        wind_df['firstV'] = wind_df.apply(lambda x: int(x['fV']),axis = 1)
        wind_df['sV'] = wind_df["V"].str.split(".", n = 1, expand = True)[1]
        wind_df['secV'] = wind_df.apply(lambda x: int(x['sV']),axis = 1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstV) + '.' + str(x.secV)),axis = 1)
        wind_df2 = wind_df.drop(columns = ['V','secV','sV','fV','firstV'])
        del(wind_df)
        wind_df2 = wind_df2.rename(columns = {'vloc':'V'})
        wind_df = wind_df2.copy()
        del(wind_df2)

        wind_df['U_cor'] = wind_df.apply(lambda row: float(row['U']) + float(row['VELOCITY']), axis=1)
        wind_df['horz_length'] = wind_df.apply(lambda row: np.sqrt(row['U_cor'] ** 2 + row['V'] ** 2), axis=1)
        wind_df['uncor_theta'] = wind_df.apply(
            lambda row: calc_bearing(row['U_cor'], row['V'], row['QUADRANT'], row['horz_length'], radians), axis=1)
        wind_df['adj_theta'] = wind_df.apply(lambda row: (row['uncor_theta'] + row['bearing']) % 360, axis=1)
        wind_df['totalWind'] = wind_df.apply(lambda row: np.sqrt(row['horz_length'] ** 2 + row['W'] ** 2), axis=1)
        wind_df['phi'] = wind_df.apply(lambda row: np.arctan(row['horz_length']), axis=1)

        wind_df['adj_v'] = wind_df.apply(lambda row: -row['horz_length']* np.cos(row['adj_theta']),axis = 1)
        wind_df['adj_u'] = wind_df.apply(lambda row: row['horz_length']* np.sin(row['adj_theta']),axis = 1)

        ## GO THROUGH WIND
        window_size = 30
        u_series = pd.Series(wind_df['adj_u'])
        u_windows = u_series.rolling(window_size)
        u_averages = pd.DataFrame(u_windows.mean())
        u_averages.columns =['U_avg']
        u_averages['key'] = u_averages.index

        v_series = pd.Series(wind_df['adj_v'])
        v_windows = v_series.rolling(window_size)
        v_averages = pd.DataFrame(v_windows.mean())
        v_averages.columns =['V_avg']
        v_averages['key'] = v_averages.index

        w_series = pd.Series(wind_df['W'])
        w_windows = w_series.rolling(window_size)
        w_averages = pd.DataFrame(w_windows.mean())
        w_averages.columns =['W_avg']
        w_averages['key'] = w_averages.index

        vw_df = w_averages.set_index('key').join(v_averages.set_index('key'))
        vw_df['key'] = vw_df.index
        uvw_df = vw_df.set_index('key').join(u_averages.set_index('key'))
        uvw_df['key'] = uvw_df.index
        wind_df2 = wind_df.copy()
        wind_df2['key'] = wind_df2.index
        wind_df = uvw_df.set_index('key').join(wind_df2.set_index('key'))

        wind_df['r_avg'] = wind_df.apply(lambda row: np.sqrt(row['U_avg']**2 + row['V_avg']**2),axis=1)
        wind_df['theta_avg'] = wind_df.apply(lambda row: np.arctan(-row['U_avg']/row['V_avg']),axis=1)

        wind_df['shift_CH4'] = wind_df.CH4.shift(periods=int(float(shift)))
        wind_df['raw_CH4'] = wind_df.apply(lambda row: row['BCH4'], axis=1)
        wind_df['BCH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['CH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['TCH4'] = wind_df.loc[:, ['shift_CH4']]

        wind_df['shift_R'] = wind_df.R.shift(periods=int(float(shift)))
        wind_df['raw_R'] = wind_df.apply(lambda row: row['R'], axis=1)

        wind_df2 = wind_df[wind_df.CH4.notnull()]

        wind_df2 = wind_df.copy()
        wind_df3 = wind_df2.drop(
            ['QUADRANT', 'secnan', 'prev_LAT', 'next_LAT', 'prev_LONG', 'next_LONG', 'prev_TIME', 'next_TIME',
             'timediff', 'uncor_theta', 'CH4','R'], axis=1)
        wind_df3['CH4'] = wind_df3.loc[:, 'shift_CH4']
        wind_df3['R'] = wind_df3.loc[:, 'shift_R']
        wind_df3 = wind_df3.drop(['shift_CH4','shift_R'], axis=1)
        # wind_df4 = wind_df3.loc[wind_df3.totalWind.notnull(),:]
        wind_df3['odometer'] = wind_df3.loc[:, 'distance'].cumsum()
        wind_df4 = wind_df3.loc[:,
                   ['DATE', 'TIME', 'SECONDS', 'NANOSECONDS', 'VELOCITY', 'U', 'V', 'W', 'BCH4', 'BRSSI', 'TCH4',
                    'TRSSI',
                    'PRESS_MBAR', 'INLET', 'TEMPC', 'CH4', 'H20', 'C2H6', 'R', 'C2C1', 'BATTV', 'POWMV', 'CURRMA',
                    'SOCPER',
                    'LAT', 'LONG', 'bearing', 'U_cor', 'horz_length', 'adj_theta', 'totalWind',
                    'phi', 'raw_CH4','raw_R','U_avg','V_avg','W_avg','r_avg','theta_avg','distance','odometer']]

        # wind_df7 = add_odometer(wind_df4,'LAT','LONG')

        # wind_df4 = wind_df7.copy()
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > xMinCarSpeed, :]
        wind_df6 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :]

        #wind_df6 = wind_df6a.loc[wind_df6a.R > .6999, :]

        del (wind_df4)
        wind_df4 = wind_df6.copy().drop_duplicates()
        wind_df5 = wind_df4.loc[wind_df4.CH4.notnull(), :]
        wind_df4 = wind_df5.copy()
        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
        os.remove(fnOutTemp)
        return True
    except ValueError:
        return False

def process_raw_data_aerisold(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack, shift, maxSpeed='45',
                        minSpeed='2'):
    """ input a raw .txt file with data (from aeris data file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import pandas as pd
    from datetime import datetime
    import os
    import gzip
    import numpy as np
    # import csv
    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)

        ########################################################################
        #### WE DON'T HAVE AN RSSI INPUT
        ### (SO THIS IS A PLACEHOLDER FOR SOME SORT OF QA/QC VARIABLE)
        ##  xMinRSSI = 50  #if RSSI is below this we don't like it
        ##################################################################

        # reading in the data with specific headers
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        sHeader = 'Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude,U (m/sec),V (m/sec),W (m/sec),T (degC),Dir (deg),Speed (m/sec),Compass (deg)'
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"
        infoHeader = "FILENAME\n"
        # somehow gZIP is indicating if  it is the first file name (I think if it is 0 then it is the first file)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename,
                          'r')  # if in python 3, change this to "r" or just "b" can't remember but something about a bit not a string
        else:
            f = open(xDir + xFilename, 'r')

        infoHeader = "FILENAME\n"

        # process - if first time on this car/date, then write header out
        headerNames = sHeader.split(',')
        xdat = str('20') + xFilename[11:17]
        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        fnLog = xOut + xCar + "_" + xdat + "_log.csv"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"
        #

        dtime = open(xDir + xFilename).readlines().pop(1).split(',')[0]
        firstdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                             int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
        #firsttime = firstdate.strftime('%s.%f')
        firsttime = dt_to_epoch(firstdate)
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        if bFirst:
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")
        if not bFirst:
            fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        fOut = open(fnOutTemp, 'w')
        fOut.write(sOutHeader)

        # read all lines
        xCntObs = -1
        xCntGoodValues = 0
        for row in f:
            # print(row)
            bGood = True
            if xCntObs < 0:
                bGood = False
                xCntObs += 1
            if bGood:
                lstS = row.split(",")
                dtime = lstS[0]
                dateob = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                  int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                fdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                 int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                #seconds = fdate.strftime('%s.%f')
                seconds = dt_to_epoch(fdate)

                import sys
                if sys.platform.startswith('win'):
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(
                        dateob.strftime('%H:%M:%S')) + ',' + str(
                        int(pd.to_numeric(dateob.strftime('%S.%f')))) + ',' + str(
                        pd.to_numeric(dateob.strftime('%f')) * 1000) + str(',')
                    csvWrite += str('50') + ',' + str('0') + ',' + str('0') + ',' + str('0') + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(
                        lstS[3]) + ',' + str(lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(
                        ',') + str(lstS[14])
                if not sys.platform.startswith('win'):
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(
                        dateob.strftime('%H:%M:%S')) + ',' + str(seconds[:10]) + ',' + str(
                        pd.to_numeric(seconds[11:]) * 1000) + str(',')
                    csvWrite += str(lstS[20]) + ',' + str(lstS[15]) + ',' + str(lstS[16]) + ',' + str(
                        lstS[17]) + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(
                        lstS[3]) + ',' + str(lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(
                        ',') + str(lstS[14]) + '\n'
                fOut.write(csvWrite)
                xCntObs += 1
            infOut.write(str(xFilename) + '\n')
        fOut.close()
        fLog.close()
        infOut.close()
        # print(xCar + "\t" + xdat + "\t" + fnOut[-22:] + "\t" + str(xCntObs) + "\t" + str(xCntGoodValues) + "\t" + str(
        #    gZIP))
        print(f"{xCar} \t {xdat} \t {fnOut[-(17 + len(xCar)):]} \t  {xCntObs} \t {xCntGoodValues} \t {gZIP}")

        wind_df = pd.read_csv(fnOutTemp)
        wind_df_not_null = wind_df.loc[wind_df['LAT'].notnull(),].reset_index(drop=True)
        del (wind_df)
        wind_df = wind_df_not_null.copy()

        radians = False
        wind_df['QUADRANT'] = wind_df.apply(lambda row: get_quadrant(row['U'], row['V']), axis=1)
        wind_df['secnan'] = wind_df.apply(lambda row: row['SECONDS'] + row['NANOSECONDS'] * 1e-9,
                                          axis=1)  # + row['NANOSECONDS']*1e-9,axis=1)
        wind_df['prev_LAT'] = wind_df.LAT.shift(periods=1)
        wind_df['next_LAT'] = wind_df.LAT.shift(periods=-1)
        wind_df['prev_LONG'] = wind_df.LONG.shift(periods=1)
        wind_df['next_LONG'] = wind_df.LONG.shift(periods=-1)
        wind_df['prev_TIME'] = wind_df.secnan.shift(periods=1)
        wind_df['next_TIME'] = wind_df.secnan.shift(periods=-1)
        wind_df['distance'] = wind_df.apply(
            lambda row: haversine(row['prev_LAT'], row['prev_LONG'], row['next_LAT'], row['next_LONG']), axis=1)

        wind_df['bearing'] = wind_df.apply(
            lambda row: calc_bearing(row['prev_LAT'], row['next_LAT'], row['prev_LONG'], row['next_LONG'], radians),
            axis=1)
        wind_df['timediff'] = wind_df.apply(lambda row: row['next_TIME'] - row['prev_TIME'], axis=1)
        # wind_df['VELOCITY_calc'] = wind_df.apply(lambda row:calc_velocity(row['timediff'],row['distance']),axis=1)
        wind_df['U_cor'] = wind_df.apply(lambda row: row['U'] + row['VELOCITY'], axis=1)
        wind_df['horz_length'] = wind_df.apply(lambda row: np.sqrt(row['U_cor'] ** 2 + row['V'] ** 2), axis=1)
        wind_df['uncor_theta'] = wind_df.apply(
            lambda row: calc_bearing(row['U_cor'], row['V'], row['QUADRANT'], row['horz_length'], radians), axis=1)
        wind_df['adj_theta'] = wind_df.apply(lambda row: (row['uncor_theta'] + row['bearing']) % 360, axis=1)
        wind_df['totalWind'] = wind_df.apply(lambda row: np.sqrt(row['horz_length'] ** 2 + row['W'] ** 2), axis=1)
        wind_df['phi'] = wind_df.apply(lambda row: np.arctan(row['horz_length']), axis=1)
        wind_df['shift_CH4'] = wind_df.CH4.shift(periods=int(float(shift)))
        wind_df['raw_CH4'] = wind_df.apply(lambda row: row['BCH4'], axis=1)
        wind_df['BCH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['CH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['TCH4'] = wind_df.loc[:, ['shift_CH4']]

        wind_df2 = wind_df[wind_df.CH4.notnull()]
        wind_df2 = wind_df.copy()
        wind_df3 = wind_df2.drop(
            ['QUADRANT', 'secnan', 'prev_LAT', 'next_LAT', 'prev_LONG', 'next_LONG', 'prev_TIME', 'next_TIME',
             'timediff', 'uncor_theta', 'CH4'], axis=1)

        wind_df3['CH4'] = wind_df3.loc[:, 'shift_CH4']
        wind_df3 = wind_df3.drop(['shift_CH4'], axis=1)
        # wind_df4 = wind_df3.loc[wind_df3.totalWind.notnull(),:]
        wind_df3 = wind_df3.loc[:,
                   ['DATE', 'TIME', 'SECONDS', 'NANOSECONDS', 'VELOCITY', 'U', 'V', 'W', 'BCH4', 'BRSSI', 'TCH4',
                    'TRSSI',
                    'PRESS_MBAR', 'INLET', 'TEMPC', 'CH4', 'H20', 'C2H6', 'R', 'C2C1', 'BATTV', 'POWMV', 'CURRMA',
                    'SOCPER',
                    'LAT', 'LONG', 'bearing', 'U_cor', 'horz_length', 'adj_theta', 'totalWind', 'phi', 'raw_CH4',
                    'distance']]
        wind_df3['odometer'] = wind_df3.loc[:, 'distance'].cumsum()
        wind_df4 = wind_df3.copy()

        # wind_df7 = add_odometer(wind_df4,'LAT','LONG')

        # wind_df4 = wind_df7.copy()
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > xMinCarSpeed, :]
        wind_df6 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :]

        del (wind_df4)
        wind_df4 = wind_df6.copy().drop_duplicates()
        wind_df5 = wind_df4.loc[wind_df4.CH4.notnull(), :]
        wind_df4 = wind_df5.copy()
        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
        os.remove(fnOutTemp)
        return True
    except ValueError:
        return False

def add_odometer(df, lat, lon):
    """ add column with running odometer
    input:
        df
        lat: name of latitude column
        lon: name of longitude column
    output:
        df with odometer reading
    """
    import pandas as pd
    import math
    df_use = df.loc[:, [(lat), (lon)]]
    df_use['prev_LAT'] = df_use.loc[:, (lat)].shift(periods=1)
    df_use['prev_LON'] = df_use.loc[:, (lon)].shift(periods=1)
    df_use['distance2'] = df_use.apply(lambda row: haversine(row['prev_LAT'], row['prev_LON'], row[(lat)], row[(lon)]),
                                       axis=1)
    df_use = df_use.reset_index(drop=True)
    df_use.loc[:, 'distance'] = df_use.apply(lambda x: nanthing(x.distance2), axis=1)
    df_use['prev_dist'] = df_use.loc[:, 'distance'].shift(periods=1)
    df_use['odometer'] = df_use['distance'].cumsum()
    df_use['prevod'] = df_use.loc[:, 'odometer'].shift(periods=1)
    df_use['dif'] = df_use.apply(lambda x: x.odometer - x.prevod, axis=1)
    df_use['dif'] = df_use.apply(lambda x: nanthing(x.dif), axis=1)
    return (pd.merge(df, df_use.loc[:, [(lat), (lon), 'odometer', 'distance']], on=[(lat), (lon)]))

def str_list(x):
    """ convert a string of a list to just a list
    input:
        string of a list thing
    output:
        list
    """
    #import ast
    #x = ast.literal_eval(x)
    x = x.strip('][').split(', ')
    x1 = [n.strip('\'') for n in x]
    return (x1)

def str_list_works(x):
    """ convert a string of a list to just a list
    input:
        string of a list thing
    output:
        list
    """
    import ast
    x = ast.literal_eval(x)
    x = [n.strip() for n in x]
    return (x)

def count_times(opList, xCar):
    """ count number of times a peak seen (not in same 5 min period)
    input:
        list of peak times in a given combined peak
    output:
        counts # of times peaks seen not in same 5 min period
    """
    try:
        if isinstance(opList, str):
            opList = str_list(opList)
        if len(opList) == 1:
            numtimes = 1
            return (numtimes)
        else:
            opList.sort()
            numtimes = 1
            index = 1
            for x in opList:
                if index == 1:
                    initTime = float(x[len(xCar) + 1:])
                    initStart = initTime
                    initEnd = initTime + 300
                    index = index + 1
                if index != 1:
                    curTime = float(x[len(xCar) + 1:])
                    within = curTime < initEnd
                    if curTime < initEnd:
                        index = index + 1
                    elif curTime >= initEnd:
                        numtimes = numtimes + 1
                        initTime = curTime
                        initStart = initTime
                        initEnd = curTime + 300
                        index = index + 1
            return (numtimes)
    except:
        return(0)


def nameFiles(outDir, processedFileLoc, xCar, xDate, SC):
    if SC:
        fnOut = outDir + "Peaks" + "_" + xCar + "_" + xDate + ".csv"
        fnShape = outDir + "Peaks" + "_" + xCar + "_" + xDate + ".shp"
        fnLog = outDir + "Peaks" + "_" + xCar + "_" + xDate + ".log"
        pkLog = outDir + "Peaks" + "_" + xCar + "_" + xDate + "_info.csv"
        jsonOut = outDir + "Peaks" + "_" + xCar + "_" + xDate + ".geojson"
        infOut = processedFileLoc + xCar + "_" + xDate + "_info.csv"

    elif not SC:
        fnOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".csv"
        fnShape = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".shp"
        fnLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".log"
        pkLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + "_info.csv"
        jsonOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".geojson"
        infOut = processedFileLoc + xCar + "_" + xDate.replace("-", "") + "_info.csv"

    filenames = {"fnOut": fnOut, 'fnShape': fnShape, 'fnLog': fnLog,
                 'pkLog': pkLog, 'jsonOut': jsonOut, 'infOut': infOut}
    return (filenames)


def identify_peaks_nowind(xCar, xDate, xDir, xFilename, outDir, processedFileLoc, Engineering, threshold='.1',
                   rthresh = '.7',
                  xTimeThreshold='5.0', minElevated='2', xB='102', basePerc='50'):
    """ input a processed data file, and finds the locations of the elevated readings (observed peaks)
    input:
        xCar: name of the car (to make filename)
        xDate: date of the reading
        xDir: directory where the file is located
        xFilename: name of the file
        outDir: directory to take it
        processedFileLoc
        Engineering: T/F if the processed file was made using an engineering file
        threshold: the proportion above baseline that is marked as elevated (i.e. .1 corresponds to 10% above
        xTimeThreshold: not super sure
        minElevated: # of elevated readings that need to be there to constitute an observed peak
        xB: Number of observations used in background average
        basePerc: percentile used for background average (i.e. 50 is median)
    output:
        saved log file
        saved csv file with identified peaks
        saved info.csv file
        saved json file
    """
    import csv, numpy
    import shutil
    from shapely.geometry import Point
    import pandas as pd
    import geopandas as gpd


    try:
        baseCalc = float(basePerc)
        xABThreshold = float(threshold)
        minElevated = float(minElevated)
        rMin = float(rthresh)
        xDistThreshold = 160.0  # find the maximum CH4 reading of observations within street segments of this grouping distance in meters
        xSDF = 4  # multiplier times standard deviation for floating baseline added to mean

        xB = int(xB)
        xTimeThreshold = float(xTimeThreshold)
        fn = xDir + xFilename  # set raw text file to read in
        fnOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".csv"
        fnShape = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".shp"
        fnLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".log"
        pkLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + "_info.csv"
        jsonOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + ".geojson"
        infOut = processedFileLoc + xCar + "_" + xDate.replace("-", "") + "_info.csv"

        ### TEST THING
        fn = xDir + xFilename  # set raw text file to read in
        filenames = nameFiles(outDir,processedFileLoc,xCar,xDate,True)
        fnOut = filenames['fnOut']
        fnShape = filenames['fnShape']
        fnLog = filenames['fnLog']
        pkLog = filenames['pkLog']
        jsonOut = filenames['jsonOut']
        infOut = filenames['infOut']

        print(f"{outDir}Peaks_{xCar}_{xDate}_info.csv")
        fLog = open(fnLog, 'w')
        shutil.copy(infOut, pkLog)

        # field column indices for various variables
        if Engineering == True:
            fDate = 0;  fTime = 1; fEpochTime = 2
            fNanoSeconds = 3; fVelocity = 4;  fU = 5
            fV = 6; fW = 7; fBCH4 = 10
            fBCH4 = 8;  fBRSSI = 9; fTCH4 = 10
            TRSSI = 11;PRESS = 12; INLET = 13
            TEMP = 14;  CH4 = 15;H20 = 16
            C2H6 = 17;  R = 18;  C2C1 = 19
            BATT = 20;  POWER = 21; CURR = 22
            SOCPER = 23;fLat = 24; fLon = 25
        elif not Engineering:
            fDate = 0; fTime = 1; fEpochTime = 2
            fNanoSeconds = 3;fVelocity = 4; fU = 5
            fV = 6;  fW = 7
            fBCH4 = 8; fBRSSI = 9
            fTCH4 = 10;  TRSSI = 11;  PRESS = 12
            INLET = 13;  TEMP = 14; CH4 = 15
            H20 = 16;C2H6 = 17;  R = 18; C2C1 = 19
            BATT = 20; POWER = 21; CURR = 22
            SOCPER = 23; fLat = 24;fLon = 25; fDist = 33;  fOdometer = 34
            fUavg = 35; fVavg = 36; fWavg = 37

            # read data in from text file and extract desired fields into a list, padding with 5 minute and hourly average
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,x11,x12,x13,x14,x15,x16 = [[] for _ in range(16)]

            count = -1
            with open(fn, 'r') as f:
                t = csv.reader(f)
                for row in t:
                    woo = row
                    # print(count)
                    if count < 0:
                        count += 1
                        continue
                    elif count >= 0:
                        datet = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                        ## if not engineering
                        epoch = float(row[fEpochTime] + "." + row[fNanoSeconds][0])
                        datetime = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                        x1.append(epoch); x2.append(datetime)
                        if row[fLat] == '':
                            x3.append('')
                        elif row[fLat] != '':
                            x3.append(float(row[fLat]))
                        if row[fLon] == '':
                            x4.append('')
                        elif row[fLon] != '':
                            x4.append(float(row[fLon]))

                        x5.append(float(row[fBCH4]))
                        x6.append(float(row[fTCH4]))
                        x7.append(0.0)
                        x8.append(0.0)
                        x9.append(row[fOdometer])
                        x11.append(float(row[C2H6]))
                        x12.append(float(row[C2C1]))
                        x13.append(float(row[R]))
                        x14.append(float(row[fUavg]))
                        x15.append(float(row[fVavg]))
                        x16.append(float(row[fWavg]))

                        count += 1
            print(f"Number of observations processed:{count}")

        # convert lists to numpy arrays
        aEpochTime = numpy.array(x1)
        aDateTime = numpy.array(x2)
        aLat = numpy.array(x3)
        aLon = numpy.array(x4)
        aCH4 = numpy.array(x5)
        aTCH4 = numpy.array(x6)
        aMean = numpy.array(x7)
        arealMean = numpy.array(x7)
        astd = numpy.array(x7)

        aMeanC2H6 = numpy.array(x7)
        aThreshold = numpy.array(x8)
        aOdom = numpy.array(x9)

        # adding ethane stuff
        aC2H6 = numpy.array(x11)
        aC2C1 = numpy.array(x12)
        aR = numpy.array(x13)
        aUavg = numpy.array(x14)
        aVavg = numpy.array(x15)
        aWavg = numpy.array(x16)


        xLatMean = numpy.mean(aLat)
        xLonMean = numpy.mean(aLon)
        #xCH4Mean = numpy.mean(aCH4)
        #xC2H6Mean = numpy.mean(aC2H6)
        #xC2C1Mean = numpy.mean(aC2C1)

        fLog.write("Day CH4_mean = " + str(numpy.mean(aCH4)) +
                   ", Day CH4 SD = " + str(numpy.std(aCH4)) + "\n")
        fLog.write("Day C2H6 Mean = " + str(numpy.mean(aC2H6)) +
                   ", Day C2H6 SD = " + str(numpy.std(aC2H6)) + "\n")
        fLog.write("Center lon/lat = " + str(xLonMean) + ", " + str(xLatMean) + "\n")

        lstCH4_AB = []

        # generate list of the index for observations that were above the threshold
        for i in range(0, count - 2):
            if ((count - 2) > xB):
                topBound = min((i + xB), (count - 2))
                botBound = max((i - xB), 0)

                for t in range(min((i + xB), (count - 2)), i, -1):
                    if aEpochTime[t] < (aEpochTime[i] + (xB / 2)):
                        topBound = t
                        break
                for b in range(max((i - xB), 0), i):
                    if aEpochTime[b] > (aEpochTime[i] - (xB / 2)):
                        botBound = b
                        break

                xCH4Mean = numpy.percentile(aCH4[botBound:topBound], baseCalc)
                xCH4_actualMean = numpy.mean(aCH4[botBound:topBound])
                xCH4_stdev = numpy.mean(aCH4[botBound:topBound])
                xC2H6Mean = numpy.percentile(aC2H6[botBound:topBound], baseCalc)

            # xCH4SD = numpy.std(aCH4[botBound:topBound])
            else:
                xCH4Mean = numpy.percentile(aCH4[0:(count - 2)], baseCalc)
                xCH4_actualMean = numpy.mean(aCH4[0:(count - 2)], baseCalc)
                xCH4_stdev = numpy.std(aCH4[0:(count - 2)], baseCalc)
                xC2H6Mean = numpy.percentile(aC2H6[0:(count - 2)], baseCalc)

                # xCH4SD = numpy.std(aCH4[0:(count-2)])
            xThreshold = xCH4Mean + (xCH4Mean * xABThreshold)
            xThreshold_c2h6 = xC2H6Mean + (xC2H6Mean * xABThreshold)

            if (aCH4[i] > xThreshold and aR[i]>rMin):
            #if (aCH4[i] > xThreshold):
                lstCH4_AB.append(i)
                aMean[i] = xCH4Mean
                aMeanC2H6[i] = xC2H6Mean
                aThreshold[i] = xThreshold
                arealMean[i] = xCH4_actualMean
                astd[i] = xCH4_stdev

        # now group the above baseline threshold observations into groups based on distance threshold
        lstCH4_ABP = []
        xDistPeak = 0.0
        xCH4Peak = 0.0
        xTime = 0.0
        cntPeak = 0
        cnt = 0
        sID = ""
        sPeriod5Min = ""
        prevIndex = 0
        for i in lstCH4_AB:
            if (cnt == 0):
                xLon1 = aLon[i]
                xLat1 = aLat[i]
                xOdom = aOdom[i]
            else:
                # calculate distance between points
                xDist = haversine(xLat1, xLon1, aLat[i], aLon[i])
                xDistPeak += xDist
                xCH4Peak += (xDist * (aCH4[i] - aMean[i]))
                xLon1 = aLon[i]
                xLat1 = aLat[i]
                xOdom = aOdom[i]
                if (sID == ""):
                    xTime = aEpochTime[i]
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                if ((aEpochTime[i] - aEpochTime[prevIndex]) > xTimeThreshold):  # initial start of a observed peak
                    cntPeak += 1
                    xTime = aEpochTime[i]
                    xDistPeak = 0.0
                    xCH4Peak = 0.0
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                    # print str(i) +", " + str(xDist) + "," + str(cntPeak) +"," + str(xDistPeak)
                lstCH4_ABP.append(
                    [sID, xTime, aEpochTime[i], aDateTime[i], aCH4[i], aLon[i], aLat[i], aMean[i], aThreshold[i],
                     xDistPeak, xCH4Peak, aTCH4[i],aC2H6[i],aC2C1[i],aR[i],aMeanC2H6[i], sPeriod5Min, xOdom,
                     aUavg[i],aVavg[i],aWavg[i]])
            cnt += 1
            prevIndex = i

        # Finding peak_id larger than 160.0 m
        tmpsidlist = []
        for r in lstCH4_ABP:
            if (float(r[9]) > 160.0) and (r[0] not in tmpsidlist):
                tmpsidlist.append(r[0])
        cntPeak -= len(tmpsidlist)

        fLog.write("Number of peaks found: " + str(cntPeak) + "\n")
        print(f"{xCar} \t {xDate} \t {xFilename} \t {count} \t {len(lstCH4_ABP)}")

        # write out the observed peaks to a csv to be read into a GIS
        fOut = open(fnOut, 'w')
        # s = "PEAK_NUM,EPOCHSTART,EPOCH,DATETIME,CH4,LON,LAT,CH4_BASELINE,CH4_THRESHOLD,PEAK_DIST_M,PEAK_CH4,TCH4,PERIOD5MIN\n"
        s = "OP_NUM,OP_EPOCHSTART,OB_EPOCH,OB_DATETIME,OB_CH4,OB_LON,OB_LAT,OB_CH4_BASELINE," \
            "OB_CH4_THRESHOLD,OP_PEAK_DIST_M,OP_PEAK_CH4,OB_TCH4,OB_C2H6," \
            "OB_C2C1,OB_R,OB_C2H6_BASELINE,OB_PERIOD5MIN,ODOMETER,OB_U_AVG,OB_V_AVG,OB_W_AVG\n"
        fOut.write(s)

        truecount = 0
        for r in lstCH4_ABP:
            if r[0] not in tmpsidlist:
                s = ''
                for rr in r:
                    s += str(rr) + ','
                s = s[:-1]
                s += '\n'
                fOut.write(s)
                truecount += 1
        fOut.close()
        fLog.close()

        openFile = pd.read_csv(fnOut)
        if openFile.shape[0] != 0:
            pkDistDf = openFile.copy().groupby('OP_NUM', as_index=False).apply(
                lambda x: max(x.ODOMETER) - min(x.ODOMETER))
            pkDistDf.columns = ['OP_NUM', 'OP_DISTANCE']
            openFile = pd.merge(openFile.copy(), pkDistDf)
            tempCount = openFile.groupby('OP_NUM', as_index=False).OP_EPOCHSTART.count().rename(
                columns={'OP_EPOCHSTART': 'Frequency'})
            tempCount = tempCount.loc[tempCount.Frequency >= minElevated, :]
            if tempCount.shape[0] == 0:
                print(f"No Observed Peaks with enough Elevated Readings Found in the file: {xFilename}")

            elif tempCount.shape[0] != 0:
                oFile = pd.merge(openFile, tempCount, on=['OP_NUM'])
                openFile = oFile.copy()
                del (oFile)
                openFile["minElevated"] = openFile.apply(lambda x: int(minElevated), axis=1)
                openFile['OB_CH4_AB'] = openFile.loc[:, 'OB_CH4'].sub(openFile.loc[:, 'OB_CH4_BASELINE'], axis=0)
                openFile['OB_C2H6_AB'] = openFile.loc[:, 'OB_C2H6'].sub(openFile.loc[:, 'OB_C2H6_BASELINE'],axis=0)
                openFile.to_csv(fnOut, index=False)


                fileWt = weighted_loc(openFile, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB').loc[:, :].rename(
                    columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'}).reset_index(drop=True)
                geometry_temp = [Point(lon, lat) for lon, lat in zip(fileWt['pk_LON'], fileWt['pk_LAT'])]
                crs = 'EPSG:4326'
                # geometry is the point of the lat/lon
                # gdf_buff = gpd.GeoDataFrame(datFram, crs=crs, geometry=geometry_temp)

                ## BUFFER AROUND EACH 'OP_NUM' WITH BUFFER DISTANCE
                gdf_buff = gpd.GeoDataFrame(fileWt, crs=crs, geometry=geometry_temp)
                # gdf_buff = makeGPD(datFram,'LON','LAT')

                ##maybe this is the issue?
                #gdf_buff = gdf_buff.to_crs(epsg=32610)
                #gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(30)
                try:
                    gdf_buff.to_file(jsonOut, driver="GeoJSON")
                    #gdf_buff.to_file('testthing.geojson', driver="GeoJSON")
                except:
                    print("Error Saving JSON File")
        elif openFile.shape[0] == 0:
            print(f"No Observed Peaks Found in the file:{xFilename}")
    except ValueError:
        print("Error in Identify Peaks")
        return False


def identify_peaks(xCar, xDate, xDir, xFilename, outDir, processedFileLoc, Engineering, threshold='.1',
                   rthresh = '.7',
                  xTimeThreshold='5.0', minElevated='2', xB='102', basePerc='50'):
    """ input a processed data file, and finds the locations of the elevated readings (observed peaks)
    input:
        xCar: name of the car (to make filename)
        xDate: date of the reading
        xDir: directory where the file is located
        xFilename: name of the file
        outDir: directory to take it
        processedFileLoc
        Engineering: T/F if the processed file was made using an engineering file
        threshold: the proportion above baseline that is marked as elevated (i.e. .1 corresponds to 10% above
        xTimeThreshold: not super sure
        minElevated: # of elevated readings that need to be there to constitute an observed peak
        xB: Number of observations used in background average
        basePerc: percentile used for background average (i.e. 50 is median)
    output:
        saved log file
        saved csv file with identified peaks
        saved info.csv file
        saved json file
    """
    import csv, numpy
    import shutil
    from shapely.geometry import Point
    import pandas as pd
    import geopandas as gpd


    try:
        baseCalc = float(basePerc)
        xABThreshold = float(threshold)
        minElevated = float(minElevated)
        rMin = float(rthresh)
        xDistThreshold = 160.0  # find the maximum CH4 reading of observations within street segments of this grouping distance in meters
        xSDF = 4  # multiplier times standard deviation for floating baseline added to mean

        xB = int(xB)
        xTimeThreshold = float(xTimeThreshold)
        fn = xDir + xFilename  # set processed csv file to read in
        fnOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".csv"
        fnShape = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".shp"
        fnLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".log"
        pkLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + "_info.csv"
        jsonOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + ".geojson"
        infOut = processedFileLoc + xCar + "_" + xDate.replace("-", "") + "_info.csv"

        ### TEST THING
        fn = xDir + xFilename  # set raw text file to read in
        filenames = nameFiles(outDir,processedFileLoc,xCar,xDate,True)
        fnOut = filenames['fnOut']
        fnShape = filenames['fnShape']
        fnLog = filenames['fnLog']
        pkLog = filenames['pkLog']
        jsonOut = filenames['jsonOut']
        infOut = filenames['infOut']

        print(f"{outDir}Peaks_{xCar}_{xDate}_info.csv")
        fLog = open(fnLog, 'w')
        shutil.copy(infOut, pkLog)

        # field column indices for various variables
        if Engineering == True:
            fDate = 0;  fTime = 1; fEpochTime = 2
            fNanoSeconds = 3; fVelocity = 4;  fU = 5
            fV = 6; fW = 7; fBCH4 = 10
            fBCH4 = 8;  fBRSSI = 9; fTCH4 = 10
            TRSSI = 11;PRESS = 12; INLET = 13
            TEMP = 14;  CH4 = 15;H20 = 16
            C2H6 = 17;  R = 18;  C2C1 = 19
            BATT = 20;  POWER = 21; CURR = 22
            SOCPER = 23;fLat = 24; fLon = 25
        elif not Engineering:
            fDate = 0; fTime = 1; fEpochTime = 2
            fNanoSeconds = 3;fVelocity = 4; fU = 5
            fV = 6;  fW = 7
            fBCH4 = 8; fBRSSI = 9
            fTCH4 = 10;  TRSSI = 11;  PRESS = 12
            INLET = 13;  TEMP = 14; CH4 = 15
            H20 = 16;C2H6 = 17;  R = 18; C2C1 = 19
            BATT = 20; POWER = 21; CURR = 22
            SOCPER = 23; fLat = 24;fLon = 25;
            fUavg = 33; fVavg = 34; fWavg = 35;
            fRavg = 36; fthetavg=37;
            fDist = 38; fOdometer = 39

            # read data in from text file and extract desired fields into a list, padding with 5 minute and hourly average
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,x11,x12,x13,x14,x15,x16,x17,x18 = [[] for _ in range(18)]

            count = -1
            with open(fn, 'r') as f:
                t = csv.reader(f)
                for row in t:
                    woo = row
                    # print(count)
                    if count < 0:
                        count += 1
                        continue
                    elif count >= 0:
                        datet = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                        ## if not engineering
                        epoch = float(row[fEpochTime] + "." + row[fNanoSeconds][0])
                        datetime = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                        x1.append(epoch); x2.append(datetime)
                        if row[fLat] == '':
                            x3.append('')
                        elif row[fLat] != '':
                            x3.append(float(row[fLat]))
                        if row[fLon] == '':
                            x4.append('')
                        elif row[fLon] != '':
                            x4.append(float(row[fLon]))

                        if row[fUavg] == '':
                            x14.append('')
                        elif row[fUavg] != '':
                            x14.append(float(row[fUavg]))
                        if row[fVavg] == '':
                            x15.append('')
                        elif row[fVavg] != '':
                            x15.append(float(row[fVavg]))
                        if row[fWavg] == '':
                            x16.append('')
                        elif row[fWavg] != '':
                            x16.append(float(row[fWavg]))

                        if row[fthetavg] == '':
                            x18.append('')
                        elif row[fthetavg] != '':
                            x18.append(float(row[fthetavg]))
                        if row[fRavg] == '':
                            x17.append('')
                        elif row[fRavg] != '':
                            x17.append(float(row[fRavg]))

                        x5.append(float(row[fBCH4]))
                        x6.append(float(row[fTCH4]))
                        x7.append(0.0)
                        x8.append(0.0)
                        x9.append(row[fOdometer])
                        x11.append(float(row[C2H6]))
                        x12.append(float(row[C2C1]))
                        x13.append(float(row[R]))
                        count += 1
            print(f"Number of observations processed:{count}")

        # convert lists to numpy arrays
        aEpochTime = numpy.array(x1)
        aDateTime = numpy.array(x2)
        aLat = numpy.array(x3)
        aLon = numpy.array(x4)
        aCH4 = numpy.array(x5)
        aTCH4 = numpy.array(x6)
        aMean = numpy.array(x7)
        aMeanC2H6 = numpy.array(x7)
        aThreshold = numpy.array(x8)
        aOdom = numpy.array(x9)

        # adding ethane stuff
        aC2H6 = numpy.array(x11)
        aC2C1 = numpy.array(x12)
        aR = numpy.array(x13)
        aUavg = numpy.array(x14)
        aVavg = numpy.array(x15)
        aWavg = numpy.array(x16)
        aRavg = numpy.array(x17)
        aThavg = numpy.array(x18)


        xLatMean = numpy.mean(aLat)
        xLonMean = numpy.mean(aLon)
        #xCH4Mean = numpy.mean(aCH4)
        #xC2H6Mean = numpy.mean(aC2H6)
        #xC2C1Mean = numpy.mean(aC2C1)

        fLog.write("Day CH4_mean = " + str(numpy.mean(aCH4)) +
                   ", Day CH4 SD = " + str(numpy.std(aCH4)) + "\n")
        fLog.write("Day C2H6 Mean = " + str(numpy.mean(aC2H6)) +
                   ", Day C2H6 SD = " + str(numpy.std(aC2H6)) + "\n")
        fLog.write("Center lon/lat = " + str(xLonMean) + ", " + str(xLatMean) + "\n")

        lstCH4_AB = []

        # generate list of the index for observations that were above the threshold
        for i in range(0, count - 2):
            if ((count - 2) > xB):
                topBound = min((i + xB), (count - 2))
                botBound = max((i - xB), 0)

                for t in range(min((i + xB), (count - 2)), i, -1):
                    if aEpochTime[t] < (aEpochTime[i] + (xB / 2)):
                        topBound = t
                        break
                for b in range(max((i - xB), 0), i):
                    if aEpochTime[b] > (aEpochTime[i] - (xB / 2)):
                        botBound = b
                        break

                xCH4Mean = numpy.percentile(aCH4[botBound:topBound], baseCalc)
                xC2H6Mean = numpy.percentile(aC2H6[botBound:topBound], baseCalc)

            # xCH4SD = numpy.std(aCH4[botBound:topBound])
            else:
                xCH4Mean = numpy.percentile(aCH4[0:(count - 2)], baseCalc)
                xC2H6Mean = numpy.percentile(aC2H6[0:(count - 2)], baseCalc)

                # xCH4SD = numpy.std(aCH4[0:(count-2)])
            xThreshold = xCH4Mean + (xCH4Mean * xABThreshold)
            xThreshold_c2h6 = xC2H6Mean + (xC2H6Mean * xABThreshold)

            if (aCH4[i] > xThreshold and aR[i]>rMin):
            #if (aCH4[i] > xThreshold):
                lstCH4_AB.append(i)
                aMean[i] = xCH4Mean
                aMeanC2H6[i] = xC2H6Mean
                aThreshold[i] = xThreshold

        # now group the above baseline threshold observations into groups based on distance threshold
        lstCH4_ABP = []
        xDistPeak = 0.0
        xCH4Peak = 0.0
        xTime = 0.0
        cntPeak = 0
        cnt = 0
        sID = ""
        sPeriod5Min = ""
        prevIndex = 0
        for i in lstCH4_AB:
            if (cnt == 0):
                xLon1 = aLon[i]
                xLat1 = aLat[i]
                xOdom = aOdom[i]
            else:
                # calculate distance between points
                xDist = haversine(xLat1, xLon1, aLat[i], aLon[i])
                xDistPeak += xDist
                xCH4Peak += (xDist * (aCH4[i] - aMean[i]))
                xLon1 = aLon[i]
                xLat1 = aLat[i]
                xOdom = aOdom[i]
                if (sID == ""):
                    xTime = aEpochTime[i]
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                if ((aEpochTime[i] - aEpochTime[prevIndex]) > xTimeThreshold):  # initial start of a observed peak
                    cntPeak += 1
                    xTime = aEpochTime[i]
                    xDistPeak = 0.0
                    xCH4Peak = 0.0
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                    # print str(i) +", " + str(xDist) + "," + str(cntPeak) +"," + str(xDistPeak)
                lstCH4_ABP.append(
                    [sID, xTime, aEpochTime[i], aDateTime[i], aCH4[i], aLon[i], aLat[i], aMean[i], aThreshold[i],
                     xDistPeak, xCH4Peak, aTCH4[i],aC2H6[i],aC2C1[i],aR[i],aMeanC2H6[i], sPeriod5Min, xOdom,
                     aUavg[i],aVavg[i],aWavg[i],aRavg[i],aThavg[i]])
            cnt += 1
            prevIndex = i

        # Finding peak_id larger than 160.0 m
        tmpsidlist = []
        for r in lstCH4_ABP:
            if (float(r[9]) > 160.0) and (r[0] not in tmpsidlist):
                tmpsidlist.append(r[0])
        cntPeak -= len(tmpsidlist)

        fLog.write("Number of peaks found: " + str(cntPeak) + "\n")
        print(f"{xCar} \t {xDate} \t {xFilename} \t {count} \t {len(lstCH4_ABP)}")

        # write out the observed peaks to a csv to be read into a GIS
        fOut = open(fnOut, 'w')
        # s = "PEAK_NUM,EPOCHSTART,EPOCH,DATETIME,CH4,LON,LAT,CH4_BASELINE,CH4_THRESHOLD,PEAK_DIST_M,PEAK_CH4,TCH4,PERIOD5MIN\n"
        s = "OP_NUM,OP_EPOCHSTART,OB_EPOCH,OB_DATETIME,OB_CH4,OB_LON,OB_LAT,OB_CH4_BASELINE," \
            "OB_CH4_THRESHOLD,OP_PEAK_DIST_M,OP_PEAK_CH4,OB_TCH4,OB_C2H6," \
            "OB_C2C1,OB_R,OB_C2H6_BASELINE,OB_PERIOD5MIN,ODOMETER,OB_U_AVG,OB_V_AVG,OB_W_AVG," \
            "OB_R_AVG,OB_THETA_AVG\n"
        fOut.write(s)

        truecount = 0
        for r in lstCH4_ABP:
            if r[0] not in tmpsidlist:
                s = ''
                for rr in r:
                    s += str(rr) + ','
                s = s[:-1]
                s += '\n'
                fOut.write(s)
                truecount += 1
        fOut.close()
        fLog.close()

        openFile = pd.read_csv(fnOut)
        if openFile.shape[0] != 0:
            pkDistDf = openFile.copy().groupby('OP_NUM', as_index=False).apply(
                lambda x: max(x.ODOMETER) - min(x.ODOMETER))
            pkDistDf.columns = ['OP_NUM', 'OP_DISTANCE']
            openFile = pd.merge(openFile.copy(), pkDistDf)
            tempCount = openFile.groupby('OP_NUM', as_index=False).OP_EPOCHSTART.count().rename(
                columns={'OP_EPOCHSTART': 'Frequency'})
            tempCount = tempCount.loc[tempCount.Frequency >= minElevated, :]
            if tempCount.shape[0] == 0:
                print(f"No Observed Peaks with enough Elevated Readings Found in the file: {xFilename}")
                tempCount.to_csv(fnOut) ## added to deal with issue where it wasn't being filtered out
            elif tempCount.shape[0] != 0:
                oFile = pd.merge(openFile, tempCount, on=['OP_NUM'])
                openFile = oFile.copy()
                del (oFile)
                openFile["minElevated"] = openFile.apply(lambda x: int(minElevated), axis=1)
                openFile['OB_CH4_AB'] = openFile.loc[:, 'OB_CH4'].sub(openFile.loc[:, 'OB_CH4_BASELINE'], axis=0)
                openFile['OB_C2H6_AB'] = openFile.loc[:, 'OB_C2H6'].sub(openFile.loc[:, 'OB_C2H6_BASELINE'],axis=0)
                openFile.to_csv(fnOut, index=False)


                fileWt = weighted_loc(openFile, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB').loc[:, :].rename(
                    columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'}).reset_index(drop=True)
                geometry_temp = [Point(lon, lat) for lon, lat in zip(fileWt['pk_LON'], fileWt['pk_LAT'])]
                crs = 'EPSG:4326'
                # geometry is the point of the lat/lon
                # gdf_buff = gpd.GeoDataFrame(datFram, crs=crs, geometry=geometry_temp)

                ## BUFFER AROUND EACH 'OP_NUM' WITH BUFFER DISTANCE
                gdf_buff = gpd.GeoDataFrame(fileWt, crs=crs, geometry=geometry_temp)
                # gdf_buff = makeGPD(datFram,'LON','LAT')

                ##maybe this is the issue?
                #gdf_buff = gdf_buff.to_crs(epsg=32610)
                #gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(30)
                try:
                    gdf_buff.to_file(jsonOut, driver="GeoJSON")
                    #gdf_buff.to_file('testthing.geojson', driver="GeoJSON")
                except:
                    print("Error Saving JSON File")
        elif openFile.shape[0] == 0:
            print(f"No Observed Peaks Found in the file:{xFilename}")
    except ValueError:
        print("Error in Identify Peaks")
        return False


def identify_peaks_CSU(xCar, xDate, xDir, xFilename, outDir, processedFileLoc, threshold='.1', xTimeThreshold='5.0',minElevated='2', xB='1020', basePerc='50'):
    import csv, numpy
    import geopandas as gpd
    import shutil
    import swifter
    try:
        baseCalc = float(basePerc)
        xABThreshold = float(threshold)
        minElevated = float(minElevated)
        xDistThreshold = 160.0  # find the maximum CH4 reading of observations within street segments of this grouping distance in meters
        xSDF = 4  # multiplier times standard deviation for floating baseline added to mean

        xB = int(xB)
        xTimeThreshold = float(xTimeThreshold)

        fn = xDir + "/" + xFilename  # set raw text file to read in

        filenames = nameFiles(outDir,processedFileLoc,xCar,xDate,False)
        fnOut = filenames['fnOut']
        fnShape = filenames['fnShape']
        fnLog = filenames['fnLog']
        pkLog = filenames['pkLog']
        jsonOut = filenames['jsonOut']
        infOut = filenames['infOut']

        #fnOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + ".csv"
        #fnShape = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".shp"
        #fnLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".log"
        #pkLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + "_info.csv"
        #jsonOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".json"
        #infOut = processedFileLoc + xCar + "_" + xDate.replace("-", "") + "_info.csv"

        print(str(outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + "_info.csv"))
        fLog = open(fnLog, 'w')
        shutil.copy(infOut, pkLog)

        # field column indices for various variables
        fDate = 0; fTime = 1;  fEpochTime = 2; fNanoSeconds = 3
        fLat = 4; fLon = 5; fVelocity = 6; fU = 7; fV = 8; fW = 9; fBCH4 = 10; fTCH4 = 12

        # read data in from text file and extract desired fields into a list, padding with 5 minute and hourly average
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [[] for _ in range(10)]

        count = -1
        with open(fn, 'r') as f:
            t = csv.reader(f)
            for row in t:
                if count < 0:
                    count += 1
                    continue

                datet = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                x1.append(float(str(row[fEpochTime]) + '.' + str(row[fNanoSeconds])))
                x2.append(float(int(datet)))
                x3.append(float(row[fLat]))
                x4.append(float(row[fLon]))
                x5.append(float(row[fBCH4]))
                x6.append(float(row[fTCH4]))
                x7.append(0.0)
                x8.append(0.0)
                count += 1
        print("Number of observations processed: " + str(count))

        # convert lists to numpy arrays
        aEpochTime = numpy.array(x1)
        aDateTime = numpy.array(x2)
        aLat = numpy.array(x3)
        aLon = numpy.array(x4)
        aCH4 = numpy.array(x5)
        aTCH4 = numpy.array(x6)
        aMean = numpy.array(x7)
        aThreshold = numpy.array(x8)
        xLatMean = numpy.mean(aLat)
        xLonMean = numpy.mean(aLon)

        fLog.write("Day CH4_mean = " + str(numpy.mean(aCH4)) + ", Day CH4_SD = " + str(numpy.std(aCH4)) + "\n")
        fLog.write("Center lon/lat = " + str(xLonMean) + ", " + str(xLatMean) + "\n")
        lstCH4_AB = []

        # generate list of the index for observations that were above the threshold
        for i in range(0, count - 2):
            if ((count - 2) > xB):
                topBound = min((i + xB), (count - 2))
                botBound = max((i - xB), 0)

                for t in range(min((i + xB), (count - 2)), i, -1):
                    if aEpochTime[t] < (aEpochTime[i] + (xB / 2)):
                        topBound = t
                        break
                for b in range(max((i - xB), 0), i):
                    if aEpochTime[b] > (aEpochTime[i] - (xB / 2)):
                        botBound = b
                        break

                xCH4Mean = numpy.percentile(aCH4[botBound:topBound], baseCalc)
            else:
                xCH4Mean = numpy.percentile(aCH4[0:(count - 2)], baseCalc)
            xThreshold = xCH4Mean + (xCH4Mean * xABThreshold)

            if (aCH4[i] > xThreshold):
                lstCH4_AB.append(i)
                aMean[
                    i] = xCH4Mean  # insert mean + SD as upper quartile CH4 value into the array to later retreive into the peak calculation
                aThreshold[i] = xThreshold

        # now group the above baseline threshold observations into groups based on distance threshold
        lstCH4_ABP = []
        xDistPeak = 0.0
        xCH4Peak = 0.0
        xTime = 0.0
        cntPeak = 0
        cnt = 0
        sID = ""
        sPeriod5Min = ""
        prevIndex = 0
        for i in lstCH4_AB:
            if (cnt == 0):
                xLon1 = aLon[i];
                xLat1 = aLat[i]
            else:
                # calculate distance between points
                xDist = haversine(xLat1, xLon1, aLat[i], aLon[i])
                xDistPeak += xDist
                xCH4Peak += (xDist * (aCH4[i] - aMean[i]))
                xLon1 = aLon[i];
                xLat1 = aLat[i]
                if (sID == ""):
                    xTime = aEpochTime[i]
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                if ((aEpochTime[i] - aEpochTime[prevIndex]) > xTimeThreshold):  # initial start of a observed peak
                    cntPeak += 1
                    xTime = aEpochTime[i]
                    xDistPeak = 0.0
                    xCH4Peak = 0.0
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                    # print str(i) +", " + str(xDist) + "," + str(cntPeak) +"," + str(xDistPeak)
                lstCH4_ABP.append(
                    [sID, xTime, aEpochTime[i], aDateTime[i], aCH4[i], aLon[i], aLat[i], aMean[i], aThreshold[i],
                     xDistPeak, xCH4Peak, aTCH4[i], sPeriod5Min])
            cnt += 1
            prevIndex = i

        # Finding peak_id larger than 160.0 m
        tmpsidlist = []
        for r in lstCH4_ABP:
            if (float(r[9]) > 160.0) and (r[0] not in tmpsidlist):
                tmpsidlist.append(r[0])
        cntPeak -= len(tmpsidlist)

        fLog.write("Number of peaks found: " + str(cntPeak) + "\n")
        print(xCar + "\t" + xDate + "\t" + xFilename + "\t" + str(count) + "\t" + str(len(lstCH4_ABP)))
        #### calculate attribute for the area under the curve -- PPM

        # write out the observed peaks to a csv to be read into a GIS
        fOut = open(fnOut, 'w')
        s = "OP_NUM,OP_EPOCHSTART,OB_EPOCH,OB_DATETIME,OB_CH4,OB_LON,OB_LAT,OB_CH4_BASELINE,OB_CH4_THRESHOLD,OP_PEAK_DIST_M,OP_PEAK_CH4,OB_TCH4,OB_PERIOD5MIN\n"

        fOut.write(s)

        truecount = 0
        for r in lstCH4_ABP:
            if r[0] not in tmpsidlist:
                s = ''
                for rr in r:
                    s += str(rr) + ','
                s = s[:-1]
                s += '\n'
                fOut.write(s)
                truecount += 1
        fOut.close()
        fLog.close()
        import pandas as pd
        openFile = pd.read_csv(fnOut)
        from shapely.geometry import Point
        if openFile.shape[0] != 0:
            tempCount = openFile.groupby('OP_NUM', as_index=False).OP_EPOCHSTART.count().rename(
                columns={'OP_EPOCHSTART': 'Frequency'})
            tempCount = tempCount.loc[tempCount.Frequency >= minElevated, :]
            if tempCount.shape[0] == 0:
                print("No Observed Peaks with enough Elevated Readings Found in the file: " + str(xFilename))
            elif tempCount.shape[0] != 0:
                oFile = pd.merge(openFile, tempCount, on=['OP_NUM'])
                openFile = oFile.copy()
                del (oFile)
                openFile['minElevated'] = openFile.swifter.apply(lambda x: int(minElevated), axis=1)
                openFile.to_csv(fnOut, index=False)
                openFile['OB_CH4_AB'] = openFile.loc[:, 'OB_CH4'].sub(openFile.loc[:, 'OB_CH4_BASELINE'], axis=0)

                fileWt = weighted_loc(openFile, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB').loc[:, :].rename(
                    columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'}).reset_index(drop=True)
                geometry_temp = [Point(xy) for xy in zip(fileWt['pk_LON'], fileWt['pk_LAT'])]
                #crs = {'init': 'epsg:4326'}
                crs = 'EPSG:4326'

                # geometry is the point of the lat/lon
                # gdf_buff = gpd.GeoDataFrame(datFram, crs=crs, geometry=geometry_temp)

                ## BUFFER AROUND EACH 'OP_NUM' OF 30 M
                gdf_buff = gpd.GeoDataFrame(fileWt, crs=crs, geometry=geometry_temp)
                # gdf_buff = makeGPD(datFram,'LON','LAT')
                gdf_buff = gdf_buff.to_crs(epsg=32610)
                gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(30)
                gdf_buff.to_file(jsonOut, driver="GeoJSON")
        elif openFile.shape[0] == 0:
            print("No Observed Peaks Found in the file: " + str(xFilename))

    except ValueError:
        print("Error in Identify Peaks")
        return False
def filter_peaks(xCar, xDate, xDir, xFilename, outFolder, buffer='30', whichpass=0):
    """ goes through a given peak file to see if there are any verifications within that file
    input:
        xCar: name of car
        xDate: date of the reading
        xDir: directory
        xFilename: name of file
        outFolder: where to save it to
        buffer: size (m) of the buffer to use to find overlapping observed peaks to combine/verify
        whichpass: doesn't really matter, but could be used to identify stuff
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    ## NECESSARY MODULES
    import pandas as pd  #
    import geopandas as gpd
    import shutil
    from datetime import datetime
    from shapely.geometry import Point
    buffer = float(buffer)

    # MOVING THE FILES NECESSARY & CREATING NEW FILES
    file_loc = xDir + xFilename
    new_loc = outFolder + "Filtered" + xFilename
    new_loc_json = new_loc[:-3] + 'geojson'

    oldInfo = xDir + 'Peaks_' + xCar + "_" + xDate.replace("-", "") + "_info.csv"
    newInfo = outFolder + 'FilteredPeaks_' + xCar + "_" + xDate.replace("-", "") + "_info.csv"

    shutil.copy(oldInfo, newInfo)
    datFram = pd.read_csv(file_loc)  # READING IN THE FILE
    #datFram = datFram_original.drop_duplicates('OP_NUM')

    if datFram.shape[0] == 0:  # IF THE OBSERVED PEAK FILE WAS EMPTY, MOVE ON
        print("Not filtering this file, no peak in it!")
        return True
    elif datFram.shape[0] == 1:  ## IF ONLY HAD ONE OBSERVED PEAK
        datFram_cent = datFram.copy()
        #datFram_cent['OB_CH4_AB'] = datFram.loc[:, 'OB_CH4'].sub(datFram.loc[:, 'OB_CH4_BASELINE'], axis=0)
        maxch4 = datFram_cent.groupby('OP_NUM', as_index=False).OB_CH4_AB.max().rename(
            columns={'OB_CH4_AB': 'pk_maxCH4_AB'})

        maxc2h6 = datFram_cent.groupby('OP_NUM', as_index=False).OB_C2H6_AB.max().rename(
            columns={'OB_C2H6_AB': 'pk_maxC2H6_AB'})

        datFram_wtLoc = weighted_loc(datFram_cent, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB').loc[:, :].rename(
            columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'})


        datFram_wtLocMax1 = pd.merge(datFram_wtLoc, maxch4, on=['OP_NUM'])
        datFram_wtLocMax = pd.merge(datFram_wtLocMax1, maxc2h6, on=['OP_NUM'])

        pass_info = datFram.copy()
        geometry_temp = [Point(lon, lat) for lon, lat in zip(datFram_wtLocMax['pk_LON'], datFram_wtLocMax['pk_LAT'])]
        #crs = {'init': 'epsg:4326'}
        crs = 'EPSG:4326'

        gdf_buff = gpd.GeoDataFrame(datFram_wtLocMax, crs=crs, geometry=geometry_temp)
        gdf_buff = gdf_buff.to_crs(epsg=32610)
        # gdf_buff['geometry'] = gdf_buff.loc[:,'geometry'].buffer(30)
        gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(buffer)
        gdf_tog = pd.merge(gdf_buff, datFram, on=['OP_NUM'])
        gdf_bind_pks = gdf_buff.copy()
        gdf_pass_pks = gdf_bind_pks.copy()
        gdf_pass_pks['min_read'] = gdf_pass_pks.loc[:, 'OP_NUM']
        gdf_pass_pks['numtimes'] = 1
        gdf_pass_pks['numdays'] = 1
        gdf_pass_pks['newgeo'] = gdf_pass_pks.loc[:, 'geometry']
        gdf_pass_pks['recombine'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['OP_NUM']].to_numpy())].copy()
        gdf_pass_pks['dates'] = gdf_pass_pks.apply(
            lambda x: datetime.fromtimestamp(int(x.OP_NUM[len(xCar) + 1:x.OP_NUM.find('.')])).strftime('%Y-%m-%d'),
            axis=1)
        gdf_pass_pks['pk_Dates'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['dates']].to_numpy())]
        gdf_pass_pks['min_Date'] = gdf_pass_pks.loc[:, 'dates']
        gdf_pass_pks = gdf_pass_pks.drop(columns=['dates'])

        gdf_pass_pks['verified'] = False
        gdf_pass_pks['geometry'] = gdf_pass_pks.loc[:, "newgeo"]
        together = pd.merge(gdf_pass_pks, gdf_tog, on=['OP_NUM', 'pk_LON', 'pk_LAT',
                                                       'pk_maxCH4_AB','pk_maxC2H6_AB', 'geometry'])
        together['pass'] = whichpass
        gdf_pass_pks = together.copy()

        gdf_pass_pks['pkGEO'] = gdf_pass_pks.loc[:, "geometry"]
        gdf_pass_pks['geometry'] = gdf_pass_pks.loc[:, "newgeo"]
        del (gdf_pass_pks['newgeo'])
        gdf_pass_pks['pass'] = whichpass

        gdf_op_unique = gdf_pass_pks.loc[:,
                        ['numtimes', 'min_read', 'numdays', 'min_Date', 'verified', 'pass', 'OB_LON',
                         'OB_LAT']].drop_duplicates()
        gdfcop = gdf_pass_pks.loc[:,
                 ['OP_NUM', 'min_read', 'min_Date', 'numtimes', 'verified', 'pass', 'pk_LAT', 'pk_LON',
                  'pk_maxCH4_AB','pk_maxC2H6_AB']].drop_duplicates()
        combinedOP = weighted_loc(gdfcop, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB').loc[:, :].rename(
            columns={'pk_LAT': 'Overall_LAT', 'pk_LON': 'Overall_LON'}).reset_index(drop=True)
        combinedOP1 = pd.merge(combinedOP, gdfcop, on=['min_read'])
        geometry_temp = [Point(lon, lat) for lon, lat in zip(combinedOP1['Overall_LON'], combinedOP1['Overall_LAT'])]
        crs = 'EPSG:4326'
        gdf_OP = gpd.GeoDataFrame(combinedOP1, crs=crs, geometry=geometry_temp)
        gdf_OP = gdf_OP.to_crs(epsg=32610).copy()
        gdf_OP_reduced = gdf_OP.loc[:, ['min_read', 'geometry',
                                        'numtimes', 'Overall_LON',
                                        'Overall_LAT', 'min_Date',
                                        'pk_maxCH4_AB','pk_maxC2H6_AB',
                                        'verified']].drop_duplicates().reset_index(drop=True)


        gdf_OP_reduced.to_file(new_loc_json, driver="GeoJSON")
        #gdf_OP_reduced.to_file('op.geojson', driver="GeoJSON")

        gdf_OP_wrecombine = pd.merge(gdf_OP.drop(columns=['geometry']),
                                     gdf_pass_pks.drop(columns=['geometry']),
                                     on=['min_read', 'min_Date', 'numtimes',
                                         'pass', 'verified', 'pk_LAT',
                                         'pk_LON','OP_NUM', 'pk_maxCH4_AB',
                                         'pk_maxC2H6_AB'])
        gdf_OP_wrecombine.to_csv(new_loc, index=False)

        gdf_buff = gpd.GeoDataFrame(datFram_wtLocMax, crs=crs, geometry=geometry_temp)
        unique_peaks = gdf_pass_pks.loc[:, ['OP_NUM', 'pk_LAT',
                                            'pk_LON', 'min_read', 'min_Date']].drop_duplicates()
        unique_peaks['save'] = True
        good_pks = list(unique_peaks.index)

        def get_thing(index):
            if index in good_pks:
                return True
            else:
                return False

        gdf_pass_pks['wooind'] = gdf_pass_pks.index
        gdf_pass_pks['save'] = gdf_pass_pks.apply(lambda x: get_thing(x.wooind), axis=1)
        unique_pks_tog = gdf_pass_pks.loc[gdf_pass_pks.save == True, :].reset_index(drop=True)
        unique_pks_tog['Latitude'] = unique_pks_tog.loc[:, 'pk_LAT']
        unique_pks_tog['Longitude'] = unique_pks_tog.loc[:, 'pk_LON']

        unique_pks_tog = gdf_pass_pks.loc[gdf_pass_pks.save == True, :].reset_index(drop=True)
        unique_pks_tog['Latitude'] = unique_pks_tog.loc[:, 'pk_LAT']
        unique_pks_tog['Longitude'] = unique_pks_tog.loc[:, 'pk_LON']

        ## adding option to add in the overall lat and lon if there is shape 1

        unique_pks_tog['Overall_LON'] = unique_pks_tog.loc[:,'pk_LON']
        unique_pks_tog['Overall_LAT'] = unique_pks_tog.loc[:,'pk_LAT']

        unique_pks_tog_stripped = unique_pks_tog.loc[:,
                                  ['OP_NUM', 'pk_LAT','pk_LON', 'pkGEO','pk_maxCH4_AB', 'pk_maxC2H6_AB', 'geometry',
                                   'min_read', 'numtimes', 'numdays', 'recombine', 'pk_Dates', 'min_Date',
                                   'verified','Latitude','Longitude','Overall_LON','Overall_LAT','wooind','save','pass',
                                   ]]
        unique_pk_names = unique_pks_tog.OP_NUM.drop_duplicates().tolist()
        unique_all = datFram.loc[datFram['OP_NUM'].isin(unique_pk_names), :]
        finaldf = pd.merge(unique_pks_tog_stripped, unique_all, on='OP_NUM')
        #unique_pks_tog.to_csv(new_loc, index=False)
        finaldf.to_csv(new_loc, index=False)

        #unique_pks_tog.to_csv(new_loc, index=False)

        # return(gdf_OP_wrecombine)

    elif datFram.shape[0] != 1:
        datFram_cent = datFram.copy()
        ### MAXCH4 is a df with the max methane (above baseline) in the given observed peak
        maxch4 = datFram_cent.groupby('OP_NUM', as_index=False).OB_CH4_AB.max().rename(
            columns={'OB_CH4_AB': 'pk_maxCH4_AB'})
        maxc2h6 = datFram_cent.groupby('OP_NUM', as_index=False).OB_C2H6_AB.max().rename(
            columns={'OB_C2H6_AB': 'pk_maxC2H6_AB'})
        ### FINDING WEIGHTED LOCATION OF THE OP, BY THE ABOVE BASELINE CH4 LEVEL
        datFram_wtLoc = weighted_loc(datFram_cent, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB').loc[:, :].rename(
            columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'})
        # datFram_wtLoc = weighted_loc(datFram_cent,'LAT','LON','PEAK_NUM','CH4_AB').rename(columns = {'LAT':'pk_LAT','LON':'pk_LON'}).copy()
        datFram_wtLocMax1 = pd.merge(datFram_wtLoc, maxch4, on=['OP_NUM'])
        datFram_wtLocMax = pd.merge(datFram_wtLocMax1, maxc2h6, on=['OP_NUM'])
        pass_info = datFram.copy()
        geometry_temp = [Point(lon, lat) for lon, lat in zip(datFram_wtLocMax['pk_LON'], datFram_wtLocMax['pk_LAT'])]
        crs = 'EPSG:4326'

        # geometry is the point of the lat/lon
        # gdf_buff = gpd.GeoDataFrame(datFram, crs=crs, geometry=geometry_temp)

        ## BUFFER AROUND EACH 'OP_NUM' OF 30 M
        gdf_buff = gpd.GeoDataFrame(datFram_wtLocMax, crs=crs, geometry=geometry_temp)
        # gdf_buff = makeGPD(datFram,'LON','LAT')
        gdf_buff = gdf_buff.to_crs(epsg=32610)
        # gdf_buff['geometry'] = gdf_buff.loc[:,'geometry'].buffer(30)
        gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(buffer)
        gdf_tog = pd.merge(gdf_buff, datFram, on=['OP_NUM'])
        gdf_bind_pks = gdf_buff.copy()

        if gdf_bind_pks.shape[0] > 1:
            data_overlap = gpd.GeoDataFrame(crs=gdf_bind_pks.crs)
            data_temp = gdf_bind_pks.copy()
            data_temp = data_temp.to_crs(epsg=32610)

            for index, row in data_temp.iterrows():
                data_temp1 = data_temp.loc[data_temp.OP_NUM != row.OP_NUM, :]
                data_temp1 = data_temp1.to_crs(epsg=32610)

                # check if intersection occured
                overlaps = data_temp1[data_temp1.geometry.overlaps(row.geometry)]['OP_NUM'].tolist()
                if len(overlaps) > 0:

                    # compare the area with threshold
                    for y in overlaps:
                        temp_area = gpd.overlay(data_temp.loc[data_temp.OP_NUM == y,],
                                                data_temp.loc[data_temp.OP_NUM == row.OP_NUM,], how='intersection')
                        temp_area = temp_area.loc[temp_area.geometry.area >= 0.001]
                        if temp_area.shape[0] > 0:
                            temp_union = gpd.overlay(data_temp.loc[data_temp.OP_NUM == y,],
                                                     data_temp.loc[data_temp.OP_NUM == row.OP_NUM,], how='union')
                            data_overlap = gpd.GeoDataFrame(pd.concat([temp_union, data_overlap], ignore_index=True),
                                                            crs=data_temp.crs)
            if data_overlap.size > 0:
                firstnull2 = data_overlap.loc[data_overlap.OP_NUM_1.isnull(), :]
                firstnull = firstnull2.copy()
                firstnull.loc[:, 'OP_NUM_1'] = firstnull2.loc[:, 'OP_NUM_2']

                secnull2 = data_overlap.loc[data_overlap.OP_NUM_2.isnull(), :]

                secnull = secnull2.copy()
                secnull.loc[:, 'OP_NUM_2'] = secnull2.loc[:, 'OP_NUM_1']

                withoutNA = data_overlap.copy().dropna()
                allTog2 = pd.concat([firstnull, secnull, withoutNA]).reset_index().copy()

                allTog2['notsame'] = allTog2.apply(lambda x: x.OP_NUM_1 == x.OP_NUM_2, axis=1)
                allTog = allTog2.loc[allTog2.notsame == False, :].drop(columns=['notsame'])

                over = allTog.copy()
                over['sorted'] = over.apply(lambda y: sorted([y['OP_NUM_1'], y['OP_NUM_2']]), axis=1)
                over['sorted'] = over.sorted.apply(lambda y: ''.join(y))
                over = over.drop_duplicates('sorted')
                over['combined'] = [list(x) for x in list(over.loc[:, ['OP_NUM_1', 'OP_NUM_2']].to_numpy())]
                # over['date1'] = over.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM_1'][6:-2])).strftime('%Y-%m-%d'),axis=1)
                # over['date2'] = over.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM_2'][6:-2])).strftime('%Y-%m-%d'),axis=1)
                over['date1'] = over.apply(
                    lambda x: datetime.fromtimestamp(int(x.OP_NUM_1[len(xCar) + 1:x.OP_NUM_1.find('.')])).strftime(
                        '%Y-%m-%d'),
                    axis=1)
                over['date2'] = over.apply(
                    lambda x: datetime.fromtimestamp(int(x.OP_NUM_2[len(xCar) + 1:x.OP_NUM_2.find('.')])).strftime(
                        '%Y-%m-%d'),
                    axis=1)

                def unique(list1):
                    # intilize a null list
                    unique_list = []
                    # traverse for all elements
                    for x in list1:
                        # check if exists in unique_list or not
                        if x not in unique_list:
                            unique_list.append(x)
                    return (unique_list)

                over['dates'] = [list(x) for x in list(over.loc[:, ['date1', 'date2']].to_numpy())]
                over['pk_Dates'] = over.apply(lambda x: unique(x.dates), axis=1)
                over = over.drop(columns=['dates'])

                over['VER_NUM'] = over.apply(lambda y: y.combined, axis=1)
                over['min_val'] = over.apply(lambda y: min(y.combined), axis=1)
                over2 = over.reset_index().loc[:,
                        ['OP_NUM_1', 'OP_NUM_2', 'geometry', 'combined', 'min_val', 'pk_Dates']]

                overcop = over2.copy().rename(columns={'combined': 'recombine'})
                # overcop.loc[:,'recombine'] = overcop.loc[:,'combined']

                for index, row in overcop.iterrows():
                    united = row.recombine
                    undate = row.pk_Dates
                    for index2, row2 in overcop.iterrows():
                        united_temp = unIfInt(united, row2.recombine)
                        undate_temp = unIfInt(undate, row2.pk_Dates)
                        if united_temp != None:
                            united = united_temp
                        if undate_temp != None:
                            undate = undate_temp
                    overcop.at[index, 'recombine'] = united.copy()
                    overcop.at[index, 'pk_Dates'] = undate.copy()

                    del (united)
                    del (undate)

                overcop['recombine'] = overcop.apply(lambda y: sorted(y.recombine), axis=1).copy()
                overcop['pk_Dates'] = overcop.apply(lambda y: sorted(y.pk_Dates), axis=1).copy()
                overcop['min_read'] = overcop.apply(lambda y: min(y.recombine), axis=1).copy()
                overcop['min_Date'] = overcop.apply(lambda y: min(y.pk_Dates), axis=1).copy()

                newOverlap = overcop.dissolve(by='min_read', as_index=False).loc[:,
                             ['min_read', 'geometry', 'recombine', 'min_Date', 'pk_Dates']].copy()

                combined = gdf_bind_pks.copy()
                combined['recombine'] = [list(x) for x in list(combined.loc[:, ['OP_NUM']].to_numpy())]
                # combined['dates'] = combined.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM'][6:-2])).strftime('%Y-%m-%d'),axis=1)
                combined['dates'] = combined.apply(
                    lambda x: datetime.fromtimestamp(int(x.OP_NUM[len(xCar) + 1:x.OP_NUM.find('.')])).strftime(
                        '%Y-%m-%d'), axis=1)

                combined['pk_Dates'] = [list(x) for x in list(combined.loc[:, ['dates']].to_numpy())]
                combined['min_Date'] = combined.loc[:, 'dates']
                combined['numtimes'] = 1
                combined['newgeo'] = combined.loc[:, 'geometry']
                combined['min_read'] = combined.loc[:, "OP_NUM"]

                for index, row in combined.iterrows():
                    for index2, row2 in newOverlap.iterrows():
                        if row.OP_NUM in row2.recombine:
                            combined.at[index, 'recombine'] = row2.recombine.copy()
                            # combined.at[index, 'newgeo']  = row2.copy().geometry
                            combined.at[index, 'min_read'] = row2.copy().min_read
                            combined.at[index, 'pk_Dates'] = row2.pk_Dates
                            combined.at[index, 'min_Date'] = row2.min_Date

                # combined['numtimes'] = combined.apply(lambda y: len(y.recombine),axis = 1).copy()
                combined['numtimes'] = combined.apply(lambda x: count_times(x.recombine, xCar), axis=1)

                combined['numdays'] = combined.apply(lambda y: len(y.pk_Dates), axis=1).copy()
                combined_reduced = combined.loc[:,
                                   ['OP_NUM', 'newgeo', 'recombine', 'numtimes', 'min_read', 'numdays', 'pk_Dates',
                                    'min_Date']]
                gdf_pass_pks = pd.merge(gdf_tog, combined_reduced, on=['OP_NUM']).copy()
                gdf_pass_pks['verified'] = gdf_pass_pks.apply(lambda y: (True if y.numtimes > 1 else False),
                                                              axis=1).copy()
            if data_overlap.size == 0:
                gdf_pass_pks = gdf_bind_pks.copy()
                gdf_pass_pks['min_read'] = gdf_pass_pks.loc[:, 'OP_NUM']
                gdf_pass_pks['numtimes'] = 1
                gdf_pass_pks['numdays'] = 1

                gdf_pass_pks['newgeo'] = gdf_pass_pks.loc[:, 'geometry']
                gdf_pass_pks['recombine'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['OP_NUM']].to_numpy())].copy()
                # gdf_pass_pks['dates'] = gdf_pass_pks.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM'][6:-2])).strftime('%Y-%m-%d'),axis=1)
                gdf_pass_pks['dates'] = gdf_pass_pks.apply(
                    lambda x: datetime.fromtimestamp(int(x.OP_NUM[len(xCar) + 1:x.OP_NUM.find('.')])).strftime(
                        '%Y-%m-%d'), axis=1)

                gdf_pass_pks['pk_Dates'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['dates']].to_numpy())]
                gdf_pass_pks['min_Date'] = gdf_pass_pks.loc[:, 'dates']
                gdf_pass_pks = gdf_pass_pks.drop(columns=['dates'])

                gdf_pass_pks['verified'] = False
                #           gdf_pass_pks['oldgeo'] = gdf_pass_pks.loc[:,'geometry']
                gdf_pass_pks['geometry'] = gdf_pass_pks.loc[:, "newgeo"]
                together = pd.merge(gdf_pass_pks, gdf_tog,
                                    on=['OP_NUM', 'pk_LON', 'pk_LAT', 'pk_maxCH4_AB','pk_maxC2H6_AB', 'geometry'])
                together['pass'] = whichpass
                gdf_pass_pks = together.copy()

        if gdf_bind_pks.shape[0] == 1:
            gdf_pass_pks = gdf_bind_pks.copy()
            gdf_pass_pks['min_read'] = gdf_pass_pks.loc[:, 'OP_NUM']
            gdf_pass_pks['numtimes'] = 1
            gdf_pass_pks['newgeo'] = gdf_pass_pks.loc[:, 'geometry']

            gdf_pass_pks['recombine'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['OP_NUM']].to_numpy())].copy()
            # gdf_pass_pks['dates'] = gdf_pass_pks.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM'][6:-2])).strftime('%Y-%m-%d'),axis=1)
            gdf_pass_pks['dates'] = gdf_pass_pks.apply(
                lambda x: datetime.fromtimestamp(int(x.OP_NUM[len(xCar) + 1:x.OP_NUM.find('.')])).strftime('%Y-%m-%d'),
                axis=1)

            gdf_pass_pks['pk_Dates'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['dates']].to_numpy())]
            gdf_pass_pks['min_Date'] = gdf_pass_pks.loc[:, 'dates']
            gdf_pass_pks['numdays'] = 1
            gdf_pass_pks = gdf_pass_pks.drop(columns=['dates'])

            gdf_pass_pks['verified'] = False
            epdat = pass_info.loc[:, ['OP_NUM', 'OP_EPOCHSTART']]
            gdf_pass_pks = pd.merge(gdf_pass_pks, epdat, on=['OP_NUM']).copy()
            data_overlap = pd.DataFrame(columns=['what', 'oh'])

#####sot
        gdf_pass_pks['pkGEO'] = gdf_pass_pks.loc[:, "geometry"]
        gdf_pass_pks['geometry'] = gdf_pass_pks.loc[:, "newgeo"]
        del (gdf_pass_pks['newgeo'])
        gdf_pass_pks['pass'] = whichpass
        gdf_pass_pks['Overall_LON'] = gdf_pass_pks['pk_LON']
        gdf_pass_pks['Overall_LAT'] = gdf_pass_pks['pk_LAT']
        combinedOP1 = gdf_pass_pks.drop(columns=['recombine', 'pk_Dates']).drop_duplicates()

        if data_overlap.size != 0:
            gdf_op_unique = gdf_pass_pks.loc[:,
                            ['numtimes', 'min_read', 'numdays', 'min_Date', 'verified', 'pass', 'OB_LON',
                             'OB_LAT']].drop_duplicates()
            gdfcop = gdf_pass_pks.loc[:,
                     ['OP_NUM', 'min_read', 'min_Date', 'numtimes', 'verified', 'pass', 'pk_LAT', 'pk_LON',
                      'pk_maxCH4_AB']].drop_duplicates()
            combinedOP = weighted_loc(gdfcop, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB').loc[:, :].rename(
                columns={'pk_LAT': 'Overall_LAT', 'pk_LON': 'Overall_LON'}).reset_index(drop=True)
            combinedOP1 = pd.merge(combinedOP, gdfcop, on=['min_read'])

        if data_overlap.size == 0 and gdf_bind_pks.shape[0] != 1:
            gdf_op_unique = gdf_pass_pks.loc[:,
                            ['numtimes', 'min_read', 'numdays', 'min_Date', 'verified', 'pass', 'OB_LON',
                             'OB_LAT']].drop_duplicates()
            gdfcop = gdf_pass_pks.loc[:,
                     ['OP_NUM', 'min_read', 'min_Date', 'numtimes', 'verified', 'pass', 'pk_LAT', 'pk_LON',
                      'pk_maxCH4_AB','pk_maxC2H6_AB']].drop_duplicates()
            combinedOP = weighted_loc(gdfcop, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB').loc[:, :].rename(
                columns={'pk_LAT': 'Overall_LAT', 'pk_LON': 'Overall_LON'}).reset_index(drop=True)
            combinedOP1 = pd.merge(combinedOP, gdfcop, on=['min_read'])

        geometry_temp = [Point(lon, lat) for lon, lat in zip(combinedOP1['Overall_LON'], combinedOP1['Overall_LAT'])]

        crs = 'EPSG:4326'
        gdf_OP = gpd.GeoDataFrame(combinedOP1, crs=crs, geometry=geometry_temp)
        #gdf_OP = gdf_OP.to_crs(epsg=32610).copy()

        gdf_OP_reduced = gdf_OP.loc[:, ['min_read', 'geometry', 'numtimes', 'Overall_LON', 'Overall_LAT', 'min_Date',
                                        'verified']].drop_duplicates().reset_index(drop=True)
        gdf_OP_reduced.to_file(new_loc_json, driver="GeoJSON")
        gdf_OP_wrecombine = pd.merge(gdf_OP.drop(columns=['geometry']), gdf_pass_pks.drop(columns=['geometry']),
                                     on=['min_read', 'min_Date', 'numtimes', 'pass', 'verified', 'pk_LAT', 'pk_LON',
                                         'OP_NUM', 'pk_maxCH4_AB'])

        gdf_OP_wrecombine.to_csv(new_loc, index=False)
        gdf_buff = gpd.GeoDataFrame(datFram_wtLocMax, crs=crs, geometry=geometry_temp)
        unique_peaks = gdf_pass_pks.loc[:, ['OP_NUM', 'pk_LAT', 'pk_LON', 'min_read', 'min_Date']].drop_duplicates()
        unique_peaks['save'] = True
        good_pks = list(unique_peaks.index)

### back2 here

        def get_thing(index):
            if index in good_pks:
                return True
            else:
                return False

        gdf_pass_pks['wooind'] = gdf_pass_pks.index
        gdf_pass_pks['save'] = gdf_pass_pks.apply(lambda x: get_thing(x.wooind), axis=1)

        unique_pks_tog = gdf_pass_pks.loc[gdf_pass_pks.save == True, :].reset_index(drop=True)
        unique_pks_tog['Latitude'] = unique_pks_tog.loc[:, 'pk_LAT']
        unique_pks_tog['minElevated'] = datFram.minElevated[0]
        unique_pks_tog['Longitude'] = unique_pks_tog.loc[:, 'pk_LON']
        unique_pks_tog_stripped = unique_pks_tog.loc[:,
                                  ['OP_NUM', 'pk_LAT','pk_LON', 'pkGEO','pk_maxCH4_AB', 'pk_maxC2H6_AB', 'geometry',
                                   'min_read', 'numtimes', 'numdays', 'recombine', 'pk_Dates', 'min_Date',
                                   'verified','Latitude','Longitude','Overall_LON','Overall_LAT','wooind','save','pass'
                                   ]]
        unique_pk_names = unique_pks_tog.OP_NUM.drop_duplicates().tolist()
        unique_all = datFram.loc[datFram['OP_NUM'].isin(unique_pk_names), :]

        finaldf = pd.merge(unique_pks_tog_stripped, unique_all, on='OP_NUM')
        #unique_pks_tog.to_csv(new_loc, index=False)
        finaldf.to_csv(new_loc, index=False)
        return

## without adding the return TRUE option
def filter_peaks_old(xCar, xDate, xDir, xFilename, outFolder, buffer='30', whichpass=0):
    """ goes through a given peak file to see if there are any verifications within that file
    input:
        xCar: name of car
        xDate: date of the reading
        xDir: directory
        xFilename: name of file
        outFolder: where to save it to
        buffer: size (m) of the buffer to use to find overlapping observed peaks to combine/verify
        whichpass: doesn't really matter, but could be used to identify stuff
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    ## NECESSARY MODULES
    import pandas as pd  #
    import geopandas as gpd
    import shutil
    from datetime import datetime
    from shapely.geometry import Point
    buffer = float(buffer)

    # MOVING THE FILES NECESSARY & CREATING NEW FILES
    file_loc = xDir + xFilename
    new_loc = outFolder + "Filtered" + xFilename
    new_loc_json = new_loc[:-3] + 'geojson'

    oldInfo = xDir + 'Peaks_' + xCar + "_" + xDate.replace("-", "") + "_info.csv"
    newInfo = outFolder + 'FilteredPeaks_' + xCar + "_" + xDate.replace("-", "") + "_info.csv"

    shutil.copy(oldInfo, newInfo)
    datFram = pd.read_csv(file_loc)  # READING IN THE FILE
    #datFram = datFram_original.drop_duplicates('OP_NUM')

    if datFram.shape[0] == 0:  # IF THE OBSERVED PEAK FILE WAS EMPTY, MOVE ON
        print("Not filtering this file, no peak in it!")
    elif datFram.shape[0] == 1:  ## IF ONLY HAD ONE OBSERVED PEAK
        datFram_cent = datFram.copy()
        #datFram_cent['OB_CH4_AB'] = datFram.loc[:, 'OB_CH4'].sub(datFram.loc[:, 'OB_CH4_BASELINE'], axis=0)
        maxch4 = datFram_cent.groupby('OP_NUM', as_index=False).OB_CH4_AB.max().rename(
            columns={'OB_CH4_AB': 'pk_maxCH4_AB'})

        maxc2h6 = datFram_cent.groupby('OP_NUM', as_index=False).OB_C2H6_AB.max().rename(
            columns={'OB_C2H6_AB': 'pk_maxC2H6_AB'})

        datFram_wtLoc = weighted_loc(datFram_cent, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB').loc[:, :].rename(
            columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'})


        datFram_wtLocMax1 = pd.merge(datFram_wtLoc, maxch4, on=['OP_NUM'])
        datFram_wtLocMax = pd.merge(datFram_wtLocMax1, maxc2h6, on=['OP_NUM'])

        pass_info = datFram.copy()
        geometry_temp = [Point(lon, lat) for lon, lat in zip(datFram_wtLocMax['pk_LON'], datFram_wtLocMax['pk_LAT'])]
        #crs = {'init': 'epsg:4326'}
        crs = 'EPSG:4326'

        gdf_buff = gpd.GeoDataFrame(datFram_wtLocMax, crs=crs, geometry=geometry_temp)
        gdf_buff = gdf_buff.to_crs(epsg=32610)
        # gdf_buff['geometry'] = gdf_buff.loc[:,'geometry'].buffer(30)
        gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(buffer)
        gdf_tog = pd.merge(gdf_buff, datFram, on=['OP_NUM'])
        gdf_bind_pks = gdf_buff.copy()
        gdf_pass_pks = gdf_bind_pks.copy()
        gdf_pass_pks['min_read'] = gdf_pass_pks.loc[:, 'OP_NUM']
        gdf_pass_pks['numtimes'] = 1
        gdf_pass_pks['numdays'] = 1
        gdf_pass_pks['newgeo'] = gdf_pass_pks.loc[:, 'geometry']
        gdf_pass_pks['recombine'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['OP_NUM']].to_numpy())].copy()
        gdf_pass_pks['dates'] = gdf_pass_pks.apply(
            lambda x: datetime.fromtimestamp(int(x.OP_NUM[len(xCar) + 1:x.OP_NUM.find('.')])).strftime('%Y-%m-%d'),
            axis=1)
        gdf_pass_pks['pk_Dates'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['dates']].to_numpy())]
        gdf_pass_pks['min_Date'] = gdf_pass_pks.loc[:, 'dates']
        gdf_pass_pks = gdf_pass_pks.drop(columns=['dates'])

        gdf_pass_pks['verified'] = False
        gdf_pass_pks['geometry'] = gdf_pass_pks.loc[:, "newgeo"]
        together = pd.merge(gdf_pass_pks, gdf_tog, on=['OP_NUM', 'pk_LON', 'pk_LAT',
                                                       'pk_maxCH4_AB','pk_maxC2H6_AB', 'geometry'])
        together['pass'] = whichpass
        gdf_pass_pks = together.copy()

        gdf_pass_pks['pkGEO'] = gdf_pass_pks.loc[:, "geometry"]
        gdf_pass_pks['geometry'] = gdf_pass_pks.loc[:, "newgeo"]
        del (gdf_pass_pks['newgeo'])
        gdf_pass_pks['pass'] = whichpass

        gdf_op_unique = gdf_pass_pks.loc[:,
                        ['numtimes', 'min_read', 'numdays', 'min_Date', 'verified', 'pass', 'OB_LON',
                         'OB_LAT']].drop_duplicates()
        gdfcop = gdf_pass_pks.loc[:,
                 ['OP_NUM', 'min_read', 'min_Date', 'numtimes', 'verified', 'pass', 'pk_LAT', 'pk_LON',
                  'pk_maxCH4_AB','pk_maxC2H6_AB']].drop_duplicates()
        combinedOP = weighted_loc(gdfcop, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB').loc[:, :].rename(
            columns={'pk_LAT': 'Overall_LAT', 'pk_LON': 'Overall_LON'}).reset_index(drop=True)
        combinedOP1 = pd.merge(combinedOP, gdfcop, on=['min_read'])
        geometry_temp = [Point(lon, lat) for lon, lat in zip(combinedOP1['Overall_LON'], combinedOP1['Overall_LAT'])]
        crs = 'EPSG:4326'
        gdf_OP = gpd.GeoDataFrame(combinedOP1, crs=crs, geometry=geometry_temp)
        gdf_OP = gdf_OP.to_crs(epsg=32610).copy()
        gdf_OP_reduced = gdf_OP.loc[:, ['min_read', 'geometry',
                                        'numtimes', 'Overall_LON',
                                        'Overall_LAT', 'min_Date',
                                        'pk_maxCH4_AB','pk_maxC2H6_AB',
                                        'verified']].drop_duplicates().reset_index(drop=True)


        gdf_OP_reduced.to_file(new_loc_json, driver="GeoJSON")
        #gdf_OP_reduced.to_file('op.geojson', driver="GeoJSON")

        gdf_OP_wrecombine = pd.merge(gdf_OP.drop(columns=['geometry']),
                                     gdf_pass_pks.drop(columns=['geometry']),
                                     on=['min_read', 'min_Date', 'numtimes',
                                         'pass', 'verified', 'pk_LAT',
                                         'pk_LON','OP_NUM', 'pk_maxCH4_AB',
                                         'pk_maxC2H6_AB'])
        gdf_OP_wrecombine.to_csv(new_loc, index=False)

        gdf_buff = gpd.GeoDataFrame(datFram_wtLocMax, crs=crs, geometry=geometry_temp)
        unique_peaks = gdf_pass_pks.loc[:, ['OP_NUM', 'pk_LAT',
                                            'pk_LON', 'min_read', 'min_Date']].drop_duplicates()
        unique_peaks['save'] = True
        good_pks = list(unique_peaks.index)

        def get_thing(index):
            if index in good_pks:
                return True
            else:
                return False

        gdf_pass_pks['wooind'] = gdf_pass_pks.index
        gdf_pass_pks['save'] = gdf_pass_pks.apply(lambda x: get_thing(x.wooind), axis=1)
        unique_pks_tog = gdf_pass_pks.loc[gdf_pass_pks.save == True, :].reset_index(drop=True)
        unique_pks_tog['Latitude'] = unique_pks_tog.loc[:, 'pk_LAT']
        unique_pks_tog['Longitude'] = unique_pks_tog.loc[:, 'pk_LON']

        unique_pks_tog = gdf_pass_pks.loc[gdf_pass_pks.save == True, :].reset_index(drop=True)
        unique_pks_tog['Latitude'] = unique_pks_tog.loc[:, 'pk_LAT']
        unique_pks_tog['Longitude'] = unique_pks_tog.loc[:, 'pk_LON']
        unique_pks_tog_stripped = unique_pks_tog.loc[:,
                                  ['OP_NUM', 'pk_LAT','pk_LON', 'pkGEO','pk_maxCH4_AB', 'pk_maxC2H6_AB', 'geometry',
                                   'min_read', 'numtimes', 'numdays', 'recombine', 'pk_Dates', 'min_Date',
                                   'verified','Latitude','Longitude','Overall_LON','Overall_LAT','wooind','save','pass',
                                   ]]
        unique_pk_names = unique_pks_tog.OP_NUM.drop_duplicates().tolist()
        unique_all = datFram.loc[datFram['OP_NUM'].isin(unique_pk_names), :]
        finaldf = pd.merge(unique_pks_tog_stripped, unique_all, on='OP_NUM')
        #unique_pks_tog.to_csv(new_loc, index=False)
        finaldf.to_csv(new_loc, index=False)

        #unique_pks_tog.to_csv(new_loc, index=False)

        # return(gdf_OP_wrecombine)

    elif datFram.shape[0] != 1:
        datFram_cent = datFram.copy()
        ### MAXCH4 is a df with the max methane (above baseline) in the given observed peak
        maxch4 = datFram_cent.groupby('OP_NUM', as_index=False).OB_CH4_AB.max().rename(
            columns={'OB_CH4_AB': 'pk_maxCH4_AB'})
        maxc2h6 = datFram_cent.groupby('OP_NUM', as_index=False).OB_C2H6_AB.max().rename(
            columns={'OB_C2H6_AB': 'pk_maxC2H6_AB'})
        ### FINDING WEIGHTED LOCATION OF THE OP, BY THE ABOVE BASELINE CH4 LEVEL
        datFram_wtLoc = weighted_loc(datFram_cent, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB').loc[:, :].rename(
            columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'})
        # datFram_wtLoc = weighted_loc(datFram_cent,'LAT','LON','PEAK_NUM','CH4_AB').rename(columns = {'LAT':'pk_LAT','LON':'pk_LON'}).copy()
        datFram_wtLocMax1 = pd.merge(datFram_wtLoc, maxch4, on=['OP_NUM'])
        datFram_wtLocMax = pd.merge(datFram_wtLocMax1, maxc2h6, on=['OP_NUM'])
        pass_info = datFram.copy()
        geometry_temp = [Point(lon, lat) for lon, lat in zip(datFram_wtLocMax['pk_LON'], datFram_wtLocMax['pk_LAT'])]
        crs = 'EPSG:4326'

        # geometry is the point of the lat/lon
        # gdf_buff = gpd.GeoDataFrame(datFram, crs=crs, geometry=geometry_temp)

        ## BUFFER AROUND EACH 'OP_NUM' OF 30 M
        gdf_buff = gpd.GeoDataFrame(datFram_wtLocMax, crs=crs, geometry=geometry_temp)
        # gdf_buff = makeGPD(datFram,'LON','LAT')
        gdf_buff = gdf_buff.to_crs(epsg=32610)
        # gdf_buff['geometry'] = gdf_buff.loc[:,'geometry'].buffer(30)
        gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(buffer)
        gdf_tog = pd.merge(gdf_buff, datFram, on=['OP_NUM'])
        gdf_bind_pks = gdf_buff.copy()

        if gdf_bind_pks.shape[0] > 1:
            data_overlap = gpd.GeoDataFrame(crs=gdf_bind_pks.crs)
            data_temp = gdf_bind_pks.copy()
            data_temp = data_temp.to_crs(epsg=32610)

            for index, row in data_temp.iterrows():
                data_temp1 = data_temp.loc[data_temp.OP_NUM != row.OP_NUM, :]
                data_temp1 = data_temp1.to_crs(epsg=32610)

                # check if intersection occured
                overlaps = data_temp1[data_temp1.geometry.overlaps(row.geometry)]['OP_NUM'].tolist()
                if len(overlaps) > 0:

                    # compare the area with threshold
                    for y in overlaps:
                        temp_area = gpd.overlay(data_temp.loc[data_temp.OP_NUM == y,],
                                                data_temp.loc[data_temp.OP_NUM == row.OP_NUM,], how='intersection')
                        temp_area = temp_area.loc[temp_area.geometry.area >= 0.001]
                        if temp_area.shape[0] > 0:
                            temp_union = gpd.overlay(data_temp.loc[data_temp.OP_NUM == y,],
                                                     data_temp.loc[data_temp.OP_NUM == row.OP_NUM,], how='union')
                            data_overlap = gpd.GeoDataFrame(pd.concat([temp_union, data_overlap], ignore_index=True),
                                                            crs=data_temp.crs)
            if data_overlap.size > 0:
                firstnull2 = data_overlap.loc[data_overlap.OP_NUM_1.isnull(), :]
                firstnull = firstnull2.copy()
                firstnull.loc[:, 'OP_NUM_1'] = firstnull2.loc[:, 'OP_NUM_2']

                secnull2 = data_overlap.loc[data_overlap.OP_NUM_2.isnull(), :]

                secnull = secnull2.copy()
                secnull.loc[:, 'OP_NUM_2'] = secnull2.loc[:, 'OP_NUM_1']

                withoutNA = data_overlap.copy().dropna()
                allTog2 = pd.concat([firstnull, secnull, withoutNA]).reset_index().copy()

                allTog2['notsame'] = allTog2.apply(lambda x: x.OP_NUM_1 == x.OP_NUM_2, axis=1)
                allTog = allTog2.loc[allTog2.notsame == False, :].drop(columns=['notsame'])

                over = allTog.copy()
                over['sorted'] = over.apply(lambda y: sorted([y['OP_NUM_1'], y['OP_NUM_2']]), axis=1)
                over['sorted'] = over.sorted.apply(lambda y: ''.join(y))
                over = over.drop_duplicates('sorted')
                over['combined'] = [list(x) for x in list(over.loc[:, ['OP_NUM_1', 'OP_NUM_2']].to_numpy())]
                # over['date1'] = over.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM_1'][6:-2])).strftime('%Y-%m-%d'),axis=1)
                # over['date2'] = over.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM_2'][6:-2])).strftime('%Y-%m-%d'),axis=1)
                over['date1'] = over.apply(
                    lambda x: datetime.fromtimestamp(int(x.OP_NUM_1[len(xCar) + 1:x.OP_NUM_1.find('.')])).strftime(
                        '%Y-%m-%d'),
                    axis=1)
                over['date2'] = over.apply(
                    lambda x: datetime.fromtimestamp(int(x.OP_NUM_2[len(xCar) + 1:x.OP_NUM_2.find('.')])).strftime(
                        '%Y-%m-%d'),
                    axis=1)

                def unique(list1):
                    # intilize a null list
                    unique_list = []
                    # traverse for all elements
                    for x in list1:
                        # check if exists in unique_list or not
                        if x not in unique_list:
                            unique_list.append(x)
                    return (unique_list)

                over['dates'] = [list(x) for x in list(over.loc[:, ['date1', 'date2']].to_numpy())]
                over['pk_Dates'] = over.apply(lambda x: unique(x.dates), axis=1)
                over = over.drop(columns=['dates'])

                over['VER_NUM'] = over.apply(lambda y: y.combined, axis=1)
                over['min_val'] = over.apply(lambda y: min(y.combined), axis=1)
                over2 = over.reset_index().loc[:,
                        ['OP_NUM_1', 'OP_NUM_2', 'geometry', 'combined', 'min_val', 'pk_Dates']]

                overcop = over2.copy().rename(columns={'combined': 'recombine'})
                # overcop.loc[:,'recombine'] = overcop.loc[:,'combined']

                for index, row in overcop.iterrows():
                    united = row.recombine
                    undate = row.pk_Dates
                    for index2, row2 in overcop.iterrows():
                        united_temp = unIfInt(united, row2.recombine)
                        undate_temp = unIfInt(undate, row2.pk_Dates)
                        if united_temp != None:
                            united = united_temp
                        if undate_temp != None:
                            undate = undate_temp
                    overcop.at[index, 'recombine'] = united.copy()
                    overcop.at[index, 'pk_Dates'] = undate.copy()

                    del (united)
                    del (undate)

                overcop['recombine'] = overcop.apply(lambda y: sorted(y.recombine), axis=1).copy()
                overcop['pk_Dates'] = overcop.apply(lambda y: sorted(y.pk_Dates), axis=1).copy()
                overcop['min_read'] = overcop.apply(lambda y: min(y.recombine), axis=1).copy()
                overcop['min_Date'] = overcop.apply(lambda y: min(y.pk_Dates), axis=1).copy()

                newOverlap = overcop.dissolve(by='min_read', as_index=False).loc[:,
                             ['min_read', 'geometry', 'recombine', 'min_Date', 'pk_Dates']].copy()

                combined = gdf_bind_pks.copy()
                combined['recombine'] = [list(x) for x in list(combined.loc[:, ['OP_NUM']].to_numpy())]
                # combined['dates'] = combined.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM'][6:-2])).strftime('%Y-%m-%d'),axis=1)
                combined['dates'] = combined.apply(
                    lambda x: datetime.fromtimestamp(int(x.OP_NUM[len(xCar) + 1:x.OP_NUM.find('.')])).strftime(
                        '%Y-%m-%d'), axis=1)

                combined['pk_Dates'] = [list(x) for x in list(combined.loc[:, ['dates']].to_numpy())]
                combined['min_Date'] = combined.loc[:, 'dates']
                combined['numtimes'] = 1
                combined['newgeo'] = combined.loc[:, 'geometry']
                combined['min_read'] = combined.loc[:, "OP_NUM"]

                for index, row in combined.iterrows():
                    for index2, row2 in newOverlap.iterrows():
                        if row.OP_NUM in row2.recombine:
                            combined.at[index, 'recombine'] = row2.recombine.copy()
                            # combined.at[index, 'newgeo']  = row2.copy().geometry
                            combined.at[index, 'min_read'] = row2.copy().min_read
                            combined.at[index, 'pk_Dates'] = row2.pk_Dates
                            combined.at[index, 'min_Date'] = row2.min_Date

                # combined['numtimes'] = combined.apply(lambda y: len(y.recombine),axis = 1).copy()
                combined['numtimes'] = combined.apply(lambda x: count_times(x.recombine, xCar), axis=1)

                combined['numdays'] = combined.apply(lambda y: len(y.pk_Dates), axis=1).copy()
                combined_reduced = combined.loc[:,
                                   ['OP_NUM', 'newgeo', 'recombine', 'numtimes', 'min_read', 'numdays', 'pk_Dates',
                                    'min_Date']]
                gdf_pass_pks = pd.merge(gdf_tog, combined_reduced, on=['OP_NUM']).copy()
                gdf_pass_pks['verified'] = gdf_pass_pks.apply(lambda y: (True if y.numtimes > 1 else False),
                                                              axis=1).copy()
            if data_overlap.size == 0:
                gdf_pass_pks = gdf_bind_pks.copy()
                gdf_pass_pks['min_read'] = gdf_pass_pks.loc[:, 'OP_NUM']
                gdf_pass_pks['numtimes'] = 1
                gdf_pass_pks['numdays'] = 1

                gdf_pass_pks['newgeo'] = gdf_pass_pks.loc[:, 'geometry']
                gdf_pass_pks['recombine'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['OP_NUM']].to_numpy())].copy()
                # gdf_pass_pks['dates'] = gdf_pass_pks.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM'][6:-2])).strftime('%Y-%m-%d'),axis=1)
                gdf_pass_pks['dates'] = gdf_pass_pks.apply(
                    lambda x: datetime.fromtimestamp(int(x.OP_NUM[len(xCar) + 1:x.OP_NUM.find('.')])).strftime(
                        '%Y-%m-%d'), axis=1)

                gdf_pass_pks['pk_Dates'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['dates']].to_numpy())]
                gdf_pass_pks['min_Date'] = gdf_pass_pks.loc[:, 'dates']
                gdf_pass_pks = gdf_pass_pks.drop(columns=['dates'])

                gdf_pass_pks['verified'] = False
                #           gdf_pass_pks['oldgeo'] = gdf_pass_pks.loc[:,'geometry']
                gdf_pass_pks['geometry'] = gdf_pass_pks.loc[:, "newgeo"]
                together = pd.merge(gdf_pass_pks, gdf_tog,
                                    on=['OP_NUM', 'pk_LON', 'pk_LAT', 'pk_maxCH4_AB','pk_maxC2H6_AB', 'geometry'])
                together['pass'] = whichpass
                gdf_pass_pks = together.copy()

        if gdf_bind_pks.shape[0] == 1:
            gdf_pass_pks = gdf_bind_pks.copy()
            gdf_pass_pks['min_read'] = gdf_pass_pks.loc[:, 'OP_NUM']
            gdf_pass_pks['numtimes'] = 1
            gdf_pass_pks['newgeo'] = gdf_pass_pks.loc[:, 'geometry']

            gdf_pass_pks['recombine'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['OP_NUM']].to_numpy())].copy()
            # gdf_pass_pks['dates'] = gdf_pass_pks.apply(lambda x: datetime.fromtimestamp(int(x['OP_NUM'][6:-2])).strftime('%Y-%m-%d'),axis=1)
            gdf_pass_pks['dates'] = gdf_pass_pks.apply(
                lambda x: datetime.fromtimestamp(int(x.OP_NUM[len(xCar) + 1:x.OP_NUM.find('.')])).strftime('%Y-%m-%d'),
                axis=1)

            gdf_pass_pks['pk_Dates'] = [list(x) for x in list(gdf_pass_pks.loc[:, ['dates']].to_numpy())]
            gdf_pass_pks['min_Date'] = gdf_pass_pks.loc[:, 'dates']
            gdf_pass_pks['numdays'] = 1
            gdf_pass_pks = gdf_pass_pks.drop(columns=['dates'])

            gdf_pass_pks['verified'] = False
            epdat = pass_info.loc[:, ['OP_NUM', 'OP_EPOCHSTART']]
            gdf_pass_pks = pd.merge(gdf_pass_pks, epdat, on=['OP_NUM']).copy()
            data_overlap = pd.DataFrame(columns=['what', 'oh'])

#####sot
        gdf_pass_pks['pkGEO'] = gdf_pass_pks.loc[:, "geometry"]
        gdf_pass_pks['geometry'] = gdf_pass_pks.loc[:, "newgeo"]
        del (gdf_pass_pks['newgeo'])
        gdf_pass_pks['pass'] = whichpass
        gdf_pass_pks['Overall_LON'] = gdf_pass_pks['pk_LON']
        gdf_pass_pks['Overall_LAT'] = gdf_pass_pks['pk_LAT']
        combinedOP1 = gdf_pass_pks.drop(columns=['recombine', 'pk_Dates']).drop_duplicates()

        if data_overlap.size != 0:
            gdf_op_unique = gdf_pass_pks.loc[:,
                            ['numtimes', 'min_read', 'numdays', 'min_Date', 'verified', 'pass', 'OB_LON',
                             'OB_LAT']].drop_duplicates()
            gdfcop = gdf_pass_pks.loc[:,
                     ['OP_NUM', 'min_read', 'min_Date', 'numtimes', 'verified', 'pass', 'pk_LAT', 'pk_LON',
                      'pk_maxCH4_AB']].drop_duplicates()
            combinedOP = weighted_loc(gdfcop, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB').loc[:, :].rename(
                columns={'pk_LAT': 'Overall_LAT', 'pk_LON': 'Overall_LON'}).reset_index(drop=True)
            combinedOP1 = pd.merge(combinedOP, gdfcop, on=['min_read'])

        if data_overlap.size == 0 and gdf_bind_pks.shape[0] != 1:
            gdf_op_unique = gdf_pass_pks.loc[:,
                            ['numtimes', 'min_read', 'numdays', 'min_Date', 'verified', 'pass', 'OB_LON',
                             'OB_LAT']].drop_duplicates()
            gdfcop = gdf_pass_pks.loc[:,
                     ['OP_NUM', 'min_read', 'min_Date', 'numtimes', 'verified', 'pass', 'pk_LAT', 'pk_LON',
                      'pk_maxCH4_AB','pk_maxC2H6_AB']].drop_duplicates()
            combinedOP = weighted_loc(gdfcop, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB').loc[:, :].rename(
                columns={'pk_LAT': 'Overall_LAT', 'pk_LON': 'Overall_LON'}).reset_index(drop=True)
            combinedOP1 = pd.merge(combinedOP, gdfcop, on=['min_read'])

        geometry_temp = [Point(lon, lat) for lon, lat in zip(combinedOP1['Overall_LON'], combinedOP1['Overall_LAT'])]

        crs = 'EPSG:4326'
        gdf_OP = gpd.GeoDataFrame(combinedOP1, crs=crs, geometry=geometry_temp)
        #gdf_OP = gdf_OP.to_crs(epsg=32610).copy()

        gdf_OP_reduced = gdf_OP.loc[:, ['min_read', 'geometry', 'numtimes', 'Overall_LON', 'Overall_LAT', 'min_Date',
                                        'verified']].drop_duplicates().reset_index(drop=True)
        gdf_OP_reduced.to_file(new_loc_json, driver="GeoJSON")
        gdf_OP_wrecombine = pd.merge(gdf_OP.drop(columns=['geometry']), gdf_pass_pks.drop(columns=['geometry']),
                                     on=['min_read', 'min_Date', 'numtimes', 'pass', 'verified', 'pk_LAT', 'pk_LON',
                                         'OP_NUM', 'pk_maxCH4_AB'])

        gdf_OP_wrecombine.to_csv(new_loc, index=False)
        gdf_buff = gpd.GeoDataFrame(datFram_wtLocMax, crs=crs, geometry=geometry_temp)
        unique_peaks = gdf_pass_pks.loc[:, ['OP_NUM', 'pk_LAT', 'pk_LON', 'min_read', 'min_Date']].drop_duplicates()
        unique_peaks['save'] = True
        good_pks = list(unique_peaks.index)

### back2 here

        def get_thing(index):
            if index in good_pks:
                return True
            else:
                return False

        gdf_pass_pks['wooind'] = gdf_pass_pks.index
        gdf_pass_pks['save'] = gdf_pass_pks.apply(lambda x: get_thing(x.wooind), axis=1)

        unique_pks_tog = gdf_pass_pks.loc[gdf_pass_pks.save == True, :].reset_index(drop=True)
        unique_pks_tog['Latitude'] = unique_pks_tog.loc[:, 'pk_LAT']
        unique_pks_tog['minElevated'] = datFram.minElevated[0]
        unique_pks_tog['Longitude'] = unique_pks_tog.loc[:, 'pk_LON']
        unique_pks_tog_stripped = unique_pks_tog.loc[:,
                                  ['OP_NUM', 'pk_LAT','pk_LON', 'pkGEO','pk_maxCH4_AB', 'pk_maxC2H6_AB', 'geometry',
                                   'min_read', 'numtimes', 'numdays', 'recombine', 'pk_Dates', 'min_Date',
                                   'verified','Latitude','Longitude','Overall_LON','Overall_LAT','wooind','save','pass'
                                   ]]
        unique_pk_names = unique_pks_tog.OP_NUM.drop_duplicates().tolist()
        unique_all = datFram.loc[datFram['OP_NUM'].isin(unique_pk_names), :]

        finaldf = pd.merge(unique_pks_tog_stripped, unique_all, on='OP_NUM')
        #unique_pks_tog.to_csv(new_loc, index=False)
        finaldf.to_csv(new_loc, index=False)
        return

def summarize_data_2(mainDF):
    """ summarize data from after all analysis has been done
    input:
        mainDf
    output:
        finds the log ch4 mean, the minimum distance and max distance of an observed peak, etc.
    """
    from numpy import log
    import pandas as pd
    todo = mainDF.loc[:, ['OP_NUM', 'min_Date', 'pk_LON', 'pk_LAT', 'pk_maxCH4_AB','pk_maxC2H6_AB','numtimes',
                          'min_read', 'OP_DISTANCE']].drop_duplicates().reset_index(drop=True)
    todo['logCH4'] = todo.apply(lambda y: log(y.pk_maxCH4_AB), axis=1)
    mnVals = todo.groupby('min_read', as_index=False).logCH4.mean().rename(columns={'logCH4': 'mnlogCH4'}).loc[:,
             ['min_read', 'mnlogCH4']]

    mnCH4 = todo.groupby('min_read', as_index=False).pk_maxCH4_AB.mean().rename(columns={'pk_maxCH4_AB': 'mn_maxch4_ab'}).loc[:,
             ['min_read', 'mn_maxch4_ab']]
    mnC2H6 = todo.groupby('min_read', as_index=False).pk_maxC2H6_AB.mean().rename(columns={'pk_maxC2H6_AB': 'mn_maxc2h6_ab'}).loc[:,
             ['min_read', 'mn_maxc2h6_ab']]

    opMin = todo.groupby('min_read', as_index=False).OP_DISTANCE.min().rename(columns={'OP_DISTANCE': 'minDist'}).loc[:,
            ['min_read', 'minDist']]
    opMax = todo.groupby('min_read', as_index=False).OP_DISTANCE.max().rename(columns={'OP_DISTANCE': 'maxDist'}).loc[:,
            ['min_read', 'maxDist']]

    verLoc = weighted_loc(todo, 'pk_LAT', 'pk_LON', 'min_read', 'pk_maxCH4_AB').rename(
        columns={'pk_LAT': 'overallLAT', 'pk_LON': 'overallLON'}).reset_index(drop=True)
    together1 = pd.merge(verLoc, mnVals, on=['min_read'])

    together2 = pd.merge(together1, mnCH4, on=['min_read'])
    together = pd.merge(together2, mnC2H6, on=['min_read'])

    final = pd.merge(together, mainDF, on=['min_read'])
    final = pd.merge(final, opMin, on=['min_read'])
    final = pd.merge(final, opMax, on=['min_read'])
    return (final)

def pass_combine(firstgroup, secondgroup, xCar, buffer='30'):
    """ used to combine two days' filtered peak files, to find any overlaps or verified peaks
    input:
        firstgroup: filtered peak file 1
        secondgroup: filtered peak file 2
        xCar: name of car
        buffer: radius of buffer (m) to be used to find overlapping peaks
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    #import ast
    import pandas as pd  #
    import geopandas as gpd
    from shapely.geometry import Point
    buffer = float(buffer)

    if not "prev_read" in secondgroup.columns:
        sgrp = secondgroup.copy()
        secondgroup['prev_read'] = sgrp.loc[:, 'min_read']

    ### MAKE GEOPANDAS FRAME OF THE FIRST GROUP PEAKS
    first_geo = [Point(lon, lat) for lon, lat in zip(firstgroup['pk_LON'], firstgroup['pk_LAT'])]
    #   crs = {'init': 'epsg:4326'}
    #crs = {'init': 'epsg:32610'}
    crs = 'EPSG:32610'

    #if 'geometry' in firstgroup.columns:
    #    firstgrp = gpd.GeoDataFrame(firstgroup.drop(columns=['geometry', 'pkGEO']), crs=crs, geometry=first_geo)
    #else:
    #    firstgrp = gpd.GeoDataFrame(firstgroup.drop(columns=['pkGEO']), crs=crs, geometry=first_geo)
    firstgrp = gpd.GeoDataFrame(firstgroup[firstgroup.columns.difference(['geometry','pkGEO'])].reset_index(drop=True),crs = crs,geometry=first_geo)

    first_buffg = firstgrp.copy()
    first_buff = first_buffg.copy().drop(columns=['geometry'])
    first_buff['geometry'] = first_buffg.apply(lambda x: x.geometry.buffer(0.00001 * buffer), axis=1)
    firstgrp = first_buff.copy()

    sec_geo = [Point(lon, lat) for lon, lat in zip(secondgroup['pk_LON'], secondgroup['pk_LAT'])]
    secgrp = gpd.GeoDataFrame(secondgroup.drop(columns=['geometry', 'pkGEO']), crs=crs, geometry=sec_geo)

    sec_buffg = secgrp.copy()
    sec_buff = sec_buffg.copy().drop(columns=['geometry'])
    sec_buff['geometry'] = sec_buffg.apply(lambda x: x.geometry.buffer(0.00001 * buffer), axis=1)
    secgrp = sec_buff.copy()

    ## choosing the unique OPs from each group
    first_pks = firstgrp.loc[:, ['OP_NUM', 'min_read', 'pk_LAT', 'pk_LON', 'pk_maxCH4_AB','pk_maxC2H6_AB']].drop_duplicates()
    sec_pks = secgrp.loc[:, ['OP_NUM', 'min_read', 'pk_LAT', 'pk_LON', 'pk_maxCH4_AB','pk_maxC2H6_AB']].drop_duplicates()

    # combining to make one large dataframe of unique OPs
    tot_pks = pd.concat([first_pks, sec_pks])

    ### COMBINE EACH GROUP (GEOMETRICALLY) BY THEIR OVERALL NAME (COULD BE VERIFIED)
    first_dis = firstgrp.dissolve(by='min_read', as_index=False)[
        ['min_read', 'geometry', 'recombine', 'verified', 'pass']].copy()
    sec_dis = secgrp.dissolve(by='min_read', as_index=False)[
        ['min_read', 'geometry', 'recombine', 'verified', 'pass']].copy()

    gdf_bind_pks = pd.concat([first_dis, sec_dis]).loc[:, ['min_read', 'geometry', 'recombine']]
    gdf_tog = pd.concat([firstgroup.drop(['pk_LAT', 'pk_LON', 'pk_maxCH4_AB','pk_maxC2H6_AB'], axis=1),
                         secondgroup.drop(['pk_LAT', 'pk_LON', 'pk_maxCH4_AB','pk_maxC2H6_AB'], axis=1)]).copy()
    gdf_tog2 = pd.concat([firstgroup, secondgroup]).copy()

    gdf_bind_pks['prev_read'] = gdf_bind_pks.loc[:, 'min_read']
    gdf_tog['prev_read'] = gdf_tog.loc[:, 'min_read']
    if gdf_bind_pks.shape[0] > 1:
        data_overlap = gpd.GeoDataFrame(crs=gdf_bind_pks.crs).copy()
        data_temp = gdf_bind_pks.copy()
        data_temp = data_temp.to_crs(epsg=32610)

        for index, row in data_temp.iterrows():
            data_temp1 = data_temp.loc[data_temp.min_read != row.min_read,]
            data_temp1 = data_temp1.to_crs(epsg=32610)
            # check if intersection occured
            overlaps = data_temp1[data_temp1.geometry.overlaps(row.geometry)]['min_read'].tolist()
            if len(overlaps) > 0:
                # compare the area with threshold
                for y in overlaps:
                    temp_area = gpd.overlay(data_temp.loc[data_temp.min_read == y,],
                                            data_temp.loc[data_temp.min_read == row.min_read,], how='intersection')
                    temp_area = temp_area.to_crs(epsg=32610)
                    temp_area = temp_area.loc[temp_area.geometry.area >= 0]
                    # temp_union = gpd.overlay(data_temp.loc[data_temp.PEAK_NUM==y,],data_temp.loc[data_temp.PEAK_NUM==row.PEAK_NUM,],how='union')
                    if temp_area.shape[0] > 0:
                        temp_union = gpd.overlay(data_temp.loc[data_temp.min_read == y,],
                                                 data_temp.loc[data_temp.min_read == row.min_read,], how='union')
                        data_overlap = gpd.GeoDataFrame(pd.concat([temp_union, data_overlap], ignore_index=True),
                                                        crs=data_temp.crs)

        if data_overlap.size != 0:
            firstnull = data_overlap[data_overlap.min_read_1.isnull()].copy()
            firstnull.loc[:, 'min_read_1'] = firstnull.loc[:, 'min_read_2']

            secnull = data_overlap[data_overlap.min_read_2.isnull()].copy()
            secnull['min_read_2'] = secnull['min_read_1'].copy()

            withoutNA = data_overlap.dropna().copy()
            allTog = pd.concat([firstnull, secnull, withoutNA]).reset_index().copy()

            over = allTog.copy().drop(columns=['index'])
            over['sorted'] = over.apply(lambda y: sorted([y['min_read_1'], y['min_read_2']]), axis=1).copy()
            over['sorted'] = over.sorted.apply(lambda y: ','.join(y)).copy()

            over['same'] = over.apply(lambda x: x.min_read_1 != x.min_read_2, axis=1)
            over = over.loc[over.same == True, :].drop(columns=['same'])
            over = over.copy().drop_duplicates('sorted').reset_index(drop=True)



            over['bothcombine'] = over.apply(lambda x: sorted(check_lst(x.recombine_1) + check_lst(x.recombine_2)),
                                             axis=1)

            over['combined'] = [list(x) for x in list(over.loc[:, ['min_read_1', 'min_read_2']].to_numpy())].copy()
            over['VER_NUM'] = over.apply(lambda y: y.combined, axis=1).copy()
            over['min_val'] = over.apply(lambda y: min(y.combined), axis=1).copy()
            over = over.reset_index().loc[:,
                   ['min_read_1', 'min_read_2', 'geometry', 'combined', 'min_val', 'bothcombine']]

            new_1 = over.copy().drop(columns=['min_read_2']).rename(columns={'min_read_1': 'min_read'})
            new_2 = over.copy().drop(columns=['min_read_1']).rename(columns={'min_read_2': 'min_read'})
            newtog = pd.concat([new_1, new_2])
            minreads = newtog.min_read.unique().tolist()

            if 'prev_read' in gdf_tog2.columns:
                toChange = gdf_tog2[gdf_tog2['min_read'].isin(minreads)].drop(columns=['geometry', 'prev_read'])
            elif 'prev_read' not in gdf_tog2.columns:
                toChange = gdf_tog2[gdf_tog2['min_read'].isin(minreads)].drop(columns=['geometry'])

            toChangecombined = pd.merge(toChange, newtog, on=['min_read']).drop(
                columns=['geometry', 'numtimes', 'verified', 'recombine'])
            toChangecombined = toChangecombined.rename(columns={'min_read': 'prev_read', 'bothcombine': 'recombine'})
            toChangecombined['numtimes'] = toChangecombined.apply(lambda x: count_times(x.recombine, xCar), axis=1)

            toChangecombined['verified'] = toChangecombined.apply(lambda x: x.numtimes > 1, axis=1)
            toChangecombined = toChangecombined.rename(columns={'min_val': 'min_read'}).drop(columns=['combined'])

            toNotChange = gdf_tog2[~gdf_tog2['min_read'].isin(minreads)].drop(columns=['geometry'])

            if 'prev_read' in toNotChange.columns:
                toNotChange = toNotChange.drop(columns=['prev_read'])
                toNotChange['prev_read'] = toNotChange.min_read
            elif 'prev_read' not in toNotChange.columns:
                toNotChange.loc[:, 'prev_read'] = gdf_tog2[~gdf_tog2['min_read'].isin(minreads)].min_read

            newCombined = pd.concat([toChangecombined, toNotChange])
        elif data_overlap.size == 0:
            newCombined = gdf_tog2.copy()
            if 'prev_read' not in newCombined.columns:
                newCombined.loc[:, 'prev_read'] = gdf_tog2.loc[:, 'min_read']
    elif gdf_bind_pks.shape[0] == 1:
        newCombined = gdf_tog2.copy()
        if 'prev_read' not in newCombined.columns:
            newCombined.loc[:, 'prev_read'] = gdf_tog2.loc[:, 'min_read']
    opCount = newCombined.groupby('OB_EPOCH').count()
    samesies = list(opCount.loc[opCount.numtimes > 1, :].index)
    if len(samesies) == 0:
        smallCombined = newCombined.copy()
    elif len(samesies) != 0:
        smallCombined = newCombined.loc[~newCombined.OB_EPOCH.isin(samesies), :]
        for index, op in enumerate(samesies):
            smalldf = newCombined.loc[newCombined.OB_EPOCH == op, :]
            recombined2 = (list(set([a for b in smalldf.recombine.tolist() for a in b])))
            newminread = min(recombined2)
            smalldf = smalldf.drop(columns=['min_read', 'recombine'])
            smalldf['min_read'] = smalldf.apply(lambda x: newminread, axis=1)
            smalldf = smalldf.drop_duplicates().reset_index(drop=True)
            smalldf['recombine'] = smalldf.apply(lambda x: recombined2, axis=1)
            smallCombined = pd.concat([smallCombined, smalldf]).reset_index(drop=True)
    return (smallCombined)

def print_results(addingFiles,to_filter,threshold,baseline_percentile,back_obs_num,min_car_speed,max_car_speed,final_results_dir,
                  start,mainThing):
    import time
    if not addingFiles:
        print(
            f"I processed {len(to_filter)} days of driving. I analysed the data using a threshold of {100 + float(threshold) * 100}% for an elevated reading, \n \
        where the threshold was calculated using the {baseline_percentile}th percentile over {back_obs_num} observations. \n \
        I filtered the speed of the car to be between {min_car_speed}mph and {max_car_speed}mph.\n \
        I created 3 summary files located here:{final_results_dir}.\n \
        The processing took {round((time.time() - start) / 60, 3)} minutes. \n \
        I found {len(mainThing.min_read.unique())} observed peaks.")

    elif addingFiles:
        print(
            f"I processed an additional {len(to_filter)} days of driving. I analysed the data using a threshold of {100 + float(threshold) * 100}% for an elevated reading, \n \
        where the threshold was calculated using the {baseline_percentile}th percentile over {back_obs_num} observations. \n \
        I filtered the speed of the car to be between {min_car_speed}mph and {max_car_speed}mph.\n \
        I created 3 summary files located here:{final_results_dir}.\n \
        The processing took {round((time.time() - start) / 60, 3)} minutes. \n \
        I found {len(mainThing.min_read.unique()) - curOP} additional observed peaks, and {vpNew - curVP} VPs.")
    return

### new
def process_raw_data_aeris(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack, shift, maxSpeed='45',
                           minSpeed='2'):
    """ input a raw .txt file with data (from aeris data file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import os
    import sys
    if sys.platform == 'win32':
        windows = True
    elif sys.platform != 'win32':
        windows = False

    import pandas as pd
    from datetime import datetime
    import os
    import gzip
    import numpy as np
    # import csv
    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)
        xMinCarSpeed = -10
        ########################################################################
        #### WE DON'T HAVE AN RSSI INPUT
        ### (SO THIS IS A PLACEHOLDER FOR SOME SORT OF QA/QC VARIABLE)
        ##  xMinRSSI = 50  #if RSSI is below this we don't like it
        ##################################################################

        # reading in the data with specific headers
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        sHeader = 'Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude,U (m/sec),V (m/sec),W (m/sec),T (degC),Dir (deg),Speed (m/sec),Compass (deg)'
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"
        infoHeader = "FILENAME\n"
        # somehow gZIP is indicating if  it is the first file name (I think if it is 0 then it is the first file)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename,
                          'r')  # if in python 3, change this to "r" or just "b" can't remember but something about a bit not a string
        else:
            f = open(xDir + xFilename, 'r')

        infoHeader = "FILENAME\n"

        # process - if first time on this car/date, then write header out
        headerNames = sHeader.split(',')
        xdat = str('20') + xFilename[11:17]
        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        removeOut = xOut + xCar + "_" + xdat + "_removed.csv"
        fnLog = xOut + xCar + "_" + xdat + ".log"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"
        #

        dtime = open(xDir + xFilename).readlines().pop(2).split(',')[0]
        firstdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                             int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
        # firsttime = firstdate.strftime('%s.%f')
        firsttime = dt_to_epoch(firstdate)
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        if bFirst:
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")

        if not bFirst:
            fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        fLog.write("Processing file: " + str(xFilename) + "\n")

        fOut = open(fnOutTemp, 'w')
        fOut.write(sOutHeader)

        # read all lines
        xCntObs = -1
        xCntGoodValues = 0
        for row in f:
            woo = row
            bGood = True
            if xCntObs != -1:
                lstS = row.split(",")
                if float(lstS[2]) < 20:
                    bGood = False
                    xCntObs += 1
            if xCntObs < 0:
                bGood = False
                xCntObs += 1
            if bGood:
                lstS = row.split(",")
                dtime = lstS[0]
                dateob = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                  int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                fdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                 int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                # seconds = fdate.strftime('%s.%f')
                seconds = dt_to_epoch(fdate)

                def getNS(seconds):
                    ns = str(float(seconds) * 1e-3)[11:]
                    # str(pd.to_numeric(str(float(seconds) * 1e-3)[11:]) * 100000)[:9]
                    return (str(ns).ljust(15, '0'))[:9]

                if len(lstS) > 6 and float(lstS[2]) > 20:
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(
                        dateob.strftime('%H:%M:%S')) + ',' + str(str(float(seconds) * 1e-3)[:10]) + ',' + getNS(
                        seconds) + str(',')
                    csvWrite += str(lstS[20]) + ',' + str(lstS[15]) + ',' + str(lstS[16]) + ',' + str(
                        lstS[17]) + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(
                        lstS[3]) + ',' + str(lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(
                        ',') + str(lstS[14]) + '\n'
                    fOut.write(csvWrite)
                    xCntObs += 1
        fLog.write("Imported " + str(xCntObs) + " lines" + "\n")

        infOut.write(str(xFilename) + '\n')
        fOut.close()
        # fLog.close()
        infOut.close()
        # print(xCar + "\t" + xdat + "\t" + fnOut[-22:] + "\t" + str(xCntObs) + "\t" + str(xCntGoodValues) + "\t" + str(
        #    gZIP))
        print(f"{xCar} \t {xdat} \t {fnOut[-(17 + len(xCar)):]} \t  {xCntObs} \t {xCntGoodValues} \t {gZIP}")

        wind_df = pd.read_csv(fnOutTemp)
        wind_df_not_null = wind_df.loc[wind_df['LAT'].notnull(),].reset_index(drop=True)
        wind_df_null = wind_df.loc[~wind_df['LAT'].notnull(),].reset_index(drop=True)
        if wind_df_null.shape[0] > 0:
            wind_df_null=wind_df_null.assign(Reason='GPS NA')

        del (wind_df)
        wind_df = wind_df_not_null.copy()

        radians = False
        wind_df['QUADRANT'] = wind_df.apply(lambda row: get_quadrant(row['U'], row['V']), axis=1)
        wind_df['secnan'] = wind_df.apply(lambda row: row['SECONDS'] + row['NANOSECONDS'] * 1e-9,
                                          axis=1)  # + row['NANOSECONDS']*1e-9,axis=1)
        wind_df['prev_LAT'] = wind_df.LAT.shift(periods=1)
        wind_df['next_LAT'] = wind_df.LAT.shift(periods=-1)
        wind_df['prev_LONG'] = wind_df.LONG.shift(periods=1)
        wind_df['next_LONG'] = wind_df.LONG.shift(periods=-1)
        wind_df['prev_TIME'] = wind_df.secnan.shift(periods=1)
        wind_df['next_TIME'] = wind_df.secnan.shift(periods=-1)
        wind_df['distance'] = wind_df.apply(
            lambda row: haversine(row['prev_LAT'], row['prev_LONG'], row['next_LAT'], row['next_LONG']), axis=1)
        wind_df['bearing'] = wind_df.apply(
            lambda row: calc_bearing(row['prev_LAT'], row['next_LAT'], row['prev_LONG'], row['next_LONG'], radians),
            axis=1)
        wind_df['timediff'] = wind_df.apply(lambda row: row['next_TIME'] - row['prev_TIME'], axis=1)
        #wind_df['VELOCITY_calc'] = wind_df.apply(lambda row: row['distance']/row['timediff'],axis=1)
        wind_df['VELOCITY_calc'] = wind_df.apply(lambda row:calc_velocity(row['timediff'], row['distance']),axis=1)

        wind_df['VELOCITY'] = wind_df.apply(lambda x: (str(x.VELOCITY)), axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda x: 0 if x.VELOCITY == 'XX.X' else x.VELOCITY, axis=1)
        wind_df['fVel'] = wind_df["VELOCITY"].str.split(".", n=1, expand=True)[0]
        wind_df = wind_df.loc[wind_df['fVel'].notnull(),].reset_index(drop=True)
        wind_df['firstVel'] = wind_df.apply(lambda x: int(x['fVel']), axis=1)

        wind_df['sVel'] = wind_df["VELOCITY"].str.split(".", n=1, expand=True)[1]
        wind_df = wind_df.loc[wind_df['sVel'].notnull(),].reset_index(drop=True)
        wind_df['secVel'] = wind_df.apply(lambda x: int(x['sVel']), axis=1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstVel) + '.' + str(x.secVel)), axis=1)
        wind_df2 = wind_df.drop(columns=['VELOCITY', 'secVel', 'sVel', 'fVel', 'firstVel'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'vloc': 'VELOCITY'})
        wind_df = wind_df2.copy()
        del (wind_df2)
        ## CORRECT W WIND THING
        wind_df['W'] = wind_df.apply(lambda x: (str(x.W)), axis=1)
        wind_df['W'] = wind_df.apply(lambda x: 0 if x.W == 'XX.X' else x.W, axis=1)
        wind_df['fW'] = wind_df["W"].str.split(".", n=1, expand=True)[0]
        # wind_df = wind_df.loc[wind_df['fW'].notnull(),].reset_index(drop=True)
        wind_df['firstW'] = wind_df.apply(lambda x: int(x['fW']), axis=1)
        wind_df['sW'] = wind_df["W"].str.split(".", n=1, expand=True)[1]
        # wind_df = wind_df.loc[wind_df['sW'].notnull(),].reset_index(drop=True)
        wind_df['secW'] = wind_df.apply(lambda x: int(x['sW']), axis=1)
        wind_df['wloc'] = wind_df.apply(lambda x: float(str(x.firstW) + '.' + str(x.secW)), axis=1)
        wind_df2 = wind_df.drop(columns=['W', 'secW', 'sW', 'fW', 'firstW'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'wloc': 'W'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        ## CORRECT U WIND THING
        wind_df['U'] = wind_df.apply(lambda x: (str(x.U)), axis=1)
        wind_df['U'] = wind_df.apply(lambda x: 0 if x.U == 'XX.X' else x.U, axis=1)
        wind_df['fU'] = wind_df["U"].str.split(".", n=1, expand=True)[0]
        wind_df['firstU'] = wind_df.apply(lambda x: int(x['fU']), axis=1)
        wind_df['sU'] = wind_df["U"].str.split(".", n=1, expand=True)[1]
        wind_df['secU'] = wind_df.apply(lambda x: int(x['sU']), axis=1)
        wind_df['uloc'] = wind_df.apply(lambda x: float(str(x.firstU) + '.' + str(x.secU)), axis=1)
        wind_df2 = wind_df.drop(columns=['U', 'secU', 'sU', 'fU', 'firstU'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'uloc': 'U'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        ## CORRECT V WIND THING
        wind_df['V'] = wind_df.apply(lambda x: (str(x.V)), axis=1)
        wind_df['V'] = wind_df.apply(lambda x: 0 if x.V == 'XX.X' else x.V, axis=1)
        wind_df['fV'] = wind_df["V"].str.split(".", n=1, expand=True)[0]
        wind_df['firstV'] = wind_df.apply(lambda x: int(x['fV']), axis=1)
        wind_df['sV'] = wind_df["V"].str.split(".", n=1, expand=True)[1]
        wind_df['secV'] = wind_df.apply(lambda x: int(x['sV']), axis=1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstV) + '.' + str(x.secV)), axis=1)
        wind_df2 = wind_df.drop(columns=['V', 'secV', 'sV', 'fV', 'firstV'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'vloc': 'V'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        wind_df['U_cor'] = wind_df.apply(lambda row: float(row['U']) + float(row['VELOCITY_calc']), axis=1)
        wind_df['horz_length'] = wind_df.apply(lambda row: np.sqrt(row['U_cor'] ** 2 + row['V'] ** 2), axis=1)
        wind_df['uncor_theta'] = wind_df.apply(
            lambda row: calc_bearing(row['U_cor'], row['V'], row['QUADRANT'], row['horz_length'], radians), axis=1)
        wind_df['adj_theta'] = wind_df.apply(lambda row: (row['uncor_theta'] + row['bearing']) % 360, axis=1)
        wind_df['totalWind'] = wind_df.apply(lambda row: np.sqrt(row['horz_length'] ** 2 + row['W'] ** 2), axis=1)
        wind_df['phi'] = wind_df.apply(lambda row: np.arctan(row['horz_length']), axis=1)

        wind_df['adj_v'] = wind_df.apply(lambda row: -row['horz_length'] * np.cos(row['adj_theta']), axis=1)
        wind_df['adj_u'] = wind_df.apply(lambda row: row['horz_length'] * np.sin(row['adj_theta']), axis=1)

        ## GO THROUGH WIND
        window_size = 30
        u_series = pd.Series(wind_df['adj_u'])
        u_windows = u_series.rolling(window_size)
        u_averages = pd.DataFrame(u_windows.mean())
        u_averages.columns = ['U_avg']
        u_averages['key'] = u_averages.index

        v_series = pd.Series(wind_df['adj_v'])
        v_windows = v_series.rolling(window_size)
        v_averages = pd.DataFrame(v_windows.mean())
        v_averages.columns = ['V_avg']
        v_averages['key'] = v_averages.index

        w_series = pd.Series(wind_df['W'])
        w_windows = w_series.rolling(window_size)
        w_averages = pd.DataFrame(w_windows.mean())
        w_averages.columns = ['W_avg']
        w_averages['key'] = w_averages.index

        vw_df = w_averages.set_index('key').join(v_averages.set_index('key'))
        vw_df['key'] = vw_df.index
        uvw_df = vw_df.set_index('key').join(u_averages.set_index('key'))
        uvw_df['key'] = uvw_df.index
        wind_df2 = wind_df.copy()
        wind_df2['key'] = wind_df2.index
        wind_df = uvw_df.set_index('key').join(wind_df2.set_index('key'))

        wind_df['r_avg'] = wind_df.apply(lambda row: np.sqrt(row['U_avg'] ** 2 + row['V_avg'] ** 2), axis=1)
        wind_df['theta_avg'] = wind_df.apply(lambda row: np.arctan(-row['U_avg'] / row['V_avg']), axis=1)

        wind_df['shift_CH4'] = wind_df.CH4.shift(periods=int(float(shift)))
        wind_df['raw_CH4'] = wind_df.apply(lambda row: row['BCH4'], axis=1)
        wind_df['BCH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['CH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['TCH4'] = wind_df.loc[:, ['shift_CH4']]

        wind_df['shift_R'] = wind_df.R.shift(periods=int(float(shift)))
        wind_df['raw_R'] = wind_df.apply(lambda row: row['R'], axis=1)

        wind_df2 = wind_df[wind_df.CH4.notnull()]
        wind_df2_null = wind_df[~wind_df.CH4.notnull()]
        if wind_df2_null.shape[0] > 0:
            wind_df2_null=wind_df2_null.assign(Reason='GPS NA')
        nullCH4 = pd.concat([wind_df_null,wind_df2_null])


        wind_df2 = wind_df.copy()
        wind_df3 = wind_df2.drop(
            ['QUADRANT', 'secnan', 'prev_LAT', 'next_LAT', 'prev_LONG', 'next_LONG', 'prev_TIME', 'next_TIME',
             'timediff', 'uncor_theta', 'CH4', 'R','VELOCITY'], axis=1)
        wind_df3['CH4'] = wind_df3.loc[:, 'shift_CH4']
        wind_df3['R'] = wind_df3.loc[:, 'shift_R']
        wind_df3 = wind_df3.drop(['shift_CH4', 'shift_R'], axis=1)
        # wind_df4 = wind_df3.loc[wind_df3.totalWind.notnull(),:]
        wind_df3['odometer'] = wind_df3.loc[:, 'distance'].cumsum()
        wind_df3a = wind_df3.copy().rename(columns = {'VELOCITY_calc':'VELOCITY'})
        wind_df4 = wind_df3a.loc[:,
                   ['DATE', 'TIME', 'SECONDS', 'NANOSECONDS', 'VELOCITY', 'U', 'V', 'W', 'BCH4', 'BRSSI', 'TCH4',
                    'TRSSI',
                    'PRESS_MBAR', 'INLET', 'TEMPC', 'CH4', 'H20', 'C2H6', 'R', 'C2C1', 'BATTV', 'POWMV', 'CURRMA',
                    'SOCPER',
                    'LAT', 'LONG', 'bearing', 'U_cor', 'horz_length', 'adj_theta', 'totalWind',
                    'phi', 'raw_CH4', 'raw_R', 'U_avg', 'V_avg', 'W_avg', 'r_avg', 'theta_avg', 'distance', 'odometer']]

        # wind_df7 = add_odometer(wind_df4,'LAT','LONG')

        # wind_df4 = wind_df7.copy()
        #wind_df5 = wind_df4.loc[wind_df4.VELOCITY > xMinCarSpeed, :]
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > -1, :]

        wrongSpeed = wind_df4.loc[wind_df4.VELOCITY <= xMinCarSpeed,:]
        wrongSpeed=wrongSpeed.assign(Reason='velocity too slow')

        #wind_df6 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :]
        wind_df6 = wind_df5.loc[wind_df5.VELOCITY < 1000, :]

        wrongSpeed2 =  wind_df5.loc[wind_df5.VELOCITY >= xMaxCarSpeed, :]
        wrongSpeed2 = wrongSpeed2.assign(Reason='velocity too fast')

        wrongSpeeds = pd.concat([wrongSpeed,wrongSpeed2])


        notGood = pd.concat([wrongSpeeds,nullCH4])
        # wind_df6 = wind_df6a.loc[wind_df6a.R > .6999, :]

        del (wind_df4)
        wind_df4 = wind_df6.copy().drop_duplicates()
        wind_df5 = wind_df4.loc[wind_df4.CH4.notnull(), :]

        nullCH4 = wind_df4.loc[~wind_df4.CH4.notnull(), :]
        if nullCH4.shape[0] > 0:
            nullCH4 = nullCH4.assign(Reason='CH4 NA')
            removedDF = pd.concat([notGood,nullCH4])
        if nullCH4.shape[0]==0:
            removedDF = notGood
        wind_df4 = wind_df5.copy()

        ## if you want to filter out high temperatures
        #wind_df4 = wind_df5.loc[wind_df5.TEMPC < 95, :].reset_index(drop=True)

        fLog.write("Usable lines - " + str(wind_df4.shape[0]) + "." + "\n")
        fLog.close()

        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
            removedDF.to_csv(removeOut,index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
            removed = pd.read_csv(removeOut)
            pd.concat([removed, removedDF]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(removeOut, index=False)

        os.remove(fnOutTemp)
        return True
    except ValueError:
        return False

def process_raw_data_aeris_notchanged(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack, shift, maxSpeed='45',
                           minSpeed='2'):
    """ input a raw .txt file with data (from aeris data file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import sys
    if sys.platform == 'win32':
        windows = True
    elif sys.platform != 'win32':
        windows = False

    import pandas as pd
    from datetime import datetime
    import os
    import gzip
    import numpy as np
    # import csv
    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)

        ########################################################################
        #### WE DON'T HAVE AN RSSI INPUT
        ### (SO THIS IS A PLACEHOLDER FOR SOME SORT OF QA/QC VARIABLE)
        ##  xMinRSSI = 50  #if RSSI is below this we don't like it
        ##################################################################

        # reading in the data with specific headers
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        sHeader = 'Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude,U (m/sec),V (m/sec),W (m/sec),T (degC),Dir (deg),Speed (m/sec),Compass (deg)'
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"
        infoHeader = "FILENAME\n"
        # somehow gZIP is indicating if  it is the first file name (I think if it is 0 then it is the first file)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename,
                          'r')  # if in python 3, change this to "r" or just "b" can't remember but something about a bit not a string
        else:
            f = open(xDir + xFilename, 'r')

        infoHeader = "FILENAME\n"

        # process - if first time on this car/date, then write header out
        headerNames = sHeader.split(',')
        xdat = str('20') + xFilename[11:17]
        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        fnLog = xOut + xCar + "_" + xdat + "_log.csv"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"
        #

        dtime = open(xDir + xFilename).readlines().pop(2).split(',')[0]
        firstdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                             int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
        # firsttime = firstdate.strftime('%s.%f')
        firsttime = dt_to_epoch(firstdate)
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        if bFirst:
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")
        if not bFirst:
            fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        fOut = open(fnOutTemp, 'w')
        fOut.write(sOutHeader)

        # read all lines
        xCntObs = -1
        xCntGoodValues = 0
        for row in f:
            woo = row
            bGood = True
            if xCntObs != -1:
                lstS = row.split(",")
                if float(lstS[2]) < 20:
                    bGood = False
                    xCntObs += 1
            if xCntObs < 0:
                bGood = False
                xCntObs += 1
            if bGood:
                lstS = row.split(",")
                dtime = lstS[0]
                dateob = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                  int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                fdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
                                 int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
                # seconds = fdate.strftime('%s.%f')
                seconds = dt_to_epoch(fdate)

                def getNS(seconds):
                    ns = str(float(seconds) * 1e-3)[11:]
                    # str(pd.to_numeric(str(float(seconds) * 1e-3)[11:]) * 100000)[:9]
                    return (str(ns).ljust(15, '0'))[:9]

                if len(lstS) > 6 and float(lstS[2]) > 20:
                    csvWrite = str(dateob.strftime('%Y-%m-%d')) + ',' + str(
                        dateob.strftime('%H:%M:%S')) + ',' + str(str(float(seconds) * 1e-3)[:10]) + ',' + getNS(
                        seconds) + str(',')
                    csvWrite += str(lstS[20]) + ',' + str(lstS[15]) + ',' + str(lstS[16]) + ',' + str(
                        lstS[17]) + ',' + str(
                        lstS[4]) + ',' + str('0') + ',' + str(lstS[4]) + ','
                    csvWrite += str('0') + ',' + str(lstS[2]) + ',' + str(lstS[1]) + ',' + str(
                        lstS[3]) + ',' + str(lstS[4]) + ',' + str(lstS[5]) + ',' + str(lstS[6]) + ','
                    csvWrite += str(lstS[7]) + ',' + str(lstS[8]) + ',' + str(lstS[9]) + ',' + str(
                        lstS[10]) + ',' + str(lstS[11]) + ',' + str(lstS[12]) + ',' + str(lstS[13]) + str(
                        ',') + str(lstS[14]) + '\n'
                    fOut.write(csvWrite)
                    xCntObs += 1
        infOut.write(str(xFilename) + '\n')
        fOut.close()
        fLog.close()
        infOut.close()
        # print(xCar + "\t" + xdat + "\t" + fnOut[-22:] + "\t" + str(xCntObs) + "\t" + str(xCntGoodValues) + "\t" + str(
        #    gZIP))
        print(f"{xCar} \t {xdat} \t {fnOut[-(17 + len(xCar)):]} \t  {xCntObs} \t {xCntGoodValues} \t {gZIP}")

        wind_df = pd.read_csv(fnOutTemp)
        wind_df_not_null = wind_df.loc[wind_df['LAT'].notnull(),].reset_index(drop=True)
        del (wind_df)
        wind_df = wind_df_not_null.copy()

        radians = False
        wind_df['QUADRANT'] = wind_df.apply(lambda row: get_quadrant(row['U'], row['V']), axis=1)
        wind_df['secnan'] = wind_df.apply(lambda row: row['SECONDS'] + row['NANOSECONDS'] * 1e-9,
                                          axis=1)  # + row['NANOSECONDS']*1e-9,axis=1)
        wind_df['prev_LAT'] = wind_df.LAT.shift(periods=1)
        wind_df['next_LAT'] = wind_df.LAT.shift(periods=-1)
        wind_df['prev_LONG'] = wind_df.LONG.shift(periods=1)
        wind_df['next_LONG'] = wind_df.LONG.shift(periods=-1)
        wind_df['prev_TIME'] = wind_df.secnan.shift(periods=1)
        wind_df['next_TIME'] = wind_df.secnan.shift(periods=-1)
        wind_df['distance'] = wind_df.apply(
            lambda row: haversine(row['prev_LAT'], row['prev_LONG'], row['next_LAT'], row['next_LONG']), axis=1)
        wind_df['bearing'] = wind_df.apply(
            lambda row: calc_bearing(row['prev_LAT'], row['next_LAT'], row['prev_LONG'], row['next_LONG'], radians),
            axis=1)
        wind_df['timediff'] = wind_df.apply(lambda row: row['next_TIME'] - row['prev_TIME'], axis=1)
        # wind_df['VELOCITY_calc'] = wind_df.apply(lambda row:calc_velocity(row['timediff'],row['distance']),axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda x: (str(x.VELOCITY)), axis=1)
        wind_df['VELOCITY'] = wind_df.apply(lambda x: 0 if x.VELOCITY == 'XX.X' else x.VELOCITY, axis=1)
        wind_df['fVel'] = wind_df["VELOCITY"].str.split(".", n=1, expand=True)[0]
        wind_df = wind_df.loc[wind_df['fVel'].notnull(),].reset_index(drop=True)
        wind_df['firstVel'] = wind_df.apply(lambda x: int(x['fVel']), axis=1)

        wind_df['sVel'] = wind_df["VELOCITY"].str.split(".", n=1, expand=True)[1]
        wind_df = wind_df.loc[wind_df['sVel'].notnull(),].reset_index(drop=True)
        wind_df['secVel'] = wind_df.apply(lambda x: int(x['sVel']), axis=1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstVel) + '.' + str(x.secVel)), axis=1)
        wind_df2 = wind_df.drop(columns=['VELOCITY', 'secVel', 'sVel', 'fVel', 'firstVel'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'vloc': 'VELOCITY'})
        wind_df = wind_df2.copy()
        del (wind_df2)
        ## CORRECT W WIND THING
        wind_df['W'] = wind_df.apply(lambda x: (str(x.W)), axis=1)
        wind_df['W'] = wind_df.apply(lambda x: 0 if x.W == 'XX.X' else x.W, axis=1)
        wind_df['fW'] = wind_df["W"].str.split(".", n=1, expand=True)[0]
        # wind_df = wind_df.loc[wind_df['fW'].notnull(),].reset_index(drop=True)
        wind_df['firstW'] = wind_df.apply(lambda x: int(x['fW']), axis=1)
        wind_df['sW'] = wind_df["W"].str.split(".", n=1, expand=True)[1]
        # wind_df = wind_df.loc[wind_df['sW'].notnull(),].reset_index(drop=True)
        wind_df['secW'] = wind_df.apply(lambda x: int(x['sW']), axis=1)
        wind_df['wloc'] = wind_df.apply(lambda x: float(str(x.firstW) + '.' + str(x.secW)), axis=1)
        wind_df2 = wind_df.drop(columns=['W', 'secW', 'sW', 'fW', 'firstW'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'wloc': 'W'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        ## CORRECT U WIND THING
        wind_df['U'] = wind_df.apply(lambda x: (str(x.U)), axis=1)
        wind_df['U'] = wind_df.apply(lambda x: 0 if x.U == 'XX.X' else x.U, axis=1)
        wind_df['fU'] = wind_df["U"].str.split(".", n=1, expand=True)[0]
        wind_df['firstU'] = wind_df.apply(lambda x: int(x['fU']), axis=1)
        wind_df['sU'] = wind_df["U"].str.split(".", n=1, expand=True)[1]
        wind_df['secU'] = wind_df.apply(lambda x: int(x['sU']), axis=1)
        wind_df['uloc'] = wind_df.apply(lambda x: float(str(x.firstU) + '.' + str(x.secU)), axis=1)
        wind_df2 = wind_df.drop(columns=['U', 'secU', 'sU', 'fU', 'firstU'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'uloc': 'U'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        ## CORRECT V WIND THING
        wind_df['V'] = wind_df.apply(lambda x: (str(x.V)), axis=1)
        wind_df['V'] = wind_df.apply(lambda x: 0 if x.V == 'XX.X' else x.V, axis=1)
        wind_df['fV'] = wind_df["V"].str.split(".", n=1, expand=True)[0]
        wind_df['firstV'] = wind_df.apply(lambda x: int(x['fV']), axis=1)
        wind_df['sV'] = wind_df["V"].str.split(".", n=1, expand=True)[1]
        wind_df['secV'] = wind_df.apply(lambda x: int(x['sV']), axis=1)
        wind_df['vloc'] = wind_df.apply(lambda x: float(str(x.firstV) + '.' + str(x.secV)), axis=1)
        wind_df2 = wind_df.drop(columns=['V', 'secV', 'sV', 'fV', 'firstV'])
        del (wind_df)
        wind_df2 = wind_df2.rename(columns={'vloc': 'V'})
        wind_df = wind_df2.copy()
        del (wind_df2)

        wind_df['U_cor'] = wind_df.apply(lambda row: float(row['U']) + float(row['VELOCITY']), axis=1)
        wind_df['horz_length'] = wind_df.apply(lambda row: np.sqrt(row['U_cor'] ** 2 + row['V'] ** 2), axis=1)
        wind_df['uncor_theta'] = wind_df.apply(
            lambda row: calc_bearing(row['U_cor'], row['V'], row['QUADRANT'], row['horz_length'], radians), axis=1)
        wind_df['adj_theta'] = wind_df.apply(lambda row: (row['uncor_theta'] + row['bearing']) % 360, axis=1)
        wind_df['totalWind'] = wind_df.apply(lambda row: np.sqrt(row['horz_length'] ** 2 + row['W'] ** 2), axis=1)
        wind_df['phi'] = wind_df.apply(lambda row: np.arctan(row['horz_length']), axis=1)

        wind_df['adj_v'] = wind_df.apply(lambda row: -row['horz_length'] * np.cos(row['adj_theta']), axis=1)
        wind_df['adj_u'] = wind_df.apply(lambda row: row['horz_length'] * np.sin(row['adj_theta']), axis=1)

        ## GO THROUGH WIND
        window_size = 30
        u_series = pd.Series(wind_df['adj_u'])
        u_windows = u_series.rolling(window_size)
        u_averages = pd.DataFrame(u_windows.mean())
        u_averages.columns = ['U_avg']
        u_averages['key'] = u_averages.index

        v_series = pd.Series(wind_df['adj_v'])
        v_windows = v_series.rolling(window_size)
        v_averages = pd.DataFrame(v_windows.mean())
        v_averages.columns = ['V_avg']
        v_averages['key'] = v_averages.index

        w_series = pd.Series(wind_df['W'])
        w_windows = w_series.rolling(window_size)
        w_averages = pd.DataFrame(w_windows.mean())
        w_averages.columns = ['W_avg']
        w_averages['key'] = w_averages.index

        vw_df = w_averages.set_index('key').join(v_averages.set_index('key'))
        vw_df['key'] = vw_df.index
        uvw_df = vw_df.set_index('key').join(u_averages.set_index('key'))
        uvw_df['key'] = uvw_df.index
        wind_df2 = wind_df.copy()
        wind_df2['key'] = wind_df2.index
        wind_df = uvw_df.set_index('key').join(wind_df2.set_index('key'))

        wind_df['r_avg'] = wind_df.apply(lambda row: np.sqrt(row['U_avg'] ** 2 + row['V_avg'] ** 2), axis=1)
        wind_df['theta_avg'] = wind_df.apply(lambda row: np.arctan(-row['U_avg'] / row['V_avg']), axis=1)

        wind_df['shift_CH4'] = wind_df.CH4.shift(periods=int(float(shift)))
        wind_df['raw_CH4'] = wind_df.apply(lambda row: row['BCH4'], axis=1)
        wind_df['BCH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['CH4'] = wind_df.loc[:, ['shift_CH4']]
        wind_df['TCH4'] = wind_df.loc[:, ['shift_CH4']]

        wind_df['shift_R'] = wind_df.R.shift(periods=int(float(shift)))
        wind_df['raw_R'] = wind_df.apply(lambda row: row['R'], axis=1)

        wind_df2 = wind_df[wind_df.CH4.notnull()]

        wind_df2 = wind_df.copy()
        wind_df3 = wind_df2.drop(
            ['QUADRANT', 'secnan', 'prev_LAT', 'next_LAT', 'prev_LONG', 'next_LONG', 'prev_TIME', 'next_TIME',
             'timediff', 'uncor_theta', 'CH4', 'R'], axis=1)
        wind_df3['CH4'] = wind_df3.loc[:, 'shift_CH4']
        wind_df3['R'] = wind_df3.loc[:, 'shift_R']
        wind_df3 = wind_df3.drop(['shift_CH4', 'shift_R'], axis=1)
        # wind_df4 = wind_df3.loc[wind_df3.totalWind.notnull(),:]
        wind_df3['odometer'] = wind_df3.loc[:, 'distance'].cumsum()
        wind_df4 = wind_df3.loc[:,
                   ['DATE', 'TIME', 'SECONDS', 'NANOSECONDS', 'VELOCITY', 'U', 'V', 'W', 'BCH4', 'BRSSI', 'TCH4',
                    'TRSSI',
                    'PRESS_MBAR', 'INLET', 'TEMPC', 'CH4', 'H20', 'C2H6', 'R', 'C2C1', 'BATTV', 'POWMV', 'CURRMA',
                    'SOCPER',
                    'LAT', 'LONG', 'bearing', 'U_cor', 'horz_length', 'adj_theta', 'totalWind',
                    'phi', 'raw_CH4', 'raw_R', 'U_avg', 'V_avg', 'W_avg', 'r_avg', 'theta_avg', 'distance', 'odometer']]

        # wind_df7 = add_odometer(wind_df4,'LAT','LONG')

        # wind_df4 = wind_df7.copy()
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > xMinCarSpeed, :]
        wind_df6 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :]

        # wind_df6 = wind_df6a.loc[wind_df6a.R > .6999, :]

        del (wind_df4)
        wind_df4 = wind_df6.copy().drop_duplicates()
        wind_df5 = wind_df4.loc[wind_df4.CH4.notnull(), :]
        # wind_df4 = wind_df5.copy()
        wind_df4 = wind_df5.loc[wind_df5.TEMPC < 95, :].reset_index(drop=True)

        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
        os.remove(fnOutTemp)
        return True
    except ValueError:
        return False

def identify_peaks(xCar, xDate, xDir, xFilename, outDir, processedFileLoc, Engineering, threshold='.1',
                   rthresh = '.7',
                  xTimeThreshold='5.0', minElevated='2', xB='102', basePerc='50'):
    """ input a processed data file, and finds the locations of the elevated readings (observed peaks)
    input:
        xCar: name of the car (to make filename)
        xDate: date of the reading
        xDir: directory where the file is located
        xFilename: name of the file
        outDir: directory to take it
        processedFileLoc
        Engineering: T/F if the processed file was made using an engineering file
        threshold: the proportion above baseline that is marked as elevated (i.e. .1 corresponds to 10% above
        xTimeThreshold: not super sure
        minElevated: # of elevated readings that need to be there to constitute an observed peak
        xB: Number of observations used in background average
        basePerc: percentile used for background average (i.e. 50 is median)
    output:
        saved log file
        saved csv file with identified peaks
        saved info.csv file
        saved json file
    """
    import csv, numpy
    import shutil
    from shapely.geometry import Point
    import pandas as pd
    import geopandas as gpd


    try:
        baseCalc = float(basePerc)
        xABThreshold = float(threshold)
        minElevated = float(minElevated)
        rMin = float(rthresh)
        xDistThreshold = 160.0  # find the maximum CH4 reading of observations within street segments of this grouping distance in meters
        xSDF = 4  # multiplier times standard deviation for floating baseline added to mean

        xB = int(xB)
        xTimeThreshold = float(xTimeThreshold)
        fn = xDir + xFilename  # set processed csv file to read in
        fnOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".csv"
        fnShape = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".shp"
        fnLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".log"
        pkLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + "_info.csv"
        jsonOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + ".geojson"
        infOut = processedFileLoc + xCar + "_" + xDate.replace("-", "") + "_info.csv"

        ### TEST THING
        fn = xDir + xFilename  # set raw text file to read in
        filenames = nameFiles(outDir,processedFileLoc,xCar,xDate,True)
        fnOut = filenames['fnOut']
        fnShape = filenames['fnShape']
        fnLog = filenames['fnLog']
        pkLog = filenames['pkLog']
        jsonOut = filenames['jsonOut']
        infOut = filenames['infOut']

        print(f"{outDir}Peaks_{xCar}_{xDate}_info.csv")
        fLog = open(fnLog, 'w')
        shutil.copy(infOut, pkLog)

        # field column indices for various variables
        if Engineering == True:
            fDate = 0;  fTime = 1; fEpochTime = 2
            fNanoSeconds = 3; fVelocity = 4;  fU = 5
            fV = 6; fW = 7; fBCH4 = 10
            fBCH4 = 8;  fBRSSI = 9; fTCH4 = 10
            TRSSI = 11;PRESS = 12; INLET = 13
            TEMP = 14;  CH4 = 15;H20 = 16
            C2H6 = 17;  R = 18;  C2C1 = 19
            BATT = 20;  POWER = 21; CURR = 22
            SOCPER = 23;fLat = 24; fLon = 25
        elif not Engineering:
            fDate = 0; fTime = 1; fEpochTime = 2
            fNanoSeconds = 3;fVelocity = 4; fU = 5
            fV = 6;  fW = 7
            fBCH4 = 8; fBRSSI = 9
            fTCH4 = 10;  TRSSI = 11;  PRESS = 12
            INLET = 13;  TEMP = 14; CH4 = 15
            H20 = 16;C2H6 = 17;  R = 18; C2C1 = 19
            BATT = 20; POWER = 21; CURR = 22
            SOCPER = 23; fLat = 24;fLon = 25;
            fUavg = 33; fVavg = 34; fWavg = 35;
            fRavg = 36; fthetavg=37;
            fDist = 38; fOdometer = 39

            # read data in from text file and extract desired fields into a list, padding with 5 minute and hourly average
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,x11,x12,x13,x14,x15,x16,x17,x18 = [[] for _ in range(18)]

            count = -1
            with open(fn, 'r') as f:
                t = csv.reader(f)
                for row in t:
                    woo = row
                    # print(count)
                    if count < 0:
                        count += 1
                        continue
                    elif count >= 0:
                        datet = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                        ## if not engineering
                        epoch = float(row[fEpochTime] + "." + row[fNanoSeconds][0])
                        datetime = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                        x1.append(epoch); x2.append(datetime)
                        if row[fLat] == '':
                            x3.append('')
                        elif row[fLat] != '':
                            x3.append(float(row[fLat]))
                        if row[fLon] == '':
                            x4.append('')
                        elif row[fLon] != '':
                            x4.append(float(row[fLon]))

                        if row[fUavg] == '':
                            x14.append('')
                        elif row[fUavg] != '':
                            x14.append(float(row[fUavg]))
                        if row[fVavg] == '':
                            x15.append('')
                        elif row[fVavg] != '':
                            x15.append(float(row[fVavg]))
                        if row[fWavg] == '':
                            x16.append('')
                        elif row[fWavg] != '':
                            x16.append(float(row[fWavg]))

                        if row[fthetavg] == '':
                            x18.append('')
                        elif row[fthetavg] != '':
                            x18.append(float(row[fthetavg]))
                        if row[fRavg] == '':
                            x17.append('')
                        elif row[fRavg] != '':
                            x17.append(float(row[fRavg]))

                        x5.append(float(row[fBCH4]))
                        x6.append(float(row[fTCH4]))
                        x7.append(0.0)
                        x8.append(0.0)
                        x9.append(row[fOdometer])
                        x11.append(float(row[C2H6]))
                        x12.append(float(row[C2C1]))
                        x13.append(float(row[R]))
                        count += 1
            print(f"Number of observations processed:{count}")

        # convert lists to numpy arrays
        aEpochTime = numpy.array(x1)
        aDateTime = numpy.array(x2)
        aLat = numpy.array(x3)
        aLon = numpy.array(x4)
        aCH4 = numpy.array(x5)
        aTCH4 = numpy.array(x6)
        aMean = numpy.array(x7)
        aMeanC2H6 = numpy.array(x7)

        aMeanCH4_true = numpy.array(x7)
        aMedianCH4 = numpy.array(x7)
        aMaxCH4 = numpy.array(x7)
        aMinCH4 = numpy.array(x7)

        aSTDCH4 = numpy.array(x7)

        aThreshold = numpy.array(x8)
        aOdom = numpy.array(x9)

        # adding ethane stuff
        aC2H6 = numpy.array(x11)
        aC2C1 = numpy.array(x12)
        aR = numpy.array(x13)
        aUavg = numpy.array(x14)
        aVavg = numpy.array(x15)
        aWavg = numpy.array(x16)
        aRavg = numpy.array(x17)
        aThavg = numpy.array(x18)


        xLatMean = numpy.mean(aLat)
        xLonMean = numpy.mean(aLon)
        #xCH4Mean = numpy.mean(aCH4)
        #xC2H6Mean = numpy.mean(aC2H6)
        #xC2C1Mean = numpy.mean(aC2C1)

        fLog.write("Day CH4_mean = " + str(numpy.mean(aCH4)) +
                   ", Day CH4 SD = " + str(numpy.std(aCH4)) + "\n")
        fLog.write("Day C2H6 Mean = " + str(numpy.mean(aC2H6)) +
                   ", Day C2H6 SD = " + str(numpy.std(aC2H6)) + "\n")
        fLog.write("Center lon/lat = " + str(xLonMean) + ", " + str(xLatMean) + "\n")

        lstCH4_AB = []

        # generate list of the index for observations that were above the threshold
        for i in range(0, count - 2):
            if ((count - 2) > xB):
                topBound = min((i + xB), (count - 2))
                botBound = max((i - xB), 0)

                for t in range(min((i + xB), (count - 2)), i, -1):
                    if aEpochTime[t] < (aEpochTime[i] + (xB / 2)):
                        topBound = t
                        break
                for b in range(max((i - xB), 0), i):
                    if aEpochTime[b] > (aEpochTime[i] - (xB / 2)):
                        botBound = b
                        break

                xCH4Mean = numpy.percentile(aCH4[botBound:topBound], baseCalc)
                xC2H6Mean = numpy.percentile(aC2H6[botBound:topBound], baseCalc)
                xMeanCH4_true = numpy.mean(aCH4[botBound:topBound])
                xSTDCH4 = numpy.std(aCH4[botBound:topBound])
                xMaxCH4 = numpy.max(aCH4[botBound:topBound])
                xMinCH4 =  numpy.min(aCH4[botBound:topBound])
                xMedianCH4 = numpy.percentile(aCH4[botBound:topBound], 50)

            # xCH4SD = numpy.std(aCH4[botBound:topBound])
            else:
                xCH4Mean = numpy.percentile(aCH4[0:(count - 2)], baseCalc)
                xC2H6Mean = numpy.percentile(aC2H6[0:(count - 2)], baseCalc)
                xMeanCH4_true = numpy.mean(aCH4[0:(count - 2)])
                xSTDCH4 = numpy.std(aCH4[0:(count - 2)])
                xMaxCH4 = numpy.max(aCH4[0:(count - 2)])
                xMinCH4 = numpy.min(aCH4[0:(count - 2)])
                xMedianCH4 = numpy.percentile(aCH4[0:(count - 2)], 50)


                # xCH4SD = numpy.std(aCH4[0:(count-2)])
            xThreshold = xCH4Mean + (xCH4Mean * xABThreshold)
            xThreshold_c2h6 = xC2H6Mean + (xC2H6Mean * xABThreshold)

            if (aCH4[i] > xThreshold and aR[i]>rMin):
            #if (aCH4[i] > xThreshold):
                lstCH4_AB.append(i)
                aMean[i] = xCH4Mean
                aMeanC2H6[i] = xC2H6Mean
                aThreshold[i] = xThreshold
                aMeanCH4_true[i] = xMeanCH4_true
                aSTDCH4[i] = xSTDCH4
                aMaxCH4[i] = xMaxCH4
                aMinCH4[i] = xMinCH4
                aMedianCH4[i] = xMedianCH4
        # now group the above baseline threshold observations into groups based on distance threshold
        lstCH4_ABP = []
        xDistPeak = 0.0
        xCH4Peak = 0.0
        xTime = 0.0
        cntPeak = 0
        cnt = 0
        sID = ""
        sPeriod5Min = ""
        prevIndex = 0
        for i in lstCH4_AB:
            if (cnt == 0):
                xLon1 = aLon[i]
                xLat1 = aLat[i]
                xOdom = aOdom[i]
            else:
                # calculate distance between points
                xDist = haversine(xLat1, xLon1, aLat[i], aLon[i])
                xDistPeak += xDist
                xCH4Peak += (xDist * (aCH4[i] - aMean[i]))
                xLon1 = aLon[i]
                xLat1 = aLat[i]
                xOdom = aOdom[i]
                if (sID == ""):
                    xTime = aEpochTime[i]
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                if ((aEpochTime[i] - aEpochTime[prevIndex]) > xTimeThreshold):  # initial start of a observed peak
                    cntPeak += 1
                    xTime = aEpochTime[i]
                    xDistPeak = 0.0
                    xCH4Peak = 0.0
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                    # print str(i) +", " + str(xDist) + "," + str(cntPeak) +"," + str(xDistPeak)
                    #aMeanCH4_true[i], aSTDCH4[i]

                lstCH4_ABP.append(
                    [sID, xTime, aEpochTime[i], aDateTime[i], aCH4[i], aLon[i], aLat[i], aMean[i],aMeanCH4_true[i],aSTDCH4[i],
                     aMaxCH4[i],aMinCH4[i],aMedianCH4[i],aThreshold[i],
                     xDistPeak, xCH4Peak, aTCH4[i],aC2H6[i],aC2C1[i],aR[i],aMeanC2H6[i], sPeriod5Min, xOdom,
                     aUavg[i],aVavg[i],aWavg[i],aRavg[i],aThavg[i]])
            cnt += 1
            prevIndex = i

        # Finding peak_id larger than 160.0 m
        tmpsidlist = []
        for r in lstCH4_ABP:
            if (float(r[9]) > 160.0) and (r[0] not in tmpsidlist):
                tmpsidlist.append(r[0])
        cntPeak -= len(tmpsidlist)

        fLog.write("Number of peaks found: " + str(cntPeak) + "\n")
        print(f"{xCar} \t {xDate} \t {xFilename} \t {count} \t {len(lstCH4_ABP)}")

        # write out the observed peaks to a csv to be read into a GIS
        fOut = open(fnOut, 'w')
        # s = "PEAK_NUM,EPOCHSTART,EPOCH,DATETIME,CH4,LON,LAT,CH4_BASELINE,CH4_THRESHOLD,PEAK_DIST_M,PEAK_CH4,TCH4,PERIOD5MIN\n"
        s = "OP_NUM,OP_EPOCHSTART,OB_EPOCH,OB_DATETIME,OB_CH4,OB_LON,OB_LAT,OB_CH4_BASELINE,OB_CH4_MEAN,OB_CH4_STD,OB_CH4_MAX,OB_CH4_MIN,OB_CH4_MED," \
            "OB_CH4_THRESHOLD,OP_PEAK_DIST_M,OP_PEAK_CH4,OB_TCH4,OB_C2H6," \
            "OB_C2C1,OB_R,OB_C2H6_BASELINE,OB_PERIOD5MIN,ODOMETER,OB_U_AVG,OB_V_AVG,OB_W_AVG," \
            "OB_R_AVG,OB_THETA_AVG\n"
        fOut.write(s)

        truecount = 0
        for r in lstCH4_ABP:
            if r[0] not in tmpsidlist:
                s = ''
                for rr in r:
                    s += str(rr) + ','
                s = s[:-1]
                s += '\n'
                fOut.write(s)
                truecount += 1
        fOut.close()
        fLog.close()

        openFile = pd.read_csv(fnOut)
        if openFile.shape[0] != 0:
            pkDistDf = openFile.copy().groupby('OP_NUM', as_index=False).apply(
                lambda x: max(x.ODOMETER) - min(x.ODOMETER))
            pkDistDf.columns = ['OP_NUM', 'OP_DISTANCE']
            openFile = pd.merge(openFile.copy(), pkDistDf)
            tempCount = openFile.groupby('OP_NUM', as_index=False).OP_EPOCHSTART.count().rename(
                columns={'OP_EPOCHSTART': 'Frequency'})
            tempCount = tempCount.loc[tempCount.Frequency >= minElevated, :]
            if tempCount.shape[0] == 0:
                print(f"No Observed Peaks with enough Elevated Readings Found in the file: {xFilename}")
                tempCount.to_csv(fnOut) ## added to deal with issue where it wasn't being filtered out
            elif tempCount.shape[0] != 0:
                oFile = pd.merge(openFile, tempCount, on=['OP_NUM'])
                openFile = oFile.copy()
                del (oFile)
                openFile["minElevated"] = openFile.apply(lambda x: int(minElevated), axis=1)
                openFile['OB_CH4_AB'] = openFile.loc[:, 'OB_CH4'].sub(openFile.loc[:, 'OB_CH4_BASELINE'], axis=0)
                openFile['OB_C2H6_AB'] = openFile.loc[:, 'OB_C2H6'].sub(openFile.loc[:, 'OB_C2H6_BASELINE'],axis=0)
                openFile.to_csv(fnOut, index=False)


                fileWt = weighted_loc(openFile, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB').loc[:, :].rename(
                    columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'}).reset_index(drop=True)
                geometry_temp = [Point(lon, lat) for lon, lat in zip(fileWt['pk_LON'], fileWt['pk_LAT'])]
                crs = 'EPSG:4326'
                # geometry is the point of the lat/lon
                # gdf_buff = gpd.GeoDataFrame(datFram, crs=crs, geometry=geometry_temp)

                ## BUFFER AROUND EACH 'OP_NUM' WITH BUFFER DISTANCE
                gdf_buff = gpd.GeoDataFrame(fileWt, crs=crs, geometry=geometry_temp)
                # gdf_buff = makeGPD(datFram,'LON','LAT')

                ##maybe this is the issue?
                #gdf_buff = gdf_buff.to_crs(epsg=32610)
                #gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(30)
                try:
                    gdf_buff.to_file(jsonOut, driver="GeoJSON")
                    #gdf_buff.to_file('testthing.geojson', driver="GeoJSON")
                except:
                    print("Error Saving JSON File")
        elif openFile.shape[0] == 0:
            print(f"No Observed Peaks Found in the file:{xFilename}")
    except ValueError:
        print("Error in Identify Peaks")
        return False


def save_results(mainInfo,mainThing,final_info_loc,final_main_csv_loc,shp_file_loc,op_shp_file_loc,all_op_csv_loc,threshold,xCar):
    import pandas as pd
    mainInfo.drop_duplicates().reset_index(drop=True).FILENAME.to_csv(final_info_loc)
    mainThing.reset_index(drop=True).to_csv(final_main_csv_loc)

    combined = summarize_data_2(
        mainThing)  ## finds locations and mean log ch4 for each peak (either verified or non yet)

    ## combined so only with the same overall peak
    uniquePk = combined.loc[:, ['min_read']].drop_duplicates()
    uniqueList = combined.loc[uniquePk.index, ['min_read', 'recombine']]
    uniqueOther = combined.loc[:, ['min_read', 'overallLON', 'overallLAT', 'mnlogCH4',
                                   'verified', 'numtimes', 'minDist', 'maxDist','mn_maxc2h6_ab','mn_maxch4_ab']].drop_duplicates()
    allTog = pd.merge(make_GPD(uniqueOther, 'overallLAT', 'overallLON'), uniqueList, on=['min_read'])
    #allTog['em'] = allTog['mnlogCH4'].swifter.apply(lambda y: estimate_emissions(y))
    allTog['em'] = allTog.apply(lambda y: estimate_emissions(y['mnlogCH4']),axis=1)

    #allTog['threshold'] = allTog['em'].swifter.apply(lambda x: threshold)
    allTog['threshold'] = allTog.apply(lambda x: threshold,axis=1)

    def minread_to_date(min_read, xCar):
        import datetime
        try:
            m_date = int(float(min_read[len(xCar) + 1:]))
            return (datetime.datetime.fromtimestamp(m_date).strftime('%Y-%m-%d %H:%M:%S'))
        except:
            return ('1970-01-01 12:00:00')


    allTog['First_Time'] = allTog.apply(lambda x: minread_to_date(x.min_read, xCar), axis=1)

    ##### SPLITTING IF THE PEAKS WERE VERIFIED OR NOT
    verTog = allTog.loc[allTog.numtimes != 1, :]

    if verTog.size > 0:
        verTog.drop(columns=['recombine']).to_file(shp_file_loc, driver="GeoJSON")
        #verTog.drop(columns=['recombine']).to_file('/Users/ewilliams/Documents/Trussville_data/FinalShpFiles/OP_Final.geojson', driver="GeoJSON")

        print(f'I found {len(verTog.min_read.unique())} verified peaks')
        vpNew = len(verTog.min_read.unique())
    if verTog.size == 0:
        print("Sorry, no verified peaks were found.")
        vpNew = 0
    if allTog.size > 0:
        allTog.drop(columns=['recombine']).to_file(op_shp_file_loc, driver="GeoJSON")
        #allTog.drop(columns=['recombine']).to_file('allOP.geojson', driver="GeoJSON")
        allTog.to_csv(all_op_csv_loc)

    if allTog.size == 0:
        print("Sorry, no observed peaks were found in the given data")
    return


def process_raw_data_amld(xCar, xDate, xDir, xFilename, bFirst, gZIP, xOut, initialTimeBack, shift, maxSpeed='45',
                           minSpeed='2'):
    """ input a raw .txt file with data (from aeris data file)
    input:
        txt file
    output:
        saved log file
        saved csv file with processed data
        saved info.csv file
    """
    import os
    import sys
    if sys.platform == 'win32':
        windows = True
    elif sys.platform != 'win32':
        windows = False

    import pandas as pd
    import datetime
    import os
    import gzip
    import numpy as np
    # import csv
    try:
        xMaxCarSpeed = float(maxSpeed) / 2.23694  # CONVERTED TO M/S (default is 45mph)
        xMinCarSpeed = float(minSpeed) / 2.23694  # CONVERTED TO M/S (default is 2mph)
        xMinCarSpeed = -10
        ########################################################################
        #### WE DON'T HAVE AN RSSI INPUT
        ### (SO THIS IS A PLACEHOLDER FOR SOME SORT OF QA/QC VARIABLE)
        ##  xMinRSSI = 50  #if RSSI is below this we don't like it
        ##################################################################

        # reading in the data with specific headers
        #          0     1    2    3       4           5    6       7        8        9          10                 11              12           13            14      15      16      17        18         19         20         21         22         23        24   25  26       27           28       29           30       31       32       33  34        35   36   37  38   39       40       41   42       43   44   45   46   47   48   49   50   51     52     53     54
        sHeader = "Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude"
        sHeader = 'Time Stamp,Inlet Number,P (mbars),T (degC),CH4 (ppm),H2O (ppm),C2H6 (ppb),R,C2/C1,Battery Charge (V),Power Input (mV),Current (mA),SOC (%),Latitude,Longitude,U (m/sec),V (m/sec),W (m/sec),T (degC),Dir (deg),Speed (m/sec),Compass (deg)'
        sOutHeader = "DATE,TIME,SECONDS,NANOSECONDS,VELOCITY,U,V,W,BCH4,BRSSI,TCH4,TRSSI,PRESS_MBAR,INLET,TEMPC,CH4,H20,C2H6,R,C2C1,BATTV,POWMV,CURRMA,SOCPER,LAT,LONG\n"
        infoHeader = "FILENAME\n"
        # somehow gZIP is indicating if  it is the first file name (I think if it is 0 then it is the first file)
        if gZIP == 0:
            f = gzip.open(xDir + "/" + xFilename,
                          'r')  # if in python 3, change this to "r" or just "b" can't remember but something about a bit not a string
        else:
            f = open(xDir + xFilename, 'r')

        infoHeader = "FILENAME\n"

        # process - if first time on this car/date, then write header out
        headerNames = sHeader.split(',')
        #xdat = str('20') + xFilename[10:16]
        xdat = xDate
        fnOut = xOut + xCar + "_" + xdat + "_dat.csv"  # set CSV output for raw data
        removeOut = xOut + xCar + "_" + xdat + "_removed.csv"
        fnLog = xOut + xCar + "_" + xdat + ".log"  # output for logfile
        infOut = xOut + xCar + "_" + xdat + "_info.csv"
        #

        #dtime = open(xDir + xFilename).readlines().pop(2).split(',')[0]
        #firstdate = datetime(int(dtime[6:10]), int(dtime[0:2]), int(dtime[3:5]), int(dtime[11:13]),
        #                     int(dtime[14:16]), int(dtime[17:19]), int(float(dtime[19:23]) * 1000000))
        # firsttime = firstdate.strftime('%s.%f')
        #firsttime = dt_to_epoch(firstdate)
        firsttime = float(open(xDir + xFilename).readlines().pop(2).split(',')[0])
        fnOutTemp = xOut + xCar + "_" + xdat + "temp_dat.csv"  #

        ## read in file
        tempFile = pd.read_csv(xDir+xFilename)
        tempFile['DATE'] = tempFile.apply(lambda x: datetime.datetime.fromtimestamp(x.nearest10hz).strftime('%Y-%m-%d'),axis=1)
        tempFile['TIME'] = tempFile.apply(lambda x: datetime.datetime.fromtimestamp(x.nearest10hz).strftime('%H:%M:%S'),axis=1)
        tempFile['SECONDS'] = tempFile.apply(lambda x: int(float(str(x.nearest10hz)[10:])*1e9),axis=1)
        tempFile = tempFile.rename(columns = {'Velocity':'VELOCITY',
                                              'Latitude':'LAT',
                                              'Longitude':'LONG'})
        tempFile1 = tempFile.copy().sort_values('nearest10hz').reset_index(drop=True)

        if bFirst:
            #tempFile.sort_values('nearest10hz').reset_index(drop=True).to_csv(fnOutTemp)
            fLog = open(fnLog, 'w')
            infOut = open(infOut, 'w')
            infOut.write(infoHeader)
            print(f"fnLog:{fnOut}")

        if not bFirst:
            #fOut = open(fnOut, 'a')
            fLog = open(fnLog, 'a')
            infOut = open(infOut, 'a')

        #fLog.write("Processing file: " + str(xFilename) + "\n")

        wind_df4 = tempFile1.copy()
        wind_df5 = wind_df4.loc[wind_df4.VELOCITY > -1, :]
        wrongSpeed = wind_df4.loc[wind_df4.VELOCITY <= xMinCarSpeed,:]
        wrongSpeed=wrongSpeed.assign(Reason='velocity too slow')

        #wind_df6 = wind_df5.loc[wind_df5.VELOCITY < xMaxCarSpeed, :]
        wind_df6 = wind_df5.loc[wind_df5.VELOCITY < 1000, :]

        wrongSpeed2 =  wind_df5.loc[wind_df5.VELOCITY >= xMaxCarSpeed, :]
        wrongSpeed2 = wrongSpeed2.assign(Reason='velocity too fast')

        wrongSpeeds = pd.concat([wrongSpeed,wrongSpeed2])
        #notGood = pd.concat([wrongSpeeds,nullCH4])
        notGood = pd.concat([wrongSpeeds])

        # wind_df6 = wind_df6a.loc[wind_df6a.R > .6999, :]

        del (wind_df4)
        wind_df4 = wind_df6.copy().drop_duplicates()
        wind_df5 = wind_df4.loc[wind_df4.CH4.notnull(), :]

        nullCH4 = wind_df4.loc[~wind_df4.CH4.notnull(), :]
        if nullCH4.shape[0] > 0:
            nullCH4 = nullCH4.assign(Reason='CH4 NA')
            removedDF = pd.concat([notGood,nullCH4])
        elif nullCH4.shape[0]==0:
            removedDF = notGood
        wind_df4 = wind_df5.copy()

        def rolling_cor(df, first, sec, window, newname):
            if (window % 2 == 1):
                sidewind = (window - 1) / 2
            else:
                sidewind = window / 2

            length = df.shape[0]
            cor_i = []
            for i in range(length):
                if (((i) < sidewind) or (i >= (length - sidewind))):
                    cor_i.append(0)
                else:
                    xvals = df.loc[(i - sidewind):(i + sidewind + 1), first]
                    yvals = df.loc[(i - sidewind):(i + sidewind + 1), sec]
                    cor_i.append(xvals.corr(yvals))
            df.loc[:, newname] = cor_i
            return (df)

        def rolling_c2h6(df, colname, window, percentile, newname):
            import numpy as np
            if (window % 2 == 1):
                sidewind = (window - 1) / 2
            else:
                sidewind = window / 2

            length = df.shape[0]
            cor_i = []
            for i in range(length):
                if (((i) < sidewind) or (i >= (length - sidewind))):
                    cor_i.append(0)
                else:
                    c2h6vals = df.loc[(i - sidewind):(i + sidewind + 1), colname]
                    cor_i.append(np.percentile(c2h6vals, percentile))
            df.loc[:, newname] = cor_i
            return (df)

        wind_df5 = rolling_cor(wind_df4,'CH4','C2H6',80,'rollingR_8')
        wind_df6 = rolling_cor(wind_df5,'CH4','C2H6',150,'rollingR_15')
        wind_df7 = rolling_cor(wind_df6,'CH4','C2H6',300,'rollingR_30')
        wind_df8 = rolling_cor(wind_df7,'CH4','C2H6',450,'rollingR_45')
        wind_df9 = rolling_cor(wind_df8,'CH4','C2H6',600,'rollingR_60')
        wind_df10 = rolling_c2h6(wind_df9,'C2H6',300,50,'rollingc2h6_30')
        wind_df11 = rolling_c2h6(wind_df10,'C2H6',150,50,'rollingc2h6_15')
        wind_df12 = rolling_c2h6(wind_df11,'C2H6',450,50,'rollingc2h6_45')

        wind_df13 = rolling_c2h6(wind_df12,'CH4',450,50,'rollingch4_45')
        wind_df14 = rolling_c2h6(wind_df13,'CH4',300,50,'rollingch4_30')
        wind_df15 = rolling_c2h6(wind_df14,'CH4',150,50,'rollingch4_15')
        wind_df16 = rolling_c2h6(wind_df15,'CH4',600,50,'rollingch4_60')


        del(wind_df4)
        wind_df4 = wind_df16.copy()
        ## if you want to filter out high temperatures
        #wind_df4 = wind_df5.loc[wind_df5.TEMPC < 95, :].reset_index(drop=True)

        #fLog.write("Usable lines - " + str(wind_df4.shape[0]) + "." + "\n")
        #fLog.close()

        if bFirst:
            wind_df4.to_csv(fnOut, index=False)
            removedDF.to_csv(removeOut,index=False)
        elif not bFirst:
            norm = pd.read_csv(fnOut)
            pd.concat([norm, wind_df4]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(fnOut, index=False)
            removed = pd.read_csv(removeOut)
            pd.concat([removed, removedDF]).sort_values(by='SECONDS').reset_index(drop=True).to_csv(removeOut, index=False)

        #os.remove(fnOutTemp)
        return True
    except ValueError:
        return False
def identify_peaks_amld(xCar, xDate, xDir, xFilename, outDir, processedFileLoc, Engineering, threshold='.1',
                   rthresh = '.7',
                  xTimeThreshold='5.0', minElevated='2', xB='102', basePerc='50'):
    """ input a processed data file, and finds the locations of the elevated readings (observed peaks)
    input:
        xCar: name of the car (to make filename)
        xDate: date of the reading
        xDir: directory where the file is located
        xFilename: name of the file
        outDir: directory to take it
        processedFileLoc
        Engineering: T/F if the processed file was made using an engineering file
        threshold: the proportion above baseline that is marked as elevated (i.e. .1 corresponds to 10% above
        xTimeThreshold: not super sure
        minElevated: # of elevated readings that need to be there to constitute an observed peak
        xB: Number of observations used in background average
        basePerc: percentile used for background average (i.e. 50 is median)
    output:
        saved log file
        saved csv file with identified peaks
        saved info.csv file
        saved json file
    """
    import csv, numpy
    import shutil
    from shapely.geometry import Point
    import pandas as pd
    import geopandas as gpd

    try:
        amld = True
        baseCalc = float(basePerc)
        xABThreshold = float(threshold)
        minElevated = float(minElevated)
        rMin = float(rthresh)
        xDistThreshold = 160.0  # find the maximum CH4 reading of observations within street segments of this grouping distance in meters
        xSDF = 4  # multiplier times standard deviation for floating baseline added to mean

        xB = int(xB)
        xTimeThreshold = float(xTimeThreshold)
        fn = xDir + xFilename  # set processed csv file to read in
        fnOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".csv"
        fnShape = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".shp"
        fnLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-", "") + ".log"
        pkLog = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + "_info.csv"
        jsonOut = outDir + "Peaks" + "_" + xCar + "_" + xDate.replace("-","") + ".geojson"
        infOut = processedFileLoc + xCar + "_" + xDate.replace("-", "") + "_info.csv"

        ### TEST THING
        fn = xDir + xFilename  # set raw text file to read in
        filenames = nameFiles(outDir,processedFileLoc,xCar,xDate,True)
        fnOut = filenames['fnOut']
        fnShape = filenames['fnShape']
        fnLog = filenames['fnLog']
        pkLog = filenames['pkLog']
        jsonOut = filenames['jsonOut']
        infOut = filenames['infOut']

        print(f"{outDir}Peaks_{xCar}_{xDate}_info.csv")
        fLog = open(fnLog, 'w')
        shutil.copy(infOut, pkLog)

        # field column indices for various variables
        if Engineering == True:
            fDate = 0;  fTime = 1; fEpochTime = 2
            fNanoSeconds = 3; fVelocity = 4;  fU = 5
            fV = 6; fW = 7; fBCH4 = 10
            fBCH4 = 8;  fBRSSI = 9; fTCH4 = 10
            TRSSI = 11;PRESS = 12; INLET = 13
            TEMP = 14;  CH4 = 15;H20 = 16
            C2H6 = 17;  R = 18;  C2C1 = 19
            BATT = 20;  POWER = 21; CURR = 22
            SOCPER = 23;fLat = 24; fLon = 25
        elif not Engineering and not amld:
            fDate = 0; fTime = 1; fEpochTime = 2
            fNanoSeconds = 3;fVelocity = 4; fU = 5
            fV = 6;  fW = 7
            fBCH4 = 8; fBRSSI = 9
            fTCH4 = 10;  TRSSI = 11;  PRESS = 12
            INLET = 13;  TEMP = 14; CH4 = 15
            H20 = 16;C2H6 = 17;  R = 18; C2C1 = 19
            BATT = 20; POWER = 21; CURR = 22
            SOCPER = 23; fLat = 24;fLon = 25;
            fUavg = 33; fVavg = 34; fWavg = 35;
            fRavg = 36; fthetavg=37;
            fDist = 38; fOdometer = 39

            # read data in from text file and extract desired fields into a list, padding with 5 minute and hourly average
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,x11,x12,x13,x14,x15,x16,x17,x18 = [[] for _ in range(18)]

            count = -1
            with open(fn, 'r') as f:
                t = csv.reader(f)
                for row in t:
                    woo = row
                    # print(count)
                    if count < 0:
                        count += 1
                        continue
                    elif count >= 0:
                        datet = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                        ## if not engineering
                        epoch = float(row[fEpochTime] + "." + row[fNanoSeconds][0])
                        datetime = row[fDate].replace("-", "") + row[fTime].replace(":", "")
                        x1.append(epoch); x2.append(datetime)
                        if row[fLat] == '':
                            x3.append('')
                        elif row[fLat] != '':
                            x3.append(float(row[fLat]))
                        if row[fLon] == '':
                            x4.append('')
                        elif row[fLon] != '':
                            x4.append(float(row[fLon]))

                        if row[fUavg] == '':
                            x14.append('')
                        elif row[fUavg] != '':
                            x14.append(float(row[fUavg]))
                        if row[fVavg] == '':
                            x15.append('')
                        elif row[fVavg] != '':
                            x15.append(float(row[fVavg]))
                        if row[fWavg] == '':
                            x16.append('')
                        elif row[fWavg] != '':
                            x16.append(float(row[fWavg]))

                        if row[fthetavg] == '':
                            x18.append('')
                        elif row[fthetavg] != '':
                            x18.append(float(row[fthetavg]))
                        if row[fRavg] == '':
                            x17.append('')
                        elif row[fRavg] != '':
                            x17.append(float(row[fRavg]))

                        x5.append(float(row[fBCH4]))
                        x6.append(float(row[fTCH4]))
                        x7.append(0.0)
                        x8.append(0.0)
                        x9.append(row[fOdometer])
                        x11.append(float(row[C2H6]))
                        x12.append(float(row[C2C1]))
                        x13.append(float(row[R]))
                        count += 1
            print(f"Number of observations processed:{count}")

        # convert lists to numpy arrays
        tempFile = pd.read_csv(fn)
        colnames = tempFile.columns

        #aEpochTime = numpy.array(x1)
        #aDateTime = numpy.array(x2)
        #aLat = numpy.array(x3)
        #aLon = numpy.array(x4)
        #aCH4 = numpy.array(x5)
        #aTCH4 = numpy.array(x6)
        #aMean = numpy.array(x7)
        #aMeanC2H6 = numpy.array(x7)
        #aThreshold = numpy.array(x8)
        #aOdom = numpy.array(x9)
        # aC2H6 = numpy.array(x11)
        # aC2C1 = numpy.array(x12)
        # aR = numpy.array(x13)
        # aUavg = numpy.array(x14)
        # aVavg = numpy.array(x15)
        # aWavg = numpy.array(x16)
        # aRavg = numpy.array(x17)
        # aThavg = numpy.array(x18)

        aEpochTime = numpy.array(tempFile.iloc[:,colnames.get_loc('nearest10hz')])
        aDateTime = numpy.array(tempFile.apply(lambda x: x.DATE.replace('-','') + x.TIME.replace(':',''),axis=1))
        aLat = numpy.array(tempFile.iloc[:,colnames.get_loc('LAT')])
        aLon = numpy.array(tempFile.iloc[:,colnames.get_loc('LONG')])
        aCH4 = numpy.array(tempFile.iloc[:,colnames.get_loc('CH4')])
        aTCH4 = numpy.array(tempFile.iloc[:,colnames.get_loc('CH4')])
        aMean = numpy.zeros(len(aEpochTime))
        aCH4Mean_true = numpy.zeros(len(aEpochTime))
        aCH4STD= numpy.zeros(len(aEpochTime))
        aCH4Max= numpy.zeros(len(aEpochTime))
        aCH4Min= numpy.zeros(len(aEpochTime))
        aCH4Median= numpy.zeros(len(aEpochTime))



        aMeanC2H6 = numpy.zeros(len(aEpochTime))
        aThreshold = numpy.zeros(len(aEpochTime))
        aOdom = numpy.array(tempFile.apply(lambda x: x.VELOCITY*.1,axis=1).cumsum())
        aC2H6 = numpy.array(tempFile.iloc[:,colnames.get_loc('C2H6')])
        aC2C1 = numpy.array(tempFile.iloc[:,colnames.get_loc('C1C2')])
        aR = numpy.array(tempFile.iloc[:,colnames.get_loc('R')])
        aBearingCCWE = numpy.array(tempFile.iloc[:,colnames.get_loc('Bearing_ccwe')])
        aBearingCWN = numpy.array(tempFile.iloc[:,colnames.get_loc('Bearing_cwn')])

        arolling8= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingR_8')])
        arolling15= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingR_15')])
        arolling30= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingR_30')])
        arolling45= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingR_45')])
        arolling60= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingR_60')])

        arollingc2h6_15= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingc2h6_15')])
        arollingc2h6_30= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingc2h6_30')])
        arollingc2h6_45= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingc2h6_45')])

        arollingch4_60= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingch4_60')])
        arollingch4_45= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingch4_45')])
        arollingch4_30= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingch4_30')])
        arollingch4_15= numpy.array(tempFile.iloc[:,colnames.get_loc('rollingch4_15')])


        aWS_cor = numpy.array(tempFile.iloc[:,colnames.get_loc('airmar_ws')])
        aWD_CCWE_cor = numpy.array(tempFile.iloc[:,colnames.get_loc('airmar_wd_cor_ccwe')])
        aWD_CWN_cor = numpy.array(tempFile.iloc[:,colnames.get_loc('airmar_wd_cor_cwn')])

        xLatMean = numpy.mean(aLat)
        xLonMean = numpy.mean(aLon)
        #xCH4Mean = numpy.mean(aCH4)
        #xC2H6Mean = numpy.mean(aC2H6)
        #xC2C1Mean = numpy.mean(aC2C1)

        fLog.write("Day CH4_mean = " + str(numpy.mean(aCH4)) +
                   ", Day CH4 SD = " + str(numpy.std(aCH4)) + "\n")
        fLog.write("Day C2H6 Mean = " + str(numpy.mean(aC2H6)) +
                   ", Day C2H6 SD = " + str(numpy.std(aC2H6)) + "\n")
        fLog.write("Center lon/lat = " + str(xLonMean) + ", " + str(xLatMean) + "\n")

        lstCH4_AB = []
        count = tempFile.shape[0]
        # generate list of the index for observations that were above the threshold
        for i in range(0, count - 2):
            if ((count - 2) > xB):
                topBound = min((i + xB), (count - 2))
                botBound = max((i - xB), 0)

                for t in range(min((i + xB), (count - 2)), i, -1):
                    if aEpochTime[t] < (aEpochTime[i] + (xB / 2)):
                        topBound = t
                        break
                for b in range(max((i - xB), 0), i):
                    if aEpochTime[b] > (aEpochTime[i] - (xB / 2)):
                        botBound = b
                        break

                xCH4Mean = numpy.percentile(aCH4[botBound:topBound], baseCalc)
                xC2H6Mean = numpy.percentile(aC2H6[botBound:topBound], baseCalc)
                xCH4Mean_true = numpy.mean(aCH4[botBound:topBound])
                xCH4STD = numpy.std(aCH4[botBound:topBound])
                xCH4Min = numpy.min(aCH4[botBound:topBound])
                xCH4Max = numpy.max(aCH4[botBound:topBound])
                xCH4Median = numpy.percentile(aCH4[botBound:topBound],50)


            # xCH4SD = numpy.std(aCH4[botBound:topBound])
            else:
                xCH4Mean = numpy.percentile(aCH4[0:(count - 2)], baseCalc)
                xC2H6Mean = numpy.percentile(aC2H6[0:(count - 2)], baseCalc)
                xCH4Mean_true = numpy.mean(aCH4[0:(count - 2)])
                xCH4STD = numpy.std(aCH4[0:(count - 2)])
                xCH4Min = numpy.min(aCH4[0:(count - 2)])
                xCH4Max = numpy.max(aCH4[0:(count - 2)])
                xCH4Median = numpy.percentile(aCH4[0:(count - 2)],50)


                # xCH4SD = numpy.std(aCH4[0:(count-2)])
            xThreshold = xCH4Mean + (xCH4Mean * xABThreshold)
            xThreshold_c2h6 = xC2H6Mean + (xC2H6Mean * xABThreshold)

            if (aCH4[i] > xThreshold and aR[i]>rMin):
            #if (aCH4[i] > xThreshold):
                lstCH4_AB.append(i)
                aMean[i] = xCH4Mean
                aMeanC2H6[i] = xC2H6Mean
                aThreshold[i] = xThreshold
                aCH4STD[i] = xCH4STD
                aCH4Max[i] = xCH4Max
                aCH4Min[i] = xCH4Min
                aCH4Mean_true[i] = xCH4Mean_true
                aCH4Median[i] = xCH4Median

        # now group the above baseline threshold observations into groups based on distance threshold
        lstCH4_ABP = []
        xDistPeak = 0.0
        xCH4Peak = 0.0
        xTime = 0.0
        cntPeak = 0
        cnt = 0
        sID = ""
        sPeriod5Min = ""
        prevIndex = 0
        for i in lstCH4_AB:
            if (cnt == 0):
                xLon1 = aLon[i]
                xLat1 = aLat[i]
                xOdom = aOdom[i]
            else:
                # calculate distance between points
                xDist = haversine(xLat1, xLon1, aLat[i], aLon[i])
                xDistPeak += xDist
                xCH4Peak += (xDist * (aCH4[i] - aMean[i]))
                xLon1 = aLon[i]
                xLat1 = aLat[i]
                xOdom = aOdom[i]
                if (sID == ""):
                    xTime = aEpochTime[i]
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                if ((aEpochTime[i] - aEpochTime[prevIndex]) > xTimeThreshold):  # initial start of a observed peak
                    cntPeak += 1
                    xTime = aEpochTime[i]
                    xDistPeak = 0.0
                    xCH4Peak = 0.0
                    sID = str(xCar) + "_" + str(xTime)
                    sPeriod5Min = str(int((aEpochTime[i] - 1350000000) / (30 * 1)))  # 30 sec
                    # print str(i) +", " + str(xDist) + "," + str(cntPeak) +"," + str(xDistPeak)
                #lstCH4_ABP.append(
                #    [sID, xTime, aEpochTime[i], aDateTime[i], aCH4[i], aLon[i], aLat[i], aMean[i], aThreshold[i],
                #     xDistPeak, xCH4Peak, aTCH4[i],aC2H6[i],aC2C1[i],aR[i],aMeanC2H6[i], sPeriod5Min, xOdom,
                #     aUavg[i],aVavg[i],aWavg[i],aRavg[i],aThavg[i]])
                lstCH4_ABP.append(
                    [sID, xTime, aEpochTime[i], aDateTime[i], aCH4[i], aLon[i], aLat[i], aMean[i],aCH4Mean_true[i],aCH4STD[i],
                     aCH4Max[i],aCH4Min[i],aCH4Median[i], aThreshold[i],
                     xDistPeak, xCH4Peak, aTCH4[i],aC2H6[i],aC2C1[i],aR[i],aMeanC2H6[i], sPeriod5Min, xOdom,
                    aWD_CCWE_cor[i],aWD_CWN_cor[i],aWS_cor[i],aBearingCCWE[i],aBearingCWN[i],arolling8[i],
                     arolling15[i],arolling30[i],arolling60[i],arollingc2h6_15[i],arollingc2h6_30[i],arollingc2h6_45[i],
                     arollingch4_15[i],arollingch4_30[i],arollingch4_45[i],arollingch4_60[i]
                     ])

            cnt += 1
            prevIndex = i

        # Finding peak_id larger than 160.0 m
        tmpsidlist = []
        for r in lstCH4_ABP:
            if (float(r[9]) > 160.0) and (r[0] not in tmpsidlist):
                tmpsidlist.append(r[0])
        cntPeak -= len(tmpsidlist)

        fLog.write("Number of peaks found: " + str(cntPeak) + "\n")
        print(f"{xCar} \t {xDate} \t {xFilename} \t {count} \t {len(lstCH4_ABP)}")

        # write out the observed peaks to a csv to be read into a GIS
        fOut = open(fnOut, 'w')
        # s = "PEAK_NUM,EPOCHSTART,EPOCH,DATETIME,CH4,LON,LAT,CH4_BASELINE,CH4_THRESHOLD,PEAK_DIST_M,PEAK_CH4,TCH4,PERIOD5MIN\n"
        s = "OP_NUM,OP_EPOCHSTART,OB_EPOCH,OB_DATETIME,OB_CH4,OB_LON,OB_LAT,OB_CH4_BASELINE,OB_CH4_MEAN,OB_CH4_STD,OB_CH4_MAX,OB_CH4_MIN,OB_CH4_MED," \
            "OB_CH4_THRESHOLD,OP_PEAK_DIST_M,OP_PEAK_CH4,OB_TCH4,OB_C2H6," \
            "OB_C2C1,OB_R,OB_C2H6_BASELINE,OB_PERIOD5MIN,ODOMETER,OB_WD_CCWE,OB_WD_CWN,OB_WS," \
            "OB_BEARING_CCWE,OB_BEARING_CWN,OB_R_8,OB_R_15,OB_R_30,OB_R_60,OB_C2H6_15,OB_C2H6_30,OB_C2H6_45," \
            "OB_CH4_15,OB_CH4_30,OB_CH4_45,OB_CH4_60\n"


        fOut.write(s)

        truecount = 0
        for r in lstCH4_ABP:
            if r[0] not in tmpsidlist:
                s = ''
                for rr in r:
                    s += str(rr) + ','
                s = s[:-1]
                s += '\n'
                fOut.write(s)
                truecount += 1
        fOut.close()
        fLog.close()

        openFile = pd.read_csv(fnOut)
        if openFile.shape[0] != 0:
            pkDistDf = openFile.copy().groupby('OP_NUM', as_index=False).apply(
                lambda x: max(x.ODOMETER) - min(x.ODOMETER))
            pkDistDf.columns = ['OP_NUM', 'OP_DISTANCE']
            openFile = pd.merge(openFile.copy(), pkDistDf)
            tempCount = openFile.groupby('OP_NUM', as_index=False).OP_EPOCHSTART.count().rename(
                columns={'OP_EPOCHSTART': 'Frequency'})
            tempCount = tempCount.loc[tempCount.Frequency >= minElevated, :]
            if tempCount.shape[0] == 0:
                print(f"No Observed Peaks with enough Elevated Readings Found in the file: {xFilename}")
                tempCount.to_csv(fnOut) ## added to deal with issue where it wasn't being filtered out
            elif tempCount.shape[0] != 0:
                oFile = pd.merge(openFile, tempCount, on=['OP_NUM'])
                openFile = oFile.copy()
                del (oFile)
                openFile["minElevated"] = openFile.apply(lambda x: int(minElevated), axis=1)
                openFile['OB_CH4_AB'] = openFile.loc[:, 'OB_CH4'].sub(openFile.loc[:, 'OB_CH4_BASELINE'], axis=0)
                openFile['OB_C2H6_AB'] = openFile.loc[:, 'OB_C2H6'].sub(openFile.loc[:, 'OB_C2H6_BASELINE'],axis=0)
                openFile.to_csv(fnOut, index=False)


                fileWt = weighted_loc(openFile, 'OB_LAT', 'OB_LON', 'OP_NUM', 'OB_CH4_AB',).loc[:, :].rename(
                    columns={'OB_LAT': 'pk_LAT', 'OB_LON': 'pk_LON'}).reset_index(drop=True)
                geometry_temp = [Point(lon, lat) for lon, lat in zip(fileWt['pk_LON'], fileWt['pk_LAT'])]
                crs = 'EPSG:4326'
                # geometry is the point of the lat/lon
                # gdf_buff = gpd.GeoDataFrame(datFram, crs=crs, geometry=geometry_temp)

                ## BUFFER AROUND EACH 'OP_NUM' WITH BUFFER DISTANCE
                gdf_buff = gpd.GeoDataFrame(fileWt, crs=crs, geometry=geometry_temp)
                # gdf_buff = makeGPD(datFram,'LON','LAT')

                ##maybe this is the issue?
                #gdf_buff = gdf_buff.to_crs(epsg=32610)
                #gdf_buff['geometry'] = gdf_buff.loc[:, 'geometry'].buffer(30)
                try:
                    gdf_buff.to_file(jsonOut, driver="GeoJSON")
                    #gdf_buff.to_file('testthing.geojson', driver="GeoJSON")
                except:
                    print("Error Saving JSON File")
        elif openFile.shape[0] == 0:
            print(f"No Observed Peaks Found in the file:{xFilename}")
    except ValueError:
        print("Error in Identify Peaks")
        return False

