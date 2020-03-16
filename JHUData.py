#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import datetime
import matplotlib.dates as mdates
import numpy as np

basedate=mdates.date2num(datetime.datetime(2020,1,20))

# Data from https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
# Use inspect element on graph to get precise numbers
# Starts 2020-01-20:
#china=np.array([278,326,547,639,916,1979,2737,4409,5970,7678,9658,11221,14341,17187,19693,23680,27409,30553,34075,36778,39790,42306,44327,44699,59832,66292,68347,70446,72364,74139,74546,74999,75472,76922,76938,77152,77660,78065,78498,78824,79251,79826,80026,80151,80271,80422,80573,80652,80699,80735,80757,80921,80932,80945,80977,80003],dtype=np.float64)
other=np.array([  4,  6,  8, 14, 25,  40,  57,  64,  87, 105, 118,  153,  173,  183,  188,  212,  227,  265,  317,  343,  361,  457,  476,  523,  538,  595,  685,  780,  896, 1013, 1095, 1200, 1371, 1677, 2047, 2418, 2755, 3332, 4258, 5300, 6762, 8545,10283,12693,14853,17464,21227,25184,29136,32847,37825,44944,47411,63569,75122,81716],dtype=np.float64)

# Keys keep changing.
# Republic of Korea -> Korea, South
# Iran (Islamic Republic of) -> Iran

# csv file starts 2020-01-22, so zero pad, except for China which has a couple of extra days data
csvfile=open('data/time_series_19-covid-Confirmed.csv','rb')
reader=csv.reader(csvfile)
timeseries={}
interesting=frozenset(['China','UK','Italy','South Korea','US','Iran','France','Germany','Spain','Japan','Switzerland','Netherlands','Sweden','Norway','Denmark','Belgium','Austria'])
for row in reader:
    where=row[1]
    if where=='Republic of Korea' or where=='Korea, South':
        where='South Korea'
    if where=='Iran (Islamic Republic of)':
        where='Iran'
    if where=='United Kingdom':
        where='UK'

    if where in interesting:
        if not where in timeseries:
            if where=='China':
                pad=np.array([278.0,326.0])
            else:
                pad=np.array([0.0,0.0])
            timeseries[where]=np.concatenate([pad,np.zeros(len(row[4:]))])
        timeseries[where]+=np.concatenate([np.array([0.0,0.0]),np.array(map(lambda x: int(x),row[4:]),dtype=np.float64)])

#timeseries['China']=china
timeseries['Other']=other
timeseries['Total']=timeseries['China']+other

timeseriesKeys=['Total','Other','China','Iran','South Korea','Italy','France','Spain','Germany','US','Japan','Netherlands','Switzerland','UK','Sweden','Norway','Belgium','Denmark','Austria']
for k in timeseriesKeys:
    assert len(timeseries[k])==len(timeseries['China'])

timeseriesKeys.sort(key=lambda k: timeseries[k][-1],reverse=True)

descriptions=dict(zip(timeseriesKeys,timeseriesKeys))
descriptions['China']='Mainland China'
descriptions['Other']='Global ex-China'
descriptions['Total']='Global Total'

populations={
    'China'      :1.4e9,
    'Other'      :7.7e9-1.4e9,
    'Total'      :7.7e9,
    'UK'         :6.6e7,
    'Italy'      :6e7,
    'Netherlands':1.7e7,
    'France'     :6.7e7,
    'Germany'    :8.3e7,
    'Spain'      :4.7e7,
    'Switzerland':8.6e6,
    'US'         :3.3e8,
    'South Korea':5.1e7,
    'Japan'      :1.3e8,
    'Iran'       :8.1e7,
    'Sweden'     :1e7,
    'Norway'     :5e6,
    'Denmark'    :5.6e6,
    'Belgium'    :1.1e7,
    'Austria'    :8.8e6
    }

def rgb(r,g,b):
    return (r/255.0,g/255.0,b/255.0)

# Tableau20 looks useful (bookmarked goodstuff).  Unused 197,176,213.
colors={
    'China'      :rgb(214, 39, 40),  
    'Other'      :rgb(  0,  0,  0),
    'Total'      :rgb(127,127,127),  # Or 199x3

    'US'         :rgb( 31,119,180),

    'UK'         :rgb( 23,190,207),
    'France'     :rgb(140, 86, 75),
    'Germany'    :rgb(196,156,148),
    'Spain'      :rgb(152,223,138),
    'Italy'      :rgb( 44,160, 44),

    'Sweden'     :rgb(158,218,229),
    'Norway'     :rgb(174,199,232),
    'Denmark'    :rgb(219,219,141),
    'Netherlands':rgb(188,189, 34),
    'Belgium'    :rgb(247,182,210),
    'Austria'    :rgb(225,187,120),
    'Switzerland':rgb(255,152,150),

    'South Korea':rgb(255,127, 14),
    'Japan'      :rgb(227,119,194),
    'Iran'       :rgb(148,103,189)
    }
