#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import datetime
import matplotlib.dates as mdates
import numpy as np

# csv file starts 2020-01-22, so zero pad, except for China which has a couple of extra days data on main tracker chart
basedate=mdates.date2num(datetime.datetime(2020,1,20))

csvfile=open('data/time_series_19-covid-Confirmed.csv','rb')
reader=csv.reader(csvfile)
timeseries={}
interesting=frozenset(['China','UK','Italy','South Korea','US','Iran','France','Germany','Spain','Japan','Switzerland','Netherlands','Sweden','Norway','Denmark','Belgium','Austria'])
firstRow=True
for row in reader:

    if firstRow:
        firstRow=False
        continue
    
    where=row[1]
    # Keys sometimes change.  Or just shorten them.
    if where=='Republic of Korea' or where=='Korea, South':
        where='South Korea'
    if where=='Iran (Islamic Republic of)':
        where='Iran'
    if where=='United Kingdom':
        where='UK'

    if not where in timeseries:
        if where=='China':
            pad=np.array([278.0,326.0])
        else:
            pad=np.array([0.0,0.0])

    if where in interesting:
        if not where in timeseries:
            timeseries[where]=np.concatenate([pad,np.zeros(len(row[4:]))])
        timeseries[where]+=np.concatenate([np.array([0.0,0.0]),np.array(map(lambda x: int(x),row[4:]),dtype=np.float64)])

    if where!='China':
        if not 'Other' in timeseries:
            timeseries['Other']=np.concatenate([pad,np.zeros(len(row[4:]))])
        timeseries['Other']+=np.concatenate([np.array([0.0,0.0]),np.array(map(lambda x: int(x),row[4:]),dtype=np.float64)])
        
timeseries['Total']=timeseries['China']+timeseries['Other']

timeseriesKeys=['Total','Other','China','Iran','South Korea','Italy','France','Spain','Germany','US','Japan','Netherlands','Switzerland','UK','Sweden','Norway','Belgium','Denmark','Austria']

def clean(a):
    c=np.concatenate(
        [
            np.minimum(a[:-1],a[1:]),
            np.array([a[-1]])
        ]
    )
    if not np.array_equal(a,c):
        print "Cleaned",a,"to",c
    return c

for k in timeseriesKeys:
    timeseries[k]=clean(timeseries[k])

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
