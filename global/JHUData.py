#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import csv
import datetime
import matplotlib.dates as mdates
import numpy as np

pyBasedate=datetime.datetime(2020,1,22)  # csv file starts 2020-01-22
basedate=mdates.date2num(pyBasedate)

descriptions={
    'Total'      :'Global Total',
    'China'      :'Mainland China',
    'China:Hubei':'China (Hubei province)',
    'China:Other':'China (Other provinces)',
    'UK'         :'UK',
    'Italy'      :'Italy',
    'Netherlands':'Netherlands',
    'France'     :'France'     ,
    'Germany'    :'Germany'    ,
    'Spain'      :'Spain'      ,
    'Portugal'   :'Portugal'   ,
    'Switzerland':'Switzerland',
    'US'         :'US'         ,
    'Canada'     :'Canada'     ,
    'South Korea':'South Korea',
    'Japan'      :'Japan'      ,
    'Iran'       :'Iran'       ,
    'Sweden'     :'Sweden'     ,
    'Norway'     :'Norway'     ,
    'Denmark'    :'Denmark'    ,
    'Belgium'    :'Belgium'    ,
    'Austria'    :'Austria'    ,
    'Malaysia'   :'Malaysia'   ,
    'Brazil'     :'Brazil'     ,
    'Australia'  :'Australia'  ,
    'Israel'     :'Israel'     ,
    'Turkey'     :'Turkey'     ,
    'Czechia'    :'Czechia'    ,
    'Ireland'    :'Ireland'    ,
    'Singapore'  :'Singapore'  ,
    'Russia'     :'Russia'     ,
    'Chile'      :'Chile'      ,
    'India'      :'India'      ,
    'Peru'       :'Peru'       ,
}

news={
    'China':[
        ((2020,1,22),'Wuhan Lockdown'),  # Actually, on the 20th
        ((2020,2,13),'Hubei Lockdown'),
        ((2020,2,20),'Hubei Lockdown inc. schools')
    ],
    'China:Hubei':[
        ((2020,1,20),'Wuhan Lockdown'),
        ((2020,2,13),'Hubei Lockdown'),
        ((2020,2,20),'Hubei Lockdown inc. schools'),
    ],
    'China:Other':[],
    'Total':[],
    'UK':[
        ((2020,3,16),'Drastic action'),
        ((2020,3,24),'National Lockdown')
    ],
    'Italy':[
        ((2020,3, 9),'National lockdown'),
        ((2020,3,23),'Tighten restrictions')
    ],
    'Netherlands':[
        ((2020,3,16),'Closures')
    ],
    'France':[
        ((2020,3,17),'National lockdown')
    ],
    'Germany':[
        ((2020,3,16),'Closures')
    ],
    'Spain':[
        ((2020,3,16),'National lockdown'),
        ((2020,3,30),'Tighten restrictions')
    ],
    'Portugal':[
        ((2020,3,19),'State of Emergency')
    ],
    'Switzerland':[
        ((2020,3,16),'National lockdown')
    ],
    'US':[
        ((2020,3,16),'Limited closures')
    ],
    'Canada':[
        ((2020,3,17),'Emergency')
    ],
    'South Korea':[
        ((2020,2,21),'Emergency measures')
    ],
    'Japan':[
        ((2020,2,25),'Basic Policies'),
        ((2020,4,16),'State of Emergency')
    ],
    'Iran':[
        ((2020,3,13),'Closures')
    ],
    'Sweden':[
        ((2020,3,11),'Large gathering ban')
    ],
    'Norway':[
        ((2020,3,12),'National lockdown')
    ],
    'Denmark':[
        ((2020,3,13),'National lockdown')
    ],
    'Belgium':[
        ((2020,3,18),'National lockdown')
    ],
    'Austria':[
        ((2020,3,16),'National lockdown')
    ],
    'Malaysia':[
        ((2020,3,16),'National lockdown')
    ],
    'Brazil':[
        ((2020,3,20),'State of emergency')
    ],
    'Australia':[
        # Borders closed 2020,3,20.
    ],
    'Israel':[
        ((2020,03,17),'Phone tracking')
        # Lockdown anticipated
    ],
    'Turkey':[
        ((2020,03,16),'Limited closures')
    ],
    'Czechia':[
        ((2020,03,16),'National lockdown')
    ],
    'Ireland':[
        ((2020,03,16),'Limited closures')
    ],
    'Singapore':[
    ],
    'Russia':[
        ((2020,03,30),'Stay-at-home regimes')
    ],
    'Chile':[
        ((2020,03,26),'School closures, Santiago lockdown')
    ],
    'India':[
        ((2020,03,24),'National lockdown')
    ],
    'Peru':[
        ((2020,03,16),'National lockdown')
    ]
}

populations={
    'China'      :1.4e9,
    'China:Hubei':5.8e7,
    'China:Other':1.4e9-5.8e7,
    'Total'      :7.7e9,
    'UK'         :6.6e7,
    'Italy'      :6e7,
    'Netherlands':1.7e7,
    'France'     :6.7e7,
    'Germany'    :8.3e7,
    'Spain'      :4.7e7,
    'Portugal'   :1.0e7,
    'Switzerland':8.6e6,
    'US'         :3.3e8,
    'Canada'     :3.8e7,
    'South Korea':5.1e7,
    'Japan'      :1.3e8,
    'Iran'       :8.1e7,
    'Sweden'     :1e7,
    'Norway'     :5e6,
    'Denmark'    :5.6e6,
    'Belgium'    :1.1e7,
    'Austria'    :8.8e6,
    'Malaysia'   :3.2e7,
    'Brazil'     :2.1e8,
    'Australia'  :2.5e7,
    'Israel'     :8.6e6,
    'Turkey'     :8.1e7,
    'Czechia'    :1.1e7,
    'Ireland'    :6.6e6,
    'Russia'     :1.45e8,
    'Chile'      :1.8e7,
    'Singapore'  :5.6e6,
    'India'      :1.4e9,
    'Peru'       :3.2e7,
    }

def rgb(r,g,b):
    return (r/255.0,g/255.0,b/255.0)

# Tableau20 looks useful (bookmarked goodstuff).  Unused.  199x3?
colors={
    'China'      :rgb(214, 39, 40),
    'China:Hubei':rgb(214, 39, 40),
    'China:Other':rgb(214, 39, 40),
    
    'Total'      :rgb(  0,  0,  0),

    'US'         :rgb( 31,119,180),
    'Canada'     :rgb( 31,119,180),  # Same as US

    'UK'         :rgb( 23,190,207),
    'Australia'  :rgb( 23,190,207),  # Same as UK
    
    'France'     :rgb(140, 86, 75),
    'Ireland'    :rgb(140, 86, 75),  # Same as France
    
    'Germany'    :rgb(196,156,148),
    'Chile'      :rgb(196,156,148),  # Same as Germany
    
    'Spain'      :rgb(152,223,138),
    'Portugal'   :rgb(152,223,138),  # Same as Spain
    
    'Italy'      :rgb( 44,160, 44),
    'Brazil'     :rgb( 44,160, 44),  # Same as Italy

    'Sweden'     :rgb(158,218,229),
    'Norway'     :rgb(174,199,232),
    'Denmark'    :rgb(219,219,141),
    'Netherlands':rgb(188,189, 34),

    'Belgium'    :rgb(247,182,210),
    'Peru'       :rgb(247,182,210), # Same as Belgium

    'Austria'    :rgb(225,187,120),
    'Czechia'    :rgb(225,187,120), # Same as Austria

    'Switzerland':rgb(197,176,213),
    'Israel'     :rgb(197,176,213), # Same as Switzerland

    'South Korea':rgb(255,127, 14),
    'India'      :rgb(255,127, 14), # Same as South Korea

    'Japan'      :rgb(227,119,194),

    'Iran'       :rgb(148,103,189),
    'Turkey'     :rgb(148,103,189), # Same as Iran

    'Malaysia'   :rgb(255,152,150),
    'Singapore'  :rgb(255,152,150), # Same as Malaysia

    'Russia'     :rgb(127,127,127)
    }

widthScale=defaultdict(lambda: 1.0)
widthScale['China:Other']=0.5
widthScale['Portugal']=0.5
widthScale['Canada']=0.5
widthScale['Brazil']=0.5
widthScale['Australia']=0.5
widthScale['Israel']=0.5
widthScale['Turkey']=0.5
widthScale['Czechia']=0.5
widthScale['Ireland']=0.5
widthScale['Singapore']=0.5
widthScale['Chile']=0.5
widthScale['India']=0.5
widthScale['Peru']=0.5

# Recursive in case error spans more than one day
def clean(a,where):
    c=np.concatenate(
        [
            np.minimum(a[:-1],a[1:]),
            np.array([a[-1]])
        ]
    )
    if not np.array_equal(a,c):
        print 'Cleaned',where,':',a,'to',c
        return clean(c,where)
    else:
        return c

def value(x):
    if x=='':
        return 0
    else:
        return int(x)

def getJHUData(all,splitChina):

    results=[]

    # Actual list
    timeseriesKeys=['Total','Iran','South Korea','Italy','France','Spain','Portugal','Germany','US','Canada','Japan','Netherlands','Switzerland','UK','Sweden','Norway','Belgium','Denmark','Austria','Malaysia','Brazil','Australia','Israel','Turkey','Czechia','Ireland','Singapore','Russia','Chile','India','Peru']
    if splitChina:
        timeseriesKeys.append('China:Hubei')
        timeseriesKeys.append('China:Other')
    else:
        timeseriesKeys.append('China')            

    # Names from the CSV file
    interesting=frozenset(['China','China:Hubei','China:Other','UK','Italy','South Korea','US','Canada','Iran','France','Germany','Spain','Portugal','Japan','Switzerland','Netherlands','Sweden','Norway','Denmark','Belgium','Austria','Malaysia','Brazil','Australia','Israel','Turkey','Czechia','Ireland','Singapore','Russia','Chile','India','Peru'])

    for what in {False:range(1),True:range(2)}[all]:

        csvfile=open(['data/time_series_covid19_confirmed_global.csv','data/time_series_covid19_deaths_global.csv'][what],'rb')
        reader=csv.reader(csvfile)
        timeseries={}
        firstRow=True
        for row in reader:
        
            if firstRow:
                firstRow=False
                continue

            #if row[0]=='From Diamond Princess':
            #    continue
            
            where=row[1]
            # Keys sometimes change.  Or just shorten them.
            if where=='Republic of Korea' or where=='Korea, South':
                where='South Korea'
            if where=='Iran (Islamic Republic of)':
                where='Iran'
            if where=='United Kingdom':
                where='UK'

            if splitChina and where=='China':
                if row[0]=='Hubei':
                    where='China:Hubei'
                else:
                    where='China:Other'

            if where in interesting:
                if not where in timeseries:
                    timeseries[where]=np.zeros(len(row[4:]))
                timeseries[where]+=np.array(map(lambda x: value(x),row[4:]),dtype=np.float64)

            if not 'Total' in timeseries:
                timeseries['Total']=np.zeros(len(row[4:]))
            timeseries['Total']+=np.array(map(lambda x: value(x),row[4:]),dtype=np.float64)
            
        #for k in timeseriesKeys:
        #    timeseries[k]=clean(timeseries[k],k)

        assert len(set([len(t) for t in timeseries.values()]))==1 # Should all be the same length

        # Sort on current case count, highest first
        if what==0:
            timeseriesKeys.sort(key=lambda k: timeseries[k][-1],reverse=True)

        results.append(timeseries)

    if not all:
        results=results[0]
    
    return timeseriesKeys,results
    
