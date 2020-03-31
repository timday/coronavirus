#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import datetime
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# GDP per head data from Worldometers 2017.
# https://www.worldometers.info/gdp/gdp-per-capita/
# CSV a bit of a mess with trailing spaces.

def GDPvalue(x):
    if x=='N.A.':
        return None
    else:
        print x
        assert x[0]=='$'
        return float(x[1:].replace(',',''))

gdpRewrites={
    'South Korea'             :'Korea, South',
    'United States'           :'US',
    'DR Congo'                :'Congo (Kinshasa)',
    'Congo (Brazzaville)'     :'Congo (Brazzaville)',
    'Czech Republic (Czechia)':'Czechia',
    }
    
def getGDPPerHead(ppp):   # ppp=True for Purchasing Power Parity

    gdp={}

    csvfile=open('data/gdp.csv','rb')
    reader=csv.reader(csvfile)
    timeseries={}
    for row in reader:
        
        where=row[1].strip()

        if where in gdpRewrites:
            where=gdpRewrites[where]
        
        if ppp:
            v=GDPvalue(row[2].strip())
        else:
            v=GDPvalue(row[3].strip())

        if v!=None:
            gdp[where]=v

    return gdp

def JHUvalue(x):
    if x=='':
        return 0
    else:
        return int(x)

def getJHUData(deaths):

    csvfile=open({False:'data/time_series_covid19_confirmed_global.csv',True:'data/time_series_covid19_deaths_global.csv'}[deaths],'rb')
    reader=csv.reader(csvfile)
    timeseries={}
    firstRow=True
    for row in reader:
        
        if firstRow:
            firstRow=False
            continue

        where=row[1]
        if not where in timeseries:
            timeseries[where]=np.zeros(len(row[4:]))
        timeseries[where]+=np.array(map(lambda x: JHUvalue(x),row[4:]),dtype=np.float64)

    return timeseries

gdp=getGDPPerHead(True)
timeseries=getJHUData(True)

window=7

growth={k:100.0*((timeseries[k][-1]/timeseries[k][-1-window])**(1.0/window)-1.0) for k in timeseries.keys() if timeseries[k][-1-window]>0.0}

#print 'Odd growth:'
#for k in growth.keys():
#    if growth[k]<=0.0:
#        print k,growth[k],timeseries[k]

plottable=[k for k in growth.keys() if k in gdp and growth[k]>0.0]

x=[gdp[k] for k in plottable]
y=[growth[k] for k in plottable]
        
plt.scatter(x,y)

plt.show()
