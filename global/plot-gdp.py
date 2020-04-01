#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: China a big influence.  Some other statistic like max growth?

import csv
import datetime
import distutils.dir_util
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def cov(x, y, w):
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def GDPvalue(x):
    if x=='N.A.':
        return None
    else:
        assert x[0]=='$'
        return float(x[1:].replace(',',''))

gdpRewrites={
    'South Korea'             :'Korea, South',
    'United States'           :'US',
    'DR Congo'                :'Congo (Kinshasa)',
    'Congo (Brazzaville)'     :'Congo (Brazzaville)',
    'Czech Republic (Czechia)':'Czechia',
    }

# GDP per head data from Worldometers 2017.
# https://www.worldometers.info/gdp/gdp-per-capita/
# CSV a bit of a mess with trailing spaces.
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

def popvalue(x):
    return float(x.replace(',',''))

# Get population
# Data from https://www.worldometers.info/world-population/population-by-country/
def getPopulation():

    population={}

    csvfile=open('data/population.csv','rb')
    reader=csv.reader(csvfile)
    timeseries={}
    for row in reader:
        
        where=row[1].strip()

        if where in gdpRewrites:
            where=gdpRewrites[where]
        
        v=popvalue(row[2].strip())

        if v!=None:
            population[where]=v

    return population

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
population=getPopulation()
timeseries=getJHUData(False)

print timeseries['China']

window=7

growth={k:100.0*((timeseries[k][-1]/timeseries[k][-1-window])**(1.0/window)-1.0) for k in timeseries.keys() if timeseries[k][-1-window]>0.0}

for k in growth.keys():
    if k in gdp:
        if not k in population:
            print 'No population for',k

#print 'Odd growth:'
#for k in growth.keys():
#    if growth[k]<=0.0:
#        print k,growth[k],timeseries[k]

plottable=[k for k in growth.keys() if k in gdp and k in population and growth[k]>0.0] # and k!='China'

fig=plt.figure(figsize=(8,6))

x=np.array([np.log10(gdp[k]) for k in plottable])
y=np.array([growth[k] for k in plottable])
w=np.array([population[k] for k in plottable])
s=np.sqrt(w/1e4)
plt.scatter(x,y,s=s,color='tab:blue')

texts=sorted(zip(s,x,y,plottable),key=lambda x: x[0],reverse=True)[:40]
for txt in texts:
    plt.text(txt[1],txt[2]+0.0025*txt[0],txt[3],verticalalignment='bottom',horizontalalignment='center')

# Unweighted regression line
gradient,intercept,r_value,p_value,std_err=scipy.stats.linregress(x,y)
print 'Unweighted',r_value,gradient,intercept
rx=np.linspace(min(x),max(x),100)
ry=gradient*rx+intercept
plt.plot(rx,ry,color='tab:orange',label='Linear regression (unweighted) r={:.3f}'.format(r_value))

# Weighted regression line
coef=np.polyfit(x,y,1,w=w)
print 'Weighted',coef[0],coef[1]
ry=coef[1]+coef[0]*rx  # Highest power first
rw=corr(x,y,w)
plt.plot(rx,ry,color='tab:red',label='Linear regression (weighted) r={:.3f}'.format(rw))

coef=np.polyfit(x,y,2,w=w)
print coef
qy=coef[2]+coef[1]*rx+coef[0]*rx**2
plt.plot(rx,qy,color='tab:green',label='Quadratic best fit')

plt.legend(loc='upper right')

distutils.dir_util.mkpath('output')
plt.savefig('output/gdp.png',dpi=96)

plt.show()
