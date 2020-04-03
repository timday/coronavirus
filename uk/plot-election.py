#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Constituency to UTLA from http://geoportal.statistics.gov.uk/datasets/b7435625a4d442bcb331d610f16aacde_0

# 2019 Election data from https://commonslibrary.parliament.uk/research-briefings/cbp-8749/ (the accompanying results by constituency csv)
# 2017 Election data from https://commonslibrary.parliament.uk/research-briefings/cbp-7979/

# Look at votes for con, lab, non-mainstream (including libdem!?), green?, brexit?

import csv
from collections import defaultdict
import distutils.dir_util
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import UKCovid19Data

def cov(x, y, w):
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def value(x):
    return float(x)

# Accumulate votes into the areas given (wards and constituencies).  No translation of areas.
def getRawVotes():

    votes={}

    csvfile=open('data/HoC-2019GE-results-by-constituency.csv')
    reader=csv.reader(csvfile)
    firstRow=True
    for row in reader:
        if firstRow:
            assert row[0]=='ons_id'
            assert row[15]=='valid_votes'
            assert row[18]=='con'
            assert row[19]=='lab'
            assert row[20]=='ld'
            assert row[21]=='brexit'
            assert row[22]=='green'
            assert row[23]=='snp'
            firstRow=False
            continue

        ward=row[0]
        pcon=row[1]

        for where in set([ward,pcon]):   # Avoid double counting if any duplicates

            if not where in votes:
                votes[where]=defaultdict(float)
        
            votes[where]['Total']+=value(row[15])
            votes[where]['Con']+=value(row[18])
            votes[where]['Lab']+=value(row[19])
            votes[where]['LibDem']+=value(row[20])
            votes[where]['Brexit']+=value(row[21])
            votes[where]['Green']+=value(row[22])

    return votes

# For interesting areas, want to collect all their sub-regions
def getAreas(interesting):

    areas=defaultdict(set)
    
    csvfile=open('data/Ward_to_Westminster_Parliamentary_Constituency_to_Local_Authority_District_to_Upper_Tier_Local_Authority_December_2019_Lookup_in_the_United_Kingdom.csv')
    reader=csv.reader(csvfile)
    firstRow=True
    for row in reader:
        if firstRow:
            assert row[0][-6:]=='WD19CD'
            assert row[2]=='PCON19CD'
            assert row[4]=='LAD19CD'
            assert row[6]=='UTLA19CD'
            firstRow=False
            continue

        ward=row[0]
        pcon=row[2]
        lad=row[4]
        utla=row[6]

        if utla in interesting:
            areas[utla].add(utla)
            areas[utla].add(lad)
            areas[utla].add(pcon)
            areas[utla].add(ward)

        if lad in interesting:
            areas[lad].add(lad)
            areas[lad].add(pcon)
            areas[lad].add(ward)

        if pcon in interesting:
            areas[pcon].add(pcon)
            areas[pcon].add(ward)

        if ward in interesting:
            areas[ward].add(ward)

    return dict(areas)

window=7

timeseries,dates,codes=UKCovid19Data.getUKCovid19Data('England',window+1,None)

rate={k:(timeseries[k][-1]/timeseries[k][-1-window])**(1.0/(window))-1.0 for k in timeseries.keys() if timeseries[k][-1-window]>0.0}
interesting=frozenset(rate.keys())

rawvotes=getRawVotes()
areas=getAreas(interesting)

votes={}

for c in interesting:
    
    if not c in votes:
        votes[c]=defaultdict(float)

    for a in areas[c]:
        if a in rawvotes:
            for k in rawvotes[a]:
                votes[c][k]+=rawvotes[a][k]
            
print len(interesting),'interesting'
novotes=[k for k in interesting if votes[k]['Total']==0.0]
print 'No votes for',len(novotes),':',novotes

for party in ['Con','Lab','LibDem','Brexit','Green']:

    fig=plt.figure(figsize=(8,6))

    print 'Top 5 for',party,sorted([(100.0*votes[k][party]/votes[k]['Total'],k) for k in rate.keys()],key=lambda it: it[0],reverse=True)[:5]
    
    y=np.array([100.0*rate[k] for k in rate.keys()])
    x=np.array([100.0*votes[k][party]/votes[k]['Total'] for k in rate.keys()])
    w=np.array([votes[k]['Total'] for k in rate.keys()])
    s=np.sqrt(w/100.0)
    
    plt.scatter(x,y,s=s)

    # Unweighted regression line
    r=scipy.stats.linregress(x,y)
    gradient,intercept,r_value,p_value,std_err=r
    print 'Unweighted',gradient,intercept
    
    rx=np.linspace(min(x),max(x),100)
    ry=gradient*rx+intercept
    plt.plot(rx,ry,color='tab:orange',label='Linear regression (unweighted)')
    
    # Weighted regression line
    coef=np.polyfit(x,y,1,w=w)
    print 'Weighted',coef[0],coef[1]
    ry=coef[1]+coef[0]*rx  # Highest power first
    plt.plot(rx,ry,color='tab:red',label='Linear regression (weighted by total votes)')
    rw=corr(x,y,w)
    
    # Weighted quadratic regression line
    coef=np.polyfit(x,y,2,w=w)
    qy=coef[2]+coef[1]*rx+coef[0]*rx**2
    plt.plot(rx,qy,color='tab:green',label='Quadratic best fit (weighted)')

    plt.title('Case-count growth rate vs. {} vote share'.format(party))
    
plt.show()
