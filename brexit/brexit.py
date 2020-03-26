#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import UKCovid19Data

# Dashboard at https://www.arcgis.com/apps/opsdashboard/index.html#/f94c3c90da5b4e9f9a0b19484dd4bb14
# Seems to be some csv files at https://www.gov.uk/government/publications/covid-19-track-coronavirus-cases
# Daily indicators contains nothing but today's data.
# Daily confirmed cases is a timeseries but just cases and totals.
# NHSR_Cases and UTLA cases table not a timeseries.

# Aha... solving the timeseries problem: https://github.com/tomwhite/covid-19-uk-data
#   Downloaded https://github.com/tomwhite/covid-19-uk-data/raw/master/data/covid-19-cases-uk.csv

# Referendum results from https://data.london.gov.uk/dataset/eu-referendum-results
#   Downloaded https://data.london.gov.uk/download/eu-referendum-results/52dccf67-a2ab-4f43-a6ba-894aaeef169e/EU-referendum-result-data.csv

# Lower tier to upper tier local authority info at
# https://geoportal.statistics.gov.uk/datasets/lower-tier-local-authority-to-upper-tier-local-authority-december-2017-lookup-in-england-and-wales
# (England and Wales only)

def cov(x, y, w):
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def getCodeRewrites():

    codes={}

    codes['S12000015']='S08000029'  # Fife code in referendum data.

    codes['E06000028']='E06000058'  # Somthing funny about Bournemouth

    upperMod={
        'E10000009':'E06000059',  # Change to Dorset code?
        # Looking for
        #'W11000023' # Betsi Cadwaladr
        #'W11000024' # Powys
        #'W11000025' # Hywel Dda
        #'W11000028' # Aneurin Bevan
        #'W11000029' # Cardiff and Vale
        #'W11000030' # Cwm Taf
        #'W11000031' # Swansea Bay
        }

    # First, read the England and Wales data
    csvfile=open('data/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority_December_2017_Lookup_in_England_and_Wales.csv')
    reader=csv.reader(csvfile)
    firstRow=True
    for row in reader:
        if firstRow:
            firstRow=False
            continue

        lower=row[0]
        upper=row[2]

        if not lower==upper:
            assert not lower in codes
            if upper in upperMod:
                upper=upperMod[upper]
            codes[lower]=upper

    # Now read the Scottish data
    csvfile=open('data/Datazone2011Lookup.csv','rb')
    reader=csv.reader(csvfile)
    firstRow=True
    for row in reader:
        if firstRow:
            firstRow=False
            continue

        council=row[5]
        healthboard=row[6]

        if council in codes:
            assert codes[council]==healthboard  # Check no contradictions
        else:
            codes[council]=healthboard
            
    return codes

def getVotesLeave(codeRewrites,interesting):
    csvfile=open('data/EU-referendum-result-data.csv')
    reader=csv.reader(csvfile)
    votesTotal=defaultdict(float)
    votesLeave=defaultdict(float)
    firstRow=True
    for row in reader:
        if firstRow:
            assert row[10]=='Valid_Votes'
            assert row[11]=='Remain'
            assert row[12]=='Leave'
            firstRow=False
            continue

        code=row[3]

        if code in codeRewrites and not code in interesting:
            code=codeRewrites[code]
        
        votesTotal[code]+=int(row[10])
        votesLeave[code]+=int(row[12])

    return votesTotal,votesLeave

for what in [('England',7,'England'),('Scotland',7,'Scotland')]: #,('Wales',5,'Wales')]:

    print what[2]

    fig=plt.figure(figsize=(8,6))
    
    window=what[1]

    timeseries,dates,codes=UKCovid19Data.getUKCovid19Data(what[0],window)

    print len(timeseries),'timeseries'
    
    codeRewrites=getCodeRewrites()
    votesTotal,votesLeave=getVotesLeave(codeRewrites,frozenset(codes.keys()))
    
    for k in sorted(codes.keys()):
        if not k in votesTotal:
            print 'No votes for',k,codes[k]
    
    rate={k:(timeseries[k][-1]/timeseries[k][-window])**(1.0/window)-1.0 for k in timeseries.keys() if timeseries[k][-window]>0.0}

    print len(rate),'rates computed'
    
    for k in rate.keys():
        if rate[k]<0.0:
            print 'Negative rate:',k,areas[k],rate[k]
    
    matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
    matplotlib.rcParams['font.family'] = "sans-serif"
    
    x=np.array([100.0*votesLeave[k]/votesTotal[k] for k in rate.keys()])
    y=np.array([100.0*rate[k] for k in rate.keys()])
    w=np.array([votesTotal[k] for k in rate.keys()])
    s=np.sqrt(w/50.0)

    plt.scatter(x,y,s=s,color='tab:blue',alpha=0.8)
    
    x=np.array([100.0*votesLeave[k]/votesTotal[k] for k in rate.keys()])
    y=np.array([100.0*rate[k] for k in rate.keys()])
    w=np.array([votesTotal[k] for k in rate.keys()])
    
    # Unweighted regression line
    r=scipy.stats.linregress(x,y)
    gradient,intercept,r_value,p_value,std_err=r
    print 'Unweighted',gradient,intercept
    
    rx=np.linspace(min(x),max(x),100)
    ry=gradient*rx+intercept
    plt.plot(rx,ry,color='tab:orange')
    
    # Weighted regression line
    coef=np.polyfit(x,y,1,w=w)
    print 'Weighted',coef[0],coef[1]
    ry=coef[1]+coef[0]*rx  # Highest power first
    plt.plot(rx,ry,color='tab:red')
    rw_value=corr(x,y,w)
    
    plt.xlabel('Leave vote')
    plt.ylabel('Daily % increase rate (last {} days)'.format(window))
    
    ax=plt.gca()
    vals=ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])
    vals=ax.get_xticks()
    ax.set_xticklabels(['{:,.1f}%'.format(x) for x in vals])
    
    plt.title("{} UTLA's cases growth rate vs. 2016 Leave vote.\nRegression lines: weighted r={:.2f} (red), unweighted r={:.2f} (orange)".format(what[2],rw_value,r_value))
    
    plt.savefig('output/brexit-{}.png'.format(what[0]),dpi=96)

plt.show()
