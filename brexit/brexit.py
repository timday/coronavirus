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

# Wales from
# https://geoportal.statistics.gov.uk/datasets/unitary-authority-to-local-health-board-april-2019-lookup-in-wales
# https://geoportal.statistics.gov.uk/datasets/unitary-authority-to-local-health-board-april-2019-lookup-in-wales

def cov(x, y, w):
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def getCodeRewrites(interesting):

    codes={}
    
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

        if upper=='E10000009':
            upper='E06000059'

        if lower in interesting:
            continue

        if upper in interesting:
            assert not lower in codes
            codes[lower]=upper
        
    # Now read the Scottish data
    csvfile=open('data/Datazone2011Lookup.csv','rb')
    reader=csv.reader(csvfile)
    firstRow=True
    for row in reader:
        if firstRow:
            firstRow=False
            continue

        lower=row[5]
        upper=row[6]

        if lower in interesting:
            continue

        if upper in interesting:
            
            if lower in codes:
                assert codes[lower]==upper  # Check no contradictions
            else:
                codes[lower]=upper

    # Now read the Welsh data
    csvfile=open('data/Unitary_Authority_to_Local_Health_Board_April_2019_Lookup_in_Wales.csv','rb')
    reader=csv.reader(csvfile)
    firstRow=True
    for row in reader:
        if firstRow:
            firstRow=False
            continue

        lower=row[0]
        upper=row[2]

        if lower in interesting:
            continue

        if upper in interesting:
            
            if lower in codes:
                assert codes[lower]==upper  # Check no contradictions
            else:
                codes[lower]=upper    
                
    # Some overrides
    codes['S12000015']='S08000029'  # Fife code in referendum data.    
    codes['E06000028']='E06000058'  # Somthing funny about Bournemouth?

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

        if code in codeRewrites:
            code=codeRewrites[code]

        if code in interesting:
            votesTotal[code]+=int(row[10])
            votesLeave[code]+=int(row[12])

    return votesTotal,votesLeave

def getDemographics(codeRewrites,interesting):
    csvfile=open('data/census.csv')
    reader=csv.reader(csvfile)

    populationTotal=defaultdict(float)
    populationAged=defaultdict(float)

    firstRow=True
    for row in reader:
        if firstRow:
            assert row[1]=='Code'
            assert row[3]=='All Residents'
            assert row[13]=='Age 45 to 49'
            assert row[14]=='Age 50 to 54'
            assert row[22]=='Age 90 and Over'
            firstRow=False
            continue
    
        code=row[1]

        if code in codeRewrites and not code in interesting:
            code=codeRewrites[code]

        if code in interesting:
            print row
            populationTotal[code]+=float(row[3])
            populationAged[code]+=sum([float(row[i]) for i in xrange(14,23) if row[i]!=''])  # 14: works; >=50

    return populationTotal,populationAged

matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
matplotlib.rcParams['font.family'] = "sans-serif"

def plot(x,y,w,s):

    fig=plt.figure(figsize=(8,6))

    plt.scatter(x,y,s=s,color='tab:blue',alpha=0.5)
    
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
    rw=corr(x,y,w)
    
    ax=plt.gca()
    vals=ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])
    vals=ax.get_xticks()
    ax.set_xticklabels(['{:,.1f}%'.format(x) for x in vals])

    return r.rvalue,rw

plots=[('England',7,'England'),('Scotland',7,'Scotland'),('Wales',5,'Wales'),(None,5,'England, Scotland and Wales')]
for p in range(0,4):

    what=plots[p]
    print what[2]

    window=what[1]

    timeseries,dates,codes=UKCovid19Data.getUKCovid19Data(what[0],window+1)   # Need 8 days to get 7 growth rates.

    print len(timeseries),'timeseries'
    for c in timeseries.keys():
        print '  ',c,codes[c],timeseries[c]

    interesting=frozenset(timeseries.keys())
    codeRewrites=getCodeRewrites(interesting)
    votesTotal,votesLeave=getVotesLeave(codeRewrites,interesting)

    # Couple of fixups to census data
    codeRewrites['E06000048']='E06000057' # Northumberland
    codeRewrites['E08000020']='E08000037' # Gateshead
    
    populationTotal,populationAged=getDemographics(codeRewrites,interesting)
    oldies={k:populationAged[k]/populationTotal[k] for k in populationTotal.keys()}

    print len(votesTotal),'votes'
    print len(oldies),'demographics'
    
    for k in sorted(timeseries.keys()):
        if not k in votesTotal:
            print 'No votes for',k,codes[k]
        if not k in oldies:
            print 'No demographics for',k,codes[k]

    for c in sorted(oldies.keys(),key=lambda c: oldies[c],reverse=True):
        print c,codes[c],oldies[c]
    
    rate={k:(timeseries[k][-1]/timeseries[k][-1-window])**(1.0/(window))-1.0 for k in timeseries.keys() if timeseries[k][-1-window]>0.0}

    print len(rate),'rates computed'
    
    for k in rate.keys():
        if rate[k]<0.0:
            print 'Negative rate:',k,codes[k],rate[k]
    
    x=np.array([100.0*votesLeave[k]/votesTotal[k] for k in rate.keys()])
    y=np.array([100.0*rate[k] for k in rate.keys()])
    w=np.array([votesTotal[k] for k in rate.keys()])
    s=np.sqrt(w/10.0)

    r,rw=plot(x,y,w,s)

    plt.xlabel('Leave vote')
    plt.ylabel('Daily % increase rate')
    plt.title("{}\nAreas' case growth rates {} to {} vs. 2016 Leave vote.\nRegression lines: weighted r={:.2f} (red), unweighted r={:.2f} (orange)".format(what[2],dates[0],dates[-1],rw,r))

    if p==3:
        outputfile='output/brexit-all.png'.format(what[0])
    else:
        outputfile='output/brexit-{}.png'.format(what[0])
    plt.savefig(outputfile,dpi=96)

    x=np.array([100.0*oldies[k] for k in rate.keys()])
    w=np.array([populationTotal[k] for k in rate.keys()])
    s=np.sqrt(w/50.0)

    r,rw=plot(x,y,w,s)

    plt.xlabel('% Population >=50 in 2011 census')
    plt.ylabel('Daily % increase rate')
    plt.title("{}\nAreas' case growth rates {} to {} vs. demographics.\nRegression lines: weighted r={:.2f} (red), unweighted r={:.2f} (orange)".format(what[2],dates[0],dates[-1],rw,r))

    if p==3:
        outputfile='output/oldies-all.png'.format(what[0])
    else:
        outputfile='output/oldies-{}.png'.format(what[0])
    plt.savefig(outputfile,dpi=96)

    y=np.array([100.0*votesLeave[k]/votesTotal[k] for k in rate.keys()])
    
    r,rw=plot(x,y,w,s)

    plt.xlabel('% Population >=50 in 2011 census')
    plt.ylabel('Leave vote')
    plt.title("{}\nAreas' 2016 Leave vote vs. demographics.\nRegression lines: weighted r={:.2f} (red), unweighted r={:.2f} (orange)".format(what[2],rw,r))

    if p==3:
        outputfile='output/oldies-vote-all.png'.format(what[0])
    else:
        outputfile='output/oldies-vote-{}.png'.format(what[0])
    plt.savefig(outputfile,dpi=96)

    
plt.show()
