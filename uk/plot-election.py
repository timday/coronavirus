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

matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
matplotlib.rcParams['font.family'] = "sans-serif"

def cov(x, y, w):
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def value(x):
    return float(x)

# Accumulate votes into the areas given (wards and constituencies).  No translation of areas.
def getRawVotes(year):

    votes={}

    filename={
        2019:'data/HoC-2019GE-results-by-constituency.csv',
        2017:'data/HoC-GE2017-constituency-results.csv'
        }[year]

    # Less colunms in the 2017 data
    offset={
        2019:0,
        2017:3
        }[year]

    csvfile=open(filename)
    reader=csv.reader(csvfile)
    firstRow=True
    for row in reader:
        if firstRow:
            assert row[0]=='ons_id'
            assert row[15-offset]=='valid_votes'
            assert row[18-offset]=='con'
            assert row[19-offset]=='lab'
            assert row[20-offset]=='ld'
            assert row[21-offset]=={2019:'brexit',2017:'ukip'}[year]
            assert row[22-offset]=='green'
            assert row[23-offset]=='snp'
            firstRow=False
            continue

        ward=row[0]
        pcon=row[1]

        for where in set([ward,pcon]):   # Avoid double counting if any duplicates

            if not where in votes:
                votes[where]=defaultdict(float)
        
            votes[where]['Total'] +=value(row[15-offset])
            votes[where]['Con']   +=value(row[18-offset])
            votes[where]['Lab']   +=value(row[19-offset])
            votes[where]['LibDem']+=value(row[20-offset])
            votes[where]['Brexit']+=value(row[21-offset])
            votes[where]['Green'] +=value(row[22-offset])

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
print len(interesting),'interesting areas (from growth rate)'
areas=getAreas(interesting)

rawvotes={2017:getRawVotes(2017),2019:getRawVotes(2019)}

votes={2017:{},2019:{}}

for year in [2017,2019]:
    for c in interesting:
        
        if not c in votes[year]:
            votes[year][c]=defaultdict(float)
    
        for a in areas[c]:
            if a in rawvotes[year]:
                for k in rawvotes[year][a]:
                    votes[year][c][k]+=rawvotes[year][a][k]
            
    novotes=[k for k in interesting if votes[year][k]['Total']==0.0]
    print 'No',year,'votes for',len(novotes),':',novotes

for party in ['Con','Lab','LibDem','Brexit','Green','SwingCon','SwingBrexit','SwingConBrexit','SwingGreen']:

    fig=plt.figure(figsize=(8,6))

    print 'Top 5 for',party,sorted([(100.0*votes[year][k][party]/votes[year][k]['Total'],k) for k in rate.keys()],key=lambda it: it[0],reverse=True)[:5]

    interesting=sorted(rate.keys(),key=lambda k: votes[2019][k]['Total'],reverse=True)

    if party=='SwingCon':
        x=np.array([100.0*votes[2019][k]['Con']/votes[2019][k]['Total'] - 100.0*votes[2017][k]['Con']/votes[2017][k]['Total'] for k in interesting])
    elif party=='SwingBrexit':
        x=np.array([100.0*votes[2019][k]['Brexit']/votes[2019][k]['Total'] - 100.0*votes[2017][k]['Brexit']/votes[2017][k]['Total'] for k in interesting])
    elif party=='SwingConBrexit':
        x=np.array([100.0*(votes[2019][k]['Con']+votes[2019][k]['Brexit'])/votes[2019][k]['Total'] - 100.0*(votes[2017][k]['Con']+votes[2017][k]['Brexit'])/votes[2017][k]['Total'] for k in interesting])
    elif party=='SwingGreen':
        x=np.array([100.0*votes[2019][k]['Green']/votes[2019][k]['Total'] - 100.0*votes[2017][k]['Green']/votes[2017][k]['Total'] for k in interesting])        
    else:
        x=np.array([100.0*votes[2019][k][party]/votes[2019][k]['Total'] for k in interesting])

    y=np.array([100.0*rate[k] for k in interesting])

    c=[UKCovid19Data.colorsByRegion[UKCovid19Data.whichRegion(k)] for k in interesting]

    w=np.array([votes[2019][k]['Total'] for k in interesting])
    s=np.sqrt(w/100.0)
    
    plt.scatter(x,y,c=c,s=s)

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

    ax=plt.gca()
    vals=ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])
    vals=ax.get_xticks()
    ax.set_xticklabels(['{:,.1f}%'.format(x) for x in vals])

    plt.ylabel('Daily % growth rate {} to {}'.format(dates[-1-window],dates[-1]))
    if party[:5]!='Swing':
        plt.xlabel('{} % of vote'.format(party))
    else:
        plt.xlabel('Percentage swing')

    regionsUsed=sorted(list(set([UKCovid19Data.whichRegion(k) for k in interesting])))
    handles,labels = ax.get_legend_handles_labels()
    handles.extend([matplotlib.patches.Patch(color=UKCovid19Data.colorsByRegion[k],label=k) for k in regionsUsed])
    plt.legend(handles=handles,loc='upper right',prop={'size':6})

    if party[:5]!='Swing':
        plt.title('Case-count growth rate vs. {} vote share.\nRegression lines: weighted (red) r={:.3f}, unweighted (orange) r={:.3f}'.format(party,r_value,rw))
    elif party=='SwingCon':
        plt.title('Case-count growth rate vs. 2017-2019 Conservative swing\nRegression lines: weighted (red) r={:.3f}, unweighted (orange) r={:.3f}'.format(r_value,rw))
    elif party=='SwingBrexit':
        plt.title('Case-count growth rate vs. 2017-2019 UKIP-Brexit swing\nRegression lines: weighted (red) r={:.3f}, unweighted (orange) r={:.3f}'.format(r_value,rw))
    elif party=='SwingConBrexit':
        plt.title('Case-count growth rate vs. 2017-2019 Conservative+Brexit swing\nRegression lines: weighted (red) r={:.3f}, unweighted (orange) r={:.3f}'.format(r_value,rw))
    elif party=='SwingGreen':
        plt.title('Case-count growth rate vs. 2017-2019 Green swing\nRegression lines: weighted (red) r={:.3f}, unweighted (orange) r={:.3f}'.format(r_value,rw))

    distutils.dir_util.mkpath('output')
    plt.savefig('output/election-{}.png'.format(party),dpi=96)
        
plt.show()
