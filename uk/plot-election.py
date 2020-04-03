#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Constituency to UTLA from http://geoportal.statistics.gov.uk/datasets/b7435625a4d442bcb331d610f16aacde_0

# Election data from https://commonslibrary.parliament.uk/research-briefings/cbp-8749/ (the accompanying results by constituency csv)

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

def getWardToUTLA():

    wardToUTLA={}
    
    csvfile=open('data/Ward_to_Westminster_Parliamentary_Constituency_to_Local_Authority_District_to_Upper_Tier_Local_Authority_December_2019_Lookup_in_the_United_Kingdom.csv')
    reader=csv.reader(csvfile)
    firstRow=True
    for row in reader:
        if firstRow:
            assert row[2]=='PCON19CD'
            assert row[6]=='UTLA19CD'
            firstRow=False
            continue

        ward=row[2]
        utla=row[6]

        wardToUTLA[ward]=utla

    return wardToUTLA

def value(x):
    return float(x)

def getVotes(wardToUTLA):

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
        utla=wardToUTLA[ward]

        if not utla in votes:
            votes[utla]=defaultdict(float)
            
        votes[utla]['Total']+=value(row[15])
        votes[utla]['Con']+=value(row[18])
        votes[utla]['Lab']+=value(row[19])
        votes[utla]['LibDem']+=value(row[20])
        votes[utla]['Brexit']+=value(row[21])
        votes[utla]['Green']+=value(row[22])

    return votes

wardToUTLA=getWardToUTLA()
votes=getVotes(wardToUTLA)

window=7

timeseries,dates,codes=UKCovid19Data.getUKCovid19Data('England',window+1,None)

fig=plt.figure(figsize=(8,6))

rate={k:(timeseries[k][-1]/timeseries[k][-1-window])**(1.0/(window))-1.0 for k in timeseries.keys() if timeseries[k][-1-window]>0.0}

x=[100.0*(rate[k]-1.0) for k in rate.keys()]
y=[100.0*votes[k]['Con']/votes[k]['Total'] for k in rate.keys()]
w=[votes[k]['Total'] for k in rate.keys()]
s=np.sqrt(w/1000.0)

plt.scatter(x,y,s=s)

plt.show()
