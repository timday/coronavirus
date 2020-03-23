#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

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

def value(s):
    if s=='1 to 4':
        return 2.5
    else:
        return float(s)

def getCases():
    csvfile=open('data/covid-19-cases-uk.csv')
    reader=csv.reader(csvfile)
    timeseries=defaultdict(list)
    areas=defaultdict(set)
    firstRow=True
    for row in reader:
        if firstRow:
            firstRow=False
            continue

        code=row[2]
        area=row[3]
        cases=value(row[4])

        if code=='':
            continue

        if area=='Orkney':
            assert code=='S08000024'
            code='S08000025'

        if area=='Western Isles':
            assert code=='S08000030'
            code='S08000028'
        
        timeseries[code].append(cases)

        areas[code].add(area)
        
    return areas,timeseries


def getVotes(interesting,areas):
    csvfile=open('data/EU-referendum-result-data.csv')
    reader=csv.reader(csvfile)
    votes={}
    firstRow=True
    for row in reader:
        if firstRow:
            assert row[10]=='Valid_Votes'
            assert row[11]=='Remain'
            assert row[12]=='Leave'
            firstRow=False
            continue

        if row[3] in interesting:
            votes[row[3]]=value(row[12])/value(row[10])


    print 'Missing votes for',[areas[k] for k in interesting if not k in votes]
            
    return votes

areas,timeseries=getCases()
#print areas
#print timeseries
votes=getVotes(frozenset(areas.keys()),areas)

# TODO: Need to figure out things like Devon and Surrey which seem to be split in referendum data
# Need to link to lower tier authorities.  UTLA E10000030 includes a bunch of lower tier stuff
# This looks useful: https://geoportal.statistics.gov.uk/datasets/lower-tier-local-authority-to-upper-tier-local-authority-december-2017-lookup-in-england-and-wales

print votes

print len(votes),len(areas)


exit()

# Data captured Upper Tier Local Authorities (UTLA) and NHS Regions tab
#   2020-03-21
#   2020-03-22
cases={
    'London'                   :[1965,2189],
    'South East'               :[ 492, 624],
    'Midlands'                 :[ 491, 536],
    'North West'               :[ 312, 390],
    'North East\nand Yorkshire':[ 298, 368],
    'East of England'          :[ 221, 274],
    'South West'               :[ 216, 242]
}

# From wikipedia https://en.wikipedia.org/wiki/Results_of_the_2016_United_Kingdom_European_Union_membership_referendum#Greater_London
# Just sort the above from least-to-most Leave
order=[
    'London',
    'South East',
    'South West',                 
    'North West',                 
    'East of England',            
    'North East\nand Yorkshire',
    'Midlands'
]

def expand(s):
    if s=='London':
        return s+'\n(Most Remain)'
    elif s=='Midlands':
        return s+'\n(Most Leave)'
    else:
        return s

growth=[100.0*(float(cases[k][1])/float(cases[k][0])-1.0) for k in order]

pos=np.arange(len(order))

matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
matplotlib.rcParams['font.family'] = "sans-serif"

fig=plt.figure(figsize=(8,6))

plt.bar(
    pos,
    growth
)

ax=plt.gca()
vals=ax.get_yticks()
ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])
ax.set_xticks(pos)
ax.set_xticklabels(map(expand,order))

for tick in ax.get_xticklabels():
    #tick.set_fontname("Comic Sans MS")
    tick.set_fontsize(8)

for tick in ax.get_yticklabels():
    #tick.set_fontname("Comic Sans MS")
    tick.set_fontsize(8)

plt.title('Virus cases growth rate 2020/03/21 - 2020/03/22 by NHS Region')

plt.savefig('output/brexit.png',dpi=96)

plt.show()
