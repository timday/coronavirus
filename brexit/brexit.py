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
    csvfile=open('data/covid-19-cases-uk.csv')  # Update from https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-cases-uk.csv
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

def getCodeRewrites():

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
        
        votesTotal[code]+=value(row[10])
        votesLeave[code]+=value(row[12])

    return votesTotal,votesLeave

areas,timeseries=getCases()
#print areas
#print timeseries

codeRewrites=getCodeRewrites()
votesTotal,votesLeave=getVotesLeave(codeRewrites,frozenset(areas.keys())

for k in timeseries.keys():
    if not k in votesTotal:
        print 'No votes for',k
    print k,areas[k],timeseries[k][-1],100.0*votesLeave[k]/votesTotal[k]
