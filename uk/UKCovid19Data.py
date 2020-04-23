#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import datetime
import numpy as np

def value(s):
    if s=='1 to 4':
        return 2.5
    elif s=='':
        return 0.0
    else:
        return float(s.replace(',',''))

# NB Due to messing around in Wales switching between Local Authority and Health Board, probably can only trust stuff from the 21st March for Wales.
# See https://github.com/tomwhite/covid-19-uk-data/blob/master/README.md

# Wales areas we want are
#  W11000028 Aneurin Bevan
#  W11000023 Betsi Cadwaladr
#  W11000029 Cardiff and Vale
#  W11000030 Cwm Taf Morgannwg
#  W11000025 Hywel Dda
#  W11000024 Powys
#  W11000031 Swansea Bay

def getUKCovid19Data(nation,window,startdate):

    csvfile=open('data/covid-19-cases-uk.csv')  # Update from https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-cases-uk.csv
    reader=csv.reader(csvfile)
    firstRow=True

    timeseries={}
    days=set()
    codes={}

    for row in reader:
        if firstRow:
            firstRow=False
            continue
        
        where=row[1]
        
        if where!=nation and nation!=None:
            continue

        code=row[2]
        if code=='':
            continue

        ymd=map(int,row[0].split('-'))
        date=datetime.date(*ymd)

        if startdate==None or date>=startdate:
        
            area=row[3]
    
            codes[code]=area  # Taking the last one seems to be OK.  Some instability earlier.  https://github.com/tomwhite/covid-19-uk-data/issues/15
            
            cases=value(row[4])
    
            if not code in timeseries:
                timeseries[code]={}
            timeseries[code][date]=cases
    
            days.add(date)

    # Fold City of London into Hackney (is split out later, but small)
    if 'E09000001' in timeseries:
        for d in timeseries['E09000001']:
            timeseries['E09000012'][d]+=timeseries['E09000001'][d]
        del timeseries['E09000001']
            
    days=sorted(list(days))
    if window!=None:
        assert len(days)>=window
        days=days[-window:]
        assert len(days)==window
        
    def trim(ts):
        return [ts[d] for d in days if d in ts]

    timeseries={a:trim(timeseries[a]) for a in timeseries.keys()}

    timeseries={a:timeseries[a] for a in timeseries.keys() if len(timeseries[a])==len(days)}

    codes={c:codes[c] for c in timeseries.keys()}

    for k in timeseries.keys():
        for i in xrange(1,len(timeseries[k])):
            if np.isnan(timeseries[k][i]):
                print 'Replacing NaN in {} at day {}'.format(k,i)
                timeseries[k][i]=2.5  # NaN means 1-4.  So, use average.  
                
    return timeseries,days,codes

def getUKCodeRewrites(interesting):

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

# Get higher-level regions for UTLAs (just England)
# File from http://geoportal.statistics.gov.uk/datasets/local-authority-district-to-county-april-2019-lookup-in-england
# Specifically http://geoportal.statistics.gov.uk/datasets/local-authority-district-to-region-april-2019-lookup-in-england

def getUKRegions():
    
    csvfile=open('data/Local_Authority_District_to_Region_April_2019_Lookup_in_England.csv','rb')
    reader=csv.reader(csvfile)
    firstRow=True

    regions={}
    
    for row in reader:
        if firstRow:
            firstRow=False
            continue

        code=row[0]
        region=row[3]

        regions[code]=region


    # Hmmm... seems to have quite a lot missing
    # Check at e.g http://statistics.data.gov.uk/atlas/resource?uri=http://statistics.data.gov.uk/id/statistical-geography/E10000011

    for k in ['E10000021','E10000024','E10000007','E10000018','E10000019']:
        regions[k]='East Midlands'

    for k in ['E10000028','E10000034','E10000031']:
        regions[k]='West Midlands'

    for k in ['E10000020','E10000015','E10000029','E10000003','E10000012']:
        regions[k]='East of England'

    for k in ['E10000023']:
        regions[k]='Yorkshire and The Humber'

    for k in ['E10000011','E10000025','E10000032','E10000030','E10000002','E10000014','E10000016','E10000017']:
        regions[k]='South East'

    for k in ['E10000008','E10000013']:
        regions[k]='South West'

    for k in ['E10000006']:
        regions[k]='North West'

    for k in ['E10000027']:
        regions[k]='South West'

    return regions


regions=None

def whichRegion(code):

    global regions
    
    if regions==None:
        regions=getUKRegions()
    
    if code[0]=='E':
        return regions[code]
    elif code[0]=='W':
        return 'Wales'
    elif code[0]=='S':
        return 'Scotland'
    elif code[0]=='N':
        return 'Northern Ireland'
    else:
        assert False
        return '???'

colorsByRegion={
    'East Midlands'            :'tab:olive',
    'East of England'          :'tab:pink',
    'London'                   :'black',
    'North East'               :'tab:cyan',
    'North West'               :'tab:purple',
    'South East'               :'tab:gray',
    'South West'               :'tab:orange',
    'West Midlands'            :'tab:green',
    'Yorkshire and The Humber' :'tab:brown',
    'Wales'                    :'tab:red',
    'Scotland'                 :'tab:blue',
    'Northern Ireland'         :'purple'
}
