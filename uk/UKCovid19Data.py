#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import datetime

def value(s):
    if s=='1 to 4':
        return 2.5
    else:
        return float(s)

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
