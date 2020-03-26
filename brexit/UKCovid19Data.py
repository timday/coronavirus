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

def getUKCovid19Data(nation,window):

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

        area=row[3]

        codes[code]=area  # Taking the last one seems to be OK.  Some instability earlier.  https://github.com/tomwhite/covid-19-uk-data/issues/15
        
        cases=value(row[4])

        if not code in timeseries:
            timeseries[code]={}
        timeseries[code][date]=cases

        days.add(date)

    days=sorted(list(days))[-window:]
    assert len(days)==window
    
    def trim(ts):
        return [it[1] for it in sorted(ts.items(),key=lambda x: x[0])][-window:]

    timeseries={a:trim(timeseries[a]) for a in timeseries.keys()}

    timeseries={a:timeseries[a] for a in timeseries.keys() if len(timeseries[a])==window}
        
    return timeseries,days,codes
                    
