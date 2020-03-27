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
        return [ts[d] for d in days if d in ts]

    timeseries={a:trim(timeseries[a]) for a in timeseries.keys()}

    timeseries={a:timeseries[a] for a in timeseries.keys() if len(timeseries[a])==window}

    codes={c:codes[c] for c in timeseries.keys()}
    
    return timeseries,days,codes
                    
