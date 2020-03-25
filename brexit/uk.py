#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def value(s):
    if s=='1 to 4':
        return 2.5
    else:
        return float(s)

# NB Due to messing around in Wales switching between Local Authority and Health Board, probably can only trust stuff from the 21st March for Wales.
# See https://github.com/tomwhite/covid-19-uk-data/blob/master/README.md

def getUK(nation,window):

    csvfile=open('data/covid-19-cases-uk.csv')  # Update from https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-cases-uk.csv
    reader=csv.reader(csvfile)
    firstRow=True

    timeseries={}
    days=set()

    for row in reader:
        if firstRow:
            firstRow=False
            continue
        
        where=row[1]
        if where!=nation and nation!=None:
            continue

        ymd=map(int,row[0].split('-'))
        date=datetime.date(*ymd)
            
        area=row[3]
        cases=value(row[4])

        if not area in timeseries:
            timeseries[area]={}
        timeseries[area][date]=cases

        days.add(date)

    days=sorted(list(days))[-window:]
    assert len(days)==window
    
    def trim(ts):
        return [it[1] for it in sorted(ts.items(),key=lambda x: x[0])][-window:]

    timeseries={a:trim(timeseries[a]) for a in timeseries.keys()}

    timeseries={a:timeseries[a] for a in timeseries.keys() if len(timeseries[a])==window}
        
    return timeseries,days
                    
for what in [('England',14),('Scotland',14),('Wales',4),(None,4)]:
    timeseries,days=getUK(*what)

    print '------'
    print what[0],days[0],days[-1],len(days)

    assert len(days)==what[1]

    print 'Top 10 case counts'
    for k in sorted(timeseries,key=lambda k: timeseries[k][-1],reverse=True)[:10]:
        print '  {:32s}: {:d}'.format(k,int(timeseries[k][-1]))
    
    print

    window=min(what[1],7)
    growth={
        k:(timeseries[k][-1]/timeseries[k][-window])**(1.0/window)
        for k in timeseries if timeseries[k][-window]>0.0
    }
    print 'Top 10 growth ({} days)'.format(window)
    for k in sorted(growth.keys(),key=lambda k: growth[k],reverse=True)[:10]:
        print '  {:32s}: {:.1f}%'.format(k,100.0*(growth[k]-1.0))

    print
