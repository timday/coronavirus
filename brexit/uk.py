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

csvfile=open('data/covid-19-cases-uk.csv')  # Update from https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-cases-uk.csv
reader=csv.reader(csvfile)
firstRow=True

timeseries={}
days=set()

# NB Due to messing around in Wales switching between Local Authority and Health Board, probably can only trust stuff from the 21st March.
# See https://github.com/tomwhite/covid-19-uk-data/blob/master/README.md

for row in reader:
    if firstRow:
        firstRow=False
        continue

    ymd=map(int,row[0].split('-'))
    date=datetime.date(*ymd)
    if date<datetime.date(2020,3,21):
        continue
    
    area=row[3]
    cases=value(row[4])

    if not area in timeseries:
        timeseries[area]={}
    timeseries[area][date]=cases

    days.add(date)

days=sorted(list(days))

window=4
days=days[-1-window:-1]
assert len(days)==window

print 'Date range',days[0],days[-1]

incomplete=set()
print 'Incomplete timeseries '.format(days)
for k in timeseries.keys():
    has=set(timeseries[k].keys()).intersection(days)
    if len(has)!=len(days):
        print '  ',k,'missing',sorted(list(set(days).difference(has)))
        incomplete.add(k)

print

usable=[k for k in sorted(timeseries.keys()) if not k in incomplete]

print 'Top 10 case counts'
for k in sorted(usable,key=lambda k: timeseries[k][days[-1]],reverse=True)[:10]:
    print '  {:32s}: {:d}'.format(k,int(timeseries[k][days[-1]]))

print

growth={
    k:(timeseries[k][days[-1]]/timeseries[k][days[0]])**(1.0/window)
    for k in usable if len(timeseries[k])>=window and timeseries[k][days[0]]>0.0
}
print 'Top 10 growth'
for k in sorted(growth.keys(),key=lambda k: growth[k],reverse=True)[:10]:
    print '  {:32s}: {:.1f}%'.format(k,100.0*(growth[k]-1.0))
