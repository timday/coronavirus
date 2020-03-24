#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
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

for row in reader:
    if firstRow:
        firstRow=False
        continue

    date=row[0]
    area=row[3]
    cases=value(row[4])

    if not area in timeseries:
        timeseries[area]={}
    timeseries[area][date]=cases

    days.add(date)

days=sorted(list(days))

days=days[-8:-1]
assert len(days)==7

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
    print '  ',k,timeseries[k][days[-1]]

print

growth={
    k:(timeseries[k][days[-1]]/timeseries[k][days[0]])**(1.0/7.0)
    for k in usable if len(timeseries[k])>=8 and timeseries[k][days[0]]>0.0
}
print 'Top 10 growth'
for k in sorted(growth.keys(),key=lambda k: growth[k],reverse=True)[:10]:
    print '  ',k,100.0*(growth[k]-1.0)
