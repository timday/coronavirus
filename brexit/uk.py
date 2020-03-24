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

for row in reader:
    if firstRow:
        firstRow=False
        continue

    date=row[0]
    area=row[3]
    cases=value(row[4])

    if not area in timeseries:
        timeseries[area]=[]
    timeseries[area].append(cases)

print 'Top 10 case counts'
for k in sorted(timeseries.keys(),key=lambda k: timeseries[k][-1],reverse=True)[:10]:
    print k,timeseries[k][-1]

print

growth={k:(timeseries[k][-1]/timeseries[k][-1-7])**(1.0/7.0) for k in timeseries.keys() if len(timeseries[k])>=8 and timeseries[k][-1-7]>0.0}
print 'Top 10 growth'
for k in sorted(growth.keys(),key=lambda k: growth[k],reverse=True)[:10]:
    print k,100.0*(growth[k]-1.0)
