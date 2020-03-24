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

for k in sorted(timeseries.keys(),key=lambda k: timeseries[k][-1]):
    print k,timeseries[k][-1]
