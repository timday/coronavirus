#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np

from JHUData import *

timeseries=getJHUData(True)

for k in timeseriesKeys:

    casesNow=timeseries[0][k][-1]
    recoveredNow=timeseries[1][k][-1]
    deathsNow=timeseries[2][k][-1]

    lo=deathsNow/casesNow
    hi=deathsNow/(recoveredNow+deathsNow)

    t7=deathsNow/timeseries[0][k][-7]
    t14=deathsNow/timeseries[0][k][-14]
    t21=deathsNow/timeseries[0][k][-21]
    
    print '{:20s}: {:.1%} - {:.1%}  {:.1%} {:.1%} {:.1%}'.format(k,lo,hi,t7,t14,t21)
