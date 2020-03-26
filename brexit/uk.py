#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import UKCovid19Data

for what in [('England',7),('Scotland',7),('Wales',5),(None,5)]:
    
    timeseries,days,codes=UKCovid19Data.getUKCovid19Data(*what)

    print '------'
    print what[0],days[0],days[-1],len(days)

    assert len(days)==what[1]

    print 'Top 10 case counts'
    for k in sorted(timeseries,key=lambda k: timeseries[k][-1],reverse=True)[:10]:
        print '  {:32s}: {:d}'.format(codes[k],int(timeseries[k][-1]))
    
    print

    window=min(what[1],7)
    growth={
        k:(timeseries[k][-1]/timeseries[k][-window])**(1.0/window)
        for k in timeseries if timeseries[k][-window]>0.0
    }
    print 'Top 10 growth ({} days)'.format(window)
    for k in sorted(growth.keys(),key=lambda k: growth[k],reverse=True)[:10]:
        print '  {:32s}: {:.1f}%'.format(codes[k],100.0*(growth[k]-1.0))

    print
