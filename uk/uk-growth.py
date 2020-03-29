#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

import UKCovid19Data

for what in [('England',7,None),('Scotland',7,None),('Wales',7,None),(None,7,None)]:
    
    timeseries,days,codes=UKCovid19Data.getUKCovid19Data(what[0],what[1]+1,what[2])

    print '------'
    print what[0],days[0],days[-1],len(days)

    assert len(days)==what[1]+1

    print 'Top 20 case counts'
    for k in sorted(timeseries,key=lambda k: timeseries[k][-1],reverse=True)[:20]:
        print '  {:32s}: {:d}'.format(codes[k],int(timeseries[k][-1]))
    
    print

    window=what[1]
    growth={
        k:(timeseries[k][-1]/timeseries[k][-1-window])**(1.0/window)
        for k in timeseries if timeseries[k][-window]>0.0
    }
    print 'Top growth ({} days, {} to {})'.format(window,days[0],days[-1])
    for k in sorted(growth.keys(),key=lambda k: growth[k],reverse=True):
        print '  {:32s}: {:.1f}%'.format(codes[k],100.0*(growth[k]-1.0))

    print
