#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import UKCovid19Data

for what in [('England',7),('Scotland',7),('Wales',7),(None,7)]:
    
    timeseries,days,codes=UKCovid19Data.getUKCovid19Data(*what)

    print '------'
    print what[0],days[0],days[-1],len(days)

    assert len(days)==what[1]

    print 'Top 20 case counts'
    for k in sorted(timeseries,key=lambda k: timeseries[k][-1],reverse=True)[:20]:
        print '  {:32s}: {:d}'.format(codes[k],int(timeseries[k][-1]))
    
    print

    window=min(what[1],7)
    growth={
        k:(timeseries[k][-1]/timeseries[k][-window])**(1.0/window)
        for k in timeseries if timeseries[k][-window]>0.0
    }
    print 'Top growth ({} days, {} to {})'.format(window,days[0],days[-1])
    for k in sorted(growth.keys(),key=lambda k: growth[k],reverse=True):
        print '  {:32s}: {:.1f}%'.format(codes[k],100.0*(growth[k]-1.0))

    print



#for k in growth.keys():
#    plt.plot(days,growth,label=k)

fig=plt.figure(figsize=(16,9))

mdays=[mdates.date2num(d) for d in days]

colors={'E':'tab:gray','W':'tab:red','S':'tab:blue'}
labels={'E':'England','W':'Wales','S':'Scotland'}
for k in timeseries.keys():
    plt.plot(mdays,timeseries[k],color=colors[k[0]],alpha=0.75)
    plt.text(mdays[-1]+0.05,timeseries[k][-1],codes[k],horizontalalignment='left',verticalalignment='center',fontdict={'size':8,'alpha':0.75,'weight':'bold','color':colors[k[0]]})

legends=[matplotlib.patches.Patch(color=colors[k],label=labels[k]) for k in ['E','W','S']]
plt.legend(handles=legends)

plt.legend(loc='upper left')
    
plt.subplots_adjust(right=0.8)

plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=75,fontsize=10)

plt.xlim(left=mdays[0],right=mdays[-1])
plt.ylim(bottom=0.0)
#plt.yscale('symlog')
plt.gca().set_ylabel('Cumulative cases')

plt.title('Cases by UTLA')
   
#plt.legend(fontsize='xx-small')

plt.show()
