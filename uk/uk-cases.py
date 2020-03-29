#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import UKCovid19Data

fig=plt.figure(figsize=(16,9))

mdayslo=None
mdayshi=None

for what in [('England',None,datetime.date(2020,3,7)),('Scotland',None,datetime.date(2020,3,7)),('Wales',None,datetime.date(2020,3,21))]:
    
    timeseries,days,codes=UKCovid19Data.getUKCovid19Data(*what)
    
    mdays=[mdates.date2num(d) for d in days]

    if mdayslo==None:
        mdayslo=mdays[0]
    else:
        mdayslo=min(mdayslo,mdays[0])

    if mdayshi==None:
        mdayshi=mdays[-1]
    else:
        mdayshi=max(mdayshi,mdays[-1])

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

plt.xlim(left=mdayslo,right=mdayshi)
plt.ylim(bottom=0.0)
#plt.yscale('symlog')
plt.gca().set_ylabel('Cumulative cases')

plt.title('Cases by UTLA')

plt.show()
