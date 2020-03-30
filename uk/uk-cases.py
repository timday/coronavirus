#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import UKCovid19Data

regions=UKCovid19Data.getUKRegions()

def whichRegion(code):
    if code[0]=='E':
        return regions[code]
    elif code[0]=='W':
        return 'Wales'
    elif code[0]=='S':
        return 'Scotland'
    else:
        assert False
        return '???'

colorsByRegion={
    'East Midlands'            :'tab:olive',
    'East of England'          :'tab:pink',
    'London'                   :'black',
    'North East'               :'tab:cyan',
    'North West'               :'tab:purple',
    'South East'               :'tab:gray',
    'South West'               :'tab:orange',
    'West Midlands'            :'tab:green',
    'Yorkshire and The Humber' :'tab:brown',
    'Wales'                    :'tab:red',
    'Scotland'                 :'tab:blue'
}

for chart in [0,1]:

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
    
        for k in sorted(timeseries.keys(),key=lambda k: timeseries[k][-1]):   # Plot highest current case counts last
            cases=np.array([y for y in timeseries[k]])
            if chart==1:
                cases[cases<10.0]=np.nan

            region=whichRegion(k)
                
            plt.plot(mdays,cases,color=colorsByRegion[region],alpha=0.8)
            if not np.isnan(cases[-1]):
                plt.text(mdays[-1]+0.05,cases[-1],codes[k],horizontalalignment='left',verticalalignment='center',fontdict={'size':8,'alpha':0.75,'weight':'bold','color':colorsByRegion[region]})
        
    legends=[matplotlib.patches.Patch(color=colorsByRegion[k],label=k) for k in sorted(colorsByRegion.keys())]
    plt.legend(handles=legends)
    
    plt.legend(loc='upper left')
        
    plt.subplots_adjust(right=0.8)
    
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=75,fontsize=10)
    
    plt.xlim(left=mdayslo,right=mdayshi)
    plt.ylim(bottom={0:0.0,1:10.0}[chart])
    plt.gca().set_ylabel('Cumulative cases')
    if chart==1:
        plt.yscale('symlog')
    
    plt.title({0:'Cases by UTLA',1:'Cases by UTLA (log scale) from $\geq10$'}[chart])
    
    plt.savefig(
        {
            0:'output/cases.png',
            1:'output/cases-log.png'
        }[chart]
        ,dpi=96
    )

plt.show()
