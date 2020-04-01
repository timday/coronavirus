#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Some health by UTLA data from https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/healthinequalities/datasets/indicatorsoflifestylesandwidercharacteristicslinkedtohealthylifeexpectancyinengland
# "Indicators of lifestyles and wider characteristics linked to healthy life expectancy in England"
# 2017 release from 2015 data.

import csv
import distutils.dir_util
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import UKCovid19Data

window=7

timeseries,dates,codes=UKCovid19Data.getUKCovid19Data('England',window+1,None)   # Need 8 days to get 7 growth rates.

rate={k:(timeseries[k][-1]/timeseries[k][-1-window])**(1.0/(window))-1.0 for k in timeseries.keys() if timeseries[k][-1-window]>0.0}

for k in sorted(rate.keys(),key=lambda k: rate[k],reverse=True):
    print k,codes[k],rate[k]

def getHealth(column,what):
    csvfile=open('data/health/Persons-Table 1.csv','rb')
    reader=csv.reader(csvfile)
    rowCount=0

    result={}
    
    for row in reader:
        if rowCount<=3:
            pass
        elif rowCount==4:
            assert row[column].strip()==what
        elif rowCount<=6:
            pass
        elif len(row[0])!=9 or row[0][0]!='E':
            pass
        else:
            code=row[0]

            if code=='E10000009':
                code='E06000059'
            if code=='E06000028':
                code='E06000058'
            
            if row[column]!='':
                result[code]=float(row[column])

        rowCount+=1
        
    return result

correlation={}

def probe(column,what,desc):
    
    health=getHealth(column,what)

    print desc

    for k in rate.keys():
        if not k in health:
            print 'No health data for',k,codes[k]
    
    print '  Highest'
    for k in sorted(rate.keys(),key=lambda k: health[k],reverse=True)[:10]:
        print '    {:32s}: {:.2f}'.format(codes[k],health[k])
    print '  Lowest'
    for k in sorted(rate.keys(),key=lambda k: health[k],reverse=False)[:10]:
        print '    {:32s}: {:.2f}'.format(codes[k],health[k])
    
    y=np.array([100.0*rate[k] for k in rate.keys()])
    x=np.array([health[k] for k in rate.keys()])
    
    fig=plt.figure(figsize=(8,6))
    plt.scatter(x,y,color='tab:blue',alpha=0.5,label='UTLAs')
    r=scipy.stats.linregress(x,y)
    gradient,intercept,r_value,p_value,std_err=r
    
    rx=np.linspace(min(x),max(x),100)
    ry=gradient*rx+intercept
    plt.plot(rx,ry,color='tab:red',label='Linear regression')

    coef=np.polyfit(x,y,2)
    qy=coef[2]+coef[1]*rx+coef[0]*rx**2
    plt.plot(rx,qy,color='tab:green',label='Quadratic best fit')

    plt.legend(loc='upper left')
    
    ax=plt.gca()
    vals=ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])

    plt.ylabel('Daily % increase rate\n({} to {})'.format(dates[0],dates[-1]))
    plt.xlabel(desc)

    plt.title("England UTLAs: Virus case-count growth rate vs. {}\nr={:.3f}".format(desc,r_value))

    distutils.dir_util.mkpath('output')
    plt.savefig('output/health-{}.png'.format(desc),dpi=96)

    correlation[desc]=r_value

probe(2,'ILO employment rate (%)1','Employment rate (%)')
probe(3,'ILO unemployment rate (%)2','Unemployment rate (%)')
probe(4,'ILO economically inactive (%)1','Economically inactive (%)')
probe(5,'Smoking prevalence (%)3','Smoking prevalence (%)')
probe(6,'Proportion of adults defined as obese (%)4','Obesity rate (%)')
probe(7,'Alcohol-related admissions (per 100,000)7','Alcohol-related admissions (per 100,000)')
probe(8,'Physically active adults (%)5','Physically active adults (%)')
probe(9,'Proportion of adults eating 5-a day (%)6','Adults eating 5-a-day (%)')
probe(10,'Preventable Mortality (deaths per 100,000)8','Preventable Mortality (deaths per 100,000)')

print

print 'Correlations, highest to lowest (absolute)'
for k in sorted(correlation.keys(),key=lambda k: math.fabs(correlation[k]),reverse=True):
    print '  {:50s}: {:.3f}'.format(k.replace('- Average score',''),correlation[k]);

plt.show()
