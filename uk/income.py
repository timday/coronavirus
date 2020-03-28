#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Some income by UTLA data from https://www.ons.gov.uk/economy/regionalaccounts/grossdisposablehouseholdincome/datasets/regionalgrossdisposablehouseholdincomegdhibylocalauthorityintheuk
# HOWEVER, need to roll up lower to upper local authorities, probably weighted by population.


from collections import defaultdict
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import UKCovid19Data

import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import UKCovid19Data

window=7

timeseries,dates,codes=UKCovid19Data.getUKCovid19Data('England',8)   # Need 8 days to get 7 growth rates.

rate={k:(timeseries[k][-1]/timeseries[k][-1-window])**(1.0/(window))-1.0 for k in timeseries.keys() if timeseries[k][-1-window]>0.0}

for k in sorted(rate.keys(),key=lambda k: rate[k],reverse=True):
    print k,codes[k],rate[k]

def getIncome(filename,column,what):
    csvfile=open('data/income/{}-Table 1.csv'.format(filename),'rb')
    reader=csv.reader(csvfile)
    rowCount=0

    result={}
    
    for row in reader:
        if rowCount<2:
            pass
        elif rowCount==2:
            assert row[column].strip()==what
        else:
            code=row[1]
            if row[column]!='':
                result[code]=float(row[column])

        rowCount+=1
        
    return result

correlation={}

def probe(filename,column,what):
    
    income=getIncome(filename,column,what)

    print what

    for k in rate.keys():
        if not k in income:
            print 'No income for',k,codes[k]
    
    print '  Highest'
    for k in sorted(rate.keys(),key=lambda k: income[k],reverse=True)[:10]:
        print '    {:32s}: {:.2f}'.format(codes[k],income[k])
    print '  Lowest'
    for k in sorted(rate.keys(),key=lambda k: income[k],reverse=False)[:10]:
        print '    {:32s}: {:.2f}'.format(codes[k],income[k])
    
    y=np.array([100.0*rate[k] for k in rate.keys()])
    x=np.array([income[k] for k in rate.keys()])
    
    fig=plt.figure(figsize=(8,6))
    plt.scatter(x,y,color='tab:blue',alpha=0.5)
    r=scipy.stats.linregress(x,y)
    gradient,intercept,r_value,p_value,std_err=r
    
    rx=np.linspace(min(x),max(x),100)
    ry=gradient*rx+intercept
    plt.plot(rx,ry,color='tab:red')

    ax=plt.gca()
    vals=ax.get_yticks()
    ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])

    plt.ylabel('Daily % increase rate\n({} to {})'.format(dates[0],dates[-1]))
    plt.xlabel(what)

    plt.title('{}\nr={:.3f}'.format(filename,r_value))

    plt.savefig('output/income-{}.png'.format(filename),dpi=96)

    correlation[what]=r_value
    
probe('Gross disposable income',22,'2016')

print

print 'Correlations, highest to lowest'
for k in sorted(correlation.keys(),key=lambda k: correlation[k],reverse=True):
    print '  {:40s}: {:.3f}'.format(k.replace('- Average score',''),correlation[k]);

plt.show()
