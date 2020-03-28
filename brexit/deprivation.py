#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Deprivation data from
# https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/834001/File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx
# at
# https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019

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

def getDeprivation(filename,column,what):
    csvfile=open('data/deprivation/{}-Table 1.csv'.format(filename),'rb')
    reader=csv.reader(csvfile)
    firstRow=True

    result={}
    
    for row in reader:
        if firstRow:
            firstRow=False
            assert row[column].strip()==what
            continue

        code=row[0]
        result[code]=float(row[column])

    return result

correlation={}

def probe(filename,column,what):
    
    deprivation=getDeprivation(filename,column,what)

    print what
    print '  Highest'
    for k in sorted(rate.keys(),key=lambda k: deprivation[k],reverse=True)[:10]:
        print '    {:32s}: {:.2f}'.format(codes[k],deprivation[k])
    print '  Lowest'
    for k in sorted(rate.keys(),key=lambda k: deprivation[k],reverse=False)[:10]:
        print '    {:32s}: {:.2f}'.format(codes[k],deprivation[k])
    
    y=np.array([100.0*rate[k] for k in rate.keys()])
    x=np.array([deprivation[k] for k in rate.keys()])
    
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

    plt.title('Deprivation: {}\nr={:.3f}'.format(what,r_value))

    plt.savefig('output/deprivation-{}.png'.format(filename),dpi=96)

    correlation[what]=r_value
    
probe('Education',4,'Education, Skills and Training - Average score')
probe('Health',4,'Health Deprivation and Disability - Average score')
probe('Employment',4,'Employment - Average score')
probe('Living',4,'Living Environment - Average score')
probe('IDACI',4,'IDACI - Average score')   # Income Deprivation Affecting Children Index
probe('IMD',4,'IMD - Average score')
probe('Barriers',4,'Barriers to Housing and Services - Average score')
probe('Income',4,'Income - Average score')
probe('Crime',4,'Crime - Average score')
probe('IDAOPI',4,'IDAOPI - Average score')   # Income Deprivation Affecting Older People Index

print

print 'Correlations, highest to lowest'
for k in sorted(correlation.keys(),key=lambda k: correlation[k],reverse=True):
    print '  {:40s}: {:.3f}'.format(k.replace('- Average score',''),correlation[k]);

plt.show()
