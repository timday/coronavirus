#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# TODO: Some useful stuff on setting fonts at http://jonathansoma.com/lede/data-studio/matplotlib/changing-fonts-in-matplotlib/

# Data from https://www.arcgis.com/apps/opsdashboard/index.html#/f94c3c90da5b4e9f9a0b19484dd4bb14

# Upper Tier Local Authorities (UTLA) and NHS Regions tab

# 2020-03-21, dummy data
cases={
    'London'                   :[1965,2000],
    'South East'               :[ 492, 500],
    'Midlands'                 :[ 491, 500],
    'North West'               :[ 312, 500],
    'North East\nand Yorkshire':[ 298, 500],
    'East of England'          :[ 221, 500],
    'South West'               :[ 216, 500]
}

# From wikipedia https://en.wikipedia.org/wiki/Results_of_the_2016_United_Kingdom_European_Union_membership_referendum#Greater_London
# Just sort the above from least-to-most Leave
order=[
    'London',
    'South East',
    'South West',                 
    'North West',                 
    'East of England',            
    'North East\nand Yorkshire',
    'Midlands'
]

def expand(s):
    if s=='London':
        return s+'\n(Most Remain)'
    elif s=='Midlands':
        return s+'\n(Most Leave)'
    else:
        return s

growth=[100.0*float(cases[k][1])/float(cases[k][0]) for k in order]

pos=np.arange(len(order))

fig=plt.figure(figsize=(8,6))

plt.bar(
    pos,
    growth
)

ax=plt.gca()
vals=ax.get_yticks()
ax.set_yticklabels(['{:,.1f}%'.format(x) for x in vals])
ax.set_xticks(pos)
ax.set_xticklabels(map(expand,order))

for tick in ax.get_xticklabels():
    tick.set_fontname("Comic Sans MS")
    tick.set_fontsize(8)

for tick in ax.get_yticklabels():
    tick.set_fontname("Comic Sans MS")
    tick.set_fontsize(8)

plt.show()
