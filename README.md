Data sources
============

* JHU tracker <https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6>

* JHU time series data <https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series>

Notes
=====

Better model of infectious period than numerical queue?

z is number contagious
y is number infectious
v is conversion rate from infectious to contagious
u is recovery rate from contagious
w is infection rate

DSolve[{z'[t]=v*y[t]-u*z[t],y'[t]=-v*y[t]+w*z[t]},{y[t],z[t]},t]

Does have an insanely complicated solution according to WolframAlpha (doesn't simplify enough, runs out of time).  But basically just exponentials.

Try with sympy


