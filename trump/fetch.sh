#!/bin/bash

# Use data from https://github.com/coviddata/covid-api 'cos JHU seem to have abandoned US states in timeseries.
# Now https://coviddata.github.io/coviddata/

mkdir -p data

(cd data ; rm -f cases.csv ; wget --backups=0 https://coviddata.github.io/coviddata/v1/regions/cases.csv )
