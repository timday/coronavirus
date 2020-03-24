#!/bin/bash

mkdir -p data

(cd data ; rm -f time_series_covid19_confirmed_global.csv ; wget --backups=0 https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv )

(cd data ; rm -f time_series_covid19_deaths_global.csv ; wget --backups=0 https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv )

