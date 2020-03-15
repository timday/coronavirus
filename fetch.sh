#!/bin/bash

(cd data ; rm -f time_series_19-covid-Confirmed.csv ; wget --backups=0 https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv )
(cd data ; rm -f time_series_19-covid-Recovered.csv ; wget --backups=0 https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv )
(cd data ; rm -f time_series_19-covid-Deaths.csv    ; wget --backups=0 https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv )

