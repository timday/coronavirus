#!/bin/bash

mkdir -p data

(cd data ; rm -r covid-19-cases-uk.csv ; wget --backups=0 https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-cases-uk.csv )
