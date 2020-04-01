#!/bin/sh

./plot-growth.py &
sleep 1
./plot-active.py &
sleep 1
./plot-time.py &
sleep 1
./plot-gdp.py &
sleep 1
./plot-projections.py &
