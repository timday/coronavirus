#!/bin/sh

./plot-growth.py &
sleep 2
./plot-active.py &
sleep 2
./plot-time.py &
sleep 2
./plot-gdp.py &
sleep 2
./plot-projections.py &
