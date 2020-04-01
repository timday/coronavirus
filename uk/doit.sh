#!/bin/sh

./plot-cases.py &
sleep 1
./plot-deprivation.py &
sleep 1
./plot-health.py &
sleep 1
./plot-brexit.py &
sleep 1
./plot-income.py &

# ./plot-growth.py  # Just prints stuff currently.
