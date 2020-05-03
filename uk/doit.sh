#!/bin/sh

./plot-cases.py &
sleep 10
./plot-health.py &
sleep 10
./plot-deprivation.py &
sleep 10
./plot-income.py &
sleep 10
./plot-brexit.py &
sleep 10
./plot-election.py &

# ./plot-growth.py  # Just prints stuff currently.
