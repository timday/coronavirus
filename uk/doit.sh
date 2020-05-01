#!/bin/sh

./plot-cases.py &
sleep 5
./plot-deprivation.py &
sleep 5
./plot-health.py &
sleep 5
./plot-brexit.py &
sleep 5
./plot-election.py &
sleep 5
./plot-income.py &

# ./plot-growth.py  # Just prints stuff currently.
