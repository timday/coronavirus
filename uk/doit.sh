#!/bin/sh

./plot-cases.py &
sleep 2
./plot-deprivation.py &
sleep 2
./plot-health.py &
sleep 2
./plot-brexit.py &
sleep 2
./plot-election.py &
sleep 2
./plot-income.py &

# ./plot-growth.py  # Just prints stuff currently.
