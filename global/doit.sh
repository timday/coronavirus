#!/bin/sh

./analyse-growth.py &
sleep 1
./analyse-active.py &
sleep 1
./analyse-time.py &
sleep 1
./analyse.py &
