#!/bin/sh

DST=~/Dropbox/coronavirus/

mkdir -p ${DST}
rm -f ${DST}/*.png
rm -r -f ${DST}/all

cp output/growth.png ${DST}/
cp output/aligned-*.png ${DST}/

cp -r output/ ${DST}/all
