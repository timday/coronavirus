#!/bin/sh

DST=~/Dropbox/coronavirus/

rm -r -f ${DST}
mkdir -p ${DST}

cp output/active-log.png ${DST}/
cp output/growth.png ${DST}/
cp output/aligned-*.png ${DST}/

cp -r output/ ${DST}/all
