#!/bin/sh

DST=~/Dropbox/coronavirus/

mkdir -p ${DST}
rm -f ${DST}/*.png
rm -r -f ${DST}/all
rm -r -f ${DST}/small

cp output/growth.png     ${DST}/
cp output/aligned-cases.png  ${DST}/
cp output/aligned-deaths.png  ${DST}/
cp output/active-log.png ${DST}/

mkdir -p ${DST}/small
for f in growth.png aligned-cases.png aligned-deaths.png active-log.png ; do convert -geometry 50% ${DST}/${f} ${DST}/small/${f} ; done

cp -r output/ ${DST}/all
