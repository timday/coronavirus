#!/bin/sh

DST=../docs/img/global

mkdir -p ${DST}

cp output/growth.png     ${DST}/
cp output/aligned-cases.png  ${DST}/
cp output/aligned-deaths.png  ${DST}/
cp output/active-log.png ${DST}/
cp output/projections-0.png ${DST}/
cp output/projections-1.png ${DST}/

mkdir -p ${DST}/small
for f in growth.png aligned-cases.png aligned-deaths.png active-log.png ; do convert -geometry 50% ${DST}/${f} ${DST}/small/${f} ; done

