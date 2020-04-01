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

# Use -define png:exclude-chunks=date to stop trivial rebuilds from spamming git
# see https://imagemagick.org/discourse-server/viewtopic.php?t=21711
( cd ${DST} ; for f in *.png ; do convert -define png:exclude-chunks=date -geometry 50% ${f} small/${f} ; done )
