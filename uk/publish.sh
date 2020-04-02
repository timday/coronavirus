#!/bin/sh

DST=../docs/img/uk

mkdir -p ${DST}
mkdir -p ${DST}/small

# Use -define png:exclude-chunks=date to stop trivial rebuilds from spamming git
# see https://imagemagick.org/discourse-server/viewtopic.php?t=21711
CP="convert -define png:exclude-chunks=date"

for f in cases-log.png cases-aligned-log.png "health-Obesity rate percentage.png" deprivation-Education.png "income-GDHI per head.png" brexit-England.png oldies-England.png oldies-vote-England.png ; do
    ${CP}               "output/${f}" "${DST}/${f}"
    ${CP} -geometry 50% "output/${f}" "${DST}/small/${f}"
done
