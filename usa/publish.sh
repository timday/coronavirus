#!/bin/sh

DST=../docs/img/usa

mkdir -p ${DST}

cp output/president-2016.png     ${DST}/
cp output/president-2012.png     ${DST}/
cp output/president-2008.png     ${DST}/

mkdir -p ${DST}/small

# Use -define png:exclude-chunks=date to stop trivial rebuilds from spamming git
# see https://imagemagick.org/discourse-server/viewtopic.php?t=21711
( cd ${DST} ; for f in *.png ; do convert -define png:exclude-chunks=date -geometry 50% ${f} small/${f} ; done )
