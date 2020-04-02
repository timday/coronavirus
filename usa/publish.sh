#!/bin/sh

DST=../docs/img/usa

mkdir -p ${DST}
mkdir -p ${DST}/small

# Use -define png:exclude-chunks=date to stop trivial rebuilds from spamming git
# see https://imagemagick.org/discourse-server/viewtopic.php?t=21711
CP="convert -define png:exclude-chunks=date"

for f in president-2016.png president-2016.png president-2016.png ; do
    ${CP}               "output/${f}" "${DST}/${f}"
    ${CP} -geometry 50% "output/${f}" "${DST}/small/${f}"
done
