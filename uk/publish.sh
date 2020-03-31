#!/bin/sh

DST=~/Dropbox/coronavirus/uk

mkdir -p ${DST}
rm -f ${DST}/*.png
rm -r -f ${DST}/small

cp output/*.png ${DST}/

mkdir -p ${DST}/small
( cd ${DST} ; for f in *.png ; do convert -geometry 50% "${f}" "small/${f}" ; done )
