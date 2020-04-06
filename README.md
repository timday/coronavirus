Messing around charting coronavirus data (mainly global and UK case counts) and some other things.

Selected output at <https://timday.github.io/coronavirus/>, updated most days.

Contents
--------
Processing relevant to each area:

* global
* uk
* usa

In each folder, `./fetch.sh` updates daily-updated data (see code for the origins of other static csv files also in the repo), `./plot-*.py` generate various charts (saved to `./output/` as well as displayed), `./publish.sh` updates them to the GitHub Pages' `docs` directory.

* docs - content for GitHub Pages website at <https://timday.github.io/coronavirus/>

* experiments - Mainly playing around with DDEs for (largely abandoned) model-fitting projections.

Note to self
============

https repo access: Avoid username prompt on pushes with `git remote set-url origin https://<username>@github.com/timday/coronavirus.git`
