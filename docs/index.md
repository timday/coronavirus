#ArmchairEpidemiology
=====================

Some dubious charting with coronavirus data.  Images can be clicked-through to full-resolution versions.  Note that most (all?) the non-scatterplot charts are log-scale on the y-axis.  Straight lines are actually exponential growth!

Sections on this page for [global](#global), [UK](#uk) and [USA](#usa) data, and some [Links](#links) to much better pages than this one.

Code (Python, matplotlib) which generated these plots at <https://github.com/timday/coronavirus>.

Global
======

Plots created from the JHU tracker's data (available at <https://github.com/CSSEGISandData/COVID-19>).

Cases
-----

Case-count growth rates by country (China split Hubei/non-Hubei).  Day-to-day increases plotted as points, with the lines showing growth over a 1-week window (this smooths out quite volatile daily rates and should remove any quirks of case reporting at weekends).  

[![Country's growth](img/global/small/growth.png)](img/global/growth.png)

A couple of aligned plots of the cumulative case and death count curves.

[![Aligned cases](img/global/small/aligned-cases.png)](img/global/aligned-cases.png)
[![Aligned deaths](img/global/small/aligned-deaths.png)](img/global/aligned-deaths.png)

"Active cases", assuming a model where newly identified cases become "active" for 2-3 weeks (uniformly distributed; eventual outcome doesn't matter).  This is an alternative to using JHU's "recovered" counts as there seems to be some doubt about how reliable they are in many countries.

[![Active cases](img/global/small/active-log.png)](img/global/active-log.png)

And alternative views of the same "active cases" but plotting the number as a proportion of a country's population (although splitting China between Hubei and all other provinces).
Log and linear scale plots shown (the linear scale makes it clearer which countries are over the hump).

[![Active cases](img/global/small/active-prop-log.png)](img/global/active-prop-log.png)
[![Active cases](img/global/small/active-prop-lin.png)](img/global/active-prop-lin.png)

Projections
-----------

Some simple (and very naive) next two months projections of case-counts (and derived "active cases") for the worst affected countries, simply by fitting (least squares) some simple models to the data so far.
Models include straight exponential growth, variants with the growth rate decaying linearly or exponentially, with or without a constant floor, the logistic equation and linear decay to zero growth at a given time.
In the chart legends there are 1-3 tick marks against the top 3 models best fitting the data.
The best 3 models are also shown rendered as "active cases" (assuming each new case then has a uniform duration of 2-3 weeks of being "active" before it is resolved).

[![Projections](img/global/small/projections-0.png)](img/global/projections-0.png)
[![Projections](img/global/small/projections-1.png)](img/global/projections-1.png)
[![Projections](img/global/small/projections-2.png)](img/global/projections-2.png)
[![Projections](img/global/small/projections-3.png)](img/global/projections-3.png)

Note: Model parameters are constrained to only allow growth rates to fall, because it never occurred to me that they'd do anything else!  So the best-fit "straight exponential growth" line (green) acts as an upper limit, and the variable-rate models can't actually flex above it even if that was a better fit.  Of course in practice, "case counts" must be at least as much a function of test and diagnostic capacity and policies as they are of any actual underlying growth of the disease itself.

UK
==

Data from the [PHE tracker](https://www.arcgis.com/apps/opsdashboard/index.html#/f94c3c90da5b4e9f9a0b19484dd4bb14) as curated into `.csv` file timeseries by <https://github.com/tomwhite/covid-19-uk-data>.

A messy plot of all the individual UTLAs/health-boards case counts:

[![UK case counts](img/uk/small/cases-log.png)](img/uk/cases-log.png)

Aligning all those on a constant number of cases:

[![UK case counts aligned](img/uk/small/cases-aligned-log.png)](img/uk/cases-aligned-log.png)

Correlations
------------
From the ONS, there is some UTLA-level data available (for England) for various health and deprivation metrics.  Is there some correlation between these measures and the rate that virus case-counts are growing at?

### Health

From the health data (from <https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/healthinequalities/datasets/indicatorsoflifestylesandwidercharacteristicslinkedtohealthylifeexpectancyinengland>, a 2017 release of 2015 data?), the score most correlated with the virus case-count growth rate appears to be the area's obesity rate:

[![England case count growth rate vs. obesity rate](img/uk/small/health-Obesity rate percentage.png)](img/uk/health-Obesity rate percentage.png)

A snapshot of the correlations with all the various health metrics (using case-count growth rates over the week prior to the date shown, sorted by the magnitude of the latest figures):
```
                                               Week to     Week to
                                              01/04/2020  08/04/2020
  Obesity rate (%)                           :  0.421       0.505
  Alcohol-related admissions (per 100,000)   :  0.221       0.470 
  Preventable Mortality (deaths per 100,000) :  0.232       0.429
  Smoking prevalence (%)                     :  0.230       0.318
  Physically active adults (%)               : -0.272      -0.266
  Economically inactive (%)                  :  0.121       0.161
  Employment rate (%)                        : -0.128      -0.157
  Adults eating 5-a-day (%)                  : -0.091      -0.145
  Unemployment rate (%)                      :  0.107       0.101
```

So - probably unsurprisingly - a general picture of unhealthy things being associated with faster case-count growth, and more health-positive things like physical activity, employment and even "5-a-day" being weakly linked with slower growth.

### Deprivation

From the "deprivation index" data (from <https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019>, the score most correlated with the virus case-count growth rate initially appeared to be "Education, Skills and Training" (higher score = worse!), but other deprivation indices have since taken over:

[![England case count growth rate vs. Barriers to Housing and Services deprivation](img/uk/small/deprivation-Barriers.png)](img/uk/deprivation-Barriers.png)
[![England case count growth rate vs. Health deprivation](img/uk/small/deprivation-Health.png)](img/uk/deprivation-Health.png)
[![England case count growth rate vs. Employment deprivation](img/uk/small/deprivation-Employment.png)](img/uk/deprivation-Employment.png)
[![England case count growth rate vs. Education deprivation](img/uk/small/deprivation-Education.png)](img/uk/deprivation-Education.png)

A snapshot of the correlations with all the various deprivation metrics (using case-count growth rates over the week prior to the date shown, sorted by the magnitude of the latest figures):
```
                                       Week to     Week to
                                      01/04/2020  08/04/2020
  Barriers to Housing and Services  :  -0.332     -0.552   
  Health Deprivation and Disability :   0.282      0.512
  Employment                        :   0.300      0.498
  Education, Skills and Training    :   0.395      0.463
  IMD                               :   0.205      0.322  
  Income                            :   0.183      0.293  
  Living Environment                :  -0.308     -0.256   
  IDACI                             :   0.162      0.242  
  Crime                             :   0.087      0.090  
  IDAOPI                            :  -0.061     -0.073   
```
(IMD - "Index of Multiple Deprivations", an aggregate score; IDACI - "Income Deprivation Affecting Children Index"; IDAOPI - "Income Deprivation Affecting Older People Index")

Note that while for most of these "deprivation index" numbers a higher score implies more deprivation, that appears to be reversed for the "Living Environment" score.  I'm not sure of the interpretation of the "Barriers to Housing and Services" score either, but a high number seems to be frequently associated with wealthy ( and therefore unaffordable) areas.

### Brexit and demographics

Plotting case-count growth against the 2016 Leave vote also shows some correlation:

[![UK case count growth rate vs. 2016 Leave vote](img/uk/small/brexit-England.png)](img/uk/brexit-England.png)

Reminder: correlation is not causation! And calling COVID-19 "The Brexit Disease" or "Leave Fever" would just be silly.

However, perhaps surprisingly, plotting case-count growth against demographics from the 2011 census there seems to be very little connection:

[![UK case count growth rate vs. demographics](img/uk/small/oldies-England.png)](img/uk/oldies-England.png)

which seems all the more surprising considering that there is actually a rather large correlation between the same demographic measure and the Leave vote:

[![Leave vote vs. demographics](img/uk/small/oldies-vote-England.png)](img/uk/oldies-vote-England.png)

### Elections

Unsurprisingly, given the previous section's results, the highest correlation I can find with 2017-2018 election data (England only) is the swing from the Conservative+UKIP vote share in 2017 to the Conservative+Brexit party vote share in 2019:

[![England case count growth rate vs. swing to Con+Brexit](img/uk/small/election-SwingConBrexit.png)](img/uk/election-SwingConBrexit.png)

### Income

Finally, case-count growth rate vs. GDHI ("gross disposable household income per head")

[![Leave vote vs. demographics](img/uk/small/income-GDHI per head.png)](img/uk/income-GDHI per head.png)

Money is quite an effective anti-viral it seems.

USA
===

A plot of coronavirus cases growth rate over the last week (JHU's data again) vs. each state's 2016 Trump vote (sized/weighted by total vote).

[![Case growth rate vs. 2016 vote](img/usa/small/president-2016.png)](img/usa/president-2016.png)

Unsurprisingly, the chart is pretty much flipped left-right for the 2012 & 2008 Obama votes.

[![Case growth rate vs. 2012 vote](img/usa/small/president-2012.png)](img/usa/president-2012.png)
[![Case growth rate vs. 2008 vote](img/usa/small/president-2008.png)](img/usa/president-2008.png)

Not particularly convincing looking trend-lines though.  California and Texas possibly dominate the weighted regression line.

Links
=====

* If you're not running the Covid-19 Symptom Tracker app... why not?  <https://covid.joinzoe.com/>.
    * Updates and webinars by the team behind this app at <https://covid.joinzoe.com/blog>.
    * App now also available for the USA at <https://covid.joinzoe.com/us>.
* The FT is doing some interesting charting: <https://www.ft.com/coronavirus-latest>.
* But Information Is Beautiful has the best looking: <https://informationisbeautiful.net/visualizations/covid-19-coronavirus-infographic-datapack/>
* Various "dashboard" pages:
    * [JHU's global page.](https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6)
    * [Worldometer's large collection of data.](https://www.worldometers.info/coronavirus/)
    * [PHE's UK page.](https://www.arcgis.com/apps/opsdashboard/index.html#/f94c3c90da5b4e9f9a0b19484dd4bb14)
    * [A page tracking Scotland.](https://www.travellingtabby.com/scotland-coronavirus-tracker/)
* Interesting read on why pandemic modelling is hard: <https://fivethirtyeight.com/features/why-its-so-freaking-hard-to-make-a-good-covid-19-model/>
* ...and another one on all the problems with case-counts: <https://fivethirtyeight.com/features/coronavirus-case-counts-are-meaningless/>
