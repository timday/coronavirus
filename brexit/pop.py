#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Population from 2011 census from https://www.kaggle.com/electoralcommission/brexit-results#census.csv
# in data/census.csv

from collections import defaultdict
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import UKCovid19Data

def cov(x, y, w):
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def corr(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

