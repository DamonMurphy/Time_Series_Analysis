# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:29:11 2020

@author: Damon
"""

#import os
#import numpy as np
import pandas as pd
#from datetime import datetime
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from statsmodels.tsa.filters.hp_filter import hpfilter
#from statsmodels.tsa.seasonal import seasonal_decompose


# SES

file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\airline_passengers.csv'

df = pd.read_csv(file1,index_col=0,parse_dates=True)

df.columns
df.plot()

df = df.dropna()

df.index

df.index.freq='MS'

df.head()

span = 12
alpha = 2/(span+1)

df.rename(columns={'Thousands of Passengers':'Pass_K'},inplace=True)
df['EWMA12'] = df['Pass_K'].ewm(alpha=alpha,adjust=False).mean()


model = SimpleExpSmoothing(df['Pass_K'])
fitted_model = model.fit(smoothing_level=alpha,optimized=False)

fitted_model.fittedvalues.shift(-1)

df['SES12'] = fitted_model.fittedvalues.shift(-1)


# DOUBLE EXP SM
df['DES_add_12'] = ExponentialSmoothing(df['Pass_K'],trend='add').fit().fittedvalues.shift(-1)

df.iloc[:24].plot(figsize=(12,5))
df.iloc[-24:].plot(figsize=(12,5))

df['DES_mul_12'] = ExponentialSmoothing(df['Pass_K'],trend='mul').fit().fittedvalues.shift(-1)

df[['Pass_K','DES_add_12','DES_mul_12']].iloc[-24:].plot(figsize=(12,5))


df['TES_mul_12'] = ExponentialSmoothing(df['Pass_K'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues

df[['Pass_K','DES_mul_12','TES_mul_12']].plot(figsize=(12,5))
df[['Pass_K','DES_mul_12','TES_mul_12']].iloc[-24:].plot(figsize=(12,5))

