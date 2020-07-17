# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:02:05 2020

@author: Damon
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose


file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\macrodata.csv'

df = pd.read_csv(file1,index_col=0,parse_dates=True)

df.columns
df['realgdp'].plot()

gpd_cyc,gpd_tr = hpfilter(df['realgdp'],lamb=1600)

type(gpd_tr)

df['trend'] = gpd_tr

df[['realgdp','trend']].plot()

df[['realgdp','trend']]['2005-01-01':].plot()

df['cycle'] = gpd_cyc

df['cycle'].plot()


file2 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\airline_passengers.csv'
#df2 = pd.read_csv(file2,index_col=0,parse_dates=True)
df2 = pd.read_csv(file2,index_col=0)

df2.head()


df2.isnull().values.any()
df2 = df2.dropna(inplace=True)
df2.index
df2.columns
column = 'Passengers_K'

df2.rename(columns={'Thousands of Passengers':column}, inplace=True)


df2.dropna(inplace=True) 

df2.index = pd.to_datetime(df2.index)

df2.plot()

#res = seasonal_decompose(df2['Thousands of Passengers'],model='multiplicative')
res = seasonal_decompose(df2['Passengers_K'],model='multiplicative')

res.plot();
res.resid.plot()


# SMA
df2.head(10)
df2.plot()

df2['6mo_MA'] = df2['Passengers_K'].rolling(window=6).mean()
df2['12mo_MA'] = df2['Passengers_K'].rolling(window=12).mean()
df2.plot()



#EWMA
df2['EWMA-12'] = df2['Passengers_K'].ewm(span=12).mean()
df2['EWMA-6'] = df2['Passengers_K'].ewm(span=6).mean()
df2[['Passengers_K','EWMA-12']].plot()

df2.head()

df2[['Passengers_K','EWMA-12','12mo_MA']].plot()
df2[['Passengers_K','EWMA-6','6mo_MA']].plot()



