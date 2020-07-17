# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:25:21 2020

@author: Damon
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

year1 = 2020
month1 = 1
day1 = 2
hour1 = 13
min1 = 30
sec1 = 15

time1 = datetime(year1, month1, day1,hour1,min1,sec1)

# NUMPY
np.array(['2020-03-15','2020-03-16','2020-03-17'],dtype='datetime64')
np.array(['2020-03-15','2020-03-16','2020-03-17'],dtype='datetime64[s]')  # FOR SECONDS

np.arange('2018-06-01','2018-06-23',7,dtype='datetime64[D]')


# PANDAS
pd.date_range('2020-01-01',periods=7,freq='D')

pd.to_datetime(['1/2/2018','Jan 03, 2018'])

pd.to_datetime(['2/1/2018','3/1/2018'],format='%d/%m/%Y')

data = np.random.randn(3,2)
cols = ['A','B']
idx = pd.date_range('2020-01-01','2020-01-03')
df = pd.DataFrame(data,index=idx,columns=cols)

df.index
df.index.max()
df.index.argmax()

file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\starbucks.csv'

df = pd.read_csv(file1,index_col='Date',parse_dates=True)
df.head()

len(df)
df.index


# TIME RESAMPLE

df.resample(rule='A').mean()

def first_day(entry):
    if len(entry):
        #if len(entry)!=0:
        return entry[0]

df.resample(rule='A').apply(first_day)

df['Close'].resample('A').mean().plot.bar()

# TIME SHIFTING

df.shift(1)

df.shift(-1).head()

df.shift(periods=1,freq='M')



# ROLLING AND EXPANDING

df['Close'].plot(figsize=(12,5))

# CREATE 7-DAY MA
df.rolling(window=90).mean()['Close'].plot()

df['Close_30MA'] = df['Close'].rolling(window=30).mean()

df[['Close','Close_30MA']].plot(figsize=(12,5))


df['Close'].expanding().mean().plot(figsize=(12,5))

df['Close'].plot();
title = 'SBUX'
y_label = 'Price per Share'
df['Close'].plot(figsize=(12,5),title=title);

ax = df['Close'].plot(figsize=(12,5),title=title)
ax.set(ylabel=y_label)

# LIMITING X-AXIS
df['Close']['2017-01-01':'2017-12-31'].plot(figsize=(12,5))

df['Close'].plot(figsize=(12,5),xlim=['2017-01-01','2017-12-31'],ylim=[51,63])

df[['Close','Close_30MA']].plot(figsize=(12,5),xlim=['2017-01-01','2017-12-31'],ylim=[51,63],color=['k','red'])

from matplotlib import dates

ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57])

# REMOVE PANDAS DEFAULT "Date" LABEL
ax.set(xlabel='')

# SET THE TICK LOCATOR AND FORMATTER FOR THE MAJOR AXIS
ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))

# FORMAT DATE TO MMM DD YYYY
#ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d %Y'))
ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))

ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter('\n\n%b'))

ax.xaxis.grid(True)
ax.yaxis.grid(True)






file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\UMTMVS.csv'
df = pd.read_csv(file1)
df.index

df.set_index('DATE',inplace=True)
df.index


df.index = pd.to_datetime(df.index)
df.index

df = pd.read_csv(file1,index_col='DATE',parse_dates=True)

((df['UMTMVS'].loc['2019-01-01'] / df['UMTMVS'].loc['2009-01-01']) - 1) * 100
((df.loc['2019-01-01'] / df.loc['2009-01-01']) - 1) * 100

df['UMTMVS'].loc['2019-01-01'] # INT
df.loc['2019-01-01'] # SERIES


yearly_data = df.resample('Y').mean()
yearly_data_shift = yearly_data.shift(1)
change = yearly_data - yearly_data_shift
change.idxmax()










import random
lat = [random.uniform(30, 50) for i in range(100)]
lon = [random.uniform(-130, -100) for i in range(100)]

loc_df = pd.DataFrame([lat, lon]).T
loc_df.columns = ['lat', 'lon']

# Sort loc_df by lat/lon
loc_df = loc_df.sort_values(['lat', 'lon'])




data1 = pd.read_csv(file1)
X= data1.iloc[:, :0].values
Y=data1.iloc[:, 1].values
data1.head()
type(X)

n_samples = 27

from sklearn.model_selection import train_test_split

test_pct = 1 - (n_samples / len(loc_df))


X = loc_df.iloc[:,0]
Y = loc_df.iloc[:,1]

X_sample, X_remain, Y_sample, Y_remain = train_test_split( X, Y, test_size=test_pct, random_state=0)

sample_df = X_sample.to_frame().join(Y_sample).reset_index(drop=True)
