# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:14:34 2020

@author: Damon
"""

#import os
import numpy as np
import pandas as pd
#from datetime import datetime
from statsmodels.tsa.ar_model import AR,ARResults
import warnings

warnings.filterwarnings('ignore')

file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\uspopulation.csv'

df = pd.read_csv(file1)
df = pd.read_csv(file1,index_col='DATE',parse_dates=True)

df.columns
df.plot()
df.plot(ylim=[0,350000])

df = df.dropna()

df.index

df.index.freq='MS'

df.head()

len(df)

train = df.iloc[:84]
test = df.iloc[84:]

# DEFINE MODEL
model = AR(train['PopEst'])


# 1st ORDER AR MODEL
AR1fit = model.fit(maxlag=1)

AR1fit.k_ar

AR1fit.params

start = len(train)
end = start + len(test) - 1

AR1fit.predict(start=start,end=end)

pred1 = AR1fit.predict(start=start,end=end)
pred1 = pred1.rename('AR1_pred')

test.plot(figsize=(12,5),legend=True)
pred1.plot(legend=True)


# 2nd ORDER AR MODEL
AR2fit = model.fit(maxlag=2)
AR2fit.params

AR2fit.predict(start=start,end=end)

pred2 = AR2fit.predict(start=start,end=end)
pred2 = pred2.rename('AR2_pred')

test.plot(figsize=(12,5),legend=True)
pred1.plot(legend=True)
pred2.plot(legend=True)


# BEST 'P' VALUE (AR Order)
ARfit = model.fit(maxlag=None,ic='t-stat')
ARfit.params   # ARfit chose 8th order

ARfit.predict(start=start,end=end)

pred8 = ARfit.predict(start=start,end=end)
pred8 = pred8.rename('AR8_pred')


from sklearn.metrics import mean_squared_error

labels = ['AR1','AR2','AR8']

preds = [pred1,pred2,pred8]

for i in range(3):
    error = mean_squared_error(test['PopEst'], preds[i])
    print(f'{labels[i]} MSE was:{error}')


test.plot(figsize=(12,5),legend=True)
pred1.plot(legend=True)
pred2.plot(legend=True)
pred8.plot(legend=True)


# FORECASTING

model2 = AR(df['PopEst'])
ARfit2 = model2.fit(maxlag=None,ic='t-stat')
ARfit2.params   # ARfit chose 8th order

forecast2 = ARfit2.predict(start=len(df),end=len(df)+24)

df['PopEst'].plot(figsize=(12,8),legend=True)
forecast2.plot(legend=True)

