# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 07:19:24 2020

@author: Damon
"""

import numpy as np
import pandas as pd
#from datetime import datetime
#from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import warnings

warnings.filterwarnings('ignore')

# REPLICATE FOR PFM VOLUME   


file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\co2_mm_mlo.csv'

df = pd.read_csv(file1)
df['Date'] = pd.to_datetime({'year':df['year'],'month':df['month'],'day':1})

df.info()

df = df.set_index('Date')

df.index.freq='MS'

df.plot()
df.plot(ylim=[0,350000])


if df.isnull().values.any():
    df = df.dropna(inplace=True)


df.head()
df.tail()


df['interpolated'].plot(figsize=(12,8))

result = seasonal_decompose(df['interpolated'],model='mult')
result = seasonal_decompose(df['interpolated'],model='add')
result.plot();


auto_arima(df['interpolated'],seasonal=True,m=12).summary()
# MY RESULT IS SARIMAX(0, 1, 3)x(1, 0, 1, 12)
# CLASS RESULT IS SARIMAX(0, 1, 1)x(1, 0, 1, 12)


len(df)


train_df = df.iloc[:717]
test_df = df.iloc[717:]

model = SARIMAX(train_df['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
results.summary()


start = len(train_df)
end = len(train_df) + len(test_df) - 1

predict = results.predict(start,end,typ='levels').rename('SARIMA Pred')

test_df['interpolated'].plot(legend=True,figsize=(12,5))
predict.plot(legend=True)

from statsmodels.tools.eval_measures import rmse

error = rmse(test_df['interpolated'],predict)
test_df['interpolated'].mean()

# REAL PREDICTION / FORECAST INTO FUTURE

model = SARIMAX(df['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
results.summary()


predict = results.predict(start=len(df),end=len(df)+11,typ='levels').rename('SARIMA_Pred')

df['interpolated'].plot(legend=True,figsize=(12,8))
predict.plot(legend=True)
len(predict)
type(predict)
predict.index


f_cast_df = predict.to_frame()

f_cast = results.get_forecast(steps=12,alpha=0.05)
f_cast.conf_int()

f_cast_df = pd.concat([f_cast_df,f_cast.conf_int()],axis=1)
f_cast_df.columns


f_cast_df.rename(columns={'lower interpolated':'LowEst','upper interpolated':'HighEst'},inplace=True)



f_cast_df.plot(figsize=(12,8))


dfa = df.copy()
dfb = pd.concat([dfa,f_cast_df],axis=0)

dfb.columns
dfb.index[-1]

dfb[['HighEst', 'interpolated', 'LowEst', 'SARIMA_Pred']].plot(figsize=(12,8),xlim=[dfb.index[-36],dfb.index[-1]])


f_cast_df.plot(legend=True)





