# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:26:41 2020

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

file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\RestaurantVisitors.csv'

df = pd.read_csv(file1)
df.head()
df.tail()


df = pd.read_csv(file1,index_col=0,parse_dates=True)
df.index.freq='D'

df.columns

df.plot()

df1 = df.dropna()

df1.head()
df1.tail()


df1.columns

cols = ['rest1', 'rest2', 'rest3','rest4', 'total']

for col in cols:
    df1[col] = df1[col].astype(int)


df1['total'].plot(figsize=(12,5))


df1[df1['holiday']==1].index == df1.query('holiday==1').index

ax = df1['total'].plot(figsize=(12,5))
for day in df1.query('holiday==1').index:
    ax.axvline(x=day,color='black',alpha=0.8);


result = seasonal_decompose(df1['total'],model='add')
result.plot();

result.seasonal.plot(figsize=(12,5))    # SEASONALITY APPEARS TO BE WEEKLY



auto_arima(df1['total'],seasonal=True,m=7).summary()
# RESULT IS  SARIMAX(1, 0, 0)x(2, 0, 0, 7)


len(df1)

train = df1.iloc[:436]
test = df1.iloc[436:]

model = SARIMAX(train['total'],order=(1,0,0),seasonal_order=(2,0,0,7),enforce_invertibility=False)
results = model.fit()
results.summary()

start = len(train)
end = len(train) + len(test) - 1

predict = results.predict(start,end,typ='levels').rename('SARIMA_Pred')

test['total'].plot(legend=True,figsize=(12,5))
predict.plot(legend=True)

from statsmodels.tools.eval_measures import rmse

error1 = rmse(test['total'],predict)
test['total'].mean()

ax = test['total'].plot(legend=True,figsize=(12,5))
predict.plot(legend=True)
for day in test.query('holiday==1').index:
    ax.axvline(x=day,color='black',alpha=0.8);


# ADD EXOGENOUS DATA

auto_arima(df1['total'],exogenous=df1[['holiday']],seasonal=True,m=7).summary()
# RESULT IS  SARIMAX(0, 0, 1)x(2, 0, 0, 7)
# CLASS RESULT IS  SARIMAX(1, 0, 1)x(1, 0, 1, 7)


len(df1)

train = df1.iloc[:436]
test = df1.iloc[436:]

model = SARIMAX(train['total'],exog=train[['holiday']],order=(0,0,1),seasonal_order=(2,0,0,7),enforce_invertibility=False)
results = model.fit()
results.summary()

start = len(train)
end = len(train) + len(test) - 1

predict2 = results.predict(start,end,exog=test[['holiday']],typ='levels').rename('SARIMA_Pred2')

test['total'].plot(legend=True,figsize=(12,5))
predict.plot(legend=True)
predict2.plot(legend=True)


error2 = rmse(test['total'],predict2)
test['total'].mean()
print('error1:',error1)
print('error2:',error2)



#model = SARIMAX(train['total'],exog=train[['holiday']],order=(1,0,1),seasonal_order=(1,0,1,7),enforce_invertibility=False)
model = SARIMAX(train['total'],exog=train[['holiday']],order=(1,0,1),seasonal_order=(1,0,1,7))
results = model.fit()
results.summary()

start = len(train)
end = len(train) + len(test) - 1

predict3 = results.predict(start,end,exog=test[['holiday']],typ='levels').rename('SARIMA_Pred3')

test['total'].plot(legend=True,figsize=(12,5))
#predict.plot(legend=True)
#predict2.plot(legend=True)
predict3.plot(legend=True)


error3 = rmse(test['total'],predict3)
test['total'].mean()
print('error1:',error1)
print('error2:',error2)
print('error3:',error3)


model = SARIMAX(df1['total'],exog=df1[['holiday']],order=(1,0,1),seasonal_order=(1,0,1,7))
results = model.fit()
results.summary()

exog_forecast = df[478:][['holiday']]
predict_final = results.predict(len(df1),len(df1)+38,exog=exog_forecast,typ='levels').rename('SARIMA_Pred_Final')

df1['total'].plot(figsize=(12,5),legend=True)
predict_final.plot(legend=True)



ax = df1['total'].loc['2017-03-01':].plot(figsize=(12,5),legend=True,xlim=('2017-03-01',predict_final.index[-1]))
predict_final.plot(legend=True)
for day in df.query('holiday==1').index:
    ax.axvline(x=day,color='black',alpha=0.8);



