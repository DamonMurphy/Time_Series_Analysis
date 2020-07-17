# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:19:03 2020

@author: Damon
"""

import numpy as np
import pandas as pd
#from datetime import datetime
from statsmodels.tsa.ar_model import AR,ARResults
import warnings

warnings.filterwarnings('ignore')

file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\airline_passengers.csv'
file2 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\DailyTotalFemaleBirths.csv'

df1 = pd.read_csv(file1)
df1 = pd.read_csv(file1,index_col='Month',parse_dates=True)
df1.head()

df2 = pd.read_csv(file2)
df2 = pd.read_csv(file2,index_col='Date',parse_dates=True)
df2.head()

df1.columns
df1.rename(columns={'Thousands of Passengers':'Pass_K'},inplace=True)
df1.plot()
df1.plot(ylim=[0,350000])

df1.isnull().values.any()
df1 = df1.dropna()

df1.index
df1.index.freq='MS'



df2.columns
df2.head()
df2.plot()

df2.isnull().values.any()
df2 = df2.dropna()


df2.index
df2.index.freq='D'

from statsmodels.tsa.stattools import adfuller

adfuller(df1['Pass_K'])
help(adfuller)

dftest = adfuller(df1['Pass_K'],autolag='AIC')
dfout = pd.Series(dftest[0:4],index=['adf_test_statistic','p-value','usedlag','nobs'])

for key,val in dftest[4].items():
    dfout[f'critical value ({key})']=val
print(dfout)


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


adf_test(df1['Pass_K'])

adf_test(df2['Births'])



file3 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\samples2.csv'


df3 = pd.read_csv(file3)
df3 = pd.read_csv(file3,index_col=0,parse_dates=True)
df3.head()


df3.columns

df3.plot()
df3[['a','b','d']].plot()

df3.isnull().values.any()
#df3 = df3.dropna()

df3.index
df3.index.freq='MS'

df3['a'].iloc[2:].plot(figsize=(12,8),legend=True)
df3['d'].shift(2).plot(legend=True)

from statsmodels.tsa.stattools import grangercausalitytests

grangercausalitytests(df3[['a','d']],maxlag=5);
grangercausalitytests(df3[['b','d']],maxlag=5);


np.random.seed(42)

df = pd.DataFrame(np.random.randint(20,30,(50,2)),columns=['test','predictions'])

df.head()

df.plot(figsize=(12,8))

from statsmodels.tools.eval_measures import mse,rmse,meanabs

mse(df['test'],df['predictions'])
rmse(df['test'],df['predictions'])
meanabs(df['test'],df['predictions'])



df1.head()
df1.index

from statsmodels.graphics.tsaplots import month_plot,quarter_plot

month_plot(df1['Pass_K']);

df1q = df1['Pass_K'].resample(rule='Q').sum()
quarter_plot(df1q);



