# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 08:30:18 2020

@author: Damon
"""

import numpy as np
import pandas as pd
#from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error,mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# REPLICATE FOR PFM VOLUME   LINES 19-70  AND 140-
# REMEMBER => DICKEY-FULLER TEST FOR SEASONALITY
# REVIEW 05-ARMA-and-ARIMA.ipynb WITH PFM VOLUME DATA

file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\airline_passengers.csv'

#df = pd.read_csv(file1)
df = pd.read_csv(file1,index_col=0,parse_dates=True)

df.rename(columns={'Thousands of Passengers':'Pass_K'},inplace=True)

df.columns

df.plot()
df.plot(ylim=[0,350000])


if df.isnull().values.any():
    df = df.dropna(inplace=True)


df.index

df.index.freq='MS'

df.head()
df.tail()

len(df)

train_df = df.iloc[:109]
test_df = df.iloc[108:]

fitted_model = ExponentialSmoothing(train_df['Pass_K'],trend='mul',seasonal='mul',seasonal_periods=12).fit()

test_predictions = fitted_model.forecast(36)

train_df['Pass_K'].plot(legend=True,label='Train',figsize=(12,8))
test_df['Pass_K'].plot(legend=True,label='Test')
test_predictions.plot(legend=True,label='Pred',xlim=['1958-01-01','1961-01-01'],ylim=[250,650])

test_df.describe()

mean_absolute_error(test_df,test_predictions)
np.sqrt(mean_squared_error(test_df,test_predictions))

final_df = ExponentialSmoothing(df['Pass_K'],trend='mul',seasonal='mul',seasonal_periods=12).fit()

final_pred = final_df.forecast(36)

df['Pass_K'].plot(legend=True,label='Actual',figsize=(12,8))
final_pred.plot(legend=True,label='Pred')
test_predictions.plot(legend=True,label='TestPred')






file2 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\samples.csv'

#df = pd.read_csv(file1)
df = pd.read_csv(file2,index_col=0,parse_dates=True)


df.columns

df['a'].plot()
df['b'].plot()


if df.isnull().values.any():
    df = df.dropna(inplace=True)


df.index

df.index.freq='MS'

df.head()
df.tail()

len(df)



from statsmodels.tsa.statespace.tools import diff


df['b'] - df['b'].shift(1)

diff(df['b'],k_diff=1).plot()


# ACF and PACF

import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols

file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\airline_passengers.csv'
df1 = pd.read_csv(file1,index_col=0,parse_dates=True)
df1.rename(columns={'Thousands of Passengers':'Pass_K'},inplace=True)
df1.index.freq='MS'


file2 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\DailyTotalFemaleBirths.csv'
df2 = pd.read_csv(file2,index_col='Date',parse_dates=True)
df2.index.freq='D'


df1.head()
df2.head()


df = pd.DataFrame({'a':[13,5,11,12,9]})
acf(df['a'])
pacf_yw(df['a'],nlags=4,method='mle')
pacf_ols(df['a'],nlags=4)

from pandas.plotting import lag_plot
lag_plot(df1['Pass_K'])     # STRONG CORRELATION

lag_plot(df2['Births'])     # WEAK/NO CORRELATION





from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_acf(df1,lags=40);
plot_acf(df2,lags=40);

plot_pacf(df2,lags=40,title='PACF BIRTHS');



plot_pacf(df1,lags=40,title='PACF Passengers')
df1['d1'] = diff(df1['Pass_K'],k_diff=1)
df1['d1'].plot(figsize=(12,5));

title = 'PACF Pass_K 1st Diff'
lags=40
plot_pacf(df1['d1'].dropna(),title=title,lags=np.arange(lags));


# ARIMA

from pmdarima import auto_arima

help(auto_arima)

stepwise_fit = auto_arima(df2['Births'],start_p=0,start_q=0,max_p=6,max_q=3,seasonal=False,trace=True)
stepwise_fit.summary()
# ARIMA Model output is ARIMA(1,1,1) for df2

stepwise_fit = auto_arima(df1['Pass_K'],start_p=0,start_q=0,max_p=6,max_q=6,seasonal=True,trace=True,m=4)
# TRY m=12 (Monthly) and m=4 (Quarterly)
stepwise_fit.summary()
# ARIMA Model output is SARIMAX(2, 1, 2)x(0, 0, 1, 12) for df1 (m=12) / 1267.601
# ARIMA Model output is SARIMAX(2, 1, 2)x(2, 0, 2, 4) for df1 (m=4) / 1205.174


from statsmodels.tsa.arima_model import ARMA,ARIMA,ARMAResults,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

df2 = df2[:120]

#ARMA MODEL   => No 'I'

df2.plot(figsize=(12,5))
df1.plot(figsize=(12,5))

from statsmodels.tsa.stattools import adfuller

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

auto_arima(df2['Births'],seasonal=False).summary()


train2 = df2.iloc[:90]
test2 = df2.iloc[90:]

model2 = ARMA(train2['Births'],order=(2,2))
results2 = model2.fit()
results2.summary()

start2 = len(train2)
end2 = len(train2) + len(test2) - 1

pred2 = results2.predict(start2,end2).rename('ARMA (2,2) Predictions')

test2['Births'].plot(figsize=(12,8),legend=True)
pred2.plot(legend=True)

test2.mean()
pred2.mean()





#ARIMA MODEL   

file3 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\Data\TradeInventories.csv'
#df3 = pd.read_csv(file3)
df3 = pd.read_csv(file3,index_col=0,parse_dates=True)

df3.rename(columns={'Inventories':'Inv'},inplace=True)

df3.columns

df3.plot()

if df3.isnull().values.any():
    df3 = df.dropna(inplace=True)

df3.index

df3.index.freq='MS'

df3.head()
df3.tail()

len(df3)


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df3['Inv'],model='add')
result.plot();

df3['Diff1'] = diff(df3['Inv'],k_diff=1)

adf_test(df3['Diff1'])

plot_acf(df3['Inv'],lags=40);
plot_pacf(df3['Inv'],lags=40);

stepwise_fit = auto_arima(df3['Inv'],start_p=0,start_q=0,seasonal=False,max_p=2,max_q=2,trace=True)
stepwise_fit.summary()


train3 = df3.iloc[:252,0:1]
test3 = df3.iloc[252:,0:1]

model3 = ARIMA(train3['Inv'],order=(1,1,1))
results3 = model3.fit()
results3.summary()

start3 = len(train3)
end3 = len(train3) + len(test3) - 1

pred3 = results3.predict(start3,end3,typ='levels').rename('ARIMA (1,1,1) Predictions')

test3['Inv'].plot(figsize=(12,8),legend=True)
pred3.plot(legend=True)
#train3['Inv'].plot(legend=True)


test3['Inv'].mean()
pred3.mean()

from statsmodels.tools.eval_measures import rmse

error = rmse(test3['Inv'],pred3)


# FORECAST FUTURE TIME PERIODS

model3F = ARIMA(df3['Inv'],order=(1,1,1))
results3F = model3F.fit()

predict3 = results3F.predict(start=len(df3),end=len(df3)+12,typ='levels').rename('ARIMA (1,1,1) Forecast')

df3['Inv'].plot(figsize=(12,8),legend=True)
predict3.plot(legend=True)
type(predict3)


help(ARIMAResults.forecast)
#forecast3a = results3F.forecast(steps=12,alpha=0.05)
forecast3a = results3F.forecast(steps=12,alpha=0.25)
len(forecast3a)
type(forecast3a)
type(forecast3a[0])  # predictions
type(forecast3a[1]) # std errors
type(forecast3a[2]) # range
forecast3a[2].shape

print(forecast3a)


f_cast = pd.DataFrame(data={'pred':forecast3a[0],'std_err':forecast3a[1]},index=predict3.index[:12])
range_df = pd.DataFrame(data=forecast3a[2],index=predict3.index[:12],columns=['LowEst','HighEst'])

f_cast3 = pd.concat([f_cast,range_df],axis=1)
f_cast3.columns

f_cast3[['pred','LowEst','HighEst']].plot(figsize=(12,8))

df3['Inv'].plot(figsize=(12,8),legend=True)
f_cast3[['pred','LowEst','HighEst']].plot(legend=True)

df3a = df3.copy()
df3b = pd.concat([df3a,f_cast3],axis=0)

df3b.columns

df3b[['HighEst', 'Inv', 'LowEst', 'pred']].plot(figsize=(12,8))


