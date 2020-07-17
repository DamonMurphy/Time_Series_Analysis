# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:01:13 2020

@author: Damon
"""

import numpy as np
from numpy.random import randn
import pandas as pd
import os

labels = ['a','b','c']

list1 = [10,20,30]

arr = np.array(list1)

d = {'a':10,'b':20,'c':30}

ser1 = pd.Series(data=list1)

ser2 = pd.Series(data=list1,index=labels)


#ALL OF THESE ARE THE SAME
ser2['c']
ser2[2]
ser2.c


np.random.seed(101)

rand_mat = randn(5,4)

rand_df = pd.DataFrame(data=rand_mat,index='A B C D E'.split(),columns='W X Y Z'.split())

rand_df['W']

list1 = ['W','Y']
rand_df[['W','Y']]
rand_df[list1]

rand_df['New'] = rand_df['W'] + rand_df['Y']

df = rand_df.copy()

df.drop(labels='New',axis=1)

df.drop(labels='New',axis=1,inplace=True)

# ROWS
df.loc['A']
df.iloc[0]
df.loc[['A','C']]
df.iloc[[0,2]]

# 2-D SELECTIONS  (ROWS, COLUMNS)
df.loc[['A','C'],['X','Z']]

df > 0

df_bool = df[df > 0]

df[df['W']>0]
df[df['W']>0]['Y']
df[df['W']>0][['Y','Z']]

cond1 = df['W']>0
cond2 = df['Y']>1

df[(cond1)&(cond2)]
df[(df['W']>0)&(df['Y']>1)]

row_labels = 'CA NY WY OR CO'.split()

df['States'] = row_labels

df.set_index('States',inplace=True)



# MISSING DATA
df = pd.DataFrame({'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]})

df.dropna()
df.dropna(axis=1)

df.dropna(thresh=2)

df.fillna(df.mean())
df['A'].fillna(value=df['A'].mean())

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
        'CompName':['Google','Google','Microsoft','Microsoft','Facebook','Facebook'],
        'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
        'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data)

df.groupby(['Company']).mean()

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

df['col2'].unique()
df['col2'].nunique()

df['col2'].value_counts()

new_df = df[(df['col1']>2)&(df['col2']==444)]

def times2(number):
    return number * 2

times2(4)

df['new'] = df['col1'].apply(times2)

df

del df['new']

df.columns

df.sort_values(['col2','col3'],ascending=[False,True])

os.chdir(r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\02-Pandas')


df = pd.read_csv('example.csv')

df1 = pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
df1.columns
df1.drop('Unnamed: 0',axis=1,inplace=True)

df2 = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
# BRINGS IN A LIST OF TABLES (EVEN THOUGH THERE IS ONLY 1 TABLE)
len(df2)

df2 = df2[0]

df2.columns

df3 = pd.read_html('https://www.baseball-reference.com/play-index/inning_summary.cgi?request=1&year=2019&team_id=ANY')

len(df3)

i = 2019
final_df = pd.DataFrame()


for i in range(2010,2020):
    url = 'https://www.baseball-reference.com/play-index/inning_summary.cgi?request=1&year=' + str(i) + '&team_id=ANY&utm_source=direct&utm_medium=Share&utm_campaign=ShareTool#runs_allowed_by_inning::none'
    df3 = pd.read_html(url)
    df3 = df3[0]

i = 2018
df4 = df4[0]
df3 + df4

'https://www.baseball-reference.com/play-index/inning_summary.cgi?request=1&year=' + str(i) + '&team_id=ANY&utm_source=direct&utm_medium=Share&utm_campaign=ShareTool#runs_allowed_by_inning::none'






