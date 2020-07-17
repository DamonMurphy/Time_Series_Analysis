# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:09:29 2020

@author: Damon
"""

import os
import numpy as np
import pandas as pd


file1 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\03-Pandas-Visualization\df1.csv'
file2 = r'C:\Damon\Udemy\Python for Time Series Data Analysis\TSA_COURSE_NOTEBOOKS\03-Pandas-Visualization\df2.csv'

df1 = pd.read_csv(file1,index_col=0)
df2 = pd.read_csv(file2)


df1['A'].plot.hist()
df1['A'].plot.hist(edgecolor='k')
df1['A'].plot.hist(edgecolor='k').autoscale(enable=True,axis='both',tight=True)

df1['A'].plot.hist(bins=20,edgecolor='k').autoscale(enable=True,axis='both',tight=True)
df1['A'].plot.hist(bins=20,edgecolor='k',grid=True).autoscale(enable=True,axis='both',tight=True)


df2.plot.bar()
df2.plot.bar(stacked=True)
df2.plot.barh()
df2.plot.barh(stacked=True)


df2.plot.line(y='a')
df2.plot.line(y=['a','b','c'],figsize=(10,4),lw=4)

df2.plot.area()
df2.plot.area(alpha=0.4)   # alpha IS TRANSPARENCY VALUE


# SCATTER PLOT WITH COLOR
df1.plot.scatter(x='A',y='B')
df1.plot.scatter(x='A',y='B',c='D',colormap='coolwarm')   # c paramter is 'color'value

# SCATTER PLOT WITH SIZE
df1.plot.scatter(x='A',y='B',s=df1['C']*df1['C']*10,c='D',edgecolor='k',alpha=0.3,colormap='coolwarm')


# BOX PLOTS
df2.plot.box()
df2.boxplot()



# KERNEL DENSITY ESTIMATION (KDE)
df2['a'].plot.kde()
df2.plot.kde()



df = pd.DataFrame(np.random.randn(1000,2),columns=['a','b'])
df.head()

df.plot.scatter(x='a',y='b')
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='Oranges')



df2['c'].plot.line(figsize=(10,3),ls=':',c='red',lw=4)


title = 'PLOT TITLE'
xlabel = 'X DATA'
ylabel = 'Y DATA'
ax = df2['c'].plot.line(figsize=(10,3),ls=':',c='red',lw=4,title=title)
ax.set(xlabel=xlabel,ylabel=ylabel)


# TO  PLACE LEGEND
ax = df2.plot()
ax.legend(loc=4)   # 4 is lower right

# PLACING LEGEN OUTSIDE PLOT
ax = df2.plot()
ax.legend(loc=4,bbox_to_anchor=(1.2,0.0))


