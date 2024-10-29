# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 08:13:38 2024

@author: OM
"""

"""
python for data science:
    pandas=series,coloums,one colums;matplot ib;numpy;seaborn"""
    
#a series is used to model one dimentional data,
#similar to a list in python
#the series object also has a few more bits
#of data, including an #index and a name

import pandas as pd
songs2=pd.Series([145,142,38,13],name='count')
#it is easy to inspect the index of series(or data for)
songs2.index
songs2
#the index can be string based as well,
#in which case pandas indicates
#that the datatype for the index is object (not string)
songs3=pd.Series([145,142,38,13],name='count',
                 index=['Paul','John','George','Ringo'])
songs3.index
songs3
#The NaN value
#operations.(similar to name in SQL)
#if you load data from a csv file
import pandas as pd
f1=pd.read_csv('age.csv')
f1


import pandas as pd
df=pd.read_excel("C:/1-python/Bahaman.excel")
df



#None, NaN, nan, and null are synonyms
#The series object behaves similarly to
#a NumPy array
import numpy as np
numpy_ser=np.array([145,142,38,13])
songs3[1]
#142
numpy_ser[1]
#they both have method in common
songs3.mean()
numpy_ser.mean()
#################################
#the pandas series data structure provides
#support for the basic crud
#operations-create,read,update and delete
#creation
import pandas as pd
george=pd.Series([10,7,1,22],
index=['1968','1969','1970','1970'],
name='George_songs')
george
#The previous example illustrates an
#intresting features of pandas-the
#index values are string and they
#are not unique. this can cause some 
#confusion, but can also useful 
#when duplicate index items are needed
############################
#reading
#to read or select the data from series
george['1968']
george['1970']
#we can iterate over data in series
#as well. when iterating over a series 
for item in george:
    print(item)
###########################
#updating
#updating value in a series can be a little tricky as well
#to update a value
#for a given index label
#the standard index assignment operation
#works
george['1969']=68
george['1969']
george
#Deletion
#the del statement appears to have
#problems with duplicate index
s=pd.Series([2,3,4],index=[1,2,3])
del s[1]
s
#convert types


#string use.astype(str)
#numeric use pd.to_numeric
#integer use .astype(int),
#note that this will fail with NaN
#datetime use pd.to_datetime
songs_66=pd.Series([3,None,11,9],
index=['George','Ringo','John','Paul'],
name='counts')
songs_66.dtypes
#dtype('float64')
pd.to_numeric(songs_66.apply(str))
#there will be error
pd.to_numeric(songs_66.astype(str),error='coerce')
#if we pass error='coerce',
#we can see that it supports many formats
songs_66.dtypes
#dealing with none
#the .fillna method will replace them with a given value,
import pandas as pd
songs_66=pd.Series([3,None,11,9],
index=['George','Ringo','John','Paul'],
name='counts')
songs_66.dtypes
songs_66=songs_66.fillna(-1)
songs_66=songs_66.astype(int)
songs_66.dtypes
songs_66
#NaN values can be dropped from
#the series using.dropna
songs_66=pd.Series([3,None,11,9],
index=['George','Ringo','John','Paul'],
name='counts')
songs_66=songs_66.dropna()
songs_66
############################
#append,combining, and joining two series
songs_69=pd.Series([7,16,21,39],
index=['Ram','Sham','Ghansham','Krishna'],
name='Counts')
#to concatenate two series together, simply use the .append
songs=pd.concat([songs_66,songs_69])
songs
#####################################
#plotting series
#plotting of line graph 
import matplotlib.pyplot as plt
fig = plt.figure()
songs_69.plot()
plt.legend()
######################
#songs it is the handle of series
#plotting of bar graps
fig=plt.figure()
songs_69.plot(kind='bar')
songs_66.plot(kind='bar',color='r')
plt.legend()
##################################
"""
this is output which is called object
<matplotlib.legend.Legend at 0x2164c1eea50>"""
##############
#histograph
import numpy as np
#name of the series is data
data=pd.Series(np.random.randn(500),name='500_random')
fig=plt.figure()
ax=fig.add_subplot(111)
data.hist()
#########################



    
