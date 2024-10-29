# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 08:27:31 2024

@author: OM
"""

import pandas as pd
technologies={
    'Courses':["Spark","Pyspark","Hadoop","Python","Pandas","Oracel","Java"],
    'Fee':[20000,25000,26000,22000,24000,21000,22000],
    'Duration':["30days","40days","35days","40days","60days","50days","55days"],}
df = pd.DataFrame(technologies)
df.columns
print(df.columns)
#Quick example of get the number of rows in dataframe
rows_count=len(df.index)
rows_count
rows_count=len(df.axes[0])
rows_count
column_count=len(df.axes[1])
column_count
######################################
df=pd.DataFrame(technologies)
row_count=df.shape[0]#return number of rows
row_count
col_count=df.shape[1]#return number of columns
print(row_count)
print(column_count)
#outputs-4
###################################
#using DataFrame.apply() to apply function add column
import pandas as pd
import numpy as np
data={"A":[1,2,3],
      "B":[4,5,6],
      "C":[7,8,9]
      }
df=pd.DataFrame(data)
print(df)
def add_3(x):
    return x+3
df2=df.apply(add_3)
df2
df2=((df.A).apply(add_3))
#########################
#using apply function single column
def add_4(x):
    return x+4
df["B"]=df["B"].apply(add_4)
df["B"]
#apply to multiple columns
df[['A','B']]=df[['A','B']].apply(add_4)
df
#apply a lambda function to each column
df2=df.apply(lambda x:x+10)
df2
##########################################
#apply lambda function to single column
#using DataFrame.apply() and lambda function
df["A"]=df["A"].apply(lambda x:x-2)
print(df)
##############################
#using pandas.DataFrame.transform() to apply function column
#using DataFrame.transform()
def add_2(x):
    return x+2
df=df.transform(add_2)
print(df)
######################################
#using pandas.DataFrame.map() to single column
df['A']=df['A'].map(lambda A:A/2.)
print(df)
##################################
#using Numpy function on single Column
#using DataFrme.apply() & [] operator
import numpy as np
df['A']=df['A'].apply(np.square)
print(df)
#################################
#using NumPy.square() Method
#using numpy.square() and [] operator
df['A']=np.square(df['A'])
print(df)
#############################################
#Pandas groupby()with examples


import pandas as pd
technologies=({
    'Courses':["Spark","Pyspark","Hadoop","Python","Pandas","Hadoop","Python","NA"],
    'Fee':[22000,25000,23000,24000,25000,25000,22000,15000],
    'Duration':["30days","50days","35days","55days","40days","60days","50days","55days"],
    'Discount':[2300,1000,1200,2500,None,1400,1600,0]})
df=pd.DataFrame(technologies)
print(df) 

#use groupby() to coumpute the sum
df2=df.groupby(['Courses']).sum()
print(df2)   


########################
#Group by multiple columns
df2=df.groupby(['Courses','Duration']).sum()
print(df2)
##############################
#add index to the grouped data
#add row index to the group by result
df2=df.groupby(['Courses','Duration']).sum().reset_index()
print(df2)
################################
#using list(df) to get the list of 
