# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:21:08 2024

@author: OM
"""
#first we write all descreption all types of data dictionary from YT
#and understand bussiness logic of data not learn only its method
#pandas dataframe - basic operations
#create dataframe with None/Null to work with examples

import pandas as pd
import numpy as np
technologies=({
    'Courses':["Spark","Pyspark","Hadoop","Python","Pandas",None,"Spark","Python"],
    'Fee':[20000,25000,23000,24000,np.nan,25000,25000,22000],
    'Duration':["30days","50days","55days","40days","60days","35days","50days","40days"],
    'Discount':[1000,2300,1000,1200,2500,1300,1400,1600]})
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df=pd.DataFrame(technologies, index=row_labels)
print(df)
#################################
#Dataframe properties
df.shape
#(8,4)#(row,column)
df.size
#32
df.columns
df.columns.values
df.index
df.dtypes
df.info
##########################
#accessing one column contents
df['Fee']
#accessing two column contents
df[['Fee','Duration']]
#select certain row and assign it to another dataframe
df2=df[6:]
df2=df[:6]
df2
#######################
#accessing certain cell from column 'Duration'
#accesing certain cell from column 'Duration'
df['Duration'][3]
#substrating specific value from a column
df['Fee'] = df['Fee']-500
df['Fee']
#####################
#pandas to mainpulate DataFrame
#Describe DataFrame
#describe DtaFrame for all numeric columns
df.describe()
#it will show 5 number summary
##########################
#rename() -Rename panadas dataframe columns
df = pd.DataFrame(technologies, index=row_labels)

#assign new header by setting new column names.
df.columns=['A','B','C','D']
df
#######################
#Rename Column Name using rename() method
#axis=0 is row and axis=1 is column
df.columns=['A','B','C','D']
df2=df.rename({'A':'c1','B':'c2'},axis=1)
df2=df.rename({'C':'c3','D':'c4'},axis='columns')
df2=df.rename(columns={'A':'c1','B':'c2'})
df
#################################
#drop rows by labels
df1 = pd.DataFrame(technologies, index=row_labels)
#drops rows by labels
df1=df.drop['r1','r2']
df1
#delete rows by position/index
df1=df.drop(df.index[1])
df1
df1=df.drop(df.index[1,3])
df1
#delete rows by index range
df1=df.drop(df.index[2:])
df1
#when you have default indexes for rows
df=pd.DataFrame(technologies)
df1=df.drop(0)
df1
df=pd.DataFrame(technologies)
df1=df.drop([0,3],axis=0)#it will delete row0 & row3
df1
df1=df.drop(range(0,2))
