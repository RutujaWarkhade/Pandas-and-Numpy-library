# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 08:12:26 2025

@author: om
"""

#np.nan, None and ''
import pandas as pd
import numpy as np
technologies={
    'Categories':["Spark","Pyspark","Hadoop","Python"],
    'Fee':[20000,25000,27000,22000],
    'Duration':['','40days',np.nan,None],
    'Discount':[1000,1500,1200,1700] 
    }
indexes=['r1','r2','r3','r4']
df=pd.DataFrame(technologies,index=indexes)
print(df)
df=pd.DataFrame(technologies,index=indexes)
df2=df.dropna() ##dropping np.nan and None
print(df2)


#drop nan and None containing rows
df_clean=df.dropna(subset=['Duration'])

#removing empty strings containing rows as well
df_clean=df_clean[df_clean['Duration']!='']


#############################################
#Converting all the columns into same data type using astype

df=df.astype(str)
df.dtypes

#now converting some specific columni into integer and some to float
df=df.astype({"Fee":int,"Discount":float})
print(df.dtypes)

#doing the same conversionn using a list
cols=["Fee","Discount"]
df=pd.DataFrame(technologies)
df[cols]=df[cols].astype('float') #u are giving the data type with ''
df.dtypes

"""But Python (specifically, pandas) is designed to interpret that
 string as the name of a data type.
 'float' is a string → pandas internally
 looks it up in a type mapping dictionary.
It knows 'float' → maps to Python's float class (float).
So pandas reads it as: "Convert this column to type float."
"""
#by using for loop
for col in ['Fee','Discount']:
    df[col]=df[col].astype('float')
    
    
#Raising or ignoring error while conversion of column is failed
df=df.astype({"Categories":int},errors='ignore')
df.dtypes

df=df.astype({"Categories":int},errors='raise')

####################################################
#using DataFrame.to_numeric() to convert data to numeric
df["Fee"]=pd.to_numeric(df["Fee"])
df.dtypes
##
#converting multiple to numeric type using apply method
df=pd.DataFrame(technologies)
df.dtypes
df[['Fee','Discount']]=df[['Fee','Discount']].apply(pd.to_numeric)
df.dtypes

#quic example to get number of rows in Data frame
rows_count=len(df.index)
row_count=len()

#Using DataFrame.apply to apply somwthing to column
import pandas as pd
import numpy as np
data=[(3,5,7),(2,4,6),(5,8,9)]
df=pd.DataFrame(data,columns=['A','B','C'])


#adding 3 into all cells using apply()
def add_3(x):
    return x+3
df2=df.apply(add_3)

#using apply() modify only single column
import pandas as pd
df=pd.DataFrame(data,columns=['A','B','C'])
def add_4(x):
    return x+4
df['B']=df['B'].apply(add_4)

#for multiple columns
#using lambda function

df["A"]=df["A"].apply(lambda x:x-2)
df

#using panads.dataframe.transform
df
def add_2(x):
    return x+2 
df=df.transform(add_2)

#using map function
df['A']=df['A'].map(lambda A:A/2)
print(df)

"""
why it becomes int 32?
Pandas internally chooses the most efficient
platform dependent numpy dtypes that corresponds
to python's int. This depends on your operating 
system and python/numpy version:
On 32-bit system int usually maps to int32
on 64-bit system int maps to int64

"""
#you can explicitly control the dtypes like this
import pandas as pd
import numpy as np
technologies={
    'Categories':["Spark","Pyspark","Hadoop","Python"],
    'Fee':[20000,25000,27000,22000],
    'Duration':['','40days',np.nan,None],
    'Discount':[1000,1500,1200,1700] 
    }
indexes=['r1','r2','r3','r4']
df=pd.DataFrame(technologies,index=indexes)
print(df)
df.dtypes
df = df.astype({"Fee":"int64", "Discount":"float64"})
df.dtypes
############################