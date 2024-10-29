# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:12:18 2024

@author: OM
"""

#what is pandas dataframe?
#pandas dataframe is two dimensional data structure
#an immutable, hetrogeneous tabular
#data structure with labeled axes rows,  and columns.

#dataframe features
#dataframe support name

#step-1 go to anaconda navigator
#step-2 select environvent tap
#step-3 by default it will be base terminal 
#step-4 on base terminal-pip install pandas 
#or conda install pandas 
################################
#upgrade pandas to latest or specific version
#on base terminal write
#conda install -c anaconda pandas

#upgrade to specific version
#conda install pandas==2.2.1
###########################
#to ceck the version of pandas 
import pandas as pd
pd.__version__
###################
#create using construter
#crreate one DataFrame from list
import pandas as pd
technologies=[["Spark",20000,"30days"],
              ["Pandas",2000,"40days"]]
df=pd.DataFrame(technologies)
print(df)

#since we have not given labels to columns and
#indexes, DataFrame by default assigns
#incremental sequence number as labels
#to both rows and columns ,these are called index
#add column and row label to the DataFrame
column_names=["Courses","Free","Duration"]
row_label=["a","b"]
df=pd.DataFrame(technologies,columns=column_names,index=row_label)
print(df)
#################################
df.dtypes
########################
#you can also assign custom
#data types to columns
#set custom types to DataFrame
#all array must me same length otherwise it gives an error
import pandas as pd
technologies={
    'Courses':["Spark","Pyspark","Hadoop","Python","Pandas","Oracel","Java"],
    'Fee':[20000,25000,26000,22000,24000,21000,22000],
    'Duration':["30days","40days","35days","40days","60days","50days","55days"],
    'Discount':[11.8,23.7,13.4,15.7,12.5,25.4,18.4]}
df = pd.DataFrame(technologies)
print(df)
print(df.dtypes)
#convert all types to best possible types
#object and string data type characteristics more and less are same but it's storing form is different
df2=df.convert_dtypes()#object convert into string
print(df2.dtypes)
#change all columns to same type
df=df.astype(str)#string convert into object
print(df.dtypes)
#change type for one or multiple columns
df=df.astype({"Fee":int,"Discount":float})
print(df.dtypes)
#convert data type for all columns in a list
df=pd.DataFrame(technologies)
df.dtypes
cols=["Fee","Discount"]
df[cols]=df[cols].astype('float')
df.dtypes
#Ignore error
df=df.astype({"Courses":int},errors='ignore')
df.dtypes
#generator error
df=df.astype({"Courses":int},errors='raise')
#converts feed column to numeric type
df=df.astype(str)
print(df.dtypes)
df['Discount']=pd.to_numeric(df['Discount'])
df.dtypes


######################
import pandas as pd
#create DataFrame from Dictionary
technologies={
    'Courses':["Spark","Pyspark","Hadoop"],
    'Fee':[20000,25000,26000],
    'Duration':["30days","40days","35days"],
    'Discount':[1000,2300,1500]
    }
df=pd.DataFrame(technologies)
df
##############################
#convert dataframe to csv
df.to_csv('data_file.csv')
df
###########################
#pandas dataframe - basic operations
#create dataframe with None/Null to work with examples

