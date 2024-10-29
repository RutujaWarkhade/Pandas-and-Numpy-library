# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:18:06 2024

@author: OM
"""
"""
iloc:(integer location)
select data by row and column indices
eg:
    df.iloc[0,1]
    select the value in 1st row and 2nd column
    
loc:(label location)
select data by row and column labels
eg:
    df.loc['row1','b']

"""




import pandas as pd
import numpy as np
technologies={
    'Courses':["Spark","Pyspark","Hadoop","Python","Pandas","Oracel","Java"],
    'Fee':[20000,25000,26000,22000,24000,21000,22000],
    'Duration':["30days","40days","35days","40days","60days","50days","55days"],
    'Discount':[11.8,23.7,13.4,15.7,12.5,25.4,18.4]}
df = pd.DataFrame(technologies)
df
df2=df.iloc[2]#select 2nd row
df2
df2=df.iloc[[2,3,6]]#select rows by index list
df2
df2=df.iloc[1:5]#select rows by integer 
df2
df2=df.iloc[:1]#select first row
df2
df2=df.iloc[:3]#select first 3 row
df2
df2=df.iloc[-1:]#select last row
df2
df2=df.iloc[-3:]#select last 3 rows
df2
df2=df.iloc[::2]#select alternate rows
df2
#in iloc it is index value hece it's end value is excluding value
#select rows by index labels
import pandas as pd
import numpy as np
technologies={
    'courses':["Spark","Pyspark","Hadoop","Python","Pandas","Oracel","Java"],
    'Fee':[20000,25000,26000,22000,24000,21000,22000],
    'Duration':["30days","40days","35days","40days","60days","50days","55days"],
    'Discount':[11.8,23.7,13.4,15.7,12.5,25.4,18.4]}
row=('r1','r2','r3','r4','r5','r6','r7')
df = pd.DataFrame(technologies,index=row)
df

df2=df.loc['r2']#select rows by labels
df2
df2=df.loc[['r2','r3','r6']]#select rows by index list
df2=df.loc['r1':'r5':2]#select rows by label index row
df2=df.loc['r1':'r5':2]#select alternate rows with index
#in loc it is row name hence end value is not excluding value
################################
#by using df[] notation
df2=df['courses']
df2
#select multiple columns
df2=df[["courses","Fee","Duration"]]
df2
#using loc[] to take column slice
#loc[] syntax to slice columns
#df.loc[:,start:stop:step]#stop value is not excluding
#select multiple columns
df2=df.loc[:,["courses","Fee","Duration"]]
df2
#select random columns
df2=df.loc[:,["courses","Fee","Discount"]]
#select cloumns between two columns




df2=df.loc[:,"Fee","Discount"]
df2



#select column by range



df2=df.loc[:,"Duration":]
df2


#select column by range
#all the columns upto 'Duration'
df2=df.loc[:,:'Duration']
df2
#select every alternate column
df2=df.loc[:,::2]
df2
##########################################

######################################
#Pandas.DataFrame.query() by examples
#Query all rows with courses equals 'Spark'
df2=df.query("courses=='Spark'")#spark should be in single cout
print(df2)
########################
#not equals condtion 
df2=df.query("courses != 'Spark'")
df2
################################
######################

#pandas add columns to DataFrame
import pandas as pd
import numpy as np
technologies={
    'Courses':["Spark","Pyspark","Hadoop","Python","Pandas"],
    'Fee':[20000,25000,23000,24000,26000],
    'Discount':[0.1,0.2,0,0.5,0.1]}#it only 5 element otherwise give error
#i.e it must have equal number of entities
df=pd.DataFrame(technologies)
print(df)
###########################
#Pandas add columns to dataframe
#add new columns to dataframe
tutors = ['Ram','Sham','Ghansham','Ganesh','Ramesh']
df2=df.assign(TutorAssigned=tutors)
print(df2)
##################################
#add mulitple columns to the DataFrame
MNCCompanies=["TATA","HCL","Infosys","Google","Amazon"]
df2=df.assign(MNC=MNCCompanies,tutors=tutors)
df2
#########################
#Derived New Column from existing column
df=pd.DataFrame(technologies)
df2=df.assign(Discount_Percent=lambda x: x.Fee* x.Discount/100)
print(df2)
####################################
#######################
#append column to excisting pandas dataframe
#add new column to the existing dataframe
df=pd.DataFrame(technologies)
df["MNCCompanies"]=MNCCompanies
print(df)
########################



#add new column at specific position
df=pd.DataFrame(technologies)
df.insert(0,'Tutors',tutors)
print(df)



########################################
#pandas rename columns with example
import pandas as pd
technologies={
    'Courses':["Spark","Pyspark","Hadoop","Python","Pandas","Oracel","Java"],
    'Fee':[20000,25000,26000,22000,24000,21000,22000],
    'Duration':["30days","40days","35days","40days","60days","50days","55days"],}
df = pd.DataFrame(technologies)
df.columns
print(df.columns)
#pandas rename column name
#rename a single column name



df2=df.rename(columns={'Courses':'Courses_list'},axis=1)
df2
df2=df.rename({'Courses':'Courses_list'},axi='columns')



##################################
#in order to chnage columns on existing dataframe
#without copying to the new dataframe
#you have to use inplace=True
