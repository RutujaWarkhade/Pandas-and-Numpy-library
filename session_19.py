# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:05:59 2024

@author: OM
"""

#pandas shuffle dataframe rows
import pandas as pd
technologies=({
    'Courses':["Spark","Pyspark","Hadoop","Python","Pandas","Oracle","Java"],
    'Fee':[20000,25000,26000,22000,24000,21000,22000],
    'Duration':["30days","40days","35days","40days","60days","50days","55days"],
    'Discount':[1000,2300,1500,1200,2500,2100,2000]})
df=pd.DataFrame(technologies)
print(df) 
#pandas shuffle DataFrame Rows
#shuffle the DataFrame rows & return all rows
df1=df.sample(frac=1)
print(df1)
######################################
#create a new index starting from zero
df1=df.sample(frac=1).reset_index()
print(df1)
#####################################
#drop shuffle index
df1=df.sample(frac=1).reset_index(drop=True)
print(df1)
##########################
import pandas as pd
technologies={
    'Courses':["Spark","Pyspark","Python","Pandas"],
    'Fee':[20000,25000,22000,30000],
    'Duration':["30days","40days","35days","50days"],
    }
index_labels=['r1','r2','r3','r4']
df1=pd.DataFrame(technologies,index=index_labels)
technologies2={
    'Courses':["Spark","Java","Python","Go"],
    'Discount':[2000,2300,1200,2000], }
index_labels2=['r1','r6','r3','r5']
df2=pd.DataFrame(technologies2,index=index_labels2)
#pandas join
df3=df1.join(df2,lsuffix="_left",rsuffix="_right")
print(df3)
#######################################
#pandas inner join dataframe
df3=df1.join(df2,lsuffix="_left",rsuffix="right",how="inner")
print(df3)
#####################################
#pandas left join DataFrame
df3=df1.join(df2,lsuffix="_left",rsuffix="_right",how='inner')
print(df3)
#######################################
#pandas right join DataFrame
df3=df1.join(df2,lsuffix="_left",rsuffix="_right",how='right')
print(df3)
########################################
#pandas merge DataFrame
import pandas as pd
technologies={
    'Courses':["Spark","Pyspark","Python","Pandas"],
    'Fee':[20000,25000,22000,30000],
    'Duration':["30days","40days","35days","50days"],
    }
index_labels=['r1','r2','r3','r4']
df1=pd.DataFrame(technologies,index=index_labels)
technologies2={
    'Courses':["Spark","Java","Python","Go"],
    'Discount':[2000,2300,1200,2000], }
index_labels2=['r1','r6','r3','r5']
df2=pd.DataFrame(technologies2,index=index_labels2)
#usind pandas.merge()
df3=pd.merge(df1,df2)
#using DataFrame.merge()
df3=df1.merge(df2)
########################
#use pandas.concat() to concat two DataFrame
import pandas as pd
df1=pd.DataFrame({ 'Courses':["Spark","Pyspark","Python","Pandas"],
 'Fee':[20000,25000,22000,24000]})
df1=pd.DataFrame({'Courses':["Pandas","HSdoop","Hyperio","Java"],
 'Fee':[25000,25200,24500,24900]})
#using pandas.concat() to concat two DataFrames
data=[df,df1]
df2=pd.concat(data)
df2
##################################
#concatinate multiple DataFrame using pandas,concat()
import pandas as pd
df1=pd.DataFrame({ 'Courses':["Spark","Pyspark","Python","Pandas"],
 'Fee':[20000,25000,22000,24000]})
df1=pd.DataFrame({'Courses':["Pandas","Hadoop","Hyperio","Java"],
 'Fee':[25000,25200,24500,24900]})
df2=pd.DataFrame({'Duration':['30day','40day','35day','60day','55day',],
                  'Discount':[1000,2300,2500,2000,3000]})
#Appending multiple DataFrame
df3=pd.concat([df,df1,df2])
print(df3)
##################################