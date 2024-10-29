# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:21:35 2024

@author: OM
"""
import numpy as np
np.__version__
#output:'1.26.4'
    

import pandas as pd
#read excel file
df=pd.read_excel('#any excel file address which end with .xlsx')
print(df)

####################################
#using Series.values.tolist()
import pandas as pd
df=pd.DataFrame({ 'Courses':["Spark","Pyspark","Python","Pandas"],
 'Fee':[20000,25000,22000,24000]})
df
col_list=df.Courses.values#data will convert into array form
print(col_list)

col_list=df.Courses.values.tolist()#data will convert into list form
print(col_list)

#using Series.Values.tolist()
col_list=df["Courses"].values.tolist()
print(col_list)

#using list() function
col_list=list(df["Courses"])
print(col_list)

import numpy as np
#convert to numpy array
col_list=df["Courses"].to_numpy()
print(col_list)

###############################
#what is numpy?
'''the  numpy library is a popular open source python
stands for numerical python
'''
#use in image processing
#use in matrix
#use in linear algebra
#x=3(scalar)
#x=[11 20 30] 1 dimentional array called vector
# 
"""
While a python list can contain different 
data types within a single list,
all of the element in a Numpy array 
should be homogeneous
"""
#array in Numpy
#create ndarray
import numpy as np
arr=np.array([10,20,30])
print(arr)
#output:
#[10 20 30]
#create a Multi-Dimentional Array
#create a multidimentional array
arr=np.array([[10,20,30],[40,50,60]])
print(arr)
#output:
 # [[10 20 30]
 #  [40 50 60]]  
 #represent the minimum dimensions
 #use ndmin param to specify how many minimum
 #dimensions you wanted to create an array with 
 #minimum dimenstion
 #[[[ ]]] -->> 3-dimentional array
arr=np.array([10,20,30,40],ndmin=3)
print(arr)
#output:
#[[[10 20 30 40]]]
#change the datatype
#dttype parameter
arr=np.array([10,20,30],dtype=complex)
print(arr)
#output:
#[10.+0.j 20.+0.j 30.+0.j]

#get the dimentions of array
#get dimentions of the array
arr=np.array([[1,2,3,4],[7,8,6,7],[9,10,11,12]])
print(arr.ndim)
print(arr)

#output:
#2

#[[ 1  2  3  4]
#[ 7  8  6  7]
#[ 9 10 11 12]]

############################

#finding the size of each data item in the array
arr=np.array([10,20,30])
print("Each item contain in bytes:",arr.item.size)

#################

arr=np.array([10,20,30])
print("Each item is of the type")










############################################

#create NumPy Array from list
#creation of array
import numpy as np
arr=np.array([10,20,30])
print("Array:",arr)
######################################
#creating array from list with type float
arr=np.array([[10,20,40],[30,40,50]],dtype='float')
print("Array created by using list: \n",arr)
#output:
#
#[[10. 20. 40.]
#[30. 40. 50.]]
##########################################
#create a sequence of integers using arange()
#create a sequence of integers
#from 0 to 20 with step of 3
import numpy as np
arr=np.arrange(0,20,3)
print("A sequential array with step of 3:\n",arr)
###############################
#################################
#array indexing in numpy
#output:
#a sequential array with steps of 3:
#[0 3 6 9 12 15 18]
#access single element using index
import numpy as np
arr=np.arrange(11)
print(arr)
#[0 1 2 3 4 5 6 7 8 9 10]
print(arr[2])
#2
print(arr[-2])
#9
#####################################
#Multi-Dimentional Array indexing
#access multi-dimentional array element
#using array indexing
arr=np.array([[10,20,30,40,50],[20,30,50,10,30]])
print(arr)
#[[10 20 30 40 50]
#[20 30 50 10 30]]
print(arr.shape)
#(2,5)#now x is 2-dimentional
print(arr[1,1])
#30
print(arr[0,4])
#50
print(arr[1,-1])#rows start from 0, we need 1 st row
#30
#################################
#access array elements using slicing
arr=np.array([0,1,2,3,4,5,6,7,8,9])
x=arr[1:8:2]
print(x)
#output:[1 3 5 7]

#example
x=arr[-2:3:-1]#start last but one(-2) upto 3 but not
print(x)
#output:[8 7 6 5 4]

#example
x=arr[-2:10]
print(x)
#[8 9]



#what is numpy?
'''the  numpy library is a popular open source python
stands for numerical python
'''

'''lists are heterogenious while arrays are homogeneous'''

#arrays in numpy
#create ndarray
import numpy as np
arr = np.array([10,20,30])
print(arr)

#create a multi-dimensional array
arr = np.array([[10,20,30],[40,50,60]])
print(arr)

#represent the minimum dimensions
#use ndmin param to specify how many minimum
#dimensions u want to create an array with minimum dimension
arr = np.array([10,20,30], ndmin = 3)
print(arr)

#change the data type
#dtype parameter
arr = np.array([10,20,30], dtype = complex)
print(arr)

#get the dimensions of array
#get dimension of array
arr = np.array([[1,2,3,4],[7,8,7,6],[9,10,11,12]])
print(arr.ndim)
print(arr)

#finding the size of each item int the array
arr = np.array([10,20,30])
print("each item contain in bytes :",arr.itemsize)

##########################
#get shape ans size of array
arr = np.array([[10,20,30,40],[60,70,80,90]])
print("array size:",arr.size)
print("array shape",arr.shape)

###############################
#create numpy array from list
#creation of arrays
arr = np.array([10,20,30])
print("array:",arr)
###########################
#creating array from list with type float
arr = np.array([[10,20,30],[30,40,50]],dtype = 'float')
print('array created by using list:\n',arr)

#create sequence of integers using arrange()
#create a sequence of integers
#from 0 to 20 with steps of 3
arr = np.arange(0,20,3)
print("a sequential array with steps of 3:\n",arr)
#####################################

#access single element using index
arr = np.arange(11)
print(arr)
print(arr[2])
print(arr[-2])

#multi dimensional array indexing
#access multi dimensional array element
#using array indexing
arr = np.array([[10,20,30,40,50],[20,30,50,10,30]])
print(arr)
print(arr.shape)
print(arr[1,1])
print(arr[0,4])
print(arr[1,-1])#1st row and last column

##########################
#access array element using slicing
arr = np .array([0,1,2,3,4,5,6,7,8,9])
x= arr[1:8:2]
print(x)

x=arr[-2:3:-1]
print(x)
#indexing in numpy
import numpy as np
multi_arr=np.array([[10,20,10,40],
                    [40,50,70,90],
                    [60,10,70,80],
                    [30,90,40,30]])
multi_arr
#Slicing array
#For multi-dimentional Numpy arrays,
#you can access the elements as below
multi_arr [1, 2]#-To access the value at row
multi_arr [1,:]#-To get the value at row 1 and
multi_arr [:,1] #-Access the value at all rows

x=multi_arr[:3,::2]#columns from 0 to 3,in array
print(x)
#output:
  #  [[10 10]
    # [40 70]
    # [60 70]]
"""session_21"""
#integer array indexing
#integer array indexing allows theselection
#integer array indexing
arr=np.arrange(35).reshape(5,7)
print(arr)
#[[0 1 2 3]
#[4 5 6 7]
#[8 9 10 11]]
rows =np.array([False,True,True])#not 0th row only
rows
wanted_rows=arr[rows, : ]#in selected rows all rows
print(wanted_rows)
#[[4 5 6 7]
#[8 9 10 11]]
#convert array
array=np.array(list)
print("Array:",array)
#output:Array:[20 40 60 80]

#numpy.asarray():Using numpy
#ndarray.itemsize
#ndarray.size
#ndarray.dtype

#ndarray.shape
#to get shape of a python NumPy array use numpy
#shape
array=np.array([[1,2,3],[4,5,6]])
array
print(array.shape)
#o/p:
#(2,3)

#Resize the array
array=np.array([[10,20,30],[40,50,60]])
array.shape=(3,2)
print(array)
#o/p:
"""
   [[10 20]
     [30 40]
     [50 60]]"""
#Numpy also provide a numpy.reshape() function to ...

#reshape usage
array=np.array([[10,20,30],[40,50,60]])
new_array=array.reshape(3,2)#3 is row & 2 is column
print(new_array)    
"""[[10 20]
 [30 40]
 [50 60]]"""
    
#arithmatic operations on numpy arrays
import numpy as np
arr1=np.arange(16).reshape(4,4)
arr2=np.array([1,3,2,4])
#add()
add_arr=np.add(arr1,arr2)
print(f"Adding two arrays:\n{add_arr}")
"""Adding two arrays:
[[ 1  4  4  7]
 [ 5  8  8 11]
 [ 9 12 12 15]
 [13 16 16 19]]"""
#substract()


#multiply()
mul_arr=np.multiply(arr1,arr2)
print(f"Multiply two arrays:\n{mul_arr}")
"""
Multiply two arrays:
[[ 0  3  4 12]
 [ 4 15 12 28]
 [ 8 27 20 44]
 [12 39 28 60]]
"""
#divide()
div_arr=np.divide(arr1,arr2)
print(f"Dividing two arrays:\n{div_arr}")
"""
Dividing two arrays:
[[ 0.          0.33333333  1.          0.75      ]
 [ 4.          1.66666667  3.          1.75      ]
 [ 8.          3.          5.          2.75      ]
 [12.          4.33333333  7.          3.75      ]]
"""
#numpy.reciprocal()
#this function returns the reciprocal of argument
#element-wise. for elements with absolute values
#larger than 1, the result is always zero because of the way in which...

#to perform reciprocal operation
import numpy as np
arr1=np.array([50,10.3,5,1,200])
rep_arr1=np.reciprocal(arr1)
print(f"After applying function to array:\n{rep_arr1}")



#to perform power operation
arr1=np.array([3,10,5])
pow_arr1=np.power(arr1,3)
print(f"After applying power function to array:\n{pow_arr1}")
"""
After applying power function to array:
[  27 1000  125]
"""
arr1=np.array([3,10,5])
arr2=np.array([3,2,1])
print("My second array:\n",arr2)
pow_arr2=np.power(arr1,arr2)
print(f"After applying power function to array:\n{pow_arr2}")
"""
After applying power function to array:
[ 27 100   5]
"""

#To perform mod function
#on numpy array
import numpy as np
arr1=np.array([7,20,13])
arr2=np.array([3,5,2])
arr1
arr1.dtype
#rem()




#mod()
mod_arr=np.mod(arr1,arr2)
print(f"After applying mod function to array:\n{mod_arr}")
#Applying mod() function: [1 0 1]

#################################################

#create empty array
from numpy import empty
a=empty([3,3])
print(a)
"""
[[6.23042070e-307 9.45734480e-308 1.78022342e-306]
 [6.23058028e-307 6.23053954e-307 1.60218763e-306]
 [8.34451504e-308 1.60205318e-306 2.56765117e-312]]
"""
##################################
#create zero array
from numpy import zeros
a=zeros([3,5])
print(a)
"""
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
"""
########################################
#create one array
from numpy import ones
a=ones([5])
print(a)
#o/p:[1. 1. 1. 1. 1.]
##############################
#create array with vstack
from numpy import array
from numpy import vstack
#create first array
a1=array([1,2,3])
print(a1)
#[1 2 3]
#create second array
a2=array([4,5,6])
print(a2)
#[4 5 6]
#vertical array
a3=vstack((a1,a2))
print(a3)
"""[[1 2 3]
 [4 5 6]]"""
print(a3.shape)
#(2,3)
################################
#create array with hstack
from numpy import array
from numpy import hstack
#ctreate first array
a1=array([1,2,3])
print(a1)
#create second array
a2=array([4,5,6])
print(a2)
#create horizontal stack
a3=hstack((a1,a2))
print(a3)
print(a3.shape)

#############################################
"""
23 April 2024
session 21
"""
#index data 

#index data out of bounds
from numpy import array
#define array
data=array([11,22,33,44,55])
print(data[5])
#########################
#index data
print(data[0,0])
###########################
#index row of two-dimensional array
from numpy import array
#definr array
data=array([
    [11,22],
    [33,44],
    [55,66]])
#index data
print(data[0,])#0th row and all columns
#[11 22]
###############################
#slice a one dimensional array
from numpy import array
#define array
data=array([11,22,33,44,55])
print(data[1:4])
#[22 33 44]
##############################
#negative slicing of a one dimentional array
from numpy import array
#define array
data=array([
    [11,22,33],
    [44,55,66],
    [77,88,99]])
#seprate array
x,y=data[:, :-1],data[:, -1]
x
y
#data[:,:-1]-all rows and columns 0 and 1
#all rows and last column
#broadcast scalar to one-dimentional array
from numpy import array
#define array
a=array([1,2,3])
print(a)
#define scalar
b=4
print(b)
c=a+b
print(c)
######################################
"""
vector l1 norm
the L1 norm is calculated as the sum of the absolute vector values,
where the absolute value of a scalar uses the notation |a1|
In effect,the norm is a calculation of the Manhattan distance
from the origin of the vector space.
||v||=|a1|+|a2|+|a3|
"""
from numpy import array
from numpy.linalg import norm
#define vector
a=array([1,2,3])
print(a)
#[1 2 3]
#calcculate array
l1=norm(a,1)
print(l1)
#6.0
########################################
#vector L2 norm
"""
the notation for the L2 norm of a vector x is ||x|| power of 2.
To calculated the L2 norm of a vector,
taken the square root of the sum of the squared vector values
Another names for L2 norm of a vector is Eulidean distance.
This is often used for calculating the error in machine learning models.
"""
from numpy import array
from numpy.linalg import norm
#define vector
a=([1,2,3])
print(a)
#[1, 2, 3]
########################
#calculate norm
l2=norm(a)
print(l2)
#3.7416573867739413
########################
#triangular matrix
"""
in this triangular matrix use print() to written output 
otherwise it gives ',' in it's output
"""
from numpy import array
from numpy import tril#tril=lower triangular matrix
from numpy import triu#triu=upper triangular matrix
#define square matrix
M=array([
[1,2,3],
[1,2,3],
[1,2,3]])
print(M)
"""
[[1 2 3]
 [1 2 3]
 [1 2 3]]
"""
#lower triangular matrix
lower=tril(M)
print(lower)
"""
[[1 0 0]
 [1 2 0]
 [1 2 3]]
"""
#upper triangular matrix
upper=triu(M)
print(upper)
"""
[[1 2 3]
 [0 2 3]
 [0 0 3]]
"""
####################################
#diagonal matrix
from numpy import array
from numpy import diag
#define square matrix
M=array([
    [1,2,3],
    [1,2,3],
    [1,2,3]])
print(M)
"""
[[1 2 3]
 [1 2 3]
 [1 2 3]]
"""
#extract diagonal vector
d=diag(M)
print(d)
"""
[1 2 3]
"""
#create diagonal matrix from vector
D=diag(d)
print(D)
"""
[[1 0 0]
 [0 2 0]
 [0 0 3]]

"""
######################
#identity matrix
from numpy import identity
I=identity(3)
print(I)
#########################
#orthogonal matrix
"""
the matrix is said to be orthogoanal if the product of 
matrix and its transpose gives an identity value
"""
from numpy import array
from numpy.linalg import inv
#define orthogonal matrix
Q=array([
    [1,0],
    [0,-1]])
print(Q)
#######################
#inverse equivalence
v=inv(Q)
print(Q.T)
print(v)
#identity equivalence
I=Q.dot(Q.T)
print(I)
###############################

################################

"""
24 april 2024
"""
from numpy import array
#define matrix
A=array([
    [1,2],
    [3,4],
    [5,6]])
print(A)
"""
[[1 2]
 [3 4]
 [5 6]]

"""
#calculate transpose
C=A.T
print(C)
"""
[[1 3 5]
 [2 4 6]]
"""

#################################

#inverse matrix
from numpy import array
from numpy.linalg import inv
#define matrix
A=array([
    [1.0,2.0],
    [3.0,4.0]])
print(A)
"""
[[1. 2.]
 [3. 4.]]
"""
#invert matrix
B=inv(A)
print(B)
"""
[[-2.   1. ]
 [ 1.5 -0.5]]

"""
#multiply A and B
I=A.dot(B)
print(I)
"""
[[1.00000000e+00 1.11022302e-16]
 [0.00000000e+00 1.00000000e+00]]

"""
#sparce matrix
#count number of 1 and 0 in rows 
from numpy import array
from scipy.sparse import csr_matrix
#create dense matrix
A=array([
    [1,0,0,1,0,0],
    [0,0,2,0,0,1],
    [0,0,0,2,0,0]])
print(A)
"""
[[1 0 0 1 0 0]
 [0 0 2 0 0 1]
 [0 0 0 2 0 0]]
"""
#convert to sparce matrix (CSR method)
S=csr_matrix(A)
print(S)
"""
(0, 0)	1
(0, 3)	1
(1, 2)	2
(1, 5)	1
(2, 3)	2
"""
#reconstruction dense matrix
B=S.todense()
print(B)
"""
[[1 0 0 1 0 0]
 [0 0 2 0 0 1]
 [0 0 0 2 0 0]]
"""
#write a python to draw a line with suitable label 
#matplotlib is a major pakage in that pyplot
import matplotlib.pyplot as plt
X=range(1,50)
Y=[value*3 for value in X]
print("values of x:")
print(*range(1,50))
"""
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
17 18 19 20 21 22 23 24 25 26 27 28 29
30 31 32 33 34 35 36 37 38 39 40 41 42
43 44 45 46 47 48 49
"""
"""
this is equivalent to-
i in range(1,50):
    print(i,end=' ')
"""
print("values in Y:")
print(Y)
"""
[3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
 33, 36, 39, 42, 45, 48, 51, 54, 57, 60,
 63, 66, 69, 72, 75, 78, 81, 84, 87, 90,
 93, 96, 99, 102, 105, 108, 111, 114, 117,
 120, 123, 126, 129, 132, 135, 138, 141,
 144, 147]

"""
#plot lines and/or markers to the axes
plt.plot(X, Y)
#set the x axis label of the  current axis
plt.xlabel('x-axis')
#set the y axis label of the currentaxis
plt.ylabel('y-axis')
#set a title
plt.title("Draw a line")
#display the figure
plt.show()
#this all run at a time i.e block of statement
#########################################
#label in the x axis, y axis and a title
import matplotlib.pyplot as plt
#x axis value
x=[1,2,3]
#y axis value
y=[2,4,1]
#plot lines and/or markers to the axes
plt.plot(x,y)
#set the x axis label of the current axis
plt.xlabel('x-axis')
#set the y axis label of the current current axis
plt.ylabel('y-axis')
#set a title
plt.title("Sample graph!")
#display the figure
plt.show()
#########################################
#write a pyhton program to plot two or more lines
#on same plot with suitable legend of each line
import matplotlib.pyplot as plt
#line 1 points
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]
#plotting the line 1 points
plt.plot (x1,y1,label="line 1")
#plotting the line 2 points
plt.plot(x2,y2,label="line 2")
plt.xlabel('x-axis')
#set the y-axis label of the current axis
plt.ylabel('y-axis')
#set a title
plt.title('Two or more lines on same plot with suitable legend')
#show legend on the plot
plt.legend()
#display a figure
plt.show()
############################
#write a python 
#line with diffrent colour and width
#line 1 points
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]
#plotting the line 1 points
plt.plot (x1,y1,label="line 1")
#plotting the line 2 points
plt.plot(x2,y2,label="line 2")
plt.xlabel('x-axis')
#set the y-axis label of the current axis
plt.ylabel('y-axis')
#set a title
plt.title('Two or more lines on same plot with suitable legend')
#display the figure
plt.plot(x1,y1, color="blue", linewidth=3, label='linewidth-3')
plt.plot(x2,y2, color="red", linewidth=5, label='linewidth-5')
#show legend on the plot
plt.legend()
#display a figure
plt.show()
######################################
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]
#plotting the line 1 points
plt.plot (x1,y1,label="line 1")
#plotting the line 2 points
plt.plot(x2,y2,label="line 2")
plt.xlabel('x-axis')
#set the y-axis label of the current axis
plt.ylabel('y-axis')
#set a title
plt.title('Two or more lines on same plot with suitable legend')
#display the figure
plt.plot(x1,y1, color="blue", linewidth=3, label='line1-dotted',linestyle="dotted")
plt.plot(x2,y2, color="red", linewidth=5, label='line2-dashed',linestyle='dashed')
#show legend on the plot
plt.legend()
#display a figure
plt.show()

########################################
"""
25 April
"""
#introdusing the marker in the graph
#write a pyhton program to plot two or more lines
#and set the line markers
import matplotlib.pyplot as plt
#x axis values
x=[1,4,5,6,7]
#y axis values
y=[2,6,3,6,3]
#plotting the points
plt.plot(x,y, color='red',linestyle='dashdot',linewidth=3,
         marker='o',markerfacecolor='blue',markersize=12)
#set the y-limits of the current axes
plt.ylim(1,8)
#set the x-limits of the current axes
plt.xlim(1,8)
#naming the x axis
plt.xlabel('x-axis')
#naming the y axis
plt.ylabel('y-axis')
#giving a titile to my graph
plt.title('Display marker')
#function to show
plt.show()
#############################################
#imp
#write a python programm to display
#a bar chart of the popularity of programming languages
import matplotlib.pyplot as plt
x=['java','python','php','javascript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i, _ in enumerate(x)]
plt.bar(x_pos, popularity, color='blue')
plt.xlabel("Language")
plt.ylabel("popularity")
plt.title("popularity of programming language\n"+
          "Worldwide, oct 2017 compared to a year ago")
plt.xticks(x_pos,x)#take a bar graph on x axis i.e horizontally
plt.minorticks_on()
plt.grid(which='major', linestyle='-',linewidth='0.5',color='red')
plt.show()
############################################
import matplotlib.pyplot as plt
x=['java','python','php','javascript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i, _ in enumerate(x)]
plt.barh(x_pos, popularity, color='green')#instead of bar write barh
plt.xlabel("popularity")
plt.ylabel("language")
plt.title("popularity of programming language\n"+
          "Worldwide, oct 2017 compared to a year ago")
plt.yticks(x_pos,x)#take a bar graph on y axis i.e horizontally
plt.minorticks_on()
plt.grid(which='major', linestyle='-',linewidth='0.5',color='red')
plt.show()
#############################################
#by default bar graph color is blue
import matplotlib.pyplot as plt
x=['java','python','php','javascript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i, _ in enumerate(x)]
plt.bar(x_pos, popularity, color=['red','green','black','yellow','cyan','blue'])#instead of bar write barh
plt.xlabel("popularity")
plt.ylabel("language")
plt.title("popularity of programming language\n"+
          "Worldwide, oct 2017 compared to a year ago")
plt.xticks(x_pos,x)#take a bar graph on y axis i.e horizontally
plt.minorticks_on()
plt.grid(which='major', linestyle='-',linewidth='0.5',color='red')
plt.show()
############################################
#historical distribution means normall or symmetrically distrubuted
import matplotlib.pyplot as plt
blood_sugar=[113,85,90,150,149,88,93,115,135,80,77,82,129]
plt.hist(blood_sugar,rwidth=0.8)#by default number
                                    #of bins is set of 10
plt.hist(blood_sugar,rwidth=0.5,bins=4)
"""
histogram showing normal, pre-diabetic and diabectic patients distributed 
80-100:normal
100-125:Pre-diabetic
125:diabetic
"""
plt.xlabel("sugar level")
plt.ylabel("number of patients")
plt.title("blood sugar chart")
plt.hist(blood_sugar,bins=[80,100,125,150],rwidth=0.95,color='g')
######################################
#multiply data samples in a histogram


#####################################
#box plot
#import libraries
import matplotlib.pyplot as plt
import numpy as np
#create dataset
np.random.seed(10)
data=np.random.normal(100,20,200) 
fig=plt.figure(figsize=(10,7))
#creating plot
plt.boxplot(data)
#show plot
plt.show()
########################################     
import matplotlib.pyplot as plt
import numpy as np
#creating dataset
np.random.seed(10)
data_1=np.random.normal(100,10,200) 
data_2=np.random.normal(90,20,200) 
data_3=np.random.normal(80,30,200)
data_4=np.random.normal(70,40,200)
data=[data_1,data_2,data_3,data_4] 
fig=plt.figure(figsize=(10,7))
#creating axes instance
ax=fig.add_axes([0,0,1,1])
#creating plot
bp=ax.boxplot(data) 
#show plot
###########################################                     