# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:07:12 2024

@author: om
"""

"""
1. Write a Pandas program to convert Series of lists to one Series. 
Sample Output:  
Original Series of list 
0    [Red, Green, White] 
1    [Red, Black] 
2    [Yellow] 
dtype: object 
One Series 
0     Red 
1     Green 
2     White 
3     Red 
4     Black 
5    Yellow 
dtype: object
"""
import pandas as pd

data = pd.Series([
    ['Red', 'Green', 'White'],
    ['Red', 'Black'],
    ['Yellow']
])
data
data1 = pd.Series([item for i in data for item in i])
data1



"""
2. Create a result array by adding the following two NumPy arrays. Next, 
modify the result array by calculating the square of each element 
arrayOne = numpy.array([[5, 6, 9], [21 ,18, 27]]) 
arrayTwo = numpy.array([[15 ,33, 24], [4 ,7, 1]])
"""
import numpy 
import pandas as pd
arrayOne = numpy.array([[5, 6, 9], [21 ,18, 27]]) 
arrayTwo = numpy.array([[15 ,33, 24], [4 ,7, 1]])
array = numpy.add(arrayOne,arrayTwo)
array
sq = list(map(lambda x:x**x,array))
sq


"""
3. Write a NumPy program to compute the mean, standard deviation, and 
variance of a given array along the second axis. 
array:  
[0 1 2 3 8 5]
"""
import numpy as np
data = np.array([0,1, 2, 3, 8, 5])
data
mean = np.mean(data)
mean
std = np.std(data)
std
var = np.mean(data)
var
