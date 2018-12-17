#!/Applications/anaconda/bin/python

import pandas as pd
import numpy as np
import timeit

animals = ['Tiger', 'Bear', 'Moose']
s1 = pd.Series(animals)
#print(s1)

"""
0    Tiger
1     Bear
2    Moose
dtype: object
"""
############################
numbers = [1, 2, 3, 4]
s2 = pd.Series(numbers)
#print(s2)
"""
0    1
1    2
2    3
3    4
dtype: int64
"""
############################
animals_2 = ['Tiger', 'Bear', None]
s3 = pd.Series(animals_2)
#print(s3)
"""
dtype: int64
0    Tiger
1     Bear
2     None
dtype: object
"""
############################
numbers_2 = [1,2,3,None]
s4 = pd.Series(numbers_2)
#print(s4)
"""
0    1.0
1    2.0
2    3.0
3    NaN
dtype: float64
"""

############################
#print(np.nan == None)
"""
False
"""

#print(np.isnan(np.nan))
"""
True
"""

############################
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
ss = pd.Series(sports)
#print(ss)
"""
Archery           Bhutan
Golf            Scotland
Sumo               Japan
Taekwondo    South Korea
dtype: object
"""

#print(ss.index)
"""
Index(['Archery', 'Golf', 'Sumo', 'Taekwondo'], dtype='object')
"""

############################
sss = s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
#print(sss)
"""
India      Tiger
America     Bear
Canada     Moose
dtype: object
"""

## This will replace the indexes indexes in sports
ssss = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
#print(ssss)
"""
Golf      Scotland
Sumo         Japan
Hockey         NaN
dtype: object
"""

#######################
sports_2 = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
## Create the series
s_2 = pd.Series(sports_2)

#print(s_2.iloc[3])
#print(s_2.loc['Golf'])
"""
South Korea ## iloc[3]
Scotland    ## loc['Golf']
"""

#############################
sports_3 = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s_3 = pd.Series(sports_3)

#s_3[0] ## This generates an error

###################################
s5 = pd.Series([100.00, 120.00, 101.00, 3.00])

total = 0
for item in s5:
    total += item

#print(s5)
#print(total)

"""
0    100.0
1    120.0
2    101.0
3      3.0
dtype: float64
324.0
"""

###################################################
## This is vectorization of the above operation
total = np.sum(s5)
#print(total)

### Create a series of random integers
s6 = pd.Series(np.random.randint(0,1000,10000))

# prints the first five elements
#print( s6.head() )

###### This prints 10,000
#print(len(s6))

#Timer.timeit -n 100
summary = 0

def addLoop():
    summary = 0
    for item in s6:
        summary += item

#Timer.timeit -n 100

def addNP():
    summary = np.sum(s6)

loop_time = timeit.timeit(addLoop, number=100)
nump_time = timeit.timeit(addNP, number=100)

#print("Loop time = ", loop_time)
#print("Nump time = ", nump_time)
"""
Loop time =  0.10554639299516566
Nump time =  0.0040935060096671805
"""

###########################################
s7 = pd.Series(np.random.randint(0,1000,10000))
#print( s7.head() )

### This adds 2 to each item in s using broadcasting
s7 += 2
#print( s7.head() )

"""
0    502
1    793
2    371
3    784
4     27
dtype: int64
0    504
1    795
2    373
3    786
4     29
dtype: int64
"""

def addIter():
    sa = pd.Series(np.random.randint(0,1000,10000))
    sa += 2

##################
def seriesAddOp():
    sn = pd.Series(np.random.randint(0,1000,10000))
    for label, value in sn.iteritems():
        sn.loc[label]= value+2

#time_iter = timeit.timeit(addIter, number=10)
#time_sers = timeit.timeit(seriesAddOp, number=10)

#print("Iteration time = ", time_iter)
#print("Series Op Time = ", time_sers)

"""
Iteration time =  0.002623969005071558
Series Op Time =  6.556441731998348
"""

###################################################
## Mixing data types in a series
sx = pd.Series([1, 2, 3])
sx.loc['Animal'] = 'Bears'
#print(sx)
"""
0             1
1             2
2             3
Animal    Bears
"""

###################################################
original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                     index=['Cricket',
                                            'Cricket',
                                            'Cricket',
                                            'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)

print(original_sports)
print("%%%%%%%%%%%%%")
print(cricket_loving_countries)
print("%%%%%%%%%%%%%")
print(all_countries)
print("%%%%%%%%%%%%%")
print(all_countries.loc['Cricket'])
