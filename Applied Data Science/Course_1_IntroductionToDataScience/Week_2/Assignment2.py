#!/Applications/anaconda/bin/python

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download
# notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform,
# visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG)
# course resource._
# 
# ---

# # Assignment 2 - Pandas Introduction
# All questions are weighted the same in this assignment.
# ## Part 1
# The following code loads the olympics dataset (olympics.csv), which was derrived
# from the Wikipedia entry on
# [All Time Olympic Games Medals]
# (https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table),
# and does some basic data cleaning. Use this dataset to answer the questions below.

# In[1]:

import pandas as pd

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

# print( 'Before cleaning' )
# print( df.head() )
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

# split the index by '('
names_ids = df.index.str.split('\s\(') 

# the [0] element is the country name (new index)
df.index = names_ids.str[0] 
# the [1] element is the abbreviation or ID (take first 3 characters from that)
df['ID'] = names_ids.str[1].str[:3]

df = df.drop('Totals')

# print( '\n\n ****************  After cleaning' )
# print( df.head() )


# ### Question 0 (Example)
# 
# What is the first country in df?
# 
# *This function should return a Series.*

# In[4]:

# You should write your whole answer within the function provided. The autograder
# will call
# this function and compare the return value against the correct solution value
def answer_zero():
    """
    # This function returns the row for Afghanistan, which is a Series object.
    The assignment question description will tell you the general format the 
    autograder is expecting    
    """
    return df.iloc[0]


# You can examine what your function returns by calling it in the cell.
#If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
#answer_zero() 


# ### Question 1
# Which country has won the most gold medals in summer games?
# 
# *This function should return a single string value.*

# In[ ]:

def answer_one():
    return df.loc[df['Total'].idxmax()].name

def answer_one_one():
    return df['Gold'].idxmax()

# ### Question 2
# Which country had the biggest difference between their summer and winter gold medal counts?
# 
# *This function should return a single string value.*

# In[ ]:

def answer_two():
    ndf = pd.DataFrame(columns=['Country', 'Difference'])
    
    for index, row in df.iterrows():
        dif = abs(row['Total'] - row['Total.1'] ) 
        ndf = ndf.append(pd.Series(data={'Country': index, 'Difference': dif}, \
                                   name=(index)))
        
    return ndf.loc[ndf['Difference'].idxmax()].name


# ### Question 3
# Which country has the biggest difference between their summer and winter gold medal counts
# relative to their total gold medal count? Only include countries that have won at least
# 1 gold in both summer and winter.
# 
# *This function should return a single string value.*

# In[ ]:

def answer_three():
    """
    The relative difference is the summer gold minus winter gold divided by the total 
    of those two. Once you have that for each country you can report back the country 
    with the largest. 
    """
    
    ndf = pd.DataFrame(columns=['Country', 'Diff'])

    
    """
    for index, row in df.iterrows():
        if ( (row['Gold.1'] > 0) & (row['Gold'] > 0) ):
            dif = abs( row['Combined total'] - abs( row['Gold.1'] - row['Gold'] ) )
            ndf = ndf.append(pd.Series(data={'Country': index, 'Diff': dif},name=(index)))
    """

    for index, row in df.iterrows():
        if ( (row['Gold.1'] > 0) & (row['Gold'] > 0) ):
            rel_diff = abs( row['Gold'] - row['Gold.1'] ) / row['Gold.2']
            ndf = ndf.append(pd.Series(data={'Country': index, 'Diff': rel_diff},name=(index)))
            
    return ndf.loc[ndf['Diff'].idxmax()].name


# ### Question 4
# Write a function to update the dataframe to include a new column called "Points" which is a
# weighted value where each gold medal counts for 3 points, silver medals for 2 points, and
# bronze mdeals for 1 point. The function should return only the column (a Series object)
# which you created.
# 
# *This function should return a Series named `Points` of length 146*

# In[ ]:

def answer_four():
    ns = pd.Series()

    for index, row in df.iterrows():
        points = (row['Gold'] + row['Gold.1']) * 3
        points += (row['Silver'] + row['Silver.1']) * 2
        points += (row['Bronze'] + row['Bronze.1'])
        ns = ns.set_value(index, points)

    df['Points'] = ns

    return ns


# ## Part 2
# For the next set of questions, we will be using census data from the [United States Census Bureau]
# (http://www.census.gov/popest/data/counties/totals/2015/CO-EST2015-alldata.html). Counties are
# political and geographic subdivisions of states in the United States. This dataset contains
# population data for counties and states in the US from 2010 to 2015. [See this document]
# (http://www.census.gov/popest/data/counties/totals/2015/files/CO-EST2015-alldata.pdf) for a
# description of the variable names.
# 
# The census dataset (census.csv) should be loaded as census_df. Answer questions using this
# as appropriate.
#

# ### Question 5
# Which state has the most counties in it? (hint: consider the sumlevel key
# carefully! You'll need this for future questions too...)
# 
# *This function should return a single string value.*
# In[ ]:

census_df = pd.read_csv('census.csv')
#print( census_df.head() )


# In[ ]:

def answer_five():

    #return df.loc[df['Total'].idxmax()].name
    #print( census_df.loc[census_df['SUMLEV'] == 50] )
    #    print( census_df.loc[census_df['SUMLEV'] == 50] )
    #print("current state = ", state)
    #print( len(census_df.loc[ (census_df['SUMLEV'] == 50) & (census_df['STATE'] == 1 ) ] ))
    #print(states)
    #states = census_df['STATE'].unique()

    state = "" 
    max_amount = 0
    state_names = census_df['STNAME'].unique()
    
    for state_name in state_names:
        tot_recs = len(census_df.loc[ (census_df['SUMLEV'] == 50) & \
                                      (census_df['STNAME'] == state_name )])
        if tot_recs > max_amount:
            max_amount = tot_recs
            state = state_name  

    return state


#### Question 6
# Only looking at the three most populous counties for each state, what are the three most populous
# states (in order of highest population to lowest population)?
# 
# *This function should return a list of string values.*

# In[ ]:

def answer_six():
    ns = pd.Series()
    
    state_names = census_df['STNAME'].unique()

    for state_name in state_names:
        s_df = census_df.loc[ (census_df['STNAME'] == state_name) & \
                              (census_df['SUMLEV'] == 50 ) ]
        s_df = s_df.sort_values('CENSUS2010POP', ascending=False)

        total_population = 0
        
        if len(s_df) >= 3:          
            for i in range(0, 3):
                temp_df = s_df.iloc[i]
                total_population += temp_df['CENSUS2010POP']
        else:        
            for i in range(0, len(s_df)):
                temp_df = s_df.iloc[i]
                total_population += temp_df['CENSUS2010POP']
        ns = ns.set_value(state_name, total_population)

    ns = ns.sort_values(ascending=False)

    return [ns.index[0], ns.index[1], ns.index[2]]
 

# ### Question 7
# Which county has had the largest change in population within the five year period
# (hint: population values are stored in columns POPESTIMATE2010 through
# POPESTIMATE2015,you need to consider all five columns)?
# *This function should return a single string value.*

# In[ ]:

def answer_seven():
    ns = pd.Series()
    columns_to_keep = ['STNAME',
                       'SUMLEV',
                       'COUNTY',
                       'CTYNAME',
                       'POPESTIMATE2010',
                       'POPESTIMATE2011',
                       'POPESTIMATE2012',
                       'POPESTIMATE2013',
                       'POPESTIMATE2014',
                       'POPESTIMATE2015']
    
    df = census_df[columns_to_keep]
    
    for index, row in df.iterrows():
        min_value = 100000000
        max_value = -1
        #print( row[1] )
        if row[1] == 50:
            for i in range(4,10):
                if row[i] > max_value:
                    max_value = row[i]
                    if row[i] < min_value:
                        min_value = row[i]
                        
            ns = ns.set_value(row[3], abs(min_value - max_value) )

    ns = ns.sort_values(ascending=False)
                  
    return ns.index[0]
                  

# ### Question 8
# In this datafile, the United States is broken up into four regions using the
# "REGION" column. 
# 
# Create a query that finds the counties that belong to regions 1 or 2, whose
# name starts with 'Washington', and whose POPESTIMATE2015 was greater than
# their POPESTIMATE 2014.
# 
# *This function should return a 5x2 DataFrame with the
# columns = ['STNAME', 'CTYNAME'] and the same index ID as the census_df
# (sorted ascending by index).*

# In[ ]:

def answer_eight():

    regions = [1,2]
    name = 'Washington'
    
    df = census_df.loc[ ((census_df['REGION'] == 1) | (census_df['REGION'] == 2) ) & \
                        (census_df['CTYNAME'].str.startswith(name) ) & \
                        (census_df['POPESTIMATE2015'] > census_df['POPESTIMATE2014']) ]

    cols_to_keep = ['STNAME', 'CTYNAME']
    df = df[cols_to_keep].sort_index()
    
    return df

###############################################################
"""
print('###################### Number 0 ################')
print( answer_zero() )

print('###################### Number 1 ################')
country = answer_one()
print(country)

print('###################### Number 2 ################')
result = answer_two()
print( result )

print('###################### Number 3 ################')
result = answer_three()
print( result )

print('###################### Number 4 ################')
result = answer_four()
print( result )

print('###################### Number 5 ################')
result = answer_five()
print(result)
print( type(result) )

print('###################### Number 6 ################')
result = answer_six()
print(result)

print('###################### Number 7 ################')
result = answer_seven()
print( result )

print('###################### Number 8 ################')
result = answer_eight()
print( result )
print( answer_eight() )
"""

result = answer_four()
print( result )

print( result.shape )
