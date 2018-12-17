#!/Applications/anaconda/bin/python

from IPython import get_ipython
import pandas as pd
import numpy as np


"""
2, 9, 12
"""

# # Assignment 3 - More Pandas
# All questions are weighted the same in this assignment. This assignment requires
# more individual learning then the last one did - you are encouraged to check out
# the [pandas documentation]
# (http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you
# might not have used yet, or ask questions on
# [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related.
# And of course, the discussion forums are open for interaction with your peers
# and the course staff.

#########################################################
# ### Question 1
# Load the energy data from the file `Energy Indicators.xls`, which is a list of
# indicators of
# [energy supply and renewable electricity production](Energy%20Indicators.xls)
# from the [United Nations]
# (http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls)
# for the year 2013, and should be put into a DataFrame with the variable name
# of **energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file.
# Also, make sure to exclude the footer and header information from the datafile.
# The first two columns are unneccessary, so you should
# get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable's]`
# 
# Convert the energy supply and the energy supply per capita to gigajoules
# (there are 1,000,000 gigajoules in a petajoule). For all countries which have
# missing data (e.g. data with "...")
# make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with parenthesis in their name. Be sure to remove
# these, e.g. `'Bolivia (Plurinational State of)'` should be `'Bolivia'`.
# 
# <br>
# 
# Next, load the GDP data from the file `world_bank.csv`, which is a csv containing
# countries' GDP from 1960 to 2015 from
# [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD).
# Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# <br>
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering
# and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102),
# which ranks countries based on their journal contributions in the aforementioned
# area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the
# intersection of country names). Use only the last 10 years (2006-2015) of GDP
# data and only the top 15
# countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country.
# 
# *This function should return a DataFrame with 20 columns and 15 entries.*
####################################################################
def answer_one():

    """
    The question asks to label the energy table's columns as 
    ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable's] 
    but it should be 
    ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'] 
    That tiny little naughty s !! Also refer to this thread for correct order of columns
    """

    """
    correct_column_order = ['Rank', 'Documents', 'Citable documents',
                            'Citations', 'Self-citations', 'Citations per document',
                            'H index', 'Energy Supply', 'Energy Supply per Capita',
                            '% Renewable', '2006', '2007', '2008', '2009', '2010', 
                            '2011', '2012', '2013', '2014', '2015']
    df = df[correct_column_order]

    df=answer_one()
    print(df.index)
    
    ##output should be
    Index(['China', 'United States', 'Japan', 'United Kingdom',
    'Russian Federation', 'Canada', 'Germany', 'India', 'France',
    'South Korea', 'Italy', 'Spain', 'Iran', 'Australia', 'Brazil'],
    dtype='object', name='Country')

    """


    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 100
    
    df = pd.read_excel('Energy Indicators.xls')
    #print(df.head())
    df = df.drop(df.index[(df.index <= 15) | (df.index >= 243 )])
    df.drop(df.columns[[0,1]], axis=1, inplace=True )

    #print(df.head())
    df = df.rename(columns={'Environmental Indicators: Energy': 'Country', \
                            'Unnamed: 3': 'Energy Supply', \
                            'Unnamed: 4': 'Energy Supply per Capita', \
                            'Unnamed: 5': '% Renewable'})

    df = df.replace('...',np.NaN)

    df['Energy Supply'] =  df['Energy Supply'] * 1000000
    # df['Energy Supply per Capita'] =  df['Energy Supply per Capita'] * 1000000

    df = df.replace( ["Republic of Korea", 'United States of America20', \
                      'United Kingdom of Great Britain and Northern Ireland19', \
                      'China, Hong Kong Special Administrative Region3', \
                      'Falkland Islands (Malvinas)', \
                      'Iran (Islamic Republic of)','Bolivia (Plurinational State of)', \
                      'Venezuela (Bolivarian Republic of)',\
                      'Micronesia (Federated States of)', 'Sint Maarten (Dutch part)', \
                      'China2', 'Japan10','France6', 'Italy9','Spain16','Australia1'], \
                     ['South Korea', 'United States', 'United Kingdom', 'Hong Kong', \
                       'Falkland Islands', 'Iran', 'Bolivia','Venezuela','Micronesia', \
                       'Sint Maarten', 'China', 'Japan','France', 'Italy','Spain', \
                       'Australia'] )
    #print(df)
    


    ###################  Data 2  ########################
    df2 = pd.read_csv('world_bank.csv')

    #print( df2 )
    df2.drop(df2.head(4).index, inplace=True)
    
    df2 = df2.replace(['Korea, Rep.', 'Iran, Islamic Rep.', 'Hong Kong SAR, China'],\
                      ['South Korea','Iran','Hong Kong'])

    df2 = df2.rename(columns ={'Data Source': 'Country'})

    columns_to_keep = [ 'Country',
                        'Unnamed: 50',
                        'Unnamed: 51',
                        'Unnamed: 52',
                        'Unnamed: 53',
                        'Unnamed: 54',
                        'Unnamed: 55',
                        'Unnamed: 56',
                        'Unnamed: 57',
                        'Unnamed: 58',
                        'Unnamed: 59' ]
    df2 = df2[columns_to_keep]

    df2 = df2.rename(columns = {'Unnamed: 50': '2006',
                                'Unnamed: 51': '2007',
                                'Unnamed: 52': '2008',
                                'Unnamed: 53': '2009',
                                'Unnamed: 54': '2010',
                                'Unnamed: 55': '2011',
                                'Unnamed: 56': '2012',
                                'Unnamed: 57': '2013',
                                'Unnamed: 58': '2014',
                                'Unnamed: 59': '2015'} )

    #######################   Data 3  ####################
    df3 = pd.read_excel('scimagojr-3.xlsx')

    #print(df3)
    # print("Column names for Sci Major = ", df3.columns.values )

    df4 = pd.merge(df, df2, how='inner', \
                   left_on=['Country'], right_on=['Country'] )
    
    df5 = pd.merge(df4, df3, how='inner', \
                          left_on=['Country'], right_on=['Country'] )

    # print([df5])

    df6 = df5.loc[ df5['Rank'] <= 15 ]

    df6 = df6.set_index(['Country'])

    #image_name_data['id'] = image_name_data['id'].astype('str')
    df6['Rank'] = df6['Rank'].astype('int')
    #print( len(df6) )

    correct_column_order = ['Rank', 'Documents', 'Citable documents',
                            'Citations', 'Self-citations', 'Citations per document',
                            'H index', 'Energy Supply', 'Energy Supply per Capita',
                            '% Renewable', '2006', '2007', '2008', '2009', '2010', 
                            '2011', '2012', '2013', '2014', '2015']
    df6 = df6[correct_column_order]
    return  df6


#########################################################
# ### Question 2 (6.6%)
# The previous question joined three datasets then reduced this to just
# the top 15 entries. When you joined the datasets, but before you reduced
# this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*
def answer_two():

    df = pd.read_excel('Energy Indicators.xls')
    df = df.drop(df.index[(df.index <= 15) | (df.index >= 243 )])
    df.drop(df.columns[[0,1]], axis=1, inplace=True )

    #print(df)
    df = df.rename(columns={'Environmental Indicators: Energy':'Country', \
                            'Unnamed: 3': 'Energy Supply', \
                            'Unnamed: 4': 'Energy Supply per Capita', \
                            'Unnamed: 5': '% Renewable'})

    df = df.replace('...',np.NaN)

    df['Energy Supply'] =  df['Energy Supply'] * 1000000

    df = df.replace( ["Republic of Korea", 'United States of America20', \
                      'United Kingdom of Great Britain and Northern Ireland19', \
                      'China, Hong Kong Special Administrative Region3', \
                      'Falkland Islands (Malvinas)', \
                      'Iran (Islamic Republic of)','Bolivia (Plurinational State of)', \
                      'Venezuela (Bolivarian Republic of)',\
                      'Micronesia (Federated States of)', 'Sint Maarten (Dutch part)', \
                      'China2', 'Japan10','France6', 'Italy9','Spain16','Australia1'], \
                      ['South Korea', 'United States', \
                       'United Kingdom', \
                       'Hong Kong', \
                       'Falkland Islands', \
                       'Iran', 'Bolivia','Venezuela','Micronesia', \
                       'Sint Maarten', 'China', 'Japan','France', 'Italy','Spain', \
                       'Australia'] )
    
    ###################  Data 2  ########################
    df2 = pd.read_csv('world_bank.csv')
    df2.drop(df2.head(4).index, inplace=True)
    
    df2 = df2.replace(['Korea, Rep.', 'Iran, Islamic Rep.', 'Hong Kong SAR, China'],\
                      ['South Korea','Iran','Hong Kong'])

    df2 = df2.rename(columns ={'Data Source': 'Country'})

    columns_to_keep = [ 'Country',
                        'Unnamed: 50',
                        'Unnamed: 51',
                        'Unnamed: 52',
                        'Unnamed: 53',
                        'Unnamed: 54',
                        'Unnamed: 55',
                        'Unnamed: 56',
                        'Unnamed: 57',
                        'Unnamed: 58',
                        'Unnamed: 59' ]
    df2 = df2[columns_to_keep]

    """
    df2 = df2.rename(columns = {'Unnamed: 50': 'GDP 2006',
                                'Unnamed: 51': 'GDP 2007',
                                'Unnamed: 52': 'GDP 2008',
                                'Unnamed: 53': 'GDP 2009',
                                'Unnamed: 54': 'GDP 2010',
                                'Unnamed: 55': 'GDP 2011',
                                'Unnamed: 56': 'GDP 2012',
                                'Unnamed: 57': 'GDP 2013',
                                'Unnamed: 58': 'GDP 2014',
                                'Unnamed: 59': 'GDP 2015'} )
    """

     
    #######################   Data 3  ####################
    
    df3 = pd.read_excel('scimagojr-3.xlsx')

    # print(df3)

    """
    print("Total rows in Energy Indicators = ", len(df) )
    print("Total rows in World Bank = ", len(df2) )
    print("Total rows in Sci Major = ", len(df3) )

    Total rows in Energy Indicators =  227
    Total rows in World Bank =  264
    Total rows in Sci Major =  191
    Energy + World Bank =  315
    Total rows in all three =  328

    Column names for Energy =  ['Country' 'Energy Supply' 'Energy Supply per Capita' 
    '% Renewable']

    Column names for World Bank =  ['Country' 'GDP 2006' 'GDP 2007' 'GDP 2008' 
    'GDP 2009' 'GDP 2010' 'GDP 2011' 'GDP 2012' 'GDP 2013' 'GDP 2014' 'GDP 2015']

    Column names for Sci Major =  ['Rank' 'Country' 'Documents' 'Citable documents' 'Citations'
    'Self-citations' 'Citations per document' 'H index']
    """
    
    ## Should this be a union
    df4 = pd.merge(df3, df2, how='inner', \
                   left_on=['Country'], right_on=['Country'] )

    # print( df4 )

    # This is another union 
    df5 = pd.merge(df4, df, how='inner', \
                       left_on=['Country'], right_on=['Country'] )
    
    df6 = df5.loc[ df5['Rank'] <= 15 ]
    
    return (len(df5) - len(df6) )


#########################################################
# ### Question 3 (6.6%)
# What are the top 15 countries for average GDP over the last 10 years?
# 
# *This function should return a Series named `avgGDP` with 15 countries and
# their average GDP sorted in descending order.*

# In[4]:

def answer_three():
    Top15 = answer_one()
    #print( Top15.columns.values )
    avgGDP = pd.Series()
    avgGDP.name = 'avgGDP'
    
    """
    Column names for result =  ['Energy Supply' 'Energy Supply per Capita' '% Renewable' 
    '2006' '2007' '2008' '2009' '2010' '2011' '2012'
    'GDP 2013' 'GDP 2014' 'GDP 2015' 'Rank' 'Documents' 'Citable documents'
    'Citations' 'Self-citations' 'Citations per document' 'H index']
    """

    countries = Top15.index

    for index, row in Top15.iterrows():
        sum = 0
        num_of_values = 0
        for i in range(10, 20):
            if np.isnan(row.iloc[i]) == False:
                sum += row.iloc[i]
                num_of_values += 1

        average = sum / num_of_values
        avgGDP.set_value(index, average)

    avgGDP = avgGDP.sort_values(ascending=False)
    return avgGDP

#########################################################
# ### Question 4 (6.6%)
# By how much had the GDP changed over the 10 year span for the country
# with the 6th largest average GDP?
# 
# *This function should return a single number.*

def answer_four():
    Top15 = answer_one()
    #print(Top15.columns.values)
    
    avgGDP = answer_three()

    value = avgGDP.index[5]
    
    df = Top15.loc[value]

    return ( df[19] - df[10] )

#########################################################
# ### Question 5 (6.6%)
# What is the mean energy supply per capita?
# 
# *This function should return a single number.*

def answer_five():
    Top15 = answer_one()
    mean = Top15['Energy Supply per Capita'].mean()
    return mean

#########################################################
# ### Question 6 (6.6%)
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the
# percentage.*

def answer_six():
    Top15 = answer_one()

    country = Top15.loc[Top15['% Renewable'].idxmax()].name
    row = Top15.loc[country]
    value = row[9]
    print( country, value )
    
    return (country, value)

#########################################################
# ### Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

def answer_seven():
    Top15 = answer_one()
    # Citations  Self-citations

    Top15['Ratio'] = Top15['Self-citations'] / Top15['Citations']
    #print(Top15)
    country = Top15.loc[Top15["Ratio"].idxmax()].name
    row = Top15.loc[country]
    value = row['Ratio']
    
    return (country, value)

#########################################################
# ### Question 8 (6.6%)
# 
# Create a column that estimates the population using Energy Supply and Energy Supply
# per capita. What is the third most populous country according to this estimate?
# 
# *This function should return a single string value.*

def answer_eight():
    Top15 = answer_one()
    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15 = Top15.sort_values('Population', ascending=False)
    
    return Top15.index[2]

#########################################################
# ### Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita
# and the energy supply per capita?
# 
# This function should return a single number.*
# 
# (Optional: Use the built-in function `plot9()` to visualize the relationship between
# Energy Supply per Capita vs. Citable docs per Capita).*
def answer_nine():
    Top15 = answer_one()
    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citation Per Capita'] = Top15['Citable documents'] / Top15['Population']
    correlation = Top15['Energy Supply per Capita'].corr(Top15['Citable documents'])
    return correlation


#########################################################
# ### Question 10 (6.6%)
# Create a new columns
# above the median for all countries in the top 15.
# 
# This function should return a series named `HighRenew` whose index is the
# country name sorted in ascending order of rank.*

def answer_ten():

    Top15 = answer_one()    
    new_series = pd.Series()
    new_series.name = 'HighRenew'
    print(new_series.name)

    median = Top15['% Renewable'].median()

    #Top15 = Top15.sort_values('Rank',ascending=True)
    Top15 = Top15.sort_values('% Renewable', ascending=True)


    for index, row in Top15.iterrows():
        if row['% Renewable'] >= median:
            new_series.set_value(index, 1)
        else :
            new_series.set_value(index, 0)
    
    return new_series

#########################################################
# ### Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent, then create
# a dateframe that displays the sample size (the number of countries in each
# continent bin), and the sum, mean, and std deviation for the estimated population
# of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named
# Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']`
# and columns `['size', 'sum', 'mean', 'std']`*

def answer_eleven():
    Top15 = answer_one()

    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']

    cont_df = pd.DataFrame( index = ['Asia', 'Australia', 'Europe', 'North America', \
                                     'South America'], \
                            columns = ['size', 'sum', 'mean', 'std'] )

    ContinentDict  = {'China':'Asia', 
                      'United States':'North America', 
                      'Japan':'Asia', 
                      'United Kingdom':'Europe', 
                      'Russian Federation':'Europe', 
                      'Canada':'North America', 
                      'Germany':'Europe', 
                      'India':'Asia',
                      'France':'Europe', 
                      'South Korea':'Asia', 
                      'Italy':'Europe', 
                      'Spain':'Europe', 
                      'Iran':'Asia',
                      'Australia':'Australia', 
                      'Brazil':'South America'}

    map_series = pd.Series()

    for index, row in Top15.iterrows():
        map_series.set_value(index, ContinentDict[index])

    Top15['Continent'] = map_series

    continents = Top15['Continent'].unique()

    return Top15.set_index('Continent').groupby(level=0)['Population'] \
                                       .aggregate({'size': len, 'sum': np.sum, \
                                                   'mean': np.mean, 'std': np.std})
          
#########################################################
# ### Question 12 (6.6%)
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these
# new % Renewable bins. How many countries are in each of these groups?
# 
# This function should return a Series with a MultiIndex of `Continent`,
# then the bins for `% Renewable`.
# Do not include groups with no country 


def answer_twelve():
    """
    Try simply using the .count() on your groupby ... if you get 
    your groupby correct, this should return the multiindexed series 
    the grader is looking for
    """
    Top15 = answer_one()

    #   Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']

    cont_df = pd.DataFrame( index = ['Asia', 'Australia', 'Europe', 'North America', \
                                     'South America'], \
                            columns = ['size', 'sum', 'mean', 'std'] )

    ContinentDict  = {'China':'Asia', 
                      'United States':'North America', 
                      'Japan':'Asia', 
                      'United Kingdom':'Europe', 
                      'Russian Federation':'Europe', 
                      'Canada':'North America', 
                      'Germany':'Europe', 
                      'India':'Asia',
                      'France':'Europe', 
                      'South Korea':'Asia', 
                      'Italy':'Europe', 
                      'Spain':'Europe', 
                      'Iran':'Asia',
                      'Australia':'Australia', 
                      'Brazil':'South America'}

    map_series = pd.Series()
    
    for index, row in Top15.iterrows():
        map_series.set_value(index, ContinentDict[index])

    Top15['Continent'] = map_series

    r_series = Top15['% Renewable']
    series_cut = pd.cut(r_series, 5)

    #print(series_cut)
    #print(r_series)
    
    Top15.groupby('Continent')
    #print(Top15)
    #print(Top15)

    
    return "ANSWER"

#########################################################
# ### Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator
# (using commas)
# 
# e.g. 12345678.90 -> 12,345,678.90
# 
# *This function should return a Series `PopEst` whose index is the country name
# and whose values are the population estimate string.*

# In[16]:

def answer_thirteen():
    """
    # Top15['PopEst'].apply(lambda x: '{:,}'.format(x)) 
    
    pd.options.display.float_format = '{:,.2f}'.format
    Top15 = answer_one()


    # df['col_name'] = df['col_name'].astype(object)
    
    pop_series = Top15['Population'].astype(object)
    pop_series.name = 'PopEst'

    Country
    Australia               23,316,017.32
    Brazil                 205,915,254.24
    Canada                  35,239,864.86
    China                1,367,645,161.29
    France                  63,837,349.40
    Germany                 80,369,696.97
    India                1,276,730,769.23
    Iran                    77,075,630.25
    Italy                   59,908,256.88
    Japan                  127,409,395.97
    South Korea             49,805,429.86
    Russian Federation     143,500,000.00
    Spain                   46,443,396.23
    United Kingdom          63,870,967.74
    United States          317,615,384.62
    """

    Top15 = answer_one()
    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    #result = Top15['Population'].apply(lambda x: '{:,.2f}'.format(x) )
    result = Top15['Population'].apply(lambda x: '{:,}'.format(x) )

    return result
    

##################################   Test Area   #############################
#print( answer_one() )
# result = answer_one()
# print("Column names for result = ", result.columns.values )
# print( result['Rank'] )
#print( result.index )

######
#print( answer_two() )

##### 
#print( answer_three() )

####
#print( answer_four() )

####
#print( answer_five() )

#####
#print( answer_six() )

########
#print( answer_seven() )

##########
print( answer_eight() )

##########
#print( answer_nine() )

########
#print( answer_ten() )
#######
#print( answer_eleven() )

##########
#print( answer_twelve() )


##########
#print( answer_thirteen() )
