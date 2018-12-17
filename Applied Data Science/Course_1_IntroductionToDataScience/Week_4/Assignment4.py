#!/Applications/anaconda/bin/python 

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments
# - you are encouraged to check out the [pandas documentation]
# (http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods
# you might not have used yet, or ask questions on
# [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python
# related. And of course, the discussion forums are open for interaction with your
# peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March,
# Q2 is April through June, Q3 is July through September, Q4 is October through
# December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline,
# and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
#
# * A _university town_ is a city which has a high percentage of university students
# compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by
# recessions. Run a t-test to compare the ratio of the mean price of houses in
# university towns the quarter before the recession starts compared to the recession
# bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
#############################################
# * From the [Zillow research data site](http://www.zillow.com/research/data/)
# there is housing data for the United States. In particular the datafile for
# [all homes at a city level]
# (http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv),
# ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
#############################################
# * From the Wikipedia page on college towns is a list of [university towns in the
# United States]
# (https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States)
# which has been copy and pasted into the file ```university_towns.txt```.
# ############################################
# * From Bureau of Economic Analysis, US Department of Commerce,
# the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States
# in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in
# the file ```gdplev.xls```. For this assignment, only look at GDP data from the first
# quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of
# ```run_ttest()```,
# which is worth 50%.


# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


###################################################################
def get_list_of_university_towns():
    """
    Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan","Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State","RegionName"]  )
    """

    pd.set_option('display.max_rows', 530)

    univ_towns_df = pd.DataFrame( columns = ['State', 'RegionName']  )
    df = pd.read_table( 'university_towns.txt', header=None, index_col=None )
    #df = pd.read_table( 'university_towns.txt', header=None)

    current_state = ""
    for index, row in df.iterrows():
        if ( 'edit' in row[0] ):
             current_state = row[0].split('[')[0].strip()
        else:
            region = row[0].split('(')[0].strip()
            univ_towns_df = univ_towns_df.append(pd.Series(data={'State': current_state, \
                                                                 'RegionName': region}, \
                                                           name=index) )
            
    return univ_towns_df

##################################################################
def get_recession_start():
    """
    Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3


    'Year' 'GDP in billions of current dollars'
    'GDP in billions of chained 2009 dollars' 'Quarter'
    'GDP in billions of current dollars.1'
    'GDP in billions of chained 2009 dollars.1']
    """

    ## skiprows=range(1, 10))
    # df.drop(df.index[[1,3]])
    gdp_df = pd.read_excel('gdplev.xls', skiprows=range(0,5) )
    gdp_df.drop(gdp_df.columns[[3,7]], axis=1, inplace=True)

    gdp_df = gdp_df.rename(columns={'Unnamed: 0': 'Year', \
                                    'GDP in billions of current dollars': 'GDP_CURR_YEAR', \
                                    'GDP in billions of chained 2009 dollars': 'GDP_2009_YEAR', \
                                    'Unnamed: 4': 'Quarter', \
                                    'GDP in billions of current dollars.1': 'GDP_CURR_QUART', \
                                    'GDP in billions of chained 2009 dollars.1': 'GDP_2009_QUART'} )
    
    gdp_df.drop(gdp_df.index[[0,1]], inplace=True)

    # Start from here "214 2000q1 12359.1"
    gdp_df = gdp_df.drop(gdp_df.index[gdp_df.index < 214])
    gdp_df = gdp_df.reset_index()

    #print( gdp_df.head() )
    #print( gdp_df.columns.values )
    # print(len(gdp_df))
    # print(gdp_df)   
    # for i in range(0, len(gdp_df)):
    #   print( gdp_df.iloc[i] )
            
    for i in range(2, len(gdp_df) - 2):
        if ( (gdp_df.iloc[i]['GDP_2009_QUART'] < gdp_df.iloc[i-1]['GDP_2009_QUART']) &
              (gdp_df.iloc[i-1]['GDP_2009_QUART'] < gdp_df.iloc[i-2]['GDP_2009_QUART']) ):
            if ( (gdp_df.iloc[i+1]['GDP_2009_QUART'] > gdp_df.iloc[i]['GDP_2009_QUART']) &
                 (gdp_df.iloc[i+2]['GDP_2009_QUART'] > gdp_df.iloc[i+1]['GDP_2009_QUART']) ):
                return gdp_df.iloc[i-3]['Quarter']

    return None

##################################################################
def get_recession_end():
    """
    Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005 q3
    """
    gdp_df = pd.read_excel('gdplev.xls', skiprows=range(0,5) )
    gdp_df.drop(gdp_df.columns[[3,7]], axis=1, inplace=True)

    gdp_df = gdp_df.rename(columns={'Unnamed: 0': 'Year', \
                                    'GDP in billions of current dollars': 'GDP_CURR_YEAR', \
                                    'GDP in billions of chained 2009 dollars': 'GDP_2009_YEAR', \
                                    'Unnamed: 4': 'Quarter', \
                                    'GDP in billions of current dollars.1': 'GDP_CURR_QUART', \
                                    'GDP in billions of chained 2009 dollars.1': 'GDP_2009_QUART'} )
    
    gdp_df.drop(gdp_df.index[[0,1]], inplace=True)

    gdp_df = gdp_df.drop(gdp_df.index[gdp_df.index < 214])
    gdp_df = gdp_df.reset_index()
            
    for i in range(2, len(gdp_df) - 2):
        if ( (gdp_df.iloc[i]['GDP_2009_QUART'] < gdp_df.iloc[i-1]['GDP_2009_QUART']) &
              (gdp_df.iloc[i-1]['GDP_2009_QUART'] < gdp_df.iloc[i-2]['GDP_2009_QUART']) ):
            if ( (gdp_df.iloc[i+1]['GDP_2009_QUART'] > gdp_df.iloc[i]['GDP_2009_QUART']) &
                 (gdp_df.iloc[i+2]['GDP_2009_QUART'] > gdp_df.iloc[i+1]['GDP_2009_QUART']) ):
                return gdp_df.iloc[i+2]['Quarter']

    return None

##################################################################
def get_recession_bottom():
    """
    Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3
    """

    gdp_df = pd.read_excel('gdplev.xls', skiprows=range(0,5) )
    gdp_df.drop(gdp_df.columns[[3,7]], axis=1, inplace=True)

    gdp_df = gdp_df.rename(columns={'Unnamed: 0': 'Year', \
                                    'GDP in billions of current dollars': 'GDP_CURR_YEAR', \
                                    'GDP in billions of chained 2009 dollars': 'GDP_2009_YEAR', \
                                    'Unnamed: 4': 'Quarter', \
                                    'GDP in billions of current dollars.1': 'GDP_CURR_QUART', \
                                    'GDP in billions of chained 2009 dollars.1': 'GDP_2009_QUART'} )
    
    gdp_df.drop(gdp_df.index[[0,1]], inplace=True)

    gdp_df = gdp_df.drop(gdp_df.index[gdp_df.index < 214])
    gdp_df = gdp_df.reset_index()
            
    for i in range(2, len(gdp_df) - 2):
        if ( (gdp_df.iloc[i]['GDP_2009_QUART'] < gdp_df.iloc[i-1]['GDP_2009_QUART']) &
              (gdp_df.iloc[i-1]['GDP_2009_QUART'] < gdp_df.iloc[i-2]['GDP_2009_QUART']) ):
            if ( (gdp_df.iloc[i+1]['GDP_2009_QUART'] > gdp_df.iloc[i]['GDP_2009_QUART']) &
                 (gdp_df.iloc[i+2]['GDP_2009_QUART'] > gdp_df.iloc[i+1]['GDP_2009_QUART']) ):
                return gdp_df.iloc[i]['Quarter']

    return None


##################################################################
def convert_housing_data_to_quarters():
    """
    Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    """

    df = pd.read_csv("City_Zhvi_AllHomes.csv")
    df['State'] = df['State'].map(states)
    
    df.set_index(['State','RegionName'], inplace=True)

    df = df.drop(df.ix[:,'RegionID':'1999-12'].head(0).columns, axis=1)

    columns_to_keep = []

    for year in range(2000, 2017):
        y_str = str(year).strip()
        if year == 2016:                   
            df[y_str + "q" + str(1)] = df[[y_str + "-01", y_str + "-02", y_str+"-03"]].mean(axis=1)
            columns_to_keep.append( y_str + "q" + str(1) )
            df[y_str + "q" + str(2)] = df[[y_str + "-04", y_str + "-05", y_str+"-06"]].mean(axis=1)
            columns_to_keep.append( y_str + "q" + str(2) )
            df[y_str + "q" + str(3)] = df[[y_str + "-07", y_str +"-08"]].mean(axis=1)
            columns_to_keep.append( y_str + "q" + str(3)  )
        else:
            df[y_str + "q" + str(1)] = df[[y_str+'-01', y_str+'-02', y_str+'-03']].mean(axis=1)
            columns_to_keep.append( y_str + "q" + str(1)  )
            df[y_str + "q" + str(2)] = df[[y_str+'-04', y_str+'-05', y_str+'-06']].mean(axis=1)
            columns_to_keep.append( y_str + "q" + str(2)  )
            df[y_str + "q" + str(3)] = df[[y_str+'-07', y_str+'-08', y_str+'-09']].mean(axis=1)
            columns_to_keep.append( y_str + "q" + str(3)  )
            df[y_str + "q" + str(4)] = df[[y_str+'-10', y_str+'-11', y_str+'-12']].mean(axis=1)
            columns_to_keep.append( y_str + "q" + str(4)  )

    df = df[columns_to_keep]
    
    return df

##################################################################
def run_ttest():
    """
    First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).
    """


    utowns_df = get_list_of_university_towns()

    utowns_df.set_index(['State', 'RegionName'], inplace=True)

    housing_df = convert_housing_data_to_quarters()

    rec_start = get_recession_start()
    rec_end = get_recession_end()
    rec_bot = get_recession_bottom()
     
    columns_to_keep = [rec_start, rec_bot]
    housing_df = housing_df[columns_to_keep]

    # calculate the price_ratio
    housing_df['Price Ratio'] = housing_df[rec_start] / housing_df[rec_bot]

    ut_df = pd.merge(utowns_df, housing_df, how='inner', \
                     left_index=True, right_index=True)
    
    hs_df = housing_df[~housing_df.index.isin(ut_df.index)]

    result = ttest_ind(ut_df['Price Ratio'].dropna(), hs_df['Price Ratio'].dropna(), \
                       equal_var = False)

    #print( type(result))
    #print( result.pvalue )
    # Ttest_indResult(statistic=-2.7770133719318872, pvalue=0.0054964273536938875)
    # df[["A", "B"]].mean(axis=1)
    mean_ut = ut_df['Price Ratio'].mean()
    mean_hs = hs_df['Price Ratio'].mean()
    #print( mean_ut, mean_hs )
    
    part1 = (result.pvalue < 0.1)
    part2 = result.pvalue
    part3 = ""

    if ( mean_ut < mean_hs ):
        part3 = 'university town'
    else:
        part3 = 'non-university town'
    
    
    return (part1, part2, part3)

############################################
# A dataframe with consisting of university towns - "State" and "RegionName"
# print( get_list_of_university_towns() )
# q1 = get_list_of_university_towns()
# print( q1.columns.names )
# print( get_list_of_university_towns() )
# print( get_recession_start() )
# print( get_recession_end() )
# print( get_recession_bottom() )
# print( convert_housing_data_to_quarters() )
print(convert_housing_data_to_quarters() )
print( convert_housing_data_to_quarters().loc["Texas"].loc["Austin"].loc["2010q3"] )
# tests()

#print(run_ttest())
