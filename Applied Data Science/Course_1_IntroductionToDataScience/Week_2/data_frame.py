#!/Applications/anaconda/bin/python
import pandas as pd
import numpy as np
import timeit

def function1():
    purchase_1 = pd.Series({'Name': 'Chris',
                            'Item Purchased': 'Dog Food',
                            'Cost': 22.50})
    purchase_2 = pd.Series({'Name': 'Kevyn',
                            'Item Purchased': 'Kitty Litter',
                            'Cost': 2.50})
    purchase_3 = pd.Series({'Name': 'Vinod',
                            'Item Purchased': 'Bird Seed',
                            'Cost': 5.00})
    df = pd.DataFrame([purchase_1, purchase_2, purchase_3], \
                      index=['Store 1', 'Store 1', 'Store 2'])
    #print(df.head())
    """
             Cost Item Purchased   Name
    Store 1  22.5       Dog Food  Chris
    Store 1   2.5   Kitty Litter  Kevyn
    Store 2   5.0      Bird Seed  Vinod
    """

    #print(df.loc['Store 2'])
    """
    Cost                      5
    Item Purchased    Bird Seed
    Name                  Vinod
    Name: Store 2, dtype: object
    """

    #print( type(df.loc['Store 2']) )
    #### <class 'pandas.core.series.Series'>

    #print( df.loc['Store 1'] )
    """
             Cost Item Purchased   Name
    Store 1  22.5       Dog Food  Chris
    Store 1   2.5   Kitty Litter  Kevyn
    """

    # print( df.loc['Store 1', 'Cost'] )
    """
    Store 1    22.5
    Store 1     2.5
    Name: Cost, dtype: float64
    """

    # print( df.T )
    """
                      Store 1       Store 1    Store 2
     Cost                22.5           2.5          5
     Item Purchased  Dog Food  Kitty Litter  Bird Seed
     Name               Chris         Kevyn      Vinod
    """

    #print( df.T.loc['Cost'] )
    """
    Store 1    22.5
    Store 1     2.5
    Store 2       5
    Name: Cost, dtype: object
    """

   # print( df['Cost'] )
    """
    Store 1    22.5
    Store 1     2.5
    Store 2     5.0
    Name: Cost, dtype: float64
    """

    #print( df.loc['Store 1']['Cost'] )
    """
    Store 1    22.5
    Store 1     2.5
    Name: Cost, dtype: float64
    """

    #print( df.loc[:,['Name', 'Cost']] )
    """
              Name  Cost
    Store 1  Chris  22.5
    Store 1  Kevyn   2.5
    Store 2  Vinod   5.0
    """

    ### This demonstrates that modifications are not done in place
    #print( df.drop('Store 1') )
    #print( df )
    """
             Cost Item Purchased   Name
    Store 2   5.0      Bird Seed  Vinod

             Cost Item Purchased   Name
    Store 1  22.5       Dog Food  Chris
    Store 1   2.5   Kitty Litter  Kevyn
    Store 2   5.0      Bird Seed  Vinod
    """

    copy_df = df.copy()
    copy_df = copy_df.drop('Store 1')
    #print( copy_df )
    """
               Cost Item Purchased   Name
      Store 2   5.0      Bird Seed  Vinod
    """

    df['Location'] = None 
    #print( df )
    """
             Cost Item Purchased   Name Location
    Store 1  22.5       Dog Food  Chris     None
    Store 1   2.5   Kitty Litter  Kevyn     None
    Store 2   5.0      Bird Seed  Vinod     None
    """

    ## This reduces the value in the cost column by 20%
    #print(df)
    ### These two approaches are both right
    #df['Cost'] *= 0.8
    df['Cost'] -= df['Cost'] * .2;
    #print(df)
    """
    ### Before
             Cost Item Purchased   Name Location
    Store 1  22.5       Dog Food  Chris     None
    Store 1   2.5   Kitty Litter  Kevyn     None
    Store 2   5.0      Bird Seed  Vinod     None
    ### After
             Cost Item Purchased   Name Location
    Store 1  18.0       Dog Food  Chris     None
    Store 1   2.0   Kitty Litter  Kevyn     None
    Store 2   4.0      Bird Seed  Vinod     None
    """

    ## This will print the names of people who bought things
    ## with cost > 3
    print( df['Name'][df['Cost']>3] )
    
###############################

def function2():
    purchase_1 = pd.Series({'Name': 'Chris',
                            'Item Purchased': 'Dog Food',
                            'Cost': 22.50})
    purchase_2 = pd.Series({'Name': 'Kevyn',
                            'Item Purchased': 'Kitty Litter',
                            'Cost': 2.50})
    purchase_3 = pd.Series({'Name': 'Vinod',
                            'Item Purchased': 'Bird Seed',
                            'Cost': 5.00})
    df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

    ## Create a series consisting of the cost values
    costs = df['Cost']
    #print(costs)
    """
    Store 1    22.5
    Store 1     2.5
    Store 2     5.0
    Name: Cost, dtype: float64
    """

    ## Add 2 to the cost values
    costs += 2
    #print(costs)
    """
    Store 1    22.5
    Store 1     2.5
    Store 2     5.0
    Name: Cost, dtype: float64
    ###  After
    Store 1    24.5
    Store 1     4.5
    Store 2     7.0
    Name: Cost, dtype: float64
    """
    ### Print the original df; notice the change done to the series
    ### above is shown in the original df
    ### If this is not desired, use the copy() method on the dataframe
    #print(df)
    """
             Cost Item Purchased   Name
    Store 1  24.5       Dog Food  Chris
    Store 1   4.5   Kitty Litter  Kevyn
    Store 2   7.0      Bird Seed  Vinod
    """

    ### Reindex the purchase records DataFrame to be indexed hierarchically,
    ### first by store, then by person. Name these indexes 'Location' and 'Name'.
    ### Then add a new entry to it with the value of:
    ### Name: 'Kevyn', Item Purchased: 'Kitty Food', Cost:3.00 Location: 'Store 2'

    df = df.set_index([df.index, 'Name'])
    df.index.names = ['Location', 'Name']
    df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))
    #print( df )
    """
                    Cost Item    Purchased
    Location Name                      
    Store 1  Chris       24.5       Dog Food
             Kevyn        4.5   Kitty Litter
    Store 2  Vinod        7.0      Bird Seed
             Kevyn        3.0     Kitty Food
    """
    
    ## Read in a CSV file 
    # df = pd.read_csv('olympics.csv')
    # print( df.head() )

    ## Read a CSV file,
    ## 1. Set which column to use as index
    ## 2. Tell the load process to skip the first row
    df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
    #print(df.head())

    ## Do the read and fix the column names
    #print(df.columns)
    for col in df.columns:
        if col[:2]=='01':
            df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
        if col[:2]=='02':
            df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
        if col[:2]=='03':
            df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
        if col[:1]=='№':
            df.rename(columns={col:'#' + col[1:]}, inplace=True) 

    #print( df.head())

    ##  This will return a series with boolean values indicating whether a
    ##  country has won at least one gold medal. An excerpt is shown below.
    ##  This is a small excerpt. The actual result has more values than this.
    ##  The result of this is a boolean mask
    ##  print( df['Gold'] > 0 )
    """
    Afghanistan (AFG)                               False
    Algeria (ALG)                                    True
    Argentina (ARG)                                  True
    Armenia (ARM)                                    True
    """

    ## This would actually return another df with only the countries that had won
    ## at least one gold medal.
    only_gold = df.where(df['Gold'] > 0)
    #print( only_gold.head() )

    ## Count of countries that have won a gold medal
    #print( only_gold['Gold'].count() )
    """
    100
    """
    ## The total number of countries in the original df
    #print( df['Gold'].count() )
    """
    147
    """

    ## Drop those rows that have no data
    only_gold = only_gold.dropna()
    #print( only_gold.head() )

    ## This selects countries that have at least a gold medal, and it also
    ## removes rows without data.
    only_gold = df[df['Gold'] > 0]
    #print( only_gold.head() )

    ### Countries that have won at least a gold medal in the summer or the winter
    #print( len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)]) )
    """
    101
    """

    ### Countries that have won at least a gold medal in the summer but not in
    ### the winter
    print( df[(df['Gold.1'] > 0) & (df['Gold'] == 0)] )
    
#############################################
    
def function3():
    df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
    
    ## Do the read and fix the column names
    #print(df.columns)
    for col in df.columns:
        if col[:2]=='01':
            df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
        if col[:2]=='02':
            df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
        if col[:2]=='03':
            df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
        if col[:1]=='№':
            df.rename(columns={col:'#' + col[1:]}, inplace=True) 

    #print(df.head())

    ## Changing the index involves
    ## 1. Create a new column for the previous index
    ## 2. Creating the new index using the set_index command
    #df['country'] = df.index
    #df = df.set_index('Gold')
    #print( df.head() )

    ### This will move the original index to a new column and then add
    ### a numerical index
    # df = df.reset_index()
    # print( df.head() )

    ## We are now dealing with census data
    df = pd.read_csv('census.csv')
    #print( df.head() )

    ### Summary Level - show all the distinct values, this is similar
    ### to SQL distinct
    print( df['SUMLEV'].unique() )
    """
    [5 rows x 100 columns]
    [40 50]
    """
    
    ## Get rid of all the summaries at state level and just keep the county
    ## data. SUMLEV == 50
    df=df[df['SUMLEV'] == 50]
    #print( df.head() )
    #print( df )

    ### Eliminate some columns from the data and specify the ones to keep
    columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
    df = df[columns_to_keep]
    #print(df.head())

    ### Setting the index to multiple columns
    df = df.set_index(['STNAME', 'CTYNAME'])
    #print( df.head() ) 

    
    #print( df.loc['Michigan', 'Washtenaw County'] )


    #print( df.loc[ [('Michigan', 'Washtenaw County'), ('Michigan', 'Wayne County')] ] )

#######################################
def function4():
    df = pd.read_csv('log.csv')
    #print( df )
    
    #df.fillna? 

    ## Set the index to 'time' and then sort on the index
    df = df.set_index('time')
    df = df.sort_index()
    #print( df )


    ### The above operation may have issues because more than one user
    ### can be using the system at the same time. As such the index will
    ### be unique.
    ### As such we can use a multilevel index.
    df = df.reset_index()
    df = df.set_index(['time', 'user'])
    print( df )

    ## This fills in the missing values
    df = df.fillna(method='ffill')
    print( df.head())

    
###############################
### The DataFrame datastructure intro
#function1()

### The DataFram Indexing and Loading
#function2()

### Indexing Dataframes
function3()

### Missing Values
#function4()
    
