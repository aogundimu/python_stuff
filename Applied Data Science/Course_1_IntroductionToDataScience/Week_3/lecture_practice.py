#!/Applications/anaconda/bin/python

import pandas as pd
import numpy as np

#########################################################
def lecture_merging_dataframes():
    df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},\
                       {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},\
                       {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],\
                      index=['Store 1', 'Store 1', 'Store 2'])
    #print(df)
    """
             Cost Item Purchased   Name
    Store 1  22.5         Sponge  Chris
    Store 1   2.5   Kitty Litter  Kevyn
    Store 2   5.0          Spoon  Filip
    """

    # Add another column
    df['Date'] = ['December 1', 'January 1', 'mid-May']

    #print(df)
    """
             Cost Item Purchased   Name        Date
    Store 1  22.5         Sponge  Chris  December 1
    Store 1   2.5   Kitty Litter  Kevyn   January 1
    Store 2   5.0          Spoon  Filip     mid-May
    """

    # Add another column
    df['Delivered'] = True

    #print(df)
    """
             Cost Item Purchased   Name        Date Delivered
    Store 1  22.5         Sponge  Chris  December 1      True
    Store 1   2.5   Kitty Litter  Kevyn   January 1      True
    Store 2   5.0          Spoon  Filip     mid-May      True
    """

    # Add another column
    df['Feedback'] = ['Positive', None, 'Negative']

    #print(df)
    """
             Cost Item Purchased   Name        Date Delivered  Feedback
    Store 1  22.5         Sponge  Chris  December 1      True  Positive
    Store 1   2.5   Kitty Litter  Kevyn   January 1      True      None
    Store 2   5.0          Spoon  Filip     mid-May      True  Negative
    """

    # Reset the index
    #print(df)
    adf = df.reset_index()
    #print(adf)
    adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
    
    #print(adf)
    """
         index  Cost Item Purchased   Name        Date Delivered  Feedback
    0  Store 1  22.5         Sponge  Chris  December 1      True  Positive
    1  Store 1   2.5   Kitty Litter  Kevyn         NaN      True      None
    2  Store 2   5.0          Spoon  Filip     mid-May      True  Negative
    """

    ######################## Joining Dataframes ##########################
    staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},\
                             {'Name': 'Sally', 'Role': 'Course liasion'},\
                             {'Name': 'James', 'Role': 'Grader'}])
    staff_df = staff_df.set_index('Name')
    
    student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},\
                               {'Name': 'Mike', 'School': 'Law'},\
                               {'Name': 'Sally', 'School': 'Engineering'}])
    student_df = student_df.set_index('Name')

    # print(staff_df.head())
    #print(staff_df)
    """
    Name             Role  
    Kelly  Director of HR
    Sally  Course liasion
    James          Grader
    """
    #print()
    
    #print(student_df.head())
    """
                School
    Name              
    James     Business
    Mike           Law
    Sally  Engineering
    """
    
    ##  Merge both dataframes - Note that the missing data is added - this an outer join
    ##  or union
    union_df = pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
    # print( union_df )
    
    """
                     Role       School
    Name                              
    James          Grader     Business
    Kelly  Director of HR          NaN
    Mike              NaN          Law
    Sally  Course liasion  Engineering
    """
    ## This is the intersection - inner join
    int_df = pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)
    #print(int_df)

    """
                     Role       School
    Name                              
    James          Grader     Business
    Sally  Course liasion  Engineering
    """

    ##################   left join
    left_df = pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)

    #print(left_df)
    """
                     Role       School
    Name                              
    Kelly  Director of HR          NaN
    Sally  Course liasion  Engineering
    James          Grader     Business
    """

    ## right join
    right_df = pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)
    #print(right_df)
    """
                     Role       School
    Name                              
    James          Grader     Business
    Mike              NaN          Law
    Sally  Course liasion  Engineering
    """

    ##  Join not using index
    staff_df = staff_df.reset_index()
    student_df = student_df.reset_index()
    no_idx_df = pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')
    #print(no_idx_df)    
    """
        Name            Role       School
    0  Kelly  Director of HR          NaN
    1  Sally  Course liasion  Engineering
    2  James          Grader     Business 
    """

    ### New Dataframes with Location added
    staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},\
                             {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},\
                             {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
    
    student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},\
                               {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},\
                               {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])
    
    ss_df = pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')
    #print(ss_df)
    #### Note that two locations are added for the two different types of locations
    """
              Location_x   Name            Role            Location_y       School
    0       State Street  Kelly  Director of HR                   NaN          NaN
    1  Washington Avenue  Sally  Course liasion   512 Wilson Crescent  Engineering
    2  Washington Avenue  James          Grader  1024 Billiard Avenue     Business
    """

    ## print(pd.merge(products, invoices, left_index=True, right_on='Product ID'))
    
    ### Inner join
    staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                             {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                             {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
    student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                               {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                               {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
    #print(staff_df)
    #print(student_df)
    last_df = pd.merge(staff_df, student_df, how='inner', \
                       left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])
    print(last_df)
    """
      First Name Last Name            Role       School
    0      Sally    Brooks  Course liasion  Engineering
    """
########################################################
def lecture_pandas_idioms():
    
    df = pd.read_csv('census.csv')
    #print(df)

    
    
    df2 = (df.where(df['SUMLEV']==50)
           .dropna()
           .set_index(['STNAME','CTYNAME'])
           .rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))
    
    #print(df2)

    #df = df[df['SUMLEV']==50]
    #df.set_index(['STNAME','CTYNAME'], inplace=True)
    #df.rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'})

    ##  Drop all entries where 'Quantity' == 0 and change column name 'Weight'
    # print(df.drop(df[df['Quantity'] == 0].index).rename(columns={'Weight': 'Weight (oz.)'}))
    
    def min_max(row):
        data = row[['POPESTIMATE2010',
                    'POPESTIMATE2011',
                    'POPESTIMATE2012',
                    'POPESTIMATE2013',
                    'POPESTIMATE2014',
                    'POPESTIMATE2015']]
        return pd.Series({'min': np.min(data), 'max': np.max(data)})

    dm = df.apply(min_max, axis=1)
    #print(dm)

    def min_max(row):
        data = row[['POPESTIMATE2010',
                    'POPESTIMATE2011',
                    'POPESTIMATE2012',
                    'POPESTIMATE2013',
                    'POPESTIMATE2014',
                    'POPESTIMATE2015']]
        row['max'] = np.max(data)
        row['min'] = np.min(data)
        return row
    dm = df.apply(min_max, axis=1)
    #print(dm)
    
    rows = ['POPESTIMATE2010',
            'POPESTIMATE2011',
            'POPESTIMATE2012',
            'POPESTIMATE2013',
            'POPESTIMATE2014',
            'POPESTIMATE2015']
    dm = df.apply(lambda x: np.max(x[rows]), axis=1)
    # print(dm)

#########################################################
def lecture_groupby() :

    """
    print(df.groupby('Category').apply(lambda df,a,b: sum(df[a] * df[b]), 'Weight (oz.)', 'Quantity'))

    # Or alternatively without using a lambda:
    def totalweight(df, w, q):        
        return sum(df[w] * df[q])        
    """
    df = pd.read_csv('census.csv')
    df = df[df['SUMLEV']==50]
    df


    # In[ ]:

    get_ipython().run_cell_magic('timeit', '-n 10', "for state in df['STNAME'].unique():\n   avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])\n    print('Counties in state ' + state + ' have an average population of ' + str(avg))")


    # In[ ]:

    get_ipython().run_cell_magic('timeit', '-n 10', "for group, frame in df.groupby('STNAME'):\n    avg = np.average(frame['CENSUS2010POP'])\n    print('Counties in state ' + group + ' have an average population of ' + str(avg))")


    # In[ ]:

    df.head()


    # In[ ]:

    df = df.set_index('STNAME')

    def fun(item):
        if item[0]<'M':
            return 0
        if item[0]<'Q':
            return 1
        return 2
    
    for group, frame in df.groupby(fun):
        print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')


        # In[ ]:

    df = pd.read_csv('census.csv')
    df = df[df['SUMLEV']==50]
    
    df.groupby('STNAME').agg({'CENSUS2010POP': np.average})
    
    print(type(df.groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']))
    print(type(df.groupby(level=0)['POPESTIMATE2010']))
    
    (df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
     .agg({'avg': np.average, 'sum': np.sum}))
    
    (df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
     .agg({'avg': np.average, 'sum': np.sum}))
    
    (df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
     .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum}))
    
#########################################################
def lecture_scales(): 
    
    df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],\
                      index=['excellent', 'excellent', 'excellent', 'good', 'good', \
                             'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
    df.rename(columns={0: 'Grades'}, inplace=True)

    #print( df.loc['excellent'] )
    
    # print(df['Grades'].astype('category').head())
    #print(df['Grades'].astype('category'))
    
    grades = df['Grades'].astype('category',\
                                 categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+',
                                             'A-', 'A', 'A+'],ordered=True)
    #print(grades)

    #print(grades > 'C')

    df = pd.read_csv('census.csv')
    df = df[df['SUMLEV']==50]

    #print(df)

    s = pd.Series(['Low', 'Low', 'High', 'Medium', 'Low', 'High', 'Low'])
    cts = s.astype('category', categories=['Low', 'Medium', 'High'], ordered=True)
    #print(cts)
    
    df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})

    #print("***********  After **************")
    # print(df)
    # print(pd.cut(df['avg'],10))

    s = pd.Series([168, 180, 174, 190, 170, 185, 179, 181, 175, 169, 182, 177, 180, 171])
    # This groups everything into three bins
    print( pd.cut(s, 3) )
    print( pd.cut(s, 3, labels=['Small', 'Medium', 'Large']))
#########################################################
def lecture_pivot_tables():
    # # Pivot Tables
    
    # In[ ]:
    
    #http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
    df = pd.read_csv('cars.csv')


    # In[ ]:

    df.head()
    

    # In[ ]:

    df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)
    

    # In[ ]:

    df.pivot_table(values='(kW)', index='YEAR', columns='Make', \
                   aggfunc=[np.mean,np.min], margins=True)




#########################################################
def lecture_date() :
    # # Date Functionality in Pandas
    pd.Timestamp('9/1/2016 10:05AM')
    # ### Period
    
    # In[4]:

    pd.Period('1/2016')


    # In[5]:

    pd.Period('3/5/2016')


    # ### DatetimeIndex
    
    # In[6]:
    
    t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
    t1
    
    
    # In[7]:
    
    type(t1.index)

    
    # ### PeriodIndex
    
    # In[8]:

    t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
    t2

    
    # In[9]:
    
    type(t2.index)


    # ### Converting to Datetime

    # In[10]:
    
    d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
    ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
    ts3
    
    
    # In[11]:
    
    ts3.index = pd.to_datetime(ts3.index)
    ts3


    # In[12]:
    
    pd.to_datetime('4.7.12', dayfirst=True)


    # ### Timedeltas

    # In[13]:

    pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')


    # In[14]:

    pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')


    # ### Working with Dates in a Dataframe

    # In[15]:
    
    dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')
    dates


    # In[16]:

    df = pd.DataFrame({'Count 1': 100 + np.random.randint(-5, 10, 9).cumsum(),\
                       'Count 2': 120 + np.random.randint(-5, 10, 9)}, index=dates)
    df


    # In[17]:

    df.index.weekday_name

    
    # In[18]:

    df.diff()

    
    # In[19]:

    df.resample('M').mean()

    
    # In[20]:
    
    df['2017']
    

    # In[21]:
    
    df['2016-12']


    # In[22]:

    df['2016-12':]


    # In[ ]:

    df.asfreq('W', method='ffill')


    # In[ ]:

    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib inline')
    
    df.plot()

##################################

########################################################
#lecture_merging_dataframes()
#lecture_pandas_idioms()

# lecture_groupby()
#lecture_scales()
#lecture_pivot_table()
#lecture_dates()

#lecture_merging_dataframes()
#lecture_pandas_idioms()
