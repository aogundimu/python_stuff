#!/Applications/anaconda/bin/python

import pandas as pd
import numpy as np

def distributions():
    # Numbers from a numpy binomial distribution
    # parameter 1 - number of times we want it to run
    # parameter 2 - chances we get a 0
    #print( np.random.binomial(1, 0.5) )
    #print( np.random.binomial(1000, 0.5) / 1000 )

    # To simulate the probability of flipping a fair coin 20 times and
    # getting a number greater than or equal 15. Use np.random.binomial(n,p,size)
    # to do 10000 simulations of flipping a fair coin 20 times, then see what
    # proportion of the simulations are 15 or greater.
    x = np.random.binomial(20, .5, 10000)
    #print(x)
    #print((x>=15).mean())

    #chance of a tornado happening during our lecture
    chance_of_tornado = 0.01 / 100
    tornado_events = np.random.binomial(100000, chance_of_tornado)
    #print( tornado_events )

    #### Chances of tornado happening 2 days in a row
    chance_of_tornado = 0.01
    tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)
    two_days_in_a_row = 0
    for j in range(1, len(tornado_events)-1 ):
        if tornado_events[j]==1 and tornado_events[j-1]==1:
            two_days_in_a_row += 1

    print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))

#################################################    
def more_distributions():

    print( np.random.uniform(0, 1) )
    # 0.04741757055213702

    print( np.random.normal(0.75) )
    # -0.5375820887432665
           
    # Draw 1000 samples from a normal distribution with an
    # expected value of 0.75 and a standard deviatio of 1
    distribution = np.random.normal(0.75,size=1000)
    print( np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution)) )
    print( np.std(distribution) )

    import scipy.stats as stats
    ## Kurtosis
    print( stats.kurtosis(distribution) )

    ### Skew
    print( stats.skew(distribution) )

    ## Chi Squared Distribution
    ## 2 is the degree of freedom
    chi_squared_df2 = np.random.chisquare(2, size=10000)
    print( stats.skew(chi_squared_df2) )

    ## 5 is the degree of freedom
    chi_squared_df5 = np.random.chisquare(5, size=10000)
    print( stats.skew(chi_squared_df5) )

    import matplotlib
    import matplotlib.pyplot as plt

    output = plt.hist([chi_squared_df2,chi_squared_df5], bins=50, histtype='step', 
                      label=['2 degrees of freedom','5 degrees of freedom'])
    plt.legend(loc='upper right')
    plt.show()

def hypothesis_testing():
    df = pd.read_csv('grades.csv')
    #print( df.head() )

    early = df[df['assignment1_submission'] <= '2015-12-31']
    late = df[df['assignment1_submission'] > '2015-12-31']

    print( type(early) )
    print( type(late) )

    #print( early.mean() )
    #print( late.mean() )

    from scipy import stats
    stats.ttest_ind

    print( stats.ttest_ind(early['assignment1_grade'], late['assignment1_grade']) )
    # Ttest_indResult(statistic=1.400549944897566, pvalue=0.16148283016060577)

    print( stats.ttest_ind(early['assignment2_grade'], late['assignment2_grade']) )
    # Ttest_indResult(statistic=1.3239868220912567, pvalue=0.18563824610067967)
    
    print( stats.ttest_ind(early['assignment3_grade'], late['assignment3_grade']) )
    # Ttest_indResult(statistic=1.7116160037010733, pvalue=0.087101516341556676)
    
###########################
# distributions()
# more_distributions()
hypothesis_testing()
