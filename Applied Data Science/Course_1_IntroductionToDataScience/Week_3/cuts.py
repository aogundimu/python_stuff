#!/Applications/anaconda/bin/python

import numpy as np
import pandas as pd
np.random.seed(2015)

defT = pd.DataFrame({'AgeGrp': [1, 2, 3, 4, 5, 6, 7],
                     'MaxAge': [18, 21, 24, 34, 44, 54, 65],
                     'MinAge': [14, 19, 22, 25, 35, 45, 55]})

cutoff = np.hstack([np.array(defT['MinAge'][0]), defT['MaxAge'].values])

print(cutoff)

labels = defT['AgeGrp']

N = 50
df = pd.DataFrame(np.random.randint(100, size=(N,2)), columns=['Age', 'Year'])
df['ageGrp'] = pd.cut(df['Age'], bins=cutoff, labels=labels, include_lowest=True)

"""
grouped = df.groupby(['Year', 'ageGrp'], as_index=False)
final = grouped.agg(np.sum)
print(final)
"""

grouped = df.groupby(['Year', 'ageGrp'], as_index=True)
final = grouped.agg(np.sum).dropna()
#print(final)
