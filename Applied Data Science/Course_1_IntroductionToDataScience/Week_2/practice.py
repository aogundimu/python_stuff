#!/Applications/anaconda/bin/python

import pandas as pd

animals = ['Tiger', 'Bear', 'Moose']
s1 = pd.Series(animals)
print(s1)

numbers = [1, 2, 3, 4]
s2 = pd.Series(numbers)
print(s2)
