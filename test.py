
from rfhack import AdversarialHacker


import pandas


df = pandas.read_csv('missForest.csv')
hacker = AdversarialHacker(df)
r = hacker.hack(0.7)

for name, value in zip(r._fields, r):
    print(name, value)

