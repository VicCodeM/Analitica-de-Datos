import numpy as np
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv('Coffe.csv')
# df
df['amount'] = 1

#Preprocesamiento
df_pivot = df.pivot_table(index='transaction_number', columns='item', values='amount', aggfunc='sum').fillna(0)
print('Dataset size: ', df_pivot.shape)

#print(df_pivot.astype(int))

df_pivot

df_pivot = df_pivot.astype(int)

def encode(x):
    if x <= 0:
        return 0
    else:
        return 1

df_pivot = df_pivot.applymap(encode)
#-----------------------------------------------#
support = 0.01
frequent_items = apriori(df_pivot, min_support=support, use_colnames=True)

metric = 'lift'

min_treshold = 1

#reglas de asosiacion con sus diferentes valores
rules = association_rules(frequent_items, metric=metric, min_threshold=min_treshold)

rules.reset_index(drop=True).sort_values('confidence', ascending=False, inplace=True)

rules