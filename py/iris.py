import pandas as pd

dir='machine-learning-with-opentutorials\csv\iris.csv'
iris=pd.read_csv(dir)

print(iris.shape)
print(iris.columns)

independent=iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependent=iris[['품종']]

print(independent.shape, dependent.shape)
print(iris.head())