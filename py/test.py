import pandas as pd

dir = 'machine-learning-with-opentutorials/csv/boston.csv' # 데이터 디렉토리
boston = pd.read_csv(dir) # 위의 디렉토리에서 파일의 데이터 읽어오기

print(boston.shape) # 불러온 데이터의 모양 확인
print(boston.columns) # 불러운 데이터의 컬럼(특성) 확인
print(boston.head())  # 데이터의 전체 행렬

# 데이터의 독립변수
independent = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
# 데이터의 종속변수
dependent = boston[['medv']]

# 독립변수와 종속변수의 모양 확인
print(independent.shape)
print(dependent.shape)