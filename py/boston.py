import pandas as pd
import tensorflow as tf

# 데이터 준비
dir = 'machine-learning-with-opentutorials/csv/boston.csv'
boston = pd.read_csv(dir)

print(boston.columns)
print(boston.head())

independent = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
dependent = boston[['medv']]

print(independent.shape)
print(dependent.shape)

# 모델 만들기
X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)

model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 모델 학습
model.fit(independent, dependent, epochs=10, verbose=0)

# 모델 예측값
prediction = model.predict(independent)

print(f"predicted house prices : {prediction[:10]}")
print(f"actual house prices : {dependent[:10]}")

# 모델의 수식 확인
formula = model.get_weights()

print(formula)

