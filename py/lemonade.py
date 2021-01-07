import pandas as pd
import tensorflow as tf

dir='machine-learning-with-opentutorials\csv\lemonade.csv'
lemonade=pd.read_csv(dir)

print(lemonade.shape)
print(lemonade.columns)

independent=lemonade[['온도']]
dependent=lemonade[['판매량']]

print(independent.shape, dependent.shape)
print(lemonade.head())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

X=tf.keras.layers.Input(shape=[1])
Y=tf.keras.layers.Dense(1)(X)

model=tf.keras.models.Model(X,Y)
model.compile(loss="mse")
model.fit(independent, dependent, epochs=10000, verbose=0)

print(model.predict(independent)) # 모델에 독립변수를 넣어서 종속변수의 값과 가깝게 나오는지 확인
print(model.predict([[15]]))  # 모델에 원하는 독립변수 넣어보기