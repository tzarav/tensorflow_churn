from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import pandas as pd

print("loading data file...")
df = pd.read_csv('Churn.csv')

X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))

y = df['Churn'].apply( lambda x: 1 if x == 'Yes' else 0)

print("datafile loaded")
print("training model")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print("creating model...")
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=44, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

model.fit(X_train, y_train, epochs=200, batch_size=43)
y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

