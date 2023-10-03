import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def random_forest(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using Random Forest is being made...")
  rf_model = RandomForestRegressor(random_state=0, n_jobs=-1).fit(X_train, y_train)
  rf_model_results = rf_model.predict(X_test)

  rf_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':rf_model_results, 'Real Values':y_test})

  return rf_train_results

def extraTrees(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using ExtraTrees is being made...")
  et_model = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
  et_model_results = et_model.predict(X_test)

  et_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':et_model_results, 'Real Values':y_test})

  return et_train_results

def kNN(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using k-NN is being made...")
  knn_model = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
  knn_model_results = knn_model.predict(X_test)

  knn_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':knn_model_results, 'Real Values':y_test})

  return knn_train_results

def LSTM(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=10):
  print("Prediction using LSTM is being made...")
  lstm_model = Sequential()
  lstm_model.add(InputLayer((X_train.shape[1], 1)))
  lstm_model.add(LSTM(64))
  lstm_model.add(Dense(8, 'relu'))
  lstm_model.add(Dense(1, 'linear'))

  lstm_model.summary()

  lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
  lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_value)

  lstm_model_results = lstm_model.predict(X_test).flatten()
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':lstm_model_results, 'Real Values':y_test})

  return lstm_train_results

def RBF(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=10):
  print("Prediction using RBF is being made...")

  rbf_model = Sequential()
  rbf_model.add(InputLayer((X_train.shape[1], 1)))
  rbf_model.add(RBFLayer(64))
  rbf_model.add(Dense(8, 'relu'))
  rbf_model.add(Dense(1, 'linear'))
  

  rbf_model.summary()

  rbf_model.compile(loss='binary_crossentropy', optimizer='adam')
  rbf_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_value)

  rbf_model_results = rbf_model.predict(X_test).flatten()
  print(rbf_model_results)
  print(rbf_model_results.shape, y_test.shape, dates_test.shape)
  rbf_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':rbf_model_results, 'Real Values':y_test})

  return rbf_train_results
