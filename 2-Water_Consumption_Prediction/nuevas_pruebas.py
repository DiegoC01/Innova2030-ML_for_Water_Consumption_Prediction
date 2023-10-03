import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, F1Score, R2Score
from keras.optimizers import Adam

# Defining helper functions

def iqr(df, minimum_value_allowed = 2):
    # IQR
    Q1 = np.percentile(df['waterMeasured'][df['waterMeasured']>=minimum_value_allowed], 25)
    Q3 = np.percentile(df['waterMeasured'][df['waterMeasured']>=minimum_value_allowed], 75)
    IQR = Q3 - Q1

    # Above Upper bound
    upper=Q3+1.5*IQR
    upper_array=np.array(df['waterMeasured']>=upper)
    
    #Below Lower bound
    lower=Q1-1.5*IQR
    lower_array=np.array(df['waterMeasured']<=lower)

    return (upper, lower)

def deleting_outliers(df):
    
    # Identifying outliers with the iqr function
    (upper, lower) = iqr(df)

    # Deleting outliers
    df['waterMeasured'] = np.where(df['waterMeasured'] > upper, None, df['waterMeasured'])
    df['waterMeasured'] = np.where(df['waterMeasured'] < lower, None, df['waterMeasured'])

    # applying the method
    count_nan = df['waterMeasured'].isnull().sum()
    
    # printing the number of values present
    # in the column
    print('Number of NaN values present: ' + str(count_nan))

    df['waterMeasured'] = df['waterMeasured'].ffill()
    print(df)
    # applying the method
    count_nan = df['waterMeasured'].isnull().sum()
    
    # printing the number of values present
    # in the column
    print('Number of NaN values present: ' + str(count_nan))

    return df

def stationarity_test(df):
    # Stationarity test
    print("Observations of Dickey-fuller test")
    dftest = adfuller(df["waterMeasured"],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['critical value (%s)'%key]= value
    print(dfoutput)

def supervised_problem(df, previous_examples=2, future_predictions=4):
   dates = np.array(df.index)
   last_date = dates[-1]
   waterMeasured = np.array(df['waterMeasured'].values)

   X = np.lib.stride_tricks.sliding_window_view(waterMeasured[1:], (previous_examples,))
   Y = np.lib.stride_tricks.sliding_window_view(waterMeasured[:len(waterMeasured) - 1], (future_predictions,))
   dates_targets = np.lib.stride_tricks.sliding_window_view(dates, (future_predictions,))

   return dates_targets, X, Y

def get_performance(predicted_values, real_values, model_name):
  rmse = sklearn.metrics.mean_squared_error(real_values, predicted_values,squared=True)
  mae = sklearn.metrics.mean_absolute_error(real_values, predicted_values)
  r2 = sklearn.metrics.r2_score(real_values, predicted_values)

  print("\tMetrics of "+str(model_name))
  print("RMSE: "+str(rmse))
  print("MAE: "+str(mae))
  print("Coefficient of determination: "+str(r2))
  print()


# Defining models

def predict_with_LSTM(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  # Message of indentification
  print("Prediction using LSTM is being made...")

  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    LSTM(64, input_shape=(number_of_inputs, 1)),
    Dense(number_of_outputs)]
  )

  # Model summary
  lstm_model.summary()

  # Training model
  lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
  #lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
  lstm_model.fit(X_train, y_train, epochs=epochs)

  # Predicting and saving results
  lstm_model_results = lstm_model.predict(X_val)
  print(lstm_model_results.shape)
  print(y_val.shape)
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return lstm_train_results

def predict_with_GRU(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=100):
  print("Prediction using GRU is being made...")
  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    GRU(64, input_shape=(number_of_inputs, 1)),
    Dense(number_of_outputs)]
  )

  # Model summary
  lstm_model.summary()

  # Training model
  lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
  #lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
  lstm_model.fit(X_train, y_train, epochs=epochs)

  # Predicting and saving results
  lstm_model_results = lstm_model.predict(X_val)
  print(lstm_model_results.shape)
  print(y_val.shape)
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return lstm_train_results

def predict_with_RNN(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=100):
  print("Prediction using Recurrent Neural Network (simple) is being made...")
  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    SimpleRNN(64, input_shape=(number_of_inputs, 1)),
    Dense(number_of_outputs)]
  )

  # Model summary
  lstm_model.summary()

  # Training model
  lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
  #lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
  lstm_model.fit(X_train, y_train, epochs=epochs)

  # Predicting and saving results
  lstm_model_results = lstm_model.predict(X_val)
  print(lstm_model_results.shape)
  print(y_val.shape)
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return lstm_train_results

def predict_with_CNN(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=100):
  print("Prediction using Convolutional Neural Networks is being made...")
  CONV_WIDTH = 3
  OUT_STEPS = 24
  num_features = X_train.shape[1]

  cnn_model = Sequential()
  cnn_model.add(InputLayer((X_train.shape[1], 1)))
  #cnn_model.add(Conv1D(256, activation='relu', kernel_size=(X_train.shape[1])))
  #cnn_model.add(Dense(1, kernel_initializer=tf.initializers.zeros()))
  cnn_model.add(Conv1D(32, (3), activation='relu'))
  cnn_model.add(Flatten())
  cnn_model.add(Dense(64, activation='relu'))
  cnn_model.add(Dense(1))

  #cnn_model.summary()

  cnn_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
  cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_value)

  cnn_model_results = cnn_model.predict(X_test).flatten()
  cnn_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':cnn_model_results, 'Real Values':y_test})

  return cnn_train_results

def predict_with_MLP(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using MLP is being made...")
  mlp_model = MLPRegressor().fit(X_train, y_train)
  mlp_model_results = mlp_model.predict(X_val)
  print(mlp_model_results)

  mlr_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':mlp_model_results[0], 'Real Values':y_val[0]})

  return mlr_train_results

def predict_with_random_forest(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using Random Forest is being made...")

  rf_model = RandomForestRegressor().fit(X_train, y_train)
  rf_model_results = rf_model.predict(X_val)

  rf_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':rf_model_results[0], 'Real Values':y_val[0]})

  return rf_train_results

def predict_with_SVR(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using SVR is being made...")
  svr_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1).fit(X_train, y_train)
  svr_model_results = svr_model.predict(X_val)

  svr_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':svr_model_results[0], 'Real Values':y_val[0]})

  return svr_train_results

def predict_with_MLR(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using MLR is being made...")
  mlr_model = LinearRegression().fit(X_train, y_train)
  mlr_model_results = mlr_model.predict(X_val)

  mlr_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':mlr_model_results[0], 'Real Values':y_val[0]})

  return mlr_train_results

def predict_with_Lasso(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using Lasso Regression is being made...")
  lasso_model = Lasso().fit(X_train, y_train)
  lasso_model_results = lasso_model.predict(X_val)

  lasso_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lasso_model_results[0], 'Real Values':y_val[0]})

  return lasso_train_results

def predict_with_Ridge(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using Ridge Regression is being made...")
  ridge_model = Ridge().fit(X_train, y_train)
  ridge_model_results = ridge_model.predict(X_val)

  ridge_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':ridge_model_results[0], 'Real Values':y_val[0]})

  return ridge_train_results

def predict_with_Elastic(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using Elastic Regression is being made...")
  elastic_model = ElasticNet().fit(X_train, y_train)
  elastic_model_results = elastic_model.predict(X_val)

  elastic_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':elastic_model_results[0], 'Real Values':y_val})

  return elastic_train_results

def predict_with_extraTrees(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using ExtraTrees is being made...")
  et_model = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
  et_model_results = et_model.predict(X_val)

  et_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':et_model_results[0], 'Real Values':y_val})

  return et_train_results

def predict_with_kNN(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using k-NN is being made...")
  knn_model = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
  knn_model_results = knn_model.predict(X_val)

  knn_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':knn_model_results[0], 'Real Values':y_val})

  return knn_train_results

######################################################
#CODE
######################################################



# Open data
raw_historical_data= ("1-Data_Preprocessing/Working_with_historical_data/5-final_dataset/historical_data.csv")
df_historical_data = pd.read_csv(raw_historical_data)

# Defining timestamp
df_historical_data['time'] = pd.to_numeric(df_historical_data['time'])
df_historical_data['time_format'] = (pd.to_datetime(df_historical_data['time'],unit='s', dayfirst=True) .dt.tz_localize('utc').dt.tz_convert('America/Santiago'))
df_historical_data.index = df_historical_data['time_format']


# Converting waterMeasured to numerical
df_historical_data['waterMeasured'] = pd.to_numeric(df_historical_data['waterMeasured'])

# Testing different granularities
df_historical_data['time_format_granularity'] = df_historical_data['time_format'].dt.floor('Min')
df_historical_data_granularity = df_historical_data.groupby(['time_format_granularity']).mean().iloc[1:,:]
df_historical_data_granularity = df_historical_data_granularity.drop(columns=['time'])
print(df_historical_data_granularity)


# Plotting data
df_historical_data_granularity.plot(y=["waterMeasured"])
plt.title("Historical water consumption")
plt.show()

# Showing data sample
print(df_historical_data_granularity)


# Defining amount of previous data to use to create the supervised problem
PREVIOUS_DATA = 3
FUTURE_DATA = 1

# Creating data sets for analysis of ML techniques

dates, X, y = supervised_problem(df_historical_data_granularity, previous_examples=PREVIOUS_DATA, future_predictions=FUTURE_DATA)

# Dividing data
q_64 = int(len(dates) * .64)
q_80 = int(len(dates) * .8)

dates_train, X_train, y_train = dates[:q_64], X[:q_64], y[:q_64]
dates_val, X_val, y_val = dates[q_64:q_80], X[q_64:q_80], y[q_64:q_80]
dates_test, X_test, y_test = dates[q_80:], X[q_80:], y[q_80:]

# Sample 
print(X_train)
print(y_train)

# Plotting subsets
#plt.plot(y_train)
#plt.plot(y_val)
#plt.plot(y_test)
#plt.legend(['Train', 'Validation', 'Test'])
#plt.show()

#################################################
# PREDICTIONS
#################################################

# Defining epochs to be used in neural networks
epochs = 25


# Getting results from every model

# CNN
#cnn_results = predict_with_CNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs_value=epoch)

# RNN
rnn_results = predict_with_RNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs_value=epochs)

# GRU
gru_results = predict_with_GRU(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs_value=epochs)

# MLP
#mlp_results = predict_with_MLP(dates_train, dates_val, X_train, y_train, X_val, y_val)

# SVR
#svr_results = predict_with_SVR(dates_train, dates_val, X_train, y_train, X_val, y_val)

# MLR
#mlr_results = predict_with_MLR(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Random Forest
#rf_results = predict_with_random_forest(dates_train, dates_val, X_train, y_train, X_val, y_val)

# ExtraTrees
#et_results = predict_with_extraTrees(dates_train, dates_val, X_train, y_train, X_val, y_val)

# kNN
#knn_results = predict_with_kNN(dates_train, dates_val, X_train, y_train, X_val, y_val)

#LSTM
lstm_results = predict_with_LSTM(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs_value=epochs)

# Ridge
#ridge_results = predict_with_Ridge(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Lasso
#lasso_results = predict_with_Lasso(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Elastic
#elastic_results = predict_with_Elastic(dates_train, dates_val, X_train, y_train, X_val, y_val)


# Performance of every model
#get_performance(rf_results['Predicted Values'], rf_results['Real Values'], "Random Forest")
#get_performance(et_results['Predicted Values'], et_results['Real Values'], "ExtraTrees")
#get_performance(knn_results['Predicted Values'], knn_results['Real Values'], "k-NN")
#get_performance(svr_results['Predicted Values'], svr_results['Real Values'], "SVR")
#get_performance(mlr_results['Predicted Values'], mlr_results['Real Values'], "MLR")
#get_performance(mlp_results['Predicted Values'], mlp_results['Real Values'], "MLP")
#get_performance(lasso_results['Predicted Values'], lasso_results['Real Values'], "Lasso")
#get_performance(ridge_results['Predicted Values'], ridge_results['Real Values'], "Ridge")
#get_performance(elastic_results['Predicted Values'], elastic_results['Real Values'], "Elastic")
get_performance(rnn_results['Predicted Values'], rnn_results['Real Values'], "RNN")
get_performance(gru_results['Predicted Values'], gru_results['Real Values'], "GRU")
#get_performance(cnn_results['Predicted Values'], gru_results['Real Values'], "CNN")
get_performance(lstm_results['Predicted Values'], lstm_results['Real Values'], "LSTM")


# Plotting results
#rf_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Random Forest results')
#plt.show()

#et_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('ExtraTrees results')
#plt.show()

lstm_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('LSTM results')
plt.show()

#knn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('k-Nearest Neighbors results')
#plt.show()

#svr_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Support Vector Regression results')
#plt.show()

#mlr_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Multi-Linear Regression results')
#plt.show()

#mlp_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Multi-layer Perceptron results')
#plt.show()

#lasso_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Lasso Regression results')
#plt.show()

#ridge_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Ridge Regression results')
#plt.show()

#elastic_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Elastic Regression results')
#plt.show()

rnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Recurrent Neural Network results')
plt.show()

gru_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Gated Recurrent Unit results')
plt.show()

#cnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Convolutional Neural Network results')
#plt.show()
