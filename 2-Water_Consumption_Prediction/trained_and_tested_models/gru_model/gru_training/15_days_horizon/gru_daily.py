import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
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
from keras.callbacks import EarlyStopping
from keras.losses import *
from keras.metrics import RootMeanSquaredError, F1Score, R2Score
from keras.optimizers import *

import pickle

# Defining helper functions

def iqr(df, minimum_value_allowed = 0):
    # IQR
    Q1 = np.percentile(df['waterMeasured'][df['waterMeasured'] > minimum_value_allowed], 25)
    Q3 = np.percentile(df['waterMeasured'][df['waterMeasured'] > minimum_value_allowed], 75)
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

   X = np.lib.stride_tricks.sliding_window_view(waterMeasured[:len(waterMeasured) - previous_examples], (previous_examples,))
   Y = np.lib.stride_tricks.sliding_window_view(waterMeasured[previous_examples:], (future_predictions,))
   dates_targets = np.lib.stride_tricks.sliding_window_view(dates[previous_examples:], (future_predictions,))

   return dates_targets, X, Y

def get_performance(predicted_values, real_values, model_name):
  rmse = sklearn.metrics.mean_squared_error(real_values, predicted_values,squared=False)
  mae = sklearn.metrics.mean_absolute_error(real_values, predicted_values)
  mse = sklearn.metrics.mean_squared_error(real_values, predicted_values)
  r2 = sklearn.metrics.r2_score(real_values, predicted_values, multioutput='variance_weighted')

  print("\tMetrics of "+str(model_name))
  print("RMSE: "+str(rmse))
  print("MAE: "+str(mae))
  print("MSE: "+str(mse))
  print("R^2: "+str(r2))
  print()


# Defining models

def predict_with_GRU(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  # Message of indentification
  print("Prediction using GRU is being made...")

  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    GRU(32, input_shape=(number_of_inputs, 1), dropout=0.1, recurrent_dropout=0.5, return_sequences=True),
    GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5),
    Dense(number_of_outputs)
    ]
  )

  # Model summary
  lstm_model.summary()

 # Early Stopping Callback
  early_stopping_monitor = EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=25,         
    verbose=1,           
    restore_best_weights=True 
  )

  # Training model
  lstm_model.compile(loss=MeanAbsoluteError(), optimizer=RMSprop(), metrics=[RootMeanSquaredError()])
  #lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
  fit_model = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping_monitor])
  lstm_model.save('gru_model/gru_training/15_days_horizon/models/gru_daily_15_days_horizon.h5')

  # Graficar la pérdida (loss) durante el entrenamiento
  #plt.figure(figsize=(12, 6))
  #plt.plot(fit_model.history['root_mean_squared_error'], label='Validation Loss')
  #plt.xlabel('Epochs')
  #plt.ylabel('Loss')
  #plt.legend()
  #plt.show()

  # Predicting and saving results

  lstm_model_results = lstm_model.predict(X_val)

  print(lstm_model_results.shape)
  print(y_val.shape)
  print("Predicho: "+str(np.sum((lstm_model_results[0])[lstm_model_results[0] > 0])))
  print("Real: "+str(np.sum((y_val[0])[y_val[0] > 0])))
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return (lstm_train_results, lstm_model_results, y_val)



######################################################
#CODE
######################################################



# Open data
raw_historical_data= ("https://media.githubusercontent.com/media/DiegoC01/Innova2030-ML_for_Water_Consumption_Prediction/main/1-Data_Preprocessing/Working_with_historical_data/5-final_dataset/historical_data_v2.csv")
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
#print(df_historical_data_granularity)

# Converting to 15 minutes
df_historical_data_granularity['hour'] = df_historical_data_granularity.index.floor('1440Min')
df_historical_data = df_historical_data_granularity.groupby(['hour']).sum().iloc[1:,:]


# Plotting data
#df_historical_data.plot(y=["waterMeasured"])
#plt.title("Historical water consumption")
#plt.show()

# Plotting data
df_historical_data.plot(y=["waterMeasured"])
plt.title("Historical water consumption")
plt.show()


# Showing data sample
print(df_historical_data)


# Defining amount of previous data to use to create the supervised problem
PREVIOUS_DATA = 15
FUTURE_DATA = 15

# Creating data sets for analysis of ML techniques

dates, X, y = supervised_problem(df_historical_data, previous_examples=PREVIOUS_DATA, future_predictions=FUTURE_DATA)

# Dividing data
q_64 = int(len(dates) * .64)
q_80 = int(len(dates) * .8)

dates_train, X_train, y_train = dates[:q_64], X[:q_64], y[:q_64]
dates_val, X_val, y_val = dates[q_64:q_80], X[q_64:q_80], y[q_64:q_80]
dates_test, X_test, y_test = dates[q_80:], X[q_80:], y[q_80:]

# Sample 
print(X_train)
print(y_train)
print(X_train.shape)
print(y_train.shape)


#################################################
# PREDICTIONS
#################################################

# Defining epochs to be used in neural networks
epochs = 10000


# Getting results from every model

#GRU
(gru_results, pred_val_gru, real_val_gru) = predict_with_GRU(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)



get_performance(pred_val_gru, real_val_gru, "GRU")

gru_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Gated Recurrent Unit results')
plt.show()
