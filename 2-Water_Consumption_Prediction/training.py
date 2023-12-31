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
   waterMeasuredX = np.array(df[['hour', 'waterMeasured']].values)
   waterMeasuredY = np.array(df['waterMeasured'].values)
   print(waterMeasuredX)
   print(waterMeasuredY)

   X = np.lib.stride_tricks.sliding_window_view(waterMeasuredX[:len(waterMeasuredY) - previous_examples], (previous_examples,2))
   Y = np.lib.stride_tricks.sliding_window_view(waterMeasuredY[previous_examples:], (future_predictions,))
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

def predict_with_LSTM(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  # Message of indentification
  print("Prediction using LSTM is being made...")

  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    LSTM(32, input_shape=(number_of_inputs, X_train[0].shape[1], 2), dropout=0.1, recurrent_dropout=0.5, return_sequences=True),
    LSTM(64, activation='relu', dropout=0.1, recurrent_dropout=0.5),
    #LSTM(100, activation='relu'),
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

def predict_with_GRU(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  # Message of indentification
  print("Prediction using GRU is being made...")

  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    GRU(32, input_shape=(number_of_inputs, X_train[0].shape[1], 2), dropout=0.1, recurrent_dropout=0.5, return_sequences=True),
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

def predict_with_RNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  # Message of indentification
  print("Prediction using Simple RNN is being made...")
  X_train = X_train.reshape(len(X_train), 168, 2)
  X_val = X_val.reshape(len(X_val), 168, 2)
  print(X_train.shape)

  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    GRU(32, input_shape=(number_of_inputs, 2), dropout=0.1, recurrent_dropout=0.5, return_sequences=True),
    GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5),
    #SimpleRNN(32, activation='relu'),
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
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return (lstm_train_results, lstm_model_results, y_val)

def predict_with_CNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  print("Prediction using Convolutional Neural Networks is being made...")
  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  X_train = X_train.reshape(len(X_train), 168, 2)
  X_val = X_val.reshape(len(X_val), 168, 2)

  # Model creation
  lstm_model = Sequential([
    Conv1D(32, (number_of_inputs), activation='relu', input_shape=(168, 2)),
    Flatten(),
    Dense(number_of_outputs)]
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
  lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping_monitor])
  loss, accuracy = lstm_model.evaluate(X_val, y_val)
  # Imprimir las métricas
  print("Loss:", loss)
  print("Accuracy:", accuracy)




  # Predicting and saving results

  lstm_model_results = lstm_model.predict(X_val)

  print(lstm_model_results.shape)
  print(y_val.shape)
  print("Predicho: "+str(np.sum((lstm_model_results[0])[lstm_model_results[0] > 0])))
  print("Real: "+str(np.sum((y_val[0])[y_val[0] > 0])))
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return (lstm_train_results, lstm_model_results, y_val)



  ###################################################
  ###################################################
def predict_with_CNN_SimpleRNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  print("Prediction using CNN-SimpleRNN is being made...")
  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    Conv1D(32, (number_of_inputs), activation='relu', input_shape=(number_of_inputs, X_train[0].shape[1], 2)),
    #Flatten(),
    SimpleRNN(32),
    Dense(number_of_outputs)]
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
  lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping_monitor])
  loss, accuracy = lstm_model.evaluate(X_test, y_test)
  # Imprimir las métricas
  print("Loss:", loss)
  print("Accuracy:", accuracy)




  # Predicting and saving results

  lstm_model_results = lstm_model.predict(X_val)
  

  print("Predicho: "+str(np.sum((lstm_model_results[0])[lstm_model_results[0] > 0])))
  print("Real: "+str(np.sum((y_val[0])[y_val[0] > 0])))
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return (lstm_train_results, lstm_model_results, y_val)



def predict_with_CNN_LSTM(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  print("Prediction using CNN-LSTM is being made...")
  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    Conv1D(32, (number_of_inputs), activation='relu', input_shape=(number_of_inputs, X_train[0].shape[1], 2)),
    #Flatten(),
    LSTM(32),
    Dense(number_of_outputs)]
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
  lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping_monitor])
  loss, accuracy = lstm_model.evaluate(X_test, y_test)
  # Imprimir las métricas
  print("Loss:", loss)
  print("Accuracy:", accuracy)



  # Predicting and saving results

  lstm_model_results = lstm_model.predict(X_val)

  print(lstm_model_results.shape)
  print(y_val.shape)
  print("Predicho: "+str(np.sum((lstm_model_results[0])[lstm_model_results[0] > 0])))
  print("Real: "+str(np.sum((y_val[0])[y_val[0] > 0])))
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return (lstm_train_results, lstm_model_results, y_val)



def predict_with_CNN_GRU(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  print("Prediction using CNN-GRU is being made...")
  number_of_inputs = int(X_train.shape[1])
  number_of_outputs = int(y_train.shape[1])

  # Model creation
  lstm_model = Sequential([
    Conv1D(32, (number_of_inputs), activation='relu', input_shape=(number_of_inputs, X_train[0].shape[1], 2)),
    #Flatten(),
    GRU(32),
    Dense(number_of_outputs)]
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
  lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping_monitor])
  loss, accuracy = lstm_model.evaluate(X_test, y_test)
  # Imprimir las métricas
  print("Loss:", loss)
  print("Accuracy:", accuracy)



  # Predicting and saving results

  lstm_model_results = lstm_model.predict(X_val)

  print(lstm_model_results.shape)
  print(y_val.shape)
  print("Predicho: "+str(np.sum((lstm_model_results[0])[lstm_model_results[0] > 0])))
  print("Real: "+str(np.sum((y_val[0])[y_val[0] > 0])))
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lstm_model_results[0], 'Real Values':y_val[0]})

  return (lstm_train_results, lstm_model_results, y_val)
###################################################
###################################################

def predict_with_MLP(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=100):
  print("Prediction using MLP is being made...")
  mlp_model = MLPRegressor(solver='lbfgs', early_stopping=True, max_iter=10000).fit(X_train, y_train)

  mlp_model_results = mlp_model.predict(X_val)

  mlr_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':mlp_model_results[0], 'Real Values':y_val[0]})

  return (mlr_train_results, mlp_model_results, y_val)

def predict_with_random_forest(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using Random Forest is being made...")

  rf_model = RandomForestRegressor().fit(X_train, y_train)

  rf_model_results = rf_model.predict(X_val)

  rf_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':rf_model_results[0], 'Real Values':y_val[0]})

  return (rf_train_results, rf_model_results, y_val)

def predict_with_SVR(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using SVR is being made...")
  svr_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
  multioutput_regressor = MultiOutputRegressor(svr_model).fit(X_train, y_train)
  svr_model_results = multioutput_regressor.predict(X_val)

  svr_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':svr_model_results[0], 'Real Values':y_val[0]})

  return (svr_train_results, svr_model_results, y_val)

def predict_with_MLR(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using MLR is being made...")
  mlr_model = LinearRegression().fit(X_train, y_train)

  mlr_model_results = mlr_model.predict(X_val)

  mlr_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':mlr_model_results[0], 'Real Values':y_val[0]})

  return (mlr_train_results, mlr_model_results, y_val)

def predict_with_Lasso(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using Lasso Regression is being made...")
  lasso_model = Lasso().fit(X_train, y_train)
  lasso_model_results = lasso_model.predict(X_val)

  lasso_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':lasso_model_results[0], 'Real Values':y_val[0]})

  return (lasso_train_results, lasso_model_results, y_val)

def predict_with_Ridge(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using Ridge Regression is being made...")
  ridge_model = Ridge().fit(X_train, y_train)

  ridge_model_results = ridge_model.predict(X_val)

  ridge_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':ridge_model_results[0], 'Real Values':y_val[0]})

  return (ridge_train_results, ridge_model_results, y_val)

def predict_with_Elastic(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using Elastic Regression is being made...")
  elastic_model = ElasticNet().fit(X_train, y_train)

  elastic_model_results = elastic_model.predict(X_val)

  elastic_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':elastic_model_results[0], 'Real Values':y_val[0]})

  return (elastic_train_results, elastic_model_results, y_val)

def predict_with_extraTrees(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using ExtraTrees is being made...")
  et_model = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
  et_model_results = et_model.predict(X_val)

  et_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':et_model_results[0], 'Real Values':y_val[0]})

  return (et_train_results, et_model_results, y_val)

def predict_with_kNN(dates_train, dates_val, X_train, y_train, X_val, y_val):
  print("Prediction using k-NN is being made...")
  knn_model = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
  knn_model_results = knn_model.predict(X_val)

  knn_train_results = pd.DataFrame(data={'Timestamp':dates_val[0], 'Predicted Values':knn_model_results[0], 'Real Values':y_val[0]})

  return (knn_train_results, knn_model_results, y_val)

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
df_historical_data_granularity['hour'] = df_historical_data_granularity.index.floor('60Min')
df_historical_data = df_historical_data_granularity.groupby(['hour']).sum().iloc[1:,:]
df_historical_data['hour'] = df_historical_data.index.hour
#print(df_historical_data)

#df_historical_data = deleting_outliers(df_historical_data)

# Plotting data
#df_historical_data.plot(y=["waterMeasured"])
#plt.title("Historical water consumption")
#plt.show()

# Normalize
#scaler = RobustScaler()
#df_historical_data = pd.DataFrame(scaler.fit_transform(df_historical_data), columns=['waterMeasured'])
#df_historical_data['waterMeasured'] = scaler.fit_transform(df_historical_data['waterMeasured'])
#df_historical_data['waterMeasured'] = (df_historical_data['waterMeasured'] - df_historical_data['waterMeasured'].min()) / (df_historical_data['waterMeasured'].max() - df_historical_data['waterMeasured'].min())

# Plotting data
df_historical_data.plot(y=["waterMeasured"])
plt.title("Historical water consumption")
plt.show()


# Showing data sample
print(df_historical_data)


# Defining amount of previous data to use to create the supervised problem
PREVIOUS_DATA = 168
FUTURE_DATA = 168

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
epochs = 10000


# Getting results from every model

#CNN
(cnn_results, pred_val_cnn, real_val_cnn) = predict_with_CNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)
get_performance(pred_val_cnn, real_val_cnn, "RNN")
cnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Recurrent Neural Network results')
plt.show()

#CNN-RNN
#(cnn_rnn_results, pred_val_cnn_rnn, real_val_cnn_rnn) = predict_with_CNN_SimpleRNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)

#CNN-LSTM
#(cnn_lstm_results, pred_val_cnn_lstm, real_val_cnn_lstm) = predict_with_CNN_LSTM(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)

#CNN-GRU
#(cnn_gru_results, pred_val_cnn_gru, real_val_cnn_gru) = predict_with_CNN_GRU(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)

#################################################


#RNN
(rnn_results, pred_val_rnn, real_val_rnn) = predict_with_RNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)
get_performance(pred_val_rnn, real_val_rnn, "RNN")
rnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Recurrent Neural Network results')
plt.show()

#GRU
(gru_results, pred_val_gru, real_val_gru) = predict_with_GRU(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)

#LSTM
(lstm_results, pred_val_lstm, real_val_lstm) = predict_with_LSTM(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)


# MLP
(mlp_results, pred_val_mlp, real_val_mlp) = predict_with_MLP(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs=epochs)

# MLR
(mlr_results, pred_val_mlr, real_val_mlr) = predict_with_MLR(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Ridge
(ridge_results, pred_val_ridge, real_val_ridge) = predict_with_Ridge(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Lasso
(lasso_results, pred_val_lasso, real_val_lasso) = predict_with_Lasso(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Elastic
(elastic_results, pred_val_elastic, real_val_elastic) = predict_with_Elastic(dates_train, dates_val, X_train, y_train, X_val, y_val)


# SVR
(svr_results, pred_val_svr, real_val_svr) = predict_with_SVR(dates_train, dates_val, X_train, y_train, X_val, y_val)



# Random Forest
(rf_results, pred_val_rf, real_val_rf) = predict_with_random_forest(dates_train, dates_val, X_train, y_train, X_val, y_val)

# ExtraTrees
(et_results, pred_val_et, real_val_et) = predict_with_extraTrees(dates_train, dates_val, X_train, y_train, X_val, y_val)

# kNN
(knn_results, pred_val_knn, real_val_knn) = predict_with_kNN(dates_train, dates_val, X_train, y_train, X_val, y_val)






# Performance of every model

get_performance(pred_val_rf, real_val_rf, "Random Forest")
get_performance(pred_val_et, real_val_et, "ExtraTrees")
get_performance(pred_val_knn, real_val_knn, "k-NN")

get_performance(pred_val_svr, real_val_svr, "SVR")

get_performance(pred_val_mlr, real_val_mlr, "MLR")
get_performance(pred_val_mlp, real_val_mlp, "MLP")

get_performance(pred_val_lasso, real_val_lasso, "Lasso")
get_performance(pred_val_ridge, real_val_ridge, "Ridge")
get_performance(pred_val_elastic, real_val_elastic, "Elastic")

get_performance(pred_val_rnn, real_val_rnn, "RNN")
get_performance(pred_val_gru, real_val_gru, "GRU")
get_performance(pred_val_lstm, real_val_lstm, "LSTM")

get_performance(pred_val_cnn, real_val_cnn, "CNN")

get_performance(pred_val_cnn_rnn, real_val_cnn_rnn, "CNN-RNN")
get_performance(pred_val_cnn_gru, real_val_cnn_gru, "CNN-GRU")
get_performance(pred_val_cnn_lstm, real_val_cnn_lstm, "CNN-LSTM")

# Plotting

#cnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
cnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('CNN results')
plt.show()

#lstm_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('LSTM results')
#plt.show()

#rnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Recurrent Neural Network results')
#plt.show()

#gru_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Gated Recurrent Unit results')
#plt.show()

#mlp_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Multi-layer Perceptron results')
#plt.show()

#mlr_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('MLR results')
#plt.show()

#cnn_lstm_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('CNN-LSTM results')
#plt.show()

#cnn_rnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('CNN-SimpleRNN results')
#plt.show()

#cnn_gru_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('CNN-GRU results')
#plt.show()

#rf_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
#plt.title('Random Forest results')
#plt.show()
