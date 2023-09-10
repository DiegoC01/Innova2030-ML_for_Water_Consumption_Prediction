import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

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


import keras.backend as K
from keras.layers import Layer

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res


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

def df_to_windowed_df(dataframe, n=3, every_x_minute=1):
  first_date = dataframe.index[n+1]
  last_date  = dataframe.index[-1]

  target_date = first_date
  
  dates = []
  X, Y = [], []

  last_time = False
  try:
    while True:
      df_subset = dataframe.loc[:target_date].tail(n+1)
      
      if len(df_subset) != n+1:
        print(f'Error: Window of size {n} is too large for date {target_date}')
        return

      values = df_subset['waterMeasured'].to_numpy()
      x, y = values[:-1], values[-1]

      dates.append(target_date)
      X.append(x)
      Y.append(y)

      next_date = target_date+timedelta(minutes=every_x_minute)
      
      if last_time:
        break
      
      target_date = next_date

      if target_date == last_date:
        last_time = True

  except:
    pass 
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1]))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

def get_performance(predicted_values, real_values, model_name):
  rmse = sklearn.metrics.mean_squared_error(real_values, predicted_values,squared=True)
  mae = sklearn.metrics.mean_absolute_error(real_values, predicted_values)
  r2 = sklearn.metrics.r2_score(real_values, predicted_values)

  print("\tMetrics of "+str(model_name))
  print("RMSE: "+str(rmse))
  print("MAE: "+str(mae))
  print("Coefficient of determination: "+str(r2))
  print()

def predict_with_random_forest(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using Random Forest is being made...")
  rf_model = RandomForestRegressor(random_state=0, n_jobs=-1).fit(X_train, y_train)
  rf_model_results = rf_model.predict(X_test)

  rf_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':rf_model_results, 'Real Values':y_test})

  return rf_train_results

def predict_with_extraTrees(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using ExtraTrees is being made...")
  et_model = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
  et_model_results = et_model.predict(X_test)

  et_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':et_model_results, 'Real Values':y_test})

  return et_train_results

def predict_with_kNN(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using k-NN is being made...")
  knn_model = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
  knn_model_results = knn_model.predict(X_test)

  knn_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':knn_model_results, 'Real Values':y_test})

  return knn_train_results

def predict_with_LSTM(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=10):
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

def predict_with_RBF(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=10):
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




# Open data
raw_historical_data= ("C:/Users/diego/OneDrive/Escritorio/Innova 2030/Datos/Datos_completos/5-final_dataset/historical_data_test.csv")
df_historical_data = pd.read_csv(raw_historical_data)

# Defining timestamp
df_historical_data['time'] = pd.to_numeric(df_historical_data['time'])
df_historical_data['time_format'] = (pd.to_datetime(df_historical_data['time'],unit='s', dayfirst=True) .dt.tz_localize('utc').dt.tz_convert('America/Santiago'))
df_historical_data.index = df_historical_data['time']


# Converting waterMeasured to numerical
df_historical_data['waterMeasured'] = pd.to_numeric(df_historical_data['waterMeasured'])

# Testing different granularities
df_historical_data['time_format_granularity'] = df_historical_data['time_format'].dt.floor('Min')
df_historical_data_granularity = df_historical_data.groupby(['time_format_granularity']).sum().iloc[1:,:]

# Plotting data
df_historical_data_granularity.plot(y=["waterMeasured"])
plt.show()

print(df_historical_data_granularity)
df_historical_data_granularity = deleting_outliers(df_historical_data_granularity)

# Defining amount of previous data to use to create the supervised problem
print("Defining window!")
WINDOW_SIZE = 3

# Creating data sets for analysis of ML techniques
print("Creating sets!")
windowed_df = df_to_windowed_df(df_historical_data_granularity, n=WINDOW_SIZE)
print(windowed_df)
dates, X, y = windowed_df_to_date_X_y(windowed_df)

q_64 = int(len(dates) * .64)
q_80 = int(len(dates) * .8)

dates_train, X_train, y_train = dates[:q_64], X[:q_64], y[:q_64]
dates_val, X_val, y_val = dates[q_64:q_80], X[q_64:q_80], y[q_64:q_80]
dates_test, X_test, y_test = dates[q_80:], X[q_80:], y[q_80:]

plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(['Train', 'Validation', 'Test'])
plt.show()


# Predictions
# Random Forest
rf_results = predict_with_random_forest(dates_train, dates_test, X_train, y_train, X_test, y_test)
rf_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.show()

# ExtraTrees
et_results = predict_with_extraTrees(dates_train, dates_test, X_train, y_train, X_test, y_test)
et_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.show()

# kNN
knn_results = predict_with_kNN(dates_train, dates_test, X_train, y_train, X_test, y_test)
knn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.show()

#LSTM
lstm_results = predict_with_LSTM(dates_train, dates_test, X_train, y_train, X_test, y_test)
lstm_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.show()



# Performance of every model
get_performance(rf_results['Predicted Values'], rf_results['Real Values'], "Random Forest")
get_performance(et_results['Predicted Values'], et_results['Real Values'], "ExtraTrees")
get_performance(lstm_results['Predicted Values'], lstm_results['Real Values'], "LSTM")
get_performance(knn_results['Predicted Values'], knn_results['Real Values'], "k-NN")
