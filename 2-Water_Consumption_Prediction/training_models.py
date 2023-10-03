import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter


from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

import sklearn
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
    df['waterMeasured'] = np.where(df['waterMeasured'] < lower, 0, df['waterMeasured'])

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
  i = 0
  try:
    while True:
      df_subset = dataframe.loc[:target_date].tail(n+1)
      i += 1
      
      if len(df_subset) != n+1:
        print(f'Error: Window of size {n} is too large for date {target_date}')
        return

      values = df_subset['waterMeasured'].to_numpy()
      x, y = values[:-1], values[-1]

      dates.append(target_date)
      X.append(x)
      Y.append(y)

      next_date = target_date+timedelta(minutes=1)
      
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

def predict_with_LSTM(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=100):
  print("Prediction using LSTM is being made...")
  lstm_model = Sequential()
  lstm_model.add(InputLayer((X_train.shape[1], 1)))
  lstm_model.add(LSTM(64))
  lstm_model.add(Dense(1))

  lstm_model.summary()

  lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
  lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_value)

  lstm_model_results = lstm_model.predict(X_test).flatten()
  lstm_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':lstm_model_results, 'Real Values':y_test})

  return lstm_train_results

def predict_with_SVR(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using SVR is being made...")
  svr_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1).fit(X_train, y_train)
  svr_model_results = svr_model.predict(X_test)

  svr_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':svr_model_results, 'Real Values':y_test})

  return svr_train_results

def predict_with_MLR(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using MLR is being made...")
  mlr_model = LinearRegression().fit(X_train, y_train)
  mlr_model_results = mlr_model.predict(X_test)

  mlr_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':mlr_model_results, 'Real Values':y_test})

  return mlr_train_results

def predict_with_MLP(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using MLP is being made...")
  mlp_model = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
  mlp_model_results = mlp_model.predict(X_test)

  mlr_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':mlp_model_results, 'Real Values':y_test})

  return mlr_train_results

def predict_with_Lasso(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using Lasso Regression is being made...")
  lasso_model = Lasso().fit(X_train, y_train)
  lasso_model_results = lasso_model.predict(X_test)

  lasso_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':lasso_model_results, 'Real Values':y_test})

  return lasso_train_results

def predict_with_Ridge(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using Ridge Regression is being made...")
  ridge_model = Ridge().fit(X_train, y_train)
  ridge_model_results = ridge_model.predict(X_test)

  ridge_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':ridge_model_results, 'Real Values':y_test})

  return ridge_train_results

def predict_with_Elastic(dates_train, dates_test, X_train, y_train, X_test, y_test):
  print("Prediction using Elastic Regression is being made...")
  elastic_model = ElasticNet().fit(X_train, y_train)
  elastic_model_results = elastic_model.predict(X_test)

  elastic_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':elastic_model_results, 'Real Values':y_test})

  return elastic_train_results

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

def predict_with_GRU(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=100):
  print("Prediction using GRU is being made...")
  gru_model = Sequential()
  gru_model.add(InputLayer((X_train.shape[1], 1)))
  gru_model.add(GRU(64))
  gru_model.add(Dense(1))

  gru_model.summary()

  gru_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
  gru_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_value)

  gru_model_results = gru_model.predict(X_test).flatten()
  gru_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':gru_model_results, 'Real Values':y_test})

  return gru_train_results

def predict_with_RNN(dates_train, dates_test, X_train, y_train, X_test, y_test, epochs_value=100):
  print("Prediction using Recurrent Neural Network (simple) is being made...")
  rnn_model = Sequential()
  rnn_model.add(InputLayer((X_train.shape[1], 1)))
  rnn_model.add(SimpleRNN(64))
  #rnn_model.add(Dense(8, 'relu'))
  #rnn_model.add(Dense(1, 'linear'))
  rnn_model.add(Dense(1))

  rnn_model.summary()

  rnn_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
  rnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_value)

  rnn_model_results = rnn_model.predict(X_test).flatten()
  rnn_train_results = pd.DataFrame(data={'Timestamp':dates_test, 'Predicted Values':rnn_model_results, 'Real Values':y_test})

  return rnn_train_results

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
plt.title("Water consumption per minute from 01-07-2023 to 01-09-2023")

months = MonthLocator()
months_fmt = DateFormatter('%B')
plt.gca().xaxis.set_major_locator(months)
plt.gca().xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=0) 
plt.show()


#df_historical_data_granularity['waterMeasured'] = (df_historical_data_granularity['waterMeasured'] - df_historical_data_granularity['waterMeasured'].min()) / (df_historical_data_granularity['waterMeasured'].max() - df_historical_data_granularity['waterMeasured'].min())


#stationarity_test(df_historical_data_granularity)

print(df_historical_data_granularity)
#df_historical_data_granularity = deleting_outliers(df_historical_data_granularity)


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

dates_train = np.array(dates_train, dtype='datetime64')
dates_val = np.array(dates_val, dtype='datetime64')
dates_test = np.array(dates_test, dtype='datetime64')

print("enetrenará con:")
print(X_train)
print(y_train)

plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

months = MonthLocator()
months_fmt = DateFormatter('%B')
plt.gca().xaxis.set_major_locator(months)
plt.gca().xaxis.set_major_formatter(months_fmt)
plt.legend(['Train', 'Validation', 'Test'])
plt.show()


# Predictions
epoch = 100

#CNN
cnn_results = predict_with_CNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs_value=epoch)

#RNN
rnn_results = predict_with_RNN(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs_value=epoch)

#GRU
gru_results = predict_with_GRU(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs_value=epoch)

# MLP
mlp_results = predict_with_MLP(dates_train, dates_val, X_train, y_train, X_val, y_val)

# SVR
svr_results = predict_with_SVR(dates_train, dates_val, X_train, y_train, X_val, y_val)

# MLR
mlr_results = predict_with_MLR(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Random Forest
rf_results = predict_with_random_forest(dates_train, dates_val, X_train, y_train, X_val, y_val)

# ExtraTrees
et_results = predict_with_extraTrees(dates_train, dates_val, X_train, y_train, X_val, y_val)

# kNN
knn_results = predict_with_kNN(dates_train, dates_val, X_train, y_train, X_val, y_val)

#LSTM
lstm_results = predict_with_LSTM(dates_train, dates_val, X_train, y_train, X_val, y_val, epochs_value=epoch)

# Ridge
ridge_results = predict_with_Ridge(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Lasso
lasso_results = predict_with_Lasso(dates_train, dates_val, X_train, y_train, X_val, y_val)

# Elastic
elastic_results = predict_with_Elastic(dates_train, dates_val, X_train, y_train, X_val, y_val)


# Performance of every model
get_performance(rf_results['Predicted Values'], rf_results['Real Values'], "Random Forest")
get_performance(et_results['Predicted Values'], et_results['Real Values'], "ExtraTrees")
get_performance(lstm_results['Predicted Values'], lstm_results['Real Values'], "LSTM")
get_performance(knn_results['Predicted Values'], knn_results['Real Values'], "k-NN")
get_performance(svr_results['Predicted Values'], svr_results['Real Values'], "SVR")
get_performance(mlr_results['Predicted Values'], mlr_results['Real Values'], "MLR")
get_performance(mlp_results['Predicted Values'], mlp_results['Real Values'], "MLP")
get_performance(lasso_results['Predicted Values'], lasso_results['Real Values'], "Lasso")
get_performance(ridge_results['Predicted Values'], ridge_results['Real Values'], "Ridge")
get_performance(elastic_results['Predicted Values'], elastic_results['Real Values'], "Elastic")
get_performance(rnn_results['Predicted Values'], rnn_results['Real Values'], "RNN")
get_performance(gru_results['Predicted Values'], gru_results['Real Values'], "GRU")
get_performance(cnn_results['Predicted Values'], gru_results['Real Values'], "CNN")


# Plotting results
rf_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Random Forest results')
plt.show()

et_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('ExtraTrees results')
plt.show()

lstm_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('LSTM results')
plt.show()

knn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('k-Nearest Neighbors results')
plt.show()

svr_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Support Vector Regression results')
plt.show()

mlr_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Multi-Linear Regression results')
plt.show()

mlp_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Multi-layer Perceptron results')
plt.show()

lasso_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Lasso Regression results')
plt.show()

ridge_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Ridge Regression results')
plt.show()

elastic_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Elastic Regression results')
plt.show()

rnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Recurrent Neural Network results')
plt.show()

gru_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Gated Recurrent Unit results')
plt.show()

cnn_results.plot(x='Timestamp', y=['Real Values', 'Predicted Values'])
plt.title('Convolutional Neural Network results')
plt.show()



