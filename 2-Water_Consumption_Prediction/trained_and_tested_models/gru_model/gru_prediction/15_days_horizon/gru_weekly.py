import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

import keras
print("Versión de TensorFlow:", tf.__version__)
print("Versión de Keras:", keras.__version__)

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
df_historical_data_granularity['hour'] = df_historical_data_granularity.index.floor('10080Min')
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
PREVIOUS_DATA = 2
FUTURE_DATA = 2

# Creating data sets for analysis of ML techniques

dates, X, y = supervised_problem(df_historical_data, previous_examples=PREVIOUS_DATA, future_predictions=FUTURE_DATA)

# Dividing data
q_64 = int(len(dates) * .64)
q_80 = int(len(dates) * .8)

dates_train, X_train, y_train = dates[:q_64], X[:q_64], y[:q_64]
dates_val, X_val, y_val = dates[q_64:q_80], X[q_64:q_80], y[q_64:q_80]
dates_test, X_test, y_test = dates[q_80:], X[q_80:], y[q_80:]



#################################################
# PREDICTIONS
#################################################

loaded_model = tf.keras.saving.load_model("gru_model/gru_training/15_days_horizon/models/gru_weekly_15_days_horizon.h5")
prediction = loaded_model.predict(X_test)

print("VALORES POR SEMANA")
print("Predicción: "+str(np.sum(prediction[-1])))
print("Reales: "+str(np.sum(y_test[-1])))

plt.plot(dates_test[-1], y_test[-1], color='red', marker='o', label='Valores Reales')
plt.plot(dates_test[-1], prediction[-1], color='blue', marker='o', label='Predicciones')

# Configurar el formato de las fechas en el eje x
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

# Rotar las fechas para que sean legibles
plt.gcf().autofmt_xdate()

plt.title('Predicciones vs Valores Reales')
plt.xlabel('Fechas')
plt.ylabel('Valores')
plt.legend()
plt.tight_layout()
plt.show()

get_performance(predicted_values=prediction, real_values=y_test, model_name="GRU: 15-Days Horizon with weekly data")