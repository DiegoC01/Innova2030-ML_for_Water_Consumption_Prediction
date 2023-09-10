import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller

def iqr(df, minimum_value_allowed = 0):
    # IQR
    Q1 = np.percentile(df['waterMeasured'][df['waterMeasured']>=minimum_value_allowed], 25)
    Q3 = np.percentile(df['waterMeasured'][df['waterMeasured']>=minimum_value_allowed], 75)
    IQR = Q3 - Q1
    print(IQR)

    # Above Upper bound
    upper=Q3+1.5*IQR
    upper_array=np.array(df['waterMeasured']>=upper)
    
    #Below Lower bound
    lower=Q1-1.5*IQR
    lower_array=np.array(df['waterMeasured']<=lower)

    return (upper, lower)

def stationarity_test(df):
    # Stationarity test
    print("Observations of Dickey-fuller test")
    dftest = adfuller(df["waterMeasured"],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['critical value (%s)'%key]= value
    print(dfoutput)

# Open data
raw_historical_data = ("1-Data_Preprocessing/Working_with_historical_data/3-cleaning_data/1-raw_data/raw_data-historical_data.csv")
df_historical_data = pd.read_csv(raw_historical_data)

# Defining the timestamp (after its convertion) of the measurement as the new index.
df_historical_data['time'] = pd.to_numeric(df_historical_data['time'])
df_historical_data['time_format'] = (pd.to_datetime(df_historical_data['time'],unit='s', dayfirst=True) .dt.tz_localize('utc').dt.tz_convert('America/Santiago'))
df_historical_data.index = df_historical_data['time']

# Plotting data
df_historical_data.plot(x='time_format', y=["waterMeasured"])
plt.show()

# Defining important parameters
# It was defined that if a value is less than 2, its considered noise
minimum_waterMeasured_allowed = 2.0

# Cleaning values < 2 (those are considered noise)
df_historical_data['waterMeasured'] = np.where(df_historical_data['waterMeasured'] < minimum_waterMeasured_allowed, 0, df_historical_data['waterMeasured'])
df_historical_data.plot(x='time_format', y=["waterMeasured"])
plt.show()

# Cleaning outliers (converting them to None and then filling them with ffill(), or converting them to 0)
(upper_bound_september, lower_bound_september) = iqr(df_historical_data, minimum_waterMeasured_allowed)
df_historical_data['waterMeasured'] = np.where(df_historical_data['waterMeasured'] < lower_bound_september, 0, df_historical_data['waterMeasured'])
df_historical_data['waterMeasured'] = np.where(df_historical_data['waterMeasured'] > upper_bound_september, None, df_historical_data['waterMeasured'])
df_historical_data = df_historical_data.ffill()

print(df_historical_data)
df_historical_data = df_historical_data.drop(columns=['time', 'time_format'])

print(df_historical_data)
df_historical_data.to_csv("1-Data_Preprocessing/Working_with_historical_data/3-cleaning_data/2-preprocessed_data/preprocessed_data-historical_data.csv")
