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
raw_data_september = ("1-Data_Preprocessing/Working_with_monthly_data/9-September/3-cleaning_data/1-raw_data/raw_data-september_from_01_to_01.csv")
df_september = pd.read_csv(raw_data_september)

# Defining the timestamp (after its convertion) of the measurement as the new index.
df_september['time'] = pd.to_numeric(df_september['time'])
df_september['time_format'] = (pd.to_datetime(df_september['time'],unit='s', dayfirst=True) .dt.tz_localize('utc').dt.tz_convert('America/Santiago'))
df_september.index = df_september['time']

# Plotting data
df_september.plot(x='time_format', y=["waterMeasured"])
plt.show()

# Defining important parameters
# It was defined that if a value is less than 2, its considered noise
minimum_waterMeasured_allowed = 2.0
volts_august_interval_1 = 9
volts_august_interval_2 = 12

# Cleaning values < 2 (those are considered noise)
df_september['waterMeasured'] = np.where(df_september['waterMeasured'] < minimum_waterMeasured_allowed, 0, df_september['waterMeasured'])
df_september.plot(x='time_format', y=["waterMeasured"])
plt.show()

# Cleaning outliers (converting them to None and then filling them with ffill(), or converting them to 0)
(upper_bound_september, lower_bound_september) = iqr(df_september, minimum_waterMeasured_allowed)
df_september['waterMeasured'] = np.where(df_september['waterMeasured'] < lower_bound_september, 0, df_september['waterMeasured'])
df_september['waterMeasured'] = np.where(df_september['waterMeasured'] > upper_bound_september, None, df_september['waterMeasured'])
df_september = df_september.ffill()



print(df_september)
df_september = df_september.drop(columns=['time', 'time_format'])

print(df_september)
df_september.to_csv("1-Data_Preprocessing/Working_with_monthly_data/9-September/3-cleaning_data/2-preprocessed_data/preprocessed_data-september_from_01_to_01.csv")




