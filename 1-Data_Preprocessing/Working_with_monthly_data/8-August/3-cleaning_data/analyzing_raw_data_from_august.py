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
raw_data_august = ("1-Data_Preprocessing/Working_with_monthly_data/8-August/3-cleaning_data/1-raw_data/raw_data-august_from_02_to_31.csv")
df_august = pd.read_csv(raw_data_august)

# Defining the timestamp (after its convertion) of the measurement as the new index.
df_august['time_format'] = (pd.to_datetime(df_august['time'],unit='s', dayfirst=True) .dt.tz_localize('utc').dt.tz_convert('America/Santiago'))
df_august.index = df_august['time']

# Plotting data
df_august.plot(x='time_format', y=["waterMeasured"])
plt.show()

# Defining important parameters
# It was defined that if a value is less than 2, its considered noise
minimum_waterMeasured_allowed = 2.0
volts_august_interval_1 = 9
volts_august_interval_2 = 12

# Cleaning values < 2 (those are considered noise)
df_august['waterMeasured'] = np.where(df_august['waterMeasured'] < minimum_waterMeasured_allowed, 0, df_august['waterMeasured'])
df_august.plot(x='time_format', y=["waterMeasured"])
plt.show()

# Extracting intervals
df_august_interval_1 = df_august.loc[(df_august['time_format'] >= '2023-08-02') & (df_august['time_format'] < '2023-08-21')]
df_august_interval_2 = df_august.loc[(df_august['time_format'] >= '2023-08-21') & (df_august['time_format'] < '2023-08-29')]

# Extracting information from the intervals
# Mean of the values >= 2 for waterMeasured
mean_waterMeasured_august_interval_1 = df_august_interval_1['waterMeasured'].loc[df_august_interval_1['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
mean_waterMeasured_august_interval_2 = df_august_interval_2['waterMeasured'].loc[df_august_interval_2['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
# Calulating IQR to detect outliers
(upper_bound_august_interval_1, lower_bound_august_interval_1) = iqr(df_august_interval_1, minimum_waterMeasured_allowed)
(upper_bound_august_interval_2, lower_bound_august_interval_2) = iqr(df_august_interval_2, minimum_waterMeasured_allowed)

# Printing info
print("Mean interval 1: "+str(mean_waterMeasured_august_interval_1))
print("Upper bound interval 1: "+str(upper_bound_august_interval_1))
print("Lower bound interval 1: "+str(lower_bound_august_interval_1))

print("Mean interval 2: "+str(mean_waterMeasured_august_interval_2))
print("Upper bound interval 2: "+str(upper_bound_august_interval_2))
print("Lower bound interval 2: "+str(lower_bound_august_interval_2))


# Plotting interval 1
df_august_interval_1.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=np.mean(mean_waterMeasured_august_interval_1), color='r', linestyle='-')
plt.axhline(y=np.mean(upper_bound_august_interval_1), color='g', linestyle='-')
plt.axhline(y=np.mean(lower_bound_august_interval_1), color='b', linestyle='-')
plt.legend(["Mean of values >= 2", "Upper bound", "Lower bound"])
plt.show()

# Plotting interval 2
df_august_interval_2.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=np.mean(mean_waterMeasured_august_interval_2), color='r', linestyle='-')
plt.axhline(y=np.mean(upper_bound_august_interval_2), color='g', linestyle='-')
plt.axhline(y=np.mean(lower_bound_august_interval_2), color='b', linestyle='-')
plt.legend(["Mean of values >= 2", "Upper bound", "Lower bound"])
plt.show()

# Escaling the data according to the changes in the measure of water given the change to a 12V battery.
proportion_between_i1_i2 = mean_waterMeasured_august_interval_2/mean_waterMeasured_august_interval_1
print("Values of interval 2 are "+str((proportion_between_i1_i2))+" percent bigger than the values of interval 1.")
df_august_interval_1.loc[:, 'waterMeasured'] *= float(proportion_between_i1_i2)

# Extracting information from the intervals... again
# Mean of the values >= 2 for waterMeasured... again
mean_waterMeasured_august_interval_1 = df_august_interval_1['waterMeasured'].loc[df_august_interval_1['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
mean_waterMeasured_august_interval_2 = df_august_interval_2['waterMeasured'].loc[df_august_interval_2['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
# Calulating IQR to detect outliers... again
(upper_bound_august_interval_1, lower_bound_august_interval_1) = iqr(df_august_interval_1, minimum_waterMeasured_allowed)
(upper_bound_august_interval_2, lower_bound_august_interval_2) = iqr(df_august_interval_2, minimum_waterMeasured_allowed)

# Printing info... again
print("Mean interval 1: "+str(mean_waterMeasured_august_interval_1))
print("Upper bound interval 1: "+str(upper_bound_august_interval_1))
print("Lower bound interval 1: "+str(lower_bound_august_interval_1))

print("Mean interval 2: "+str(mean_waterMeasured_august_interval_2))
print("Upper bound interval 2: "+str(upper_bound_august_interval_2))
print("Lower bound interval 2: "+str(lower_bound_august_interval_2))


# Plotting interval 1... again
df_august_interval_1.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=np.mean(mean_waterMeasured_august_interval_1), color='r', linestyle='-')
plt.axhline(y=np.mean(upper_bound_august_interval_1), color='g', linestyle='-')
plt.axhline(y=np.mean(lower_bound_august_interval_1), color='b', linestyle='-')
plt.legend(["Mean of values >= 2", "Upper bound", "Lower bound"])
plt.show()

# Plotting interval 2... again
df_august_interval_2.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=np.mean(mean_waterMeasured_august_interval_2), color='r', linestyle='-')
plt.axhline(y=np.mean(upper_bound_august_interval_2), color='g', linestyle='-')
plt.axhline(y=np.mean(lower_bound_august_interval_2), color='b', linestyle='-')
plt.legend(["Mean of values >= 2", "Upper bound", "Lower bound"])
plt.show()


# Inserting modified intervals
df_august.loc[(df_august['time_format'] >= '2023-08-02') & (df_august['time_format'] < '2023-08-21')] = df_august_interval_1
df_august.loc[(df_august['time_format'] >= '2023-08-21') & (df_august['time_format'] < '2023-08-29')] = df_august_interval_2

# Plotting dataset modified
df_august.plot(x='time_format', y=["waterMeasured"])
plt.show()

# Extracting information of the entire modified dataset
# Mean of the values >= 2 for waterMeasured of the entire modified dataset
mean_waterMeasured_august = df_august['waterMeasured'].loc[df_august['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
# Calulating IQR to detect outliers on the entire modified dataset
(upper_bound_august, lower_bound_august) = iqr(df_august, minimum_waterMeasured_allowed)

# Printing info of the entire modified dataset
print("Mean: "+str(mean_waterMeasured_august))
print("Upper bound: "+str(upper_bound_august))
print("Lower bound: "+str(lower_bound_august))

# Plotting the entire modified dataset with its information
df_august.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=np.mean(mean_waterMeasured_august), color='r', linestyle='-')
plt.axhline(y=np.mean(upper_bound_august), color='g', linestyle='-')
plt.axhline(y=np.mean(lower_bound_august), color='b', linestyle='-')
plt.legend(["Mean of values >= 2", "Upper bound", "Lower bound"])
plt.show()

# Cleaning dataset using the upper and lower bound, replacing outliers with None
df_august['waterMeasured'] = np.where(df_august['waterMeasured'] < lower_bound_august, 0, df_august['waterMeasured'])
df_august['waterMeasured'] = np.where(df_august['waterMeasured'] > upper_bound_august, None, df_august['waterMeasured'])

# Replacing None values with the preceding value of each row of each column.
df_august = df_august.ffill()

# Plotting data
df_august.plot(x='time_format', y=["waterMeasured"])
plt.title("Dataset with None Values Replaced")
plt.show()

#Plot waterMeasured over time: density diagram
df_august_density = df_august[(df_august['waterMeasured'] >= minimum_waterMeasured_allowed)]
df_august_density.plot(x='time_format', y=["waterMeasured"], kind="kde")
plt.show()

print(df_august)
df_august = df_august.drop(columns=['time', 'time_format'])

print(df_august)
df_august.to_csv("1-Data_Preprocessing/Working_with_monthly_data/8-August/3-cleaning_data/2-preprocessed_data/preprocessed_data-august_from_02_to_31.csv")




