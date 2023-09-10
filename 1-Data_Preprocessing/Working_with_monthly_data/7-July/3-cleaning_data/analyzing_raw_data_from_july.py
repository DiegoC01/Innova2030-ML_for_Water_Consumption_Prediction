import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller

def deleting_noisy_values(df, minimum_waterMeasured_allowed=2):
    # Deleting noise created by the electromagentic field field
    df['waterMeasured'] = np.where(df['waterMeasured'] < minimum_waterMeasured_allowed, 0, df['waterMeasured'])

    return df

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
    df['waterMeasured'] = np.where(df['waterMeasured'] > upper, 0, df['waterMeasured'])
    df['waterMeasured'] = np.where(df['waterMeasured'] < lower, 0, df['waterMeasured'])

    return df

def stationarity_test(df):
    # Stationarity test
    print("Observations of Dickey-fuller test")
    dftest = adfuller(df["waterMeasured"],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['critical value (%s)'%key]= value
    print(dfoutput)

def scaling_data_with_mean(df_interval, reference_mean, minimum_waterMeasured_allowed=2):
    """Scale the values of a dataframe using a mean value (considered as 'correct' for practical reasons) as reference.
    The mean is 
    
    Keyword arguments:
    df_interval -- Dataframe to be scaled.
    reference_mean -- Mean used as a reference when scaling.
    minimum_waterMeasured_allowed -- Values considered as not noisy or different to zero. Its default value is 2.
    Return: df_interval scaled.
    """
    df_interval = deleting_outliers(df_interval)

    # Calculating the proportion between the means
    mean_waterMeasured_interval = df_interval['waterMeasured'].loc[df_interval['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
    
    proportion_between_means = float(mean_waterMeasured_interval/reference_mean)

    # Scaling values of water measured with proportion calculated
    df_interval['waterMeasured'] = np.where(True, (df_interval['waterMeasured'])/proportion_between_means, df_interval['waterMeasured'])

    return df_interval



# Open data
raw_data_july = ("1-Data_Preprocessing/Working_with_monthly_data/7-July/3-cleaning_data/1-raw_data/raw_data-july_from_01_to_31.csv")
df_july = pd.read_csv(raw_data_july)

# Defining the timestamp (after its convertion) of the measurement as the new index.
df_july['time_format'] = (pd.to_datetime(df_july['time'],unit='s', dayfirst=True)
              .dt.tz_localize('utc')
              .dt.tz_convert('America/Santiago'))

df_july.index = df_july['time']

#df_july.drop(['time'], axis=1, inplace=True)

# Deleting noisy values
df_july = deleting_noisy_values(df_july)

# Plotting data
df_july.plot(y=["waterMeasured"])
plt.title("Raw data")
plt.show()

# Defining mean of the values measured in august
mean_waterMeasured_august =  4.557104
minimum_waterMeasured_allowed = 2

# Defining intervals to be analyzed
df_interval_1 = df_july.loc[(df_july['time_format'] >= '2023-07-09') & (df_july['time_format'] < '2023-07-14')]
df_interval_2 = df_july.loc[(df_july['time_format'] >= '2023-07-14') & (df_july['time_format'] < '2023-07-16')]
df_interval_3 = df_july.loc[(df_july['time_format'] >= '2023-07-16') & (df_july['time_format'] < '2023-07-18')]

# Plotting data
df_interval_1.plot(x='time_format', y=["waterMeasured"])
plt.title("Raw data i1")
plt.show()
df_interval_2.plot(x='time_format', y=["waterMeasured"])
plt.title("Raw data i2")
plt.show()
df_interval_3.plot(x='time_format', y=["waterMeasured"])
plt.title("Raw data i3")
plt.show()

# Substracting every value of the 3 intervals by its mean
mean_waterMeasured_interval_1 = df_interval_1['waterMeasured'].loc[df_interval_1['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
mean_waterMeasured_interval_2 = df_interval_2['waterMeasured'].loc[df_interval_2['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
mean_waterMeasured_interval_3 = df_interval_3['waterMeasured'].loc[df_interval_3['waterMeasured'] >= minimum_waterMeasured_allowed].mean()

df_interval_1['waterMeasured'] = np.where(True, (df_interval_1['waterMeasured'] - mean_waterMeasured_interval_1), df_interval_1['waterMeasured'])
df_interval_2['waterMeasured'] = np.where(True, (df_interval_2['waterMeasured'] - mean_waterMeasured_interval_2), df_interval_2['waterMeasured'])
df_interval_3['waterMeasured'] = np.where(True, (df_interval_3['waterMeasured'] -mean_waterMeasured_interval_3), df_interval_3['waterMeasured'])

# Deleting noisy values
df_interval_1 = deleting_noisy_values(df_interval_1)
df_interval_2 = deleting_noisy_values(df_interval_2)
df_interval_3 = deleting_noisy_values(df_interval_3)

# Plotting data
df_interval_1.plot(x='time_format', y=["waterMeasured"])
plt.title("Raw data i1 minus mean")
plt.show()
df_interval_2.plot(x='time_format', y=["waterMeasured"])
plt.title("Raw data i2 minus mean")
plt.show()
df_interval_3.plot(x='time_format', y=["waterMeasured"])
plt.title("Raw data i3 minus mean")
plt.show()

# Applying re-scaling to the intervals
df_interval_1 = scaling_data_with_mean(df_interval_1, mean_waterMeasured_august)
df_interval_2 = scaling_data_with_mean(df_interval_2, mean_waterMeasured_august)
df_interval_3 = scaling_data_with_mean(df_interval_3, mean_waterMeasured_august)

# Deleting noisy values
df_interval_1 = deleting_noisy_values(df_interval_1)
df_interval_2 = deleting_noisy_values(df_interval_2)
df_interval_3 = deleting_noisy_values(df_interval_3)

# Plotting data
df_interval_1.plot(x='time_format', y=["waterMeasured"])
plt.title("Scaled raw data i1")
plt.show()
df_interval_2.plot(x='time_format', y=["waterMeasured"])
plt.title("Scaled raw data i2")
plt.show()
df_interval_3.plot(x='time_format', y=["waterMeasured"])
plt.title("Scaled raw data i3")
plt.show()

# Reemplazing the scaled values of the intervals in the original dataframe
df_july.loc[(df_july['time_format'] >= '2023-07-09') & (df_july['time_format'] < '2023-07-14')] = df_interval_1
df_july.loc[(df_july['time_format'] >= '2023-07-14') & (df_july['time_format'] < '2023-07-16')] = df_interval_2
df_july.loc[(df_july['time_format'] >= '2023-07-16') & (df_july['time_format'] < '2023-07-18')] = df_interval_3

# Deleting noisy values
df_july = deleting_noisy_values(df_july)


# Calculating upper and lower bounds to detect outliers
(upper, lower) = iqr(df_july, minimum_waterMeasured_allowed)
print("Upper bound: "+str(upper))
print("Lower bound: "+str(lower))

# Plotting values of the dataframe
df_july.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=upper, color='r', linestyle='-')
plt.axhline(y=lower, color='g', linestyle='-')
plt.legend(["Upper bound", "Lower bound"])
plt.show()

# Defining new intervals to be analyzed
df_interval_4 = df_july.loc[(df_july['time_format'] >= '2023-07-25') & (df_july['time_format'] < '2023-07-26')]
df_interval_5 = df_july.loc[(df_july['time_format'] >= '2023-07-26') & (df_july['time_format'] < '2023-07-27')]
df_interval_6 = df_july.loc[(df_july['time_format'] >= '2023-07-27') & (df_july['time_format'] < '2023-07-28')]
df_interval_7 = df_july.loc[(df_july['time_format'] >= '2023-07-28') & (df_july['time_format'] < '2023-07-30')]
df_interval_8 = df_july.loc[(df_july['time_format'] >= '2023-07-30') & (df_july['time_format'] < '2023-08-01')]

# Applying re-scaling to the new intervals
df_interval_4 = scaling_data_with_mean(df_interval_4, mean_waterMeasured_august)
df_interval_5 = scaling_data_with_mean(df_interval_5, mean_waterMeasured_august)
df_interval_6 = scaling_data_with_mean(df_interval_6, mean_waterMeasured_august)
df_interval_7 = scaling_data_with_mean(df_interval_7, mean_waterMeasured_august)
df_interval_8 = scaling_data_with_mean(df_interval_8, mean_waterMeasured_august)

# Reemplazing the scaled values of the new intervals in the original dataframe
df_july.loc[(df_july['time_format'] >= '2023-07-25') & (df_july['time_format'] < '2023-07-26')] = df_interval_4
df_july.loc[(df_july['time_format'] >= '2023-07-26') & (df_july['time_format'] < '2023-07-27')] = df_interval_5
df_july.loc[(df_july['time_format'] >= '2023-07-27') & (df_july['time_format'] < '2023-07-28')] = df_interval_6
df_july.loc[(df_july['time_format'] >= '2023-07-28') & (df_july['time_format'] < '2023-07-30')] = df_interval_7
df_july.loc[(df_july['time_format'] >= '2023-07-30') & (df_july['time_format'] < '2023-08-01')] = df_interval_8

# Deleting noisy values
df_july = deleting_noisy_values(df_july)


# Calculating upper and lower bounds to detect outliers
(upper, lower) = iqr(df_july, minimum_waterMeasured_allowed)
print("Upper bound: "+str(upper))
print("Lower bound: "+str(lower))

# Plotting values of the dataframe
df_july.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=upper, color='r', linestyle='-')
plt.axhline(y=lower, color='g', linestyle='-')
plt.legend(["Upper bound", "Lower bound"])
plt.show()

# Defining new intervals to be analyzed: 3rd time
df_interval_9 = df_july.loc[(df_july['time_format'] >= '2023-07-18') & (df_july['time_format'] < '2023-07-19')]
df_interval_10 = df_july.loc[(df_july['time_format'] >= '2023-07-19') & (df_july['time_format'] < '2023-07-20')]
df_interval_11 = df_july.loc[(df_july['time_format'] >= '2023-07-20') & (df_july['time_format'] < '2023-07-21')]
df_interval_12 = df_july.loc[(df_july['time_format'] >= '2023-07-21') & (df_july['time_format'] < '2023-07-22')]
df_interval_13 = df_july.loc[(df_july['time_format'] >= '2023-07-22') & (df_july['time_format'] < '2023-08-23')]
df_interval_14 = df_july.loc[(df_july['time_format'] >= '2023-07-23') & (df_july['time_format'] < '2023-08-25')]

# Applying re-scaling to the new intervals
df_interval_9 = scaling_data_with_mean(df_interval_9, mean_waterMeasured_august)
df_interval_10 = scaling_data_with_mean(df_interval_10, mean_waterMeasured_august)
df_interval_11 = scaling_data_with_mean(df_interval_11, mean_waterMeasured_august)
df_interval_12 = scaling_data_with_mean(df_interval_12, mean_waterMeasured_august)
df_interval_13 = scaling_data_with_mean(df_interval_13, mean_waterMeasured_august)
df_interval_14 = scaling_data_with_mean(df_interval_14, mean_waterMeasured_august)

# Reemplazing the scaled values of the new intervals in the original dataframe
df_july.loc[(df_july['time_format'] >= '2023-07-18') & (df_july['time_format'] < '2023-07-19')] = df_interval_9
df_july.loc[(df_july['time_format'] >= '2023-07-19') & (df_july['time_format'] < '2023-07-20')] = df_interval_10
df_july.loc[(df_july['time_format'] >= '2023-07-20') & (df_july['time_format'] < '2023-07-21')] = df_interval_11
df_july.loc[(df_july['time_format'] >= '2023-07-21') & (df_july['time_format'] < '2023-07-22')] = df_interval_12
df_july.loc[(df_july['time_format'] >= '2023-07-22') & (df_july['time_format'] < '2023-08-23')] = df_interval_13
df_july.loc[(df_july['time_format'] >= '2023-07-23') & (df_july['time_format'] < '2023-08-25')] = df_interval_14

# Deleting noisy values
df_july = deleting_noisy_values(df_july)

# Calculating upper and lower bounds to detect outliers
(upper, lower) = iqr(df_july, minimum_waterMeasured_allowed)
print("Upper bound: "+str(upper))
print("Lower bound: "+str(lower))


# Plotting values of the dataframe
df_july.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=upper, color='r', linestyle='-')
plt.axhline(y=lower, color='g', linestyle='-')
plt.legend(["Upper bound", "Lower bound"])
plt.show()

# Defining new intervals to be analyzed: 4rd time
df_interval_15 = df_july.loc[(df_july['time_format'] >= '2023-07-01') & (df_july['time_format'] < '2023-07-03')]
df_interval_16 = df_july.loc[(df_july['time_format'] >= '2023-07-03') & (df_july['time_format'] < '2023-07-09')]

# Applying re-scaling to the new intervals
df_interval_15 = scaling_data_with_mean(df_interval_15, mean_waterMeasured_august)
df_interval_16 = scaling_data_with_mean(df_interval_16, mean_waterMeasured_august)

# Reemplazing the scaled values of the new intervals in the original dataframe
df_july.loc[(df_july['time_format'] >= '2023-07-01') & (df_july['time_format'] < '2023-07-03')] = df_interval_15
df_july.loc[(df_july['time_format'] >= '2023-07-03') & (df_july['time_format'] < '2023-07-09')] = df_interval_16

# Deleting noisy values
df_july = deleting_noisy_values(df_july)

# Calculating upper and lower bounds to detect outliers
(upper, lower) = iqr(df_july, minimum_waterMeasured_allowed)
print("Upper bound: "+str(upper))
print("Lower bound: "+str(lower))

df_july['day'] = df_july['time_format'].dt.day
print("Mean per day (July)")
print(df_july[(df_july['waterMeasured'] >= minimum_waterMeasured_allowed)].groupby('day').mean())
df_july.drop(['day'], axis=1, inplace=True)



# Plotting values of the dataframe
df_july.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=upper, color='r', linestyle='-')
plt.axhline(y=lower, color='g', linestyle='-')
plt.legend(["Upper bound", "Lower bound"])
plt.title("Water Measured Time Series")
plt.show()


# Given that some days present realy high values (+-)1, those days will be again scaled.
df_interval_17 = df_july.loc[(df_july['time_format'] >= '2023-07-05') & (df_july['time_format'] < '2023-07-06')]
df_interval_18 = df_july.loc[(df_july['time_format'] >= '2023-07-06') & (df_july['time_format'] < '2023-07-07')]
df_interval_19 = df_july.loc[(df_july['time_format'] >= '2023-07-07') & (df_july['time_format'] < '2023-07-08')]
df_interval_20 = df_july.loc[(df_july['time_format'] >= '2023-07-19') & (df_july['time_format'] < '2023-07-20')]
df_interval_21 = df_july.loc[(df_july['time_format'] >= '2023-07-23') & (df_july['time_format'] < '2023-07-24')]
df_interval_22 = df_july.loc[(df_july['time_format'] >= '2023-07-25') & (df_july['time_format'] < '2023-07-26')]
df_interval_23 = df_july.loc[(df_july['time_format'] >= '2023-07-26') & (df_july['time_format'] < '2023-07-27')]
df_interval_24 = df_july.loc[(df_july['time_format'] >= '2023-07-28') & (df_july['time_format'] < '2023-07-29')]
df_interval_25 = df_july.loc[(df_july['time_format'] >= '2023-07-29') & (df_july['time_format'] < '2023-07-30')]
df_interval_26 = df_july.loc[(df_july['time_format'] >= '2023-07-30') & (df_july['time_format'] < '2023-07-31')]

# More aditions
df_interval_27 = df_july.loc[(df_july['time_format'] >= '2023-07-11') & (df_july['time_format'] < '2023-07-12')]
df_interval_28 = df_july.loc[(df_july['time_format'] >= '2023-07-12') & (df_july['time_format'] < '2023-07-13')]
df_interval_29 = df_july.loc[(df_july['time_format'] >= '2023-07-14') & (df_july['time_format'] < '2023-07-15')]
df_interval_30 = df_july.loc[(df_july['time_format'] >= '2023-07-15') & (df_july['time_format'] < '2023-07-16')]
df_interval_31 = df_july.loc[(df_july['time_format'] >= '2023-07-16') & (df_july['time_format'] < '2023-07-17')]
df_interval_32 = df_july.loc[(df_july['time_format'] >= '2023-07-17') & (df_july['time_format'] < '2023-07-18')]
df_interval_33 = df_july.loc[(df_july['time_format'] >= '2023-07-19') & (df_july['time_format'] < '2023-07-20')]


df_interval_17 = scaling_data_with_mean(df_interval_17, mean_waterMeasured_august)
df_interval_18 = scaling_data_with_mean(df_interval_18, mean_waterMeasured_august)
df_interval_19 = scaling_data_with_mean(df_interval_19, mean_waterMeasured_august)
df_interval_20 = scaling_data_with_mean(df_interval_20, mean_waterMeasured_august)
df_interval_21 = scaling_data_with_mean(df_interval_21, mean_waterMeasured_august)
df_interval_22 = scaling_data_with_mean(df_interval_22, mean_waterMeasured_august)
df_interval_23 = scaling_data_with_mean(df_interval_23, mean_waterMeasured_august)
df_interval_24 = scaling_data_with_mean(df_interval_24, mean_waterMeasured_august)
df_interval_25 = scaling_data_with_mean(df_interval_25, mean_waterMeasured_august)
df_interval_26 = scaling_data_with_mean(df_interval_26, mean_waterMeasured_august)
df_interval_27 = scaling_data_with_mean(df_interval_27, mean_waterMeasured_august)
df_interval_28 = scaling_data_with_mean(df_interval_28, mean_waterMeasured_august)
df_interval_29 = scaling_data_with_mean(df_interval_29, mean_waterMeasured_august)
df_interval_30 = scaling_data_with_mean(df_interval_30, mean_waterMeasured_august)
df_interval_31 = scaling_data_with_mean(df_interval_31, mean_waterMeasured_august)
df_interval_32 = scaling_data_with_mean(df_interval_32, mean_waterMeasured_august)
df_interval_33 = scaling_data_with_mean(df_interval_33, mean_waterMeasured_august)

df_july.loc[(df_july['time_format'] >= '2023-07-05') & (df_july['time_format'] < '2023-07-06')] = df_interval_17
df_july.loc[(df_july['time_format'] >= '2023-07-06') & (df_july['time_format'] < '2023-07-07')] = df_interval_18
df_july.loc[(df_july['time_format'] >= '2023-07-07') & (df_july['time_format'] < '2023-07-08')] = df_interval_19
df_july.loc[(df_july['time_format'] >= '2023-07-19') & (df_july['time_format'] < '2023-07-20')] = df_interval_20
df_july.loc[(df_july['time_format'] >= '2023-07-23') & (df_july['time_format'] < '2023-07-24')] = df_interval_21
df_july.loc[(df_july['time_format'] >= '2023-07-25') & (df_july['time_format'] < '2023-07-26')] = df_interval_22
df_july.loc[(df_july['time_format'] >= '2023-07-26') & (df_july['time_format'] < '2023-07-27')] = df_interval_23
df_july.loc[(df_july['time_format'] >= '2023-07-28') & (df_july['time_format'] < '2023-07-29')] = df_interval_24
df_july.loc[(df_july['time_format'] >= '2023-07-29') & (df_july['time_format'] < '2023-07-30')] = df_interval_25
df_july.loc[(df_july['time_format'] >= '2023-07-30') & (df_july['time_format'] < '2023-07-31')] = df_interval_26
df_july.loc[(df_july['time_format'] >= '2023-07-11') & (df_july['time_format'] < '2023-07-12')] = df_interval_27 
df_july.loc[(df_july['time_format'] >= '2023-07-12') & (df_july['time_format'] < '2023-07-13')] = df_interval_28 
df_july.loc[(df_july['time_format'] >= '2023-07-14') & (df_july['time_format'] < '2023-07-15')] = df_interval_29 
df_july.loc[(df_july['time_format'] >= '2023-07-15') & (df_july['time_format'] < '2023-07-16')] = df_interval_30 
df_july.loc[(df_july['time_format'] >= '2023-07-16') & (df_july['time_format'] < '2023-07-17')] = df_interval_31 
df_july.loc[(df_july['time_format'] >= '2023-07-17') & (df_july['time_format'] < '2023-07-18')] = df_interval_32 
df_july.loc[(df_july['time_format'] >= '2023-07-19') & (df_july['time_format'] < '2023-07-20')] = df_interval_33 

# Deleting noisy values
df_july = deleting_noisy_values(df_july)
df_july = deleting_outliers(df_july)

df_july['day'] = df_july['time_format'].dt.day
print("Mean per day (July)")
print(df_july[(df_july['waterMeasured'] >= minimum_waterMeasured_allowed)].groupby('day').mean())
df_july.drop(['day'], axis=1, inplace=True)

# Plotting values of the dataframe
df_july.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=mean_waterMeasured_august, color='r', linestyle='-')
plt.legend(["Mean of reference"])
plt.title("Water Measured Time Series Final")
plt.show()

#Plot waterMeasured over time
df_july.plot(y=["waterMeasured"], style="k.")
plt.show()

#Plot waterMeasured over time: density diagram
df_july_density = df_july[(df_july['waterMeasured'] >= minimum_waterMeasured_allowed)]
df_july_density.plot(y=["waterMeasured"], kind="kde")
plt.show()

# Deleting unnecessary data or unusual values (according to data presented in August)
df_july['waterMeasured'] = np.where(df_july['waterMeasured'] > 5.3, 0, df_july['waterMeasured'])
df_july['waterMeasured'] = np.where(df_july['waterMeasured'] < 3.7, 0, df_july['waterMeasured'])

df_july = deleting_outliers(df_july)


df_july['day'] = df_july['time_format'].dt.day
print("Mean per day (July)")
print(df_july[(df_july['waterMeasured'] >= minimum_waterMeasured_allowed)].groupby('day').mean())
df_july.drop(['day'], axis=1, inplace=True)

#Plot waterMeasured over time: density diagram
df_july_density = df_july[(df_july['waterMeasured'] >= minimum_waterMeasured_allowed)]
df_july_density.plot(y=["waterMeasured"], kind="kde")
plt.show()

#Plot waterMeasured over time
df_july.plot(y=["waterMeasured"], style="k.")
plt.show()

# Plotting values of the dataframe
df_july.plot(x='time_format', y=['waterMeasured'])
plt.axhline(y=mean_waterMeasured_august, color='r', linestyle='-')
plt.legend(["Mean of reference"])
plt.title("Water Measured Time Series Final (Hopefully)")
plt.show()

df_july = df_july.drop(columns=['time', 'time_format'])

(upper, lower) = iqr(df_july)
print("IQR analysis")
print("Upper bound: "+str(upper))
print("Lower bound"+str(lower))
print()
mean_general = df_july['waterMeasured'].loc[df_july['waterMeasured'] >= minimum_waterMeasured_allowed].mean()
print("Mean: "+str(float(mean_general)))

df_july.to_csv("1-Data_Preprocessing/Working_with_monthly_data/7-July/3-cleaning_data/2-preprocessed_data/preprocessed_data-july_from_01_to_31.csv")


