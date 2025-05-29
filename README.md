# Ex.No: 6 HOLT WINTERS METHOD
# Date:
# AIM:
# ALGORITHM:
1.You import the necessary libraries 

2.You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, and perform some initial data exploration

3.You group the data by date and resample it to a monthly frequency (beginning of the month

4.You plot the time series data

5.You import the necessary 'statsmodels' libraries for time series analysis

6.You decompose the time series data into its additive components and plot them:

7.You calculate the root mean squared error (RMSE) to evaluate the model's performance

8.You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt- Winters model to the entire dataset and make future predictions

9.You plot the original sales data and the predictions
# PROGRAM:
```
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('/content/daily-minimum-temperatures-in-me.csv', parse_dates=['Date'], index_col='Date', delimiter=',')
data['Daily minimum temperatures'] = pd.to_numeric(data['Daily minimum temperatures'], errors='coerce')

print(data.head())

temperature = data['Daily minimum temperatures']
temperature_monthly = temperature.resample('MS').mean()

scaler = MinMaxScaler()
temperature_scaled = pd.Series(
    scaler.fit_transform(temperature_monthly.values.reshape(-1, 1)).flatten(),
    index=temperature_monthly.index
)



train_size = int(len(temperature_scaled) * 0.8)
train_data = temperature_scaled[:train_size]
test_data = temperature_scaled[train_size:]

if len(train_data) < 24:
    print("Warning: Not enough data for seasonal modeling. Using non-seasonal model.")
    model = ExponentialSmoothing(train_data, trend='add').fit()
else:
    model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast test set
predictions = model.forecast(len(test_data))

print("MAE :", mean_absolute_error(test_data, predictions))
rmse = mean_squared_error(test_data, predictions)**0.5  
print("RMSE:", rmse)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='Train', color='black')
plt.plot(test_data, label='Test', color='green')
plt.plot(predictions, label='Forecast', color='red')
plt.title('Temperature Forecast (Train/Test Split)')
plt.xlabel('Date')
plt.ylabel('Scaled Temperature')
plt.legend()
plt.grid(True)
plt.show()

temperature_scaled = temperature_scaled + 1e-6  # Add a small positive constant

if len(temperature_scaled) < 24:
    final_model = ExponentialSmoothing(temperature_scaled, trend='mul').fit()
else:
    final_model = ExponentialSmoothing(temperature_scaled, trend='mul', seasonal='mul', seasonal_periods=12).fit()


future_forecast = final_model.forecast(12)

plt.figure(figsize=(12, 8))
plt.plot(temperature_scaled.index, temperature_scaled, label='Observed')
plt.plot(future_forecast.index, future_forecast, label='Forecast (Next 12 Months)', linestyle='--')
plt.title('Forecast of Monthly Minimum Temperatures')
plt.xlabel('Date')
plt.ylabel('Scaled Temperature')
plt.legend()
plt.grid(True)
plt.show()
```
# OUTPUT:
TEST_PREDICTION
![image](https://github.com/user-attachments/assets/15778a88-5395-458d-b439-8c8567beb1c1)

FINAL_PREDICTION
![image](https://github.com/user-attachments/assets/f22d508e-4050-432e-81fe-57f657361563)

# RESULT:
Thus the program run successfully based on the Holt Winters Method model.
