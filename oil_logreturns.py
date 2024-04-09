# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:04:28 2024

@author: he_98
"""

import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# Setting the font sizes
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 18  # Axis label font size
plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick label size
plt.rcParams['legend.fontsize'] = 16  # Legend font size


# Setting the start and end dates for the data
start_date = datetime(2005, 1, 1)
end_date = datetime(2020, 12, 30)

# Ticker for WTI Crude Oil futures
ticker_symbol = 'CL=F'
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Using 'Close' prices for analysis
oil_prices = data['Close'].dropna()
oil_log_returns = np.log(oil_prices / oil_prices.shift(1)).dropna()
weekly_log_returns = oil_log_returns.resample('W').sum()
oil_monthly_log_returns = oil_log_returns.resample('M').sum()

# Checking stationarity of oil prices
adf_test = adfuller(oil_prices.dropna())
print('ADF Statistic for oil prices: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])

# Check stationarity of log returns
adf_test = adfuller(oil_log_returns.dropna())
print('ADF Statistic for returns: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])

# Plotting the original oil prices
plt.figure(figsize=(12, 6))
plt.plot(oil_prices.index, oil_prices, label='Oil Prices', color='blue')
plt.title('Oil Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plotting the daily log returns
plt.figure(figsize=(12, 6))
plt.plot(oil_log_returns.index, oil_log_returns, label='Daily Log Returns', color='blue')
plt.title('Daily Log Returns of Oil Prices')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.show()

# Plotting the monthly log returns
plt.figure(figsize=(12, 6))
plt.plot(oil_monthly_log_returns.index, oil_monthly_log_returns, label='Monthly Log Returns', color='blue')
plt.title('Monthly Log Returns of Oil Prices')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.show()

# DECOMPOSITION
periodicity = 12  # for annual seasonality in monthly data
decomposition = seasonal_decompose(oil_monthly_log_returns, model='additive', period=periodicity)

trend = decomposition.trend.dropna()
seasonal = decomposition.seasonal.dropna()
resid = decomposition.resid.dropna()

trend = decomposition.trend.fillna(0)
resid = decomposition.resid.fillna(0)

# Plotting the decomposed components with titles
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
decomposition.observed.plot(ax=ax1, title='Observed')
decomposition.trend.plot(ax=ax2, title='Trend')
decomposition.seasonal.plot(ax=ax3, title='Seasonal')
decomposition.resid.plot(ax=ax4, title='Residual')

fig.suptitle('Monthly Log Returns of Oil - Decomposition', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

train_size = int(len(oil_monthly_log_returns) * 0.7)

# TREND COMPONENT

# Best ARIMA model for the trend component
auto_model_trend = auto_arima(trend.dropna(), start_p=0, start_q=0, max_p=5, max_q=5,
                              seasonal=False, stepwise=True, suppress_warnings=True, 
                              error_action='ignore', trace=True)

print("Trend Component:")
print(auto_model_trend.summary())

# Auto ARIMA for Trend Component
trend_model = auto_model_trend

# Forecasting the trend component using the best ARIMA model
trend_forecast = trend_model.predict(n_periods=len(trend) - train_size)

# Linear Regression for Trend Component
lin_reg = LinearRegression()
lin_reg.fit(np.arange(train_size).reshape(-1, 1), trend[:train_size])
trend_lin_forecast = lin_reg.predict(np.arange(len(trend)).reshape(-1, 1))

test_trend = trend[train_size:]

# ARIMA forecast for the test set
arima_trend_forecast_test = trend_forecast[-len(test_trend):]

# Linear Regression forecast for the test set
lin_reg_trend_forecast_test = trend_lin_forecast[train_size:]

# Calculating MSE for ARIMA and Linear Regression
mse_arima = mean_squared_error(test_trend, arima_trend_forecast_test)
mse_lin_reg = mean_squared_error(test_trend, lin_reg_trend_forecast_test)

# Printing the MSE values
print(f'ARIMA MSE: {mse_arima}')
print(f'Linear Regression MSE: {mse_lin_reg}')

# Fitting the Exponential Smoothing model to the trend component
ets_model = ExponentialSmoothing(trend[:train_size], 
                                 trend='additive',
                                 seasonal=None, 
                                 damped_trend=False).fit()
ets_forecast = ets_model.predict(start=train_size, end=len(trend) - 1)

# Calculating MSE for ETS
mse_ets = mean_squared_error(test_trend, ets_forecast)
print(f'ETS MSE: {mse_ets}')

# Preparing the feature matrix X and target vector y
X = np.arange(len(trend)).reshape(-1, 1)
y = trend.values

# Splitting the data into train and test sets
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Fitting the Gradient Boosting Regressor
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=42)
gbm_model.fit(X_train, y_train)

# Forecasting with the trained GBM model
gbm_forecast = gbm_model.predict(X_test)

# Calculating the MSE for the GBM forecast
mse_gbm = mean_squared_error(y_test, gbm_forecast)
print(f'Gradient Boosting Model MSE: {mse_gbm}')

# SEASONAL COMPONENT

# Seasonal Naive Forecast for Seasonal Component
seasonal_naive_forecast = np.tile(seasonal[-12:], int(np.ceil((len(seasonal) - train_size) / 12)))

# Holt-Winters for Seasonal Component
hw_model = ExponentialSmoothing(seasonal[:train_size], trend='add', seasonal='add', seasonal_periods=12).fit()
seasonal_hw_forecast = hw_model.forecast(len(seasonal) - train_size)

test_seasonal = seasonal[train_size:]

# Seasonal Naive forecast for the test set
seasonal_naive_forecast_test = seasonal_naive_forecast[-len(test_seasonal):]

# Holt-Winters forecast for the test set
seasonal_hw_forecast_test = seasonal_hw_forecast[-len(test_seasonal):]

# Calculating MSE for Seasonal Naive and Holt-Winters
mse_seasonal_naive = mean_squared_error(test_seasonal, seasonal_naive_forecast_test)
mse_holt_winters = mean_squared_error(test_seasonal, seasonal_hw_forecast_test)

# Printing the MSE values
print(f'Seasonal Naive MSE: {mse_seasonal_naive}')
print(f'Holt-Winters MSE: {mse_holt_winters}')


# RESIDUALS

# Modeling the residual component with ARIMA
auto_model_resid = auto_arima(resid.dropna(), start_p=0, start_q=0, max_p=5, max_q=5,
                              seasonal=False, stepwise=True, suppress_warnings=True, 
                              error_action='ignore', trace=True)

print("Best ARIMA Model for Residual Component:")
print(auto_model_resid.summary())

# Forecasting the residual component
residual_forecast = auto_model_resid.predict(n_periods=len(trend) - train_size)


# Assuming you have the actual residuals for the test set in a variable `test_resid`
test_resid = resid[train_size:]

# Calculate the MSE for the residuals
mse_resid_arima = mean_squared_error(test_resid, residual_forecast)
print(f'Residual ARIMA MSE: {mse_resid_arima}')

# Preparing the feature matrix (time) and target vector (residuals)
X_resid = np.arange(len(resid)).reshape(-1, 1)
y_resid = resid.values

# Splitting the data into training and testing sets
X_train_resid, X_test_resid = X_resid[:train_size], X_resid[train_size:]
y_train_resid, y_test_resid = y_resid[:train_size], y_resid[train_size:]

# Fitting the Linear Regression model
lin_reg_resid = LinearRegression()
lin_reg_resid.fit(X_train_resid, y_train_resid)

# Forecasting with the trained model
lin_reg_resid_forecast = lin_reg_resid.predict(X_test_resid)

# Calculating the MSE
mse_lin_reg_resid = mean_squared_error(y_test_resid, lin_reg_resid_forecast)
print(f'Linear Regression Residual MSE: {mse_lin_reg_resid}')

# Fitting the Random Forest Regressor
rf_resid = RandomForestRegressor(n_estimators=100, random_state=42)
rf_resid.fit(X_train_resid, y_train_resid)

# Forecasting with the trained model
rf_resid_forecast = rf_resid.predict(X_test_resid)

# Calculating the MSE
mse_rf_resid = mean_squared_error(y_test_resid, rf_resid_forecast)
print(f'Random Forest Residual MSE: {mse_rf_resid}')

# COMBINED FORECAST
combined_forecast = trend_forecast + seasonal_naive_forecast[:len(trend_forecast)] + residual_forecast

plt.figure(figsize=(10, 6))
plt.plot(oil_monthly_log_returns.index, oil_monthly_log_returns, label='Original')
plt.plot(oil_monthly_log_returns.index[train_size:], combined_forecast, label='Combined Forecast', color='red')
# Titles and labels
plt.title('Monthly Log Returns and Combined Forecast')
plt.xlabel('Date')
plt.ylabel('Log Returns')
plt.grid(True)
plt.legend()
plt.show()


test_monthly_log_returns = oil_monthly_log_returns[train_size:]

if len(combined_forecast) != len(test_monthly_log_returns):
    print("Length mismatch between forecast and actual data")
else:
    # Calculating MSE
    mse_combined = mean_squared_error(test_monthly_log_returns, combined_forecast)
    print(f'Combined Forecast MSE: {mse_combined}')

# Calculating RMSE
rmse_combined = np.sqrt(mean_squared_error(test_monthly_log_returns, combined_forecast))

# Calculating MAE
mae_combined = mean_absolute_error(test_monthly_log_returns, combined_forecast)

print(f'Combined Forecast RMSE: {rmse_combined}')
print(f'Combined Forecast MAE: {mae_combined}')
