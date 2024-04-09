# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:07:16 2024

@author: he_98
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

# Setting the font sizes
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 18  # Axis label font size
plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick label size
plt.rcParams['legend.fontsize'] = 16  # Legend font size

# Data retrieval
start_date = datetime(2005, 1, 1)
end_date = datetime(2020, 12, 30)

# Tickers
steel_ticker = 'Steel'  # Steel's ticker
oil_ticker = 'CL=F'  # Crude oil futures ticker

# Fetching the data
steel_data = yf.download(steel_ticker, start=start_date, end=end_date)
oil_data = yf.download(oil_ticker, start=start_date, end=end_date)

# Using 'Close' prices for analysis
steel_prices = steel_data['Close'].dropna()
oil_prices = oil_data['Close'].dropna()

# Calculating log returns
steel_log_returns = np.log(steel_prices / steel_prices.shift(1)).dropna()
oil_log_returns = np.log(oil_prices / oil_prices.shift(1)).dropna()

# Splitting the data into training and testing sets
split_ratio = 0.8
split_index = int(len(steel_log_returns) * split_ratio)

log_returns_train = steel_log_returns[:split_index]
log_returns_test = steel_log_returns[split_index:]

# Defining ranges for p and q
p_range = range(1, 6)
q_range = range(1, 6)

best_aic = np.inf
best_bic = np.inf
best_model = None

# Finding the best GARCH model
for p in p_range:
    for q in q_range:
        try:
            # Fitting the GARCH model
            model = arch_model(log_returns_train, p=p, q=q, mean='constant', dist='Normal')
            model_fit = model.fit(disp='off')

            # Checking if the model has a lower AIC or BIC than the current best
            if model_fit.aic < best_aic or model_fit.bic < best_bic:
                best_aic = model_fit.aic
                best_bic = model_fit.bic
                best_model = model_fit
                best_params = (p, q)
                
        except Exception as e:
            print(f"Could not fit GARCH({p},{q}) model: {str(e)}")

# Best model's parameters and criteria
print(f"Best Model: GARCH({best_params[0]},{best_params[1]})")
print(f"Best AIC: {best_aic}")
print(f"Best BIC: {best_bic}")
print(best_model.summary())

# Best p, q values for the GARCH model
best_p, best_q = best_params

# Fitting the best model on the training data
best_garch_model = arch_model(log_returns_train, vol='Garch', p=best_p, q=best_q, mean='constant', dist='Normal')
best_fit = best_garch_model.fit(disp='off')

# Calculate rolling volatility
rolling_volatility_all = steel_log_returns.rolling(window=30).std()

# Dropping NaN values
rolling_volatility_all = rolling_volatility_all.dropna()

# Splitting the data
volatility_train = rolling_volatility_all[:split_index]
volatility_test = rolling_volatility_all[split_index:]

# Calculating standardized residuals from the model
std_resid = best_fit.std_resid

plt.figure(figsize=(10, 6))
plt.plot(std_resid, color='blue', linewidth=1)
plt.title('Standardized Residuals from the GARCH Model', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Standardized Residuals', fontsize=14)
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
plot_acf(std_resid**2, ax=ax, lags=40, alpha=0.05)
ax.set_title('ACF of Squared Standardized Residuals', fontsize=16)
ax.set_xlabel('Lag', fontsize=14)
ax.set_ylabel('Autocorrelation', fontsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Fitting a GARCH(1,1) model for comparison
simple_model = arch_model(log_returns_train, vol='Garch', p=1, q=1, mean='constant', dist='Normal')
simple_fit = simple_model.fit(disp='off')

print(simple_fit.summary())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(best_fit.conditional_volatility, label='Conditional Volatility', color='blue')
ax.set_title('Conditional Volatility from GARCH Model', fontsize=16)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Volatility', fontsize=14)
ax.legend()
plt.grid(True)
plt.show()

# Performing the Ljung-Box test
ljung_box_result = acorr_ljungbox(best_fit.std_resid**2, lags=[10], return_df=True)
print(ljung_box_result)

# Forecasting next day volatility
forecasted_volatility_44 = []
forecasted_volatility_11 = []

for i in range(29, len(log_returns_test)):
    current_window = steel_log_returns[:split_index + i]
    
    # Fitting the GARCH(4,4) model
    garch_44_model = arch_model(current_window, vol='Garch', p=4, q=4, mean='constant', dist='Normal')
    garch_44_fit = garch_44_model.fit(last_obs=i+split_index, disp='off')
    
    # Forecasting the next step
    forecast_44 = garch_44_fit.forecast(horizon=1).variance.values[-1, :]
    forecasted_volatility_44.append(np.sqrt(forecast_44))

    # Fitting the GARCH(1,1) model
    garch_11_model = arch_model(current_window, vol='Garch', p=1, q=1, mean='constant', dist='Normal')
    garch_11_fit = garch_11_model.fit(last_obs=i+split_index, disp='off', update_freq=5)
    
    # Forecasting the next step
    forecast_11 = garch_11_fit.forecast(horizon=1).variance.values[-1, :]
    forecasted_volatility_11.append(np.sqrt(forecast_11))

mse_44 = np.mean((forecasted_volatility_44 - volatility_test.values)**2)
mse_11 = np.mean((forecasted_volatility_11 - volatility_test.values)**2)

print(f'MSE for GARCH(4,4): {mse_44}')
print(f'MSE for GARCH(1,1): {mse_11}')

plt.plot(volatility_test.index, volatility_test, label='Actual Volatility')
plt.plot(volatility_test.index, forecasted_volatility_44, label='Forecasted Volatility', color='red')
plt.title('GARCH Model Volatility Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)

# Finding the best ARIMA model automatically
auto_arima_model = pm.auto_arima(volatility_train, 
                                 error_action='ignore', 
                                 trace=True, 
                                 suppress_warnings=True, 
                                 seasonal=False, 
                                 stepwise=True)

print(auto_arima_model.summary())

# Forecasting using the best ARIMA model found
n_periods = len(volatility_test)
arima_forecast_vol, confint = auto_arima_model.predict(n_periods=n_periods, return_conf_int=True)

# Plotting the forecast against the actual values
plt.figure(figsize=(10, 6))
plt.plot(volatility_test.index, volatility_test, label='Actual Volatility')
plt.plot(volatility_test.index, arima_forecast_vol, label='Forecasted Volatility', color='red')
plt.fill_between(volatility_test.index, confint[:, 0], confint[:, 1], color='pink', alpha=0.3)
plt.title('Volatility Forecast vs Actuals')
plt.legend()
plt.show()
arima_forecast_vol.index = volatility_test.index

mse_arima = np.mean((arima_forecast_vol - volatility_test) ** 2)
print(f'MSE for ARIMA: {mse_arima}')

#%%

# Crude Oil

# Splitting the data into training and testing sets
split_ratio = 0.8
split_index = int(len(oil_log_returns) * split_ratio)

oil_log_returns_train = oil_log_returns[:split_index]
oil_log_returns_test = oil_log_returns[split_index:]

# Defining ranges for p and q
p_range = range(1, 6)
q_range = range(1, 6)

best_aic = np.inf
best_bic = np.inf
best_model = None

# Finding the best GARCH model
for p in p_range:
    for q in q_range:
        try:
            # Fitting the GARCH model
            model = arch_model(oil_log_returns_train, p=p, q=q, mean='constant', dist='Normal')
            model_fit = model.fit(disp='off')

            # Checking if this model has a lower AIC or BIC than the current best
            if model_fit.aic < best_aic or model_fit.bic < best_bic:
                best_aic = model_fit.aic
                best_bic = model_fit.bic
                best_model = model_fit
                best_params = (p, q)
                
        except Exception as e:
            print(f"Could not fit GARCH({p},{q}) model: {str(e)}")

# Best model's parameters and criteria
print(f"Best Model: GARCH({best_params[0]},{best_params[1]})")
print(f"Best AIC: {best_aic}")
print(f"Best BIC: {best_bic}")
print(best_model.summary())

# Best p, q values for the GARCH model
best_p, best_q = best_params

# Fitting the best model on the training data
oil_best_garch_model = arch_model(oil_log_returns_train, vol='Garch', p=best_p, q=best_q, mean='constant', dist='Normal')
oil_best_fit = oil_best_garch_model.fit(disp='off')

# Calculating rolling volatility for the entire series
oil_rolling_volatility_all = oil_log_returns.rolling(window=30).std()

# Dropping NaN values
oil_rolling_volatility_all = oil_rolling_volatility_all.dropna()

# Splitting the data
oil_volatility_train = oil_rolling_volatility_all[:split_index]
oil_volatility_test = oil_rolling_volatility_all[split_index:]

# Calculating standardized residuals from the model
oil_std_resid = oil_best_fit.std_resid

plt.figure(figsize=(10, 6))
plt.plot(oil_std_resid, color='blue', linewidth=1)
plt.title('Standardized Residuals from the GARCH Model', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Standardized Residuals', fontsize=14)
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
plot_acf(oil_std_resid**2, ax=ax, lags=40, alpha=0.05)
ax.set_title('ACF of Squared Standardized Residuals', fontsize=16)
ax.set_xlabel('Lag', fontsize=14)
ax.set_ylabel('Autocorrelation', fontsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Fitting a GARCH(1,1) model for comparison
simple_model = arch_model(oil_log_returns_train, vol='Garch', p=1, q=1, mean='constant', dist='Normal')
simple_fit = simple_model.fit(disp='off')

print(simple_fit.summary())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(oil_best_fit.conditional_volatility, label='Conditional Volatility', color='blue')
ax.set_title('Conditional Volatility from GARCH Model', fontsize=16)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Volatility', fontsize=14)
ax.legend()
plt.grid(True)
plt.show()

# Performing the Ljung-Box test
oil_ljung_box_result = acorr_ljungbox(oil_best_fit.std_resid**2, lags=[10], return_df=True)
print(oil_ljung_box_result)

oil_forecasted_volatility_25 = []
oil_forecasted_volatility_11 = []

for i in range(29, len(oil_log_returns_test)):
    current_window = oil_log_returns[:split_index + i]
    
    # Fitting the GARCH(2,5) model
    garch_25_model = arch_model(current_window, vol='Garch', p=2, q=5, mean='constant', dist='Normal')
    garch_25_fit = garch_25_model.fit(last_obs=i+split_index, disp='off')
    
    # Forecasting the next step
    forecast_25 = garch_25_fit.forecast(horizon=1).variance.values[-1, :]
    oil_forecasted_volatility_25.append(np.sqrt(forecast_25))

    # Fitting the GARCH(1,1) model
    garch_11_model = arch_model(current_window, vol='Garch', p=1, q=1, mean='constant', dist='Normal')
    garch_11_fit = garch_11_model.fit(last_obs=i+split_index, disp='off', update_freq=5)
    
    # Forecasting the next step
    forecast_11 = garch_11_fit.forecast(horizon=1).variance.values[-1, :]
    oil_forecasted_volatility_11.append(np.sqrt(forecast_11))

oil_mse_25 = np.mean((oil_forecasted_volatility_25 - oil_volatility_test.values)**2)
oil_mse_11 = np.mean((oil_forecasted_volatility_11 - oil_volatility_test.values)**2)

print(f'MSE for GARCH(2,5): {oil_mse_25}')
print(f'MSE for GARCH(1,1): {oil_mse_11}')

plt.plot(oil_volatility_test.index, oil_volatility_test, label='Actual Volatility')
plt.plot(oil_volatility_test.index, oil_forecasted_volatility_25, label='Forecasted Volatility', color='red')
plt.title('GARCH Model Volatility Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)

# Finding the best ARIMA model automatically
oil_auto_arima_model = pm.auto_arima(oil_volatility_train, 
                                 error_action='ignore', 
                                 trace=True, 
                                 suppress_warnings=True, 
                                 seasonal=False, 
                                 stepwise=True)

print(oil_auto_arima_model.summary())

# Forecasting using the best ARIMA model found
n_periods = len(oil_volatility_test)
oil_arima_forecast_vol, oil_confint = oil_auto_arima_model.predict(n_periods=n_periods, return_conf_int=True)

# Plotting the forecast against the actual values
plt.figure(figsize=(10, 6))
plt.plot(oil_volatility_test.index, oil_volatility_test, label='Actual Volatility')
plt.plot(oil_volatility_test.index, oil_arima_forecast_vol, label='Forecasted Volatility', color='red')
plt.fill_between(oil_volatility_test.index, oil_confint[:, 0], oil_confint[:, 1], color='pink', alpha=0.3)
plt.title('Volatility Forecast vs Actuals')
plt.legend()
plt.show()
oil_arima_forecast_vol.index = oil_volatility_test.index

oil_mse_arima = np.mean((oil_arima_forecast_vol - oil_volatility_test) ** 2)
print(f'MSE for ARIMA: {oil_mse_arima}')


#%%

# PLOTS TOGETHER

# Plotting for steel and oil side by side with rotated x-axis labels
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Plotting for steel on the left
axs[0].plot(volatility_test.index, volatility_test, label='Actual Volatility - Steel')
axs[0].plot(volatility_test.index, forecasted_volatility_44, label='Forecasted Volatility - Steel', color='red')
axs[0].set_title('Steel Volatility Forecast vs Actuals')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Volatility')
axs[0].legend()
axs[0].grid(True)
axs[0].tick_params(axis='x', rotation=45)

# Plotting for oil on the right
axs[1].plot(oil_volatility_test.index, oil_volatility_test, label='Actual Volatility - Oil')
axs[1].plot(oil_volatility_test.index, oil_forecasted_volatility_25, label='Forecasted Volatility - Oil', color='red')
axs[1].set_title('Oil Volatility Forecast vs Actuals')
axs[1].set_xlabel('Date')
axs[1].legend()
axs[1].grid(True)
axs[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Standardized Residuals for Steel
axs[0].plot(std_resid, color='blue', linewidth=1)
axs[0].set_title('Standardized Residuals from the GARCH Model - Steel', fontsize=16)
axs[0].set_xlabel('Time', fontsize=14)
axs[0].set_ylabel('Standardized Residuals', fontsize=14)
axs[0].axhline(y=0, color='r', linestyle='--')
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Standardized Residuals for Oil
axs[1].plot(oil_std_resid, color='green', linewidth=1)
axs[1].set_title('Standardized Residuals from the GARCH Model - Oil', fontsize=16)
axs[1].set_xlabel('Time', fontsize=14)
axs[1].axhline(y=0, color='r', linestyle='--')
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# ACF of Squared Standardized Residuals for Steel
plot_acf(std_resid**2, ax=axs[0], lags=40, alpha=0.05)
axs[0].set_title('ACF of Squared Standardized Residuals - Steel', fontsize=16)
axs[0].set_xlabel('Lag', fontsize=14)
axs[0].set_ylabel('Autocorrelation', fontsize=14)
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# ACF of Squared Standardized Residuals for Oil
plot_acf(oil_std_resid**2, ax=axs[1], lags=40, alpha=0.05)
axs[1].set_title('ACF of Squared Standardized Residuals - Oil', fontsize=16)
axs[1].set_xlabel('Lag', fontsize=14)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Conditional Volatility from GARCH Model for Steel
axs[0].plot(best_fit.conditional_volatility, label='Conditional Volatility', color='blue')
axs[0].set_title('Conditional Volatility - Steel', fontsize=16)
axs[0].set_xlabel('Time', fontsize=14)
axs[0].set_ylabel('Volatility', fontsize=14)
axs[0].legend()
axs[0].grid(True)

# Conditional Volatility from GARCH Model for Oil
axs[1].plot(oil_best_fit.conditional_volatility, label='Conditional Volatility', color='green')
axs[1].set_title('Conditional Volatility - Oil', fontsize=16)
axs[1].set_xlabel('Time', fontsize=14)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
