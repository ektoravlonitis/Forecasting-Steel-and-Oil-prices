# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:36:09 2024

@author: he_98
"""

# PROBABILITY DISTRIBUTION

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import t
from scipy.stats import kstest

# Setting the font sizes
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 18  # Axis label font size
plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick label size
plt.rcParams['legend.fontsize'] = 16  # Legend font size

# Setting the start and end dates for the data
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

# Calculating log returns and monthly log returns
steel_log_returns = np.log(steel_prices / steel_prices.shift(1)).dropna()
oil_log_returns = np.log(oil_prices / oil_prices.shift(1)).dropna()

steel_monthly_log_returns = steel_log_returns.resample('M').sum()
oil_monthly_log_returns = oil_log_returns.resample('M').sum()

# Steel analysis

steel_monthly_log_returns_no_zeros = steel_monthly_log_returns[steel_monthly_log_returns != 0]

# Fitting the normal distribution to monthly log returns
mu, std = norm.fit(steel_monthly_log_returns_no_zeros)

# Fitting the log-normal distribution to monthly log returns
shape, loc, scale = lognorm.fit(steel_monthly_log_returns_no_zeros)

# Fitting the Student's t-distribution to the monthly log returns without zeros
params_no_zeros = t.fit(steel_monthly_log_returns_no_zeros)
df, t_loc, t_scale = params_no_zeros

# Kolmogorov-Smirnov tests
ks_statistic_norm, ks_pvalue_norm = kstest(steel_monthly_log_returns, 'norm', args=(mu, std))
ks_statistic_lognorm, ks_pvalue_lognorm = kstest(steel_monthly_log_returns, 'lognorm', args=(shape, loc, scale))
ks_statistic_t, ks_pvalue_t = kstest(steel_monthly_log_returns, 't', args=(df, t_loc, t_scale))

# Preparing the qqplots
qq_data = {
    "Normal": (norm, (mu, std)),
    "Log-normal": (lognorm, (shape, loc, scale)),
    "Student's t": (t, (df, t_loc, t_scale))
}

# Preparing the ks test results
ks_results = {
    "Normal": (ks_statistic_norm, ks_pvalue_norm),
    "Log-normal": (ks_statistic_lognorm, ks_pvalue_lognorm),
    "Student's t": (ks_statistic_t, ks_pvalue_t)
}

# Statistical Analysis
print("Statistical Analysis of Log Returns:")
print(f"Mean: {np.mean(steel_monthly_log_returns):.6f}")
print(f"Std Dev: {np.std(steel_monthly_log_returns):.6f}")
print(f"Skewness: {stats.skew(steel_monthly_log_returns):.6f}")
print(f"Kurtosis: {stats.kurtosis(steel_monthly_log_returns, fisher=False):.6f}")

# PDF fit plots
plt.figure(figsize=(14, 7))
plt.hist(steel_monthly_log_returns, bins=30, density=True, alpha=0.5, label='Data')

# Normal distribution PDF
x_values = np.linspace(steel_monthly_log_returns.min(), steel_monthly_log_returns.max(), 100)
plt.plot(x_values, norm.pdf(x_values, mu, std), 'r-', lw=4, label='Normal PDF')

# Log-normal distribution PDF
plt.plot(x_values, lognorm.pdf(x_values, shape, loc, scale), 'g-', lw=2, label='Log-normal PDF')

# Student's t-distribution PDF
plt.plot(x_values, t.pdf(x_values, df, t_loc, t_scale), 'b-', lw=2, label="Student's t PDF")
plt.legend()
plt.xlabel('Monthly Log Returns')
plt.ylabel('Density')
plt.title('Fitted Distributions Over Data Histogram')
plt.show()

# Combining QQ Plots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
distributions = [(norm, 'Normal'), (lognorm, 'Log-normal'), (t, 'Student\'s t')]
for i, (dist, label) in enumerate(distributions):
    sm.qqplot(steel_monthly_log_returns_no_zeros, line='s', dist=dist, ax=axs[i], fit=True)
    axs[i].set_title(f'{label} QQ Plot')
plt.tight_layout()
plt.show()

# Printing K-S test results
print(f"Normal distribution K-S statistic: {ks_statistic_norm}, p-value: {ks_pvalue_norm}")
print(f"Log-normal distribution K-S statistic: {ks_statistic_lognorm}, p-value: {ks_pvalue_lognorm}")
print(f"Student's t distribution K-S statistic: {ks_statistic_t}, p-value: {ks_pvalue_t}")

# Calculating the 95% Value at Risk (VaR)
var_95 = np.percentile(steel_monthly_log_returns_no_zeros, 5)
print(f"95% Value at Risk: {var_95:.4f}")

# Calculating the Expected Shortfall (ES) at 95%
es_95 = steel_monthly_log_returns_no_zeros[steel_monthly_log_returns_no_zeros <= var_95].mean()
print(f"95% Expected Shortfall: {es_95:.4f}")



# Crude Oil analysis

oil_monthly_log_returns_no_zeros = oil_monthly_log_returns[oil_monthly_log_returns != 0]

# Fitting the normal distribution to monthly log returns for Crude Oil
mu_oil, std_oil = norm.fit(oil_monthly_log_returns_no_zeros)

# Fitting the log-normal distribution to monthly log returns for Crude Oil
shape_oil, loc_oil, scale_oil = lognorm.fit(oil_monthly_log_returns_no_zeros)

# Fitting the Student's t-distribution to the monthly log returns without zeros for Crude Oil
params_no_zeros_oil = t.fit(oil_monthly_log_returns_no_zeros)
df_oil, t_loc_oil, t_scale_oil = params_no_zeros_oil  

# Kolmogorov-Smirnov tests for Crude Oil
ks_statistic_norm_oil, ks_pvalue_norm_oil = kstest(oil_monthly_log_returns, 'norm', args=(mu_oil, std_oil))
ks_statistic_lognorm_oil, ks_pvalue_lognorm_oil = kstest(oil_monthly_log_returns, 'lognorm', args=(shape_oil, loc_oil, scale_oil))
ks_statistic_t_oil, ks_pvalue_t_oil = kstest(oil_monthly_log_returns, 't', args=(df_oil, t_loc_oil, t_scale_oil))

# Statistical analysis for Crude Oil
print("Statistical Analysis of Crude Oil Log Returns:")
print(f"Mean: {np.mean(oil_monthly_log_returns):.6f}")
print(f"Std Dev: {np.std(oil_monthly_log_returns):.6f}")
print(f"Skewness: {stats.skew(oil_monthly_log_returns):.6f}")
print(f"Kurtosis: {stats.kurtosis(oil_monthly_log_returns, fisher=False):.6f}")

# PDF fit plots for Crude Oil
plt.figure(figsize=(14, 7))
plt.hist(oil_monthly_log_returns, bins=30, density=True, alpha=0.5, color='green', label='Data Oil')

# Normal distribution PDF for Crude Oil
x_values_oil = np.linspace(oil_monthly_log_returns.min(), oil_monthly_log_returns.max(), 100)
plt.plot(x_values_oil, norm.pdf(x_values_oil, mu_oil, std_oil), 'r-', lw=4, label='Normal PDF Oil')

# Log-normal distribution PDF for Crude Oil
plt.plot(x_values_oil, lognorm.pdf(x_values_oil, shape_oil, loc_oil, scale_oil), 'g-', lw=2, label='Log-normal PDF Oil')

# Student's t-distribution PDF for Crude Oil
plt.plot(x_values_oil, t.pdf(x_values_oil, df_oil, t_loc_oil, t_scale_oil), 'b-', lw=2, label="Student's t PDF Oil")
plt.legend()
plt.xlabel('Monthly Log Returns Oil')
plt.ylabel('Density')
plt.title('Fitted Distributions Over Data Histogram - Crude Oil')
plt.show()

# Combined QQ Plots for Crude Oil
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
distributions_oil = [(norm, 'Normal'), (lognorm, 'Log-normal'), (t, "Student's t")]
for i, (dist, label) in enumerate(distributions_oil):
    sm.qqplot(oil_monthly_log_returns_no_zeros, line='s', dist=dist, ax=axs[i], fit=True)
    axs[i].set_title(f'{label} QQ Plot - Oil')
plt.tight_layout()
plt.show()

# K-S test results for Crude Oil
print(f"Normal distribution K-S statistic Oil: {ks_statistic_norm_oil}, p-value: {ks_pvalue_norm_oil}")
print(f"Log-normal distribution K-S statistic Oil: {ks_statistic_lognorm_oil}, p-value: {ks_pvalue_lognorm_oil}")
print(f"Student's t distribution K-S statistic Oil: {ks_statistic_t_oil}, p-value: {ks_pvalue_t_oil}")

# Calculating the 95% Value at Risk (VaR) for Crude Oil
var_95_oil = np.percentile(oil_monthly_log_returns_no_zeros, 5)
print(f"95% Value at Risk - Oil: {var_95_oil:.4f}")

# Calculating the Expected Shortfall (ES) at 95% for Crude Oil
es_95_oil = oil_monthly_log_returns_no_zeros[oil_monthly_log_returns_no_zeros <= var_95_oil].mean()
print(f"95% Expected Shortfall - Oil: {es_95_oil:.4f}")


# Combined plots
plt.figure(figsize=(20, 7))

# For Steel
plt.subplot(1, 2, 1)
plt.hist(steel_monthly_log_returns, bins=30, density=True, alpha=0.5, color='blue', label='Steel Data')
x_values_steel = np.linspace(steel_monthly_log_returns.min(), steel_monthly_log_returns.max(), 100)
plt.plot(x_values_steel, norm.pdf(x_values_steel, mu, std), 'r-', lw=4, label='Normal PDF Steel')
plt.plot(x_values_steel, lognorm.pdf(x_values_steel, shape, loc, scale), 'g-', lw=2, label='Log-normal PDF Steel')
plt.plot(x_values_steel, t.pdf(x_values_steel, df, t_loc, t_scale), 'b-', lw=2, label="Student's t PDF Steel")
plt.title('Steel Fitted Distributions')
plt.xlabel('Monthly Log Returns')
plt.ylabel('Density')
plt.legend()

# For Crude Oil
plt.subplot(1, 2, 2)
plt.hist(oil_monthly_log_returns, bins=30, density=True, alpha=0.5, color='green', label='Crude Oil Data')
x_values_oil = np.linspace(oil_monthly_log_returns.min(), oil_monthly_log_returns.max(), 100)
plt.plot(x_values_oil, norm.pdf(x_values_oil, mu_oil, std_oil), 'r-', lw=4, label='Normal PDF Oil')
plt.plot(x_values_oil, lognorm.pdf(x_values_oil, shape_oil, loc_oil, scale_oil), 'g-', lw=2, label='Log-normal PDF Oil')
plt.plot(x_values_oil, t.pdf(x_values_oil, df_oil, t_loc_oil, t_scale_oil), 'b-', lw=2, label="Student's t PDF Oil")
plt.title('Crude Oil Fitted Distributions')
plt.xlabel('Monthly Log Returns')
plt.legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# For Steel
distributions = [("Normal", norm), ("Log-normal", lognorm), ("Student's t", t)]
for i, (label, dist) in enumerate(distributions):
    sm.qqplot(steel_monthly_log_returns_no_zeros, line='s', dist=dist, ax=axs[0, i], fit=True)
    axs[0, i].set_title(f'Steel {label} QQ Plot')

# For Crude Oil
distributions_oil = [("Normal", norm), ("Log-normal", lognorm), ("Student's t", t)]
for i, (label, dist) in enumerate(distributions_oil):
    sm.qqplot(oil_monthly_log_returns_no_zeros, line='s', dist=dist, ax=axs[1, i], fit=True)
    axs[1, i].set_title(f'Crude Oil {label} QQ Plot')

plt.tight_layout()
plt.show()