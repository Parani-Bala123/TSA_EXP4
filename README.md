# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 02/05/2026
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("timeseries_2year.csv")
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"])

series = df["stock_price"].diff().dropna().values
plt.rcParams["figure.figsize"] = (12, 4)
ar1 = np.array([1, -0.9])
ma1 = np.array([1,  0.4])

ARMA11 = ArmaProcess(ar1, ma1)
sim11   = ARMA11.generate_sample(nsample=1000)

plt.figure(figsize=(12, 4))
plt.plot(sim11, color='steelblue', linewidth=0.9)
plt.title("SIMULATED ARMA(1,1) PROCESS", fontsize=13, fontweight='bold')
plt.xlabel("Time")
plt.ylabel("Value")
plt.xlim(0, 1000)
plt.grid(True, alpha=0.3)
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
plot_pacf(sim11, lags=35, ax=ax, color='steelblue',
          title="Partial Autocorrelation — ARMA(1,1)")
ax.set_xlabel("Lag")
ax.set_ylabel("Partial Autocorrelation")
ax.grid(True, alpha=0.3)
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(sim11, lags=35, ax=ax, color='steelblue',
         title="Autocorrelation — ARMA(1,1)")
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.grid(True, alpha=0.3)
plt.show()

ar2 = np.array([1, -0.9,  0.5])
ma2 = np.array([1,  0.4,  0.3])

ARMA22 = ArmaProcess(ar2, ma2)
sim22   = ARMA22.generate_sample(nsample=10000)

plt.figure(figsize=(12, 4))
plt.plot(sim22, color='darkorange', linewidth=0.6)
plt.title("SIMULATED ARMA(2,2) PROCESS", fontsize=13, fontweight='bold')
plt.xlabel("Time")
plt.ylabel("Value")
plt.xlim(0, 10000)
plt.grid(True, alpha=0.3)
plt.show()
fig, ax = plt.subplots(figsize=(12, 4))
plot_pacf(sim22, lags=35, ax=ax, color='darkorange',
          title="Partial Autocorrelation — ARMA(2,2)")
ax.set_xlabel("Lag")
ax.set_ylabel("Partial Autocorrelation")
ax.grid(True, alpha=0.3)
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(sim22, lags=35, ax=ax, color='darkorange',
         title="Autocorrelation — ARMA(2,2)")
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.grid(True, alpha=0.3)
plt.show()
```

### OUTPUT:
#### SIMULATED ARMA(1,1) PROCESS:
<img width="1247" height="458" alt="image" src="https://github.com/user-attachments/assets/084e33f8-4690-4cb6-8378-b2c0eec8232b" />

#### Partial Autocorrelation:
<img width="1257" height="482" alt="image" src="https://github.com/user-attachments/assets/79517020-79b7-413d-bb49-6b9950500787" />

#### Autocorrelation:
<img width="1251" height="488" alt="image" src="https://github.com/user-attachments/assets/c2bef97c-6107-483f-a7ff-c7ef786a6533" />

#### SIMULATED ARMA(2,2) PROCESS:
<img width="1239" height="479" alt="image" src="https://github.com/user-attachments/assets/3ad5d194-4954-49a0-a7e1-6441642a5656" />

#### Partial Autocorrelation:
<img width="1248" height="470" alt="image" src="https://github.com/user-attachments/assets/c29493cb-2059-4a09-a8ac-03b13c8a56fd" />

#### Autocorrelation:
<img width="1240" height="479" alt="image" src="https://github.com/user-attachments/assets/02c5a4cd-e254-4270-a419-18a3402319bf" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
