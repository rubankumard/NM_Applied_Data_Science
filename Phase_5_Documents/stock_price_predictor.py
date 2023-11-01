import numpy as np
import os
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

open_list = []
high_list = []
low_list = []
close_list = []
adj_close_list = []
vol_list = []

model = keras.models.load_model('MSFT.csv')

print("Stock Price Prediction!!!")
print("\nPlease enter the details of past 10 days")

for i in range(10):
    print(f"\nEnter the details of Day {i + 1}")
    a = float(input("\nEnter Opening Price:"))
    b = float(input("\nEnter Highest Price:"))
    c = float(input("\nEnter Lowest Price:"))
    d = float(input("\nEnter Closing Price:"))
    e = float(input("\nEnter Adjusted Closing Price:"))
    f = float(input("\nEnter Volume:"))
    open_list.append(a)
    high_list.append(b)
    low_list.append(c)
    close_list.append(d)
    adj_close_list.append(e)
    vol_list.append(f)

# Create a dictionary with the lists and column names
details = {
    "Opening Price": open_list,
    "Highest Price": high_list,
    "Lowest Price": low_list,
    "Closing Price": close_list,
    "Adjusted Closing Price": adj_close_list,
    "Volume": vol_list,
}

# Create a DataFrame
data = pd.DataFrame(details)

data['Range'] = data['Highest Price']-data['Lowest Price']

data['Daily Average'] = (data['Highest Price']+data['Lowest Price']+data['Closing Price']+data['Opening Price'])/4
data['Market Capitalization'] = data['Closing Price'] * data['Volume']
window = 10
data['SMA'] = data['Closing Price'].rolling(window=window).mean()
data['EMA'] = data['Closing Price'].ewm(span=window, adjust=False).mean()

delta = data['Closing Price'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=window).mean()
avg_loss = loss.rolling(window=window).mean()

rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
data['RSI'] = rsi

k_window = 10
d_window = 3   

low_min = data['Lowest Price'].rolling(window=k_window).min()
high_max = data['Highest Price'].rolling(window=k_window).max()

k = 100 * ((data['Closing Price'] - low_min) / (high_max - low_min))
data['%K'] = k

data['%D'] = data['%K'].rolling(window=d_window).mean()
data = data.fillna(method='bfill') 
data = data.drop(['Closing Price'],axis=1)

X = data.values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Define the sequence length (e.g., number of days to look back for predictions)
sequence_length =10  # You can adjust this as needed

# Create sequences for training y
X_sequences = []

for i in range(len(X) - sequence_length):
    X_sequences.append(X[i:i + sequence_length])

X_sequences = np.array(X_sequences)

predictions = model.predict(X_sequences)
print(f"Closing Price for the next day is: {predictions}")