from audioop import rms
import math
import yfinance as yf 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

# Get the data
stock_name = 'AAPL'
stock_start_date = '2016-01-01'
stock_end_date = '2021-10-01'
stock_data = yf.download(stock_name, start=stock_start_date, end=stock_end_date)
stock_data.head()


# Visualize the data
plt.figure(figsize=(15,8))
plt.title('Stock Prices History for ' + stock_name)
plt.plot(stock_data['Close'])
plt.xlabel('Date')
plt.ylabel('Prices ($)')
plt.show()

# Preprocess the data, make a 80 -20 split
print('Preprocessing the data...')
close_price = stock_data['Close']
values = close_price.values
training_data_len = math.ceil(len(values) * 0.8)

scalar = MinMaxScaler(feature_range=(0, 1)) # Normalize the data, side note could apply sigmoid instead maybe ?? 
scaled_data = scalar.fit_transform(values.reshape(-1, 1))
training_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

# Create a 60-days window of historical prices (i-60) as our feature data (x_train)
# and the following 60-days window as label date (y_train)
print('Creating training data...')
for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])


# Convert the data to numpy arrays for tensorflow reasons 
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Prepare the test set
print('Preparing the test set...')
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = values[training_data_len]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



# Prepare the model - copilot
# model = keras.Sequential([
#     layers.LSTM(units=50, return_sequences=True, input_shape=[x_train.shape[1], 1]),
#     layers.LSTM(units=50, return_sequences=False),
#     layers.Dense(1)
# ])

#  Actual model
print('Preparing the model...')
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()


# Train the model
print('Training the model...')
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=1)


# Evaluate the model
print('Evaluating the model...')
predictions = model.predict(x_test)
predictions = scalar.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)


# visualize the results
data = stock_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
