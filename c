import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess the data
data = pd.read_csv('PAGS.csv')
prices = data['Close'].values.reshape(-1, 1)  # Adjust based on your data

# Normalize the data
scaler = MinMaxScaler()
prices_normalized = scaler.fit_transform(prices)

# Prepare the data
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[(i + time_steps), 0])
    return np.array(X), np.array(y)

time_steps = 10  # Adjust the number of time steps based on your preference
X, y = prepare_data(prices_normalized, time_steps)

# Reshape the input data to fit the LSTM input shape
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

# Forecast future prices
future_time_steps = 5  # Number of future time steps to predict
last_sequence = prices_normalized[-time_steps:]  # Use the last sequence from the data as input

future_predictions = []
for _ in range(future_time_steps):
    # Reshape the input sequence
    sequence = last_sequence[-time_steps:].reshape(1, time_steps, 1)
    # Make the next price prediction
    prediction = model.predict(sequence)
     # Append the prediction to the results
    future_predictions.append(prediction[0, 0])
    # Update the last sequence with the predicted value
    last_sequence = np.append(last_sequence, prediction[0, 0])
    
# Inverse transform the predictions
future_predictions = scaler.inverse_transform([future_predictions]).flatten()

# Convert dates to strings
historical_dates = data['Date'].astype(str)
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=future_time_steps + 1).tolist()[1:]
future_dates = [str(date.date()) for date in future_dates]

# Concatenate all dates
all_dates = np.concatenate((historical_dates.values, future_dates))

# Plotting the graph
plt.figure(figsize=(200, 40))
plt.plot(historical_dates, prices, label='Historical')
plt.plot(future_dates, future_predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.show()

