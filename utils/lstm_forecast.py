import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def forecast_lstm(expense_data, months=3):
    df = expense_data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.groupby(df['Date'].dt.to_period('M')).sum().reset_index()
    df['Date'] = df['Date'].dt.to_timestamp()
    df.set_index('Date', inplace=True)

    # Prepare data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Amount_Conv']])
    X, y = [], []
    time_steps = 3
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i-time_steps:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    # Forecast
    forecast_input = scaled[-time_steps:].reshape(1, time_steps, 1)
    predictions = []
    for _ in range(months):
        pred = model.predict(forecast_input)[0][0]
        predictions.append(pred)
        forecast_input = np.append(forecast_input[:, 1:, :], [[[pred]]], axis=1)

    forecasted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq='MS')
    return pd.Series(forecasted_values, index=forecast_dates)
