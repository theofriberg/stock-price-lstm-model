import os
from types import NoneType
import numpy as np
from pandas import DataFrame
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras.api.models import Sequential, load_model, Model
from keras.api.layers import Dense, LSTM, Dropout, Bidirectional

def fetch_data(scaler: MinMaxScaler) -> tuple[DataFrame, np.ndarray]:
    """Fetches data using the yfinance library.

    Args:
        scaler (MinMaxScaler): An sklearn scaler that scales data into a fixed range.

    Raises:
        ValueError: If the data could not be fetched.

    Returns:
        tuple[DataFrame, np.ndarray]: A tuple consisting of the original data as a pandas DataFrame
                                      and a np array of the scaled data.
    """
    df = yf.download('AAPL', start='2015-01-01', end='2024-01-01')

    if not isinstance(df, NoneType):
        closing_price_data = df[['Close']]

        scaled_data = scaler.fit_transform(closing_price_data)
        return df, scaled_data
    else:
        raise ValueError('Could not get data.')

def create_sequences(data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Creates time series sequences of data for training.

    Args:
        data (np.ndarray): Data to create sequences from.
        seq_length (int): Length of sequences.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple consisting of an np array with the training data
                                       and a np array of the correct price (validation) for a given sequence.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)


scaler = MinMaxScaler(feature_range=(0, 1))
df, scaled_data = fetch_data(scaler)

# Visualize the DataFrame
print(df.head())
print(df.tail())
plt.plot(df.index, df[['Close']])
plt.title('AAPL stock price data')
plt.show()

# Create data sequences
seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split into test and training data
train_size = int(len(X) * 0.8) # Use 80% as training data and 20% as test data
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = None
if os.path.exists('stock-price-lstm-model.keras'):
    print('Loading saved model...')
    model = load_model('stock-price-lstm-model.keras', compile=True)
else:
    # Define the model
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, input_shape=(60, 1))),
        Dropout(0.2),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))
    model.save('stock-price-lstm-model.keras')

if isinstance(model, Model):
    # Make predictions
    predictions = model.predict(X_test)
    
    # Quantify error
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    # Rescale predictions into price data
    predictions = scaler.inverse_transform(predictions).flatten()
    

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[train_size+seq_length:], df[['Close']][train_size+seq_length:], label='Actual Price')
    plt.plot(df.index[train_size+seq_length:], predictions, label='Predicted Price')

      # Add MAPE to plot
    print(f'\n{mape}')
    plt.text(
        0.05, 0.95, f'MAPE: {mape:.2%}', 
        transform=plt.gca().transAxes, fontsize=12, 
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8)
    )

    plt.legend()
    plt.show()