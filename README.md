# Stock Price Prediction using LSTM

This project predicts stock prices using a bidirectional LSTM (Long Short-Term Memory) model. The model is trained on historical stock price data for Apple Inc. (AAPL) obtained from Yahoo Finance.

## Features
- Fetches stock price data using `yfinance`.
- Preprocesses data by normalizing and creating sequences.
- Trains a deep learning model using bidirectional LSTM layers.
- Saves and loads the trained model for future use.
- Predicts stock prices and visualizes results.
- Computes Mean Absolute Percentage Error (MAPE) for evaluation.

## Requirements
Ensure you have Python installed along with the required dependencies:

```bash
pip install numpy pandas yfinance matplotlib scikit-learn keras tensorflow
```

## Setup and Running the Project

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Run the script**
   ```bash
   python stock_price_prediction.py
   ```

   The script will:
   - Fetch historical stock data.
   - Train a new LSTM model if no saved model exists.
   - Load a previously trained model if available.
   - Make predictions and display the actual vs. predicted stock prices.

## How It Works
1. **Data Collection:**
   - Fetches AAPL stock prices from Yahoo Finance.
   - Normalizes closing prices using MinMaxScaler.
   - Creates sequences for LSTM training.

2. **Model Training & Prediction:**
   - Defines a bidirectional LSTM model with multiple layers.
   - Trains the model using historical data.
   - Saves the trained model as `stock-price-lstm-model.keras`.
   - Predicts future stock prices.

3. **Visualization:**
   - Plots actual vs. predicted prices.
   - Displays the Mean Absolute Percentage Error (MAPE) on the graph.

## Example Output
- A graph showing actual vs. predicted stock prices.
- MAPE displayed on the plot to assess model accuracy.

## Notes
- If no model exists, training may take time depending on system performance.
- The dataset used is limited to Apple Inc. (AAPL) stock prices; modify the ticker symbol in the script to analyze different stocks.

## License
This project is open-source and available under the MIT License.

