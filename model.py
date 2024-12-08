import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(device)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(device)
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        predictions = self.linear(lstm_out[:, -1])
        return predictions

def get_binance_data(symbol, interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)
        return df
    else:
        raise Exception(f"Failed to retrieve data for {symbol}: {response.text}")

def prepare_dataset(symbol, sequence_length=10):
    logger.info(f"Fetching data for {symbol}")
    df = get_binance_data(symbol)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    
    sequences = []
    for i in range(sequence_length, len(scaled_data)):
        seq = scaled_data[i-sequence_length:i]
        label = scaled_data[i]
        sequences.append((seq, label))
    
    return sequences, scaler

def train_model(model, data, epochs=100, lr=0.001, sequence_length=10, patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for seq, label in data:
            seq = torch.FloatTensor(seq).view(1, sequence_length, -1).to(device)
            label = torch.FloatTensor(label).view(1, -1).to(device)

            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data)
        scheduler.step(avg_loss)
        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")

def main():
    logger.info("Starting model training process")
    
    symbols = ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ARBUSDT']
    
    for symbol in symbols:
        logger.info(f"Training model for {symbol}")
        model = LSTMModel(input_size=1, hidden_layer_size=128, output_size=1, num_layers=2, dropout=0.2).to(device)
        
        data, scaler = prepare_dataset(symbol)
        train_model(model, data, epochs=100)
        
        # Save the model and scaler
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), f"models/{symbol}_model.pth")
        joblib.dump(scaler, f"models/{symbol}_scaler.save")
        
        logger.info(f"Model and scaler for {symbol} saved.")

if __name__ == "__main__":
    main()