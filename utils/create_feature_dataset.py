import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os

def download_data(tickers, start_date, end_date):
    """Downloads historical data for a list of tickers from yfinance."""
    print(f"Downloading data for: {', '.join(tickers)}...")
    # Use "auto_adjust=True" to get adjusted close prices, which is standard
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d", auto_adjust=True)
    data.index = pd.to_datetime(data.index.date)
    return data

def calculate_technical_indicators(nifty_data):
    """Calculates all technical indicators on the Nifty 50 data."""
    print("Calculating technical indicators...")
    
    # Ensure columns are lowercase for pandas-ta
    nifty_data.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    }, inplace=True, errors='ignore')

    # Momentum
    nifty_data['RSI_14'] = nifty_data.ta.rsi(length=14)
    
    macd = nifty_data.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        nifty_data['MACD_12_26_9'] = macd[f'MACD_{12}_{26}_{9}']
        nifty_data['MACD_HIST_12_26_9'] = macd[f'MACDh_{12}_{26}_{9}']
        nifty_data['MACD_SIGNAL_12_26_9'] = macd[f'MACDs_{12}_{26}_{9}']
    else:
        print("Warning: MACD calculation failed.")

    stoch = nifty_data.ta.stoch(k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        nifty_data['STOCH_K_14_3_3'] = stoch[f'STOCHk_{14}_{3}_{3}']
        nifty_data['STOCH_D_14_3_3'] = stoch[f'STOCHd_{14}_{3}_{3}']
    else:
        print("Warning: Stochastic Oscillator calculation failed.")

    # Trend
    nifty_data['SMA_20'] = nifty_data.ta.sma(length=20)
    nifty_data['SMA_50'] = nifty_data.ta.sma(length=50)
    nifty_data['SMA_200'] = nifty_data.ta.sma(length=200)
    nifty_data['EMA_12'] = nifty_data.ta.ema(length=12)
    nifty_data['EMA_26'] = nifty_data.ta.ema(length=26)
    
    adx = nifty_data.ta.adx(length=14)
    if adx is not None and not adx.empty:
        nifty_data['ADX_14'] = adx[f'ADX_{14}']
    else:
        print("Warning: ADX calculation failed.")

    # Volatility
    bbands = nifty_data.ta.bbands(length=20, std=2)
    if bbands is not None and not bbands.empty:
        # Use f-strings with floats to match pandas-ta column names
        lower_col = f'BBL_{20}_{2.0}'
        mid_col = f'BBM_{20}_{2.0}'
        upper_col = f'BBU_{20}_{2.0}'
        width_col = f'BBB_{20}_{2.0}'
        
        if lower_col in bbands.columns:
            nifty_data['BBL_20_2'] = bbands[lower_col]
            nifty_data['BBM_20_2'] = bbands[mid_col]
            nifty_data['BBU_20_2'] = bbands[upper_col]
            nifty_data['BBB_20_2'] = bbands[width_col]
        else:
            print(f"Warning: Could not find expected Bollinger Bands columns (e.g., {lower_col}). Found: {bbands.columns}. Skipping BBands.")
    else:
        print("Warning: bbands calculation returned None or Empty. Skipping BBands.")
        
    nifty_data['ATR_14'] = nifty_data.ta.atr(length=14)

    # Volume
    nifty_data['OBV'] = nifty_data.ta.obv()

    # Other Transforms
    nifty_data['RET_DAILY'] = nifty_data['close'].pct_change()
    nifty_data['LOG_RET'] = nifty_data.ta.log_return(length=1)
    nifty_data['HIGH_LOW_SPREAD'] = nifty_data['high'] - nifty_data['low']
    nifty_data['OPEN_CLOSE_SPREAD'] = nifty_data['open'] - nifty_data['close']

    return nifty_data

def main():
    # --- 1. Configuration ---
    START_DATE = "2010-01-01"
    END_DATE = "2025-10-31" # <--- Updated to 2025 as you requested
    
    TICKERS = [
        '^NSEI', '^INDIAVIX', '^GSPC', '^IXIC', '^DJI', 
        '^N225', '^HSI', '^FTSE', '^GDAXI', 
        'INR=X', 'BZ=F', 'GC=F'
    ]
    
    OUTPUT_FILE = "Dataset/NIFTY50_features_wide.csv"
    
    # --- 2. Download yfinance Data ---
    # This is now the ONLY source of data, making it fully automatic
    all_data = download_data(TICKERS, START_DATE, END_DATE)
    
    # --- 3. Process Nifty 50 Data ---
    nifty_ohlcv = all_data.loc[:, (['Open', 'High', 'Low', 'Close', 'Volume'], '^NSEI')]
    nifty_ohlcv.columns = nifty_ohlcv.columns.droplevel(1) 
    nifty_ohlcv.dropna(inplace=True)
    
    # Calculate TA features
    nifty_with_ta = calculate_technical_indicators(nifty_ohlcv.copy())

    # --- 4. Prepare Other Features ---
    print("Preparing other market features...")
    other_features = all_data['Close'].drop(columns='^NSEI', errors='ignore')
    
    col_rename_map = {
        '^INDIAVIX': 'VIX_Close',
        '^GSPC': 'SP500_Close',
        '^IXIC': 'NASDAQ_Close',
        '^DJI': 'DOW_Close',
        '^N225': 'NIKKEI_Close',
        '^HSI': 'HANGSENG_Close',
        '^FTSE': 'FTSE_Close',
        '^GDAXI': 'DAX_Close',
        'INR=X': 'USDINR_Close',
        'BZ=F': 'BRENT_Close',
        'GC=F': 'GOLD_Close'
    }
    other_features.rename(columns=col_rename_map, inplace=True, errors='ignore')
    
    # --- 5. Combine Nifty TA + Other Market Features ---
    combined_df = nifty_with_ta.join(other_features)
    
    # --- DELETED: "Load and Merge Macro Data" Section ---
    # We removed the Repo Rate and CPI logic here.
    # Now the script will not ask for CSV files.

    # --- 6. Final Cleaning ---
    print("Cleaning final dataset (forward-filling and dropping NaNs)...")
    combined_df.fillna(method='ffill', inplace=True)
    
    initial_rows = len(combined_df)
    combined_df.dropna(inplace=True)
    final_rows = len(combined_df)
    
    print(f"Dropped {initial_rows - final_rows} rows with NaNs.")
    
    # Ensure 'RET_DAILY' is present (important for your return prediction)
    if 'RET_DAILY' not in combined_df.columns:
         print("Error: 'RET_DAILY' column missing! Check calculation function.")
         return

    # --- 7. Save to CSV ---
    print(f"Saving final dataset to {OUTPUT_FILE}...")
    combined_df.to_csv(OUTPUT_FILE)
    print(f"Successfully created {OUTPUT_FILE} with {len(combined_df.columns)} features.")

if __name__ == "__main__":
    main()
