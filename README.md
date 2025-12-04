# FreqSAMBA: Multi-Scale Stock Prediction System

**FreqSAMBA** is a deep learning stock forecasting framework that combines **Mamba** (State Space Models), **FFT-based Frequency Decomposition**, and **Graph Neural Networks (GNN)** to predict future stock prices.

Key features include:
- **Multi-Scale Architecture:** Splits time-series into Trend (Low Freq) and Residual (High Freq) using FFT.
- **Mamba Backbone:** Efficient sequence modeling using selective state space models.
- **Per-Window Scaling:** Handles non-stationary stock data by dynamically scaling each input window.
- **Production Workflow:** Clear separation of research (backtesting) and production (forecasting).

------------------------------------------------------------

## 📂 Project Structure

    ├── config/                   # Configuration classes
    │   └── model_config.py       # ModelArgs and TrainingConfig definitions
    ├── models/                   # Neural Network architecture
    │   ├── samba.py              # Main FreqSAMBA model (FFT + Mamba + GNN)
    │   ├── mamba.py              # Mamba Backbone implementation
    │   └── ...
    ├── trainer/                  # Training loop logic
    ├── utils/                    # Data loading, metrics, logging
    ├── create_feature_dataset.py # SCRIPT 1: Download & Prep Data
    ├── main.py                   # SCRIPT 2: Research & Hyperparameter Tuning
    ├── train_production.py       # SCRIPT 3: Final Model Training (100% Data)
    ├── predict_future.py         # SCRIPT 4: Forecasting
    ├── paper_config.py           # Central config file
    ├── test_system.py            # Environment & model verification
    └── requirements.txt          # Python dependencies

------------------------------------------------------------

## 🚀 Installation

Install dependencies:

    pip install -r requirements.txt

(Requires: torch, numpy, pandas, yfinance, pandas_ta, matplotlib, scikit-learn)

Verify installation:

    python test_system.py

------------------------------------------------------------

## 🛠️ Usage Workflow

### ✅ Step 1: Data Preparation

Downloads NIFTY50 data + technical indicators (RSI, MACD, Bollinger Bands).

    python create_feature_dataset.py

Output:

    Dataset/NIFTY50_features_wide.csv

------------------------------------------------------------

### ✅ Step 2: Research & Hyperparameter Tuning (Backtesting)

Splits data:  
- 80% → Dev  
- 20% → Test  

Runs grid search → evaluates → saves best hyperparameters.

    python main.py --mode train

Output:

    final_model_outputs/best_params.json

------------------------------------------------------------

### ✅ Step 3: Production Training (100% of Data)

Trains a fresh model using all available data with best hyperparameters.

    python train_production.py

Output:

    final_model_outputs/production_samba_model.pth

------------------------------------------------------------

### ✅ Step 4: Forecasting (Next 7 Days)

Generates future predictions using the production model.

    python predict_future.py

Output:
- Prints next 7 days of prices
- Saves forecast in final_model_outputs/

------------------------------------------------------------

## ⚙️ Configuration

All settings are in **paper_config.py**:

- **lag:** input sequence length (default: 60)
- **horizon:** days ahead to predict (default: 7)
- **cutoff_ratio:** FFT Trend/Residual split (default: 0.5)
- **d_model, n_layer:** Mamba architecture size parameters

------------------------------------------------------------

## 📊 Methodology Details

### 1️⃣ FFT Frequency Decomposition  
The input series is converted to frequency domain:

- **Trend:** lower 50% frequencies  
- **Residual:** higher 50% frequencies  

Both components capture different dynamics.

### 2️⃣ Dual-Branch Mamba Encoder  
Each component is processed by a dedicated Mamba network:

- Trend Mamba
- Residual Mamba

### 3️⃣ Fusion + Graph Neural Network  
A Chebyshev Graph Convolution (GNN) learns dependencies among features and merges both branches.

### 4️⃣ Inverse Scaling  
Every prediction is mapped back to real prices using the min/max values of the specific input window.

------------------------------------------------------------

# ✅ Summary

FreqSAMBA provides a **complete stock prediction system** consisting of:

- Automated dataset generation  
- Hyperparameter tuning  
- Backtesting  
- Production-grade training  
- 7-day forecasting pipeline  
- Multi-scale (FFT + Mamba + GNN) architecture  

All results and models are saved inside:

    final_model_outputs/

------------------------------------------------------------
