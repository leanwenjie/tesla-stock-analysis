import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from rsi import calculate_rsi

def calculate_rsi(prices, period=14):
    prices = np.array(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

# Load data
csv_path = 'tesla_stock_cleaned.csv'
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])

# Feature engineering
# Calculate RSI if not present
if 'RSI' not in df.columns:
    df['RSI'] = calculate_rsi(df['Close'].values, period=14)

# Drop rows with missing values (from moving averages, RSI, etc.)
df = df.dropna().reset_index(drop=True)

# Create next day target for regression: next day's Close
df['Close_Next'] = df['Close'].shift(-1)
# Drop last row (no next day value)
df = df[:-1]

# Features to use
features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI']
X = df[features]
y_reg = df['Close_Next']

# Time-based train/test split (80% train, 20% test)
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

# --- Linear Regression ---
reg = LinearRegression()
reg.fit(X_train, y_reg_train)
y_reg_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print(f"\n[Linear Regression] Next Day Closing Price Prediction:")
print(f"RMSE: {rmse:.4f}")

# --- Random Forest Regression ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_reg_train)
y_rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_reg_test, y_rf_pred))
print(f"\n[Random Forest Regression] Next Day Closing Price Prediction:")
print(f"RMSE: {rf_rmse:.4f}")

# --- Gradient Boosting Regression ---
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_reg_train)
y_gb_pred = gb.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_reg_test, y_gb_pred))
print(f"\n[Gradient Boosting Regression] Next Day Closing Price Prediction:")
print(f"RMSE: {gb_rmse:.4f}")

# --- Plot actual vs predicted closing price for all models ---
plt.figure(figsize=(12,5))
plt.plot(df['Date'].iloc[split_idx:], y_reg_test, label='Actual Close', color='blue')
plt.plot(df['Date'].iloc[split_idx:], y_reg_pred, label='Linear Regression', color='orange')
plt.plot(df['Date'].iloc[split_idx:], y_rf_pred, label='Random Forest', color='green')
plt.plot(df['Date'].iloc[split_idx:], y_gb_pred, label='Gradient Boosting', color='red')
plt.title('Next Day Closing Price Prediction (All Models)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# --- 30-Day Recursive Forecasting for All Models ---
import datetime

future_days = 30
last_row = df.iloc[-1].copy()
future_dates = [last_row['Date'] + pd.Timedelta(days=i+1) for i in range(future_days)]

# Prepare rolling windows for MA and RSI
close_hist = list(df['Close'].values)
ma20_hist = list(df['MA20'].values)
ma50_hist = list(df['MA50'].values)

future_preds_lr = []
future_preds_rf = []
future_preds_gb = []

for i in range(future_days):
    # Prepare features for the next day
    # For Open, High, Low, Volume: use last known value (can be improved)
    open_ = last_row['Open']
    high_ = last_row['High']
    low_ = last_row['Low']
    volume_ = last_row['Volume']
    # For MA20, MA50: recalculate with new close
    ma20 = np.mean(close_hist[-19:]) if len(close_hist) >= 19 else np.mean(close_hist)
    ma50 = np.mean(close_hist[-49:]) if len(close_hist) >= 49 else np.mean(close_hist)
    # For RSI: recalculate with new close
    rsi = calculate_rsi(np.array(close_hist), period=14)[-1]
    # Build feature vector
    features_next = np.array([[close_hist[-1], open_, high_, low_, volume_, ma20, ma50, rsi]])
    # Predict with each model
    pred_lr = reg.predict(features_next)[0]
    pred_rf = rf.predict(features_next)[0]
    pred_gb = gb.predict(features_next)[0]
    # Store predictions
    future_preds_lr.append(pred_lr)
    future_preds_rf.append(pred_rf)
    future_preds_gb.append(pred_gb)
    # Update for next iteration
    close_hist.append(pred_lr)  # Use LR prediction for rolling features (could use ensemble or other)
    last_row['Close'] = pred_lr
    last_row['MA20'] = ma20
    last_row['MA50'] = ma50
    last_row['RSI'] = rsi

# Plot future predictions
plt.figure(figsize=(12,5))
plt.plot(df['Date'].iloc[split_idx:], y_reg_test, label='Actual Close', color='blue')
plt.plot(df['Date'].iloc[split_idx:], y_reg_pred, label='Linear Regression (Test)', color='orange')
plt.plot(df['Date'].iloc[split_idx:], y_rf_pred, label='Random Forest (Test)', color='green')
plt.plot(df['Date'].iloc[split_idx:], y_gb_pred, label='Gradient Boosting (Test)', color='red')
plt.plot(future_dates, future_preds_lr, label='Linear Regression (Future)', color='orange', linestyle='dashed')
plt.plot(future_dates, future_preds_rf, label='Random Forest (Future)', color='green', linestyle='dashed')
plt.plot(future_dates, future_preds_gb, label='Gradient Boosting (Future)', color='red', linestyle='dashed')
plt.title('Next Day Closing Price Prediction (Test & 30-Day Future Forecast)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show() 