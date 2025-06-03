import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title='Tesla Stock Forecast', layout='wide')
st.title('Tesla Stock Price Forecasting App')

# --- RSI Calculation Function ---
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

# --- File Upload ---
st.sidebar.header('Upload Dataset')
file = st.sidebar.file_uploader('Upload CSV', type=['csv'])
if file is not None:
    df = pd.read_csv(file)
else:
    st.info('Using default tesla_stock_cleaned.csv')
    df = pd.read_csv('tesla_stock_cleaned.csv')

# --- Feature Engineering ---
df['Date'] = pd.to_datetime(df['Date'])
if 'RSI' not in df.columns:
    df['RSI'] = calculate_rsi(df['Close'].values, period=14)
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df = df.dropna().reset_index(drop=True)
df['Close_Next'] = df['Close'].shift(-1)
df = df[:-1]
features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI']
X = df[features]
y_reg = df['Close_Next']
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

# --- Model Selection ---
model_name = st.sidebar.selectbox('Select Model', ['Linear Regression', 'Random Forest', 'Gradient Boosting'])
if model_name == 'Linear Regression':
    model = LinearRegression()
elif model_name == 'Random Forest':
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# --- Train and Predict ---
model.fit(X_train, y_reg_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
mae = mean_absolute_error(y_reg_test, y_pred)
r2 = r2_score(y_reg_test, y_pred)
st.write(f'**Test RMSE:** {rmse:.4f}')
st.write(f'**Test MAE:** {mae:.4f}')
st.write(f'**Test R²:** {r2:.4f}')

# --- 30-Day Recursive Forecast ---
future_days = st.sidebar.slider('Forecast Horizon (days)', 1, 180, 30)
last_row = df.iloc[-1].copy()
future_dates = [last_row['Date'] + pd.Timedelta(days=i+1) for i in range(future_days)]
close_hist = list(df['Close'].values)
future_preds = []
for i in range(future_days):
    open_ = last_row['Open']
    high_ = last_row['High']
    low_ = last_row['Low']
    volume_ = last_row['Volume']
    ma20 = np.mean(close_hist[-19:]) if len(close_hist) >= 19 else np.mean(close_hist)
    ma50 = np.mean(close_hist[-49:]) if len(close_hist) >= 49 else np.mean(close_hist)
    rsi = calculate_rsi(np.array(close_hist), period=14)[-1]
    features_next = np.array([[close_hist[-1], open_, high_, low_, volume_, ma20, ma50, rsi]])
    pred = model.predict(features_next)[0]
    future_preds.append(pred)
    close_hist.append(pred)
    last_row['Close'] = pred
    last_row['MA20'] = ma20
    last_row['MA50'] = ma50
    last_row['RSI'] = rsi

# --- Date Range Selection for Zoom ---
all_dates = pd.concat([df['Date'].iloc[split_idx:], pd.Series(future_dates)])
min_date = all_dates.min()
max_date = all_dates.max()
def_date1 = df['Date'].iloc[split_idx]
def_date2 = max_date
start_date, end_date = st.sidebar.date_input('Select date range to zoom', [def_date1, def_date2], min_value=min_date, max_value=max_date)

# --- Plot ---
st.subheader('Test Set and 30-Day Forecast')
fig, ax = plt.subplots(figsize=(12,5))
# Mask for zoomed range
test_mask = (df['Date'].iloc[split_idx:] >= pd.to_datetime(start_date)) & (df['Date'].iloc[split_idx:] <= pd.to_datetime(end_date))
future_mask = (pd.Series(future_dates) >= pd.to_datetime(start_date)) & (pd.Series(future_dates) <= pd.to_datetime(end_date))
ax.plot(df['Date'].iloc[split_idx:][test_mask], y_reg_test[test_mask], label='Actual Close', color='blue')
ax.plot(df['Date'].iloc[split_idx:][test_mask], y_pred[test_mask], label=f'{model_name} (Test)', color='orange')
ax.plot(pd.Series(future_dates)[future_mask], np.array(future_preds)[future_mask], label=f'{model_name} (Future)', color='red', linestyle='dashed')
ax.set_title('Next Day Closing Price Prediction and 30-Day Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig) 