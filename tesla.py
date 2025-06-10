import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import streamlit as st
import plotly.graph_objs as go

# Load the dataset
df = pd.read_csv('Tesla_Stock_Updated_V2.csv')

# Convert 'Date' column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Remove duplicates
df = df.drop_duplicates()

# Check for missing values
print("Missing values in each column: ")
print(df.isnull().sum())

# Basic statistics
print("Basic statistics: ")
print(df.describe())

# Calculate daily returns
df['Daily_Return'] = df['Close'].pct_change() * 100

# Calculate moving averages
df['MA20'] = df['Close'].rolling(window=50).mean()
df['MA50'] = df['Close'].rolling(window=200).mean()

# Plot stock trends in 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(15,10))

# Subplot 1: Close Price & MAs
axs[0,0].plot(df.index, df['Close'], label='Close Price', color='blue')
axs[0,0].plot(df.index, df['MA20'], label='20-day MA', color='red')
axs[0,0].plot(df.index, df['MA50'], label='50-day MA', color='green')
axs[0,0].set_title('Tesla Stock Price and Moving Averages')
axs[0,0].legend()

# Subplot 2: Daily Returns Histogram
axs[0,1].hist(df['Daily_Return'].dropna(), bins=50, color='steelblue', edgecolor='black')
axs[0,1].set_title('Distribution of Daily Returns')

# Subplot 3: Volume Bar Plot
axs[1,0].bar(df.index, df['Volume'], color='dodgerblue', width=1.0)
axs[1,0].set_title('Trading Volume')

#Subplot 4: Price Range (High - Low)
df['Price_Range'] = df['High'] - df['Low']
axs[1,1].plot(df.index, df['Price_Range'], color='seagreen')
axs[1,1].set_title('Daily Price Range (High-Low)')

plt.tight_layout()
plt.show()

# Trading stats
total_days = len(df)
pos_days = len(df[df['Daily_Return'] > 0])
neg_days = len(df[df['Daily_Return'] < 0])

print(f"\nTotal trading days: {total_days}")
print(f"Positive days: {pos_days} ({(pos_days/total_days)*100:.2f}%)")
print(f"Negative days: {neg_days} ({(neg_days/total_days)*100:.2f}%)")

# Save cleaned data
df.to_csv('tesla_stock_cleaned.csv')


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

    epsilon = 1e-10
    rs = avg_gain / (avg_loss + epsilon)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi




# Apply RSI calculation
df['RSI'] = calculate_rsi(df['Close'].values, period=14)

# Plot RSI
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--', linewidth=2)
plt.axhline(30, color='green', linestyle='--', linewidth=2)
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()



csv_path = 'tesla_stock_cleaned.csv'
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])

# Calculate RSI if not available
if 'RSI' not in df.columns:
    df['RSI'] = calculate_rsi(df['Close'].values, period=14)

# Drop missing values
df = df.dropna().reset_index(drop=True)

# Create target: next day's close
df['Close_Next'] = df['Close'].shift(-1)
df = df[:-1]  # Drop last row with NaN target

features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI']
X = df[features]
y_reg = df['Close_Next']

# Time-based train/test split
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]


# Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_reg_train)
y_reg_pred = reg.predict(X_test)
print(f"[Linear Regression] RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)):.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_reg_train)
y_rf_pred = rf.predict(X_test)
print(f"[Random Forest] RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_rf_pred)):.4f}")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_reg_train)
y_gb_pred = gb.predict(X_test)
print(f"[Gradient Boosting] RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_gb_pred)):.4f}")


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


future_days = 30
last_row = df.iloc[-1].copy()
future_dates = [last_row['Date'] + pd.Timedelta(days=i+1) for i in range(future_days)]

close_hist = list(df['Close'].values)
future_preds_lr, future_preds_rf, future_preds_gb = [], [], []

for i in range(future_days):
    open_, high_, low_, volume_ = last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume']
    ma20 = np.mean(close_hist[-19:]) if len(close_hist) >= 19 else np.mean(close_hist)
    ma50 = np.mean(close_hist[-49:]) if len(close_hist) >= 49 else np.mean(close_hist)
    rsi = calculate_rsi(np.array(close_hist), period=14)[-1]
    features_next = pd.DataFrame({
    'Close': [close_hist[-1]],
    'Open': [open_],
    'High': [high_],
    'Low': [low_],
    'Volume': [volume_],
    'MA20': [ma20],
    'MA50': [ma50],
    'RSI': [rsi]
})

    pred_lr = reg.predict(features_next)[0]
    pred_rf = rf.predict(features_next)[0]
    pred_gb = gb.predict(features_next)[0]

    future_preds_lr.append(pred_lr)
    future_preds_rf.append(pred_rf)
    future_preds_gb.append(pred_gb)

    close_hist.append(pred_lr)
    last_row['Close'] = pred_lr
    last_row['MA20'] = ma20
    last_row['MA50'] = ma50
    last_row['RSI'] = rsi




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

st.set_page_config(page_title='Tesla Stock Forecast', layout='wide')
st.title('Tesla Stock Price Forecasting App')



df = pd.read_csv('Tesla_Stock_Updated_V2.csv')
df.rename(columns={'Adj Close': 'Close'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Feature engineering
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
df.dropna(inplace=True)
df['Close_Next'] = df['Close'].shift(-1)
df.dropna(inplace=True)

features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI']
X = df[features]
y = df['Close_Next']

split = int(0.8 * len(df))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Model selection
model_name = st.sidebar.selectbox('Select Model', ['Linear Regression', 'Random Forest', 'Gradient Boosting'])
if model_name == 'Linear Regression':
    model = LinearRegression()
elif model_name == 'Random Forest':
    model = RandomForestRegressor()
else:
    model = GradientBoostingRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
st.write(f'MAE: {mean_absolute_error(y_test, y_pred):.2f}')
st.write(f'R²: {r2_score(y_test, y_pred):.2f}')

# Forecast
test_dates = df['Date'].iloc[split:split + len(y_test)].values
n_days = st.sidebar.slider("Forecast Days", 1, 30, 5)
future_df = df.copy()
last_known_date = future_df['Date'].iloc[-1]
future_dates, future_prices = [], []

for i in range(n_days):
    last_row = future_df.iloc[-1]
    next_features = {
        'Close': last_row['Close'],
        'Open': last_row['Open'],
        'High': last_row['High'],
        'Low': last_row['Low'],
        'Volume': last_row['Volume'],
        'MA20': future_df['Close'].iloc[-20:].mean(),
        'MA50': future_df['Close'].iloc[-50:].mean(),
        'RSI': 100 - (100 / (1 + future_df['Close'].pct_change().iloc[-14:].mean()))
    }
    next_input = pd.DataFrame([next_features])
    next_close = model.predict(next_input)[0]
    next_date = last_known_date + pd.Timedelta(days=1)
    last_known_date = next_date
    future_dates.append(next_date)
    future_prices.append(next_close)
    next_row = {
        'Date': next_date,
        'Close': next_close,
        'Open': last_row['Open'],
        'High': last_row['High'],
        'Low': last_row['Low'],
        'Volume': last_row['Volume']
    }
    future_df = pd.concat([future_df, pd.DataFrame([next_row])], ignore_index=True)

# Plotting test period with Plotly
test_dates = pd.to_datetime(test_dates)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=test_dates, y=y_test.values,
                          mode='lines+markers', name='Actual',
                          hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'))
fig1.add_trace(go.Scatter(x=test_dates, y=y_pred,
                          mode='lines+markers', name='Predicted',
                          hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}<extra></extra>'))

fig1.update_layout(
    title='Tesla Stock Price Forecast (Test Period)',
    xaxis_title='Date',
    yaxis_title='Close Price',
    hovermode='x unified'
)

st.plotly_chart(fig1, use_container_width=True)

# Plotting future forecast with Plotly
future_dates_dt = pd.to_datetime(future_dates)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=future_dates_dt, y=future_prices,
                          mode='lines+markers',
                          name='Future Forecast',
                          hovertemplate='Date: %{x}<br>Forecasted Close: %{y:.2f}<extra></extra>'))

fig2.update_layout(
    title='Tesla Stock Price Future Forecast',
    xaxis_title='Date',
    yaxis_title='Forecasted Close Price',
    hovermode='x unified'
)

st.plotly_chart(fig2, use_container_width=True)

# Show data tables
st.subheader("Prediction Results")
st.dataframe(pd.DataFrame({'Date': test_dates, 'Actual': y_test.values, 'Predicted': y_pred}).reset_index(drop=True))

st.subheader("📈 Future Forecast")
st.dataframe(pd.DataFrame({'Date': future_dates, 'Forecasted Close': future_prices}).reset_index(drop=True))
