import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the dataset
df = pd.read_csv('Tesla_Stock_Updated_V2.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index
df.set_index('Date', inplace=True)

# Basic data cleaning
# Remove any duplicate entries
df = df.drop_duplicates()

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Calculate daily returns
df['Daily_Return'] = df['Close'].pct_change() * 100

# Calculate moving averages
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# --- 2x2 Grid of Subplots for Tesla Stock Analysis ---
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Top left: Stock price and moving averages
axs[0, 0].plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
axs[0, 0].plot(df.index, df['MA20'], label='20-day MA', color='red', linewidth=2)
axs[0, 0].plot(df.index, df['MA50'], label='50-day MA', color='green', linewidth=2)
axs[0, 0].set_title('Tesla Stock Price and Moving Averages')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Price')
axs[0, 0].legend(fontsize=8)
axs[0, 0].grid(True, alpha=0.3)

# Top right: Distribution of daily returns
axs[0, 1].hist(df['Daily_Return'].dropna(), bins=50, color='steelblue', edgecolor='black')
axs[0, 1].set_title('Distribution of Daily Returns')
axs[0, 1].set_xlabel('Daily Return (%)')
axs[0, 1].set_ylabel('Count')
axs[0, 1].grid(True, alpha=0.3)

# Bottom left: Trading volume (bar plot)
axs[1, 0].bar(df.index, df['Volume'], color='dodgerblue', width=1.0)
axs[1, 0].set_title('Trading Volume')
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel('Volume')
axs[1, 0].grid(True, alpha=0.3)

# Bottom right: Daily price range (line plot)
df['Price_Range'] = df['High'] - df['Low']
axs[1, 1].plot(df.index, df['Price_Range'], color='seagreen')
axs[1, 1].set_title('Daily Price Range (High-Low)')
axs[1, 1].set_xlabel('Date')
axs[1, 1].set_ylabel('Price Range')
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some statistics about the moving averages
print("\nMoving Averages Statistics:")
print("\n20-day Moving Average:")
print(f"Current MA20: ${df['MA20'].iloc[-1]:.2f}")
print(f"MA20 Range: ${df['MA20'].min():.2f} - ${df['MA20'].max():.2f}")

print("\n50-day Moving Average:")
print(f"Current MA50: ${df['MA50'].iloc[-1]:.2f}")
print(f"MA50 Range: ${df['MA50'].min():.2f} - ${df['MA50'].max():.2f}")

# Calculate how often price is above/below each MA
price_above_ma20 = (df['Close'] > df['MA20']).mean() * 100
price_above_ma50 = (df['Close'] > df['MA50']).mean() * 100

print(f"\nPrice Statistics:")
print(f"Price is above MA20 {price_above_ma20:.1f}% of the time")
print(f"Price is above MA50 {price_above_ma50:.1f}% of the time")

# Calculate additional metrics
volatility = df['Daily_Return'].std()
print(f"\nAnnualized Volatility: {volatility * np.sqrt(252):.2f}%")

# Calculate trading statistics
total_trading_days = len(df)
positive_days = len(df[df['Daily_Return'] > 0])
negative_days = len(df[df['Daily_Return'] < 0])
print(f"\nTrading Statistics:")
print(f"Total Trading Days: {total_trading_days}")
print(f"Positive Days: {positive_days} ({positive_days/total_trading_days*100:.2f}%)")
print(f"Negative Days: {negative_days} ({negative_days/total_trading_days*100:.2f}%)")

# Save cleaned data
df.to_csv('tesla_stock_cleaned.csv')

def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    Parameters:
    -----------
    prices : array-like
        Array of price data (typically closing prices)
    period : int, default 14
        The period over which to calculate RSI
    Returns:
    --------
    numpy.ndarray
        Array containing RSI values
    """
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

# Calculate RSI (using Close price)
df['RSI'] = calculate_rsi(df['Close'].values, period=14)

# Plot RSI
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['RSI'], label='RSI', color='C0', linewidth=1)
plt.axhline(70, color='red', linestyle='--', linewidth=2)
plt.axhline(30, color='green', linestyle='--', linewidth=2)
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.ylim(0, 100)
plt.legend(['RSI'], loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 