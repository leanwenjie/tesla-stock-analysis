import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os
warnings.filterwarnings('ignore')

class TeslaStockAnalysis:
    def __init__(self):
        self.data = None
        self.technical_indicators = None
        # Create output directory for graphs
        self.output_dir = 'analysis_graphs'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def obtain_data(self, file_path='Tesla.csv.xlsx'):
        """
        Load Tesla stock data from Kaggle dataset
        """
        try:
            # Read the Excel file
            self.data = pd.read_excel(file_path)
            
            # Ensure date column is in datetime format
            if 'Date' in self.data.columns:
                self.data.rename(columns={'Date': 'date'}, inplace=True)
            elif 'date' not in self.data.columns:
                raise ValueError("Date column not found in the dataset")
                
            # Convert date to datetime if it's not already
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            # Sort data by date
            self.data.sort_values('date', inplace=True)
            
            # Display basic information about the dataset
            print("\nDataset Information:")
            print(f"Number of records: {len(self.data)}")
            print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
            print("\nColumns in the dataset:")
            print(self.data.columns.tolist())
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def calculate_technical_indicators(self):
        """
        Calculate technical indicators for analysis
        """
        if self.data is None:
            raise ValueError("Please obtain data first using obtain_data()")
            
        # Calculate Moving Averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['SMA_200'] = self.data['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        self.data['BB_middle'] = self.data['Close'].rolling(window=20).mean()
        self.data['BB_std'] = self.data['Close'].rolling(window=20).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (self.data['BB_std'] * 2)
        self.data['BB_lower'] = self.data['BB_middle'] - (self.data['BB_std'] * 2)
        
        # Calculate Daily Returns
        self.data['Daily_Return'] = self.data['Close'].pct_change() * 100
        
        # Calculate Cumulative Returns
        self.data['Cumulative_Return'] = (1 + self.data['Daily_Return']/100).cumprod() - 1
        
        return self.data
    
    def clean_data(self):
        """
        Clean and preprocess the data
        """
        if self.data is None:
            raise ValueError("Please obtain data first using obtain_data()")
            
        # Handle missing values
        self.data.fillna(method='ffill', inplace=True)  # Forward fill
        self.data.fillna(method='bfill', inplace=True)  # Backward fill for any remaining NaNs
        
        # Remove outliers using IQR method
        for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if column in self.data.columns:
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data[column] = self.data[column].clip(lower=lower_bound, upper=upper_bound)
        
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        if self.data is None:
            raise ValueError("Please obtain and clean data first")
            
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Price and Volume Plot
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.data['date'], self.data['Close'], label='Close Price')
        ax1.set_title('Tesla Stock Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # 2. Moving Averages
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(self.data['date'], self.data['Close'], label='Close Price')
        ax2.plot(self.data['date'], self.data['SMA_20'], label='20-day SMA')
        ax2.plot(self.data['date'], self.data['SMA_50'], label='50-day SMA')
        ax2.set_title('Moving Averages')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.legend()
        
        # 3. RSI
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(self.data['date'], self.data['RSI'], label='RSI')
        ax3.axhline(y=70, color='r', linestyle='--')
        ax3.axhline(y=30, color='g', linestyle='--')
        ax3.set_title('Relative Strength Index')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('RSI')
        ax3.legend()
        
        # 4. Bollinger Bands
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(self.data['date'], self.data['Close'], label='Close Price')
        ax4.plot(self.data['date'], self.data['BB_upper'], label='Upper Band')
        ax4.plot(self.data['date'], self.data['BB_middle'], label='Middle Band')
        ax4.plot(self.data['date'], self.data['BB_lower'], label='Lower Band')
        ax4.set_title('Bollinger Bands')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'technical_analysis.png'))
        plt.show()
        
        # Additional Analysis
        # 1. Daily Returns Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Daily_Return'].dropna(), bins=50, kde=True)
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Daily Return (%)')
        plt.savefig(os.path.join(self.output_dir, 'daily_returns_distribution.png'))
        plt.show()
        
        # 2. Cumulative Returns
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['date'], self.data['Cumulative_Return'] * 100)
        plt.title('Cumulative Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'cumulative_returns.png'))
        plt.show()
        
        # Print statistical summary
        print("\nStatistical Summary:")
        print(self.data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
        
        # Calculate and print correlation matrix
        correlation_matrix = self.data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.show()

def main():
    # Create instance of TeslaStockAnalysis
    analysis = TeslaStockAnalysis()
    
    # Obtain data
    print("Loading Tesla stock data from Kaggle dataset...")
    analysis.obtain_data()
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    analysis.calculate_technical_indicators()
    
    # Clean data
    print("\nCleaning data...")
    analysis.clean_data()
    
    # Explore data
    print("\nPerforming exploratory data analysis...")
    analysis.explore_data()

if __name__ == "__main__":
    main() 