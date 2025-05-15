import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
import os
import yfinance as yf
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

class TeslaStockAnalysis:
    def __init__(self):
        self.data = None
        self.models = {}
        self.forecasts = {}
        # Create output directory for graphs
        self.output_dir = 'analysis_graphs'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def obtain_data(self, file_path='Tesla_Stock_Updated_V2.csv'):
        """
        Load Tesla stock data from CSV file
        """
        try:
            # Read the CSV file
            self.data = pd.read_csv(file_path)
            
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
            raise ValueError("Please obtain data first using obtain_data()")
            
        # Plot time series
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['date'], self.data['Close'])
        plt.title('Tesla Stock Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.show()

        # Plot moving averages
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['date'], self.data['SMA_20'], label='SMA_20')
        plt.plot(self.data['date'], self.data['SMA_50'], label='SMA_50')
        plt.plot(self.data['date'], self.data['SMA_200'], label='SMA_200')
        plt.title('Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

        # Plot Bollinger Bands
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['date'], self.data['BB_middle'], label='BB_middle')
        plt.plot(self.data['date'], self.data['BB_upper'], label='BB_upper')
        plt.plot(self.data['date'], self.data['BB_lower'], label='BB_lower')
        plt.title('Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

        # Plot daily returns
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['date'], self.data['Daily_Return'])
        plt.title('Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.show()

        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['date'], self.data['Cumulative_Return'] * 100)
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.show()

    def model_data(self):
        """Model the data using multiple approaches"""
        print("\nModeling Data...")
        
        # Prepare data for modeling
        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Create features for XGBoost
        df['Returns'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Target'] = df['Close'].shift(-1)  # Next day's closing price
        
        # Drop NaN values
        df = df.dropna()
        
        # Split data
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]
        
        # Prepare features for XGBoost
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20', 'Volatility']
        X_train = train_data[feature_columns]
        y_train = train_data['Target']
        X_test = test_data[feature_columns]
        y_test = test_data['Target']
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Prophet': Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'KNN Regression': KNeighborsRegressor(n_neighbors=5)
        }
        
        # Train and evaluate models
        results = {}
        forecasts = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            if model_name == 'Prophet':
                prophet_df = train_data.reset_index().rename(columns={'date': 'ds', 'Close': 'y'})
                model.fit(prophet_df)
                future = test_data.reset_index().rename(columns={'date': 'ds'})
                forecast = model.predict(future)['yhat']
                forecast = pd.Series(forecast.values, index=test_data.index)
                
            elif model_name == 'XGBoost':
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
                forecast = pd.Series(forecast, index=test_data.index)
                
            else:  # Linear Regression and Random Forest
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
                forecast = pd.Series(forecast, index=test_data.index)
            
            # Handle NaN values
            if forecast.isnull().any():
                print(f"Warning: {model_name} forecast contains NaN values. Filling with previous value.")
                forecast = forecast.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate metrics
            mse = mean_squared_error(test_data['Close'], forecast)
            mae = mean_absolute_error(test_data['Close'], forecast)
            r2 = r2_score(test_data['Close'], forecast)
            
            results[model_name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            }
            
            forecasts[model_name] = forecast
            
            print(f"{model_name} Metrics:")
            print(f"MSE: {mse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R2: {r2:.2f}")
        
        # Plot results
        plt.figure(figsize=(15, 8))
        plt.plot(test_data.index, test_data['Close'], label='Actual', color='black')
        
        for model_name, forecast in forecasts.items():
            plt.plot(test_data.index, forecast, label=f'{model_name} Forecast', alpha=0.7)
        
        plt.title('Tesla Stock Price: Actual vs Model Forecasts')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('multi_model_forecast.png')
        plt.show()
        plt.close()
        
        # At the end of model_data, save results and forecasts to self
        self.results = results
        self.forecasts = forecasts
        return results, forecasts

    def interpret_results(self):
        """
        Generate business insights and interpretations based on multiple models
        """
        if self.data is None or not self.forecasts:
            raise ValueError("Please run model_data() first")
            
        # Calculate key statistics
        current_price = self.data['Close'].iloc[-1]
        
        # Only use models with valid forecasts (no NaNs after alignment)
        valid_forecasts = {}
        test_data = self.data[['date', 'Close']].copy()
        test_data.set_index('date', inplace=True)
        test_data = test_data.iloc[int(len(test_data) * 0.8):]
        for model_name, forecast in self.forecasts.items():
            aligned = pd.concat([test_data['Close'], forecast], axis=1, keys=['actual', 'forecast']).dropna()
            if not aligned.empty and not aligned['forecast'].isnull().any():
                valid_forecasts[model_name] = aligned['forecast']
        if not valid_forecasts:
            print("No valid model forecasts available for interpretation.")
            return {}
        
        # Calculate average forecast from all valid models
        avg_forecast = np.mean([forecast.iloc[-1] for forecast in valid_forecasts.values()])
        price_change = ((avg_forecast - current_price) / current_price) * 100
        
        # Calculate volatility
        volatility = self.data['Daily_Return'].std()
        
        # Calculate trend strength using ADX
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        
        # Find best performing model
        best_model = min(valid_forecasts.items(), 
            key=lambda x: mean_squared_error(test_data['Close'].loc[x[1].index], x[1]))[0]
        
        # Generate insights
        insights = {
            'Current Price': f"${current_price:.2f}",
            'Average Forecasted Price': f"${avg_forecast:.2f}",
            'Expected Change': f"{price_change:.2f}%",
            'Volatility': f"{volatility:.2f}%",
            'Trend Strength': 'Strong' if atr.iloc[-1] > atr.mean() else 'Weak',
            'Trading Volume Trend': 'Increasing' if self.data['Volume'].iloc[-5:].mean() > self.data['Volume'].iloc[-10:-5].mean() else 'Decreasing',
            'Best Performing Model': best_model
        }
        
        # Print insights
        print("\nBusiness Insights:")
        for key, value in insights.items():
            print(f"{key}: {value}")
            
        # Generate recommendations
        print("\nRecommendations:")
        if price_change > 5:
            print("Strong buy signal based on positive price forecast")
        elif price_change < -5:
            print("Consider selling or waiting for better entry point")
        else:
            print("Hold position, monitor market conditions")
            
        if volatility > 3:
            print("High volatility detected - consider risk management strategies")
            
        return insights

    def generate_report(self):
        """
        Generate a comprehensive analysis report
        """
        if self.data is None or not self.forecasts:
            raise ValueError("Please run the analysis first")
        
        # Only use models with valid forecasts (no NaNs after alignment)
        valid_forecasts = {}
        test_data = self.data[['date', 'Close']].copy()
        test_data.set_index('date', inplace=True)
        test_data = test_data.iloc[int(len(test_data) * 0.8):]
        for model_name, forecast in self.forecasts.items():
            aligned = pd.concat([test_data['Close'], forecast], axis=1, keys=['actual', 'forecast']).dropna()
            if not aligned.empty and not aligned['forecast'].isnull().any():
                valid_forecasts[model_name] = aligned['forecast']
        if not valid_forecasts:
            print("No valid model forecasts available for report.")
            return {}
        
        # Calculate model performance metrics
        model_metrics = {}
        for model_name, forecast in valid_forecasts.items():
            aligned = pd.concat([test_data['Close'], forecast], axis=1, keys=['actual', 'forecast']).dropna()
            model_metrics[model_name] = {
                'MSE': mean_squared_error(aligned['actual'], aligned['forecast']),
                'MAE': mean_absolute_error(aligned['actual'], aligned['forecast']),
                'R2': r2_score(aligned['actual'], aligned['forecast'])
            }
        
        # Find best performing model
        best_model = min(valid_forecasts.items(), 
            key=lambda x: mean_squared_error(test_data['Close'].loc[x[1].index], x[1]))[0]
        
        report = {
            'Project Background': {
                'Organization': 'Tesla Inc.',
                'Target Users': 'Investors and Financial Analysts',
                'Benefits': 'Stock price forecasting and investment decision support'
            },
            'Project Objectives': [
                'Analyze Tesla stock price trends and patterns',
                'Develop multiple predictive models for price forecasting',
                'Compare model performance and generate actionable insights'
            ],
            'Models Used': [
                'Linear Regression - Trend analysis',
                'Random Forest - Complex pattern recognition',
                'Prophet - Seasonality and trend decomposition',
                'XGBoost - Gradient boosting for complex patterns',
                'KNN Regression - Non-parametric regression'
            ],
            'Model Performance': model_metrics,
            'Key Findings': {
                'Price Trends': f"Current price: ${self.data['Close'].iloc[-1]:.2f}",
                'Volatility': f"{self.data['Daily_Return'].std():.2f}%",
                'Trading Volume': f"Average: {self.data['Volume'].mean():.0f}",
                'Best Model': best_model
            },
            'Recommendations': [
                'Monitor market conditions regularly',
                'Consider technical indicators for entry/exit points',
                'Implement risk management strategies',
                'Use ensemble approach combining multiple models for better predictions'
            ]
        }
        
        # Save report to file
        with open('analysis_report.txt', 'w') as f:
            f.write("TESLA STOCK ANALYSIS REPORT\n")
            f.write("=========================\n\n")
            
            for section, content in report.items():
                f.write(f"{section}\n")
                f.write("-" * len(section) + "\n")
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, dict):
                            f.write(f"\n{key}:\n")
                            for subkey, subvalue in value.items():
                                f.write(f"  {subkey}: {subvalue:.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                elif isinstance(content, list):
                    for item in content:
                        f.write(f"- {item}\n")
                f.write("\n")
                
        print("\nAnalysis report has been generated as 'analysis_report.txt'")
        return report

def main():
    # Create instance of TeslaStockAnalysis
    analysis = TeslaStockAnalysis()
    
    # Obtain data
    print("Loading Tesla stock data from CSV file...")
    analysis.obtain_data()
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    analysis.calculate_technical_indicators()
    
    # Clean data
    print("\nCleaning data...")
    analysis.clean_data()
    
    # Model data
    print("\nPerforming time series modeling...")
    analysis.model_data()
    
    # Interpret results
    print("\nGenerating business insights...")
    analysis.interpret_results()
    
    # Generate report
    print("\nGenerating analysis report...")
    analysis.generate_report()

if __name__ == "__main__":
    main() 