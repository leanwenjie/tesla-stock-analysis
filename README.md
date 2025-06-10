# Tesla Stock Price Analysis and Forecasting

This application provides an interactive dashboard for analyzing Tesla stock data and making price predictions using machine learning models.

## Prerequisites

Before running the application, make sure you have Python installed on your system. Then install the required packages:

```bash
pip install -r requirements.txt
```

## Required Files

1. `tesla.py` - The main application file
2. `Tesla_Stock_Updated_V2.csv` - The dataset file
3. `requirements.txt` - List of required Python packages

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python -m streamlit run tesla.py
   ```

3. **Access the Dashboard**
   - The application will automatically open in your default web browser
   - If it doesn't open automatically, you can access it at http://localhost:8501

## Features

- **Stock Price Analysis**
  - Historical price trends
  - Moving averages
  - Daily returns distribution
  - Trading volume analysis
  - Price range visualization

- **Price Prediction**
  - Multiple machine learning models:
    - Linear Regression
    - Random Forest
    - Gradient Boosting
  - Model performance metrics (RMSE, MAE, R²)
  - Future price forecasting

- **Interactive Elements**
  - Model selection
  - Forecast period adjustment
  - Interactive plots
  - Data tables

## Troubleshooting

If you encounter any issues:

1. **'streamlit' not recognized**
   - Use `python -m streamlit run tesla.py` instead of `streamlit run tesla.py`

2. **Missing dependencies**
   - Run `pip install -r requirements.txt` again

3. **File not found errors**
   - Make sure `Tesla_Stock_Updated_V2.csv` is in the same directory as `tesla.py`

## Data Source

The application uses Tesla stock data obatained from trusted platform , Kaggle. The data includes:
- Date
- Open price
- High price
- Low price
- Close price
- Volume

## Note

This is a demonstration project and should not be used for actual investment decisions. Always do your own research before making investment choices. 