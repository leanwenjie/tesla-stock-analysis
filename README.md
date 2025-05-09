# Tesla Stock Analysis

This project analyzes Tesla's stock data to provide valuable insights for investors and financial analysts. The analysis includes technical indicators, price trends, and market insights.

## Features

- Historical stock data analysis
- Technical indicators calculation
- Data visualization
- Statistical analysis
- Market trend analysis

## Technical Indicators Included

- Moving Averages (SMA 20, 50, 200)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Daily and Cumulative Returns

## Setup

1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the analysis:
```bash
python tesla_analysis.py
```

## Output

The analysis generates several visualizations in the `analysis_graphs` directory:
- Technical analysis charts
- Daily returns distribution
- Cumulative returns
- Correlation heatmap

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- openpyxl

## License

MIT License 