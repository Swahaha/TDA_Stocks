import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from market_tda import MarketTDA  # assuming above code is saved as market_tda.py

def load_data():
    """
    Load market data from CSV files
    """
    try:
        # Update these paths to match your CSV file locations
        sp500 = pd.read_csv('sp500.csv')  # adjust path as needed
        russell = pd.read_csv('russell.csv')  # adjust path as needed
        nasdaq = pd.read_csv('nasdaq.csv')  # adjust path as needed
        
        # Convert dates to datetime
        for df in [sp500, russell, nasdaq]:
            df['Date'] = pd.to_datetime(df['Date'])
        
        print("Data loaded successfully!")
        print(f"Date range: {sp500['Date'].min()} to {sp500['Date'].max()}")
        
        return sp500, russell, nasdaq
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def main():
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    print("Starting market analysis...")
    
    # 1. Load data
    print("\n1. Loading market data...")
    sp500, russell, nasdaq = load_data()
    
    # 2. Initialize TDA analyzer
    print("\n2. Initializing TDA analyzer...")
    tda = MarketTDA()
    
    # 3. Prepare data
    print("\n3. Preparing data...")
    returns, rolling_data = tda.prepare_data(sp500, russell, nasdaq)
    print(f"Processed {len(returns)} days of returns data")
    
    # 4. Perform TDA analysis
    print("\n4. Performing TDA analysis...")
    diagrams, dist_matrix, mds_coords = tda.analyze_market_structure(returns, rolling_data)
    
    # 5. Create visualizations
    print("\n5. Creating visualizations...")
    tda.plot_analysis(diagrams, mds_coords, returns, save_path='output/market_analysis.png')
    
    # 6. Detect market regimes
    print("\n6. Detecting market regimes...")
    clusters, regime_analysis = detect_market_regimes(tda, mds_coords, returns)
    
    # Print regime analysis
    print("\nMarket Regime Analysis:")
    print(regime_analysis)
    
    # 7. Save results
    print("\n7. Saving results...")
    results = {
        'returns': returns,
        'rolling_data': rolling_data,
        'mds_coords': mds_coords,
        'clusters': clusters,
        'regime_analysis': regime_analysis
    }
    
    # Save important results to files
    returns.to_csv('output/processed_returns.csv')
    rolling_data.to_csv('output/rolling_features.csv')
    regime_analysis.to_csv('output/regime_analysis.csv')
    np.save('output/persistence_diagrams.npy', diagrams)
    
    print("\nAnalysis complete! Results saved to 'output' directory.")
    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise