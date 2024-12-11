import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from advancedTDA import AdvancedMarketTDA  # The class we created earlier

def load_and_prepare_data(file_path):
    """
    Load and prepare market data for analysis
    """
    try:
        # Load data
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Ensure we have required columns
        required_columns = ['SP500', 'Russell', 'NASDAQ']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        return data[required_columns]
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def analyze_time_periods(data, tda_analyzer):
    """
    Analyze different time periods of interest
    """
    # Define periods of interest
    periods = {
        'Full_Period': (data.index.min(), data.index.max()),
        'Financial_Crisis': ('2008-01-01', '2009-12-31'),
        'Dot_Com_Bubble': ('2000-01-01', '2002-12-31'),
        'Covid_Crisis': ('2020-01-01', '2020-12-31'),
        'Recent_Period': ('2021-01-01', data.index.max())
    }
    
    results = {}
    for period_name, (start, end) in periods.items():
        print(f"\nAnalyzing {period_name}...")
        period_data = data[start:end]
        results[period_name] = tda_analyzer.perform_advanced_analysis(period_data)
        
    return results

def save_results(results, output_dir):
    """
    Save analysis results to files
    """
    for period_name, period_results in results.items():
        # Create period directory
        period_dir = os.path.join(output_dir, period_name)
        os.makedirs(period_dir, exist_ok=True)
        
        # Save numerical results
        pd.Series({
            'persistence_entropy': period_results['persistence_entropy'],
            'n_market_states': len(np.unique(period_results['market_states'])),
            'n_critical_transitions': len(period_results['critical_transitions'][0]),
            'mean_stability': period_results['stability_measures']['density_mean']
        }).to_csv(os.path.join(period_dir, 'summary_metrics.csv'))
        
        # Save market states
        np.save(os.path.join(period_dir, 'market_states.npy'), 
                period_results['market_states'])
        
        # Save critical transitions
        np.save(os.path.join(period_dir, 'critical_transitions.npy'), 
                period_results['critical_transitions'][0])

def compare_periods(results):
    """
    Compare analysis results across different periods
    """
    comparison = pd.DataFrame()
    
    for period_name, period_results in results.items():
        comparison.loc[period_name, 'Persistence_Entropy'] = period_results['persistence_entropy']
        comparison.loc[period_name, 'N_Market_States'] = len(np.unique(period_results['market_states']))
        comparison.loc[period_name, 'N_Critical_Transitions'] = len(period_results['critical_transitions'][0])
        comparison.loc[period_name, 'Mean_Stability'] = period_results['stability_measures']['density_mean']
    
    return comparison

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tda_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting TDA Analysis...")
    
    # Load data
    print("\nLoading market data...")
    data = load_and_prepare_data('combined dat')
    print(f"Loaded data from {data.index.min()} to {data.index.max()}")
    
    # Initialize TDA analyzer
    window_size = 20  # Adjust as needed
    tda_analyzer = AdvancedMarketTDA(window_size=window_size)
    
    # Perform analysis for different periods
    print("\nPerforming analysis for different time periods...")
    results = analyze_time_periods(data, tda_analyzer)
    
    # Compare periods
    print("\nComparing periods...")
    comparison = compare_periods(results)
    comparison.to_csv(os.path.join(output_dir, 'period_comparison.csv'))
    print("\nPeriod comparison:")
    print(comparison)
    
    # Save results
    print("\nSaving results...")
    save_results(results, output_dir)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    for period_name, period_results in results.items():
        save_path = os.path.join(output_dir, f'{period_name}_analysis.png')
        tda_analyzer.plot_advanced_analysis(period_results, 
                                         data.loc[period_results['market_states'].index], 
                                         save_path)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    return results, comparison

if __name__ == "__main__":
    try:
        results, comparison = main()
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise