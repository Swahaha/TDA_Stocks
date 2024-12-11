import pandas as pd
import numpy as np
from market_tda import MarketTDA
from market_analysis import MarketAnalyzer
import os
from datetime import datetime

def load_data(file_path):
    """Load and preprocess market data"""
    try:
        print(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        required_columns = ['SP500', 'Russell', 'NASDAQ']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        print(f"Data loaded successfully!")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Number of trading days: {len(data)}")
        
        return data[required_columns]
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_output_directory():
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"market_analysis_{timestamp}"
    
    # Create subdirectories
    subdirs = ['tda', 'analysis', 'plots', 'reports']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    return base_dir

def analyze_market_periods(data, tda, analyzer, periods):
    """Analyze specific market periods"""
    tda_results = {}
    analysis_results = {}
    
    for name, (start, end) in periods.items():
        try:
            print(f"\n{'='*50}")
            print(f"Analyzing {name} period ({start} to {end})")
            period_data = data[start:end]
            
            if len(period_data) < tda.window_size:
                print(f"Warning: Period too short for window size {tda.window_size}")
                continue
                
            # Perform TDA analysis
            print("\nPerforming TDA analysis...")
            tda_results[name] = tda.analyze_market_structure(period_data)
            
            # Perform market analysis
            print("\nPerforming advanced market analysis...")
            analysis_results[name] = analyzer.analyze_period_results(
                tda_results[name], name
            )
            
            print(f"Analysis complete for {name}")
            
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            continue
            
    return tda_results, analysis_results

def save_results(tda_results, analysis_results, output_dir):
    """Save analysis results"""
    for period_name in tda_results.keys():
        # Create period directory
        period_dir = os.path.join(output_dir, 'analysis', period_name)
        os.makedirs(period_dir, exist_ok=True)
        
        try:
            # Save TDA results
            np.save(
                os.path.join(period_dir, 'combined_cloud.npy'),
                tda_results[period_name]['combined']['cloud']
            )
            np.save(
                os.path.join(period_dir, 'persistence_diagrams.npy'),
                tda_results[period_name]['combined']['diagrams']
            )
            
            # Save analysis results
            period_analysis = analysis_results[period_name]
            
            # Save crisis indicators
            pd.DataFrame([period_analysis['combined']['crisis_indicators']]).to_csv(
                os.path.join(period_dir, 'crisis_indicators.csv')
            )
            
            # Save regime analysis
            pd.DataFrame.from_dict(period_analysis['combined']['regimes'], 
                                 orient='index').to_csv(
                os.path.join(period_dir, 'regime_analysis.csv')
            )
            
            # Save stability measures
            pd.DataFrame([period_analysis['combined']['stability']]).to_csv(
                os.path.join(period_dir, 'stability_measures.csv')
            )
            
        except Exception as e:
            print(f"Error saving results for {period_name}: {e}")
            continue

def main():
    try:
        # Create output directory
        output_dir = create_output_directory()
        print(f"\nOutput will be saved to: {output_dir}")
        
        # Load data
        data = load_data('combined dat')
        
        # Initialize analyzers
        window_size = 20
        print(f"\nInitializing analysis with window size {window_size}")
        tda = MarketTDA(window_size=window_size)
        analyzer = MarketAnalyzer(window_size=window_size)
        
        # Define analysis periods
        periods = {
            'Full_Period': (data.index.min(), data.index.max()),
            'Financial_Crisis': ('2008-01-01', '2009-12-31'),
            'Covid_Crisis': ('2020-01-01', '2020-12-31'),
            'Recent_Period': ('2021-01-01', data.index.max()),
            'Pre_Covid': ('2019-01-01', '2019-12-31'),
            'Post_Covid': ('2021-01-01', '2021-12-31')
        }
        
        # Perform analysis
        print("\nStarting analysis...")
        tda_results, analysis_results = analyze_market_periods(
            data, tda, analyzer, periods
        )
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # TDA visualizations
        for period_name, period_results in tda_results.items():
            save_path = os.path.join(output_dir, 'tda', f'{period_name}_tda.png')
            tda.plot_analysis(period_results, save_path)
        
        # Analysis visualizations
        analyzer.plot_analysis_results(
            analysis_results,
            save_path=os.path.join(output_dir, 'plots', 'analysis_summary.png')
        )
        
        # Generate report
        print("\nGenerating analysis report...")
        report = analyzer.generate_report(analysis_results)
        with open(os.path.join(output_dir, 'reports', 'analysis_report.txt'), 'w') as f:
            f.write(report)
        
        # Save results
        print("\nSaving analysis results...")
        save_results(tda_results, analysis_results, output_dir)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print("\nOutput directory contains:")
        print("1. TDA visualizations")
        print("2. Market analysis plots")
        print("3. Comprehensive report")
        print("4. Numerical results")
        
        return tda_results, analysis_results
        
    except Exception as e:
        print(f"\nCritical error during analysis: {e}")
        raise

if __name__ == "__main__":
    try:
        tda_results, analysis_results = main()
    except Exception as e:
        print(f"\nProgram terminated due to error: {e}")
        raise