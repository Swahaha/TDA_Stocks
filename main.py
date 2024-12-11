import pandas as pd
import numpy as np
from market_tda import MarketTDA
from market_analysis import MarketAnalyzer
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load and preprocess market data"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        print(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        required_columns = ['SP500', 'Russell', 'NASDAQ']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Data loaded successfully!")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Number of trading days: {len(data)}")
        print(f"Columns in data: {list(data.columns)}")
        
        return data[required_columns]
    
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty")
        raise
    except pd.errors.ParserError:
        print(f"Error: Unable to parse {file_path}. Make sure it's a valid CSV file")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_output_directory():
    """Create timestamped output directory with all necessary subdirectories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"market_analysis_{timestamp}"
    
    # Create main subdirectories
    subdirs = [
        'tda',
        'analysis',
        'plots',
        'reports',
        'plots/individual_periods',
        'plots/combined_analysis'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        print(f"Created directory: {os.path.join(base_dir, subdir)}")
    
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

def save_period_analysis(period_name, tda_result, analysis_result, output_dir):
    """Save analysis results for a specific period with proper data structure handling"""
    # Create period directory
    period_dir = os.path.join(output_dir, 'analysis', period_name)
    os.makedirs(period_dir, exist_ok=True)
    
    try:
        # Save TDA results
        np.save(
            os.path.join(period_dir, 'combined_cloud.npy'),
            tda_result['combined']['cloud']
        )
        np.save(
            os.path.join(period_dir, 'persistence_diagrams.npy'),
            tda_result['combined']['diagrams']
        )
        print(f"Saved TDA results for {period_name}")
        
        # Save analysis results
        combined_analysis = analysis_result['combined']
        
        # Save crisis indicators as flattened dict
        crisis_dict = {}
        for key, value in combined_analysis['crisis_indicators'].items():
            if isinstance(value, (int, float, str)):
                crisis_dict[key] = value
            elif isinstance(value, np.ndarray):
                crisis_dict[key] = value.item() if value.size == 1 else value.tolist()
        pd.DataFrame([crisis_dict]).to_csv(
            os.path.join(period_dir, 'crisis_indicators.csv')
        )
        
        # Save regime analysis
        regime_data = []
        for regime_id, regime_info in combined_analysis['regimes'].items():
            regime_dict = {'regime_id': regime_id}
            regime_dict.update({k: v for k, v in regime_info.items() 
                              if isinstance(v, (int, float, str))})
            regime_data.append(regime_dict)
        
        pd.DataFrame(regime_data).to_csv(
            os.path.join(period_dir, 'regime_analysis.csv'), index=False
        )
        
        # Save stability measures
        stability_data = {
            'density_stats': combined_analysis['stability']['density_stats'],
            'network_measures': combined_analysis['stability']['network_measures'],
            'volatility_measures': combined_analysis['stability']['volatility_measures']
        }
        
        # Flatten nested dictionaries
        flat_stability = {}
        for category, metrics in stability_data.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    flat_stability[f"{category}_{metric}"] = value
                elif isinstance(value, np.ndarray):
                    flat_stability[f"{category}_{metric}"] = value.item() if value.size == 1 else value.tolist()
        
        pd.DataFrame([flat_stability]).to_csv(
            os.path.join(period_dir, 'stability_measures.csv')
        )
        
        print(f"Saved analysis results for {period_name}")
        
        # Save period summary
        summary = generate_period_summary(period_name, combined_analysis)
        with open(os.path.join(period_dir, 'period_summary.txt'), 'w') as f:
            f.write(summary)
            
        print(f"Saved period summary for {period_name}")
        
    except Exception as e:
        print(f"Error saving results for {period_name}: {e}")
        print("Detailed error info:", str(e))

def generate_period_summary(period_name, analysis):
    """Generate summary text for a period with error handling"""
    try:
        summary = [
            f"Analysis Summary for {period_name}",
            "=" * 50,
            "",
            "Crisis Indicators:"
        ]
        
        # Add crisis indicators with safe access
        crisis = analysis.get('crisis_indicators', {})
        for key, value in crisis.items():
            if isinstance(value, (int, float)):
                summary.append(f"- {key}: {value:.3f}")
            elif isinstance(value, np.ndarray):
                summary.append(f"- {key}: {value.item():.3f}" if value.size == 1 
                             else f"- {key}: {value.tolist()}")
        
        summary.extend([
            "",
            "Market Regimes:",
            f"- Number of Regimes: {len(analysis.get('regimes', {}))}"
        ])
        
        # Add stability measures with safe access
        stability = analysis.get('stability', {})
        density_stats = stability.get('density_stats', {})
        network_measures = stability.get('network_measures', {})
        
        summary.extend([
            "",
            "Stability Measures:",
            f"- Mean Local Density: {density_stats.get('mean', 0):.3f}",
            f"- Network Clustering: {network_measures.get('clustering_coef', 0):.3f}"
        ])
        
        return "\n".join(summary)
        
    except Exception as e:
        print(f"Error generating period summary: {e}")
        return f"Error generating summary for {period_name}: {str(e)}"

def generate_comparative_visualizations(analysis_results, output_dir):
    """Generate comparative visualizations across periods"""
    try:
        # Create comparison metrics
        metrics = {}
        for period, results in analysis_results.items():
            metrics[period] = {
                'persistence_entropy': results['combined']['crisis_indicators']['persistence_entropy'],
                'volatility': results['combined']['crisis_indicators']['volatility'],
                'n_regimes': len(results['combined']['regimes']),
                'stability': results['combined']['stability']['density_stats']['mean']
            }
        
        metrics_df = pd.DataFrame(metrics).T
        
        # Create comparison plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Persistence Entropy vs Volatility
        plt.subplot(221)
        plt.scatter(metrics_df['persistence_entropy'], metrics_df['volatility'])
        for idx, row in metrics_df.iterrows():
            plt.annotate(idx, (row['persistence_entropy'], row['volatility']))
        plt.xlabel('Persistence Entropy')
        plt.ylabel('Volatility')
        plt.title('Market Structure vs Volatility')
        
        # Plot 2: Number of Regimes
        plt.subplot(222)
        metrics_df['n_regimes'].plot(kind='bar')
        plt.title('Number of Market Regimes')
        plt.xticks(rotation=45)
        
        # Plot 3: Stability Comparison
        plt.subplot(223)
        metrics_df['stability'].plot(kind='bar')
        plt.title('Market Stability Comparison')
        plt.xticks(rotation=45)
        
        # Save comparison plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'combined_analysis', 'metrics_comparison.png'))
        plt.close()
        
        # Save metrics data
        metrics_df.to_csv(os.path.join(output_dir, 'analysis', 'comparative_metrics.csv'))
        
        print("Generated and saved comparative visualizations")
        
    except Exception as e:
        print(f"Error generating comparative visualizations: {e}")

def main():
    try:
        # Create output directory
        output_dir = create_output_directory()
        print(f"\nOutput will be saved to: {output_dir}")
        
        # Load data
        data = load_data('combined_data.csv')  # Update with your actual filename
        
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
        
        # Save individual period results
        print("\nSaving individual period results...")
        for period_name in tda_results.keys():
            save_period_analysis(
                period_name,
                tda_results[period_name],
                analysis_results[period_name],
                output_dir
            )
        
        # Generate and save visualizations
        print("\nGenerating visualizations...")
        
        # TDA visualizations
        for period_name, period_results in tda_results.items():
            save_path = os.path.join(output_dir, 'plots', 'individual_periods', 
                                   f'{period_name}_tda.png')
            tda.plot_analysis(period_results, save_path)
            print(f"Saved TDA visualization for {period_name}")
        
        # Analysis visualizations
        analyzer.plot_analysis_results(
            analysis_results,
            save_path=os.path.join(output_dir, 'plots', 'combined_analysis', 
                                 'analysis_summary.png')
        )
        print("Saved analysis summary plots")
        
        # Generate comparative visualizations
        generate_comparative_visualizations(analysis_results, output_dir)
        
        # Generate and save final report
        print("\nGenerating final report...")
        report = analyzer.generate_report(analysis_results)
        report_path = os.path.join(output_dir, 'reports', 'final_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print("Saved final report")
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print("\nOutput directory contains:")
        print("1. TDA visualizations for each period")
        print("2. Market analysis plots and comparative visualizations")
        print("3. Comprehensive report and period summaries")
        print("4. Numerical results and metrics")
        
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