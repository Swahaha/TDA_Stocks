import pandas as pd
import numpy as np
from market_tda import MarketTDA
from market_analysis import MarketAnalyzer
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

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
        'plots/combined_analysis',
        'plots/approach_comparisons'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        print(f"Created directory: {os.path.join(base_dir, subdir)}")
    
    return base_dir

def analyze_market_periods(data, tda, analyzer, periods):
    """Analyze specific market periods"""
    tda_results = {}
    analysis_results = {}
    comparison_results = {}
    
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
            
            # Perform approach comparison
            print("\nComparing analysis approaches...")
            comparison_results[name] = tda.compare_approaches(tda_results[name])
            
            print(f"Analysis complete for {name}")
            
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            continue
            
    return tda_results, analysis_results, comparison_results

def debug_data_structure(data, prefix=""):
    """Helper function to inspect nested data structures"""
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}: {type(value)}")
            if isinstance(value, (dict, list, np.ndarray)):
                debug_data_structure(value, prefix + "  ")
    elif isinstance(data, list):
        print(f"{prefix}List of length {len(data)}")
        if data:
            print(f"{prefix}First element type: {type(data[0])}")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}Array shape: {data.shape}, dtype: {data.dtype}")

def save_period_analysis(period_name, tda_result, analysis_result, comparison_result, output_dir):
    """Save analysis results for a specific period"""
    period_dir = os.path.join(output_dir, 'analysis', period_name)
    os.makedirs(period_dir, exist_ok=True)
    
    try:
        # Save TDA results - cloud
        np.save(
            os.path.join(period_dir, 'combined_cloud.npy'),
            tda_result['combined']['cloud']
        )
        
        # Save persistence diagrams separately for each dimension
        diagrams_dir = os.path.join(period_dir, 'persistence_diagrams')
        os.makedirs(diagrams_dir, exist_ok=True)
        for dim, diagram in enumerate(tda_result['combined']['diagrams']):
            np.save(
                os.path.join(diagrams_dir, f'dim_{dim}.npy'),
                diagram
            )
        
        print(f"Saved TDA results for {period_name}")
        
        # Save crisis indicators
        crisis_dict = {}
        for key, value in analysis_result['combined']['crisis_indicators'].items():
            if isinstance(value, (np.floating, np.integer)):
                crisis_dict[key] = float(value)
            elif isinstance(value, (float, int)):
                crisis_dict[key] = value
                
        pd.DataFrame([crisis_dict]).to_csv(
            os.path.join(period_dir, 'crisis_indicators.csv')
        )
        
        # Save regime analysis
        regime_data = []
        for regime_id, regime_info in analysis_result['combined']['regimes'].items():
            regime_dict = {'regime_id': regime_id}
            for k, v in regime_info.items():
                if isinstance(v, (np.floating, np.integer)):
                    regime_dict[k] = float(v)
                elif isinstance(v, (float, int)):
                    regime_dict[k] = v
            regime_data.append(regime_dict)
            
        pd.DataFrame(regime_data).to_csv(
            os.path.join(period_dir, 'regime_analysis.csv'),
            index=False
        )
        
        # Save stability measures
        stability = analysis_result['combined']['stability']
        stability_dict = {}
        
        # Density stats
        for k, v in stability['density_stats'].items():
            stability_dict[f'density_{k}'] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            
        # Network measures
        for k, v in stability['network_measures'].items():
            stability_dict[f'network_{k}'] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            
        # Volatility measures
        for k, v in stability['volatility_measures'].items():
            stability_dict[f'volatility_{k}'] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            
        pd.DataFrame([stability_dict]).to_csv(
            os.path.join(period_dir, 'stability_measures.csv')
        )
        
        # Save arrays separately
        arrays_dir = os.path.join(period_dir, 'arrays')
        os.makedirs(arrays_dir, exist_ok=True)
        
        np.save(os.path.join(arrays_dir, 'regime_labels.npy'), 
                analysis_result['combined']['regime_labels'])
        np.save(os.path.join(arrays_dir, 'states.npy'), 
                analysis_result['combined']['states'])
        np.save(os.path.join(arrays_dir, 'transitions.npy'), 
                analysis_result['combined']['transitions'])
        np.save(os.path.join(arrays_dir, 'local_variance.npy'), 
                analysis_result['combined']['local_variance'])
        np.save(os.path.join(arrays_dir, 'local_density.npy'), 
                analysis_result['combined']['stability']['local_density'])
        np.save(os.path.join(arrays_dir, 'distance_matrix.npy'), 
                analysis_result['combined']['stability']['distance_matrix'])
        
        # Save comparison results
        comparison_dir = os.path.join(period_dir, 'approach_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Save feature counts
        pd.DataFrame(comparison_result['feature_counts']).to_csv(
            os.path.join(comparison_dir, 'feature_counts.csv')
        )
        
        # Save persistence entropy
        pd.DataFrame({
            'combined': [comparison_result['persistence_entropy']['combined']],
            **{k: [v] for k, v in comparison_result['persistence_entropy']['separate'].items()}
        }).to_csv(os.path.join(comparison_dir, 'persistence_entropy.csv'))
        
        # Save correlations
        pd.Series(comparison_result['embedding_correlations']).to_csv(
            os.path.join(comparison_dir, 'embedding_correlations.csv')
        )
        
        # Save structure preservation
        pd.DataFrame({
            'combined': [comparison_result['structure_preservation']['combined']],
            **{k: [v] for k, v in comparison_result['structure_preservation']['separate'].items()}
        }).to_csv(os.path.join(comparison_dir, 'structure_preservation.csv'))
        
        print(f"Saved comparison results for {period_name}")
        
        # Save period summary
        summary = generate_period_summary(period_name, analysis_result['combined'])
        with open(os.path.join(period_dir, 'summary.txt'), 'w') as f:
            f.write(summary)
        
        print(f"Successfully saved all results for {period_name}")
        
    except Exception as e:
        print(f"Error saving results for {period_name}: {str(e)}")
        traceback.print_exc()

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
        data = load_data('combined_indices.csv')
        
        # Initialize analyzers
        window_size = 20
        print(f"\nInitializing analysis with window size {window_size}")
        tda = MarketTDA(window_size=window_size)
        analyzer = MarketAnalyzer(window_size=window_size)
        
        periods = {
            'Full_Period': (data.index.min(), data.index.max()),
            'Dot_Com_Crash': ('2001-03-01', '2002-10-31'),  # Tech bubble burst & post-9/11 decline
            'Financial_Crisis': ('2008-09-01', '2009-03-31'),  # Great Financial Crisis peak period
            'Covid_Crash': ('2020-02-19', '2020-03-23'),  # Sharp COVID-19 market decline
            'Post_Covid_Recovery': ('2020-03-24', '2020-08-31'),  # Rapid recovery phase
            'Inflation_Crisis': ('2022-01-01', '2022-10-31'),  # High inflation & rate hikes period
            'SVB_Crisis': ('2023-03-01', '2023-03-31'),  # Silicon Valley Bank collapse
            'Recent_Period': ('2024-01-01', data.index.max()),
            'Pre_Financial_Crisis': ('2007-01-01', '2008-08-31'),
            'Post_Financial_Crisis': ('2009-04-01', '2010-12-31'),
            'Pre_Covid': ('2019-01-01', '2019-12-31'),
            'Post_Covid': ('2021-01-01', '2021-12-31')
        }
        
        # Perform analysis
        print("\nStarting analysis...")
        tda_results, analysis_results, comparison_results = analyze_market_periods(
            data, tda, analyzer, periods
        )
        
        # Generate and save reports
        print("\nGenerating reports...")
        reports_dir = os.path.join(output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save individual period reports
        all_reports = []
        for period_name, period_analysis in analysis_results.items():
            try:
                report_text = analyzer.generate_period_summary(
                    period_name, 
                    period_analysis['combined']
                )
                
                # Save individual period report
                period_report_path = os.path.join(reports_dir, f'{period_name}_report.txt')
                with open(period_report_path, 'w') as f:
                    f.write(report_text)
                print(f"Saved report for {period_name}")
                
                all_reports.append(report_text)
                all_reports.append("\n" + "="*80 + "\n")  # Separator between periods
                
            except Exception as e:
                print(f"Error saving report for {period_name}: {str(e)}")
        
        # Save combined report with all periods
        try:
            combined_report_path = os.path.join(reports_dir, 'combined_analysis_report.txt')
            with open(combined_report_path, 'w') as f:
                f.write("COMPREHENSIVE MARKET ANALYSIS REPORT\n")
                f.write("="*40 + "\n\n")
                f.write("\n".join(all_reports))
            print("Saved combined analysis report")
        except Exception as e:
            print(f"Error saving combined report: {str(e)}")
        
        # Save individual period results
        print("\nSaving individual period results...")
        for period_name in tda_results.keys():
            save_period_analysis(
                period_name,
                tda_results[period_name],
                analysis_results[period_name],
                comparison_results[period_name],
                output_dir
            )
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        print("Creating TDA visualizations...")
        for period_name, period_results in tda_results.items():
            try:
                print(f"  Processing {period_name}...")
                save_path = os.path.join(output_dir, 'plots', 'individual_periods', 
                                       f'{period_name}_tda.png')
                tda.plot_analysis(period_results, save_path)
                print(f"  Completed {period_name}")
            except Exception as e:
                print(f"  Error generating plot for {period_name}: {str(e)}")
                plt.close('all')
                continue

        # Approach comparison visualizations
        print("Creating approach comparison visualizations...")
        for period_name, comparison in comparison_results.items():
            try:
                print(f"  Processing {period_name}...")
                save_path = os.path.join(output_dir, 'plots', 'approach_comparisons', 
                                       f'{period_name}_comparison.png')
                tda.plot_approach_comparison(comparison, save_path)
                print(f"  Completed {period_name}")
            except Exception as e:
                print(f"  Error generating comparison plot for {period_name}: {str(e)}")
                plt.close('all')
                continue

        # Generate analysis summary plots
        print("\nCreating analysis summary plots...")
        try:
            summary_path = os.path.join(output_dir, 'plots', 'combined_analysis', 
                                      'analysis_summary.png')
            analyzer.plot_analysis_results(analysis_results, summary_path)
            print("Completed analysis summary plots")
            plt.close('all')
        except Exception as e:
            print(f"Error generating analysis summary plots: {str(e)}")
            plt.close('all')

        print("\nAnalysis complete! Results saved to: {output_dir}")
        print("\nOutput directory structure:")
        print("1. /reports - Individual and combined analysis reports")
        print("2. /plots - TDA and analysis visualizations")
        print("3. /analysis - Numerical results and metrics")
        print("4. /plots/approach_comparisons - Comparison visualizations")
        
        return tda_results, analysis_results, comparison_results
        
    except Exception as e:
        print(f"\nCritical error during analysis: {e}")
        plt.close('all')
        raise

if __name__ == "__main__":
    try:
        tda_results, analysis_results, comparison_results = main()
    except Exception as e:
        print(f"\nProgram terminated due to error: {e}")
        raise