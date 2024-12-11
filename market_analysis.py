import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from persim import plot_diagrams  # Only import what we know exists
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

class MarketAnalyzer:
    def __init__(self, window_size=20):
        self.window_size = window_size
    
    def generate_period_summary(self, period_name, analysis):
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

    def compute_persistence_entropy(self, diagram):
        """
        Custom implementation of persistence entropy
        """
        try:
            # Filter out infinite death times
            finite_mask = np.isfinite(diagram[:, 1])
            if not np.any(finite_mask):
                return 0.0
                
            diagram = diagram[finite_mask]
            
            # Calculate persistence
            persistence = diagram[:, 1] - diagram[:, 0]
            
            # Normalize the persistence values
            total_persistence = np.sum(persistence)
            if total_persistence == 0:
                return 0.0
                
            probabilities = persistence / total_persistence
            
            # Calculate entropy
            return -np.sum(probabilities * np.log(probabilities + 1e-10))  # Added small constant to avoid log(0)
        except Exception as e:
            print(f"Error computing persistence entropy: {e}")
            return 0.0

    def analyze_persistent_features(self, diagrams):
        """
        Analyze persistence diagrams to identify significant market features
        """
        try:
            if not isinstance(diagrams, list) or len(diagrams) < 2:
                raise ValueError("Invalid persistence diagrams format")

            # Basic feature analysis
            components = diagrams[0]
            cycles = diagrams[1]

            if len(components) == 0 and len(cycles) == 0:
                return {
                    'n_components': 0,
                    'n_cycles': 0,
                    'avg_component_lifetime': 0,
                    'avg_cycle_lifetime': 0,
                    'max_component_lifetime': 0,
                    'max_cycle_lifetime': 0,
                    'significant_cycles': np.array([]),
                    'cycle_threshold': 0,
                    'persistence_entropy': 0
                }

            # Calculate lifetimes
            component_lifetimes = components[:, 1] - components[:, 0] if len(components) > 0 else np.array([])
            cycle_lifetimes = cycles[:, 1] - cycles[:, 0] if len(cycles) > 0 else np.array([])
            
            # Advanced cycle analysis
            significant_threshold = np.percentile(cycle_lifetimes, 90) if len(cycle_lifetimes) > 0 else 0
            significant_cycles = cycles[cycle_lifetimes > significant_threshold] if len(cycle_lifetimes) > 0 else np.array([])
            
            return {
                'n_components': len(components),
                'n_cycles': len(cycles),
                'avg_component_lifetime': np.mean(component_lifetimes) if len(component_lifetimes) > 0 else 0,
                'avg_cycle_lifetime': np.mean(cycle_lifetimes) if len(cycle_lifetimes) > 0 else 0,
                'max_component_lifetime': np.max(component_lifetimes) if len(component_lifetimes) > 0 else 0,
                'max_cycle_lifetime': np.max(cycle_lifetimes) if len(cycle_lifetimes) > 0 else 0,
                'significant_cycles': significant_cycles,
                'cycle_threshold': significant_threshold,
                'persistence_entropy': self.compute_persistence_entropy(cycles) if len(cycles) > 0 else 0
            }
        except Exception as e:
            print(f"Error in analyze_persistent_features: {str(e)}")
            # Return default values in case of error
            return {
                'n_components': 0,
                'n_cycles': 0,
                'avg_component_lifetime': 0,
                'avg_cycle_lifetime': 0,
                'max_component_lifetime': 0,
                'max_cycle_lifetime': 0,
                'significant_cycles': np.array([]),
                'cycle_threshold': 0,
                'persistence_entropy': 0
            }
        
    def create_persistence_network(self, point_cloud, epsilon=0.1):
        """
        Create network based on persistence features with robust error handling
        """
        try:
            if len(point_cloud) < 2:
                return nx.Graph(), np.zeros((len(point_cloud), len(point_cloud)))
                
            # Compute pairwise distances
            distances = pdist(point_cloud)
            dist_matrix = squareform(distances)
            
            # Create network
            G = nx.Graph()
            G.add_nodes_from(range(len(point_cloud)))
            
            # Add edges where distance is below threshold
            for i in range(len(point_cloud)):
                for j in range(i+1, len(point_cloud)):
                    if dist_matrix[i,j] < epsilon:
                        G.add_edge(i, j, weight=dist_matrix[i,j])
            
            return G, dist_matrix
            
        except Exception as e:
            print(f"Error in create_persistence_network: {e}")
            return nx.Graph(), np.zeros((len(point_cloud), len(point_cloud)))

    def detect_market_states_and_transitions(self, point_cloud, eps=0.5, min_samples=5, k_neighbors=5):
        """
        Detect market states and critical transitions
        """
        try:
            # Detect states using DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)
            states = clustering.labels_
            
            # Detect transitions using local manifold analysis
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(point_cloud)-1)).fit(point_cloud)
            distances, _ = nbrs.kneighbors(point_cloud)
            local_var = np.var(distances, axis=1)
            
            # Identify critical transitions
            threshold = np.mean(local_var) + 2*np.std(local_var)
            transitions = np.where(local_var > threshold)[0]
            
            return states, transitions, local_var
        except Exception as e:
            print(f"Error in detect_market_states_and_transitions: {e}")
            raise

    def compute_crisis_indicators(self, diagrams, returns):
        """
        Compute comprehensive crisis indicators
        """
        try:
            # Calculate persistence entropy
            persistence_entropy = self.compute_persistence_entropy(diagrams[1])
            
            # Calculate topological complexity
            n_features = len(diagrams[0]) + len(diagrams[1])
            complexity = n_features / self.window_size
            
            # Calculate feature stability
            feature_lifetimes = np.concatenate([
                diagrams[0][:, 1] - diagrams[0][:, 0],
                diagrams[1][:, 1] - diagrams[1][:, 0]
            ])
            feature_stability = np.mean(feature_lifetimes)
            
            # Calculate volatility measures
            returns = np.array(returns).flatten()
            volatility = np.std(returns)
            rolling_vol = pd.Series(returns).rolling(self.window_size).std()
            vol_of_vol = np.std(rolling_vol)
            
            # Calculate drawdown
            rolling_max = pd.Series(returns).expanding().max()
            drawdown = (rolling_max - returns) / rolling_max
            max_drawdown = np.max(drawdown)
            
            return {
                'persistence_entropy': persistence_entropy,
                'topological_complexity': complexity,
                'feature_stability': feature_stability,
                'volatility': volatility,
                'volatility_of_volatility': vol_of_vol,
                'max_drawdown': max_drawdown,
                'avg_feature_lifetime': np.mean(feature_lifetimes),
                'max_feature_lifetime': np.max(feature_lifetimes)
            }
        except Exception as e:
            print(f"Error in compute_crisis_indicators: {e}")
            raise

    def detect_market_regimes(self, mds_coords, returns, eps=0.5, min_samples=5):
        """
        Enhanced market regime detection
        """
        try:
            # Ensure returns is a 1D array with same length as mds_coords
            if isinstance(returns, pd.DataFrame):
                returns = returns.values.flatten()
            elif isinstance(returns, pd.Series):
                returns = returns.values
                
            # Trim or pad returns to match mds_coords length
            if len(returns) > len(mds_coords):
                returns = returns[:len(mds_coords)]
            elif len(returns) < len(mds_coords):
                # Pad with zeros if needed
                returns = np.pad(returns, (0, len(mds_coords) - len(returns)))
                
            # DBSCAN clustering for regime detection
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(mds_coords)
            labels = clustering.labels_
            
            # Verify dimensions match
            assert len(labels) == len(returns), f"Mismatch in dimensions: labels {len(labels)}, returns {len(returns)}"
            
            # Analyze each regime
            regimes = {}
            for label in set(labels):
                mask = labels == label
                regime_returns = returns[mask]
                
                if len(regime_returns) > 0:
                    # Calculate regime statistics
                    regime_stats = {
                        'volatility': np.std(regime_returns),
                        'mean_return': np.mean(regime_returns),
                        'duration': np.sum(mask),
                        'skewness': pd.Series(regime_returns).skew(),
                        'kurtosis': pd.Series(regime_returns).kurtosis(),
                    }
                    
                    # Add Sharpe ratio if volatility is non-zero
                    if regime_stats['volatility'] != 0:
                        regime_stats['sharpe_ratio'] = regime_stats['mean_return'] / regime_stats['volatility']
                    else:
                        regime_stats['sharpe_ratio'] = 0
                    
                    regimes[f'Regime_{label}'] = regime_stats
                
            return regimes, labels
        except Exception as e:
            print(f"Error in detect_market_regimes: {e}")
            return {}, np.zeros(len(mds_coords))

    def analyze_period_results(self, period_results, period_name):
        try:
            combined_cloud = period_results['combined']['cloud']
            combined_diagrams = period_results['combined']['diagrams']
            processed_data = period_results['processed_data']
            
            # Convert processed_data to proper format for analysis
            if isinstance(processed_data, pd.DataFrame):
                returns = processed_data.mean(axis=1).values  # Use mean across indices
            else:
                returns = processed_data.flatten()
                
            # Ensure returns length matches cloud length
            if len(returns) > len(combined_cloud):
                returns = returns[-len(combined_cloud):]
            
            # Basic topological analysis
            feature_analysis = self.analyze_persistent_features(combined_diagrams)
            
            # Crisis indicators
            crisis_indicators = self.compute_crisis_indicators(combined_diagrams, returns)
            
            # Market regimes
            regimes, regime_labels = self.detect_market_regimes(
                period_results['combined']['mds_coords'],
                returns
            )
            
            # Market states and transitions
            states, transitions, local_var = self.detect_market_states_and_transitions(combined_cloud)
            
            # Stability analysis
            stability = self.compute_stability_measures(combined_cloud, returns)
            
            # Analyze individual indices
            separate_analysis = {}
            for index, result in period_results['separate'].items():
                try:
                    index_returns = processed_data[index] if isinstance(processed_data, pd.DataFrame) else processed_data
                    if len(index_returns) > len(result['cloud']):
                        index_returns = index_returns[-len(result['cloud']):]
                        
                    separate_analysis[index] = {
                        'features': self.analyze_persistent_features(result['diagrams']),
                        'crisis_indicators': self.compute_crisis_indicators(
                            result['diagrams'],
                            index_returns
                        ),
                        'stability': self.compute_stability_measures(
                            result['cloud'],
                            index_returns
                        )
                    }
                except Exception as e:
                    print(f"Error analyzing {index}: {e}")
                    continue
            
            return {
                'period_name': period_name,
                'combined': {
                    'features': feature_analysis,
                    'crisis_indicators': crisis_indicators,
                    'regimes': regimes,
                    'regime_labels': regime_labels,
                    'states': states,
                    'transitions': transitions,
                    'local_variance': local_var,
                    'stability': stability
                },
                'separate': separate_analysis
            }
            
        except Exception as e:
            print(f"Error in analyze_period_results: {e}")
            raise

    def compute_stability_measures(self, point_cloud, returns):
        """
        Compute comprehensive stability measures with robust error handling
        """
        try:
            # Initialize default values
            default_stats = {
                'density_stats': {
                    'mean': 0,
                    'std': 0,
                    'skew': 0
                },
                'network_measures': {
                    'avg_degree': 0,
                    'clustering_coef': 0,
                    'n_components': 0
                },
                'volatility_measures': {
                    'mean_vol': 0,
                    'vol_of_vol': 0,
                    'max_vol': 0
                },
                'local_density': np.zeros(len(point_cloud)),
                'distance_matrix': np.zeros((len(point_cloud), len(point_cloud)))
            }

            # Check for valid input
            if len(point_cloud) < 2:
                print("Warning: Not enough points for stability analysis")
                return default_stats

            # Local density estimation with error handling
            try:
                k_neighbors = min(10, len(point_cloud)-1)
                if k_neighbors < 1:
                    return default_stats
                    
                nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(point_cloud)
                distances, _ = nbrs.kneighbors(point_cloud)
                
                # Add small epsilon to avoid division by zero
                eps = 1e-10
                mean_distances = np.mean(distances, axis=1) + eps
                local_density = 1 / mean_distances
                
            except Exception as e:
                print(f"Error in local density calculation: {e}")
                local_density = np.zeros(len(point_cloud))
                
            # Network measures with error handling
            try:
                G, dist_matrix = self.create_persistence_network(point_cloud, epsilon=0.1)
                
                # Compute network statistics safely
                if G.number_of_nodes() > 0:
                    avg_degree = np.mean([d for n, d in G.degree()]) if G.number_of_edges() > 0 else 0
                    clustering_coef = nx.average_clustering(G) if G.number_of_edges() > 0 else 0
                    n_components = nx.number_connected_components(G)
                else:
                    avg_degree = clustering_coef = n_components = 0
                    
            except Exception as e:
                print(f"Error in network measures calculation: {e}")
                avg_degree = clustering_coef = n_components = 0
                dist_matrix = np.zeros((len(point_cloud), len(point_cloud)))
                
            # Volatility measures with error handling
            try:
                returns_series = pd.Series(returns)
                rolling_vol = returns_series.rolling(self.window_size, min_periods=1).std()
                
                volatility_stats = {
                    'mean_vol': rolling_vol.mean() if not rolling_vol.empty else 0,
                    'vol_of_vol': rolling_vol.std() if not rolling_vol.empty else 0,
                    'max_vol': rolling_vol.max() if not rolling_vol.empty else 0
                }
            except Exception as e:
                print(f"Error in volatility measures calculation: {e}")
                volatility_stats = {'mean_vol': 0, 'vol_of_vol': 0, 'max_vol': 0}
                
            # Combine all measures
            return {
                'density_stats': {
                    'mean': np.mean(local_density),
                    'std': np.std(local_density),
                    'skew': pd.Series(local_density).skew() if len(local_density) > 2 else 0
                },
                'network_measures': {
                    'avg_degree': avg_degree,
                    'clustering_coef': clustering_coef,
                    'n_components': n_components
                },
                'volatility_measures': volatility_stats,
                'local_density': local_density,
                'distance_matrix': dist_matrix
            }
            
        except Exception as e:
            print(f"Error in compute_stability_measures: {e}")
            return default_stats



    def generate_report(self, analysis_results):
        """
        Generate comprehensive analysis report
        """
        try:
            report = []
            report.append("=== Market Structure Analysis Report ===\n")
            
            for period, results in analysis_results.items():
                report.append(f"\nPeriod: {period}")
                report.append("="*50)
                
                combined = results['combined']
                
                # Crisis indicators
                report.append("\nCrisis Indicators:")
                crisis = combined['crisis_indicators']
                report.append(f"- Persistence Entropy: {crisis['persistence_entropy']:.3f}")
                report.append(f"- Topological Complexity: {crisis['topological_complexity']:.3f}")
                report.append(f"- Volatility: {crisis['volatility']:.3f}")
                report.append(f"- Max Drawdown: {crisis['max_drawdown']:.3f}")
                
                # Market regimes
                report.append("\nMarket Regimes:")
                for regime, stats in combined['regimes'].items():
                    report.append(f"\n{regime}:")
                    report.append(f"- Duration: {stats['duration']} days")
                    report.append(f"- Volatility: {stats['volatility']:.3f}")
                    report.append(f"- Mean Return: {stats['mean_return']:.3f}")
                    report.append(f"- Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}")
                
                # Stability measures
                report.append("\nStability Measures:")
                stability = combined['stability']
                report.append(f"- Mean Local Density: {stability['density_stats']['mean']:.3f}")
                report.append(f"- Network Clustering: {stability['network_measures']['clustering_coef']:.3f}")
                report.append(f"- Mean Volatility: {stability['volatility_measures']['mean_vol']:.3f}")
                
                # Critical transitions
                report.append(f"\nCritical Transitions: {len(combined['transitions'])}")
                
                # Individual indices
                report.append("\nIndividual Index Analysis:")
                for index, analysis in results['separate'].items():
                    report.append(f"\n{index}:")
                    report.append(f"- Persistence Entropy: {analysis['features']['persistence_entropy']:.3f}")
                    report.append(f"- Crisis Probability: {analysis['crisis_indicators']['persistence_entropy']:.3f}")
                    report.append(f"- Volatility: {analysis['stability']['volatility_measures']['mean_vol']:.3f}")
            
            return "\n".join(report)
            
        except Exception as e:
            print(f"Error generating report: {e}")
            raise

    def plot_analysis_results(self, analysis_results, save_path=None):
        """Plot analysis results with proper figure management"""
        try:
            print("  Creating main comparison plot...")
            fig = plt.figure(figsize=(20, 15))
            
            periods = list(analysis_results.keys())
            print(f"  Processing {len(periods)} periods...")
            
            # Plot 1: Crisis Indicators
            plt.subplot(321)
            indicators = [res['combined']['crisis_indicators'] 
                        for res in analysis_results.values()]
            plt.plot(periods, [i['persistence_entropy'] for i in indicators], 'o-', label='Entropy')
            plt.plot(periods, [i['topological_complexity'] for i in indicators], 's-', label='Complexity')
            plt.legend()
            plt.title('Crisis Indicators')
            plt.xticks(rotation=45)
            
            # Plot 2: Market Regimes
            print("  Creating regime plot...")
            plt.subplot(322)
            for period in periods:
                regimes = analysis_results[period]['combined']['regimes']
                plt.scatter([period]*len(regimes), 
                        [r['volatility'] for r in regimes.values()],
                        alpha=0.6)
            plt.title('Regime Volatilities')
            plt.xticks(rotation=45)
            
            # Plot 3: Stability Measures
            print("  Creating stability plot...")
            plt.subplot(323)
            stabilities = [res['combined']['stability'] for res in analysis_results.values()]
            plt.plot(periods, [s['density_stats']['mean'] for s in stabilities], 'o-')
            plt.title('Market Stability')
            plt.xticks(rotation=45)
            
            # Plot 4: Transitions
            plt.subplot(324)
            transitions = [len(res['combined']['transitions']) for res in analysis_results.values()]
            plt.bar(periods, transitions)
            plt.title('Number of Critical Transitions')
            plt.xticks(rotation=45)
            
            # Plot 5: Index Comparison
            plt.subplot(325)
            for idx in ['SP500', 'Russell', 'NASDAQ']:
                values = [res['separate'][idx]['crisis_indicators']['volatility'] 
                        for res in analysis_results.values()]
                plt.plot(periods, values, 'o-', label=idx)
            plt.legend()
            plt.title('Index-wise Volatility')
            plt.xticks(rotation=45)
            
            # Plot 6: Network Analysis
            plt.subplot(326)
            network_measures = [res['combined']['stability']['network_measures'] 
                            for res in analysis_results.values()]
            plt.plot(periods, [n['clustering_coef'] for n in network_measures], 'o-', 
                    label='Clustering')
            plt.plot(periods, [n['avg_degree']/10 for n in network_measures], 's-', 
                    label='Avg Degree/10')
            plt.legend()
            plt.title('Network Measures')
            plt.xticks(rotation=45)
            
            plt.suptitle('Market Analysis Results', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Save main plot
            print("  Saving main analysis plot...")
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # Create and save heatmap
            print("  Creating heatmap...")
            fig_heatmap = plt.figure(figsize=(12, 8))
            crisis_data = pd.DataFrame([res['combined']['crisis_indicators'] 
                                    for res in analysis_results.values()],
                                    index=periods)
            sns.heatmap(crisis_data, annot=True, cmap='YlOrRd', fmt='.2f')
            plt.title('Crisis Indicators Heatmap')
            if save_path:
                heatmap_path = save_path.replace('.png', '_heatmap.png')
                plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
            plt.close(fig_heatmap)
            
            # Create and save regime transitions
            print("  Creating regime transitions plot...")
            fig_regime = plt.figure(figsize=(12, 6))
            for i, period in enumerate(periods):
                regime_labels = analysis_results[period]['combined']['regime_labels']
                plt.scatter([i]*len(regime_labels), regime_labels, alpha=0.5, label=period)
            plt.xticks(range(len(periods)), periods, rotation=45)
            plt.title('Regime Transitions Across Periods')
            if save_path:
                regime_path = save_path.replace('.png', '_regimes.png')
                plt.savefig(regime_path, bbox_inches='tight', dpi=300)
            plt.close(fig_regime)
            
            # Create and save stability comparison
            print("  Creating stability comparison plot...")
            fig_stability = plt.figure(figsize=(12, 6))
            stability_metrics = pd.DataFrame(
                {period: res['combined']['stability']['volatility_measures'] 
                for period, res in analysis_results.items()}
            ).T
            stability_metrics.plot(kind='bar')
            plt.title('Stability Metrics Comparison')
            plt.xticks(rotation=45)
            if save_path:
                stability_path = save_path.replace('.png', '_stability.png')
                plt.savefig(stability_path, bbox_inches='tight', dpi=300)
            plt.close(fig_stability)
            
            print("  All analysis plots completed and saved")
            
        except Exception as e:
            print(f"Error in plot_analysis_results: {e}")
            plt.close('all')  # Clean up on error
            raise