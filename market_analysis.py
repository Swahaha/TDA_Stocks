import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from persim import persistent_entropy as persim_entropy
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

    def analyze_persistent_features(self, diagrams):
        """
        Analyze persistence diagrams to identify significant market features
        """
        try:
            # Basic feature analysis
            components = diagrams[0]
            cycles = diagrams[1]
            component_lifetimes = components[:, 1] - components[:, 0]
            cycle_lifetimes = cycles[:, 1] - cycles[:, 0]
            
            # Advanced cycle analysis
            significant_threshold = np.percentile(cycle_lifetimes, 90)
            significant_cycles = cycles[cycle_lifetimes > significant_threshold]
            
            return {
                'n_components': len(components),
                'n_cycles': len(cycles),
                'avg_component_lifetime': np.mean(component_lifetimes),
                'avg_cycle_lifetime': np.mean(cycle_lifetimes),
                'max_component_lifetime': np.max(component_lifetimes),
                'max_cycle_lifetime': np.max(cycle_lifetimes),
                'significant_cycles': significant_cycles,
                'cycle_threshold': significant_threshold,
                'persistence_entropy': persim_entropy(cycles)
            }
        except Exception as e:
            print(f"Error in analyze_persistent_features: {e}")
            raise

    def create_persistence_network(self, point_cloud, epsilon=0.1):
        """
        Create network based on persistence features
        """
        try:
            distances = pdist(point_cloud)
            dist_matrix = squareform(distances)
            
            G = nx.Graph()
            for i in range(len(point_cloud)):
                for j in range(i+1, len(point_cloud)):
                    if dist_matrix[i,j] < epsilon:
                        G.add_edge(i, j, weight=dist_matrix[i,j])
            
            return G, dist_matrix
        except Exception as e:
            print(f"Error in create_persistence_network: {e}")
            raise

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
            persistence_entropy = persim_entropy(diagrams[1])
            
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
            # DBSCAN clustering for regime detection
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(mds_coords)
            labels = clustering.labels_
            
            # Analyze each regime
            regimes = {}
            for label in set(labels):
                mask = labels == label
                regime_returns = returns[mask]
                
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
            raise

    def compute_stability_measures(self, point_cloud, returns):
        """
        Compute comprehensive stability measures
        """
        try:
            # Local density estimation
            k_neighbors = min(10, len(point_cloud)-1)
            nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(point_cloud)
            distances, _ = nbrs.kneighbors(point_cloud)
            local_density = 1 / np.mean(distances, axis=1)
            
            # Network measures
            G, dist_matrix = self.create_persistence_network(point_cloud)
            network_stats = {
                'avg_degree': np.mean([d for n, d in G.degree()]),
                'clustering_coef': nx.average_clustering(G),
                'n_components': nx.number_connected_components(G)
            }
            
            # Volatility measures
            rolling_vol = pd.Series(returns).rolling(self.window_size).std()
            volatility_stats = {
                'mean_vol': np.mean(rolling_vol),
                'vol_of_vol': np.std(rolling_vol),
                'max_vol': np.max(rolling_vol)
            }
            
            return {
                'density_stats': {
                    'mean': np.mean(local_density),
                    'std': np.std(local_density),
                    'skew': pd.Series(local_density).skew()
                },
                'network_measures': network_stats,
                'volatility_measures': volatility_stats,
                'local_density': local_density,
                'distance_matrix': dist_matrix
            }
        except Exception as e:
            print(f"Error in compute_stability_measures: {e}")
            raise

    def analyze_period_results(self, period_results, period_name):
        """
        Comprehensive period analysis
        """
        try:
            combined_cloud = period_results['combined']['cloud']
            combined_diagrams = period_results['combined']['diagrams']
            processed_data = period_results['processed_data']
            
            # Basic topological analysis
            feature_analysis = self.analyze_persistent_features(combined_diagrams)
            
            # Crisis indicators
            crisis_indicators = self.compute_crisis_indicators(
                combined_diagrams, 
                processed_data.values.flatten()
            )
            
            # Market regimes
            regimes, regime_labels = self.detect_market_regimes(
                period_results['combined']['mds_coords'],
                processed_data.values.flatten()
            )
            
            # Market states and transitions
            states, transitions, local_var = self.detect_market_states_and_transitions(combined_cloud)
            
            # Stability analysis
            stability = self.compute_stability_measures(
                combined_cloud,
                processed_data.values.flatten()
            )
            
            # Analyze individual indices
            separate_analysis = {}
            for index, result in period_results['separate'].items():
                try:
                    separate_analysis[index] = {
                        'features': self.analyze_persistent_features(result['diagrams']),
                        'crisis_indicators': self.compute_crisis_indicators(
                            result['diagrams'],
                            processed_data[index]
                        ),
                        'stability': self.compute_stability_measures(
                            result['cloud'],
                            processed_data[index]
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
        try:
            fig = plt.figure(figsize=(20, 15))
            
            periods = list(analysis_results.keys())
            
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
            plt.subplot(322)
            for period in periods:
                regimes = analysis_results[period]['combined']['regimes']
                plt.scatter([period]*len(regimes), 
                        [r['volatility'] for r in regimes.values()],
                        alpha=0.6)
            plt.title('Regime Volatilities')
            plt.xticks(rotation=45)
            
            # Plot 3: Stability Measures
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
            
            # Add a main title
            plt.suptitle('Market Analysis Results', fontsize=16, y=1.02)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            plt.show()
            
            # Create additional plots for deeper analysis
            # Plot 7: Crisis Indicators Heatmap
            plt.figure(figsize=(12, 8))
            crisis_data = pd.DataFrame([res['combined']['crisis_indicators'] 
                                    for res in analysis_results.values()],
                                    index=periods)
            sns.heatmap(crisis_data, annot=True, cmap='YlOrRd', fmt='.2f')
            plt.title('Crisis Indicators Heatmap')
            if save_path:
                plt.savefig(save_path.replace('.png', '_heatmap.png'), 
                        bbox_inches='tight', dpi=300)
            plt.show()
            
            # Plot 8: Regime Transitions
            plt.figure(figsize=(12, 6))
            for i, period in enumerate(periods):
                regime_labels = analysis_results[period]['combined']['regime_labels']
                plt.scatter([i]*len(regime_labels), regime_labels, 
                        alpha=0.5, label=period)
            plt.xticks(range(len(periods)), periods, rotation=45)
            plt.title('Regime Transitions Across Periods')
            if save_path:
                plt.savefig(save_path.replace('.png', '_regimes.png'), 
                        bbox_inches='tight', dpi=300)
            plt.show()
            
            # Plot 9: Stability Comparison
            plt.figure(figsize=(12, 6))
            stability_metrics = pd.DataFrame(
                {period: res['combined']['stability']['volatility_measures'] 
                for period, res in analysis_results.items()}
            ).T
            stability_metrics.plot(kind='bar')
            plt.title('Stability Metrics Comparison')
            plt.xticks(rotation=45)
            if save_path:
                plt.savefig(save_path.replace('.png', '_stability.png'), 
                        bbox_inches='tight', dpi=300)
            plt.show()
            
        except Exception as e:
            print(f"Error in plot_analysis_results: {e}")
            plt.close('all')  # Clean up any open figures
            raise