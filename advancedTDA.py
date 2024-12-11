import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams, wasserstein, persistent_entropy
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import matplotlib as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors

class AdvancedMarketTDA:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.scaler = StandardScaler()
    
    def compute_persistence_entropy(self, diagrams):
        """
        Compute persistence entropy to measure complexity of topological features
        
        Returns:
        float: Entropy value indicating topological complexity
        """
        return persistent_entropy(diagrams[1])  # Focus on 1-dimensional features
    
    def create_persistence_network(self, point_cloud, epsilon=0.1):
        """
        Create a network based on persistence features
        Reveals relationships between market states
        """
        # Compute pairwise distances
        distances = pdist(point_cloud)
        dist_matrix = squareform(distances)
        
        # Create network
        G = nx.Graph()
        
        # Add edges where distance is below threshold
        for i in range(len(point_cloud)):
            for j in range(i+1, len(point_cloud)):
                if dist_matrix[i,j] < epsilon:
                    G.add_edge(i, j, weight=dist_matrix[i,j])
        
        return G
    
    def detect_market_states(self, point_cloud, eps=0.5, min_samples=5):
        """
        Use DBSCAN to detect distinct market states from point cloud
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)
        return clustering.labels_
    
    def compute_wasserstein_distances(self, period_diagrams):
        """
        Compute Wasserstein distances between persistence diagrams
        to measure similarity between different time periods
        """
        n_periods = len(period_diagrams)
        distances = np.zeros((n_periods, n_periods))
        
        for i in range(n_periods):
            for j in range(i+1, n_periods):
                dist = wasserstein(period_diagrams[i][1], period_diagrams[j][1])
                distances[i,j] = distances[j,i] = dist
                
        return distances
    
    def detect_critical_transitions(self, point_cloud, k_neighbors=5):
        """
        Detect critical transitions in market behavior
        using local manifold analysis
        """
        # Compute local dimensionality
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(point_cloud)
        distances, _ = nbrs.kneighbors(point_cloud)
        
        # Estimate local variance
        local_var = np.var(distances, axis=1)
        
        # Detect transitions as points with high local variance
        threshold = np.mean(local_var) + 2*np.std(local_var)
        transitions = np.where(local_var > threshold)[0]
        
        return transitions, local_var
    
    def analyze_cycle_persistence(self, diagrams):
        """
        Analyze the persistence of cycles in market behavior
        """
        # Extract 1-dimensional features (cycles)
        cycles = diagrams[1]
        
        # Compute persistence lengths
        persistence_lengths = cycles[:,1] - cycles[:,0]
        
        # Find significant cycles
        significant_threshold = np.percentile(persistence_lengths, 90)
        significant_cycles = cycles[persistence_lengths > significant_threshold]
        
        return {
            'persistence_lengths': persistence_lengths,
            'significant_cycles': significant_cycles,
            'threshold': significant_threshold
        }
    
    def compute_stability_measures(self, point_cloud):
        """
        Compute measures of market stability using
        topological features
        """
        # Local density estimation
        nbrs = NearestNeighbors(n_neighbors=10).fit(point_cloud)
        distances, _ = nbrs.kneighbors(point_cloud)
        local_density = 1 / np.mean(distances, axis=1)
        
        # Stability metrics
        stability = {
            'density_mean': np.mean(local_density),
            'density_std': np.std(local_density),
            'density_skew': pd.Series(local_density).skew(),
            'density_distribution': local_density
        }
        
        return stability
    
    def perform_advanced_analysis(self, data):
        """
        Perform comprehensive TDA analysis
        """
        # Create point cloud
        point_cloud = self.create_combined_point_cloud(data)
        
        # Basic persistence computation
        diagrams = ripser(point_cloud, maxdim=2)['diagrams']
        
        # Advanced analyses
        results = {
            'persistence_entropy': self.compute_persistence_entropy(diagrams),
            'market_states': self.detect_market_states(point_cloud),
            'critical_transitions': self.detect_critical_transitions(point_cloud),
            'cycle_analysis': self.analyze_cycle_persistence(diagrams),
            'stability_measures': self.compute_stability_measures(point_cloud),
            'persistence_network': self.create_persistence_network(point_cloud)
        }
        
        return results
    
    def plot_advanced_analysis(self, results, data, save_path=None):
        """
        Create visualizations for advanced analyses
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Plot critical transitions
        plt.subplot(321)
        transitions, local_var = results['critical_transitions']
        plt.plot(local_var)
        plt.scatter(transitions, local_var[transitions], c='r')
        plt.title('Critical Transitions Detection')
        
        # Plot market states
        plt.subplot(322)
        states = results['market_states']
        plt.scatter(range(len(states)), data.values, c=states, cmap='viridis')
        plt.title('Market States Classification')
        
        # Plot cycle persistence distribution
        plt.subplot(323)
        cycle_analysis = results['cycle_analysis']
        plt.hist(cycle_analysis['persistence_lengths'], bins=30)
        plt.axvline(cycle_analysis['threshold'], c='r', linestyle='--')
        plt.title('Cycle Persistence Distribution')
        
        # Plot stability measures
        plt.subplot(324)
        stability = results['stability_measures']
        plt.hist(stability['density_distribution'], bins=30)
        plt.title('Market Stability Distribution')
        
        # Plot persistence network
        plt.subplot(325)
        G = results['persistence_network']
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=20, alpha=0.6)
        plt.title('Persistence Network')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()