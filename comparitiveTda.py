import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams, wasserstein
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

class ComparativeTDA:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.scaler = StandardScaler()
    
    def create_individual_point_clouds(self, data):
        """
        Create separate point clouds for each market index
        
        Parameters:
        data: DataFrame with market indices
        
        Returns:
        dict: Point clouds for each index
        """
        point_clouds = {}
        
        for column in data.columns:
            # Scale the individual index data
            scaled_data = self.scaler.fit_transform(data[[column]])
            
            # Create point cloud using sliding window
            points = []
            for i in range(len(scaled_data) - self.window_size + 1):
                window = scaled_data[i:i + self.window_size]
                points.append(window.flatten())
            
            point_clouds[column] = np.array(points)
            
        return point_clouds
    
    def create_combined_point_cloud(self, data):
        """
        Create single point cloud using all indices as coordinates
        
        Parameters:
        data: DataFrame with market indices
        
        Returns:
        np.array: Combined point cloud
        """
        # Scale all data together
        scaled_data = self.scaler.fit_transform(data)
        
        # Each point is [SP500, Russell, NASDAQ] at each time point
        return scaled_data
    
    def analyze_both_approaches(self, data):
        """
        Perform TDA analysis using both individual and combined approaches
        
        Parameters:
        data: DataFrame with market indices
        
        Returns:
        dict: Results from both approaches
        """
        # Approach 1: Individual point clouds
        individual_clouds = self.create_individual_point_clouds(data)
        individual_results = {}
        
        for index, cloud in individual_clouds.items():
            diagrams = ripser(cloud, maxdim=2)['diagrams']
            dist_matrix = squareform(pdist(cloud))
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            mds_coords = mds.fit_transform(dist_matrix)
            
            individual_results[index] = {
                'cloud': cloud,
                'diagrams': diagrams,
                'mds_coords': mds_coords
            }
        
        # Approach 2: Combined point cloud
        combined_cloud = self.create_combined_point_cloud(data)
        combined_diagrams = ripser(combined_cloud, maxdim=2)['diagrams']
        combined_dist_matrix = squareform(pdist(combined_cloud))
        combined_mds_coords = MDS(n_components=2).fit_transform(combined_dist_matrix)
        
        return {
            'individual': individual_results,
            'combined': {
                'cloud': combined_cloud,
                'diagrams': combined_diagrams,
                'mds_coords': combined_mds_coords
            }
        }
    
    def plot_comparison(self, results, save_path=None):
        """
        Plot comparison of both approaches
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Individual approach plots
        for i, (index, result) in enumerate(results['individual'].items()):
            # Plot persistence diagrams
            plt.subplot(3, 3, i*3 + 1)
            plot_diagrams(result['diagrams'], show=False)
            plt.title(f'{index} Persistence Diagrams')
            
            # Plot MDS embedding
            plt.subplot(3, 3, i*3 + 2)
            plt.scatter(result['mds_coords'][:, 0], result['mds_coords'][:, 1], alpha=0.6)
            plt.title(f'{index} MDS Embedding')
            
            # Plot first two dimensions of point cloud
            plt.subplot(3, 3, i*3 + 3)
            plt.scatter(result['cloud'][:, 0], result['cloud'][:, 1], alpha=0.6)
            plt.title(f'{index} Point Cloud (First 2 Dims)')
        
        # Combined approach plots
        fig2 = plt.figure(figsize=(15, 5))
        
        # Combined persistence diagrams
        plt.subplot(131)
        plot_diagrams(results['combined']['diagrams'], show=False)
        plt.title('Combined Persistence Diagrams')
        
        # Combined MDS embedding
        plt.subplot(132)
        plt.scatter(results['combined']['mds_coords'][:, 0], 
                   results['combined']['mds_coords'][:, 1], alpha=0.6)
        plt.title('Combined MDS Embedding')
        
        # Combined point cloud in 3D
        ax = fig2.add_subplot(133, projection='3d')
        cloud = results['combined']['cloud']
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], alpha=0.6)
        plt.title('Combined 3D Point Cloud')
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path + '_individual.png')
            fig2.savefig(save_path + '_combined.png')
        plt.show()

def analyze_data_approaches(data, tda):
    """
    Analyze both continuous and discrete approaches
    """
    # Approach 1: Continuous data
    continuous_results = tda.analyze_both_approaches(data)
    
    # Approach 2: Discrete blocks
    crisis_periods = {
        'Financial_Crisis': ('2008-01-01', '2009-12-31'),
        'Covid_Crisis': ('2020-01-01', '2020-12-31'),
        'Dot_Com_Bubble': ('2000-01-01', '2002-12-31')
    }
    
    discrete_results = {}
    for period_name, (start, end) in crisis_periods.items():
        period_data = data[start:end]
        discrete_results[period_name] = tda.analyze_both_approaches(period_data)
    
    return continuous_results, discrete_results