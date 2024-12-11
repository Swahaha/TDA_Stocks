import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

class MarketTDA:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.scaler = StandardScaler()
        
    def prepare_data(self, data):
        """
        Prepare market data for TDA analysis.
        """
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Remove any infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with forward fill then backward fill
        returns = returns.ffill().bfill()  # Updated to use new method
        
        # Scale the returns for each index separately
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(returns),
            index=returns.index,
            columns=returns.columns
        )
        
        return scaled_data
        
    def create_combined_point_cloud(self, data):
        """
        Create point cloud using sliding windows where each point combines
        all three indices' windows into one high-dimensional point.
        """
        window_cloud = []
        
        # Create sliding windows
        for i in range(len(data) - self.window_size + 1):
            # Take window_size consecutive market states for all indices
            window = data.iloc[i:i + self.window_size].values
            # Flatten the window maintaining all indices together
            window_cloud.append(window.flatten())
            
        cloud = np.array(window_cloud)
        print(f"Combined cloud shape: {cloud.shape}")
        return cloud
    
    def create_separate_point_clouds(self, data):
        """
        Create separate point clouds for each index using sliding windows.
        """
        point_clouds = {}
        
        for column in data.columns:
            cloud = []
            # Create sliding windows for this index
            for i in range(len(data) - self.window_size + 1):
                # Take window_size consecutive states for this index
                window = data[column].iloc[i:i + self.window_size].values
                cloud.append(window)
            
            point_clouds[column] = np.array(cloud)
            print(f"{column} cloud shape: {point_clouds[column].shape}")
            
        return point_clouds
    
    def analyze_market_structure(self, data):
        """
        Perform TDA analysis using both approaches with sliding windows.
        """
        print("Preparing data...")
        processed_data = self.prepare_data(data)
        
        print("Creating point clouds...")
        # Approach 1: Combined point cloud with sliding windows
        combined_cloud = self.create_combined_point_cloud(processed_data)
        
        # Approach 2: Separate point clouds
        separate_clouds = self.create_separate_point_clouds(processed_data)
        
        print("Computing persistence diagrams...")
        # Compute persistence diagrams for combined approach
        print("Computing combined persistence...")
        ripser_output = ripser(combined_cloud, maxdim=1)
        combined_diagrams = ripser_output['dgms']
        
        print("Computing distance matrices and MDS...")
        # Compute distance matrix and MDS for combined approach
        combined_dist_matrix = squareform(pdist(combined_cloud))
        print("succesfully finished squareform")
        combined_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        combined_mds_coords = combined_mds.fit_transform(combined_dist_matrix)
        
        # Compute for separate approaches
        print("Computing separate persistences...")
        separate_results = {}
        for index, cloud in separate_clouds.items():
            print(f"Processing {index}...")
            diagrams = ripser(cloud, maxdim=1)['dgms']
            dist_matrix = squareform(pdist(cloud))
            mds_coords = MDS(n_components=2, dissimilarity='precomputed', 
                           random_state=42).fit_transform(dist_matrix)
            
            separate_results[index] = {
                'cloud': cloud,
                'diagrams': diagrams,
                'mds_coords': mds_coords
            }
        
        print("Analysis complete.")
        return {
            'processed_data': processed_data,
            'combined': {
                'cloud': combined_cloud,
                'diagrams': combined_diagrams,
                'mds_coords': combined_mds_coords
            },
            'separate': separate_results
        }
    
    def plot_analysis(self, results, save_path=None):
        """
        Create visualization comparing both approaches with proper figure management
        """
        try:
            # Create new figure
            fig = plt.figure(figsize=(20, 15))
            
            # Plot combined approach results
            plt.subplot(231)
            plot_diagrams(results['combined']['diagrams'], show=False)
            plt.title('Persistence Diagrams (Combined Approach)')
            
            plt.subplot(232)
            plt.scatter(
                results['combined']['mds_coords'][:, 0],
                results['combined']['mds_coords'][:, 1],
                alpha=0.6
            )
            plt.title('MDS Embedding (Combined Approach)')
            
            # Plot separate approach results
            for i, (index, result) in enumerate(results['separate'].items()):
                # Persistence diagrams
                plt.subplot(234 + i)
                plot_diagrams(result['diagrams'], show=False)
                plt.title(f'Persistence Diagrams ({index})')
                
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot to {save_path}")
            
            plt.close(fig)  # Close the figure explicitly
            
        except Exception as e:
            print(f"Error in plot_analysis: {str(e)}")
            plt.close('all')  # Ensure cleanup on error