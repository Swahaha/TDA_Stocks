import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_mutual_info_score
from scipy.stats import spearmanr

class MarketTDA:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.scaler = StandardScaler()
        
    def prepare_data(self, data):
        """Prepare market data for TDA analysis."""
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Remove any infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with forward fill then backward fill
        returns = returns.ffill().bfill()
        
        # Scale the returns for each index separately
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(returns),
            index=returns.index,
            columns=returns.columns
        )
        
        return scaled_data
        
    def create_combined_point_cloud(self, data):
        """Create point cloud combining all indices."""
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
        """Create separate point clouds for each index."""
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
        """Perform TDA analysis using both approaches."""
        print("Preparing data...")
        processed_data = self.prepare_data(data)
        
        print("Creating point clouds...")
        combined_cloud = self.create_combined_point_cloud(processed_data)
        separate_clouds = self.create_separate_point_clouds(processed_data)
        
        print("Computing persistence diagrams...")
        print("Computing combined persistence...")
        ripser_output = ripser(combined_cloud, maxdim=1)
        combined_diagrams = ripser_output['dgms']
        
        print("Computing distance matrices and MDS...")
        combined_dist_matrix = squareform(pdist(combined_cloud))
        print("succesfully finished squareform")
        combined_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        combined_mds_coords = combined_mds.fit_transform(combined_dist_matrix)
        
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
    
    def compare_approaches(self, results):
        """Compare the results of combined vs separate point cloud approaches."""
        comparisons = {}
        
        # 1. Compare topological features
        comparisons['feature_counts'] = {
            'combined': {
                'components': len(results['combined']['diagrams'][0]),
                'cycles': len(results['combined']['diagrams'][1])
            },
            'separate': {
                index: {
                    'components': len(result['diagrams'][0]),
                    'cycles': len(result['diagrams'][1])
                }
                for index, result in results['separate'].items()
            }
        }
        
        # 2. Compare persistence entropies
        def compute_persistence_entropy(diagram):
            if len(diagram) == 0:
                return 0
            persistence = diagram[:, 1] - diagram[:, 0]
            persistence = persistence[np.isfinite(persistence)]
            if len(persistence) == 0:
                return 0
            persistence = persistence / persistence.sum()
            return -np.sum(persistence * np.log(persistence + 1e-10))
        
        comparisons['persistence_entropy'] = {
            'combined': compute_persistence_entropy(results['combined']['diagrams'][1]),
            'separate': {
                index: compute_persistence_entropy(result['diagrams'][1])
                for index, result in results['separate'].items()
            }
        }
        
        # 3. Compare MDS embeddings correlation
        combined_mds = results['combined']['mds_coords']
        separate_mds = {
            index: result['mds_coords']
            for index, result in results['separate'].items()
        }
        
        comparisons['embedding_correlations'] = {}
        for index, mds_coords in separate_mds.items():
            combined_dist = np.sqrt(np.sum((combined_mds[None, :] - combined_mds[:, None]) ** 2, axis=2))
            separate_dist = np.sqrt(np.sum((mds_coords[None, :] - mds_coords[:, None]) ** 2, axis=2))
            
            corr, _ = spearmanr(combined_dist.flatten(), separate_dist.flatten())
            comparisons['embedding_correlations'][index] = corr
            
        # 4. Compare structure preservation
        def compute_structure_preservation(mds_coords, original_data):
            orig_dist = np.sqrt(np.sum((original_data[None, :] - original_data[:, None]) ** 2, axis=2))
            mds_dist = np.sqrt(np.sum((mds_coords[None, :] - mds_coords[:, None]) ** 2, axis=2))
            return spearmanr(orig_dist.flatten(), mds_dist.flatten())[0]
        
        comparisons['structure_preservation'] = {
            'combined': compute_structure_preservation(
                results['combined']['mds_coords'],
                results['combined']['cloud']
            ),
            'separate': {
                index: compute_structure_preservation(
                    result['mds_coords'],
                    result['cloud']
                )
                for index, result in results['separate'].items()
            }
        }
        
        return comparisons

    def plot_approach_comparison(self, comparisons, save_path=None):
        """Visualize the comparisons between approaches."""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Plot feature counts comparison
        plt.subplot(221)
        x = np.arange(len(comparisons['feature_counts']['separate']))
        width = 0.35
        
        combined_components = [comparisons['feature_counts']['combined']['components']] * len(comparisons['feature_counts']['separate'])
        combined_cycles = [comparisons['feature_counts']['combined']['cycles']] * len(comparisons['feature_counts']['separate'])
        
        separate_components = [info['components'] for info in comparisons['feature_counts']['separate'].values()]
        separate_cycles = [info['cycles'] for info in comparisons['feature_counts']['separate'].values()]
        
        plt.bar(x - width/2, combined_components, width, label='Combined Components')
        plt.bar(x + width/2, separate_components, width, label='Separate Components')
        plt.bar(x - width/2, combined_cycles, width, bottom=combined_components, label='Combined Cycles')
        plt.bar(x + width/2, separate_cycles, width, bottom=separate_components, label='Separate Cycles')
        
        plt.xlabel('Index')
        plt.ylabel('Feature Count')
        plt.title('Topological Feature Comparison')
        plt.xticks(x, comparisons['feature_counts']['separate'].keys())
        plt.legend()
        
        # 2. Plot persistence entropy comparison
        plt.subplot(222)
        indices = list(comparisons['persistence_entropy']['separate'].keys())
        combined_entropy = [comparisons['persistence_entropy']['combined']] * len(indices)
        separate_entropy = [comparisons['persistence_entropy']['separate'][idx] for idx in indices]
        
        plt.plot(indices, combined_entropy, 'o-', label='Combined')
        plt.plot(indices, separate_entropy, 's-', label='Separate')
        plt.xlabel('Index')
        plt.ylabel('Persistence Entropy')
        plt.title('Persistence Entropy Comparison')
        plt.legend()
        
        # 3. Plot embedding correlations
        plt.subplot(223)
        correlations = list(comparisons['embedding_correlations'].values())
        plt.bar(indices, correlations)
        plt.xlabel('Index')
        plt.ylabel('Correlation with Combined Embedding')
        plt.title('MDS Embedding Correlations')
        
        # 4. Plot structure preservation
        plt.subplot(224)
        combined_preservation = [comparisons['structure_preservation']['combined']] * len(indices)
        separate_preservation = [comparisons['structure_preservation']['separate'][idx] for idx in indices]
        
        plt.plot(indices, combined_preservation, 'o-', label='Combined')
        plt.plot(indices, separate_preservation, 's-', label='Separate')
        plt.xlabel('Index')
        plt.ylabel('Structure Preservation')
        plt.title('Structure Preservation Comparison')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    def plot_analysis(self, results, save_path=None):
        """Create visualization of TDA analysis results."""
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
                plt.subplot(234 + i)
                plot_diagrams(result['diagrams'], show=False)
                plt.title(f'Persistence Diagrams ({index})')
                
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot to {save_path}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error in plot_analysis: {str(e)}")
            plt.close('all')