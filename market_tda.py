import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.manifold import MDS
from datetime import datetime, timedelta

class MarketTDA:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_data(self, sp500, russell, nasdaq):
        """
        Prepare and combine market data for TDA analysis
        
        Parameters:
        sp500, russell, nasdaq: pandas DataFrames with columns ['Date', 'Close']
        """
        # Ensure all dataframes have the same dates
        common_dates = sorted(set(sp500['Date']) & set(russell['Date']) & set(nasdaq['Date']))
        
        # Create combined dataset
        data = pd.DataFrame(index=common_dates)
        data['SP500'] = sp500.set_index('Date')['Close']
        data['Russell'] = russell.set_index('Date')['Close']
        data['NASDAQ'] = nasdaq.set_index('Date')['Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate rolling windows (20-day)
        rolling_data = pd.DataFrame()
        for col in returns.columns:
            rolling_data[f'{col}_mean'] = returns[col].rolling(20).mean()
            rolling_data[f'{col}_std'] = returns[col].rolling(20).std()
            rolling_data[f'{col}_skew'] = returns[col].rolling(20).skew()
            
        return returns, rolling_data.dropna()
    
    def create_point_cloud(self, data, window_size=20):
        """
        Create point cloud from rolling windows of market data
        """
        scaled_data = self.scaler.fit_transform(data)
        point_cloud = []
        
        for i in range(len(scaled_data) - window_size + 1):
            window = scaled_data[i:i + window_size].flatten()
            point_cloud.append(window)
            
        return np.array(point_cloud)
    
    def compute_persistence(self, point_cloud, max_dim=2):
        """
        Compute persistence diagrams using Ripser
        """
        diagrams = ripser(point_cloud, maxdim=max_dim)['diagrams']
        return diagrams
    
    def analyze_market_structure(self, returns, rolling_data, window_size=20):
        """
        Perform complete TDA analysis on market data
        """
        # Create point cloud from rolling features
        point_cloud = self.create_point_cloud(rolling_data, window_size)
        
        # Compute persistence diagrams
        diagrams = self.compute_persistence(point_cloud)
        
        # Compute distance matrix
        dist_matrix = squareform(pdist(point_cloud))
        
        # Perform MDS for visualization
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        mds_coords = mds.fit_transform(dist_matrix)
        
        return diagrams, dist_matrix, mds_coords
    
    def plot_analysis(self, diagrams, mds_coords, returns, save_path=None):
        """
        Create visualization of TDA analysis
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Plot persistence diagrams
        plt.subplot(2, 2, 1)
        plot_diagrams(diagrams, show=False)
        plt.title('Persistence Diagrams')
        
        # Plot MDS embedding
        plt.subplot(2, 2, 2)
        plt.scatter(mds_coords[:, 0], mds_coords[:, 1], alpha=0.6)
        plt.title('MDS Embedding of Market States')
        
        # Plot returns distribution
        plt.subplot(2, 2, 3)
        for col in returns.columns:
            sns.kdeplot(returns[col], label=col)
        plt.title('Returns Distribution')
        plt.legend()
        
        # Plot correlation heatmap
        plt.subplot(2, 2, 4)
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def detect_market_regimes(tda_analysis, mds_coords, returns, n_clusters=3):
    """
    Detect market regimes using clustering on TDA features
    """
    from sklearn.cluster import KMeans
    
    # Perform clustering on MDS coordinates
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(mds_coords)
    
    # Analyze characteristics of each regime
    regime_analysis = pd.DataFrame()
    for i in range(n_clusters):
        mask = clusters == i
        regime_returns = returns[mask]
        
        regime_analysis.loc[f'Regime {i}', 'Mean Returns SP500'] = regime_returns['SP500'].mean()
        regime_analysis.loc[f'Regime {i}', 'Volatility SP500'] = regime_returns['SP500'].std()
        regime_analysis.loc[f'Regime {i}', 'NASDAQ'] = regime_returns['NASDAQ'].mean()
        
    return clusters, regime_analysis