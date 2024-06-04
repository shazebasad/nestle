
import pandas as pd
import gower
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class ClusterSegment:
    def __init__(self, df, clustering_features):
        self.df = df
        self.clustering_features = clustering_features
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def encode_categorical_variables(self):
        X_clustering_gower = self.df[self.clustering_features].copy()
        for col in ['Family status']:
            le = LabelEncoder()
            X_clustering_gower.loc[:, col] = le.fit_transform(X_clustering_gower[col])
            self.label_encoders[col] = le
        return X_clustering_gower

    def compute_gower_distance_matrix(self, X_clustering_gower):
        return gower.gower_matrix(X_clustering_gower)

    def perform_clustering(self, gower_dist_matrix, n_clusters=3):
        agglom_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        clusters = agglom_clustering.fit_predict(gower_dist_matrix)
        self.df['Cluster_Gower'] = clusters
        return clusters

    def reduce_dimensions(self, gower_dist_matrix):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(gower_dist_matrix)
        return pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

    def calculate_centroids(self, pca_df):
        return pca_df.groupby('Cluster').mean()

    def calculate_centroids_values(self, numeric_features):
        centroids_values = self.df.groupby('Cluster_Gower')[numeric_features].mean()
        centroids_values_scaled = pd.DataFrame(self.scaler.fit_transform(centroids_values), columns=centroids_values.columns, index=centroids_values.index)
        centroids_values_scaled['Cluster_Gower'] = centroids_values_scaled.index
        return centroids_values_scaled

    def plot_clusters(self, pca_df, centroids):
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis', s=50)
        ax.scatter(centroids['PCA1'], centroids['PCA2'], c='red', marker='x', s=200)  # plot centroids

        for cluster_num in centroids.index:
            cluster_points = pca_df[pca_df['Cluster'] == cluster_num][['PCA1', 'PCA2']]
            if len(cluster_points) > 1:  # Only if there are enough points to calculate covariance
                cov = np.cov(cluster_points.T)
                cluster_center = centroids.loc[cluster_num]
                self.plot_ellipse(ax, cluster_center, cov, cluster_num)

        ax.set_title('Cluster Visualization using PCA')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        plt.colorbar(scatter, label='Cluster')
        plt.show()

    @staticmethod
    def plot_ellipse(ax, center, cov, cluster_num):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = patches.Ellipse(xy=(center['PCA1'], center['PCA2']), width=width, height=height, angle=angle, edgecolor='black', facecolor='none')
        ax.add_patch(ellipse)
        ax.text(center['PCA1'] + 0.1, center['PCA2'] + 0.1, f'Cluster {cluster_num}', horizontalalignment='left', verticalalignment='bottom', fontsize=12, color='black', weight='bold')

    def profile_clusters(self, numeric_features):
        segment_profiles_gower = self.df.groupby('Cluster_Gower').agg({
            'Age': lambda x: x.mode()[0],
            'Income': lambda x: x.mode()[0],
            'Family status': lambda x: x.mode()[0],
            'Confectionary Shopping Value 2022': 'mean',
            'Confectionary Shopping Value 2023': 'mean',
            'Coffee Shopping Value 2022': 'mean',
            'Coffee Shopping Value 2023': 'mean'
        })
        segment_profiles_gower_scaled = pd.DataFrame(self.scaler.fit_transform(segment_profiles_gower), columns=segment_profiles_gower.columns, index=segment_profiles_gower.index)
        segment_profiles_gower_scaled['Cluster_Gower'] = segment_profiles_gower_scaled.index
        return segment_profiles_gower

    def create_radar_chart(self, ax, data, categories, title):
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        values = data.tolist()
        values += values[:1]
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(title)

    def plot_radar_charts(self, segment_profiles_standardized, categories):
        num_clusters = segment_profiles_standardized.shape[0]
        fig, axes = plt.subplots(num_clusters, 1, figsize=(6, 6 * num_clusters), subplot_kw=dict(polar=True))
        for i, ax in enumerate(axes.flatten()):
            cluster_data = segment_profiles_standardized.iloc[i][categories]
            self.create_radar_chart(ax, cluster_data, categories, f'Cluster {i}')
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        X_clustering_gower = self.encode_categorical_variables()
        gower_dist_matrix = self.compute_gower_distance_matrix(X_clustering_gower)
        clusters = self.perform_clustering(gower_dist_matrix)
        pca_df = self.reduce_dimensions(gower_dist_matrix)
        pca_df['Cluster'] = clusters
        centroids = self.calculate_centroids(pca_df)
        numeric_features = ['Age', 'Income', 'Confectionary Shopping Value 2022', 'Confectionary Shopping Value 2023',
                            'Coffee Shopping Value 2022', 'Coffee Shopping Value 2023']
        centroids_values_scaled = self.calculate_centroids_values(numeric_features)
        self.plot_clusters(pca_df, centroids)
        segment_profiles_gower = self.profile_clusters(numeric_features)
        
        # Standardize the cluster profiles
        segment_profiles_standardized = segment_profiles_gower.copy()
        segment_profiles_standardized[numeric_features] = self.scaler.fit_transform(segment_profiles_gower[numeric_features])
        
        # Plot radar charts
        self.plot_radar_charts(segment_profiles_standardized, numeric_features)

        return segment_profiles_gower
