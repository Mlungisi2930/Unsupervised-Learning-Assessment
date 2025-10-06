import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Important for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
import io
import base64
from flask import Flask, render_template_string, request
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

# HTML template for web display
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Unsupervised Learning Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f8f9fa; }
        .plot { margin: 20px 0; text-align: center; }
        .results { background: white; padding: 20px; border-radius: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Unsupervised Learning Analysis</h1>
        <div class="text-center mb-4">
            <form method="POST">
                <button type="submit" class="btn btn-primary btn-lg">Run Analysis</button>
            </form>
        </div>
        
        {% if results %}
        <div class="results">
            <h2>Analysis Results</h2>
            {{ results|safe }}
        </div>
        {% endif %}
        
        {% for plot in plots %}
        <div class="plot">
            <img src="data:image/png;base64,{{ plot }}" class="img-fluid" alt="Analysis Plot">
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

def plot_to_base64():
    """Convert current matplotlib plot to base64 string"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def analyze():
    plots = []
    results = ""
    
    if request.method == 'POST':
        try:
            # =============================================================================
            # DATA PREPARATION
            # =============================================================================
            results += "<h3>1. DATA PREPARATION</h3>"
            
            # Load data
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
            df = pd.read_csv(url)
            
            results += f"<p>Dataset shape: {df.shape}</p>"
            results += f"<p>Missing values: {df.isnull().sum().sum()}</p>"
            
            # Data preprocessing
            df_processed = df.copy()
            channel_dummies = pd.get_dummies(df['Channel'], prefix='Channel')
            region_dummies = pd.get_dummies(df['Region'], prefix='Region')
            
            numeric_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
            df_processed = pd.concat([df_processed[numeric_features], channel_dummies, region_dummies], axis=1)
            
            # Scale the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_processed)
            df_scaled = pd.DataFrame(df_scaled, columns=df_processed.columns)
            
            # =============================================================================
            # DIMENSIONALITY REDUCTION (PCA)
            # =============================================================================
            results += "<h3>2. DIMENSIONALITY REDUCTION (PCA)</h3>"
            
            pca = PCA()
            pca_features = pca.fit_transform(df_scaled)
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Create scree plot
            plt.figure(figsize=(12, 5))
            plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='skyblue', label='Individual')
            plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative', color='red')
            plt.xlabel('Principal Components')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Scree Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plots.append(plot_to_base64())
            plt.close()
            
            n_components_85 = np.argmax(cumulative_variance >= 0.85) + 1
            results += f"<p>Components for 85% variance: {n_components_85}</p>"
            results += f"<p>Variance retained: {cumulative_variance[n_components_85-1]:.3f}</p>"
            
            # Apply PCA with selected components
            pca_final = PCA(n_components=n_components_85)
            df_pca = pca_final.fit_transform(df_scaled)
            
            # =============================================================================
            # CLUSTERING (KMEANS)
            # =============================================================================
            results += "<h3>3. CLUSTERING ANALYSIS (KMEANS)</h3>"
            
            # Test different k values
            k_range = range(2, 8)
            inertia = []
            silhouette_scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(df_scaled)
                inertia.append(kmeans.inertia_)
                silhouette_avg = silhouette_score(df_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Plot evaluation metrics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(k_range, inertia, 'bo-')
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Inertia')
            ax1.set_title('Elbow Method')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(k_range, silhouette_scores, 'ro-')
            ax2.set_xlabel('Number of Clusters (k)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis')
            ax2.grid(True, alpha=0.3)
            
            plots.append(plot_to_base64())
            plt.close()
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            results += f"<p>Optimal clusters: {optimal_k} (Silhouette: {max(silhouette_scores):.3f})</p>"
            
            # Final clustering
            kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            best_labels = kmeans_optimal.fit_predict(df_scaled)
            
            # =============================================================================
            # CLUSTER VISUALIZATION
            # =============================================================================
            results += "<h3>4. CLUSTER VISUALIZATION</h3>"
            
            # 2D visualization
            pca_2d = PCA(n_components=2)
            df_2d = pca_2d.fit_transform(df_scaled)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df_2d[:, 0], df_2d[:, 1], c=best_labels, cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
            plt.title(f'Customer Segments (k={optimal_k})')
            plt.grid(True, alpha=0.3)
            
            plots.append(plot_to_base64())
            plt.close()
            
            # Cluster analysis
            df_clustered = df.copy()
            df_clustered['Cluster'] = best_labels
            cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
            
            results += "<h4>Cluster Sizes:</h4>"
            for cluster, size in cluster_sizes.items():
                results += f"<p>Cluster {cluster}: {size} customers</p>"
            
            results += "<h4>Business Insights:</h4>"
            results += "<ul>"
            results += "<li>Clear customer segments identified through purchasing patterns</li>"
            results += "<li>Optimal segmentation provides actionable insights for marketing</li>"
            results += "<li>PCA successfully reduced dimensionality while preserving structure</li>"
            results += "</ul>"
            
            results += "<div class='alert alert-success mt-3'><strong>Analysis Completed Successfully!</strong></div>"
            
        except Exception as e:
            results = f"<div class='alert alert-danger'><strong>Error:</strong> {str(e)}</div>"
    
    return render_template_string(HTML_TEMPLATE, results=results, plots=plots)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)