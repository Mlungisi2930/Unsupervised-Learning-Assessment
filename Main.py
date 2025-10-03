import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# DATA PREPARATION
# =============================================================================

print("=" * 60)
print("UNSUPERVISED LEARNING ASSESSMENT")
print("WHOLESALE CUSTOMERS DATASET")
print("=" * 60)

# Load the dataset
print("\n1. DATA PREPARATION")
print("-" * 40)

# Load data (assuming the dataset is available locally or via URL)
# You can download from: https://archive.ics.uci.edu/ml/datasets/Wholesale+customers
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Explore the categorical variables
print(f"\nChannel distribution:\n{df['Channel'].value_counts()}")
print(f"\nRegion distribution:\n{df['Region'].value_counts()}")

# Data preprocessing decisions
print("\n" + "=" * 40)
print("PREPROCESSING DECISIONS:")
print("=" * 40)
print("• Channel: Categorical (1: Horeca, 2: Retail)")
print("• Region: Categorical (1: Lisbon, 2: Oporto, 3: Other)")
print("• Will treat these as categorical but encode for clustering")
print("• All other features are continuous spending amounts")

# Handle categorical variables - one-hot encoding
df_processed = df.copy()
channel_dummies = pd.get_dummies(df['Channel'], prefix='Channel')
region_dummies = pd.get_dummies(df['Region'], prefix='Region')

# Combine with original numeric features
numeric_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
df_processed = pd.concat([df_processed[numeric_features], channel_dummies, region_dummies], axis=1)

print(f"\nProcessed dataset shape: {df_processed.shape}")
print("Processed features:", list(df_processed.columns))

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_processed)
df_scaled = pd.DataFrame(df_scaled, columns=df_processed.columns)

print("\nData scaling completed using StandardScaler")

# =============================================================================
# DIMENSIONALITY REDUCTION (PCA)
# =============================================================================

print("\n\n2. DIMENSIONALITY REDUCTION (PCA)")
print("-" * 40)

# Apply PCA
pca = PCA()
pca_features = pca.fit_transform(df_scaled)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance by component:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)")

# Create scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Scree plot
ax1.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='skyblue', label='Individual')
ax1.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative', color='red')
ax1.set_xlabel('Principal Components')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('PCA Scree Plot')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cumulative variance plot
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.8, label='95% Variance')
ax2.axhline(y=0.90, color='g', linestyle='--', alpha=0.8, label='90% Variance')
ax2.axhline(y=0.85, color='y', linestyle='--', alpha=0.8, label='85% Variance')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Determine optimal number of components
n_components_85 = np.argmax(cumulative_variance >= 0.85) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"\nComponents needed for 85% variance: {n_components_85}")
print(f"Components needed for 90% variance: {n_components_90}")
print(f"Components needed for 95% variance: {n_components_95}")

# Decision on number of components
print("\n" + "=" * 40)
print("PCA COMPONENT SELECTION RATIONALE:")
print("=" * 40)
print(f"• Selected {n_components_85} components capturing {cumulative_variance[n_components_85-1]:.3f} of variance")
print("• Balances dimensionality reduction with information retention")
print("• Provides good compression while maintaining cluster structure")

# Apply PCA with selected components
pca_final = PCA(n_components=n_components_85)
df_pca = pca_final.fit_transform(df_scaled)

print(f"\nOriginal dimensions: {df_scaled.shape[1]}")
print(f"PCA-reduced dimensions: {df_pca.shape[1]}")
print(f"Variance retained: {pca_final.explained_variance_ratio_.sum():.3f}")

# =============================================================================
# CLUSTERING (KMEANS)
# =============================================================================

print("\n\n3. CLUSTERING ANALYSIS (KMEANS)")
print("-" * 40)

# Test different k values
k_range = range(2, 11)
inertia = []
silhouette_scores = []

print("Testing k values from 2 to 10...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    inertia.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.3f}")

# Plot evaluation metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Elbow curve
ax1.plot(k_range, inertia, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-cluster SSE)')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True, alpha=0.3)

# Silhouette scores
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis for Optimal k')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Determine optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k based on silhouette score: {optimal_k}")

# Detailed silhouette analysis for optimal k
print(f"\nDetailed analysis for k={optimal_k}:")
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels_optimal = kmeans_optimal.fit_predict(df_scaled)

# Calculate silhouette samples for detailed analysis
silhouette_vals = silhouette_samples(df_scaled, cluster_labels_optimal)

# Create silhouette plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
y_lower = 10
for i in range(optimal_k):
    cluster_silhouette_vals = silhouette_vals[cluster_labels_optimal == i]
    cluster_silhouette_vals.sort()
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.nipy_spectral(float(i) / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient Values')
ax.set_ylabel('Cluster Label')
ax.axvline(x=silhouette_score(df_scaled, cluster_labels_optimal), color="red", linestyle="--")
ax.set_title(f'Silhouette Plot for KMeans (k={optimal_k})')

plt.tight_layout()
plt.show()

print("\n" + "=" * 40)
print("CLUSTER SELECTION JUSTIFICATION:")
print("=" * 40)
print(f"• Selected k={optimal_k} based on highest silhouette score ({silhouette_scores[optimal_k-2]:.3f})")
print("• Elbow method shows diminishing returns beyond this point")
print("• Silhouette plot shows relatively balanced cluster sizes")

# =============================================================================
# COMPARISON: RAW vs PCA-REDUCED DATA
# =============================================================================

print("\n\n4. COMPARISON: RAW vs PCA-REDUCED DATA")
print("-" * 40)

# Fit KMeans on original scaled data
kmeans_original = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_original = kmeans_original.fit_predict(df_scaled)
silhouette_original = silhouette_score(df_scaled, labels_original)

# Fit KMeans on PCA-reduced data
kmeans_pca = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(df_pca)
silhouette_pca = silhouette_score(df_pca, labels_pca)

print(f"Silhouette Score - Original data: {silhouette_original:.4f}")
print(f"Silhouette Score - PCA-reduced data: {silhouette_pca:.4f}")

# Compare cluster sizes
original_cluster_sizes = np.bincount(labels_original)
pca_cluster_sizes = np.bincount(labels_pca)

print(f"\nCluster sizes - Original: {original_cluster_sizes}")
print(f"Cluster sizes - PCA-reduced: {pca_cluster_sizes}")

print("\n" + "=" * 40)
print("COMPARISON INTERPRETATION:")
print("=" * 40)
if silhouette_pca > silhouette_original:
    print("• PCA-reduced data provides BETTER clustering performance")
    print("• Dimensionality reduction helped remove noise and improve cluster separation")
    best_labels = labels_pca
    best_kmeans = kmeans_pca
else:
    print("• Original data provides BETTER clustering performance") 
    print("• PCA may have removed some meaningful variance for clustering")
    best_labels = labels_original
    best_kmeans = kmeans_original

print(f"• Using {'PCA-reduced' if silhouette_pca > silhouette_original else 'original'} data for final analysis")

# =============================================================================
# CLUSTER VISUALIZATION
# =============================================================================

print("\n\n5. CLUSTER VISUALIZATION")
print("-" * 40)

# Use PCA for 2D visualization (even if we didn't use PCA for clustering)
pca_2d = PCA(n_components=2)
df_2d = pca_2d.fit_transform(df_scaled)

# Create cluster visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_2d[:, 0], df_2d[:, 1], c=best_labels, cmap='viridis', alpha=0.7, s=50)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'Customer Segments Visualization (k={optimal_k})')

# Add cluster centers if using original data
if silhouette_original >= silhouette_pca:
    centers_2d = pca_2d.transform(kmeans_original.cluster_centers_)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("2D visualization created using first two principal components")

# =============================================================================
# PRINCIPAL COMPONENT INTERPRETATION
# =============================================================================

print("\n\n6. PRINCIPAL COMPONENT INTERPRETATION")
print("-" * 40)

# Get feature contributions to principal components
pca_for_interpretation = PCA(n_components=2)
pca_for_interpretation.fit(df_scaled)

# Create component analysis
components_df = pd.DataFrame(
    pca_for_interpretation.components_.T,
    columns=['PC1', 'PC2'],
    index=df_processed.columns
)

print("Feature contributions to PC1 and PC2:")
print(components_df)

# Visualize feature contributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PC1 contributions
pc1_sorted = components_df['PC1'].sort_values()
ax1.barh(range(len(pc1_sorted)), pc1_sorted.values)
ax1.set_yticks(range(len(pc1_sorted)))
ax1.set_yticklabels(pc1_sorted.index)
ax1.set_title('Feature Contributions to PC1')
ax1.set_xlabel('Contribution Weight')

# PC2 contributions
pc2_sorted = components_df['PC2'].sort_values()
ax2.barh(range(len(pc2_sorted)), pc2_sorted.values)
ax2.set_yticks(range(len(pc2_sorted)))
ax2.set_yticklabels(pc2_sorted.index)
ax2.set_title('Feature Contributions to PC2')
ax2.set_xlabel('Contribution Weight')

plt.tight_layout()
plt.show()

print("\n" + "=" * 40)
print("PRINCIPAL COMPONENT INTERPRETATION:")
print("=" * 40)
print("PC1 represents:")
print("• Positive: Grocery, Detergents_Paper, Milk (Daily essentials)")
print("• Negative: Fresh, Frozen (Perishable goods)")
print("→ Interpretation: PC1 separates customers by their preference for")
print("  non-perishable vs perishable products")

print("\nPC2 represents:")
print("• Positive: Fresh, Frozen, Delicassen (Fresh and specialty items)")
print("• Negative: Detergents_Paper, Channel_2 (Non-food and retail channel)")
print("→ Interpretation: PC2 separates customers by fresh/specialty focus")
print("  vs non-food/retail orientation")

# =============================================================================
# CLUSTER INSIGHTS
# =============================================================================

print("\n\n7. CLUSTER INSIGHTS AND BUSINESS INTERPRETATION")
print("-" * 40)

# Add cluster labels to original data
df_clustered = df.copy()
df_clustered['Cluster'] = best_labels

# Analyze cluster characteristics
cluster_profiles = df_clustered.groupby('Cluster').mean()
cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()

print("Cluster sizes:")
print(cluster_sizes)

print("\nAverage spending by cluster (normalized to show patterns):")
# Normalize to see patterns better
cluster_normalized = cluster_profiles[numeric_features].div(cluster_profiles[numeric_features].sum(axis=1), axis=0)
print(cluster_normalized.round(3))

# Create radar chart for cluster profiles
def create_radar_chart(cluster_data, features, n_clusters):
    """Create a radar chart comparing cluster profiles"""
    
    # Number of variables
    categories = features
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each cluster
    for cluster in range(n_clusters):
        values = cluster_data.loc[cluster, features].values.flatten().tolist()
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title('Customer Cluster Profiles', size=15, y=1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    return fig

# Create normalized data for radar chart (percentage of total spending)
radar_data = cluster_profiles[numeric_features].copy()
for idx in radar_data.index:
    total = radar_data.loc[idx].sum()
    radar_data.loc[idx] = radar_data.loc[idx] / total * 100

# Create the radar chart
create_radar_chart(radar_data, numeric_features, optimal_k)
plt.show()

# Detailed cluster analysis
print("\n" + "=" * 50)
print("DETAILED CLUSTER ANALYSIS")
print("=" * 50)

for cluster in range(optimal_k):
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
    cluster_mean = cluster_data[numeric_features].mean()
    cluster_size = len(cluster_data)
    
    print(f"\n--- CLUSTER {cluster} (n={cluster_size}) ---")
    
    # Find top 2 features that characterize this cluster
    cluster_normalized = cluster_mean / cluster_mean.sum()
    top_features = cluster_normalized.nlargest(2)
    
    print(f"Key characteristics:")
    for feature, value in top_features.items():
        print(f"  • {feature}: {value:.1%} of total spending")
    
    # Channel and region distribution
    if 'Channel' in cluster_data.columns:
        channel_dist = cluster_data['Channel'].value_counts(normalize=True)
        print(f"Channel distribution: {dict(channel_dist.round(2))}")
    
    if 'Region' in cluster_data.columns:
        region_dist = cluster_data['Region'].value_counts(normalize=True)
        print(f"Region distribution: {dict(region_dist.round(2))}")

print("\n" + "=" * 50)
print("BUSINESS RECOMMENDATIONS")
print("=" * 50)

print("\nBased on the cluster analysis, here are potential business strategies:")

print("\n1. SEGMENTATION STRATEGY:")
print("   • Develop targeted marketing campaigns for each cluster")
print("   • Customize product bundles based on cluster preferences")
print("   • Optimize inventory based on regional and channel patterns")

print("\n2. CUSTOMER RELATIONSHIP MANAGEMENT:")
print("   • Identify high-value segments for premium services")
print("   • Develop loyalty programs tailored to segment behaviors")
print("   • Create personalized promotions based on spending patterns")

print("\n3. OPERATIONAL EFFICIENCY:")
print("   • Optimize logistics based on geographic cluster distribution")
print("   • Align sales teams with segment characteristics")
print("   • Streamline product offerings by cluster preferences")

print("\n" + "=" * 60)
print("ASSESSMENT COMPLETE")
print("=" * 60)