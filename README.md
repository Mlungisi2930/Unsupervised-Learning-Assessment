# Wholesale Customers Clustering Analysis

# 📊 Project Overview
This project performs an unsupervised learning analysis on the Wholesale Customers dataset to identify distinct customer segments based on their purchasing patterns. The analysis combines dimensionality reduction (PCA) with clustering techniques (K-Means) to uncover meaningful customer groups that can inform business strategies.

# 🎯 Business Objective
The goal is to segment wholesale customers into distinct groups based on their spending across different product categories, enabling targeted marketing, optimized inventory management, and improved customer relationship management.

# 📁 Dataset
* Source: UCI Machine Learning Repository

* Dataset: Wholesale Customers Data

* Records: 440 customers

* Features: 8 variables

Original Features:
* Channel: Categorical (1: Horeca, 2: Retail)

* bRegion: Categorical (1: Lisbon, 2: Oporto, 3: Other)

* Continuous Spending Features:

*Fresh

*Milk

*Grocery

*Frozen

*Detergents_Paper

*Delicassen

# 🛠️ Methodology
1. Data Preprocessing
   
* One-hot encoding for categorical variables (Channel, Region)

* Standardization of all features using StandardScaler

* Missing values analysis (none found in original dataset)

2. Dimensionality Reduction (PCA)
* Principal Component Analysis to reduce feature space

* Scree plot and cumulative variance analysis

* Selected components capturing 85% of variance

* Component interpretation and feature contribution analysis

3. Clustering Analysis
K-Means clustering with optimal k selection

* Evaluation using:

* Elbow method (inertia)

* Silhouette analysis

* Cluster stability assessment

* Comparison between original vs PCA-reduced data

4. Cluster Interpretation
Detailed profile analysis for each cluster

Radar charts for visual comparison

Business insights and recommendations

# 📈 Key Results
Optimal Configuration
Optimal Clusters: Determined through silhouette analysis

PCA Components: Selected to retain 85% variance

Best Performance: Comparison between raw and PCA-reduced data

Principal Component Interpretation
PC1: Separates customers by preference for non-perishable vs perishable products

PC2: Distinguishes fresh/specialty focus vs non-food/retail orientation

Customer Segments
* The analysis identifies distinct customer segments with unique purchasing patterns, enabling:

* Targeted marketing strategies

* Customized product bundles

* Optimized inventory management

# 🚀 Installation & Usage
Prerequisites
bash
Python 3.7+
pip install pandas numpy matplotlib seaborn scikit-learn
Running the Analysis
python
python Main.py

# 📂 Project Structure
text
├── Main.py                 # Main analysis script
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

# 📊 Visualizations
The analysis generates several key visualizations:

* PCA Scree Plot - Variance explained by components

* Elbow Curve & Silhouette Scores - Cluster evaluation

* Silhouette Plots - Cluster quality assessment

* 2D Cluster Visualization - Customer segments

* bRadar Charts - Cluster profile comparisons

* Feature Contribution Plots - PCA interpretation

# 💡 Business Applications
Marketing Strategy
* Develop targeted campaigns for each customer segment

* Create personalized promotions based on spending patterns

* Optimize customer acquisition strategies

Operations Optimization
* Inventory management based on segment preferences

* Logistics optimization using geographic patterns

* Product assortment alignment with cluster characteristics

Customer Relationship Management
* Identify high-value segments for premium services

* Develop tailored loyalty programs

* Improve customer retention through personalized engagement

# 🔍 Technical Insights
Model Performance
* Comprehensive evaluation of clustering quality

* Comparison between dimensionality reduction approaches

* Robustness analysis through multiple validation techniques

Methodological Strengths
* Systematic approach to parameter selection

* Multiple validation techniques

* Comprehensive business interpretation

* Reproducible analysis pipeline

# 📝 Conclusion
This project demonstrates a complete unsupervised learning pipeline from data preprocessing to business insights. The identified customer segments provide actionable intelligence for wholesale business optimization, enabling data-driven decision making across marketing, operations, and customer relationship management.

# 🤝 Contributing
Feel free to fork this project and contribute by submitting pull requests for additional features or improvements.

# 📄 License
This project is intended for educational and analytical purposes. Dataset sourced from UCI Machine Learning Repository.
