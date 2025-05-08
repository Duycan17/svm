import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 1. Read the Excel file
df = pd.read_excel('housing_with_property_tax.xlsx')

# 2. Set target and features
target_col = 'median_house_value'
X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Handle missing values (drop rows with any missing values)
X = X.dropna()
y = y[X.index]

# 4. Encode categorical column 'ocean_proximity'
categorical_features = ['ocean_proximity']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# 5. Reduce dimension (PCA)
pca = PCA(n_components=0.95, random_state=42)  # retain 95% variance
X_reduced = pca.fit_transform(X_processed)

# Plot PCA explained variance ratio
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.tight_layout()
plt.show()

# 6. Unsupervised learning (clustering)
n_clusters = 5  # or choose based on domain knowledge
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_reduced)

# Plot KMeans cluster distribution
plt.figure(figsize=(6, 4))
plt.hist(cluster_labels, bins=np.arange(n_clusters+1)-0.5, rwidth=0.8)
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.title('KMeans Cluster Distribution')
plt.xticks(range(n_clusters))
plt.tight_layout()
plt.show()

# 7. Augment reduced data with cluster info
X_augmented = np.hstack([X_reduced, cluster_labels.reshape(-1, 1)])

# 8. Supervised learning (regression)
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y, test_size=0.2, random_state=42)
reg = SVR()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# 9. Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Pipeline 5 - Test MSE: {mse:.4f}")

# Plot SVR predictions vs true values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('SVR Predictions vs True Values')
plt.tight_layout()
plt.show()
