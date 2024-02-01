import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
# Load the data
df = pd.read_csv('/users/mishellebitman/Desktop/Assignment3/Mall_Customers.csv')

# Convert Gender to numerical values
gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

# Standardize Annual Income, Spending Score and age data
scaler = StandardScaler()
df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']] = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']])

# Feature Selection: Annual Income and Spending Score for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Employing the Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the WCSS to find the elbow
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
plt.title('Elbow Method for Income and Spending Score')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
#plt.show()

#the selected cluster number based on elbow method is 5

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
clusters = kmeans.fit_predict(X)

df['Cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, hue='Cluster', palette='viridis')
plt.title('Clusters of Customers Based on Income and Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()


fig = plt.figure(figsize=(20, 8))

# For the scatter plot of Clusters according to Annual Income
ax1 = fig.add_subplot(121)
sns.swarmplot(x='Cluster', y='Annual Income (k$)', data=df, hue='Cluster', ax=ax1, palette='viridis')
ax1.set_title('Clusters According to Annual Income')
ax1.legend().remove() 

# For the scatter plot of Clusters according to Spending Score
ax2 = fig.add_subplot(122)
sns.swarmplot(x='Cluster', y='Spending Score (1-100)', data=df, hue='Cluster', ax=ax2, palette='viridis')
ax2.set_title('Clusters According to Spending Score')
ax2.legend().remove() 

plt.show()

silhouette_avg = silhouette_score(X, clusters)
print(f'The average silhouette score is : {silhouette_avg}')

