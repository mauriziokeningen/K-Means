import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as skl

# Load data from the CSV file using the Pandas library
dataframe = pd.read_csv('CPUs.csv')

# Remove the " GHz" unit from the frequency values and convert to float type
dataframe['Frequency'] = dataframe['Frequency'].str.replace(' GHz', '').astype(float)

# Select only the frequency and price features for the dataset
X = dataframe[['Frequency', 'Price']]

# Number of clusters (K)
k = 5

# Initialize the clustering algorithm with the specified K value
kmeansModel = skl.KMeans(n_clusters=k)

# Fit the data
kmeansModel.fit(X)

# Get the centroids
centroids = kmeansModel.cluster_centers_

# Get a list of data labels
labels = kmeansModel.predict(X)

# Add a classification label column to the dataframe
dataframe['Labels'] = labels

# Color table
colors = ['red', 'orange', 'green', 'pink', 'blue']

data_colors = []
centroid_colors = []

for label in labels:
    data_colors.append(colors[label])

for i in range(len(centroids)):
    centroid_colors.append(colors[i])

# Scatter plot
ax = plt.axes()
ax.scatter(dataframe['Frequency'], dataframe['Price'], c=data_colors, marker='o', s=40)
ax.scatter(centroids[:, 0], centroids[:, 1], c=centroid_colors, marker='+', s=200)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Price (MXN)')
plt.title('K-Means Clustering of Processors')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()

dataframe.to_csv('grouped-processors.csv', index=False)
