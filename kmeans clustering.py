import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_dist(x1, y1, x2, y2):
    return (((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) ** 0.5)


data = pd.read_csv("Points.csv")

points = np.array(data)

plt.scatter(points[:,0],points[:,1])
plt.title('1. Points.csv data plot in graph')
plt.show()

centroids = np.array([[40, 40], [100, 0], [0, 100]]) #intialized centroid

plt.scatter(points[:,0],points[:,1])
plt.scatter(centroids[:,0],centroids[:,1], c='y', s=100)
plt.title('2.Random initialize K cluster centers, [(40, 40), (100, 0), and (0, 100)] illustrate in yellow dots')
plt.show()

K = 3

m = points.shape[0]
idx = np.zeros(m)

# Assign each data points to the closest center
for i in range(m):
    temp = np.zeros(K)
    for j in range(K):
        temp[j] = np.sum((points[i,:] - centroids[j,:]) ** 2)  #Calculatung the euclidean_distance
        idx[i] = np.argmin(temp)

# Finding the new centroid and plot the final graph
colors = ['r', 'g', 'b']
fig, ax = plt.subplots()

for i in range(K):
    clusters = np.array([points[j] for j in range(len(points)) if idx[j] == i])
    centroids[i] = np.mean(clusters, axis=0)
    ax.scatter(clusters[:, 0], clusters[:, 1], c=colors[i])
ax.scatter(centroids[:,0],centroids[:,1], c='y', s=100)
plt.title('3. Final cluster centre by recompute the cluster centers and data belongs until result converge')
plt.show()



