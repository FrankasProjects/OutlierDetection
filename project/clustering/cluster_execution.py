from sklearn.cluster import AgglomerativeClustering
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt

""" Main Clustering Method """

# Adapted from
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# https://scikit-learn.org/stable/modules/clustering.html
# https://www.techladder.in/article/hierarchical-clustering-algorithm-python
# https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019

"Clusters a dataframe into the number of clusters and plots the result in a colorful scatterplot "
def calculateClusters(dataframe, cluster_nr):
    pm_sensors = dataframe

    df = pd.DataFrame(pm_sensors, columns =['sensor_id', 'lat', 'lon', 'p2', 'timestamp'])

    coord = df.iloc[:, [1, 2]].values

    all_lat_coordinates = df.iloc[:, [1]].values
    all_lon_coordinates = df.iloc[:, [2]].values

    #plt.scatter(all_lat_coordinates, all_lon_coordinates)
    #plt.show()

    aggloclust=AgglomerativeClustering(n_clusters=cluster_nr, affinity='euclidean', linkage='average').fit(coord)


    labels = aggloclust.labels_
    labels = list(labels)
    df_labels = DataFrame(labels)
    labels_grouped = df_labels.drop_duplicates(keep="last")

    df["cluster"] = labels

    # just a filler
    df["average_distance_to_cluster_members"] = labels

    # plots colorful clustering

    #plt.scatter(all_x_coordinates,all_y_coordinates, c=labels)
    plt.scatter(all_lat_coordinates, all_lon_coordinates, c=labels, cmap="Set3")
    cbar = plt.colorbar()
    cbar.set_ticks(range(1,12))
    cbar.set_ticklabels(range(1,12))
    cbar.set_label("Cluster")
    #plt.legend(labels_grouped)

    plt.show()
    return df
