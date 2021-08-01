import pandas as pd
from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
from pandas import DataFrame
from dbhandler.fetch_data import get_pm_df
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

# Adopted from
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

#variables holding database connection
localCursor = None
localCon = None



def createConnections():
    global localCon
    localCon = openConLocal()
    global localCursor
    localCursor = localCon.cursor()


def closeConnections():
    closeLocal(localCursor)


"""starting method"""
def pm_clustering_silhouette():
    createConnections()
    df = get_pm_df(localCursor)
    # Since it is a location-based clustering, the dataset is reduced to
    # one row per sensor (which actively produces measurements)
    df_grouped = df.drop_duplicates(subset='sensor_id', keep="last")
    df_result = pd.DataFrame()
    # Number of cluster formations, for which the silhouette score should be calculated
    x = 2
    K = 20
    score = []

    # calculates score for each cluster-formation
    while x <= K:
        score.append(calculate_Clusters_silhouette(df_grouped, x))
        x = x+1

    #structures data to create silhouette plot and table
    df_cluster = pd.DataFrame({'cluster': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                                           '15', '16', '17', '18', '19', '20']})

    df_wss = DataFrame(score, columns=['score'])
    df_wss['cluster'] = df_cluster
    ax = plt.gca()

    df_wss.plot(kind='line', x='cluster', y='score', ax=ax)
    plt.xlabel('Cluster')
    plt.ylabel('Silhouette Score')

    #plt.savefig('Silhouette', format='pdf')
    plt.show()

    df_wss.to_latex()
    print(df_wss.to_latex(index=False, multirow=True))

    closeConnections()


def calculate_Clusters_silhouette(dataframe, cluster_nr):

    pm_sensors = dataframe
    df = pd.DataFrame(pm_sensors, columns=['sensor_id', 'lat', 'lon', 'p2', 'timestamp'])
    coord = df.iloc[:, [1, 2]].values

    # executes agglomerative clustering for the cluster_nr
    aggloclust=AgglomerativeClustering(n_clusters=cluster_nr, affinity='euclidean', linkage='average').fit(coord)
    # print(aggloclust)

    labels = aggloclust.labels_
    # calculates silhouette score
    score = silhouette_score(coord, aggloclust.labels_, metric='euclidean')

    return score
