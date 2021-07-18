from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
import pandas as pd
from dbhandler.fetch_data import get_pm_df
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Adopted from
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318


localCursor = None
localCon = None



def createConnections():
    global localCon
    localCon = openConLocal()
    global localCursor
    localCursor = localCon.cursor()


def closeConnections():
    closeLocal(localCursor)

"""STARTING METHOD """
def plot_dendrogram():
    createConnections()
    df = get_pm_df(localCursor)
    # Since it is a location-based clustering, the dataset is reduced to
    # one row per sensor (which actively produces measurements)
    df_grouped = df.drop_duplicates(subset='sensor_id', keep="last")
    df_grouped = pd.DataFrame(df_grouped, columns=['sensor_id', 'lat', 'lon', 'p2', 'timestamp'])

    coord = df_grouped.iloc[:, [1, 2]].values
    x = range(len(coord))

    # Dendrogram is calculated and plotted
    dendrogram = sch.dendrogram(sch.linkage(coord, method="average"),  leaf_rotation=90.)
    plt.title('Dendrogram')
    plt.xlabel('Sensor Number')
    plt.ylabel('Cluster Distances')
    ax = plt.gca()
    ax.set_xticklabels(x)
    plt.locator_params(axis='x', nbins=50)
    #plt.savefig('Dendrogram', format='pdf')
    plt.show()
