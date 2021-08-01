import psycopg2
from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
from dbhandler.insert_data import execute_values
from sklearn.cluster import AgglomerativeClustering
from pandas import DataFrame
import pandas as pd

localCursor = None
localCon = None

"""Supporting Methods to read and write data from the database."""


def createConnections():
    global localCon
    localCon = openConLocal()
    global localCursor
    localCursor = localCon.cursor()


def closeConnections():
    closeLocal(localCursor)


def fetchData(localCursor, select_query, column_names):

    localCursor.itersize = 1000  # chunk size
    try:
        localCursor.execute(select_query)
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        localCursor.close()
        return 1

    tupples = localCursor.fetchall()

    df = pd.DataFrame(tupples, columns=column_names)
    return df


def fetchStations(localCursor):
    column_names_station = ["station_id", "latitude",
                            "longitude", "altitude", "geopoint", "pm_sensor_id", "pm_sensor_type", "hum_sensor_id",
                            "hum_sensor_type", "distance_to_stuttgart_center"]
    df = fetchData(localCursor, "select * from ldi_stuttgart.stuttgart_ldi_stations ; ", column_names_station)
    return df



def get_hum_df(localCursor):
    column_names_hum = ["sensor_id", "sensor_type",
                        "location", "lat", "lon", "timestamp", "pressure", "altitude", "pressure_sealevel",
                        "temperature", "humidity"]
    df = fetchData(localCursor,
                   "select * from ldi_stuttgart.hum_sensors ; ", column_names_hum)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df

def get_values_statistics(localCursor):
    column_names_pm = ["sensor_id", "sensor_type",
                       "location", "lat", "lon", "timestamp", "p1", "p2"]
    df = fetchData(localCursor, "select * from ldi_stuttgart.stuttgart_pm_sensors; ", column_names_pm)
    # df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df

def get_pm_df(localCursor):
    column_names_pm = ["sensor_id", "sensor_type",
                       "location", "lat", "lon", "timestamp", "p1", "p2"]
    df = fetchData(localCursor, "select * from ldi_stuttgart.pm_sensors_week; ", column_names_pm)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df

def get_pm_df_cleaned(localCursor):
    column_names_pm = ["sensor_id", "sensor_type",
                       "location", "lat", "lon", "timestamp", "p1", "p2"]
    df = fetchData(localCursor, "select * from ldi_stuttgart.pm_sensors_week; ", column_names_pm)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df



def get_pm_df_clustered(localCursor):
    column_names_pm = ["sensor_id", "lat",
                       "lon", "cluster", "average_distance_to_cluster_members", "p2", "timestamp", "outlier", "a_sqrt", "id"]
    df = fetchData(localCursor, "select * from ldi_stuttgart.pm_sensors_clustered_labeled_final ; ", column_names_pm)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df


def get_pm_df_week(localCursor):
    column_names_pm = ["sensor_id", "sensor_type",
                       "location", "lat", "lon", "timestamp", "p1", "p2"]
    df = fetchData(localCursor, "select * from ldi_stuttgart.pm_sensors_week_cleaned ; ", column_names_pm)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df


def calcBestDay():
    createConnections()
    column_names_hum = ["sensor_id", "sensor_type",
                        "location", "lat", "lon", "timestamp", "p1", "p2"]
    df = fetchData(localCursor,
                   "select * from ldi_stuttgart.stuttgart_pm_sensors where timestamp::date between date '2019-09-26' and date '2019-09-26'; ",
                   column_names_hum)

    execute_values(localCon, localCursor, df, 'ldi_stuttgart.pm_sensors')


def calcBestWeek():
    createConnections()
    column_names_pm = ["sensor_id", "sensor_type",
                       "location", "lat", "lon", "timestamp", "p1", "p2"]
    df = fetchData(localCursor,
                   "select * from ldi_stuttgart.stuttgart_pm_sensors where timestamp::date between date '2019-09-26' and date '2019-09-28'; ",
                   column_names_pm)

    execute_values(localCon, localCursor, df, 'ldi_stuttgart.pm_sensors_week')


def calcHumidity():
    createConnections()
    column_names_hum = ["sensor_id", "sensor_type",
                        "location", "lat", "lon", "timestamp", "pressure", "altitude", "pressure_sealevel",
                        "temperature", "humidity"]
    df = fetchData(localCursor,
                   "select * from ldi_stuttgart.stuttgart_hum_sensors where timestamp::date between date '2019-12-13' and date '2019-12-15'; ",
                   column_names_hum)
    execute_values(localCon, localCursor, df, 'ldi_stuttgart.hum_sensors_week')

def writeClustersToDB():
    createConnections()
    df = get_pm_df(localCursor)
    df_grouped = df.drop_duplicates(subset='sensor_id', keep="last")
    df_clustered = calculateClusters(df_grouped, 11)

    df_red = df_clustered.drop(['lat', 'lon', 'timestamp', 'p2', ], axis=1)
    df_merged = pd.merge(df, df_red, on='sensor_id')
    df_merged = df_merged[['sensor_id', 'lat', 'lon', 'cluster', 'average_distance_to_cluster_members', 'p2', 'timestamp']]

    execute_values(localCon, localCursor, df_merged, 'ldi_stuttgart.pm_sensors_clustered')



def calculateClusters(dataframe, cluster_nr):
    pm_sensors = dataframe
    print(pm_sensors)

    df = pd.DataFrame(pm_sensors, columns =['sensor_id', 'lat', 'lon', 'p2', 'timestamp'])

    coord = df.iloc[:, [1, 2]].values

    all_lat_coordinates = df.iloc[:, [1]].values
    all_lon_coordinates = df.iloc[:, [2]].values


    aggloclust=AgglomerativeClustering(n_clusters=cluster_nr, affinity='euclidean', linkage='average').fit(coord)
    #print(aggloclust)

    labels = aggloclust.labels_
    labels = list(labels)
    df_labels = DataFrame(labels)

    df["cluster"] = labels

    # insert distance calculations)
    df["average_distance_to_cluster_members"] = labels

    return df