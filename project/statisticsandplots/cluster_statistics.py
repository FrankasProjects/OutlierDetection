from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
from dbhandler.fetch_data import get_pm_df_clustered
import pandas as pd
localCursor = None
localCon = None


def createConnections():
    global localCon
    localCon = openConLocal()
    global localCursor
    localCursor = localCon.cursor()


def closeConnections():
    closeLocal(localCursor)

"""Starting Method"""
def calc_statistics():
    createConnections()
    df = get_pm_df_clustered(localCursor)
    statistics = pd.DataFrame()

    z = 0
    cluster_nr = 11

    # Calculates Mininum, Maximum, Mean values, standard deviation and mad per cluster
    while z < cluster_nr:
        temp_statistics = []
        df_clustered_temp = df.loc[df['cluster'] == z]
        #number of sensors
        temp_statistics.append(df_clustered_temp['sensor_id'].nunique())
        #number of measurements
        temp_statistics.append(len(df_clustered_temp))
        #min
        temp_statistics.append(df_clustered_temp['p2'].min())
        #max
        temp_statistics.append(df_clustered_temp['p2'].max())
        #mean
        temp_statistics.append(df_clustered_temp['p2'].mean())
        #std
        temp_statistics.append(df_clustered_temp['p2'].std())
        #mad
        temp_statistics.append(df_clustered_temp['p2'].mad())

        a_series = pd.Series(temp_statistics)

        statistics = statistics.append(a_series, ignore_index=True)
        # statistics = pd.concat([statistics, pd.DataFrame(temp_statistics)])
        z = z+1

    statistics.columns = ['sensor', 'measurements', 'min', 'max', 'mean', 'std', 'mad']
    statistics.to_latex()
    print(statistics.to_latex(index=False, multirow=True))