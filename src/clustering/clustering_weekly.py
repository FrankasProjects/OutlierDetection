from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
import pandas as pd
from clustering.cluster_execution import calculateClusters
from dbhandler.fetch_data import get_pm_df_week
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta


localCursor = None
localCon = None
sd_pm = []
mad_pm = []

def createConnections():
    global localCon
    localCon = openConLocal()
    global localCursor
    localCursor = localCon.cursor()


def closeConnections():
    closeLocal(localCursor)


"""starting method"""
"""Clusters the dataframe into 1-20 clusters, and calculates statistics for each cluster formation"""
def std_clustering():
    createConnections()
    # Using the cleaned dataset
    df = get_pm_df_week(localCursor)
    df_grouped = df.drop_duplicates(subset='sensor_id', keep="last")
    x = 1
    K = 20

    df_result = pd.DataFrame()

    # Clusters the dataframe into x clusters and calculates the std and mad
    while x <= K:
        print("Clustering for x = ", x)
        # returns clustered dataframe
        df_clustered = calculateClusters(df_grouped, x)
        df_count =  df_clustered['cluster'].value_counts()
        s = df_count.mean()

        # merges dataframe, so that each produced measurement is assigned to a cluster, based on
        # the location of its sensor
        df_red = df_clustered.drop(['lat', 'lon', 'timestamp', 'p2', 'average_distance_to_cluster_members'], axis=1)
        df_merged = pd.merge(df, df_red, on='sensor_id')

        global sd_pm
        sd_pm.append(calc_mean_std_pm_weekly(df_merged, x))
        global mad_pm
        mad_pm.append(calc_mmad_pm_weekly(df_merged, x))
        print("Finished clustering into ", x, " clusters!")
        x = x + 1


    # selecting and calculating number, statistics and columns for final table
    df_sd = pd.DataFrame(sd_pm)
    df_sd['max_value'] = df_sd.max(axis=1)
    df_sd['mean'] = df_sd.mean(axis=1)
    df_result['max_std'] = df_sd['max_value']
    df_result['mean_std'] = df_sd['mean']

    df_mad = pd.DataFrame(mad_pm)
    df_mad['max_value'] = df_mad.max(axis=1)
    df_mad['mean'] = df_mad.mean(axis=1)
    df_result['max_mad'] = df_sd['max_value']
    df_result['mean_mad'] = df_sd['mean']


    df_cluster_sd = pd.DataFrame({'cluster': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                                           '18', '19', '20']})

    df_cluster_sd['mean'] = df_sd['mean']
    df_cluster_sd['max'] = df_sd['max_value']

    df_cluster_sd.plot(x='cluster', y='mean', kind='line')
    plt.show()

    df_cluster_sd.plot(x='cluster', y='max', kind='line')
    plt.show()
    print(df_result.to_latex(index=False, multirow=True))

    closeConnections()

def calc_mean_std_pm_weekly(df_clustered, cluster_nr):
    sd_array_pm2 = []
    z = 0
    while z < cluster_nr:
        start_time = datetime(year=2019, month=9, day=26, hour=00, minute=00, second=00)
        end_time = datetime(year=2019, month=10, day=2, hour=23, minute=59, second=59)
        df_clustered_temp = df_clustered.loc[df_clustered['cluster'] == z]
        sd_temp = []
        while start_time < end_time:
            mid_time = start_time + timedelta(days=1)
            df_interval = df_clustered_temp[
                (df_clustered_temp['timestamp'] >= start_time) & (df_clustered_temp['timestamp'] < mid_time)]
            if not df_interval.empty:
                sd_temp.append(calc_std_pm2(df_interval))
           # else:
               # print("Interval between ", start_time, " ", mid_time, " is empty. Check if correct")

            start_time = mid_time

        if not sd_temp:
            print("empty")
        else:
            sd_array_pm2.append(calc_avg(sd_temp))
        z = z + 1
    return sd_array_pm2


def calc_std_pm2(df):
    return df.std()['p2']



def calc_avg(lst):
    return sum(lst) / len(lst)


def calc_mmad_pm_weekly(df_clustered,cluster_nr):
    mad = []
    z = 0
    while z < cluster_nr:
        start_time = datetime(year=2019, month=9, day=26, hour=00, minute=00, second=00)
        end_time = datetime(year=2019, month=10, day=2, hour=23, minute=59, second=59)
        df_clustered_temp = df_clustered.loc[df_clustered['cluster'] == z]
        mad_temp = []
        while start_time < end_time:
            mid_time = start_time + timedelta(days = 1)
            df_interval = df_clustered_temp[(df_clustered_temp['timestamp'] > start_time) & (df_clustered_temp['timestamp'] <= mid_time)]
            if not df_interval.empty:
                mad_temp.append(calc_mad_pm2(df_interval))
            #else:
                #print("Interval between ", start_time, " ", mid_time, " is empty. Check if correct")
            start_time = mid_time
        if not mad_temp:
            print("empty")
        else:
            mad.append(calc_avg(mad_temp))
        z = z+1
    return calc_avg(mad)

def calc_mad_pm2(df_interval):
    pm_col = df_interval.loc[:,'p2']
    return pm_col.mad()


