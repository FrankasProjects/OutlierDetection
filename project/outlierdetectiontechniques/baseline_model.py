
from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
from dbhandler.fetch_data import get_pm_df_clustered
import pandas as pd
import numpy as np

#Adopted From
# https://www.statology.org/z-score-python/

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
def calc_baseline():
    outliers = []
    createConnections()
    df = get_pm_df_clustered(localCursor)
    z = 0
    cluster_nr = 11
    df_result = pd.DataFrame()

    while z < cluster_nr:
        anomalies = []
        df_clustered_temp = df.loc[df['cluster'] == z]
        df_clustered_temp['outlier'] = 0
        minimum = df_clustered_temp['p2'].min()
        df_clustered_temp['p2'].hist(grid=False,
                figsize=(10, 6),
                bins=30)
        # plt.show()
        print(minimum)
        # - minimum according to the approach of van Zoest et. al
        df_clustered_temp.insert(len(df_clustered_temp.columns), 'A_Sqrt',
                  np.sqrt(df_clustered_temp['p2'] - minimum))
        df_clustered_temp['A_Sqrt'].hist(grid=False, color='green',
                          figsize=(10, 6), bins=40)
        #plt.show()
        # Set upper and lower limit to 3 standard deviation
        data_std = df_clustered_temp['A_Sqrt'].std()
        data_mean = df_clustered_temp['A_Sqrt'].mean()
        anomaly_cut_off = data_std * 3

        lower_limit = data_mean - anomaly_cut_off
        upper_limit = data_mean + anomaly_cut_off
        print(lower_limit)
        df_clustered_temp['outlier'][df_clustered_temp.A_Sqrt > upper_limit ] = 1
        df_clustered_temp['outlier'][df_clustered_temp.A_Sqrt < lower_limit ] = 1
        for outlier in df_clustered_temp['A_Sqrt']:
            if outlier > upper_limit or outlier < lower_limit:
                anomalies.append(outlier)
        df_clustered_temp.outlier.value_counts()
        outliers.append(anomalies)
        df_result = pd.concat([df_result, df_clustered_temp])
        z = z + 1
    df_result.outlier.value_counts()

    print("Calculation finished")
    print("0: Normal Data instances")
    print("1: Outliers")
    print( df_result.outlier.value_counts())

    # database table is already filled
    # execute_values(localCon, localCursor, df_result, 'ldi_stuttgart.pm_sensors_clustered_labeled_final')
