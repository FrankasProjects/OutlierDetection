from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
from datetime import datetime
from datetime import timedelta
from dbhandler.fetch_data import get_pm_df
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

"STARTING METHOD"
def clean_data():
    max = pd.DataFrame()
    createConnections()
    df = get_pm_df(localCursor)
    print("Loaded initial dataset with: ", len(df), " data instances")
    #1. Remove values <= 0
    df_2 = df[df.p2 > 0]
    removed = (len(df) - len(df_2))
    percentage = (removed / len(df)) * 100

    print(removed, " or ", round(percentage,2), "% data instances were removed, due to their value being smaller or equal to zero.")

    max = calc_max(df_2)
    df_result = calc_mean_per_hour(df_2)
    removed_2 = (len(df_2) - len(df_result))
    percentage_2 = (removed / len(df_2)) * 100

    print(removed_2, " or ", round(percentage_2,2), "%  data instances were removed, because they were higher than the threshold of ", max )

    print("Final dataset contains: ", len(df_result), " data instances.")
    # writes values to table: ldi_stuttgart.pm_sensors_week_cleaned
    # execute_values(localCon, localCursor, df_result, 'ldi_stuttgart.pm_sensors_week_cleaned')

"""
Calculates & returns the maximum value, in a 60 min time interval
"""
def calc_max(df):
    df_result = []
    start_time = datetime(year=2019, month=9, day=26, hour=00, minute=00, second=00)
    end_time = datetime(year=2019, month=9, day=28, hour=23, minute=59, second=59)

    while start_time < end_time:
        mid_time = start_time + timedelta(minutes=60)
        df_interval = df[
            (df['timestamp'] >= start_time) & (df['timestamp'] < mid_time)]
        if not df_interval.empty:
            # calc mean value
            df_result.append(df_interval['p2'].max())

        start_time = mid_time

    return df_result



"""
Calculates and removes values > 3 * mean value per hour
"""
def calc_mean_per_hour(df_2):
    removed_columns = pd.DataFrame()
    df_result = pd.DataFrame()
    mean_list = []

    start_time = datetime(year=2019, month=9, day=26, hour=00, minute=00, second=00)
    end_time = datetime(year=2019, month=9, day=28, hour=23, minute=59, second=59)

    while start_time < end_time:
        mid_time = start_time + timedelta(minutes=60)
        df_interval = df_2[
                (df_2['timestamp'] >= start_time) & (df_2['timestamp'] < mid_time)]
        if not df_interval.empty:
            # calc mean value
            mean = df_interval['p2'].mean()
            mean_list.append(mean)
            threshold = mean * 3
            removed_columns = pd.concat([removed_columns, df_interval[df_interval.p2 > threshold]])
            df_interval = df_interval[df_interval.p2 < threshold]
            df_result = pd.concat([df_result, df_interval])

        start_time = mid_time
        maxi = pd.DataFrame(mean_list)
        maax = maxi.max()
        removed_columns['p2'].mean()
        removed_columns['p2'].count()

    removed_columns['p2'].mean()

    return df_result




