from dbhandler.fetch_data import fetchData
from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal

def fetchPM():
    localCon = openConLocal()
    localCursor = localCon.cursor()

    column_names_pm = ["sensor_id", "sensor_type",
                           "location", "lat", "lon", "timestamp", "p1", "p2"]
    df = fetchData(localCursor, "select * from ldi_stuttgart.pm_sensors ;", column_names_pm)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    closeLocal(localCursor)
    return df
