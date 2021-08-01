import psycopg2
from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
import psycopg2.extras as extras

# Adopted from
# https://naysan.ca/2020/05/09/pandas-to-postgresql-using-psycopg2-bulk-insert-performance-benchmark/

def writeDataframeToDatabase(df):
    table = 'ldi_stuttgart.pm_sensors_clustered'
    localCon = openConLocal()
    localCursor = localCon.cursor()
    print("Truncating Table")

    localCursor.execute("TRUNCATE TABLE ldi_stuttgart.pm_sensors_clustered")

    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query  = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)

    try:
        extras.execute_values(localCursor, query, tuples)
        localCon.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        localCon.rollback()
        localCursor.close()
        return 1
    print("execute_values() done")

    closeLocal(localCursor)


