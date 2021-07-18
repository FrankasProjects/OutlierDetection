import psycopg2.extras as extras
import psycopg2

def execute_values(localCon, localCursor, df, table):
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    try:
        extras.execute_values(localCursor, query, tuples)
        localCon.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        localCon.rollback()
        localCursor.close()
        return 1
    print("execute_values() done")
    localCursor.close()