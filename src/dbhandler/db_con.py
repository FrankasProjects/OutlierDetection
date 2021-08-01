import psycopg2
import os
from dotenv import load_dotenv

# Adopted from
# https://kb.objectrocket.com/postgresql/how-to-install-psycopg2-in-windows-1460
# https://www.learnpython.org/en/Pandas_Basics
# https://towardsdatascience.com/how-to-handle-large-datasets-in-python-with-pandas-and-dask-34f43a897d55


con = None
myCon = None

"""Nothing to fill in here. Fill in credentials in .env """
def openConLocal():
    load_dotenv()
    try:
        global myCon
        myCon = psycopg2.connect(user=os.getenv('user'),
                                 password=os.getenv('password'),
                                 host=os.getenv('host'),
                                 port=os.getenv('port'),
                                 database=os.getenv('database'))

        cursorLocal = myCon.cursor()
        # Print PostgreSQL Connection properties
        print(myCon.get_dsn_parameters(), "\n")

        # Print PostgreSQL version
        cursorLocal.execute("SELECT version();")
        record = cursorLocal.fetchone()
        print("You are connected to - ", record, "\n")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    return myCon


def close(cursorLocal):
    if(myCon):
        cursorLocal.close()
        myCon.close()
        print("PostgreSQL connection is closed")


def closeLocal(cursor):
    if(con):
        cursor.close()
        con.close()
        print("Local PostgreSQL connection is closed")
