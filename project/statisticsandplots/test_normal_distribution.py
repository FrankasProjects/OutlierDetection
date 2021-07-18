from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot

from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
from dbhandler.fetch_data import get_pm_df_clustered
from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import anderson

localCursor = None
localCon = None

"""Includes different methods that were used to check if the clusters are normally distributed"""

def createConnections():
    global localCon
    localCon = openConLocal()
    global localCursor
    localCursor = localCon.cursor()


def closeConnections():
    closeLocal(localCursor)



def Shapiro_Wilk_Test():
    createConnections()
    df = get_pm_df_clustered(localCursor)
    z = 0
    cluster_nr = 11

    while z < cluster_nr:
        df_clustered_temp = df.loc[df['cluster'] == z]
        stat, p = shapiro(df_clustered_temp['p2'])
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
        z = z + 1
    closeConnections()


def Dagostino_Test():
    createConnections()
    df = get_pm_df_clustered(localCursor)
    z = 0
    cluster_nr = 11

    while z < cluster_nr:
        df_clustered_temp = df.loc[df['cluster'] == z]
        stat, p = normaltest(df_clustered_temp['p2'])
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
        z = z + 1
    closeConnections()

def Anderson_darling_test():
    createConnections()
    df = get_pm_df_clustered(localCursor)
    z = 0
    cluster_nr = 11

    while z < cluster_nr:
        df_clustered_temp = df.loc[df['cluster'] == z]
        result = anderson(df_clustered_temp['p2'])
        print('Statistic: %.3f' % result.statistic)
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        z = z + 1
    closeConnections()


def create_QQPlot():
    createConnections()
    df = get_pm_df_clustered(localCursor)
    z = 0
    cluster_nr = 11

    while z < cluster_nr:
        df_clustered_temp = df.loc[df['cluster'] == z]
        qqplot(df_clustered_temp['p2'], line='s')
        pyplot.show()
        z = z + 1
    closeConnections()

def create_histogram():
    createConnections()
    df = get_pm_df_clustered(localCursor)
    z = 0
    cluster_nr = 11

    while z < cluster_nr:
        df_clustered_temp = df.loc[df['cluster'] == z]
        pyplot.hist(df_clustered_temp['p2'])
        pyplot.show()
        z = z + 1
    closeConnections()



