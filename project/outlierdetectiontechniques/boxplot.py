import seaborn as sns
from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
import matplotlib.pyplot as plt
from dbhandler.fetch_data import get_pm_df_clustered
import numpy as np
import pandas as pd



#Adtoped from
# https://seaborn.pydata.org/generated/seaborn.boxplot.html
# https://stackoverflow.com/questions/10238357/finding-the-outlier-points-from-matplotlib-boxplot
# https://towardsdatascience.com/create-and-customize-boxplots-with-pythons-matplotlib-to-get-lots-of-insights-from-your-data-d561c9883643


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
def create_boxplot():
    createConnections()
    df = get_pm_df_clustered(localCursor)
    z = 0
    cluster_nr = 11
    df_result = pd.DataFrame()

    #Plots Boxplots of all 11 clusters with and without outliers
    sns.set_theme(style="whitegrid")

    ax = sns.boxplot(x="cluster", y="p2", data=df, showfliers=True)
    plt.xticks([0,1, 2, 3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10,11])
    plt.xlabel("Cluster")
    plt.ylabel("PM")
    #ax = sns.boxplot(x="cluster", y="p2", data=df, showfliers=False)
    #plt.savefig('BoxPlotTestWith', format='pdf')
    plt.show()

    ax = sns.boxplot(x="cluster", y="p2", data=df, showfliers=False)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    plt.xlabel("Cluster")
    plt.ylabel("PM")
    #plt.savefig('BoxPlotTestWithout', format='pdf')
    plt.show()

    general_stats = pd.DataFrame(columns = ['total', 'non', 'out', 'percentage', 'min', 'max'])
    outliers = pd.DataFrame()
    non_outliers = pd.DataFrame()

    # Calculates BoxPlots statistics (outliers, outlierrange, ...) for each cluster
    while z < cluster_nr:
        statistics = []
        df_clustered_temp = df.loc[df['cluster'] == z]
        df_clustered_temp['outlier'] = 0
        statistics.append(len(df_clustered_temp))

        Q1 = df_clustered_temp['p2'].quantile(0.25)
        Q3 = df_clustered_temp['p2'].quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.

        # non-outliers
        filter = (df_clustered_temp['p2'] >= Q1 - 1.5 * IQR) & (df_clustered_temp['p2'] <= Q3 + 1.5 * IQR)
        non = df_clustered_temp.loc[filter]
        statistics.append(len(non))
        non_outliers = pd.concat([non, non_outliers])

        # outliers
        bottom = (df_clustered_temp['p2'] < Q1 - 1.5 * IQR)
        top = (df_clustered_temp['p2'] > Q3 + 1.5 * IQR)
        df_clustered_temp['outlier'][df_clustered_temp.p2 > Q3 + 1.5 * IQR] = 1
        df_clustered_temp['outlier'][df_clustered_temp.p2 < Q1 - 1.5 * IQR] = 1
        top_outliers = df_clustered_temp.loc[top]
        bottom_outliers = df_clustered_temp.loc[bottom]
        temp_outliers = pd.concat([top_outliers , bottom_outliers])
        outliers = pd.concat([top_outliers, bottom_outliers, outliers])
        statistics.append(len(temp_outliers))
        percentage = len(temp_outliers) / len(df_clustered_temp)
        percentage = percentage * 100
        statistics.append(percentage)
        statistics.append(temp_outliers['p2'].min())
        statistics.append(temp_outliers['p2'].max())
        df_clustered_temp.outlier.value_counts()


        get_summary_statistics(df_clustered_temp)
        general_stats.loc[len(general_stats)] = statistics
        df_result = pd.concat([df_result, df_clustered_temp])

        z = z + 1
    sum = []
    sum.append(len(df))
    sum.append(len(non_outliers))
    sum.append(len(outliers))
    percentage = len(outliers) / len(df)
    sum.append(percentage)
    sum.append(outliers['p2'].min())
    sum.append(outliers['p2'].max())
    general_stats.loc[len(general_stats)] = sum
    general_stats.to_latex()
    print(general_stats.to_latex(index=False, multirow=True))
    df_result.outlier.value_counts()
    precision_values = pd.DataFrame(columns=['precision', 'recall', 'f-measure', 'accuracy'])
    precision_values.loc[len(precision_values)] = calc_precision(df, df_result)
    precision_values.to_latex()
    print(precision_values.to_latex(index=False, multirow=True))
    precision_values.to_csv('Resources/boxplot_precision.csv')



def calc_precision(df, df_result):
    # TODO Activate print statements for more details
    df_values = []
    df = df.rename(columns={'outlier': 'outlier_std'})
    df['outlier_std'] = df['outlier_std'].replace(['1'], 1)
    df['outlier_std'] = df['outlier_std'].replace(['0'], 0)
    positive = df.loc[df['outlier_std'] == 1 ]
    negative = df.loc[df['outlier_std'] == 0 ]

    data = [df_result['id'],df_result['outlier']]
    headers = ['id', 'outlier_box']
    df_box = pd.concat(data, axis=1, keys=headers)
    df_temp = pd.merge(df, df_box, on =['id'])

    # True positive
    df_tp = df_temp.loc[(df_temp['outlier_std'] == 1) & (df_temp['outlier_box'] == 1)]
    # False Positive
    df_fp =  df_temp.loc[(df_temp['outlier_std'] == 0) & (df_temp['outlier_box'] == 1)]
    #False nageative
    df_fn = df_temp.loc[(df_temp['outlier_std'] == 1) & (df_temp['outlier_box'] == 0)]
    #True negative
    df_tn = df_temp.loc[(df_temp['outlier_std'] == 0) & (df_temp['outlier_box'] == 0)]

    precision = len(df_tp) / (len(df_tp) + len(df_fp))
    #print('Precision: %.3f' % precision)
    df_values.append(precision)

    recall = len(df_tp) / (len(df_tp) + len(df_fn))
    #print('Recall: %.3f' % recall)
    df_values.append(recall)

    f_measure_1 = ( 2 * precision * recall) / (precision + recall)
    f_measure_2 = 2 * (( precision * recall) / (precision + recall))
    #print('F-Measure 1 : %.3f' % f_measure_1)
    #print('F-Measure 2: %.3f' % f_measure_2)
    df_values.append(f_measure_2)

    accuracy = (len(df_tp) + len (df_tn) ) / (len(positive) + len(negative))
    accuracy_2 = (len(df_tp) + len (df_tn) ) / (len(df_tp) + len(df_tn) + len(df_fp) + len(df_fn))
    #print('Accuracy 2: %.3f' % accuracy)
    #print('Accuracy 2: %.3f' % accuracy_2)
    df_values.append(accuracy)
    return df_values



def get_summary_statistics(dataset):
    mean = np.round(np.mean(dataset['p2']), 2)
    median = np.round(np.median(dataset['p2']), 2)
    min_value = np.round(dataset['p2'].min(), 2)
    max_value = np.round(dataset['p2'].max(), 2)
    quartile_1 = np.round(dataset['p2'].quantile(0.25), 2)
    quartile_3 = np.round(dataset['p2'].quantile(0.75), 2)  # Interquartile range
    iqr = np.round(quartile_3 - quartile_1, 2)
    bottom = quartile_1 - 1.5 * iqr
    top = quartile_3 + 1.5 * iqr
    # TODO Activate print statements for more details
    #print('Setosa summary statistics')
    #print('Min: %s' % min_value)
    #print('Mean: %s' % mean)
    #print('Max: %s' % max_value)
    #print('25th percentile: %s' % quartile_1)
    #print('Median: %s' % median)
    #print('75th percentile: %s' % quartile_3)
    #print('Interquartile range (IQR): %s' % iqr)
    #print('Bottom outliers below ', bottom)
    #print('Top outliers above ', top)







