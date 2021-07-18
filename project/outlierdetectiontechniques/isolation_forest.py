from dbhandler.fetch_data import get_pm_df_clustered
from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Adopted from
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html
# https://blog.paperspace.com/anomaly-detection-isolation-forest/
# https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1
# https://www.kaggle.com/kevinarvai/outlier-detection-practice-uni-multivariate


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
def calc_isolation_forest():
    createConnections()
    df_result = pd.DataFrame()
    general_stats_total = pd.DataFrame(columns=['perc', 'total', 'non', 'out', 'percentage', 'min', 'max'])
    df = get_pm_df_clustered(localCursor)

    z = 0
    cluster_nr = 11

    # calculates IsolationForest Statistics for each cluster
    while z < cluster_nr:
        df_cluster = pd.DataFrame()
        df_clustered_temp = df.loc[df['cluster'] == z]
        # temporary
        df_clustered_temp['outlier'] = 1
        df_clustered_temp['quartile'] = 100
        decriptive_statistics(df_clustered_temp)
        # statistics_with = Isolation_Forest(df_clustered_temp)
        # statistics.append(len(df_clustered_temp))

        contaminations = [0.01, 0.02, 0.05, 0.1, 0.15, 'auto']
        nr = 0
        while nr < 6:
            df_quantile_temp = df_clustered_temp.copy()
            df_quantile_temp['quartile'] = contaminations[nr]
            statistics = []

            statistics.append(contaminations[nr])
            statistics.append(len(df_quantile_temp))

            model = IsolationForest(max_samples='auto', contamination=contaminations[nr], max_features=1.0)
            model.fit(df_quantile_temp[['p2']])
            df_quantile_temp['scores'] = model.decision_function(df_quantile_temp[['p2']])
            df_quantile_temp['anomaly'] = model.predict(df_quantile_temp[['p2']])
            anomaly = df_quantile_temp.loc[df_quantile_temp['anomaly'] == -1]
            normal = df_quantile_temp.loc[df_quantile_temp['anomaly'] == 1]
            df_quantile_temp['outlier'][df_quantile_temp.anomaly == -1] = -1
            statistics.append(len(normal))
            anomaly_index = list(anomaly.index)
            print(len(anomaly))
            statistics.append(len(anomaly))
            outliers_counter = len(df_quantile_temp[df_quantile_temp['p2'] > 50])
            outliers_counter
            percentage = len(anomaly) / len(df_quantile_temp)
            statistics.append(percentage)
            statistics.append(anomaly['p2'].min())
            statistics.append(anomaly['p2'].max())
            general_stats_total.loc[len(general_stats_total)] = statistics
            df_cluster = pd.concat([df_cluster, df_quantile_temp])
            plot_Isolation_Forest(df_quantile_temp, model)

            df_result = pd.concat([df_result, df_clustered_temp])
            nr = nr + 1
        df_result = pd.concat([df_result, df_cluster])
        z = z + 1

    # TODO start here and use results from already existing calculations for faster results
    # df_result.to_csv('./Resources/results.csv')
    # df_result = pd.read_csv("./Resources/results.csv")
    # general_stats_total.to_csv('./Resources/general_stats.csv')
    # general_stats_total = pd.read_csv("./Resources/general_stats.csv")
    general_stats_total.to_latex()
    print(general_stats_total.to_latex(index=False, multirow=True))

    quantiles_second = [0.01, 0.02, 0.05, 0.1, 0.15, 'auto']
    nr_second = 0
    stats = pd.DataFrame(columns=['non', 'out', 'percentage', 'min', 'max'])
    precision_values = pd.DataFrame(columns=['precision', 'recall', 'fpr', 'f-measure', 'accuracy'])
    while nr_second < 6:
        summary = []
        #print('CALC ', quantiles_second[nr_second])
        df_to_calc = df_result.loc[df_result['quartile'] == quantiles_second[nr_second]]
        outlier = df_to_calc.loc[df_to_calc['outlier'] == -1]
        non_outlier = df_to_calc.loc[df_to_calc['outlier'] == 1]
        summary.append(len(non_outlier))
        summary.append(len(outlier))
        percentage = len(outlier) / len(df_to_calc)
        summary.append(percentage)
        summary.append(outlier['p2'].min())
        summary.append(outlier['p2'].max())
        stats.loc[len(stats)] = summary
        df_precision = calc_precision(df, df_to_calc)
        precision_values.loc[len(precision_values)] = df_precision
        nr_second = nr_second + 1
    stats.to_latex()
    print(stats.to_latex(index=False, multirow=True))

    precision_values.to_latex()
    print(precision_values.to_latex(index=False, multirow=True))
    precision_values.to_csv('Resources/IsolationForest_precision.csv')



def plot_Isolation_Forest(df, model):

    xx = np.linspace(df['p2'].min(), df['p2'].max(), len(df)).reshape(-1,1)

    # plt.figure(figsize=(10, 4))
    anomaly_score = model.decision_function(xx)
    outlier = model.predict(xx)
    print(len(outlier))
    plt.figure(figsize=(10, 4))
    plt.plot(xx, anomaly_score, label='Anomaly Score')
    plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                     where=outlier == -1, color='r',
                     alpha=.4, label='Outlier Region')
    plt.legend()
    plt.ylabel('Anomaly Score')
    plt.xlabel('PM-Measurements')
    # plt.savefig('IsolationForest', format='pdf')
    plt.show()



def Isolation_Forest(df_clustered_temp):
    statistics = []
    statistics.append(len(df_clustered_temp))

    model = IsolationForest(max_samples='auto', contamination='auto', max_features=1.0)
    model.fit(df_clustered_temp[['p2']])
    df_clustered_temp['scores'] = model.decision_function(df_clustered_temp[['p2']])
    df_clustered_temp['anomaly'] = model.predict(df_clustered_temp[['p2']])
    anomaly = df_clustered_temp.loc[df_clustered_temp['anomaly'] == -1]
    normal = df_clustered_temp.loc[df_clustered_temp['anomaly'] == 1]
    statistics.append(len(normal))
    anomaly_index = list(anomaly.index)
    print(len(anomaly))
    statistics.append(len(anomaly))
    outliers_counter = len(df_clustered_temp[df_clustered_temp['p2'] > 50])
    outliers_counter
    percentage = len(anomaly) / len(df_clustered_temp)
    statistics.append(percentage)
    statistics.append(anomaly['p2'].min())
    statistics.append(anomaly['p2'].max())
    plot_Isolation_Forest(df_clustered_temp , model)
    # print("Accuracy percentage:", 100 * list(df['anomaly']).count(-1) / (outliers_counter))
    return statistics



def calc_without_df_extension(df_temp):
    statistics = []
    outliers = pd.DataFrame()
    statistics.append(len(df_temp))

    model = IsolationForest(max_samples='auto', contamination='auto', max_features=1.0)
    model.fit(df_temp[['p2']])
    scores = model.decision_function(df_temp[['p2']])
    anomaly_calc = model.predict(df_temp[['p2']])
    anomaly = anomaly_calc[anomaly_calc == -1]
    normal = anomaly_calc[anomaly_calc == 1]
    statistics.append(len(normal))
    print(len(anomaly))
    statistics.append(len(anomaly))
    percentage = len(anomaly) / len(df_temp)
    statistics.append(percentage)
    statistics.append(anomaly.min())
    statistics.append(anomaly.max())
    outliers_counter = len(df_temp[df_temp['p2'] > 50])
    return statistics

def decriptive_statistics(df_clustered_temp):

    plt.scatter(range(df_clustered_temp.shape[0]), np.sort(df_clustered_temp['p2'].values))
    plt.xlabel('index')
    plt.ylabel('p2')
    plt.title("p2 distribution")
    sns.despine()
    plt.show()

    sns.distplot(df_clustered_temp['p2'])
    plt.title("Distribution of p2")
    sns.despine()
    plt.show()

def calc_precision(df, df_result):
    # TODO For more detailed insights activate print statements
    df_values = []
    df = df.rename(columns={'outlier': 'outlier_std'})
    df['outlier_std'] = df['outlier_std'].replace(['1'], 1)
    df['outlier_std'] = df['outlier_std'].replace(['0'], 0)
    positive = df.loc[df['outlier_std'] == 1]
    negative = df.loc[df['outlier_std'] == 0]

    data = [df_result['id'],df_result['outlier']]
    headers = ['id', 'outlier_box']
    df_box = pd.concat(data, axis=1, keys=headers)
    df_temp = pd.merge(df, df_box, on =['id'])

    # normal = 1, outlier = -1
    # True positive
    df_tp = df_temp.loc[(df_temp['outlier_std'] == 1) & (df_temp['outlier_box'] == -1)]
    # False Positive
    df_fp =  df_temp.loc[(df_temp['outlier_std'] == 0) & (df_temp['outlier_box'] == -1)]
    #False nageative
    df_fn = df_temp.loc[(df_temp['outlier_std'] == 1) & (df_temp['outlier_box'] == 1)]
    #True negative
    df_tn = df_temp.loc[(df_temp['outlier_std'] == 0) & (df_temp['outlier_box'] == 1)]

    precision = len(df_tp) / (len(df_tp) + len(df_fp))
    #print('Precision: %.3f' % precision)
    df_values.append(precision)

    recall = len(df_tp) / (len(df_tp) + len(df_fn))
    #print('Recall: %.3f' % recall)
    df_values.append(recall)

    fpr = len(df_fp) / (len(df_fp) + len(df_tn))
    df_values.append(fpr)


    f_measure_1 = ( 2 * precision * recall) / (precision + recall)
    f_measure_2 = 2 * (( precision * recall) / (precision + recall))
    #print('F-Measure 1 : %.3f' % f_measure_1)
    #print('F-Measure 2: %.3f' % f_measure_2)
    df_values.append(f_measure_2)

    accuracy = (len(df_tp) + len (df_tn) ) / (len(positive) + len(negative))
    #print('Accuracy 2: %.3f' % accuracy)
    df_values.append(accuracy)
    return df_values






