
from numpy import where, quantile
from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate

from dbhandler.fetch_data import get_pm_df_clustered
import pandas as pd
from dbhandler.db_con import openConLocal
from dbhandler.db_con import closeLocal
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np

#Adopted from
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
# https://scikit-learn.org/stable/modules/density.html
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
# https://www.datatechnotes.com/2020/05/anomaly-detection-with-kernel-density-in-python.html
# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
# https://www.statsmodels.org/stable/examples/notebooks/generated/kernel_density.html

localCursor = None
localCon = None


def createConnections():
    global localCon
    localCon = openConLocal()
    global localCursor
    localCursor = localCon.cursor()


def closeConnections():
    closeLocal(localCursor)


def calc_kde():
    # execute until while loop, when starting below
    createConnections()
    df = get_pm_df_clustered(localCursor)
    general_stats_total = pd.DataFrame(columns=['perc', 'total', 'non', 'out', 'percentage', 'min', 'max'])
    z = 5
    df_result = pd.DataFrame()
    cluster_nr = 11

    while z < cluster_nr:
        df_cluster = pd.DataFrame()
        df_clustered_temp = df.loc[df['cluster'] == z]
        df_clustered_temp['outlier'] = 0
        df_clustered_temp['quartile'] = 100

        x_ax = range(df_clustered_temp['p2'].size)
        plt.plot(x_ax, df_clustered_temp['p2'])
        plt.show()
        x = scale(df_clustered_temp[['p2']])

        # TODO activate for obtaining optimal bandwidth (takes up to an hour for each cluster)
        # Calculates optimal bandwidth
        # grid = GridSearchCV(KernelDensity(),
        #                     {'bandwidth': np.linspace(0.1, 1.0, 30)},
        #                     cv=20)  # 20-fold cross-validation
        # grid.fit(x)
        # print(grid.best_params_['bandwidth'])
        #
        # bw = round(grid.best_params_['bandwidth'], 2)

        #kde = KernelDensity(bandwidth=bw)
        #kde.fit(x)
        #print(kde)

        # TODO comment the next three lines, when part above is activated
        # here just a fixed bandwidth parameter is used
        kde = KernelDensity(bandwidth=0.22)
        kde.fit(x)
        print(kde)


        scores = kde.score_samples(x)

        thresh = quantile(scores, .15)
        print(thresh)

        index = where(scores <= thresh)
        values = x[index]

        plt.plot(x_ax, x)
        plt.scatter(index, values, color='r')
        plt.show()


        quantiles = [0.01, 0.02, 0.05, 0.1, 0.15]
        nr = 0
        # executes the KDE calculations 5 times, with varying thresholds (quantiles)
        while nr < 5:
            df_quantile_temp = df_clustered_temp.copy()
            df_quantile_temp['quartile'] = quantiles[nr]
            statistics = []
            statistics.append(quantiles[nr])
            statistics.append(len(df_quantile_temp))

            thresh = quantile(scores, quantiles[nr])
            print(thresh)
            index = where(scores <= thresh)
            values = x[index]
            plt.plot(x_ax, x)
            plt.scatter(index, values, color='r')
            plt.xlabel("PM-Measurements")
            plt.ylabel("Data Instances")
            # plt.savefig('KDE', format='pdf')
            plt.show()
            df_quantile_temp['value'] = scores
            outliers = df_quantile_temp.loc[df_quantile_temp['value'] <= thresh]
            non = df_quantile_temp.loc[df_quantile_temp['value'] > thresh]
            df_quantile_temp['outlier'][df_quantile_temp.value <= thresh] = 1
            statistics.append(len(non))
            statistics.append(len(outliers))
            percentage = len(outliers) / len(df_quantile_temp)
            statistics.append(percentage)
            statistics.append(outliers['p2'].min())
            statistics.append(outliers['p2'].max())
            general_stats_total.loc[len(general_stats_total)] = statistics
            df_cluster = pd.concat([df_cluster, df_quantile_temp])
            nr = nr + 1
        df_result = pd.concat([df_result, df_cluster])
        z = z + 1

    # TODO: START HERE (But execute the first lines of the method(until while loop))
    # df_result.to_csv('Resources/results.csv')
    # df_result = pd.read_csv("./Resources/results.csv")
    df_result = pd.read_csv("./Resources/results_0.csv")
    df_result.append(pd.read_csv("./Resources/results_1.csv"))
    df_result.append(pd.read_csv("./Resources/results_2.csv"))
    df_result.append(pd.read_csv("./Resources/results_3.csv"))
    # general_stats_total.to_csv('Resources/general_stats.csv')
    general_stats_total = pd.read_csv("./Resources/general_stats.csv")
    general_stats_total.to_latex()
    print(general_stats_total.to_latex(index=False, multirow=True))

    quantiles_second = [0.01, 0.02, 0.05, 0.1, 0.15]
    nr_second = 0
    stats = pd.DataFrame(columns=['non', 'out', 'percentage', 'min', 'max'])
    precision_values = pd.DataFrame(columns=['precision', 'recall', 'fpr', 'f-measure', 'accuracy'])
    # calculates the precision measures for each threshold
    while nr_second < 5:
        summary = []
        print('CALC ', quantiles_second[nr_second])
        df_to_calc = df_result.loc[df_result['quartile'] == quantiles_second[nr_second]]
        outlier = df_to_calc.loc[df_to_calc['outlier'] ==1]
        non_outlier = df_to_calc.loc[df_to_calc['outlier'] ==0]
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
    precision_values.to_csv('Resources/KernelDensity_precision.csv')



def calc_precision(df, df_to_calc):
    # define actual
    df_values = []
    df = df.rename(columns={'outlier': 'outlier_std'})
    df['outlier_std'] = df['outlier_std'].replace(['1'], 1)
    df['outlier_std'] = df['outlier_std'].replace(['0'], 0)
    positive = df.loc[df['outlier_std'] == 1]
    negative = df.loc[df['outlier_std'] == 0]

    data = [df_to_calc['id'], df_to_calc['outlier']]
    headers = ['id', 'outlier_box']
    df_box = pd.concat(data, axis=1, keys=headers)
    df_temp = pd.merge(df, df_box, on=['id'])

    # True positive
    df_tp = df_temp.loc[(df_temp['outlier_std'] == 1) & (df_temp['outlier_box'] == 1)]
    # False Positive
    df_fp = df_temp.loc[(df_temp['outlier_std'] == 0) & (df_temp['outlier_box'] == 1)]
    # False nageative
    df_fn = df_temp.loc[(df_temp['outlier_std'] == 1) & (df_temp['outlier_box'] == 0)]
    # True negative
    df_tn = df_temp.loc[(df_temp['outlier_std'] == 0) & (df_temp['outlier_box'] == 0)]

    precision = len(df_tp) / (len(df_tp) + len(df_fp))
    df_values.append(precision)
    # print('Precision: %.3f' % precision)

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)

    recall = len(df_tp) / (len(df_tp) + len(df_fn))
    #print('Recall: %.3f' % recall)
    df_values.append(recall)

    fpr = len(df_fp) / (len(df_fp) + len(df_tn))
    df_values.append(fpr)

    f_measure_1 = (2 * precision * recall) / (precision + recall)
    f_measure_2 = 2 * ((precision * recall) / (precision + recall))
    #print('F-Measure 1 : %.3f' % f_measure_1)
    #print('F-Measure 2: %.3f' % f_measure_2)
    df_values.append(f_measure_2)

    accuracy = (len(df_tp) + len(df_tn)) / (len(positive) + len(negative))
    #print('Accuracy 2: %.3f' % accuracy)
    df_values.append(accuracy)

    # auc = metrics.auc(recall, precision)
    # df_values.append(auc)

    return df_values

"""Additional Method for Creating Plot"""
def calc_Kernel():
    createConnections()
    df = get_pm_df_clustered(localCursor)

    z = 5
    cluster_nr = 11
    bandwidth = [0.16, 0.22, 0.1, 0.38, 0.57, 0.22, 0.1, 0.1, 0.29, 0.13, 0.5]

    while z < cluster_nr:

        df_clustered_temp = df.loc[df['cluster'] == z]
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        # Scatter plot of data samples and histogram
        ax.scatter(df_clustered_temp['p2'], np.abs(np.random.randn(df_clustered_temp['p2'].size)),
                   zorder=15, color='red', marker='x', alpha=0.5, label='Samples')
        lines = ax.hist(df_clustered_temp['p2'], bins=20, edgecolor='k', label='Histogram')
        plt.show()

        ax.legend(loc='best')
        ax.grid(True, zorder=-5)

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        # Scatter plot of data samples and histogram
        ax.scatter(df_clustered_temp['p2'], np.abs(np.random.randn(df_clustered_temp['p2'].size)),
               zorder=15, color='red', marker='x', alpha=0.5, label='Samples')
        lines = ax.hist(df_clustered_temp['p2'], bins=20, edgecolor='k', label='Histogram')

        ax.legend(loc='best', prop={'size': 10})
        ax.grid(True, zorder=-5)

        kde = KDEUnivariate(df_clustered_temp['p2'])
        print("Plotting with bandwidth: ", bandwidth[z])
        kde.fit(bw = bandwidth[z]) # Estimate the densities

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        # Plot the histrogram
        ax.hist(df_clustered_temp['p2'], bins=20, density=True, label='Histogram from samples',
                zorder=5, edgecolor='k', alpha=0.5)

        # Plot the KDE as fitted using the default arguments
        ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)



        # Plot the samples
        ax.scatter(df_clustered_temp['p2'], np.abs(np.random.randn(df_clustered_temp['p2'].size)) / 40,
                   marker='x', color='red', zorder=20, label='Samples', alpha=0.5)

        ax.legend(loc='best', prop={'size': 14})
        ax.set_xlabel("PM-Measurements", fontsize=12)
        ax.set_ylabel("Normalised Density", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, zorder=-5)
        # plt.savefig('KDE', format='pdf')
        plt.show()
        z = z +1