import seaborn as sns
from dbhandler.fetch_data import fetchStations
from dbhandler.db_con import openConLocal
import matplotlib.pyplot as plt
import pandas as pd

# Adopted from
# https://medium.com/analytics-vidhya/create-a-grouped-bar-chart-with-matplotlib-and-pandas-9b021c97e0a





localCursor = None
localCon = None


def createConnections():
    global localCon
    localCon = openConLocal()
    global localCursor
    localCursor = localCon.cursor()

"""Starting Method for ROC """
def plot_roc_curve_kernel():
    df_kern = pd.read_csv("./Resources/KernelDensity_precision.csv")
    tpr = df_kern['recall']
    fpr = df_kern['fpr']
    plt.plot(fpr, tpr, color='orange', label='ROC Kernel Density Estimation ')
    # plt.plot([0.0, 0.12], [0.0, 1], color='darkblue', linestyle='--')
    df_iso = pd.read_csv("./Resources/IsolationForest_precision.csv")
    tpr = df_iso['recall']
    fpr = df_iso['fpr']
    plt.plot(fpr, tpr, color='green', label='ROC Isolation Forest ')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    # plt.savefig('ROC', format='pdf')
    plt.show()

"""Starting Method for Plot includings all ODTs"""
def combination_plot():
    pal = sns.color_palette("husl", 9)
    pal.as_hex()
    df_combined = pd.DataFrame(columns=['Technique', 'precision', 'recall', 'f-score', 'accuracy'])

    df_box = pd.read_csv("./Resources/boxplot_precision.csv")
    df_box = df_box.rename(columns={'Unnamed: 0': 'Technique'})
    df_box['Technique'] = 'Box Plot'

    df_iso = pd.read_csv("./Resources/IsolationForest_precision_copy.csv")
    df_iso = df_iso.drop('fpr', 1)

    df_kern = pd.read_csv("./Resources/KernelDensity_precision_copy.csv")
    df_kern = df_kern.drop('fpr', 1)

    df_combined = pd.concat([df_combined, df_iso], ignore_index=True)
    df_combined = pd.concat([df_combined, df_kern], ignore_index=True)
    df_combined = pd.concat([df_combined, df_box], ignore_index=True)


    sns.set_style("whitegrid")
    sns.color_palette("husl", 9)

    labels = ['Isolation Forest \n (0.01)', 'Isolation Forest \n (0.02)','Isolation Forest \n (0.05)',
              'Isolation Forest \n (0.1)', 'Isolation Forest \n (0.15)',
              'Isolation Forest \n (auto)', 'Kernel Density \n (0.01)',
              'Kernel Density  \n (0.02)', 'Kernel Density \n (0.05)', 'Kernel Density \n (0.1)',
              'Kernel Density \n (0.15)', 'Box Plot']


    ax = df_combined.plot(x="Technique", y=["precision", "recall", "f-measure", "accuracy"], kind="bar",
                     width=0.9, figsize=(18 ,5),  color=sns.color_palette("husl",  9))


    plt.gcf().subplots_adjust(bottom=0.3)

    plt.xticks(range(12), labels)

    rects = ax.patches

    for p in rects:
        ax.annotate(format(p.get_height(), '.2f'),
                       ((p.get_x() + p.get_width() / 2.) +0.01, p.get_height()-0.01),
                       ha='center', va='center',
                       size=12,
                       xytext=(0, -12),
                       textcoords='offset points',
                       rotation=90)

    plt.legend(loc="upper right", bbox_to_anchor=(1, 1.2), ncol=2)

    plt.title("Comparison of Outlier Detection Techniques")
    plt.xlabel("Outlier Detection Techniques")
    plt.ylabel("Scores")
    plt.xticks(rotation=0, horizontalalignment="center", fontweight='bold')
    # plt.savefig('CombinationPlot', format='pdf')
    plt.show()

"""Starting Point for Scatterplot plotting coordinates of the sensor stations"""
def plot_coordinates():
    createConnections()
    df = fetchStations(localCursor)
    plt.scatter(x=df['longitude'], y=df['latitude'])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # plt.savefig('Scatter', format='pdf')
    plt.show()

