""" !!! load imports first !!! """
""" Only execute single methods"""
from statisticsandplots.combination_plot import plot_roc_curve_kernel, combination_plot, plot_coordinates
from statisticsandplots.cluster_statistics import calc_statistics
from statisticsandplots.clean_data import clean_data
from clustering.calculate_silhouette_score import pm_clustering_silhouette
from clustering.create_dendrogram import plot_dendrogram
from clustering.clustering_weekly import std_clustering
from outlierdetectiontechniques.boxplot import create_boxplot
from outlierdetectiontechniques.baseline_model import calc_baseline
from outlierdetectiontechniques.isolation_forest import calc_isolation_forest
from outlierdetectiontechniques.kernel_density import calc_kde

"""
Execute the following methods, separately! (not the whole script) to obtain the results. 
Print statements were inserted to get an overview over the variables. 
For obtaining deeper insights into the mechanisms, 
open the methods in their files and execute all statements separately.
"""


"Data Preprocessing"
# Data Cleanning (removing errors from the dataset)
# Takes values from: ldi_stuttgart.pm_sensors_week"
# Writes values to: ldi_stuttgart.pm_sensors_week_cleaned"
clean_data()

# Baseline Model
# Output: the number of normal values and outliers
# Due to the high amount plots are inactivated but can be activated (uncomment) to get insights into the data distribution
# Hint: The code produces errors, due to some columns being overwritten (intentionally!).
#       This can be ignored and has been tested. No values get lost.
calc_baseline()



"Data Clustering"

# Dendrogram
# Produces Plot
plot_dendrogram()


# Silhouette Method
# creates the silhouette plot and a (latex) table, containing the results
pm_clustering_silhouette()


# Standard Deviation Calculation
# Produces a Plot for each Cluster Formation
# Two extra plots at the end summarizing the mean and maximum standard deviation values
# Produces (latex) table containing the maximum & mean standard deviation, and mean absolute deviation
std_clustering()

"General Statistics"
# For a final cluster number of 11 clusters, this table produces statistics for each of these clusters
# Generates (latex) table, containing the number of sensors & measurements per cluster
# Minimum and Maximum Values measured per Cluster
# Mean, Standard Deviation, Mean Absolut Deviation Values per Cluster
calc_statistics()

"Box Plot"
# Produces a Plot containing 11 Box Plots, one for each cluster
# Produces two (latex) tables
# First: Containing the total number of measurements, number of (non) outliers, as well as
#        the percentage, and the minimum and maximum values
# Second: Containing the f-score, precision, recall, and accuracy
# For more details, uncomment the print statements in the boxplot.py. They give detailed results about
# the iqr, whiskers, ... for each cluster
# Hint: The code produces errors, due to some columns being overwritten (intentionally!).
#       This can be ignored and has been tested. No values get lost.
create_boxplot()

"Kernel Density Estimation"
# HINT: The execution takes a long time. For obtaining insights into the results
#       go directly into the method and start the execution, where marked "START HERE".
# The calculation of the bandwidth parameter is commented, due to its high calculation time for each cluster (~1h)
# Also the execution without bandwidth takes 30-60 min (at least on my computer)
# Hint: The code produces errors, due to some columns being overwritten (intentionally!).
#       This can be ignored and has been tested. No values get lost.
# Outputs: Plots per cluster, indicating the data distribution and marking
# the data instances detected as outliers (several plots per Cluster)
calc_kde()


"Isolation Forst"
# Hint: This calculations takes some time for calculation.
# To speed it up and obtain only the statistics, go into the file and start within the starting method,
# where marked & saved calcualtion results are used form .csv files
# For obtaining the plots, execute method below.
# First Output: a (latex) table containing all results for each contamination paramter and all clusters
# Second Output: Summary of the large table above, as a mean value of the statistiscs for all 11 clusters
# Third Output: (Latex) table containing performance measures
# Hint: The code produces errors, due to some columns being overwritten (intentionally!).
#       This can be ignored and has been tested. No values get lost.
calc_isolation_forest()

"Additional Calculations"
# Check whether the dataset / clusters are normally distributed


#ROC
# Outputs the ROC of the Kernel Density Estimation and Isolation Forest
# Uses the results stored in Resources
plot_roc_curve_kernel()


# CombinationPlot
# Ouputs a plot containing all ODT with all thresholds
# Uses results stored in Resources
combination_plot()


#Plots ScatterPlot of Sensors according to their longitude and latitude values
plot_coordinates()




