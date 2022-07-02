import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_show(name):  # name is the file name of the csv file to be plotted
    data = pd.read_csv(name)  # read the csv file into data frame data (data frame)
    data = data.drop("Unnamed: 0", axis=1)  # drop the first column (Unnamed: 0)
    ax = plt.gca()  # get the current axis (axes)
    data.plot(kind='line', y='avg', ax=ax)  # plot the dataframe data with kind='line' and y='avg'
    data.plot(kind='line', y='max', color='red',
              ax=ax)  # plot the dataframe data with kind='line' and y='max' and color='red'
    data.plot(kind='line', y='min', color='green',
              ax=ax)  # plot the dataframe data with kind='line' and y='min' and color='green'
    plt.xlabel("Generation")  # set the x label to "Generation"
    plt.ylabel("Fitness")  # set the y label to "Fitness"
    plt.show()  # show the plot


plot_show('generation_analysis_29120158.csv') # call the function plot_show with the file name as argument (name)
