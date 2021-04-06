import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import wrangle

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def explore_logerror(train):
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.boxenplot(x=train["logerror"], palette='ocean')
    
def log_error_deep_dive(train):
    plt.subplots(1, 3, figsize=(25,8), sharey=True)
    sns.set(style="darkgrid")

    plt.subplot(1,3,1)
    plt.title("Percents of Each Log Error Level in LA", size=20, color='black')
    sns.barplot(y='in_los_angeles', x='level_of_log_error', data=train,
                   palette='viridis')

    plt.subplot(1,3,2)
    plt.title("Percents of Each Log Error Level in Orange County", size=20, color='black')
    sns.barplot(y='in_orange_county', x='level_of_log_error', data=train,
                   palette='viridis')

    plt.subplot(1,3,3)
    plt.title("Percents of Each Log Error Level in Ventura", size=20, color='black')
    sns.barplot(y='in_ventura', x='level_of_log_error', data=train,
                   palette='viridis')