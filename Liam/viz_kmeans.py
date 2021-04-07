import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

def viz_zillow(train, kmeans):
    
    centroids = np.array(train.groupby('cluster')['square_feet', 'logerror'].mean())
    cen_x = [i[0] for i in centroids]
    cen_y = [i[1] for i in centroids]
    # cen_x = [i[0] for i in kmeans.cluster_centers_]
    # cen_y = [i[1] for i in kmeans.cluster_centers_]
    train['cen_x'] = train.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    train['cen_y'] = train.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

    colors = ['#DF2020','#2095DF', '#81DF20']
    train['c'] = train.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    #plot scatter chart for Actual species and those predicted by K - Means

    #specify custom palette for sns scatterplot
    colors1 = ['#2095DF','#81DF20' ,'#DF2020']
    customPalette = sns.set_palette(sns.color_palette(colors1))

    #plot the scatterplots

    #Define figure (num of rows, columns and size)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    # plot ax1 
    ax1 = plt.subplot(2,1,1) 
    sns.scatterplot(data = train, x = 'square_feet', y = 'logerror', ax = ax1, palette=customPalette)
    plt.title('County')

    #plot ax2
    ax2 = plt.subplot(2,1,2) 
    ax2.scatter(train.square_feet, train.logerror, c=train.cluster, alpha = 0.6, s=10)
    ax2.set(xlabel = 'square_feet', ylabel = 'logerror', title = 'K - Means')

    # plot centroids on  ax2
    ax2.scatter(cen_x, cen_y, marker='X', c=colors, s=200)
    
    
    train.drop(columns = ['cen_x', 'cen_y', 'c'], inplace = True)
    plt.tight_layout()
    plt.show()
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def start_longitude_latitude_houseage(train, validate, test):
    kmeans = KMeans(n_clusters=5, random_state=123)
    # identify columns we want to cluster on
    cluster_cols = ['latitude', 'longitude', 'house_age']
    # clustering on train, getting the cetnoids
    kmeans = kmeans.fit(train[cluster_cols])
    # identifying clusters in train
    train['longitude_latitude_houseage_cluster'] = kmeans.predict(train[cluster_cols])
    # identifying clusters in validate, test
    validate['longitude_latitude_houseage_cluster'] = kmeans.predict(validate[cluster_cols])
    test['longitude_latitude_houseage_cluster'] = kmeans.predict(test[cluster_cols])
    return train, validate, test

def predict_cluster_longitude_latitude_houseage(some_dataframe):
    some_dataframe['longitude_latitude_houseage_cluster'] = kmeans.predict(some_dataframe[cluster_cols])
    return some_dataframe

def get_dummy_longitude_latitude_houseage_cluster(some_dataframe):
    dummy_df =  pd.get_dummies(some_dataframe['longitude_latitude_houseage_cluster'])
    dummy_df.columns = ['Ventura', 'Orange County', 
                    'North downtown LA', 'East downtown LA', 
                    'North LA']
    some_dataframe = pd.concat([some_dataframe, dummy_df], axis=1)
    some_dataframe = some_dataframe.drop(['Orange County', 'East downtown LA', 
                    'North downtown LA', 'longitude_latitude_houseage_cluster'], axis=1)
    return some_dataframe

def prep_longitude_latitude_houseage_clusters(some_dataframe):
    some_dataframe = predict_cluster_longitude_latitude_houseage(some_dataframe)
    some_dataframe = get_dummy_longitude_latitude_houseage_cluster(some_dataframe)
    return some_dataframe

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# for taxes_cluster

def start_taxes_cluster(train, validate, test):
    kmeans = KMeans(n_clusters=6, random_state=123)
    # identify columns we want to cluster on
    cluster_cols = ['structure_tax_value', 'land_tax_value']
    # clustering on train, getting the cetnoids
    kmeans = kmeans.fit(train[cluster_cols])
    # identifying clusters in train
    train['taxes_cluster'] = kmeans.predict(train[cluster_cols])
    # identifying clusters in validate, test
    validate['taxes_cluster'] = kmeans.predict(validate[cluster_cols])
    test['taxes_cluster'] = kmeans.predict(test[cluster_cols])
    return train, validate, test

def predict_cluster_taxes(some_dataframe):
    some_dataframe['taxes_cluster'] = kmeans.predict(some_dataframe[cluster_cols])
    return some_dataframe

def get_dummy_taxes_cluster(some_dataframe):
    dummy_df =  pd.get_dummies(some_dataframe['taxes_cluster'])
    dummy_df.columns = ['low_structure_and_land_tax', 'drop1',
                        'drop2', 'medium_structure_low_land_tax', 
                        'drop4', 'drop5']
    some_dataframe = pd.concat([some_dataframe, dummy_df], axis=1)
    some_dataframe = some_dataframe.drop(['drop1', 'drop2', 'drop4', 'drop5', 'taxes_cluster'], axis=1)
    return some_dataframe

def prep_taxes_clusters(some_dataframe):
    some_dataframe = predict_cluster_taxes(some_dataframe)
    some_dataframe = get_dummy_taxes_cluster(some_dataframe)
    return some_dataframe

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# for quality_houseage_roomcount
    kmeans = KMeans(n_clusters=5, random_state=123)
    # identify columns we want to cluster on
    cluster_cols = ['quality', 'house_age', 'room_count']
    # clustering on train, getting the cetnoids
    kmeans = kmeans.fit(train[cluster_cols])
    # identifying clusters in train
    train['quality_houseage_roomcount_cluster'] = kmeans.predict(train[cluster_cols])
    # identifying clusters in validate, test
    validate['quality_houseage_roomcount_cluster'] = kmeans.predict(validate[cluster_cols])
    test['quality_houseage_roomcount_cluster'] = kmeans.predict(test[cluster_cols])

def predict_cluster_quality_houseage_roomcount(some_dataframe):
    some_dataframe['quality_houseage_roomcount_cluster'] = kmeans.predict(some_dataframe[cluster_cols])
    return some_dataframe

def get_dummy_quality_houseage_roomcount_cluster(some_dataframe):
    dummy_df =  pd.get_dummies(some_dataframe['quality_houseage_roomcount_cluster'])
    dummy_df.columns = ['house quality = 0', 
                    'Older homes low quality', 
                    'Younger homes avg. quality', 
                    'Newer Homes High Quality', 
                    'Older Homes High Quality']
    some_dataframe = pd.concat([some_dataframe, dummy_df], axis=1)
    some_dataframe = some_dataframe.drop(['Older homes low quality', 
                    'Younger homes avg. quality', 
                    'quality_houseage_roomcount_cluster'], axis=1)
    return some_dataframe

def prep_quality_houseage_roomcount_clusters(some_dataframe):
    some_dataframe = predict_cluster_quality_houseage_roomcount(some_dataframe)
    some_dataframe = get_dummy_quality_houseage_roomcount_cluster(some_dataframe)
    return some_dataframe