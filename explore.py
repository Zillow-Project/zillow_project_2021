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
    
def logerror_pairplot(train):
    sns.pairplot(data = train, hue = 'level_of_log_error', 
             x_vars = ['logerror', 'structure_tax_value', 'tax_value', 
                       'land_tax_value'],
             y_vars = ['logerror', 'latitude', 'longitude'], 
             palette='viridis_r')

def quality_age_room_count_cluster(train):
    X = train[['quality', 'house_age', 'room_count']]
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns= X.columns).set_index([X.index.values])
    #define the thing
    kmeans = KMeans(n_clusters=5)
    # fit the thing
    kmeans.fit(X_scaled)
    # Use (predict using) the thing
    kmeans.predict(X_scaled)
    # Make a dataframe
    train['quality_houseage_roomcount_cluster'] = kmeans.predict(X_scaled)
    X_scaled['quality_houseage_roomcount_cluster'] = kmeans.predict(X_scaled)
    # Cluster Centers aka (centroids)
    kmeans.cluster_centers_
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns = X.columns)
    # lets visualize the clusters along with the centers on (scaled data).
    plt.figure(figsize=(20, 40))
    # scatter plot of data with hue for cluster
    plt.subplot(5,1,1)
    sns.scatterplot(x = 'quality', y= 'room_count', data = X_scaled, hue = 'quality_houseage_roomcount_cluster', palette='viridis')
    centroids_scaled.plot.scatter(x = 'room_count', y = 'quality', ax = plt.gca(), color = 'k', alpha = 0.3, s = 500, marker = 'o',)
    plt.subplot(5,1,2)
    sns.scatterplot(x = 'house_age', y= 'quality', data = X_scaled, hue = 'quality_houseage_roomcount_cluster', palette='viridis')
    centroids_scaled.plot.scatter(x = 'house_age', y = 'quality', ax = plt.gca(), color = 'k', alpha = 0.3, s = 500, marker = 'o',)
    plt.subplot(5,1,3)
    sns.scatterplot(x = 'house_age', y= 'room_count', data = X_scaled, hue = 'quality_houseage_roomcount_cluster', palette='viridis')
    centroids_scaled.plot.scatter(x = 'house_age', y = 'room_count', ax = plt.gca(), color = 'k', alpha = 0.3, s = 500, marker = 'o',)
    plt.legend();
    # lets visualize the clusters along with the centers on (scaled data).
    plt.figure(figsize=(14, 9))

    
def quality_age_room_count_relplot(train):
    X = train[['quality', 'house_age', 'room_count']]
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns= X.columns).set_index([X.index.values])
    #define the thing
    kmeans = KMeans(n_clusters=5, random_state=123)
    # fit the thing
    kmeans.fit(X_scaled)
    # Use (predict using) the thing
    kmeans.predict(X_scaled)
    # Make a dataframe
    train['quality_houseage_roomcount_cluster'] = kmeans.predict(X_scaled)
    X_scaled['quality_houseage_roomcount_cluster'] = kmeans.predict(X_scaled)
    # Cluster Centers aka (centroids)
    kmeans.cluster_centers_
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns = X.columns)
    # scatter plot of data with hue for cluster
    sns.relplot(x = 'house_age', y= 'quality', data = X_scaled, col = X_scaled.quality_houseage_roomcount_cluster, col_wrap = 2, hue = train.level_of_log_error, palette='viridis_r')
    # plot cluster centers (centroids)
    # centroids_scaled.plot.scatter(x = 'age', y = 'annual_income', ax = plt.gca(), color = 'k', alpha = 0.3, s = 500, marker = 'o',)
    plt.show();
    
    
    
def get_dum_and_plot(train):
    dummy_df =  pd.get_dummies(train['quality_houseage_roomcount_cluster'])
    dummy_df.columns = ['zero', 'one', 'two', 'three', 'four']
    df = pd.concat([train, dummy_df], axis=1)
    # Plot the clusters
    plt.figure(figsize=(20, 36))
    plt.subplot(3,2,1)
    plt.title("Percents of Each Log Error Level for Homeq = 0", size=20, color='black')
    sns.barplot(y=df.zero, x='level_of_log_error', data=df,
                   palette='viridis')
    plt.subplot(3,2,2)
    plt.title("Percents of Each Log Error Level Older Homes Low Quality", size=20, color='black')
    sns.barplot(y=df.one, x='level_of_log_error', data=df,
                   palette='viridis')
    plt.subplot(3,2,3)
    plt.title("Percents of Each Log Error Level Younger Homes Avg. Quality", size=20, color='black')
    sns.barplot(y=df.two, x='level_of_log_error', data=df,
                   palette='viridis')
    plt.subplot(3,2,4)
    plt.title("Percents of Each Log Error Level Newer Homes High Quality", size=20, color='black')
    sns.barplot(y=df.three, x='level_of_log_error', data=df,
                   palette='viridis')
    plt.subplot(3,2,5)
    plt.title("Percents of Each Log Error Level Older Homes High Quality", size=20, color='black')
    sns.barplot(y=df.four, x='level_of_log_error', data=df,
                   palette='viridis')
    
def closer_tax_plot(train):
    plt.subplots(1, 2, figsize=(25,8), sharey=True)
    sns.set(style="darkgrid")

    plt.subplot(1,2,1)
    plt.title("Percents of Each Log Error Level in LA", size=20, color='black')
    sns.scatterplot(x = train.structure_tax_value, y = train.logerror, hue = train.level_of_log_error, palette='viridis')

    plt.subplot(1,2,2)
    plt.title("Percents of Each Log Error Level in Orange County", size=20, color='black')
    sns.scatterplot(x = train.land_tax_value, y = train.logerror, hue = train.level_of_log_error, palette='viridis')
 

    
def taxes_cluster(train):
    B = train[['structure_tax_value', 'land_tax_value']]
    scaler = StandardScaler().fit(B)
    B_scaled = pd.DataFrame(scaler.transform(B), columns= B.columns).set_index([B.index.values])
    B.head()
    #define the thing
    kmeans = KMeans(n_clusters=6, random_state=123)
    # fit the thing
    kmeans = kmeans.fit(B_scaled)
    # Use (predict using) the thing 
    kmeans.predict(B_scaled)
    # create the cluster features
    train['taxes_cluster'] = kmeans.labels_
    B_scaled['taxes_cluster'] = kmeans.labels_
    # set centroids
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns = ['structure_tax_value', 'land_tax_value'])
    centroids = train.groupby('taxes_cluster')['structure_tax_value', 'land_tax_value'].mean()
    # Rename the clusters
    train['taxes_cluster'] = 'cluster' + train.taxes_cluster.astype(str)
    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    # scatter plot of data with hue for cluster
    sns.scatterplot(x='structure_tax_value', y='land_tax_value', data=B_scaled, hue='taxes_cluster', palette='viridis')
    # plot cluster centers (centroids)
    centroids_scaled.plot.scatter(x='structure_tax_value', y='land_tax_value', ax = plt.gca(), color ='black', alpha = 0.3, s = 800, marker = 'o', label = 'centroids')
    plt.title('Visualizing Cluster Centers')
    plt.legend();

def taxes_relplot(train):
    # lets visualize the clusters along with the centers on (scaled data).
    plt.figure(figsize=(14, 9))
    # scatter plot of data with hue for cluster
    sns.relplot(x='structure_tax_value', y='land_tax_value', 
                data = train, col = train.taxes_cluster, 
                col_wrap = 2, hue = train.level_of_log_error, 
               palette='viridis_r')
    # plot cluster centers (centroids)
    # centroids_scaled.plot.scatter(x = 'age', y = 'annual_income', ax = plt.gca(), color = 'k', alpha = 0.3, s = 500, marker = 'o',)
    plt.show();
    
def cluster_longitude_latitude_houseage(train):
    X = train[['longitude', 'latitude', 'house_age']]
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns= X.columns).set_index([X.index.values])
    #define the thing
    kmeans = KMeans(n_clusters=5, random_state = 123)
    # fit the thing
    kmeans.fit(X_scaled)
    # Use (predict using) the thing
    kmeans.predict(X_scaled)
    train['longitude_latitude_houseage_cluster'] = kmeans.predict(X_scaled)
    X_scaled['longitude_latitude_houseage_cluster'] = kmeans.predict(X_scaled)
    # Cluster Centers aka (centroids)
    kmeans.cluster_centers_
    # Make a dataframe
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns = X.columns)
    # Plot the cluster
    # lets visualize the clusters along with the centers on (scaled data).
    plt.figure(figsize=(20, 20))
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'longitude', y= 'latitude', data = X_scaled, hue = X_scaled.longitude_latitude_houseage_cluster, palette='viridis_r')
    centroids_scaled.plot.scatter(x = 'longitude', y= 'latitude', ax = plt.gca(), color = 'k', alpha = 0.3, s = 500, marker = 'o',)
    plt.legend();

def lat_long_relplot(train):
    X = train[['longitude', 'latitude', 'house_age']]
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns= X.columns).set_index([X.index.values])
    #define the thing
    kmeans = KMeans(n_clusters=5, random_state = 123)
    # fit the thing
    kmeans.fit(X_scaled)
    # Use (predict using) the thing
    kmeans.predict(X_scaled)
    train['longitude_latitude_houseage_cluster'] = kmeans.predict(X_scaled)
    X_scaled['longitude_latitude_houseage_cluster'] = kmeans.predict(X_scaled)
    # Cluster Centers aka (centroids)
    kmeans.cluster_centers_
    # Make a dataframe
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns = X.columns)
    # Plot the clusers in relplot
    plt.figure(figsize=(14, 9))
    sns.relplot(x = 'longitude', y= 'latitude', data = X_scaled, col = X_scaled.longitude_latitude_houseage_cluster, col_wrap = 2, hue = train.level_of_log_error, palette='viridis')
    plt.show();


def get_dum_and_plot2(train):
    dummy_df =  pd.get_dummies(train['longitude_latitude_houseage_cluster'])
    dummy_df.columns = ['zero', 'one', 'two', 'three', 'four']
    df = pd.concat([train, dummy_df], axis=1)
    # Plot the clusters
    plt.figure(figsize=(20, 32))
    plt.subplot(3,2,1)
    plt.title("Percents of Each Log Error Level for Ventura", size=20, color='black')
    sns.barplot(y=df.zero, x='level_of_log_error', data=df,
                  palette='viridis')
    plt.subplot(3,2,2)
    plt.title("Percents of Each Log Error Level for Orange County", size=20, color='black')
    sns.barplot(y=df.one, x='level_of_log_error', data=df,
                   palette='viridis')
    plt.subplot(3,2,3)
    plt.title("Percents of Each Log Error Level for North downtown LA", size=20, color='black')
    sns.barplot(y=df.two, x='level_of_log_error', data=df,
                   palette='viridis')
    plt.subplot(3,2,4)
    plt.title("Percents of Each Log Error Level for East downtown LA", size=20, color='black')
    sns.barplot(y=df.three, x='level_of_log_error', data=df,
                   palette='viridis')
    plt.subplot(3,2,5)
    plt.title("Percents of Each Log Error Level for North LA", size=20, color='black')
    sns.barplot(y=df.four, x='level_of_log_error', data=df,
                   palette='viridis')
    
    
def OLS_Model(X_train, y_train, X_validate, y_validate)
    # create the model object
    lm = LinearRegression(normalize=True)
    # fit the model to our training data. We must specify the column in y_train,
    # since we have converted it to a dataframe from a series!
    lm.fit(X_train, y_train.logerror)
        # just call y_train.actual_target
    # predict train
    y_train['logerror_pred_lm'] = lm.predict(X_train)
    # evaluate: rmse
    rmse_train_lm = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm)**(1/2)
    # predict validate
    y_validate['logerror_pred_lm'] = lm.predict(X_validate)
    # evaluate: rmse
    rmse_validate_lm = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm)**(1/2)
        # make sure you are using x_validate an not x_train
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train_lm,
          "\nValidation/Out-of-Sample: ", rmse_validate_lm)