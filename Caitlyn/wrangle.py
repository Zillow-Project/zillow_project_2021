import env
import pandas as pd
import numpy as np

import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_connection(db, user=env.user, host=env.host, password=env.password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Acquire
def get_zillow_data():
    '''
    Grab our data from path and read as dataframe
    '''
    
    df = pd.read_sql('''
                        SELECT *
                        FROM   properties_2017 prop  
                               INNER JOIN (SELECT parcelid,
                                                  logerror,
                                                  Max(transactiondate) transactiondate 
                                           FROM   predictions_2017 
                                           GROUP  BY parcelid, logerror) pred
                                       USING (parcelid) 
                               LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
                               LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
                               LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
                               LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
                               LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
                               LEFT JOIN storytype story USING (storytypeid) 
                               LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
                       WHERE transactiondate like '2017-%%-%%'
                               AND prop.latitude IS NOT NULL
                               AND prop.longitude IS NOT NULL
                               
                               AND propertylandusetypeid between 260 AND 266
                               OR propertylandusetypeid between 273 AND 279
                               AND NOT propertylandusetypeid = 274
                               AND unitcnt = 1;''', get_connection('zillow'))
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Prepare

def drop_50_pct_null(df):
    '''This function takes in the zillow df
    removes all columns and rows with 50% nulls or more
    returns df'''
    # Drop columns with 50% or more missing values
    df = df.dropna(axis = 1, thresh = 0.5 * len(df.index))
    # went from 67 columns down to 33
    # drop rows with 50% or more missing vlaues
    df.dropna(axis = 0, thresh = 0.5 * len(df.columns))
        # ended up not dropping any rows 
            # will remain in the function in case anything were to change later on
    return df

def clean_zillow(df):
    '''This function takes in the df
    applies all the cleaning funcitons previously created
    creates features
    drops columns
    renames columns'''
    # assuming null value for pool and fireplacecnt means none
    df.poolcnt.fillna(0, inplace = True)
    df.fireplacecnt.fillna(0, inplace = True)
    # drop features/rows with more than 50% null values
    df = drop_50_pct_null(df)
    # create dummy variables and add them to the df
    dummy_df =  pd.get_dummies(df['fips'])
    dummy_df.columns = ['in_los_angeles', 'in_orange_county', 'in_ventura']
    df = pd.concat([df, dummy_df], axis=1)
    df['fips'] = df.fips.replace('6,037.00', 6037)
    df['fips'] = df.fips.replace('6,059.00', 6059)
    df['fips'] = df.fips.replace('6,111.00', 6111)
    #create new feature house_age
    today = pd.to_datetime('today')
    df['house_age'] = today.year - df['yearbuilt']
    df['tax_rate'] = df.taxvaluedollarcnt / df.taxamount
    df['acres'] = df.lotsizesquarefeet/43560
    #drop features
    df = df.drop(['propertycountylandusecode', 'propertyzoningdesc', 
                 'heatingorsystemdesc', 'transactiondate',
                  'finishedsquarefeet12', 'id', 'censustractandblock',
                 'rawcensustractandblock', 'calculatedbathnbr', 
                 'assessmentyear', 'propertylandusedesc'], axis=1)
    #rename features
    df = df.rename(columns={'heatingorsystemtypeid':'has_heating_system', 
                           'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 
                           'buildingqualitytypeid':'quality',
                           'calculatedfinishedsquarefeet':'square_feet', 
                           'fullbathcnt':'full_bathrooms',
                           'lotsizesquarefeet':'lot_square_feet', 
                           'propertylandusetypeid':'land_type',
                           'regionidcity':'city', 'regionidcounty':'county',
                           'regionidzip':'zip_code', 'roomcnt':'room_count',
                           'structuretaxvaluedollarcnt':'structure_tax_value',
                           'taxvaluedollarcnt':'tax_value', 
                           'landtaxvaluedollarcnt':'land_tax_value', 
                           'fireplacecnt':'has_fireplace',
                           'poolcnt':'has_pool'})
    # assuming that null means no heating bc it is southern CA
    df.has_heating_system.fillna('13', inplace = True)
    # change has_heating_system to binary
    df['has_heating_system'] = df.has_heating_system.replace([2.0, 7.0, 24.0, 6.0, 20.0, 13.0, 18.0, 1.0, 10.0, 11.0], 1)
    df['has_heating_system'] = df.has_heating_system.replace('13', '0')
    df['has_heating_system'] = (df['has_heating_system'] == True ).astype(int)
    # all of these are 1 unit counts
    df.unitcnt.fillna(1, inplace = True)
    df['unitcnt'] = df.unitcnt.replace([2.0, 3.0, 4.0, 6.0], 1)
    # change has_fireplace to a binary
    df['has_fireplace'] = df.has_fireplace.replace([2.0, 3.0, 4.0, 5.0], 1)
    df['has_fireplace'] = df.has_fireplace.replace(0.0, 0)
    #fix has_pool to int
    df['has_pool'] = df.has_fireplace.replace(1.0, 1)
    df['has_pool'] = df.has_fireplace.replace(0.0, 0)
    # fix unitcnt to int
    df['unitcnt'] = (df['unitcnt'] == True ).astype(int)
    # replacing null in quality feature with its median range (6)
    df.quality.fillna(6.0, inplace = True)
    # replacing null in square_feet with its median
    df.lot_square_feet.fillna(7313, inplace = True)
     # replacing null in quality feature with its median
    df.square_feet.fillna(1511, inplace = True)
     # replacing null in quality feature with its median
    df.full_bathrooms.fillna(2, inplace = True)
     # replacing null in quality feature with its median
    df.yearbuilt.fillna(1970, inplace = True)
     # replacing null in quality feature with its median
    df.structure_tax_value.fillna(134871, inplace = True)
     # replacing null in quality feature with its median
    df.house_age.fillna(51, inplace = True)
     # replacing null in quality feature with its median
    df.city.fillna(25218, inplace = True)
     # replacing null in quality feature with its median
    df.zip_code.fillna(96410, inplace = True)
    #drop remaining null values
    df = df.dropna()
    # change la, oc, and vent into int
    df['in_los_angeles'] = (df['in_los_angeles'] == True ).astype(int)
    df['in_orange_county'] = (df['in_orange_county'] == True ).astype(int)
    df['in_ventura'] = (df['in_ventura'] == True ).astype(int)
    # set index as parcelid
    df = df.set_index('parcelid')
    # finish dropping
    df = df.drop(['yearbuilt'], axis=1)
    # Handle outliers
    df = df[df.tax_value < 1153326.5]
    df = df[df.square_feet < 4506.0]
    df = df[df.acres < 0.665426997245179]
    # bin some of the large features
    # bin the square feet
    df['square_feet_bins'] = pd.cut(df.square_feet, 
                            bins = [0,500,1000,1500,2000,2500,3000,3500,4000,6000],
                            labels = [1, 2, 3, 4, 5, 6, 7, 8,9])
    df['square_feet_bins'] = (df['square_feet_bins']).astype(int)
    # bin lot square feet
    df['lot_sqft_bins'] = pd.cut(df.lot_square_feet, 
                            bins = [0,10000,20000,30000,40000,50000,60000,70000,10000000],
                            labels = [0, 1, 2, 3, 4, 5, 6, 7])
    df['lot_sqft_bins'] = (df['lot_sqft_bins']).astype(int)
    # bin acres
    df['acre_bins'] = pd.cut(df.acres, 
                            bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],
                            labels = [0, 1, 2, 3, 4, 5, 6])
    df['acre_bins'] = (df['acre_bins']).astype(int)
    # bin log error
    df['level_of_log_error'] = pd.cut(df.logerror, 
                            bins = [-5,-1,-.15,.15,1,5],
                            labels = ['Way Under', 'Under', 'Accurate', 'Over', 'Way Over'])
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Split the Data into Tain, Test, and Validate.

def split_zillow(df):
    '''This fuction takes in a df 
    splits into train, test, validate
    return: three pandas dataframes: train, validate, test
    '''
    # split the focused zillow data
    train_validate, test = train_test_split(df, test_size=.2, random_state=1234)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=1234)
    return train, validate, test


# Split the data into X_train, y_train, X_vlaidate, y_validate, X_train, and y_train

def split_train_validate_test(train, validate, test):
    ''' This function takes in train, validate and test
    splits them into X and y versions
    returns X_train, X_validate, X_test, y_train, y_validate, y_test'''
    X_train = train.drop(columns = ['logerror'])
    y_train = train.logerror
    X_validate = validate.drop(columns=['logerror'])
    y_validate = validate.logerror
    X_test = test.drop(columns=['logerror'])
    y_test = test.logerror
    return X_train, X_validate, X_test, y_train, y_validate, y_test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Scale the Data


def scale_my_data(train, validate, test):
    scale_columns = ['bathrooms', 'bedrooms', 'quality', 
              'square_feet', 'full_bathrooms', 
              'latitude', 'longitude', 'lot_square_feet',  
              'land_type', 'city', 'county', 'zip_code', 
             'room_count', 'unitcnt', 'structure_tax_value', 
             'tax_value',  'land_tax_value', 'taxamount',
              'house_age', 'tax_rate', 'acres']
    scaler = MinMaxScaler()
    scaler.fit(train[scale_columns])

    train_scaled = scaler.transform(train[scale_columns])
    validate_scaled = scaler.transform(validate[scale_columns])
    test_scaled = scaler.transform(test[scale_columns])
    #turn into dataframe
    train_scaled = pd.DataFrame(train_scaled)
    validate_scaled = pd.DataFrame(validate_scaled)
    test_scaled = pd.DataFrame(test_scaled)
    
    return train_scaled, validate_scaled, test_scaled

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# for longitude_latitude_houseage

def start_taxes_cluster(train, validate, test):
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

def start_quality_houseage_cluster(train, validate, test):
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
    return train, validate, test

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

def focused_zillow(train, validate, test):
    '''
    takes in train
    sets sepecific features to focus on
    returns a focused data frame in a pandas dataframe
    '''
    # choose features to focus on
    features = [
    'logerror',
    'latitude',
    'longitude',
    'Ventura',
    'North LA',
    'low_structure_and_land_tax',
    'medium_structure_low_land_tax',
    'house quality = 0',
    'Newer Homes High Quality',
    'Older Homes High Quality'] # the target
    # return a df based only on these features
    train = train[features]
    validate = validate[features]
    test = test[features]
    return train, validate, test