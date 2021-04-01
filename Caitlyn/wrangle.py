from env import host, user, password
import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt
from statsmodels.formula.api import ols

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def acquire_zillow():
    '''
    Grab data from Codeup SQL server
    '''
    sql_query = '''select *
    from properties_2017
    left join airconditioningtype using(airconditioningtypeid)
    left join architecturalstyletype using(architecturalstyletypeid)
    left join buildingclasstype using(buildingclasstypeid)
    left join heatingorsystemtype using(heatingorsystemtypeid)
    left join storytype using(storytypeid)
    left join typeconstructiontype using(typeconstructiontypeid)
    join (select parcelid, max(logerror) as logerror, max(transactiondate) as transactiondate
                from predictions_2017
                group by parcelid) as pred_17 using(parcelid)
    where transactiondate like '2017-%%-%%'
        and parcelid in(
            select distinct parcelid)
            and latitude is not null
                and longitude is not null;'''
    # make the connection to codeup sequel server
    connection = f'mysql+pymysql://{user}:{password}@{host}/zillow'
    # Assign the df
    df = pd.read_sql(sql_query, connection)
    return df


def clean_zillow(df):
    '''This function takes in the df
    applies all the cleaning funcitons previously created
    drops columns
    renames columns'''
    # assuming null value for pool means no pool
    df.poolcnt.fillna(0, inplace = True)
    # assuming null calie for fireplace means no pool
    df.fireplacecnt.fillna(0, inplace = True)
    df = only_one_unit_homes(df)
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
    #create new feature tax_rate which is the monthyl taxes
    df['tax_rate'] = df.taxvaluedollarcnt / df.taxamount
    # create new feature for log_error_levels
    df['level_of_log_error'] = pd.qcut(df.logerror, q=5, labels=['L1', 'L2', 'L3', 'L4', 'L5'])
    # create new feature acres
    df['acres'] = df.lotsizesquarefeet/43560
    #drop features
    df = df.drop(['propertycountylandusecode', 'propertyzoningdesc', 
                 'heatingorsystemdesc', 'transactiondate',
                  'finishedsquarefeet12', 'id', 'censustractandblock',
                 'rawcensustractandblock', 'calculatedbathnbr', 
                 'assessmentyear'], axis=1)
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
    # drop all remaining null values
    # all of these are 1 unit counts
    df.unitcnt.fillna(1, inplace = True)
    df['unitcnt'] = df.unitcnt.replace([2.0, 3.0, 4.0, 6.0], 1)
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
    # change has_fireplace to a binary
    df['has_fireplace'] = df.has_fireplace.replace([2.0, 3.0, 4.0, 5.0], 1)
    df['has_fireplace'] = df.has_fireplace.replace(0.0, 0)
    #fix has_pool to int
    df['has_pool'] = df.has_fireplace.replace(1.0, 1)
    df['has_pool'] = df.has_fireplace.replace(0.0, 0)
    # change unitcnt and has_heating_system to integers istead of objects
    df['has_heating_system'] = (df['has_heating_system'] == True ).astype(int)
    df['unitcnt'] = (df['unitcnt'] == True ).astype(int)
    #drop remaining null values
    df = df.dropna()
    # change la, oc, and vent into int
    df['in_los_angeles'] = (df['in_los_angeles'] == True ).astype(int)
    df['in_orange_county'] = (df['in_orange_county'] == True ).astype(int)
    df['in_ventura'] = (df['in_ventura'] == True ).astype(int)
    # set index as parcelid
    df = df.set_index('parcelid')
    return df

def missing_zero_values_table(df):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        null_count = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, null_count, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'null_count', 2 : '% of Total Values'})
        mz_table['Total Zeroes + Null Values'] = mz_table['Zero Values'] + mz_table['null_count']
        mz_table['% Total Zero + Null Values'] = 100 * mz_table['Total Zeroes + Null Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] >= 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " +  str((mz_table['null_count'] != 0).sum()) +
          " columns that have NULL values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
        return mz_table

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


scale_columns = ['bathrooms', 'bedrooms', 'quality', 
                  'square_feet', 'full_bathrooms', 
                  'latitude', 'longitude', 'lot_square_feet', 'has_pool', 
                  'land_type', 'city', 'county', 'zip_code', 
                 'room_count', 'unitcnt', 'structure_tax_value', 
                 'tax_value',  'land_tax_value', 'taxamount',
                  'house_age', 'tax_rate']

def scale_my_data(train, validate, test, scale_columns):
    scaler = MinMaxScaler()
    scaler.fit(train[scale_columns])
    
    train_scaled = scaler.transform(train[scale_columns])
    validate_scaled = scaler.transform(validate[scale_columns])
    test_scaled = scaler.transform(test[scale_columns])
    return train_scaled, validate_scaled, test_scaled





def null_tables(df):
    '''This function will take in a df
    counts the number of missing features
    counts the number of missing rows
    finds the percent of missing columns
    returns a table with each of theses are features'''
    # Gotta set up the new Dataframes info
    table_nulls = df.isnull().sum(axis =1).value_counts().sort_index(ascending=False)
    # Make it into an officail df
    table_nulls = pd.DataFrame(table_nulls)
    # reset the index
    table_nulls.reset_index(level=0, inplace=True)
    # create the columns num_cols_missing and num_rows_missing
    table_nulls.columns= ['num_cols_missing', 'num_rows_missing']
    # now I need to add the percent column
    table_nulls['pct_cols_missing']= round((table_nulls.num_cols_missing /df.shape[1]) * 100, 2)
    return table_nulls

def only_one_unit_homes(df):
    '''This function takes in the zillow df
    removes rows where the propertylandusetype id is not a single unite
    returns a new df'''
    # Remove rows where propertylandusetypeid is less than 260
    clean_it = df.loc[df['propertylandusetypeid'] < 260].index
    df.drop(clean_it , inplace=True)
    # Remove rows where propertylandusetypeid is 267
    clean_it = df.loc[df['propertylandusetypeid'] == 267].index
    df.drop(clean_it , inplace=True)
        # Remove rows where propertylandusetypeid is 267
    clean_it = df.loc[df['propertylandusetypeid'] == 268].index
    df.drop(clean_it , inplace=True)
        # Remove rows where propertylandusetypeid is 267
    clean_it = df.loc[df['propertylandusetypeid'] == 269].index
    df.drop(clean_it , inplace=True)
        # Remove rows where propertylandusetypeid is 267
    clean_it = df.loc[df['propertylandusetypeid'] == 270].index
    df.drop(clean_it , inplace=True)
        # Remove rows where propertylandusetypeid is 267
    clean_it = df.loc[df['propertylandusetypeid'] == 271].index
    df.drop(clean_it , inplace=True)
    # Remove rows where propertylandusetypeid is 269
    clean_it = df.loc[df['propertylandusetypeid'] == 274].index
    df.drop(clean_it , inplace=True)
    # Remove rows where propertylandusetypeid is less than 260
    clean_it = df.loc[df['propertylandusetypeid'] > 279].index
    df.drop(clean_it , inplace=True)
    return df

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